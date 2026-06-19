"""Chunkers for approaches 2A, 2B, and 2C.

2B (sentence-window) and 2C (parent-document) will follow in this same file.
Scaffold order: complete 2A first, then unlock 2B and 2C.
"""

from typing import Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter          # pip install langchain-text-splitters
from langchain_experimental.text_splitter import SemanticChunker             # pip install langchain-experimental
import hashlib

from .schema import Chunk
from ..ingestion.schema import ParsedDocument


# ---- 2A: Recursive character splitter (per-section) ----------------------

# chunk_recursive: split each ParsedSection independently; no section-boundary crossing.
# Input:  doc — a ParsedDocument from any 1A/1B/1C loader
# Output: list[Chunk] with full provenance metadata on every chunk


def chunk_recursive(
	doc: ParsedDocument,
	*,
	chunk_size: int = 1000,
	chunk_overlap: int = 150,
) -> list[Chunk]:
	"""2A baseline: split each section independently with RecursiveCharacterTextSplitter.

	Section metadata (heading_path, page_idx, slide_idx) is preserved on every
	emitted chunk. chunk_idx is a per-document counter (0-based).
	"""

	# Step 1: build the splitter once — reused across all sections
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		# Hint: the splitter accepts any Callable[[str], int] as its size oracle.
		# Background: `len` gives character count, which is the correct default here;
		#             to measure in tokens instead, swap in a tiktoken encode→len wrapper.
		# Answer: len
		length_function=len,  # (easy)
	)

	chunks: list[Chunk] = []
	chunk_idx = 0

	# Step 2: walk sections; skip any that are empty after stripping whitespace
	for section_idx, section in enumerate(doc.sections):
		if not section.text.strip():
			continue

		# Step 3: split this section into pieces and emit one Chunk per piece
		for piece in splitter.split_text(section.text):

			# heading_path mutation risk — read before filling in the blank below:
			# If you pass section.heading_path directly, every Chunk from this
			# section shares ONE list object. A downstream consumer that appends
			# a breadcrumb to any chunk's heading_path will silently corrupt all
			# the others from the same section. The one-call fix creates an
			# independent copy that can't be reached through section.heading_path.

			chunks.append(Chunk(
				source=doc.source,
				chunk_idx=chunk_idx,
				chunker="2a",
				section_idxs=[section_idx],
				text=piece,           # (easy)
				heading_path=list(section.heading_path),   # (think)
				page_idx=section.page_idx,
				slide_idx=section.slide_idx,
				meta={"chunk_id": f"{doc.source}::{chunk_idx}", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
			))
			chunk_idx += 1

	return chunks


# ---- 2B: Semantic chunker (embedding-similarity breakpoints) ---------------

# chunk_semantic: split where sentence-to-sentence embedding similarity drops sharply.
# Input:  doc; embeddings — any LangChain Embeddings object injected by the caller
# Output: list[Chunk], boundaries chosen by semantic drift rather than fixed char counts


def chunk_semantic(
	doc: ParsedDocument,
	embeddings,                                  # any LangChain Embeddings object
	*,
	breakpoint_threshold_type: str = "percentile",
	breakpoint_threshold_amount: float = 80.0,
) -> list[Chunk]:
	"""2B: split where sentence-to-sentence embedding similarity drops significantly.

	embeddings is injected so the caller controls which model is used — the same
	model used here MUST be used at index time, otherwise distances are meaningless.

	breakpoint_threshold_amount with type="percentile" means: split only where the
	similarity drop is larger than that percentile of all drops in the section.
	Higher → fewer, larger chunks.  Lower → more, smaller chunks.
	"""
	# Hint: SemanticChunker's threshold_type controls HOW breakpoints are selected.
	# Background: "percentile" splits at the top-N% sharpest drops (content-adaptive);
	#             "standard_deviation" splits where drop > mean + N*σ;
	#             "gradient" finds inflection points in the similarity curve.
	# Answer: breakpoint_threshold_type
	splitter = SemanticChunker(
		embeddings=embeddings,
		breakpoint_threshold_type=breakpoint_threshold_type,          # (easy) — re-read the Answer line above
		breakpoint_threshold_amount=breakpoint_threshold_amount,
	)

	chunks: list[Chunk] = []
	chunk_idx = 0

	for section_idx, section in enumerate(doc.sections):
		if not section.text.strip():
			continue

		for piece in splitter.split_text(section.text):
			chunks.append(Chunk(
				source=doc.source,
				chunk_idx=chunk_idx,
				chunker="2b",
				section_idxs=[section_idx],
				text=piece,                           # (easy) same pattern as chunk_recursive
				heading_path=list(section.heading_path),
				page_idx=section.page_idx,
				slide_idx=section.slide_idx,
				meta={
					"chunk_id": f"{doc.source}::{chunk_idx}",
					"breakpoint_threshold_type":   breakpoint_threshold_type,
					"breakpoint_threshold_amount": breakpoint_threshold_amount,
				},
			))
			chunk_idx += 1

	return chunks


# ---- 2C: Parent-document chunker ------------------------------------------

# chunk_parent_document: two-level split — small children are embedded; large parents are returned.
# Input:  doc; parent_size — large context window; child_size/child_overlap — embedding chunk params
# Output: list[Chunk] where text = child (embedding unit), meta["parent_text"] = full context


def chunk_parent_document(
	doc: ParsedDocument,
	*,
	parent_size: int = 2000,
	child_size: int = 400,
	child_overlap: int = 50,
) -> list[Chunk]:
	"""2C: embed small child chunks; store the parent in meta for retrieval-time expansion.

	At query time: similarity search finds relevant children (short, precise match);
	the LLM receives meta["parent_text"] (long, informative) as its context window.
	"""
	# Guard: the two-level split degenerates if child chunks are not strictly smaller than parents
	assert child_size < parent_size, "child_size must be less than parent_size"  # (think) what invariant must hold?

	parent_splitter = RecursiveCharacterTextSplitter(
		chunk_size=parent_size,
		chunk_overlap=0,         # parents don't overlap — avoids storing duplicate context in meta
		length_function=len,
	)
	child_splitter = RecursiveCharacterTextSplitter(
		chunk_size=child_size,
		chunk_overlap=child_overlap,
		length_function=len,
	)

	chunks: list[Chunk] = []
	chunk_idx = 0

	for section_idx, section in enumerate(doc.sections):
		if not section.text.strip():
			continue

		for parent in parent_splitter.split_text(section.text):
			for child in child_splitter.split_text(parent):
				chunks.append(Chunk(
					source=doc.source,
					chunk_idx=chunk_idx,
					chunker="2c",
					section_idxs=[section_idx],
					text=child,               # (easy) which variable is the embedding unit?
					heading_path=list(section.heading_path),
					page_idx=section.page_idx,
					slide_idx=section.slide_idx,
					meta={
						"parent_text": parent,  # (think) what does retrieval-time expansion need here?
						"child_size": child_size,
						"parent_size": parent_size,
					},
				))
				chunk_idx += 1

	return chunks


# ---------------------------------------------------------------------------
# Smoke test  (run: python -m src.chunking.chunkers)
# Prerequisite: place any .docx file at tests/fixtures/sample.docx
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	from pathlib import Path
	from ..ingestion.loaders import load_docx_1b

	doc = load_docx_1b(Path("tests/fixtures/sample.docx"))
	chunks = chunk_recursive(doc, chunk_size=500, chunk_overlap=80)

	print(f"total chunks  : {len(chunks)}")
	print(f"chunker       : {chunks[0].chunker}")
	print(f"text preview  : {chunks[0].text[:80]!r}")
	print(f"section_idxs  : {chunks[0].section_idxs}")
	print(f"heading_path  : {chunks[0].heading_path}")
	# expected shape (numbers vary by file):
	# total chunks  : 23
	# chunker       : 2a
	# text preview  : 'Introduction\n\nThis thesis investigates...'
	# section_idxs  : [0]
	# heading_path  : ['Introduction']

	# Defensive-copy isolation check — tests blank #3 directly.
	# If you passed section.heading_path as a shared reference, the WARN fires.
	first_section_chunks = [c for c in chunks if c.section_idxs == [0]]
	if first_section_chunks:
		doc.sections[0].heading_path.append("INJECTED")
		if "INJECTED" in first_section_chunks[0].heading_path:
			print("WARN: heading_path is a shared reference — blank #3 needs list()")
		else:
			print("isolation : OK — heading_path is an independent copy")

	print()
	print("--- 2B smoke test ---")
	# SemanticChunker needs a LangChain Embeddings object — BGE is the project default.
	# First run will download ~400 MB; subsequent runs use the local cache.
	# from langchain_community.embeddings import HuggingFaceEmbeddings  # pip install langchain-community
	# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
	from langchain_openai import OpenAIEmbeddings  # pip install langchain-openai
	embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
	chunks_2b = chunk_semantic(doc, embeddings, breakpoint_threshold_amount=95.0)
	print(f"total chunks  : {len(chunks_2b)}")
	print(f"chunker       : {chunks_2b[0].chunker}")
	print(f"text preview  : {chunks_2b[0].text[:80]!r}")
	print(f"threshold meta: {chunks_2b[0].meta}")
	# expected shape (chunk count varies — SemanticChunker is non-deterministic):
	# total chunks  : (often fewer than 2A — semantic boundaries are coarser than char limits)
	# chunker       : 2b
	# text preview  : a coherent semantic unit, not a mid-sentence cut
	# threshold meta: {'breakpoint_threshold_type': 'percentile', 'breakpoint_threshold_amount': 95.0}

	print()
	print("--- 2C smoke test ---")
	chunks_2c = chunk_parent_document(doc, parent_size=1000, child_size=200, child_overlap=30)
	print(f"total chunks  : {len(chunks_2c)}")
	print(f"chunker       : {chunks_2c[0].chunker}")
	print(f"child text    : {chunks_2c[0].text[:60]!r}")
	print(f"parent longer : {len(chunks_2c[0].meta['parent_text']) > len(chunks_2c[0].text)}")
	# expected shape:
	# total chunks  : (more than 2A — each parent splits into multiple children)
	# chunker       : 2c
	# child text    : first ~200 chars of the first child chunk
	# parent longer : True


# ---- HINTS (uncover only if stuck > 5 min) ----
# 2A — blank 1 (length_function):
#   Already given inline above the blank — re-read the Answer: comment.
# 2A — blank 2 (text):
#   What does `for piece in splitter.split_text(...)` bind the current string to?
# 2A — blank 3 (heading_path):
#   One-call Python idiom that constructs an independent list from any iterable.
#   Answer: list(section.heading_path)
#
# 2B — blank 1 (breakpoint_threshold_type):
#   Already given inline above the blank — re-read the Answer: comment.
# 2B — blank 2 (text):
#   Same as 2A — what loop variable does splitter.split_text(...) bind each string to?
#
# 2C — blank 1 (assert):
#   The invariant: child chunks must fit inside a parent chunk.
#   Answer: child_size < parent_size
# 2C — blank 2 (text):
#   Which variable — child or parent — goes into the vector store for similarity search?
# 2C — blank 3 (parent_text):
#   Which variable — child or parent — should the LLM see as its context window?


# ---- Reflection questions ----
# Q2 (2A): chunk_idx resets to 0 for every document. With 100 docs in the corpus,
#     "chunk 5" is ambiguous — it exists once per document. What is the real
#     corpus-unique identifier for a chunk, and what would break first in a
#     vector store if you used chunk_idx alone as a primary key?
#     (The answer involves a composite key and one specific failure mode.)
#
# Q3 (2B): SemanticChunker with percentile=95 splits only at the top 5% sharpest
#     similarity drops. What would you expect if you set it to 50 instead?
#     And how would you detect a "degenerate" chunking (one giant chunk per section)
#     in production before it reaches the vector store?
#     (Think: chunk length distribution, a simple assert, or a fallback splitter.)
#
# Q4 (2C): If two child chunks share the same parent, meta["parent_text"] is stored
#     twice — once per child Chunk object. At what corpus scale does this become a
#     storage problem, and what data-structure change would store each parent exactly once?
#     (Sketch the change in two sentences — the answer involves a separate lookup and an ID.)
