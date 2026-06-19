"""Approach 3E — Contextual retrieval (Anthropic-style).

Before embedding each chunk, an LLM writes 1–2 sentences situating the chunk
within its source document.  That context is prepended to chunk.text so the
embedding captures *what the chunk is about* in addition to *what it says*.

Example
-------
Original chunk text:
    "This improvement represents a 3× increase over the previous design."

After context prepend:
    "Context: This chunk is from the AAB_Manuscript, Results section,
    reporting peak power density improvements of the Ag28Cu72 cathode.
    ---
    This improvement represents a 3× increase over the previous design."

The embedding of the contextualised chunk is richer — it places the vague
pronoun "this improvement" in the right semantic neighbourhood.

Cost model
----------
One LLM call per chunk.  For a 500-chunk corpus at gpt-4o-mini pricing
(~$0.15 / 1M input tokens, ~200 tokens per call):
    500 × 200 tokens = 100K tokens ≈ $0.015
Cheap enough to run on the full corpus.  Responses are cached to
`caches/contextual_cache.jsonl` so re-runs are free.
"""

import json
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel

from ..chunking.schema import Chunk


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_CONTEXT_PROMPT = """\
Here is a document excerpt (first 1500 characters):
<document>
{doc_excerpt}
</document>

Here is a chunk from that document:
<chunk>
{chunk_text}
</chunk>

Write 1-2 sentences that situate this chunk within the document.
Mention the document's topic, the section this chunk belongs to, and
what concept this chunk specifically covers.
Reply with ONLY those 1-2 sentences — no preamble, no explanation."""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

# add_context_to_chunks: for each chunk, call the LLM to generate situating
# context, prepend it to chunk.text, return a new list of Chunks.
#
# Input:  chunks     — list[Chunk] from any chunker
#         doc_texts  — dict mapping source path → full document text
#                      (used as context for the LLM prompt)
#         llm        — any LangChain chat model
#         cache_path — where to persist generated contexts (avoid re-calling LLM)
# Output: list[Chunk] with context prepended to each chunk's text


def add_context_to_chunks(
	chunks:     list[Chunk],
	doc_texts:  dict[str, str],
	llm:        BaseLanguageModel,
	cache_path: Path = Path("caches/contextual_cache.jsonl"),
) -> list[Chunk]:
	"""Return new Chunks with LLM-generated context prepended to each text.

	Chunks whose source is not in doc_texts are returned unchanged.
	Results are cached to cache_path — re-runs load from cache instantly.
	"""
	# Step 1: load existing cache to avoid redundant LLM calls
	# The cache maps chunk_id → context string.
	cache = _load_cache(cache_path)
	cache_path.parent.mkdir(parents=True, exist_ok=True)

	contextual_chunks = []

	for chunk in chunks:
		chunk_id = chunk.meta.get("chunk_id", f"{chunk.source}::{chunk.chunk_idx}")

		# Step 2: get the document excerpt for the prompt
		# Use only the first 1500 chars of the document — enough context for the
		# LLM without blowing up the prompt for very long documents.
		doc_excerpt = doc_texts.get(chunk.source, "")[:1500]  # (easy) how many chars?

		if not doc_excerpt:
			# No document text available — return chunk unchanged
			contextual_chunks.append(chunk)
			print(f"  no doc text for chunk {chunk_idx_label(chunk)} of {len(chunks)}", end="\r")
			continue

		# Step 3: check cache first; call LLM only on cache miss
		if chunk_id in cache:
			context = cache[chunk_id]
		else:
			prompt  = _CONTEXT_PROMPT.format(
				doc_excerpt=doc_excerpt,     # (easy) pass the document excerpt
				chunk_text=chunk.text,      # (easy) pass the chunk text
			)
			# Hint: LangChain chat models expose .invoke(prompt) → AIMessage.
			# Background: .content extracts the string from the AIMessage response.
			# Answer: llm.invoke(prompt).content
			context = llm.invoke(prompt).content  # (think) call + extract string

			# Persist to cache immediately so a crash doesn't lose work
			cache[chunk_id] = context
			_append_cache(cache_path, chunk_id, context)

		# Step 4: prepend context to chunk text
		# Format: "Context: {context}\n---\n{original text}"
		# The separator "---" visually distinguishes context from chunk content.
		contextualised_text = f"Context: {context}" + "\n---\n" + chunk.text  # (think) build the full prepended string

		# Step 5: build a new Chunk with the contextualised text
		# Everything else stays the same — source, metadata, heading_path, etc.
		# Only text and chunker tag change.
		contextual_chunks.append(Chunk(
			text=contextualised_text,                         # (easy) use the contextualised text
			source=chunk.source,
			chunk_idx=chunk.chunk_idx,
			chunker="3e",                          # tag so eval can filter by approach
			section_idxs=chunk.section_idxs,
			heading_path=list(chunk.heading_path),
			page_idx=chunk.page_idx,
			slide_idx=chunk.slide_idx,
			meta={**chunk.meta, "original_text": chunk.text, "context": context},
		))

		print(f"  contextualised chunk {chunk_idx_label(chunk)} of {len(chunks)}", end="\r")

	print(f"\ncontextualised {len(contextual_chunks)} chunks")
	return contextual_chunks


def chunk_idx_label(chunk: Chunk) -> str:
	return f"{chunk.chunk_idx+1}"


# ---------------------------------------------------------------------------
# Cache helpers  (given — no blanks)
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> dict[str, str]:
	"""Load chunk_id → context mapping from a JSONL cache file."""
	if not path.exists():
		return {}
	cache = {}
	with path.open(encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if line:
				entry = json.loads(line)
				cache[entry["chunk_id"]] = entry["context"]
	return cache


def _append_cache(path: Path, chunk_id: str, context: str) -> None:
	"""Append one entry to the JSONL cache file."""
	with path.open("a", encoding="utf-8") as fh:
		fh.write(json.dumps({"chunk_id": chunk_id, "context": context}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Smoke test  (run: python3 -m src.ingestion.contextual)
# Prerequisites: chunks_recursive.jsonl, parsed_corpus.jsonl
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	import json as _json
	from langchain_openai import ChatOpenAI   # pip install langchain-openai

	chunk_path  = Path("chunks_recursive.jsonl")
	parsed_path = Path("parsed_corpus.jsonl")

	if not chunk_path.exists() or not parsed_path.exists():
		print("ERROR: need chunks_recursive.jsonl and parsed_corpus.jsonl")
		raise SystemExit(1)

	from ..chunking.stats import load_chunks_jsonl
	chunks = load_chunks_jsonl(chunk_path)

	# Build doc_texts map: source → full text
	doc_texts = {}
	with parsed_path.open(encoding="utf-8") as fh:
		for line in fh:
			doc = _json.loads(line.strip())
			full_text = "\n\n".join(s["text"] for s in doc["sections"] if s["text"].strip())
			doc_texts[doc["source"]] = full_text

	llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

	# Contextualise a small subset first to check quality and cost
	subset = chunks[:5]
	print(f"Contextualising {len(subset)} chunks (subset test)...")
	contextual = add_context_to_chunks(subset, doc_texts, llm)

	for c in contextual:
		print(f"\n--- chunk {c.chunk_idx} ({Path(c.source).name}) ---")
		print(c.text[:300])

	# expected: each chunk now starts with 1-2 sentences of context,
	# then "---", then the original chunk text
	contextual = add_context_to_chunks(chunks, doc_texts, llm)

	# save to disk
	with Path("contextual_chunks.jsonl").open("w") as fh:
		for c in contextual:
			fh.write(json.dumps(c.__dict__) + "\n")


# ---- HINTS (uncover only if stuck > 5 min) ----
# doc_excerpt blank: 1500
# prompt blanks: doc_excerpt=doc_excerpt, chunk_text=chunk.text
# llm call: llm.invoke(prompt).content
# contextualised_text: f"Context: {context}" + "\n---\n" + chunk.text
# text blank: contextualised_text


# ---- Reflection question ----
# Q12: The cache maps chunk_id → context string and is appended line-by-line.
#      If you change the LLM prompt template and re-run, the cache still
#      returns old contexts. What's the minimal change to the cache key that
#      would invalidate stale entries when the prompt changes?
#      (Hint: think about what uniquely identifies a (prompt_version, chunk) pair.)
