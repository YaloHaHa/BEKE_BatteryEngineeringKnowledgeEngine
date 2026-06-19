"""Challenge 3 — Index building: Chroma (dense, 3A) and BM25 (sparse, part of 3B).

Day-4 deliverable: run this module to embed chunks into Chroma, build the BM25
in-memory index, and confirm round-trip retrieval works on 3 probe queries.

Architecture note
-----------------
Both Chroma and BM25Retriever speak LangChain Document objects
(page_content + metadata), NOT our Chunk dataclass directly.
`_chunk_to_document` is the bridge — it flattens Chunk provenance into a
flat metadata dict, because Chroma's metadata store only accepts scalar values
(str / int / float / bool).  Nested lists or dicts raise a runtime error.
"""

import json
from pathlib import Path

from langchain_chroma import Chroma                                   # pip install langchain-chroma
from langchain_community.retrievers import BM25Retriever              # pip install langchain-community
from langchain_core.documents import Document

from ..chunking.schema import Chunk
from ..chunking.stats import load_chunks_jsonl


# ---------------------------------------------------------------------------
# Document conversion
# ---------------------------------------------------------------------------

# _chunk_to_document: flatten a Chunk into a LangChain Document.
# Input:  chunk — one Chunk from any chunker (2A / 2B / 2C)
# Output: Document(page_content=chunk.text, metadata={flat provenance dict})


def _chunk_to_document(chunk: Chunk) -> Document:
	"""Convert a Chunk to a LangChain Document for indexing.

	Chroma metadata must be flat — all values must be str / int / float / bool.
	Any list or nested dict raises a runtime error at index time.
	"""
	return Document(
		page_content=chunk.text,
		metadata={
			# Step 1: scalar provenance fields — pass through directly
			"source":    chunk.source,         # (easy) the chunk's file-path field
			"chunk_idx": chunk.chunk_idx,
			"chunker":   chunk.chunker,
			"page_idx":  chunk.page_idx,
			"slide_idx": chunk.slide_idx,

			# Step 2: corpus-unique ID — prefer the one stored in meta, fall back to composite
			# Background: chunk.meta["chunk_id"] was computed in the chunker as
			#   f"{doc.source}::{chunk_idx}".  Use .get() with a fallback so this
			#   function works even on Chunks that predate the meta["chunk_id"] field.
			"chunk_id":  chunk.meta.get("chunk_id", f"{chunk.source}::{chunk.chunk_idx}"),      # (think) what's the fallback composite key?

			# Step 3: serialise list[str] → JSON string (Chroma cannot store lists)
			# Hint: use the standard library function that converts a Python object to a JSON string.
			# Background: json.dumps(["a","b"]) → '["a", "b"]'; callers can json.loads() it back.
			# Answer: json.dumps(chunk.heading_path)
			"heading_path": json.dumps(chunk.heading_path),     # (think) — see hint above
		},
	)


# ---------------------------------------------------------------------------
# 3A: Dense index (Chroma)
# ---------------------------------------------------------------------------

# build_chroma_index: embed all chunks and persist to disk.
# Input:  chunks — list[Chunk]; embeddings — any LangChain Embeddings object
#         persist_dir — where Chroma stores its SQLite + FAISS files
# Output: Chroma vectorstore (also saved to disk for fast re-load)


def build_chroma_index(
	chunks: list[Chunk],
	embeddings,
	persist_dir: Path,
	collection_name: str = "rag_corpus",
) -> Chroma:
	"""Embed chunks and persist a Chroma vector store.

	Each chunk becomes one Chroma document.  The index is saved to disk at
	persist_dir so subsequent runs call load_chroma_index() instead of
	re-embedding everything (embedding 1 000 chunks takes ~30 s on CPU).
	"""
	# Step 1: ensure the output directory exists before Chroma tries to write
	persist_dir.mkdir(parents=True, exist_ok=True)    # (think) what happens on a second run if exist_ok=False?
	
	# Step 2: convert all Chunks to LangChain Documents
	docs = [_chunk_to_document(c) for c in chunks]                     # (easy) call the converter defined above

	# Step 3: extract the corpus-unique ID for every document
	# Background: Chroma requires a unique string id per document.
	#   Passing explicit ids prevents Chroma from generating random UUIDs,
	#   which means a second call with the same chunks will update-in-place
	#   rather than create duplicates.
	ids = [d.metadata["chunk_id"] for d in docs]            # (think) which metadata key holds the corpus-unique id?

	# Step 4: build (or overwrite) the Chroma collection
	# Hint: Chroma.from_documents() is the class method that embeds + stores in one call.
	# Background: persist_directory must be a str, not a Path — Chroma's underlying
	#   SQLite driver does not accept pathlib.Path objects.
	# Answer (persist_directory line): persist_directory=str(persist_dir)
	vectorstore = Chroma.from_documents(
		documents=docs,
		embedding=embeddings,                               # (easy) pass the embeddings object
		collection_name=collection_name,
		persist_directory=str(persist_dir),          # (think) — see hint above; what type does Chroma expect?
		ids=ids,
	)
	print(f"indexed {len(docs):,} chunks → {persist_dir}")
	return vectorstore


def load_chroma_index(
	persist_dir: Path,
	embeddings,
	collection_name: str = "rag_corpus",
) -> Chroma:
	"""Load a previously persisted Chroma index from disk (no re-embedding)."""
	return Chroma(
		collection_name=collection_name,
		embedding_function=embeddings,
		persist_directory=str(persist_dir),
	)


# ---------------------------------------------------------------------------
# 3B (sparse half): BM25 index
# ---------------------------------------------------------------------------

# build_bm25_retriever: build an in-memory BM25 inverted index over the corpus.
# Input:  chunks — list[Chunk]; k — number of results to return per query
# Output: BM25Retriever ready to call .invoke(query) → list[Document]


def build_bm25_retriever(chunks: list[Chunk], k: int = 10) -> BM25Retriever:
	"""Build an in-memory BM25 retriever over the chunk corpus.

	BM25 scores a document d against query q as:
	    score(d,q) = Σ_t  IDF(t) × TF(t,d) × (k1+1) / (TF(t,d) + k1×(1−b+b×|d|/avgdl))
	k1 ≈ 1.5 caps TF saturation; b ≈ 0.75 normalises for document length.
	No embedding call, no API cost — builds from raw tokens in memory.
	"""
	# Step 1: convert Chunks to Documents (BM25Retriever uses the same Document type)
	docs = [_chunk_to_document(c) for c in chunks]       # (easy) same converter as build_chroma_index

	# Step 2: build the BM25 index
	# Hint: BM25Retriever exposes a class method that accepts a list of Documents.
	# Background: from_documents() tokenises page_content with a default whitespace
	#   tokeniser and builds the inverted index in memory (~instant for thousands of chunks).
	# Answer: BM25Retriever.from_documents(docs)
	retriever = BM25Retriever.from_documents(docs)  # (easy) — class method name; see hint above

	# Step 3: set how many results to return per query
	retriever.k = k                    # (think) attribute name that controls result count

	return retriever


# ---------------------------------------------------------------------------
# Smoke test  (run: python -m src.retrieval.index)
# Prerequisites:
#   1. chunks_recursive.jsonl exists (produced by Day-3 stats.py smoke test)
#   2. sentence-transformers installed: pip install sentence-transformers
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface
	# from langchain_community.embeddings import HuggingFaceEmbeddings  # deprecated

	chunk_path = Path("chunks_recursive.jsonl")
	if not chunk_path.exists():
		print(f"ERROR: {chunk_path} not found — run src/chunking/stats.py first.")
		raise SystemExit(1)

	chunks = load_chunks_jsonl(chunk_path)
	print(f"loaded {len(chunks):,} chunks from {chunk_path}")

	embeddings = HuggingFaceEmbeddings(
		model_name="BAAI/bge-base-en-v1.5",
		model_kwargs={"device": "cpu"},
		encode_kwargs={"normalize_embeddings": True},
	)

	# ---- 3A: build Chroma ---------------------------------------------------
	persist_dir = Path("indices/chroma_recursive")
	vectorstore = build_chroma_index(chunks, embeddings, persist_dir)

	probe = "What is the discharge mechanism of Al-Air batteries?"
	results = vectorstore.similarity_search(probe, k=3)
	print(f"\nChroma top-3 for: {probe!r}")
	for i, doc in enumerate(results):
		print(f"  [{i+1}] chunk {doc.metadata.get('chunk_idx')} | {doc.metadata.get('source','?').split('/')[-1]}")
		print(f"       {doc.page_content[:80]!r}")

	# ---- 3B (sparse): BM25 --------------------------------------------------
	bm25 = build_bm25_retriever(chunks, k=3)
	results_bm25 = bm25.invoke(probe)
	print(f"\nBM25 top-3 for: {probe!r}")
	for i, doc in enumerate(results_bm25):
		print(f"  [{i+1}] chunk {doc.metadata.get('chunk_idx')} | {doc.metadata.get('source','?').split('/')[-1]}")
		print(f"       {doc.page_content[:80]!r}")

	# expected shape (numbers vary by corpus):
	# loaded 15 chunks from chunks_recursive.jsonl
	# indexed 15 chunks → indices/chroma_recursive
	# Chroma top-3 for: 'What is the discharge mechanism...'
	#   [1] chunk 5 | AgCathode_Manuscript.docx
	#       'The Al-Air battery discharges via oxidation of aluminium...'


# ---- HINTS (uncover only if stuck > 5 min) ----
# _chunk_to_document — blank 1 (source):
#   chunk.source is the field; it's already a str.
#
# _chunk_to_document — blank 2 (chunk_id fallback):
#   The corpus-unique format is f"{chunk.source}::{chunk.chunk_idx}".
#   This is the same format the chunkers stored in meta["chunk_id"].
#
# _chunk_to_document — blank 3 (heading_path):
#   Answer: json.dumps(chunk.heading_path)
#
# build_chroma_index — blank 1 (exist_ok):
#   True — a second run would fail with FileExistsError otherwise.
#
# build_chroma_index — blank 2 (docs list comprehension):
#   _chunk_to_document(c)
#
# build_chroma_index — blank 3 (metadata key for id):
#   "chunk_id" — the key set in _chunk_to_document.
#
# build_chroma_index — blank 4 (embedding= arg):
#   embeddings — the function parameter.
#
# build_chroma_index — blank 5 (persist_directory type):
#   str(persist_dir) — Chroma's SQLite driver doesn't accept pathlib.Path.
#
# build_bm25_retriever — blank 1 (docs list comprehension):
#   _chunk_to_document(c)
#
# build_bm25_retriever — blank 2 (class method):
#   from_documents
#
# build_bm25_retriever — blank 3 (attribute name):
#   k — BM25Retriever.k controls how many results are returned per .invoke() call.


# ---- Reflection question ----
# Q6: Chroma persists its index to disk.  If you add 50 new chunks to the corpus
#     tomorrow, what is the better workflow: (a) delete the index and rebuild from
#     scratch, or (b) open the existing collection and upsert only the new chunks?
#     What does Chroma's API offer for case (b), and what could silently go wrong
#     if you call from_documents() on a collection that already contains those IDs?
#     (Hint: look at the difference between .add() and .upsert() on a Collection.)
