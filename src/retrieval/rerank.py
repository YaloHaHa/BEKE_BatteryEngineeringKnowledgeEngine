"""Challenge 3 — Reranking: 3C cross-encoder reranker.

Day-5 deliverable: implement CrossEncoderReranker so the eval runner can
compare 3A / 3B / 3C on the same eval set.

Background
----------
Dense and hybrid retrievers score documents using embedding similarity — a
single dot product between the query vector and each chunk vector.  This is
fast (sub-millisecond per chunk) but shallow: the model never sees the query
and document *together*.

A cross-encoder fixes that.  It takes the concatenated string
    "[CLS] query [SEP] passage [SEP]"
and runs the full transformer attention over both at once.  Every token in
the query can attend to every token in the passage.  This is much more
accurate — but also O(n) inference calls per query, so you only run it on a
small shortlist (top-50) from the first-stage retriever.

bge-reranker-v2-m3 (BAAI)
    - State-of-the-art open-source cross-encoder, multilingual, 568M params.
    - API: sentence_transformers.CrossEncoder
    - Input: list of (query, passage) string pairs
    - Output: list of float scores (higher = more relevant)
    - Typical latency: ~0.5 s for 50 pairs on CPU, ~0.05 s on GPU.

Two-stage pipeline (3C):
    query
      │
      ▼
    3B hybrid retriever  ──→  top-50 candidates   (fast, recall-oriented)
      │
      ▼
    bge-reranker-v2-m3   ──→  re-scored + sorted  (slow, precision-oriented)
      │
      ▼
    top-k final results
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder                    # pip install sentence-transformers
from pydantic import Field


# ---------------------------------------------------------------------------
# 3C: Cross-encoder reranker
# ---------------------------------------------------------------------------

# CrossEncoderReranker: a LangChain BaseRetriever that wraps any first-stage
# retriever and re-scores its top-fetch_k candidates with a cross-encoder.
#
# Design:
#   base_retriever  — the first-stage retriever (use 3B hybrid for best results)
#   model           — a loaded CrossEncoder instance (injected, not constructed here)
#   top_n           — how many results to return after reranking  (default 10)
#   fetch_k         — how many candidates to pull from base_retriever (default 50)
#                     must be > top_n; larger = better recall, slower rerank
#
# LangChain custom retriever contract:
#   Subclass BaseRetriever, implement _get_relevant_documents(query, *, run_manager).
#   The public .invoke(query) method calls this internally.


class CrossEncoderReranker(BaseRetriever):
	"""3C: first-stage retrieval → cross-encoder rerank → top_n results."""

	base_retriever: Any = Field(description="First-stage retriever (e.g. 3B hybrid)")
	model:          Any = Field(description="Loaded CrossEncoder instance")
	top_n:          int = Field(default=10, description="Final results to return")
	fetch_k:        int = Field(default=50, description="Candidates to fetch before reranking")

	def _get_relevant_documents(
		self,
		query: str,
		*,
		run_manager: CallbackManagerForRetrieverRun,
	) -> list[Document]:
		"""Fetch fetch_k candidates, re-score with cross-encoder, return top_n."""

		# Step 1: pull the first-stage candidate pool
		# Temporarily widen the base retriever's k to fetch_k so we get a large pool.
		# Background: we retrieve more than we'll return (50 > 10) because the
		#   first-stage ranking is imperfect — some of the best answers sit at
		#   rank 15–40 for BM25+dense but will be lifted to rank 1 by the reranker.
		candidates = self.base_retriever.invoke(query)   # k is baked into base_retriever at construction time

		# Step 2: build (query, passage) pairs for the cross-encoder
		# Background: CrossEncoder.predict() expects a list of 2-element sequences.
		#   Each element is (query_string, passage_string).
		#   The passage is the chunk text stored in page_content.
		pairs = [(query, doc.page_content) for doc in candidates]    # (think) what are the two strings?

		# Step 3: score all pairs in one batched call
		# Hint: CrossEncoder exposes a .predict() method that takes the pairs list.
		# Background: returns a numpy array of floats; higher score = more relevant.
		#   batch_size=16 keeps CPU memory stable; increase to 32 on GPU.
		# Answer: self.model.predict(pairs, batch_size=16)
		scores = self.model.predict(pairs, batch_size=16)          # (easy) — see hint above

		# Step 4: pair each score with its document, sort by score descending
		# Background: zip(scores, candidates) produces (score, doc) tuples.
		#   sorted(..., key=lambda x: x[0], reverse=True) sorts highest-score first.
		scored_docs = sorted(
			zip(scores, candidates),                            # (easy) built-in that pairs two iterables
			key=lambda x: x[0],                              # (think) which element of the tuple is the score?
			reverse=True,                                        # (easy) highest score first = ascending or descending?
		)

		# Step 5: strip scores, return only the top_n documents
		return [doc for _, doc in scored_docs[:self.top_n]]     # (think) unpack the tuple; which attribute caps the list?


# ---------------------------------------------------------------------------
# 3D: Multi-query + rerank
# ---------------------------------------------------------------------------

# make_multiquery_reranking_retriever: generate N query paraphrases via LLM,
# retrieve independently for each, deduplicate, then rerank with cross-encoder.
#
# Why this helps:
#   A single query phrasing explores one region of the embedding space.
#   Paraphrases explore N different regions — their union has higher recall
#   than any single query alone.  The cross-encoder then precisely ranks
#   the merged candidate pool.
#
# Architecture:
#   query
#     │
#     ├─ LLM generates [q1, q2, q3]  (paraphrases)
#     │
#     ├─ base_retriever.invoke(q_original) → docs_0
#     ├─ base_retriever.invoke(q1)         → docs_1
#     ├─ base_retriever.invoke(q2)         → docs_2   } MultiQueryRetriever
#     └─ base_retriever.invoke(q3)         → docs_3
#             ↓  deduplicate by page_content
#         merged pool (~15–40 unique docs)
#             ↓  CrossEncoderReranker
#         top_n final results


def make_multiquery_reranking_retriever(
	base_retriever: BaseRetriever,
	llm,
	top_n:   int = 10,
) -> "CrossEncoderReranker":
	"""3D: MultiQueryRetriever → deduplicate → cross-encoder rerank.

	Args:
	    base_retriever : first-stage retriever (use 3B hybrid)
	    llm            : any LangChain chat model (e.g. ChatOpenAI, ChatAnthropic)
	    top_n          : final results to return after reranking
	"""
	# Hint: MultiQueryRetriever.from_llm() is the factory method.
	# Background: include_original=True adds the original unparaphrased query
	#   to the retrieval set, so you always have at least one run with the
	#   exact original phrasing — important for precise technical queries.
	# Answer line: MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm, include_original=True)
	from langchain_classic.retrievers import MultiQueryRetriever   # pip install langchain-classic

	multi_retriever = MultiQueryRetriever.from_llm(
		retriever=base_retriever,          # (think) which retriever does MultiQuery wrap?
		llm=llm,                # (easy) pass the llm through to generate multiple queries
		include_original=True,   # (think) should the original query also be retrieved?
	)

	# Step 2: load cross-encoder (reuse same model as 3C)
	# Hint: same CrossEncoder constructor as make_reranking_retriever.
	# Answer: CrossEncoder("BAAI/bge-reranker-v2-m3")
	model = CrossEncoder("BAAI/bge-reranker-v2-m3")   # (easy) — see hint above

	# Step 3: wrap MultiQueryRetriever with the cross-encoder reranker.
	# Note: fetch_k is not needed here — MultiQueryRetriever controls the
	#   pool size (n_queries × base k).  We pass a large top_n to the
	#   reranker so it sees the full merged pool.
	return CrossEncoderReranker(
		base_retriever=multi_retriever,     # (think) which retriever feeds candidates to the cross-encoder?
		model=model,
		top_n=top_n,
		fetch_k=200,                 # generous ceiling; actual pool size controlled by MultiQueryRetriever
	)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

# make_reranking_retriever: load bge-reranker-v2-m3 and wrap a base retriever.
# Input:  base_retriever — any BaseRetriever (pass your 3B hybrid here)
#         top_n          — final results returned to the caller
#         fetch_k        — candidate pool size fed to the cross-encoder
# Output: CrossEncoderReranker ready to call .invoke(query)


def make_reranking_retriever(
	base_retriever: BaseRetriever,
	top_n:   int = 10,
	fetch_k: int = 50,
) -> CrossEncoderReranker:
	"""3C: load the cross-encoder and return a reranking retriever.

	The CrossEncoder is loaded once here and reused across all queries —
	loading it inside _get_relevant_documents would re-download the model
	on every call.

	First run downloads ~1.1 GB to ~/.cache/huggingface/.
	Subsequent runs load from disk in ~3 s.
	"""
	# Hint: CrossEncoder takes a model name string as its first argument.
	# Background: "BAAI/bge-reranker-v2-m3" is the recommended open-source
	#   cross-encoder for retrieval reranking — multilingual, strong on
	#   scientific/technical text.
	# Answer: CrossEncoder("BAAI/bge-reranker-v2-m3")
	model = CrossEncoder("BAAI/bge-reranker-v2-m3")                                   # (easy) — see hint above

	return CrossEncoderReranker(
		base_retriever=base_retriever,
		model=model,
		top_n=top_n,
		fetch_k=fetch_k,                                            # (easy) pass the parameter through
	)


# ---------------------------------------------------------------------------
# Smoke test  (run: python3 -m src.retrieval.rerank)
# Prerequisites: chunks_recursive.jsonl, indices/chroma_recursive/
# First run downloads bge-reranker-v2-m3 (~1.1 GB) — be patient.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	from pathlib import Path
	from langchain_huggingface import HuggingFaceEmbeddings
	from .index import load_chroma_index, build_bm25_retriever
	from .retrievers import make_hybrid_retriever
	from ..chunking.stats import load_chunks_jsonl

	chunks     = load_chunks_jsonl(Path("chunks_recursive.jsonl"))
	embeddings = HuggingFaceEmbeddings(
		model_name="BAAI/bge-base-en-v1.5",
		encode_kwargs={"normalize_embeddings": True},
	)
	vectorstore = load_chroma_index(Path("indices/chroma_recursive"), embeddings)
	bm25        = build_bm25_retriever(chunks, k=50)
	hybrid      = make_hybrid_retriever(vectorstore, bm25, dense_weight=0.5, k=50)

	reranker = make_reranking_retriever(hybrid, top_n=10, fetch_k=50)

	probe   = "What is the peak power density achieved by the Ag28Cu72 parent alloy cathode?"
	results = reranker.invoke(probe)

	print(f"3C reranked top-{len(results)} for:\n  {probe!r}\n")
	for i, doc in enumerate(results):
		src = doc.metadata.get("source", "?").split("/")[-1]
		print(f"  [{i+1}] {src} :: chunk {doc.metadata.get('chunk_idx')}")
		print(f"       {doc.page_content[:100]!r}")

	# expected shape:
	# 3C reranked top-10:
	#   [1] AgCathode_Manuscript_04.08.2025_highlight_old.pdf :: chunk N
	#       'The Ag28Cu72 alloy achieved a peak power density of ...'

	# ---- 3D: multi-query + rerank -------------------------------------------
	from langchain_openai import ChatOpenAI   # pip install langchain-openai
	llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

	probe_3d = "How does the generation of aluminate ions (Al(OH)4-) practically limit the Al-Air battery?"
	mq_reranker = make_multiquery_reranking_retriever(hybrid, llm=llm, top_n=10)
	results_3d  = mq_reranker.invoke(probe_3d)

	print(f"\n3D multi-query+rerank top-{len(results_3d)} for:\n  {probe_3d!r}\n")
	for i, doc in enumerate(results_3d):
		src = doc.metadata.get("source", "?").split("/")[-1]
		print(f"  [{i+1}] {src} :: chunk {doc.metadata.get('chunk_idx')}")
		print(f"       {doc.page_content[:100]!r}")
	# expected: rank-1 should now be correct (this was rank-2 even for 3C)


# ---- HINTS (uncover only if stuck > 5 min) ----
# Step 1 (candidates):
#   self.base_retriever.invoke(query)
#
# Step 2 (pairs):
#   query is the string passed to _get_relevant_documents.
#   doc.page_content is the chunk text.
#   Answer: [(query, doc.page_content) for doc in candidates]
#
# Step 3 (scores):
#   self.model.predict(...)
#
# Step 4 (zip):
#   zip(scores, candidates)
#   key=lambda x: x[0]   ← index 0 is the score
#   reverse=True          ← highest score first
#
# Step 5 (return):
#   for _, doc in scored_docs[:self.top_n]
#   _ discards the score; self.top_n caps the slice.


# ---- Reflection question ----
# Q11: The cross-encoder sees (query, passage) together — why is this more
#      accurate than comparing query and passage embeddings independently?
#      And given that accuracy improvement, why don't we skip the first-stage
#      retriever entirely and just cross-encode every chunk in the corpus?
#      (Think about what changes when the corpus grows from 500 to 500,000 chunks.)
