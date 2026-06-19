"""Challenge 3 — Retriever wrappers: 3A (dense) and 3B (hybrid BM25+dense).

Day-4 deliverable: implement make_dense_retriever and make_hybrid_retriever.
Day-5 additions (3C rerank, 3D multi-query, 3E contextual) live in rerank.py.

Retriever contract
------------------
Every function here returns a LangChain BaseRetriever, meaning the caller
can always do:

    results: list[Document] = retriever.invoke("my query string")

This uniform interface is what lets run_eval.py swap retrievers without
changing any eval logic.
"""

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever   # pip install langchain-classic
from langchain_core.retrievers import BaseRetriever


# ---------------------------------------------------------------------------
# 3A: Dense retriever
# ---------------------------------------------------------------------------

# make_dense_retriever: wrap a Chroma vectorstore as a standard BaseRetriever.
# Input:  vectorstore — a Chroma index built by build_chroma_index()
#         k — number of nearest neighbours to return per query
# Output: BaseRetriever whose .invoke(query) → list[Document]


def make_dense_retriever(vectorstore: Chroma, k: int = 10) -> BaseRetriever:
	"""3A: pure dense retrieval — cosine nearest neighbours in embedding space.

	Strengths : semantic similarity, paraphrase, multi-lingual
	Weaknesses: exact keywords (acronyms, formulae, author names, model codes)
	"""
	# Hint: Chroma objects expose .as_retriever() which returns a BaseRetriever.
	# Background: search_kwargs={"k": k} controls how many neighbours are returned.
	#   search_type defaults to "similarity" (cosine); "mmr" enables diversity re-ranking.
	# Answer: vectorstore.as_retriever(search_kwargs={"k": k})
	return vectorstore.as_retriever(search_kwargs={"k": k})   # (easy) — see hint above


# ---------------------------------------------------------------------------
# 3B: Hybrid retriever (BM25 + dense via EnsembleRetriever)
# ---------------------------------------------------------------------------

# make_hybrid_retriever: combine BM25 (sparse) and Chroma (dense) rankings
# using Reciprocal Rank Fusion.
#
# How EnsembleRetriever works:
#   1. Both sub-retrievers independently produce a ranked list of k documents.
#   2. Each document's RRF score from retriever_i:
#        rrf_score(doc) = weight_i / (rank_i + 60)
#      The constant 60 prevents rank-1 from dominating.
#   3. Scores are summed across retrievers; the merged list is sorted by total score.
#
# RRF is rank-based — raw cosine distances and BM25 scores are never compared
# directly (they live on incompatible scales).  This is what makes the
# combination robust without calibration.
#
# dense_weight tuning guide:
#   0.3 → BM25 dominates (good for keyword-heavy / acronym queries)
#   0.5 → equal contribution (safe default)
#   0.7 → dense dominates (good for paraphrase / conceptual queries)


def make_hybrid_retriever(
	vectorstore: Chroma,
	bm25_retriever: BM25Retriever,
	dense_weight: float = 0.5,
	k: int = 10,
) -> EnsembleRetriever:
	"""3B: hybrid BM25 + dense via Reciprocal Rank Fusion."""

	# Step 1: wrap the vectorstore as a dense retriever at the same pool size
	dense_retriever = make_dense_retriever(vectorstore,k)                         # (easy) call make_dense_retriever with vectorstore and k

	# Step 2: match BM25's result pool to dense's so both contribute equally
	bm25_retriever.k = k                        # (easy) attribute name that controls BM25 result count

	# Step 3: wire both into EnsembleRetriever
	# Background: EnsembleRetriever.weights is a list aligned with .retrievers.
	#   weights[0] applies to retrievers[0], weights[1] to retrievers[1].
	#   They must sum to 1.0 exactly.
	# Think: if dense_weight=0.5, what is the BM25 weight?
	#   And which retriever goes first in the list — does order matter for RRF?
	return EnsembleRetriever(
		retrievers=[bm25_retriever, dense_retriever],               # (think) correct order: BM25 first, dense second
		weights=[1-dense_weight, dense_weight],              # (think) express bm25_weight in terms of dense_weight
	)


# ---------------------------------------------------------------------------
# Smoke test  (run: python -m src.retrieval.retrievers)
# Prerequisites: chunks_recursive.jsonl, indices/chroma_recursive/ must exist.
#   Run src/retrieval/index.py smoke test first if they don't.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	from pathlib import Path
	from langchain_huggingface import HuggingFaceEmbeddings
	from .index import load_chroma_index, build_bm25_retriever
	from ..chunking.stats import load_chunks_jsonl

	chunks = load_chunks_jsonl(Path("chunks_recursive.jsonl"))
	embeddings = HuggingFaceEmbeddings(
		model_name="BAAI/bge-base-en-v1.5",
		encode_kwargs={"normalize_embeddings": True},
	)
	vectorstore = load_chroma_index(Path("indices/chroma_recursive"), embeddings)
	bm25        = build_bm25_retriever(chunks, k=10)

	# ---- 3A probe -----------------------------------------------------------
	dense   = make_dense_retriever(vectorstore, k=5)
	probe   = "What electrolyte is used in the Al-Air battery?"
	results = dense.invoke(probe)
	print(f"3A dense — top-{len(results)} for: {probe!r}")
	for i, doc in enumerate(results):
		print(f"  [{i+1}] chunk {doc.metadata.get('chunk_idx')} | {doc.metadata.get('source','?').split('/')[-1]}")
		print(f"       {doc.page_content[:]!r}")

	# ---- 3B weight sweep ----------------------------------------------------
	print()
	for w in [0.3, 0.5, 0.7]:
		hybrid  = make_hybrid_retriever(vectorstore, bm25, dense_weight=w, k=5)
		results = hybrid.invoke(probe)
		sources = {doc.metadata.get("source","?").split("/")[-1] for doc in results}
		print(f"3B hybrid dense_weight={w} → {sources}")
		print(f"  top-{len(results)} chunks:")
		for i, doc in enumerate(results):
			print(f"    [{i+1}] chunk {doc.metadata.get('chunk_idx')} | {doc.metadata.get('source','?').split('/')[-1]}")
			print(f"         {doc.page_content[:]!r}")

	# expected shape:
	# 3A dense — top-5 for: 'What electrolyte is used...'
	#   [1] chunk 7 | AgCathode_Manuscript.docx
	#       'The electrolyte used in this study is ...'
	# 3B hybrid dense_weight=0.3 → {'AgCathode_Manuscript.docx', ...}
	# 3B hybrid dense_weight=0.5 → {...}
	# 3B hybrid dense_weight=0.7 → {...}


# ---- HINTS (uncover only if stuck > 5 min) ----
# make_dense_retriever:
#   method name: as_retriever
#   k goes inside search_kwargs: {"k": k}
#
# make_hybrid_retriever — blank 1 (dense_retriever):
#   make_dense_retriever(vectorstore, k=k)
#
# make_hybrid_retriever — blank 2 (attribute name):
#   bm25_retriever.k — same attribute as in build_bm25_retriever
#
# make_hybrid_retriever — blank 3 (retrievers list):
#   [bm25_retriever, dense_retriever]
#   Order matters only for the weights alignment, not for RRF math.
#
# make_hybrid_retriever — blank 4 (bm25 weight):
#   1 - dense_weight  (since both weights must sum to 1.0)


# ---- Reflection questions ----
# Q7 (3A vs 3B): Run both retrievers on a query containing a specific acronym
#     (e.g. "OER" — oxygen evolution reaction).  Which retriever ranks the
#     correct chunk higher?  Why does BM25 do better here, and what would
#     change if you used a domain-adapted embedding model trained on battery
#     literature instead of general-purpose BGE?
#
# Q8 (weight sweep): The three configs above (0.3 / 0.5 / 0.7) are manual.
#     What is the principled way to pick the best weight?  Describe in two
#     sentences using the eval set and Hit@5.
