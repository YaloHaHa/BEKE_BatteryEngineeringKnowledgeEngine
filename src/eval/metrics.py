"""Retrieval evaluation metrics — Hit@k, MRR, nDCG.

Day-4 deliverable: implement the three metric functions so evaluate_retriever()
can produce a single results dict for any retriever on any eval set.

Eval set format (eval_set_v1.jsonl, one JSON object per line):
    {
        "question"   : "What electrolyte is used in the Al-Air battery?",
        "answer"     : "KOH (6 M aqueous potassium hydroxide)",
        "source_hint": "AgCathode_Manuscript",
        "source"     : "/abs/path/to/AgCathode_Manuscript.docx",
        "page_idx"   : 1
    }

Ground truth for retrieval: the `source` field.
A retrieved Document is considered *relevant* if its metadata["source"] matches
the ground-truth source path exactly.

Metric overview
---------------
Hit@k           : 1.0 if the correct source appears in top-k, else 0.0.
                  Averaged across all queries → % of questions answered.
Reciprocal Rank : 1 / rank_of_first_relevant_doc.  If nothing relevant found → 0.0.
                  Averaged → MRR (Mean Reciprocal Rank).
nDCG@k          : DCG@k / IDCG@k.  Rewards high-ranking relevant results more
                  than low-ranking ones (logarithmic discount).
"""

import json
import math
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

# hit_at_k: did the correct source appear anywhere in the top-k results?
# Input:  results        — list[Document] from retriever.invoke(query)
#         relevant_source — ground-truth source path (str)
#         k              — how many of the ranked results to consider
# Output: 1.0 if any of results[:k] has metadata["source"] == relevant_source
#         else 0.0


def hit_at_k(results: list[Document], relevant_source: str, k: int) -> float:
	"""Return 1.0 if the correct source appears in the top-k results, else 0.0."""
	top_k = results[:k]

	# Step: check whether any document in top_k is relevant.
	# A document is relevant when its metadata["source"] matches relevant_source.
	# Wrap the result in float() to return 1.0 or 0.0 rather than True/False.
	return float(any(doc.metadata.get("source") == relevant_source for doc in top_k))


# reciprocal_rank: at what rank does the first correct result appear?
# Input:  results        — the full ranked list from the retriever
#         relevant_source — ground-truth source path
# Output: 1/rank if found (rank is 1-based), 0.0 if not found in the list
#
# Example:
#   results = [wrong, wrong, correct, wrong]
#   relevant = "thesis.docx"
#   → correct is at rank 3 → return 1/3 ≈ 0.333


def reciprocal_rank(results: list[Document], relevant_source: str) -> float:
	"""Return 1 / rank of the first relevant document (1-based), or 0.0.

	Example: correct result at rank 3 → return 1/3 ≈ 0.333
	"""
	# Step: walk results with a 0-based index; return as soon as you find a match.
	# rank is 1-based, so divide by (i + 1), not i.
	for i, doc in enumerate(results):
		if doc.metadata.get("source") == relevant_source:
			return 1 / (i + 1)   # (think) what does 1/rank look like using i?

	return 0   # (easy) what do you return when nothing matched?


# ndcg_at_k: how good is the overall ranking quality in the top-k results?
#
# DCG (Discounted Cumulative Gain) rewards relevant documents that appear
# earlier in the ranking:
#     DCG@k = Σ_{i=0}^{k-1}  rel_i / log2(i + 2)
# where rel_i = 1 if results[i] is relevant, else 0.
# The +2 shifts the denominator so rank-1 gets log2(2)=1.0, rank-2 gets log2(3)≈1.58, etc.
#
# IDCG (Ideal DCG) is the DCG you would achieve if all relevant documents in
# your top-k were ranked first.  For simplicity we compute it by sorting
# the relevance labels of results[:k] in descending order (1s first, 0s last).
#
# nDCG@k = DCG@k / IDCG@k   (in [0, 1]; higher is better)
# If IDCG == 0 (no relevant docs in top-k at all), return 0.0.


def ndcg_at_k(results: list[Document], relevant_source: str, k: int) -> float:
	"""Return nDCG@k for a single query.

	DCG@k = Σ_{i=0}^{k-1}  rel_i / log2(i + 2)
	nDCG@k = DCG@k / IDCG@k   where IDCG is DCG of the ideal (best possible) ranking.
	"""
	top_k = results[:k]

	# Step 1: build a binary relevance list — 1 if the doc matches, else 0
	rels = [1 if doc.metadata.get("source") == relevant_source else 0 for doc in top_k]                           # (think) 1 or 0 based on source match

	# Step 2: compute DCG — each position discounted by log2(rank + 1)
	# i is 0-based; rank is (i+1); discount denominator is log2(i+2)
	dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rels))   # (think) what goes in the blank?

	# Step 3: compute IDCG — same formula but on the ideal ordering of rels
	# Ideal ordering: put the 1s first, 0s last.
	ideal = sorted(rels, reverse=True)                         # (easy) which built-in sorts a list?
	idcg  = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))

	# Step 4: return the ratio; guard against division by zero
	return dcg / idcg if idcg > 0 else 0.0           # (think) what's the numerator? what's the zero-guard?


# ---------------------------------------------------------------------------
# Evaluation runner  (given — no blanks)
# ---------------------------------------------------------------------------

def evaluate_retriever(
	retriever: BaseRetriever,
	eval_set: list[dict],
	ks: list[int] = [1, 5, 10],
	verbose: bool = False,
) -> dict:
	"""Run the retriever on every question in eval_set; return averaged metrics.

	Args:
	    retriever : any LangChain BaseRetriever (3A, 3B, 3C, ...)
	    eval_set  : list of dicts with keys "question" and "source"
	    ks        : which k values to compute Hit@k and nDCG@k for
	    verbose   : if True, print per-question results

	Returns:
	    dict with keys like "hit@1", "hit@5", "hit@10", "mrr", "ndcg@5", ...
	    Values are averages across all eval_set items.
	"""
	max_k = max(ks)

	hit_sums  = {k: 0.0 for k in ks}
	ndcg_sums = {k: 0.0 for k in ks}
	rr_sum    = 0.0
	n         = len(eval_set)

	for item in eval_set:
		question = item["question"]
		gt_source = item["source"]

		results = retriever.invoke(question)            # list[Document]
		top_max = results[:max_k]

		rr_sum += reciprocal_rank(top_max, gt_source)

		for k in ks:
			hit_sums[k]  += hit_at_k(top_max, gt_source, k)
			ndcg_sums[k] += ndcg_at_k(top_max, gt_source, k)

		if verbose:
			first_hit = next(
				(i + 1 for i, d in enumerate(top_max)
				 if d.metadata.get("source") == gt_source),
				None,
			)
			print(f"  Q: {question[:60]!r}  first_hit_rank={first_hit}")

	metrics = {"mrr": round(rr_sum / n, 4)}
	for k in ks:
		metrics[f"hit@{k}"]  = round(hit_sums[k]  / n, 4)
		metrics[f"ndcg@{k}"] = round(ndcg_sums[k] / n, 4)
	return metrics


def load_eval_set(path: Path) -> list[dict]:
	"""Load a JSONL eval set (one JSON object per line)."""
	items = []
	with path.open(encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if line:
				items.append(json.loads(line))
	return items


# ---------------------------------------------------------------------------
# Smoke test  (run: python -m src.eval.metrics)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface
	# from langchain_community.embeddings import HuggingFaceEmbeddings  # deprecated
	from ..retrieval.index import load_chroma_index, build_bm25_retriever
	from ..retrieval.retrievers import make_dense_retriever, make_hybrid_retriever
	from ..chunking.stats import load_chunks_jsonl
	from ..retrieval.rerank import make_reranking_retriever

	eval_path = Path("eval/eval_set_v2.jsonl")
	chunk_path = Path("chunks_recursive.jsonl")

	if not eval_path.exists() or not chunk_path.exists():
		print("ERROR: need eval/eval_set_v2.jsonl and chunks_recursive.jsonl")
		raise SystemExit(1)

	eval_set = load_eval_set(eval_path)
	chunks   = load_chunks_jsonl(chunk_path)
	print(f"eval set  : {len(eval_set)} questions")
	print(f"chunks    : {len(chunks):,}")

	embeddings = HuggingFaceEmbeddings(
		model_name="BAAI/bge-base-en-v1.5",
		encode_kwargs={"normalize_embeddings": True},
	)
	vectorstore = load_chroma_index(Path("indices/chroma_recursive"), embeddings)
	bm25        = build_bm25_retriever(chunks, k=10)

	dense  = make_dense_retriever(vectorstore, k=10)
	hybrid = make_hybrid_retriever(vectorstore, bm25, dense_weight=0.5, k=10)

	print("\n--- 3A dense ---")
	m_dense = evaluate_retriever(dense, eval_set, verbose=True)
	for key, val in m_dense.items():
		print(f"  {key:<12} {val}")

	print("\n--- 3B hybrid (w=0.5) ---")
	m_hybrid = evaluate_retriever(hybrid, eval_set, verbose=True)
	for key, val in m_hybrid.items():
		print(f"  {key:<12} {val}")
	
	# print("\n--- 3C rerank (bge-reranker-v2-m3) ---")

	# hybrid_for_rerank = make_hybrid_retriever(vectorstore, bm25, dense_weight=0.5, k=50)
	# reranker = make_reranking_retriever(hybrid_for_rerank, top_n=10, fetch_k=50)
	# m_rerank = evaluate_retriever(reranker, eval_set, verbose=True)
	# for key, val in m_rerank.items():
	# 	print(f"  {key:<12} {val}")

	print("\n--- 3D multi-query + rerank ---")
	from langchain_openai import ChatOpenAI
	from ..retrieval.rerank import make_multiquery_reranking_retriever

	llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
	hybrid_for_3d = make_hybrid_retriever(vectorstore, bm25, dense_weight=0.5, k=10)
	mq_reranker = make_multiquery_reranking_retriever(hybrid_for_3d, llm=llm, top_n=10)
	m_3d = evaluate_retriever(mq_reranker, eval_set, verbose=True)
	for key, val in m_3d.items():
		print(f"  {key:<12} {val}")

	print("\n--- 3E contextual retrieval ---")
	# Prerequisites: run build_contextual_corpus.py first to produce:
	#   contextual_chunks.jsonl  — chunks with LLM-generated context prepended
	#   indices/chroma_contextual — Chroma index built from those chunks
	contextual_chunk_path  = Path("contextual_chunks.jsonl")
	contextual_index_path  = Path("indices/chroma_contextual")

	if not contextual_chunk_path.exists() or not contextual_index_path.exists():
		print("  SKIP: run build_contextual_corpus.py first to generate contextual chunks and index.")
	else:
		from ..chunking.stats import load_chunks_jsonl as _load
		contextual_chunks  = _load(contextual_chunk_path)
		vectorstore_3e     = load_chroma_index(contextual_index_path, embeddings)
		bm25_3e            = build_bm25_retriever(contextual_chunks, k=50)
		hybrid_3e          = make_hybrid_retriever(vectorstore_3e, bm25_3e, dense_weight=0.5, k=50)
		reranker_3e        = make_reranking_retriever(hybrid_3e, top_n=10, fetch_k=50)
		m_3e = evaluate_retriever(reranker_3e, eval_set, verbose=True)
		for key, val in m_3e.items():
			print(f"  {key:<12} {val}")

	# expected shape (numbers vary by corpus and eval set size):
	# --- 3A dense ---
	#   hit@1        0.4667
	#   hit@5        0.8000
	#   hit@10       0.8667
	#   mrr          0.5833
	#   ndcg@1       0.4667
	#   ndcg@5       0.6542
	#   ndcg@10      0.6712
	# --- 3B hybrid (w=0.5) ---
	#   hit@1        0.5333    ← typically higher than dense
	#   ...


# ---- HINTS (uncover only if stuck > 5 min) ----
# hit_at_k:
#   Concept: scan the first k results, check if any have the right source.
#   One-liner: any(d.metadata.get("source") == relevant_source for d in results[:k])
#   Wrap in float() to return 1.0 / 0.0.
#
# reciprocal_rank:
#   Concept: find the first relevant doc, return 1 / (index + 1).
#   Pattern:
#       for i, doc in enumerate(results):
#           if doc.metadata.get("source") == relevant_source:
#               return 1.0 / (i + 1)
#       return 0.0
#
# ndcg_at_k:
#   Concept: DCG = sum over top-k of rel_i / log2(i + 2); IDCG = DCG of ideal ranking.
#   Pattern:
#       top_k  = results[:k]
#       rels   = [1 if d.metadata.get("source") == relevant_source else 0 for d in top_k]
#       dcg    = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
#       ideal  = sorted(rels, reverse=True)
#       idcg   = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
#       return dcg / idcg if idcg > 0 else 0.0


# ---- Reflection questions ----
# Q9 (MRR vs Hit@k): MRR and Hit@k are related but different.
#     Give a concrete example where Hit@5 = 1.0 for two retrievers, but their
#     MRR values are 0.2 and 1.0 respectively.  Which metric is more informative
#     for a use-case where the LLM only uses the top-1 result?
#
# Q10 (nDCG vs MRR): nDCG and MRR both discount lower-ranked results, but
#     nDCG sums across all k positions.  When would nDCG@10 be a more useful
#     signal than MRR?  (Think: what if multiple relevant chunks exist?)
