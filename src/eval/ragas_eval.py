"""Day 6B — RAGAS answer-quality evaluation.

metrics.py measures retrieval quality: did we find the right chunk?
ragas_eval.py measures answer quality: is the generated answer correct?

The four RAGAS metrics
----------------------
Faithfulness       : Does every claim in the answer appear in the retrieved context?
                     Catches hallucination — the LLM inventing facts not in the passages.
                     Score: fraction of answer claims that are grounded in the context.

Answer relevance   : Does the answer actually address the question asked?
                     Catches topic drift — the LLM answering a related but different question.
                     Score: cosine similarity between question embedding and answer embedding.

Context precision  : Are the retrieved chunks ranked with the most useful ones first?
                     A retriever that puts noise at rank 1 and the answer at rank 5 scores low.
                     Score: precision@k averaged over ranks, weighted by position.

Context recall     : Did the retrieved chunks contain all the information needed to answer?
                     Requires a ground-truth answer to compare against.
                     Score: fraction of ground-truth sentences entailed by the context.

RAGAS dataset format
--------------------
RAGAS expects a HuggingFace Dataset with four columns:
    question       : str   — the original query
    answer         : str   — the LLM-generated answer
    contexts       : list[str] — the retrieved chunk texts (list, not a single string)
    ground_truth   : str   — the reference answer from the eval set

Install:  pip install ragas datasets
"""

import json
from pathlib import Path

from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

# build_ragas_dataset: convert parallel lists into a HuggingFace Dataset
# that ragas.evaluate() can consume.
#
# Input:  questions      — list[str]        one per eval item
#         answers        — list[str]        LLM-generated answers
#         contexts_list  — list[list[str]]  retrieved chunk texts per question
#         ground_truths  — list[str]        reference answers from eval set
# Output: datasets.Dataset with columns: question, answer, contexts, ground_truth


def build_ragas_dataset(
    questions:     list[str],
    answers:       list[str],
    contexts_list: list[list[str]],
    ground_truths: list[str],
):
    """Pack parallel lists into a HuggingFace Dataset for ragas.evaluate().

    The four lists must be the same length — one entry per eval question.
    """
    from datasets import Dataset   # pip install datasets

    # Step 1: assemble a dict mapping column name → list of values.
    # RAGAS requires exactly these four column names.
    data = {
        "question":     questions,   # (easy) which list?
        "answer":       answers,   # (easy)
        "contexts":     contexts_list,   # (easy) list of lists — one list of strings per question
        "ground_truth": ground_truths,   # (easy)
    }

    # Step 2: wrap the dict in a HuggingFace Dataset
    # Hint: Dataset has a class method .from_dict() that takes a column dict.
    return Dataset.from_dict(data)   # (think) class method + argument


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

# run_ragas: iterate over the eval set, generate answers, collect contexts,
# build the Dataset, call ragas.evaluate(), return a results dict.
#
# Input:  eval_set  — list[dict] from load_eval_set() — needs "question" and "answer" keys
#         retriever — the winning retriever (3E)
#         llm       — chat model used for generation
#         metrics   — list of ragas Metric objects (default: all four)
#         k         — chunks per question passed to the LLM
# Output: dict mapping metric name → float score


def run_ragas(
    eval_set:  list[dict],
    retriever: BaseRetriever,
    llm:       BaseLanguageModel,
    metrics:   list | None = None,
    k:         int = 5,
) -> dict:
    """Generate answers for every eval item, then score with RAGAS.

    Args:
        eval_set  : list of dicts with keys 'question' and 'answer'
        retriever : any BaseRetriever (use 3E for best results)
        llm       : LangChain chat model used both for generation and RAGAS LLM judge
        metrics   : RAGAS metrics to compute (default: faithfulness, answer_relevancy,
                    context_precision, context_recall)
        k         : how many retrieved chunks to pass to the LLM
    """
    from ragas import evaluate
    # Use the legacy singletons from ragas.metrics (not ragas.metrics.collections).
    # These are pre-instantiated dataclass objects with llm=None; evaluate() injects
    # the LLM and embeddings automatically.  The new ragas.metrics.collections classes
    # require llm as a required positional arg at construction time — a different API.
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_huggingface import HuggingFaceEmbeddings
    from ..generation.generator import ask

    # Step 1: default metric set — pre-instantiated singletons, no ()
    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # Step 2: iterate — generate answer + collect context texts for each question
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in eval_set:
        question     = item["question"]
        ground_truth = item["answer"]

        rag_answer  = ask(question, retriever, llm, k=k)
        chunk_texts = [doc.page_content for doc in rag_answer.sources]

        questions.append(question)
        answers.append(rag_answer.answer)
        contexts_list.append(chunk_texts)
        ground_truths.append(ground_truth)

        print(f"  answered: {question[:60]!r}")

    # Step 3: build the RAGAS dataset (HF Dataset with legacy column names;
    # evaluate() calls convert_v1_to_v2_dataset internally to rename them)
    dataset = build_ragas_dataset(questions, answers, contexts_list, ground_truths)

    # Step 4: run evaluation.
    # - llm: pass the raw LangChain chat model; evaluate() wraps it in
    #        LangchainLLMWrapper internally (accepts LangchainLLM type).
    # - embeddings: answer_relevancy needs an embeddings model for cosine similarity.
    #   Pass HuggingFaceEmbeddings directly (it's a LangchainEmbeddings subclass);
    #   evaluate() wraps it in LangchainEmbeddingsWrapper, which calls embed_query()
    #   on it.  HuggingFaceEmbeddings has embed_query; OpenAIEmbeddings does not in
    #   newer langchain-openai — that was the original AttributeError.
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=hf_embeddings,
    )

    # Step 5: convert to {metric_name: mean_float}.
    # EvaluationResult.__getitem__ takes a string key, not an integer, so dict(result)
    # fails with KeyError: 0.  result._repr_dict is populated in __post_init__ with
    # exactly the per-metric mean scores we want.
    return dict(result._repr_dict)


# ---------------------------------------------------------------------------
# Smoke test  (run: python3 -m src.eval.ragas_eval)
# Prerequisites: contextual_chunks.jsonl, indices/chroma_contextual,
#                eval_set_v1.jsonl (needs "answer" field for ground_truth)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path as _Path
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ..retrieval.index import load_chroma_index, build_bm25_retriever
    from ..retrieval.retrievers import make_hybrid_retriever
    from ..retrieval.rerank import make_reranking_retriever
    from ..chunking.stats import load_chunks_jsonl
    from .metrics import load_eval_set

    contextual_chunk_path = _Path("contextual_chunks.jsonl")
    contextual_index_path = _Path("indices/chroma_contextual")
    eval_path             = _Path("eval/eval_set_v1.jsonl")

    if not contextual_chunk_path.exists() or not contextual_index_path.exists():
        print("ERROR: run build_contextual_corpus.py first")
        raise SystemExit(1)
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found")
        raise SystemExit(1)

    # Build 3E retriever
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    chunks      = load_chunks_jsonl(contextual_chunk_path)
    vectorstore = load_chroma_index(contextual_index_path, embeddings)
    bm25        = build_bm25_retriever(chunks, k=50)
    hybrid      = make_hybrid_retriever(vectorstore, bm25, dense_weight=0.5, k=50)
    retriever   = make_reranking_retriever(hybrid, top_n=5, fetch_k=50)

    llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_set = load_eval_set(eval_path)

    # Run on first 5 questions to keep cost low during development
    subset = eval_set[:5]
    print(f"Running RAGAS on {len(subset)} questions (subset)...")
    scores = run_ragas(subset, retriever, llm)

    print("\nRAGAS scores:")
    for metric, score in scores.items():
        print(f"  {metric:<25} {score:.4f}")

    # expected shape:
    # RAGAS scores:
    #   faithfulness              0.95xx
    #   answer_relevancy          0.93xx
    #   context_precision         0.91xx
    #   context_recall            0.88xx


# ---- HINTS (uncover only if stuck > 5 min) ----
# build_ragas_dataset:
#   data = {"question": questions, "answer": answers,
#           "contexts": contexts_list, "ground_truth": ground_truths}
#   Dataset.from_dict(data)
#
# run_ragas — metrics default:
#   [faithfulness, answer_relevancy, context_precision, context_recall]
#
# run_ragas — item keys:
#   item["question"], item["answer"]
#
# chunk_texts:
#   [doc.page_content for doc in rag_answer.sources]
#
# evaluate call:
#   evaluate(dataset, metrics=metrics)
#
# return:
#   dict(result)


# ---- Reflection questions ----
# Q15: Faithfulness and context_recall both measure grounding, but in opposite
#      directions. Explain the difference with a concrete example where one
#      score is high and the other is low.
#
# Q16: RAGAS uses an LLM as a judge to score faithfulness. What does this mean
#      for reproducibility? What could cause the same pipeline to score
#      differently on two consecutive runs?
