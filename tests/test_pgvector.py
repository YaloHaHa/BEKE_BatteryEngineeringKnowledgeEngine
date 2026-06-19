"""Phase 2 Day 2 — Round-trip validation: pgvector vs Chroma baseline.

What this test does
-------------------
1. Loads the first 100 contextual chunks (Phase 1 corpus subset).
2. Upserts them into Aurora pgvector via build_pgvector_index().
3. Runs 5 eval questions whose gold answer lives in those 100 chunks.
4. Asserts Hit@1 == 1.0 — matching the Phase 1 Chroma baseline.

Prerequisites (run once before executing this test)
---------------------------------------------------
* SSH tunnel open — see docs/aws_setup.md for connection details
* .env in repo root contains AURORA_DB_URL (see docs/aws_setup.md)

Run:
    pytest tests/test_pgvector.py -v
or directly:
    python tests/test_pgvector.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Pytest skip guard — skip if tunnel / env not available
# ---------------------------------------------------------------------------

AURORA_URL = os.getenv("AURORA_DB_URL", "")
pytestmark = pytest.mark.skipif(
    not AURORA_URL,
    reason=(
        "AURORA_DB_URL not set — open the SSH tunnel and add it to .env. "
        "See the module docstring for instructions."
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
CHUNK_PATH = REPO_ROOT / "contextual_chunks.jsonl"
EVAL_PATH  = REPO_ROOT / "eval" / "eval_set_v2.jsonl"
COLLECTION = "beke_pgvector_test"   # isolated collection — won't touch production data
N_CHUNKS   = 100
N_EVAL     = 5
TOP_K      = 10                     # fetch_k for Hit@1 — generous window


def _load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_chunks(n: int = N_CHUNKS):
    from src.chunking.stats import load_chunks_jsonl
    all_chunks = load_chunks_jsonl(CHUNK_PATH)
    return all_chunks[:n]


def _covered_eval_questions(chunks) -> list[dict]:
    """Return eval questions whose gold source is covered by `chunks`."""
    covered_sources = {c.source.split("/")[-1] for c in chunks}
    with open(EVAL_PATH) as f:
        evals = [json.loads(line) for line in f]
    return [e for e in evals if e["source"].split("/")[-1] in covered_sources]


def _hit_at_k(results, gold_source: str, k: int) -> bool:
    """True if any of the top-k results comes from gold_source."""
    gold_filename = gold_source.split("/")[-1]
    for doc in results[:k]:
        doc_filename = doc.metadata.get("source", "").split("/")[-1]
        if doc_filename == gold_filename:
            return True
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embeddings():
    return _load_embeddings()


@pytest.fixture(scope="module")
def vectorstore(embeddings):
    """Build a fresh pgvector index with 100 chunks. Module-scoped for speed."""
    from src.retrieval.index_pgvector import build_pgvector_index

    chunks = _load_chunks(N_CHUNKS)
    print(f"\n[fixture] Upserting {len(chunks)} chunks to collection '{COLLECTION}'...")
    vs = build_pgvector_index(
        chunks,
        embeddings,
        collection_name=COLLECTION,
        pre_delete_collection=True,   # clean slate every test run
    )
    print("[fixture] Upsert complete.")
    return vs


@pytest.fixture(scope="module")
def eval_questions():
    chunks = _load_chunks(N_CHUNKS)
    questions = _covered_eval_questions(chunks)
    assert len(questions) >= N_EVAL, (
        f"Need at least {N_EVAL} eval questions covered by the {N_CHUNKS}-chunk subset, "
        f"but only found {len(questions)}."
    )
    return questions[:N_EVAL]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPGVectorRoundTrip:

    def test_upsert_count(self, vectorstore):
        """Vectorstore must be non-empty after upsert."""
        results = vectorstore.similarity_search("aluminium", k=1)
        assert len(results) == 1, "pgvector returned no results — upsert may have failed."

    def test_hit_at_1_equals_baseline(self, vectorstore, eval_questions):
        """Hit@1 must equal 1.0 — matching the Phase 1 Chroma baseline."""
        hits = 0
        misses = []

        for item in eval_questions:
            query  = item["question"]
            gold   = item["source"]
            results = vectorstore.similarity_search(query, k=TOP_K)
            hit = _hit_at_k(results, gold, k=1)
            if hit:
                hits += 1
            else:
                misses.append({
                    "query": query[:80],
                    "gold":  gold.split("/")[-1],
                    "top1":  results[0].metadata.get("source", "?").split("/")[-1] if results else "—",
                })

        hit_at_1 = hits / len(eval_questions)
        print(f"\nHit@1 = {hit_at_1:.4f}  ({hits}/{len(eval_questions)} questions)")
        if misses:
            print("Misses:")
            for m in misses:
                print(f"  Q: {m['query']}")
                print(f"     gold={m['gold']}  top1={m['top1']}")

        assert hit_at_1 == 1.0, (
            f"Hit@1 = {hit_at_1:.4f} — expected 1.0 (Phase 1 Chroma baseline). "
            f"See misses above."
        )

    def test_result_metadata_schema(self, vectorstore):
        """Every returned document must carry the expected metadata keys."""
        required_keys = {"source", "chunk_idx", "chunk_id", "chunker"}
        results = vectorstore.similarity_search("electrolyte conductivity", k=5)
        for doc in results:
            missing = required_keys - set(doc.metadata.keys())
            assert not missing, f"Document missing metadata keys: {missing}"

    def test_similarity_scores_ordered(self, vectorstore):
        """Scores from similarity_search_with_score must be non-decreasing (ascending distance)."""
        results = vectorstore.similarity_search_with_score("oxygen reduction reaction", k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores), (
            f"Scores not in ascending order (lower = more similar for L2): {scores}"
        )

    def test_retriever_interface(self, vectorstore):
        """as_retriever().invoke() must return Document objects (drop-in contract)."""
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke("Al-Air battery discharge")
        assert len(docs) > 0
        assert all(hasattr(d, "page_content") for d in docs)


# ---------------------------------------------------------------------------
# Stand-alone runner  (python tests/test_pgvector.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))  # allow `src.*` imports when run directly
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    if not os.getenv("AURORA_DB_URL"):
        print(
            "ERROR: AURORA_DB_URL not set.\n"
            "Open the SSH tunnel then add AURORA_DB_URL to .env.\n"
            "See docs/aws_setup.md for connection details."
        )
        sys.exit(1)

    print("Loading embeddings (first run downloads ~440 MB)...")
    emb = _load_embeddings()

    print(f"Loading first {N_CHUNKS} contextual chunks...")
    chunks = _load_chunks(N_CHUNKS)

    from src.retrieval.index_pgvector import build_pgvector_index
    vs = build_pgvector_index(
        chunks, emb,
        collection_name=COLLECTION,
        pre_delete_collection=True,
    )

    questions = _covered_eval_questions(chunks)[:N_EVAL]
    print(f"\nRunning {len(questions)} eval questions...")

    hits = 0
    for item in questions:
        q   = item["question"]
        gold = item["source"]
        res  = vs.similarity_search(q, k=TOP_K)
        hit  = _hit_at_k(res, gold, k=1)
        hits += int(hit)
        status = "✓" if hit else "✗"
        top1_src = res[0].metadata.get("source", "?").split("/")[-1] if res else "—"
        print(f"  {status} {q[:70]}")
        if not hit:
            print(f"      gold={gold.split('/')[-1]}  top1={top1_src}")

    hit_at_1 = hits / len(questions)
    print(f"\n{'='*60}")
    print(f"Hit@1 = {hit_at_1:.4f}  ({hits}/{len(questions)})")
    print(f"Phase 1 Chroma baseline: 1.0000")
    if hit_at_1 == 1.0:
        print("PASS — pgvector matches Chroma baseline ✓")
    else:
        print("FAIL — pgvector below Chroma baseline ✗")
        sys.exit(1)
