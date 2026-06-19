"""Unit tests for retrieval — Hit@1 smoke test  —  Day 7 scaffold.

These tests are *integration* tests: they load real indices from disk
and run a real retriever.  They're slower than unit tests (seconds, not
milliseconds) but they verify the full retrieval stack works end-to-end.

Convention: prefix with `test_` so pytest discovers them automatically.

Running:
    pytest tests/test_retrieval.py -v

Prerequisites (must exist before running):
    contextual_chunks.jsonl
    indices/chroma_contextual/

Skip message: if the prerequisites are missing, the tests are skipped
gracefully rather than failing with a confusing import error.

# pip install pytest langchain-huggingface chromadb rank-bm25
"""

import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Skip the whole module if the required indices don't exist yet
# ---------------------------------------------------------------------------

CHUNK_PATH = Path("contextual_chunks.jsonl")
INDEX_PATH = Path("indices/chroma_contextual")

# pytestmark applies a marker to every test in this file.
# skipif skips the test when the condition is True.
pytestmark = pytest.mark.skipif(
    not (CHUNK_PATH.exists() and INDEX_PATH.exists()),
    reason="contextual_chunks.jsonl or indices/chroma_contextual not found — run build_contextual_corpus.py first",
)


# ---------------------------------------------------------------------------
# Module-level fixture: build the 3E retriever once for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever_3e():
    """Build the winning 3E retriever once and share it across all tests.

    scope="module" means pytest runs this fixture only ONCE per test file —
    building the index + loading the model is expensive, so we don't repeat it.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.retrieval.index import load_chroma_index, build_bm25_retriever
    from src.retrieval.retrievers import make_hybrid_retriever
    from src.retrieval.rerank import make_reranking_retriever
    from src.chunking.stats import load_chunks_jsonl

    # Arrange: load embeddings + chunks + indices
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    chunks      = load_chunks_jsonl(CHUNK_PATH)
    vectorstore = load_chroma_index(INDEX_PATH, embeddings)
    bm25        = build_bm25_retriever(chunks, k=50)
    hybrid      = make_hybrid_retriever(vectorstore, bm25, dense_weight=0.5, k=50)
    retriever   = make_reranking_retriever(hybrid, top_n=5, fetch_k=50)
    return retriever


# ---------------------------------------------------------------------------
# Test 1 — retriever returns documents
# ---------------------------------------------------------------------------

def test_retriever_returns_docs(retriever_3e):
    """A basic query should return at least one Document."""
    # Act: invoke the retriever with a simple query
    # Hint: LangChain retrievers expose .invoke(query) → list[Document]
    # Background: if invoke() returns [] there's an indexing problem
    # Answer: docs = retriever_3e.invoke("silver catalyst")
    docs = retriever_3e.invoke("silver catalyst")   # (easy)

    # Assert: got at least one result
    assert len(docs) > 0


# ---------------------------------------------------------------------------
# Test 2 — Hit@1 on a known query
# ---------------------------------------------------------------------------

def test_hit_at_1_silver_cathode(retriever_3e):
    """The top result for the Ag28Cu72 query must come from AgCathode_Manuscript.docx."""
    # This is the first question in eval_set_v1.jsonl — we know the correct source.
    query           = "What is the peak power density achieved by the Ag28Cu72 parent alloy cathode?"
    expected_source = "AgCathode_Manuscript_04.08.2025_highlight_old.pdf"

    # Act
    docs = retriever_3e.invoke(query)

    # Arrange: extract the filename from the top-ranked document's metadata
    # Hint: Path(doc.metadata["source"]).name gives you just the filename
    # Background: metadata["source"] is the full file path; .name strips the directory
    # Answer: top_source = Path(docs[0].metadata["source"]).name
    top_source = Path(docs[0].metadata["source"]).name   # (think)

    # Assert: the top result is from the right file
    assert top_source == expected_source, (
        f"Hit@1 miss — expected {expected_source!r}, got {top_source!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — all returned documents have page_content
# ---------------------------------------------------------------------------

def test_returned_docs_have_content(retriever_3e):
    """Every returned Document must have non-empty page_content."""
    query = "gas diffusion layer role"
    docs  = retriever_3e.invoke(query)

    # Assert: no empty content
    # Hint: iterate and check each doc's .page_content attribute
    # Background: an empty page_content means the embedding was computed on '' —
    #   it will score badly and pollute the context window
    # Answer: for doc in docs: assert len(doc.page_content) > 0
    for doc in docs:
        assert len(doc.page_content) > 0   # (think)


# ---------------------------------------------------------------------------
# Test 4 — top-5 cap is respected
# ---------------------------------------------------------------------------

def test_retriever_respects_top_n(retriever_3e):
    """The reranker is configured with top_n=5 — result count must be ≤ 5."""
    docs = retriever_3e.invoke("aluminate ions electrolyte concentration")

    # Assert: at most 5 documents returned
    # Hint: len() + <= operator
    # Background: if more than top_n are returned, the reranker didn't apply
    # Answer: assert len(docs) <= 5
    assert len(docs) <=5   # (easy)


# ---- HINTS (uncover only if stuck > 5 min) ----
# test_retriever_returns_docs:
#   docs = retriever_3e.invoke("silver catalyst")
#
# test_hit_at_1_silver_cathode:
#   top_source = Path(docs[0].metadata["source"]).name
#
# test_returned_docs_have_content:
#   assert len(doc.page_content) > 0
#
# test_retriever_respects_top_n:
#   assert len(docs) <= 5


# ---- Reflection question ----
# Q: test_hit_at_1_silver_cathode hard-codes `expected_source = "AgCathode_Manuscript.docx"`.
#    If you added a new document to the corpus that's more relevant to that query,
#    the test would fail even though the retriever is working correctly.
#    How would you make this test more robust to corpus changes?
