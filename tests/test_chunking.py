"""Unit tests for src/chunking/chunkers.py  —  Day 7 scaffold.

What is a unit test?
--------------------
A unit test is a short, focused function that checks ONE thing about your code.
Each test follows the same 3-step pattern:

    Arrange  →  set up the inputs you need
    Act      →  call the function you're testing
    Assert   →  check the output matches what you expect

If the assertion is True  → test PASSES (green).
If the assertion is False → test FAILS (red) and shows you exactly what went wrong.

Why bother?
-----------
After you change any code, run `pytest tests/` in one second to know whether
you broke something.  Without tests, you find out by accident — usually at
the worst time.

Running these tests:
    pytest tests/test_chunking.py -v

Expected: all tests pass (green dots).

# pip install pytest  (already in requirements)
"""

import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the code under test
# ---------------------------------------------------------------------------
from src.ingestion.loaders import load_docx_1b
from src.chunking.chunkers import chunk_recursive
from src.chunking.stats import load_chunks_jsonl


# ---------------------------------------------------------------------------
# Fixture: load the sample .docx once and share it across tests
# ---------------------------------------------------------------------------
# A pytest *fixture* is a helper that creates reusable setup.
# @pytest.fixture tells pytest: "run this function and pass its return value
# to any test that names it as a parameter."

FIXTURE_PATH = Path("tests/fixtures/sample.docx")


@pytest.fixture
def sample_doc():
    """Load sample.docx once; share the result across all tests in this file."""
    # Arrange: load the document using our production loader
    # load_docx_1b returns a single ParsedDocument, not a list
    return load_docx_1b(FIXTURE_PATH)


# ---------------------------------------------------------------------------
# Test 1 — loader returns at least one document
# ---------------------------------------------------------------------------

def test_docx_loader_returns_documents(sample_doc):
    """load_docx_1b should return a non-empty list."""
    # Arrange: sample_doc is already loaded by the fixture above

    # Act: nothing to do — the loader already ran

    # Assert: check the document has at least one section
    # Background: if sections is empty the loader silently swallowed all content
    assert len(sample_doc.sections) > 0


# ---------------------------------------------------------------------------
# Test 2 — each document has the required metadata fields
# ---------------------------------------------------------------------------

def test_docx_loader_metadata_fields(sample_doc):
    """Every loaded document must carry 'source' and 'type' metadata."""
    # Act: nothing to do — check structure of what was returned

    # Assert 1: check 'source' attribute is set (it's a top-level field, not in .meta)
    assert sample_doc.source != ""      # source is a str field on ParsedDocument

    # Assert 2: check 'doc_type' attribute is set
    assert sample_doc.doc_type != ""   # doc_type is the type field on ParsedDocument


# ---------------------------------------------------------------------------
# Test 3 — chunk_recursive produces the right output type
# ---------------------------------------------------------------------------

def test_chunk_recursive_returns_chunks(sample_doc):
    """chunk_recursive should return a non-empty list of Chunk objects."""
    # Act: chunk the document with default parameters
    chunks = chunk_recursive(sample_doc)

    # Assert: result is non-empty
    assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Test 4 — every chunk carries text content
# ---------------------------------------------------------------------------

def test_chunks_have_text(sample_doc):
    """No chunk should have an empty text field."""
    # Arrange + Act
    chunks = chunk_recursive(sample_doc)

    # Assert: iterate and check each chunk's .text field is non-empty
    # Hint: use a for loop with assert inside, or use all() with a generator
    # Background: an empty chunk wastes an embedding slot and confuses the retriever
    # Answer: for chunk in chunks: assert len(chunk.text) > 0
    for chunk in chunks:
        assert len(chunk.text)  > 0   # (think) — which attribute holds the chunk text?


# ---------------------------------------------------------------------------
# Test 5 — chunk size is within the configured bound
# ---------------------------------------------------------------------------

def test_chunk_size_within_limit(sample_doc):
    """No chunk text should exceed chunk_size + overlap (a small buffer is fine)."""
    CHUNK_SIZE    = 1000
    CHUNK_OVERLAP = 150
    BUFFER        = 200   # splitter may slightly overshoot on word boundaries

    chunks = chunk_recursive(sample_doc, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for chunk in chunks:
        # Assert: text length is at most CHUNK_SIZE + BUFFER
        # Hint: len(string) gives you character count
        # Background: chunks that are far too large inflate embedding cost and confuse reranking
        # Answer: assert len(chunk.text) <= CHUNK_SIZE + BUFFER
        assert len(chunk.text) <= CHUNK_SIZE + BUFFER   # (stretch) — write the full boolean expression


# ---------------------------------------------------------------------------
# Test 6 — load_chunks_jsonl round-trips through save + load
# ---------------------------------------------------------------------------

def test_load_chunks_jsonl_round_trip(sample_doc, tmp_path):
    """Chunks saved to a .jsonl file should load back identically."""
    # tmp_path is a pytest built-in fixture that gives you a fresh temp directory
    import json

    chunks = chunk_recursive(sample_doc)

    # Arrange: save chunks to a temp file manually
    out_path = tmp_path / "chunks.jsonl"
    with open(out_path, "w") as f:
        for chunk in chunks:
            # Each chunk is one JSON line — use chunk.__dict__ to serialise
            f.write(json.dumps(chunk.__dict__) + "\n")

    # Act: load them back
    # Hint: load_chunks_jsonl takes a Path argument
    # Background: the round-trip test catches serialisation bugs — missing fields, wrong types
    # Answer: loaded = load_chunks_jsonl(out_path)
    loaded = load_chunks_jsonl(out_path)   # (think)

    # Assert: same number of chunks came back
    assert len(loaded) == len(chunks)


# ---- HINTS (uncover only if stuck > 5 min) ----
# test_docx_loader_returns_documents:
#   assert len(sample_doc) > 0
#
# test_docx_loader_metadata_fields:
#   assert "source" in first_doc.meta
#
# test_chunk_recursive_returns_chunks:
#   chunks = chunk_recursive(doc)
#
# test_chunks_have_text:
#   assert len(chunk.text) > 0
#
# test_chunk_size_within_limit:
#   assert len(chunk.text) <= CHUNK_SIZE + BUFFER
#
# test_load_chunks_jsonl_round_trip:
#   loaded = load_chunks_jsonl(out_path)


# ---- Reflection question ----
# Q: test_chunk_size_within_limit uses BUFFER=200 as a tolerance.
#    Why is a hard assert len(chunk.text) <= 1000 too strict?
#    What property of RecursiveCharacterTextSplitter causes occasional overshoot?
