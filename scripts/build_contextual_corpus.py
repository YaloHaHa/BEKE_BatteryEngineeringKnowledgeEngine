"""Driver: contextualise chunks with 3E → save contextual_chunks.jsonl → build Chroma index.

Run from the repo root:
    python3 build_contextual_corpus.py

Prerequisites (must exist before running):
    parsed_corpus.jsonl      — from build_corpus.py
    chunks_recursive.jsonl   — from build_corpus.py

Outputs:
    contextual_chunks.jsonl        — chunks with LLM context prepended
    caches/contextual_cache.jsonl  — per-chunk context cache (re-runs are free)
    indices/chroma_contextual/     — Chroma vector index of contextualised chunks

Cost estimate (gpt-4o-mini, ~200 tokens/call):
    500 chunks × 200 tokens = 100K tokens ≈ $0.015
    Re-runs cost $0 (served from cache).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.contextual import add_context_to_chunks
from src.chunking.stats import load_chunks_jsonl, write_chunks_jsonl
from src.retrieval.index import build_chroma_index

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARSED_PATH          = Path("parsed_corpus.jsonl")
CHUNKS_PATH          = Path("chunks_recursive.jsonl")
CONTEXTUAL_OUT       = Path("contextual_chunks.jsonl")
CACHE_PATH           = Path("caches/contextual_cache.jsonl")
CHROMA_OUT           = Path("indices/chroma_contextual")

LLM_MODEL            = "gpt-4o-mini"
EMBED_MODEL          = "BAAI/bge-base-en-v1.5"

# Set to True to run on the first 5 chunks only — verify quality before full run
SMOKE_TEST_ONLY      = False
SMOKE_TEST_N         = 5

# ---------------------------------------------------------------------------
# Step 1: load parsed corpus → build doc_texts map
# ---------------------------------------------------------------------------

if not PARSED_PATH.exists():
    raise SystemExit(f"ERROR: {PARSED_PATH} not found — run build_corpus.py first")
if not CHUNKS_PATH.exists():
    raise SystemExit(f"ERROR: {CHUNKS_PATH} not found — run build_corpus.py first")

print(f"Loading parsed corpus from {PARSED_PATH} ...")
doc_texts = {}
with PARSED_PATH.open(encoding="utf-8") as fh:
    for line in fh:
        doc = json.loads(line.strip())
        full_text = "\n\n".join(s["text"] for s in doc["sections"] if s["text"].strip())
        doc_texts[doc["source"]] = full_text
print(f"  {len(doc_texts)} documents loaded")

# ---------------------------------------------------------------------------
# Step 2: load chunks
# ---------------------------------------------------------------------------

print(f"\nLoading chunks from {CHUNKS_PATH} ...")
chunks = load_chunks_jsonl(CHUNKS_PATH)
print(f"  {len(chunks)} chunks loaded")

if SMOKE_TEST_ONLY:
    chunks = chunks[:SMOKE_TEST_N]
    print(f"  [SMOKE TEST] running on first {SMOKE_TEST_N} chunks only")

# ---------------------------------------------------------------------------
# Step 3: contextualise — LLM generates 1-2 situating sentences per chunk
# ---------------------------------------------------------------------------

print(f"\nContextualising {len(chunks)} chunks with {LLM_MODEL} ...")
print(f"  cache: {CACHE_PATH}  (existing entries reused for free)")

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
contextual_chunks = add_context_to_chunks(chunks, doc_texts, llm, cache_path=CACHE_PATH)

# ---------------------------------------------------------------------------
# Step 4: save contextual_chunks.jsonl
# ---------------------------------------------------------------------------

if not SMOKE_TEST_ONLY:
    write_chunks_jsonl(contextual_chunks, CONTEXTUAL_OUT)
    print(f"\nwrote {len(contextual_chunks)} contextual chunks → {CONTEXTUAL_OUT}")
else:
    print("\n[SMOKE TEST] sample output (not saved to disk):")
    for c in contextual_chunks:
        print(f"\n--- chunk {c.chunk_idx} ({Path(c.source).name}) ---")
        print(c.text[:400])
    print("\nSet SMOKE_TEST_ONLY = False and re-run to process the full corpus.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Step 5: build Chroma index from contextualised chunks
# ---------------------------------------------------------------------------

print(f"\nBuilding Chroma index → {CHROMA_OUT} ...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
build_chroma_index(contextual_chunks, embeddings, persist_dir=CHROMA_OUT)
print(f"  index saved to {CHROMA_OUT}")

print("\nDone. Run python3 -m src.eval.metrics to compare 3E against 3A–3D.")
