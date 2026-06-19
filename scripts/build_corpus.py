"""One-off driver: ingest all corpus files → parsed_corpus.jsonl → chunks_*.jsonl

Run from the repo root:
    python3 build_corpus.py

Adjust CORPUS_DIR below if your files live elsewhere.
Loaders used: 1B (native python-docx / pypdf) — rich metadata, no vision API cost.
Switch to load_*_1c if you want vision captioning later.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import json
from pathlib import Path

from src.ingestion.loaders import load_docx_1b, load_pdf_1b
from src.chunking.chunkers import chunk_recursive
from src.chunking.stats import write_chunks_jsonl

# ---------------------------------------------------------------------------
# Config — edit these paths if needed
# ---------------------------------------------------------------------------

CORPUS_DIR   = Path("eval_set_files")
PARSED_OUT   = Path("parsed_corpus.jsonl")
CHUNKS_OUT   = Path("chunks_recursive.jsonl")

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Step 1: ingest all files → parsed_corpus.jsonl
# ---------------------------------------------------------------------------

LOADER_MAP = {
    ".docx": load_docx_1b,
    ".pdf":  load_pdf_1b,
    # ".pptx": load_pptx_1b,   # uncomment if you add pptx files later
}

files = sorted(
    p for p in CORPUS_DIR.iterdir()
    if p.suffix.lower() in LOADER_MAP and not p.name.startswith(".")
)

if not files:
    raise SystemExit(f"No supported files found in {CORPUS_DIR}")

print(f"Found {len(files)} files in {CORPUS_DIR}:")
for f in files:
    print(f"  {f.name}")

parsed_docs = []
failed = []

print("\n--- Ingestion ---")
for path in files:
    loader = LOADER_MAP[path.suffix.lower()]
    try:
        doc = loader(path)
        parsed_docs.append(doc)
        print(f"  ✓  {path.name}  ({doc.section_count} sections)")
    except Exception as e:
        print(f"  ✗  {path.name}  ERROR: {e}")
        failed.append(path.name)

print(f"\ningested {len(parsed_docs)} docs, {len(failed)} failed")
if failed:
    print(f"  failed: {failed}")

# Write parsed_corpus.jsonl
PARSED_OUT.parent.mkdir(parents=True, exist_ok=True)
with PARSED_OUT.open("w", encoding="utf-8") as fh:
    for doc in parsed_docs:
        fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
print(f"wrote {len(parsed_docs)} documents → {PARSED_OUT}")

# ---------------------------------------------------------------------------
# Step 2: chunk all parsed docs → chunks_recursive.jsonl
# ---------------------------------------------------------------------------

print("\n--- Chunking (2A recursive) ---")
all_chunks = []
for doc in parsed_docs:
    chunks = chunk_recursive(doc, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {Path(doc.source).name:<55}  {len(chunks):>4} chunks")
    all_chunks.extend(chunks)

write_chunks_jsonl(all_chunks, CHUNKS_OUT)

# Quick length sanity check
lengths = [len(c.text) for c in all_chunks]
print(f"\ncorpus summary:")
print(f"  total chunks : {len(all_chunks)}")
print(f"  mean length  : {sum(lengths)/len(lengths):.0f} chars")
print(f"  min / max    : {min(lengths)} / {max(lengths)} chars")
print(f"\nNext: rebuild the Chroma index")
print(f"  python3 -m src.retrieval.index")
