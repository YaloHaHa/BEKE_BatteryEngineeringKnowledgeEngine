"""Phase 2 Day 3 — Full corpus ingestion pipeline.
Input:  S3 bucket beke-corpus-prod (1,102 files, ~50 GB)
Output: Aurora pgvector collection beke_contextual

Parsing approach per file type:
  .docx / .pdf → 1b (native, fast)
  .pptx        → 1c (vision-assisted, gpt-4o-mini describes images/charts)

Run order:
  1. Open SSH tunnel (port 5433)
  2. python ingest.py --limit 10   # smoke test on 10 files (~5 min)
  3. python ingest.py              # full overnight run (~8–10h)

Checkpoint file (ingest_checkpoint.jsonl) records every successfully
processed S3 key. Re-runs skip already-processed files automatically.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator

import boto3                                              # pip install boto3
from dotenv import load_dotenv                            # pip install python-dotenv
from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface
from langchain_openai import ChatOpenAI                   # pip install langchain-openai

from src.ingestion.loaders import load_one
from src.ingestion.contextual import add_context_to_chunks
from src.chunking.chunkers import chunk_recursive
from src.retrieval.index_pgvector import build_pgvector_index

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET          = "beke-corpus-prod"
AWS_PROFILE     = "beke"
AWS_REGION      = "us-west-1"
COLLECTION      = "beke_contextual"
CHECKPOINT_PATH = Path("ingest_checkpoint.jsonl")
CACHE_PATH      = Path("caches/contextual_cache.jsonl")
SUPPORTED_EXTS  = {".docx", ".pptx", ".pdf"}
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 150
EMBED_BATCH     = 100

# Parsing approach per file type (see configs/contextual_rerank.yaml)
# .pptx uses vision-assisted (1c): gpt-4o-mini describes embedded images/charts
# .docx and .pdf use native (1b): fast text extraction, no vision API calls
PARSE_APPROACH  = {".docx": "1b", ".pdf": "1b", ".pptx": "1c"}


# ---------------------------------------------------------------------------
# Scaffold 1 of 4 — list_s3_keys
# list_s3_keys: discover all ingestible files in the S3 bucket.
# Input:  bucket — S3 bucket name; profile — AWS profile name
# Output: Iterator[str] of S3 keys ending in .docx / .pptx / .pdf
# ---------------------------------------------------------------------------

def list_s3_keys(bucket: str, profile: str = AWS_PROFILE) -> Iterator[str]:
    # Step 1: create a boto3 S3 client scoped to the beke profile
    session = boto3.Session(profile_name=profile, region_name=AWS_REGION)
    # Hint: boto3 service name for S3
    # Background: same .client() pattern as Secrets Manager — just a different service string
    client  = session.client("s3")                           # (easy)
    # Answer: client("s3")

    # Step 2: page through all objects in the bucket
    # Hint: boto3 paginator name for listing S3 objects
    # Background: list_objects_v2 returns max 1,000 keys per call; a paginator
    #             automatically handles the continuation token so you see all keys
    paginator = client.get_paginator("list_objects_v2")                     # (think)
    # Answer: "list_objects_v2"

    # Step 3: yield only keys whose suffix is in SUPPORTED_EXTS
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Hint: pathlib attribute that returns the file extension including the dot
            # Background: Path("report.pdf").suffix == ".pdf" — works on S3 keys
            #             because they follow Unix path conventions
            if Path(key).suffix.lower() in SUPPORTED_EXTS:              # (think)
                yield key
            # Answer: .suffix


# ---------------------------------------------------------------------------
# Scaffold 2 of 4 — load_checkpoint + save_checkpoint
# load_checkpoint: read already-processed S3 keys from the checkpoint file.
# Input:  path — Path to the checkpoint JSONL file
# Output: set[str] of S3 keys that have already been successfully ingested

# save_checkpoint: append one successfully processed S3 key to the checkpoint.
# Input:  key — the S3 key just processed; path — checkpoint file path
# Output: None (writes one JSON line to disk)
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path = CHECKPOINT_PATH) -> set[str]:
    # Step 1: return an empty set if no checkpoint file exists yet
    # Hint: pathlib method that returns True if the path points to an existing file
    # Background: on the very first run there is no checkpoint file — returning
    #             an empty set means every file will be processed
    if not path.exists():                                        # (easy)
        return set()
    # Answer: exists()

    # Step 2: read each line, deserialise, and collect the "key" field
    # Hint: use a set comprehension over the file lines; skip blank lines with .strip()
    # Background: each line is {"key": "some/s3/path.pdf"} — json.loads() gives
    #             a dict, then ["key"] extracts the string
    with path.open() as f:
        return {json.loads(line)["key"] for line in f if line.strip()}  # (think)
    # Answer: json.loads(line)["key"] ... if line.strip()


def save_checkpoint(key: str, path: Path = CHECKPOINT_PATH) -> None:
    # Step 1: open in append mode — never overwrite existing checkpoint entries
    # Hint: Python file open mode that adds to the end of the file
    # Background: "w" would erase the entire checkpoint on every save; "a" adds
    #             one line at a time, making each write atomic and crash-safe
    with path.open("a") as f:                                 # (think)
        f.write(json.dumps({"key": key}) + "\n")
    # Answer: "a"


# ---------------------------------------------------------------------------
# Scaffold 3 of 4 — process_file
# process_file: run the full ingestion pipeline for one S3 file.
# Input:  key — S3 object key; embeddings — HuggingFace model;
#         llm — ChatOpenAI for contextualisation; tmp_dir — temp folder
# Output: int — number of chunks upserted (0 if file produced no chunks)
# ---------------------------------------------------------------------------

def process_file(
    key: str,
    embeddings,
    llm,
    tmp_dir: Path,
    s3_client=None,                     # pass in from main to avoid per-file client init
) -> int:
    suffix   = Path(key).suffix.lower()
    tmp_path = tmp_dir / Path(key).name

    # Step 1: download the file from S3 to tmp_path
    s3 = s3_client or boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION).client("s3")
    s3.download_file(BUCKET, key, str(tmp_path))

    # Step 2: parse using the correct approach for this file type
    # Hint: look up the approach in PARSE_APPROACH; fall back to "1b" if unknown
    # Background: PARSE_APPROACH maps suffix → approach string; .get(key, default)
    #             is safer than [] because it won't raise KeyError on new file types
    approach = PARSE_APPROACH.get(suffix, "1b")               # (think)
    doc      = load_one(tmp_path, approach=approach)
    # Answer: .get(suffix, "1b")

    # Step 3: chunk the parsed document
    chunks = chunk_recursive(doc, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        tmp_path.unlink(missing_ok=True)
        return 0

    # Step 4: contextualise — prepend LLM-generated situating sentence to each chunk
    # Hint: the contextualiser needs the full document text keyed by doc.source
    # Background: add_context_to_chunks expects doc_texts: dict[source → full_text];
    #             ParsedDocument.full_text joins all section texts automatically
    doc_texts = {doc.source: doc.full_text}                         # (think)
    chunks    = add_context_to_chunks(chunks, doc_texts, llm, cache_path=CACHE_PATH)
    # Answer: doc.full_text

    # Step 5: embed + upsert to pgvector
    # Hint: pre_delete_collection controls whether to wipe the collection before upserting
    # Background: False here because many files share one collection — wiping it
    #             on each file would erase all previously ingested chunks
    build_pgvector_index(
        chunks, embeddings,
        collection_name=COLLECTION,
        pre_delete_collection=False,                            # (think)
    )
    # Answer: False

    # Step 6: delete the local temp file to keep disk usage flat
    tmp_path.unlink(missing_ok=True)                             # (easy)
    # Answer: unlink(missing_ok=True)

    return len(chunks)


# ---------------------------------------------------------------------------
# Scaffold 4 of 4 — main
# main: orchestrate the full ingestion run with checkpointing and error recovery.
# Input:  limit — optional int to cap the number of files (None = full corpus)
# Output: None (logs progress; updates checkpoint file; upserts to Aurora)
# ---------------------------------------------------------------------------

def main(limit: int | None = None) -> None:
    logger.info("Starting ingestion — bucket=%s  collection=%s", BUCKET, COLLECTION)

    # Step 1: initialise embeddings + LLM
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create S3 client once — reused across all files
    s3_client = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION).client("s3")

    # Step 2: load checkpoint and build the todo list
    done = load_checkpoint()
    keys = list(list_s3_keys(BUCKET))
    # Hint: filter keys to only those NOT already in the done set
    # Background: `not in` on a set is O(1) — fast even for 1,102 keys
    todo = [k for k in keys if k not in done]                   # (think)
    # Answer: not in done

    # Step 3: apply the file limit for smoke tests
    # Hint: Python slice that takes the first `limit` items if limit is set
    # Background: todo[:None] returns the full list unchanged — so this one-liner
    #             works whether limit is an int or None
    todo = todo[:limit]                                          # (easy)
    # Answer: :limit

    logger.info(
        "Files: total=%d  already done=%d  to process=%d",
        len(keys), len(done), len(todo),
    )

    # Step 4: process each file; checkpoint on success; log errors without crashing
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, key in enumerate(todo, 1):
            t0 = time.time()
            try:
                n = process_file(key, embeddings, llm, tmp_dir, s3_client=s3_client)
                # Hint: save the checkpoint AFTER a successful process_file, not before
                # Background: saving before means a crash mid-process marks the file
                #             as done even though it wasn't — causing silent data loss
                save_checkpoint(key)                                      # (easy)
                # Answer: save_checkpoint(key)
                logger.info(
                    "[%d/%d] %-60s → %3d chunks  (%.1fs)",
                    i, len(todo), key, n, time.time() - t0,
                )
            except Exception as exc:
                logger.error("FAILED [%d/%d] %s — %s", i, len(todo), key, exc)

    logger.info("Ingestion complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BEKE corpus ingestion pipeline")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N files (omit for full run)",
    )
    args = parser.parse_args()
    main(limit=args.limit)


# ---- HINTS (uncover only if stuck > 5 min) ----
# All hints are placed inline directly above each blank.


# ---- Reflection question ----
# Q: save_checkpoint() writes one line per file AFTER process_file() succeeds.
#    If the machine crashes mid-file (e.g. during embedding), that file is NOT
#    in the checkpoint. On the next run, process_file() will re-run it from scratch.
#    Is this the correct behaviour? What would go wrong if you saved the checkpoint
#    BEFORE calling process_file()?
#    (Hint: think about silent data loss vs. duplicate work — which is worse?)
