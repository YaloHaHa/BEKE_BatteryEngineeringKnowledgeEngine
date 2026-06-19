"""Minimal pgvector debug script — run from repo root with tunnel open."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
conn = os.getenv("AURORA_DB_URL")
print(f"Connection string: {conn[:40]}...") # masked

from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --- Test 1: simple docs, no explicit ids ---
print("\n--- Test 1: from_documents WITHOUT explicit ids ---")
docs = [Document(page_content=f"aluminium air battery test {i}", metadata={"source": f"test_{i}"}) for i in range(5)]
vs1 = PGVector.from_documents(docs, emb, connection=conn,
    collection_name="beke_debug_noids", pre_delete_collection=True, use_jsonb=True)
r1 = vs1.similarity_search("aluminium", k=3)
print(f"Results: {len(r1)}  (expected 3)")

# --- Test 2: same docs, WITH explicit string ids (same format as chunks) ---
print("\n--- Test 2: from_documents WITH explicit string ids ---")
ids = [f"/some/path/file.docx::{i}" for i in range(5)]
vs2 = PGVector.from_documents(docs, emb, connection=conn,
    collection_name="beke_debug_withids", pre_delete_collection=True, use_jsonb=True, ids=ids)
r2 = vs2.similarity_search("aluminium", k=3)
print(f"Results: {len(r2)}  (expected 3)")

# --- Test 3: load fresh instance after upsert ---
print("\n--- Test 3: reload after upsert ---")
PGVector.from_documents(docs, emb, connection=conn,
    collection_name="beke_debug_reload", pre_delete_collection=True, use_jsonb=True, ids=ids)
vs3 = PGVector(embeddings=emb, connection=conn,
    collection_name="beke_debug_reload", use_jsonb=True)
r3 = vs3.similarity_search("aluminium", k=3)
print(f"Results: {len(r3)}  (expected 3)")

print("\nDone.")
