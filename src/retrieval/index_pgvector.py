"""Phase 2 — pgvector index: drop-in replacement for index.py (Chroma).
Input:  list[Chunk] + LangChain Embeddings object + Aurora connection string
Output: PGVector vectorstore (upsert) or BaseRetriever (query time)

Connection resolution order
---------------------------
1. Explicit `connection_string` argument
2. Environment variable AURORA_DB_URL  (set in .env for local tunnel dev)
3. AWS Secrets Manager secret (production)

Local dev setup: see docs/aws_setup.md for SSH tunnel and connection details.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv                        # pip install python-dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_postgres.vectorstores import PGVector  # pip install langchain-postgres "psycopg[binary]"


class PGVectorNoExtension(PGVector):
    """PGVector that skips CREATE EXTENSION on reconnect.

    PGVector.__post_init__ runs `CREATE EXTENSION IF NOT EXISTS vector` every
    time a new instance is created — requiring superuser privileges and adding
    unnecessary overhead at query time. The extension is already installed (Day 2
    setup). This subclass overrides create_vector_extension() to a no-op so
    load_pgvector_index() can reconnect safely without superuser rights.
    """

    def create_vector_extension(self) -> None:
        pass  # extension already installed during Day 2 provisioning

from ..chunking.schema import Chunk

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "beke_contextual"
SECRET_NAME     = "beke-aurora-credentials"
AWS_REGION      = "us-west-1"
AWS_PROFILE     = "beke"


# ---------------------------------------------------------------------------
# Credential resolution — GIVEN
# ---------------------------------------------------------------------------

def _get_connection_string(connection_string: Optional[str] = None) -> str:
    """Resolve a psycopg3 connection string: explicit arg → env var → Secrets Manager."""
    if connection_string:
        return connection_string
    env_url = os.getenv("AURORA_DB_URL")
    if env_url:
        logger.debug("Using AURORA_DB_URL from environment")
        return env_url
    return _get_connection_string_from_secrets_manager()


# ---------------------------------------------------------------------------
# Scaffold 1 of 4 — _get_connection_string_from_secrets_manager
# _get_connection_string_from_secrets_manager: fetch Aurora credentials
#   from AWS Secrets Manager and build a psycopg3 connection string.
# Input:  none — uses module constants SECRET_NAME, AWS_PROFILE, AWS_REGION
# Output: str — "postgresql+psycopg://user:pw@host:port/db"
# ---------------------------------------------------------------------------

def _get_connection_string_from_secrets_manager() -> str:
    import boto3  # pip install boto3

    # Step 1: open a Secrets Manager client scoped to the beke profile and region
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    # Hint: boto3 method that opens a low-level service client
    # Background: .client() gives direct API access; .resource() gives an ORM-like
    #             wrapper — Secrets Manager only has a .client() interface, not .resource()
    # Answer: session.client("secretsmanager")
    client = session.client("secretsmanager")                   # (easy)

    # Step 2: fetch the secret and deserialise its JSON string payload
    response = client.get_secret_value(SecretId=SECRET_NAME)
    # Hint: the response dict key that holds the raw secret string
    # Background: AWS wraps JSON secrets in "SecretString"; binary blobs go in
    #             "SecretBinary" — you will never see both populated at once
    # Answer: "SecretString"
    creds = json.loads(response["SecretString"])                        # (think)

    # Step 3: assemble and return the psycopg3 connection string
    # .get() with a default for port — why not creds["port"] directly?   (think)
    return (
        f"postgresql+psycopg://{creds['username']}:{creds['password']}"
        f"@{creds['host']}:{creds.get('port', 5432)}/{creds['dbname']}"
    )


# ---------------------------------------------------------------------------
# Document conversion — GIVEN (identical to index.py; same pattern you filled
# in for Chroma — pgvector has the same flat-metadata constraint)
# ---------------------------------------------------------------------------

def _chunk_to_document(chunk: Chunk) -> Document:
    """Convert a Chunk to a LangChain Document. Metadata must be flat scalars."""
    return Document(
        page_content=chunk.text,
        metadata={
            "source":       chunk.source,
            "chunk_idx":    chunk.chunk_idx,
            "chunker":      chunk.chunker,
            "page_idx":     chunk.page_idx,
            "slide_idx":    chunk.slide_idx,
            "chunk_id":     chunk.meta.get("chunk_id", f"{chunk.source}::{chunk.chunk_idx}"),
            "heading_path": json.dumps(chunk.heading_path),
        },
    )


# ---------------------------------------------------------------------------
# Scaffold 2 of 4 — build_pgvector_index  (coming next turn)
# ---------------------------------------------------------------------------

def build_pgvector_index(
    chunks: list[Chunk],
    embeddings,
    connection_string: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 100,
    pre_delete_collection: bool = False,
) -> PGVector:
    # Step 1: resolve the connection string (explicit arg → .env → Secrets Manager)
    # Hint: there is a private function in this file whose entire job is this resolution chain
    # Background: centralising resolution means build_pgvector_index() and
    #             load_pgvector_index() both behave consistently without duplicating logic
    conn_str = _get_connection_string(connection_string)                         # (easy)
    # Answer: _get_connection_string(connection_string)

    # Step 2: convert all Chunks to LangChain Documents + extract their IDs
    # Hint: there is a private helper in this file that converts one Chunk → Document
    # Background: PGVector.from_documents() expects LangChain Document objects,
    #             not raw Chunk dataclasses — the helper handles metadata flattening
    docs = [_chunk_to_document(c) for c in chunks]                          # (easy)
    # Answer: _chunk_to_document(c)

    # Step 3: upsert into pgvector — creates the table if it doesn't exist.
    # Note: we do NOT pass explicit ids — langchain_pg_embedding.id is a global
    # primary key (not scoped per collection). Passing chunk_ids as ids would
    # cause silent PK violations if the same chunks exist in another collection.
    # chunk_id is already stored in metadata for filtering/lookup.
    # Hint: the function parameter that holds the embeddings model object
    # Background: PGVector.from_documents() needs the embeddings to convert
    #             page_content strings into vectors before storing them
    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,                                        # (easy)
        # Answer: embeddings
        connection=conn_str,
        collection_name=collection_name,
        pre_delete_collection=pre_delete_collection,
        use_jsonb=True,
    )

    print(f"[pgvector] upserted {len(docs):,} chunks → '{collection_name}'")
    return vectorstore


# ---------------------------------------------------------------------------
# Scaffold 3 of 4 — load_pgvector_index + get_pgvector_retriever: Both connect to an already-populated pgvector table 
# ---------------------------------------------------------------------------

def load_pgvector_index(
    embeddings,
    connection_string: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
) -> PGVector:
    # Step 1: resolve connection string
    conn_str = _get_connection_string(connection_string)

    # Step 2: return a PGVector instance pointing at the existing collection
    # Hint: PGVector constructor — same class as from_documents() but no ingestion
    # Background: at query time you never re-embed; you just reconnect to the
    #             existing table and let similarity_search() do the work.
    #             create_extension_if_not_exists=False avoids a permission check
    #             on reconnect; the extension was already created at index time.
    return PGVectorNoExtension(                                    # (easy)
        embeddings=embeddings,
        connection=conn_str,
        collection_name=collection_name,
        use_jsonb=True,
    )
    # Answer: PGVectorNoExtension(...)


def get_pgvector_retriever(
    embeddings,
    connection_string: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    k: int = 10,
) -> BaseRetriever:
    # Step 1: load the existing index
    vectorstore = load_pgvector_index(embeddings, connection_string, collection_name)

    # Step 2: wrap as a BaseRetriever — mirrors make_dense_retriever() in retrievers.py
    # Hint: the PGVector method that returns a LangChain BaseRetriever
    # Background: as_retriever() is the standard LangChain interface — same call
    #             you used on the Chroma vectorstore in retrievers.py
    return vectorstore.as_retriever(search_kwargs={"k": k})            # (think)
    # Answer: as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# Scaffold 4 of 4 — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface

    from ..chunking.stats import load_chunks_jsonl

    chunk_path = Path("contextual_chunks.jsonl")
    if not chunk_path.exists():
        print(f"ERROR: {chunk_path} not found")
        sys.exit(1)

    all_chunks = load_chunks_jsonl(chunk_path)
    # Hint: Python slice syntax to take the first N items from a list
    # Background: we use 100 chunks for a fast smoke test rather than all 265 —
    #             enough to verify the round-trip without waiting for full embedding
    sample = all_chunks[:100]                                  # (easy)
    # Answer: all_chunks[:100]

    # Step 1: build embeddings — same model as Phase 1
    # Hint: HuggingFaceEmbeddings constructor — model name, device, normalize flag
    # Background: normalize_embeddings=True is required for cosine similarity;
    #             without it dot-product scores are not bounded between -1 and 1
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",                                       # (easy)
        # Answer: "BAAI/bge-base-en-v1.5"
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Step 2: upsert 100 chunks into a test collection
    vs = build_pgvector_index(
        sample,
        embeddings,
        collection_name="beke_smoke_test",
        pre_delete_collection=True,                            # (think)
        # Answer: True
    )

    # Step 3: probe — similarity search on a known battery question
    probe   = "What is the discharge mechanism of Al-Air batteries?"
    results = vs.similarity_search(probe, k=3)                              # (easy)
    # Answer: similarity_search(probe, k=3)

    print(f"\npgvector top-3 for: {probe!r}")
    for i, doc in enumerate(results):
        src = doc.metadata.get("source", "?").split("/")[-1]
        print(f"  [{i+1}] chunk {doc.metadata.get('chunk_idx')} | {src}")
        print(f"       {doc.page_content[:100]!r}")

    print("\nSmoke test passed ✓")
    # expected shape:
    # [pgvector] upserted 100 chunks → 'beke_smoke_test'
    # pgvector top-3 for: 'What is the discharge mechanism of Al-Air batteries?'
    #   [1] chunk N | AAB_Manuscript_10.09.2024.docx
    #       'Context: ...'


# ---- HINTS (uncover only if stuck > 5 min) ----
# All scaffolds complete — no open blanks remaining.


# ---- Reflection question ----
# Q: _get_connection_string() tries AURORA_DB_URL before falling back to
#    Secrets Manager. In production on App Runner you could set AURORA_DB_URL
#    as an environment variable instead of calling Secrets Manager at all.
#    What is one concrete advantage of the Secrets Manager path over an env var,
#    and one concrete advantage of the env var path?
#    (Hint: think about secret rotation and cold-start latency.)
