"""Check corpus stats from Aurora pgvector.

Run with SSH tunnel open (see docs/aws_setup.md), then:
    python check_corpus_stats.py
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

url = os.getenv("AURORA_DB_URL")
if not url:
    print("ERROR: AURORA_DB_URL not set in .env")
    raise SystemExit(1)

engine = create_engine(url)

with engine.connect() as conn:
    # Total chunks in the beke_contextual collection
    row = conn.execute(text("""
        SELECT COUNT(*) AS chunk_count
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = 'beke_contextual'
    """)).fetchone()
    chunk_count = row[0]

    # Distinct source documents
    row2 = conn.execute(text("""
        SELECT COUNT(DISTINCT e.cmetadata->>'source') AS doc_count
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = 'beke_contextual'
    """)).fetchone()
    doc_count = row2[0]

print(f"\n=== BEKE Corpus Stats (beke_contextual) ===")
print(f"  Documents : {doc_count}")
print(f"  Chunks    : {chunk_count:,}")
print(f"============================================")
print(f"\nPaste these numbers back to me!")
