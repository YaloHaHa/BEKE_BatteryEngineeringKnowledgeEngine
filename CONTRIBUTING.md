# Build Your Own RAG System

This guide walks you through forking BEKE and adapting it to your own document corpus and AWS infrastructure. By the end you'll have a deployed Streamlit app that answers questions over your own files.

---

## Prerequisites

- Python 3.11+
- An AWS account (S3, RDS Aurora PostgreSQL, EC2)
- An OpenAI API key (for contextual enrichment + answer generation)
- Your document corpus (.docx, .pptx, .pdf)

---

## 1. Clone and install

```bash
git clone https://github.com/yanghanghuang/Al_Air_Battery_Librarian.git
cd Al_Air_Battery_Librarian
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Preview the UI (no AWS needed)

```bash
MOCK_MODE=1 streamlit run app.py
```

This runs the full UI with canned answers. Edit `_build_mock_pipeline()` in `app.py` to customize the mock data for your domain.

---

## 3. Upload your corpus to S3

Create an S3 bucket and update `scripts/upload_corpus_to_s3.py`:

```python
BUCKET = "your-bucket-name"
REGION = "your-region"
PROFILE = "your-aws-profile"
CORPUS_FOLDERS = [
    "/path/to/your/documents/folder1/",
    "/path/to/your/documents/folder2/",
]
```

Then run:

```bash
python scripts/upload_corpus_to_s3.py
```

The script mirrors your local folder structure as S3 prefixes, skips already-uploaded files, and is resumable.

---

## 4. Set up Aurora PostgreSQL + pgvector

Provision an Aurora PostgreSQL Serverless v2 cluster and enable the pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Set up an SSH bastion host or VPC peering so your local machine and EC2 instance can reach Aurora. Store your credentials in a `.env` file:

```bash
# .env (never commit)
AURORA_DB_URL=postgresql+psycopg://<user>:<password>@<host>:<port>/<dbname>
OPENAI_API_KEY=sk-...
AWS_PROFILE=your-profile
AWS_REGION=your-region
```

---

## 5. Ingest your corpus

The ingestion pipeline downloads each file from S3, parses it, chunks it, enriches chunks with LLM-generated context, embeds them, and upserts into pgvector.

```bash
python scripts/ingest.py
```

Key parameters to customize in `scripts/ingest.py`:

- `BUCKET` — your S3 bucket name
- `COLLECTION` — pgvector collection name
- `EMBED_MODEL` — embedding model (default: `BAAI/bge-base-en-v1.5`)

The pipeline checkpoints progress to `ingest_checkpoint.jsonl` — if interrupted, re-run to resume where it left off.

**Expected runtime:** ~8–10 hours for ~1,000 documents (embedding + LLM contextualisation). The LLM context cache (`caches/contextual_cache.jsonl`) means re-runs are near-instant for already-processed chunks.

---

## 6. Configure the app

Update constants in `app.py` to match your setup:

```python
COLLECTION   = "your_collection_name"
CORPUS_DOCS  = "500"          # your doc count
CORPUS_CHUNKS = "40K"         # your chunk count
CORPUS_DATE  = "Jun 2026"     # when you indexed
```

Update `src/download.py` if your S3 bucket name differs:

```python
BUCKET = "your-bucket-name"
REGION = "your-region"
```

---

## 7. Run locally

With the SSH tunnel open (or direct Aurora access):

```bash
streamlit run app.py
```

---

## 8. Deploy to EC2

Build and push the Docker image:

```bash
# Build
docker build -t your-app .

# Push to ECR
aws ecr get-login-password --region <region> --profile <profile> | \
    docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker tag your-app:latest <account-id>.dkr.ecr.<region>.amazonaws.com/your-app:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/your-app:latest

# On EC2: pull and run
docker pull <account-id>.dkr.ecr.<region>.amazonaws.com/your-app:latest
docker run -d --name your-app -p 8501:8501 --env-file .env \
    <account-id>.dkr.ecr.<region>.amazonaws.com/your-app:latest
```

Assign an Elastic IP for a stable URL. The Dockerfile handles model downloads at build time so cold starts are fast.

---

## 9. Customize auth

User credentials live in `users.yaml` (gitignored). Generate a password hash:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(input('Password: ').encode(), bcrypt.gensalt()).decode())"
```

Add to `users.yaml`:

```yaml
yourname:
  name: yourname
  password_hash: <paste hash here>
```

Guest access is capped at 3 queries/day by default — change `GUEST_DAILY_LIMIT` in `src/auth.py`.

---

## 10. Build your eval set

Create an `eval/eval_set.jsonl` file with one JSON object per line:

```json
{"question": "What is the optimal temperature for X?", "source": "YourDocument.pdf", "answer": "The optimal temperature is 450°C."}
```

Run retrieval evaluation:

```bash
python -m src.eval.metrics
```

Run answer quality evaluation (RAGAS):

```bash
python -m src.eval.ragas_eval
```

---

## Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
app.py                 # Streamlit entry point — customize UI here
src/
  ingestion/           # Add new file format loaders here
  chunking/            # Adjust chunk size, overlap, strategy
  retrieval/           # Swap embedding models or vector DBs
  generation/          # Change LLM or prompt template
  eval/                # Add your own eval questions
  auth.py              # User login + guest rate limiting
  download.py          # S3 presigned URL downloads
scripts/               # Ingestion pipeline + utilities
configs/               # Pipeline parameters (contextual_rerank.yaml)
tests/                 # Unit + integration tests
```

---

## Common customizations

| Want to... | Change... |
|-----------|-----------|
| Use a different embedding model | `EMBED_MODEL` in `app.py` and `scripts/ingest.py` |
| Use a different LLM | `LLM_MODEL` in `app.py` |
| Change chunk size | `chunk_recursive()` params in `src/chunking/chunkers.py` |
| Add a new file format | Add a loader in `src/ingestion/loaders.py` |
| Change the UI theme | `.streamlit/config.toml` and CSS in `app.py` |
| Increase retrieval candidates | `FETCH_K` and `TOP_N` in `app.py` |
