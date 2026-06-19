# ⚡ BEKE — Battery Engineering Knowledge Engine

A production RAG system for semantic search over a multi-modal PhD research corpus (1,200 Word, PowerPoint, and PDF documents). Ask a natural-language question about aluminum-air batteries and micro-quadrotor drones — get a grounded answer with source citations and one-click file downloads.

![BEKE Demo](static/website_demo.png)

**Live demo:** [http://184.72.59.84:8501](http://184.72.59.84:8501) — log in or browse as a guest (3 queries/day).

---

## Architecture

```
User query
    │
    ▼
BGE-base-en-v1.5 dense retrieval       ← fetch k=50 from Aurora pgvector
    │
    ▼
bge-reranker-v2-m3 (cross-encoder)     ← re-score, keep top-5
    │
    ▼
gpt-4o-mini (grounded generation)      ← answer + inline citations
```

Each chunk is enriched at index time with 1–2 LLM-generated sentences situating it within its source document ([Anthropic-style contextual retrieval](https://www.anthropic.com/news/contextual-retrieval)). This bridges vocabulary gaps that neither BM25 nor dense embeddings can cross — a chunk about "3× power density improvement" gets context identifying the specific cathode alloy and experiment section.

### Infrastructure

| Component      | Service                                                 |
| -------------- | ------------------------------------------------------- |
| Corpus storage | S3 (1,200 files, ~50 GB)                                |
| Vector index   | Aurora PostgreSQL Serverless v2 + pgvector (84K chunks) |
| Web app        | EC2 t3.medium, Streamlit, Docker                        |
| Embeddings     | BGE-base-en-v1.5 (local, CPU)                           |
| Reranker       | BGE-reranker-v2-m3 (local, CPU)                         |
| LLM            | gpt-4o-mini (API)                                       |

---

## Retrieval Ablation (30-question eval set)

Five retriever configurations evaluated on the same eval set and chunks. Each row adds one technique on top of the previous.

| #            | Retriever                     | Hit@1          | MRR            | nDCG@5         |
| ------------ | ----------------------------- | -------------- | -------------- | -------------- |
| 3A           | Dense only                    | 0.60           | 0.76           | 0.83           |
| 3B           | + BM25 hybrid                 | 0.80           | 0.89           | 0.91           |
| 3C           | + cross-encoder rerank        | 0.97           | 0.98           | 0.97           |
| 3D           | + multi-query                 | 0.97           | 0.98           | 0.97           |
| **3E** | **+ contextual chunks** | **1.00** | **1.00** | **0.98** |

**Key takeaways:**

- **BM25 hybrid (+20pp Hit@1):** Chemical terms like "Al(OH)₄⁻", "ECSA", and measurement values that dense embeddings blurred were matched exactly by BM25. Single largest gain.
- **Cross-encoder rerank (+17pp):** The correct chunk was already in top-50 but poorly ranked. Full (query, passage) attention pushed it to rank 1.
- **Multi-query (+0pp):** LLM paraphrases didn't explore a sufficiently different embedding region — the remaining miss was a ranking problem, not a phrasing problem.
- **Contextual enrichment (+3pp to perfect):** LLM-generated context supplied the missing vocabulary at index time, resolving the final failure case.

### Answer Quality (RAGAS, 5-question subset)

| Metric            | Score |
| ----------------- | ----- |
| Faithfulness      | 0.93  |
| Answer relevancy  | 0.97  |
| Context precision | 0.93  |
| Context recall    | 0.80  |

93% of claims in generated answers are grounded in retrieved passages. Context recall (0.80) is the lowest metric — expected on a 5-file subset; the full corpus raises it as more relevant material becomes retrievable.

---

## Methodology

Built as a two-phase research project with a structured bake-off approach: for each challenge (ingestion, chunking, retrieval), three approaches were implemented and compared on the same eval set before selecting a winner.

**Challenge 1 — Ingestion:** Native parsers (python-docx, python-pptx, PyMuPDF) selected over LangChain's UnstructuredLoader for richer metadata and full control over speaker notes, headings, and table extraction.

**Challenge 2 — Chunking:** Recursive text splitting (1000 chars, 150 overlap) selected over semantic chunking (non-deterministic, high variance) and hierarchical chunking (higher complexity, marginal gain). Full analysis in `report.md`.

**Challenge 3 — Retrieval:** Contextual enrichment + hybrid BM25/dense + cross-encoder rerank selected as the winning configuration. See ablation table above.

**Phase 2 — Scale-up:** Migrated from local Chroma to Aurora pgvector. Built a streaming ingestion pipeline with checkpointing that processed 1,102 files overnight. Deployed as a Dockerized Streamlit app on EC2 with user auth, S3 presigned-URL downloads, and a cyberpunk UI theme.

---

## Tech Stack

| Layer      | Technology                                                   |
| ---------- | ------------------------------------------------------------ |
| Language   | Python 3.11                                                  |
| Framework  | LangChain, Streamlit                                         |
| Embeddings | BAAI/bge-base-en-v1.5 (HuggingFace)                          |
| Reranker   | BAAI/bge-reranker-v2-m3                                      |
| Vector DB  | Aurora PostgreSQL + pgvector (prod) / Chroma (Phase 1 local) |
| LLM        | gpt-4o-mini                                                  |
| Ingestion  | python-docx, python-pptx, PyMuPDF                            |
| Auth       | bcrypt + YAML credentials                                    |
| Deployment | Docker on EC2, S3, Aurora Serverless v2                      |

---

## Repository Layout

```
app.py                 # Streamlit entry point
src/
  ingestion/           # Word/PPT/PDF parsers + LLM context enrichment
  chunking/            # recursive/semantic/hierarchical chunkers
  retrieval/           # pgvector index, hybrid BM25/dense, cross-encoder rerank
  generation/          # grounded QA with citations
  eval/                # Hit@k, MRR, nDCG, RAGAS evaluation
  auth.py              # login + guest rate limiting
  download.py          # S3 presigned URL downloads
scripts/               # ingestion pipeline, corpus upload, utilities
eval/                  # hand-crafted eval sets (30 + 50 questions)
configs/               # pipeline parameters
tests/                 # unit + integration tests
report.md              # full Phase 1 write-up
```

---

## Build Your Own

Want to use this codebase with your own documents and AWS setup? See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide.
