# BEKE — Streamlit app container
# Build:  docker build -t beke-app .
# Run:    docker run -p 8501:8501 --env-file .env beke-app
# Deploy: push to ECR → App Runner pulls and runs

# ---------------------------------------------------------------------------
# Base image — Python 3.11 slim (plan spec; smaller than full image)
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# gcc / g++     : compile psycopg C extensions
# libpq-dev     : PostgreSQL client headers (required by psycopg2-binary)
# curl          : health-check convenience; not strictly required
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies
# torch installed first as a separate layer — cache it independently
# so code changes don't trigger a full torch re-download (~700 MB)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Pre-download HuggingFace model FILES into the image layer
# Uses snapshot_download (no torch import needed) — avoids version conflicts.
# Models are cached at /root/.cache/huggingface/
# This prevents a 30-60s cold download on first query in production.
# ---------------------------------------------------------------------------
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-base-en-v1.5'); \
snapshot_download('BAAI/bge-reranker-v2-m3'); \
print('Models pre-downloaded successfully')"

# ---------------------------------------------------------------------------
# Copy application source
# (requirements already installed above — copy code last so code changes
#  don't invalidate the dependency cache layer)
# ---------------------------------------------------------------------------
COPY app.py .
COPY src/ src/
COPY configs/ configs/
COPY users.yaml .
COPY .streamlit/ .streamlit/
COPY static/ static/

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
EXPOSE 8501

# Streamlit config: disable telemetry, bind to all interfaces
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Credentials come from App Runner environment variables (Secrets Manager)
# — NOT from .env (which is not copied into the image)

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
