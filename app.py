"""BEKE — Battery Engineering Knowledge Engine (Streamlit web interface).

Pipeline (query time):
  pgvector dense retrieval -> bge-reranker-v2-m3 -> gpt-4o-mini -> answer

Run locally (full pipeline — needs Aurora tunnel + .env):
    streamlit run app.py

Run locally (mock mode — UI preview, no DB needed):
    MOCK_MODE=1 streamlit run app.py

Prerequisites (full mode): SSH tunnel on port 5433, AURORA_DB_URL + OPENAI_API_KEY in .env
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.auth import require_auth, is_guest, guest_query_allowed
from src.download import render_download_button
from src.generation.generator import RAGAnswer

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION   = "beke_contextual"
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL    = "gpt-4o-mini"
FETCH_K      = 50
TOP_N        = 5

MOCK_MODE = os.getenv("MOCK_MODE", "").strip().lower() in ("1", "true", "yes")

# Corpus stats — update after running: python check_corpus_stats.py
CORPUS_DOCS   = "1,200"
CORPUS_CHUNKS = "84K"
CORPUS_DATE   = "Jun 2026"

# Hero banner image — place your drone image at static/drone_banner.png
BANNER_PATH = Path("static/drone_banner.png")

# Example queries for the chips
EXAMPLE_QUERIES = [
    "What electrolyte concentration gives the highest energy density?",
    "How long can the micro-quadrotor fly on an Al-air cell?",
    "What is the peak power density of the Ag28Cu72 cathode?",
    "How does parasitic corrosion limit Al-air battery performance?",
]


# ---------------------------------------------------------------------------
# Cyberpunk CSS — neon glow on title + pulsing input border + scanlines
# ---------------------------------------------------------------------------

_CYBERPUNK_CSS = """
<style>
/* Neon glow on the main title */
h1 {
    text-shadow:
        0 0 10px rgba(0, 255, 136, 0.6),
        0 0 30px rgba(0, 255, 136, 0.3),
        0 0 60px rgba(0, 255, 136, 0.15) !important;
}

/* Pulsing border on the text input */
.stTextInput > div > div > input {
    border: 1px solid #1e3a5f !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00ff88 !important;
    box-shadow: 0 0 8px rgba(0, 255, 136, 0.4),
                0 0 20px rgba(0, 255, 136, 0.15) !important;
}

/* Subtle glow on expander headers (source citations) */
.streamlit-expanderHeader {
    text-shadow: 0 0 6px rgba(0, 255, 136, 0.3) !important;
}

/* Example query buttons — outlined cyberpunk style */
div.stButton > button[kind="secondary"] {
    border: 1px solid #1e3a5f !important;
    background: transparent !important;
    color: #00d4ff !important;
    transition: all 0.2s ease !important;
    font-size: 0.85rem !important;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: #00ff88 !important;
    color: #00ff88 !important;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.3) !important;
}

/* Metric cards — subtle glow */
div[data-testid="stMetric"] {
    border: 1px solid #1e3a5f;
    border-radius: 0.2rem;
    padding: 12px 16px;
    background: rgba(17, 25, 39, 0.6);
}
div[data-testid="stMetric"] label {
    color: #5a7a8a !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #00ff88 !important;
    text-shadow: 0 0 8px rgba(0, 255, 136, 0.3) !important;
}

/* Banner image — slight vignette effect */
img[data-testid="stImage"] {
    border-radius: 4px;
    border: 1px solid #1e3a5f;
}
</style>
"""


# ---------------------------------------------------------------------------
# Mock pipeline — returns canned answers for UI development
# ---------------------------------------------------------------------------

def _build_mock_pipeline():
    """Return a fake RAG callable for UI preview without any infra."""
    from langchain_core.documents import Document

    _MOCK_SOURCES = [
        Document(
            page_content=(
                "Context: This section describes the electrochemical performance "
                "of aluminum-air batteries using 4M KOH electrolyte. The Al-0.5Mg-0.1Sn "
                "alloy anode achieved a peak energy density of 420 Wh/kg at 25 mA/cm2 "
                "current density, with an open-circuit voltage of 1.78V."
            ),
            metadata={"source": "/corpus/AAB_Manuscript.docx", "chunk_idx": 42},
        ),
        Document(
            page_content=(
                "The micro-quadrotor drone platform requires an energy source with "
                "specific energy exceeding 400 Wh/kg to achieve the target 45-minute "
                "flight endurance. Al-air batteries meet this threshold while "
                "maintaining a favorable power-to-weight ratio of 0.8 kW/kg."
            ),
            metadata={"source": "/corpus/Drone_Power_Requirements.pdf", "chunk_idx": 15},
        ),
        Document(
            page_content=(
                "Parasitic corrosion of the aluminum anode in alkaline electrolyte "
                "generates hydrogen gas, reducing coulombic efficiency. Adding 50 ppm "
                "ZnO inhibitor to the 4M KOH solution suppressed H2 evolution by 73%, "
                "improving utilization efficiency from 81% to 94%."
            ),
            metadata={"source": "/corpus/Corrosion_Inhibition_Study.docx", "chunk_idx": 8},
        ),
    ]

    _MOCK_ANSWER = (
        "The optimal KOH electrolyte concentration for Al-air batteries is **4M**, "
        "which yields a peak energy density of **420 Wh/kg** at 25 mA/cm² current "
        "density [1]. This exceeds the **400 Wh/kg** threshold required for the "
        "micro-quadrotor drone platform to achieve 45-minute flight endurance [2]. "
        "Adding 50 ppm ZnO inhibitor to the electrolyte further improves coulombic "
        "efficiency from 81% to 94% by suppressing parasitic hydrogen evolution [3]."
    )

    def mock_rag(query: str) -> RAGAnswer:
        time.sleep(0.5)  # simulate latency
        return RAGAnswer(question=query, answer=_MOCK_ANSWER, sources=_MOCK_SOURCES)

    return mock_rag


# ---------------------------------------------------------------------------
# Real pipeline — pgvector + reranker + LLM
# ---------------------------------------------------------------------------

@st.cache_resource
def build_pipeline():
    """Load all models and return a callable rag(query) -> RAGAnswer."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    from src.retrieval.index_pgvector import load_pgvector_index
    from src.retrieval.rerank import make_reranking_retriever
    from src.generation.generator import ask

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = load_pgvector_index(embeddings, collection_name=COLLECTION)
    dense       = vectorstore.as_retriever(search_kwargs={"k": FETCH_K})
    reranker    = make_reranking_retriever(dense, top_n=TOP_N, fetch_k=FETCH_K)
    llm         = ChatOpenAI(model=LLM_MODEL, temperature=0)

    return lambda query: ask(query, reranker, llm, k=TOP_N)


# ---------------------------------------------------------------------------
# Render answer + citations
# ---------------------------------------------------------------------------

def render_answer(result: RAGAnswer) -> None:
    """Display the RAGAnswer with source citations."""
    st.markdown(result.answer)

    if not result.sources:
        return

    st.markdown("---")
    st.markdown("**Sources**")

    for i, doc in enumerate(result.sources, 1):
        filename  = doc.metadata.get("source", "unknown").split("/")[-1]
        chunk_idx = doc.metadata.get("chunk_idx", "?")
        excerpt   = doc.page_content[:300]

        with st.expander(f"[{i}] {filename}  (chunk {chunk_idx})"):
            st.markdown(f"```\n{excerpt}\n```")
            render_download_button(doc.metadata.get("source", ""))

    # ---- Download All button (logged-in users only) ----
    if not is_guest():
        from src.download import generate_presigned_url, EXPIRY_S

        if st.button("⬇ Download All Sources", key="download_all"):
            urls = []
            for doc in result.sources:
                s3_key = doc.metadata.get("source", "").lstrip("/")
                if not s3_key:
                    continue
                url = generate_presigned_url(s3_key)
                if url:
                    urls.append(url)

            if urls:
                # Inject JS to trigger all downloads at once via hidden iframes
                iframes = "".join(
                    f'<iframe src="{u}" style="display:none;"></iframe>'
                    for u in urls
                )
                st.markdown(iframes, unsafe_allow_html=True)
                st.caption(f"✅ Downloading {len(urls)} file(s). Links expire in {EXPIRY_S // 60} minutes.")
            else:
                st.warning("Could not generate download links — check AWS credentials.")


# ---------------------------------------------------------------------------
# Main — page layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="BEKE — Battery Knowledge Engine",
        page_icon="⚡",
        layout="wide",
    )

    # ---- Idea 6: inject cyberpunk CSS ----
    st.markdown(_CYBERPUNK_CSS, unsafe_allow_html=True)

    # ---- Auth gate (login page inherits the cyberpunk CSS above) ----
    require_auth()

    # ---- Header + Hero banner overlay ----
    if BANNER_PATH.exists():
        import base64
        banner_b64 = base64.b64encode(BANNER_PATH.read_bytes()).decode()
        ext = BANNER_PATH.suffix.lstrip(".")
        mime = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"
        st.markdown(f"""
        <div style="
            position: relative;
            width: 100%;
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid #1e3a5f;
            margin-bottom: 1rem;
        ">
            <img src="data:{mime};base64,{banner_b64}"
                 style="width: 100%; display: block;" />
            <div style="
                position: absolute;
                bottom: 0; left: 0; right: 0;
                padding: 24px 32px;
                background: linear-gradient(transparent, rgba(10, 14, 23, 0.85));
            ">
                <div style="
                    font-size: 2.4rem;
                    font-weight: 800;
                    color: #00ff88;
                    letter-spacing: 3px;
                    text-shadow: 0 0 10px rgba(0,255,136,0.6),
                                 0 0 30px rgba(0,255,136,0.3);
                    line-height: 1.1;
                ">BEKE - Battery Engineering Knowledge Engine</div>
                <div style="
                    font-size: 0.9rem;
                    color: #c8d6e5;
                    margin-top: 4px;
                    text-shadow: 0 0 6px rgba(0,0,0,0.8);
                "> Al-Air batteries &amp; micro-quadrotor drones.
                   Answers grounded in research corpus.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback if no banner image
        st.title("⚡ BEKE")
        st.caption(
            "Battery Engineering Knowledge Engine — "
            "Al-Air batteries & micro-quadrotor drones. "
            "Answers grounded in your research corpus."
        )

    if MOCK_MODE:
        st.info("🛠️ **Mock mode** — UI preview, no database connection.", icon="⚠️")

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### Engine Core")
        st.markdown(f"`Embeddings` {EMBED_MODEL}")
        st.markdown(f"`Reranker ` {RERANK_MODEL}")
        st.markdown(f"`LLM     ` {LLM_MODEL}")
        st.markdown(f"`Fetch K ` {FETCH_K} → rerank → top {TOP_N}")

        st.markdown("---")

        st.markdown("### Data Vault")
        st.markdown("`Storage ` AWS S3")
        st.markdown("`Index   ` Aurora pgvector")
        st.markdown(f"`Docs    ` {CORPUS_DOCS}")
        st.markdown(f"`Chunks  ` {CORPUS_CHUNKS}")
        st.markdown(f"`Indexed ` {CORPUS_DATE}")

        if MOCK_MODE:
            st.caption("⚠️ Running in mock mode")

    # ---- Load pipeline ----
    if MOCK_MODE:
        rag = _build_mock_pipeline()
    else:
        with st.spinner("Loading knowledge base — first load takes ~30s..."):
            rag = build_pipeline()

    # ---- Query input ----
    query = st.text_input(
        "Ask a question:",
        placeholder="e.g. What electrolyte concentration gives the highest energy density?",
    )

    # ---- Idea 2: Example query chips (show only when no query yet) ----
    if not query:
        st.markdown("**Try an example:**")
        cols = st.columns(2)
        for i, eq in enumerate(EXAMPLE_QUERIES):
            if cols[i % 2].button(eq, key=f"eq_{i}", use_container_width=True):
                # Set the query and rerun so the answer renders below
                st.session_state["prefilled_query"] = eq
                st.rerun()

    # Check if an example chip was clicked
    if not query and "prefilled_query" in st.session_state:
        query = st.session_state.pop("prefilled_query")

    # Fall back to last query if text input is empty (happens on button-click reruns)
    if not query:
        query = st.session_state.get("last_query", "")

    # ---- Run pipeline + render ----
    if query:
        # Only call the pipeline if the query is new (avoid re-running on button clicks)
        if query != st.session_state.get("last_query"):
            if is_guest() and not guest_query_allowed():
                st.warning(
                    "Daily guest limit reached (3 queries/day). "
                    "Please log in for unlimited access."
                )
                return

            t0 = time.time()
            with st.spinner("Searching and reasoning..."):
                try:
                    result = rag(query)
                except Exception as exc:
                    st.error(
                        f"Connection error — is the Aurora tunnel open?\n\n`{exc}`"
                    )
                    return

            elapsed = time.time() - t0
            st.session_state["last_query"] = query
            st.session_state["last_result"] = result
            st.session_state["last_elapsed"] = elapsed

        # Display cached result (survives button-click reruns)
        result = st.session_state.get("last_result")
        elapsed = st.session_state.get("last_elapsed", 0)
        if result:
            st.caption(f"Query time: {elapsed:.1f}s")
            render_answer(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
