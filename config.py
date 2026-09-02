"""Central configuration for the RAG telecom customer-care chatbot.

Every tunable lives here so that ingestion, retrieval, the CLI and the Streamlit
UI all agree on paths, model names and retrieval sizes.

Secrets are never hard-coded: they are read from a local `.env` file (NFR-03).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_store"          # NFR-05: persisted on disk
LOG_DIR = PROJECT_ROOT / "logs"

FAQ_CSV = DATA_DIR / "faq.csv"                      # FR-14
TICKETS_DB = DATA_DIR / "tickets.db"                # FR-15
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"          # FR-16

INTERACTION_LOG = LOG_DIR / "interactions.jsonl"    # FR-05a

# --------------------------------------------------------------------------
# Embeddings (FR-09 / NFR-02) - runs locally, no embedding API cost
# --------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------------------------------
# Collections (FR-06). Adding a source = add an ingest_*.py and register the
# collection name here + in retriever.COLLECTIONS (NFR-06).
# --------------------------------------------------------------------------
FAQ_COLLECTION = "faq"
TICKETS_COLLECTION = "tickets"
GUIDES_COLLECTION = "guides"

TOP_K = int(os.getenv("TOP_K", "3"))                # FR-07: top-3 per collection

# Minimum relevance for a retrieved document to reach the prompt. Chroma's l2
# relevance function maps our normalised embeddings to roughly [-0.41, 1].
# Measured against this knowledge base: in-domain questions score 0.46-0.74 at
# best, while off-topic ones never exceed -0.19, so 0.10 sits in the gap with
# margin on both sides. Below it, retrieval returns nothing and the prompt's
# "no context" path produces a refusal instead of an answer built on noise.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.10"))

# --------------------------------------------------------------------------
# PDF chunking (FR-16)
# --------------------------------------------------------------------------
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --------------------------------------------------------------------------
# LLM (FR-12, FR-13)
# --------------------------------------------------------------------------
def _read_secret(name: str, default: str = "") -> str:
    """Environment variable first, then Streamlit Cloud's secrets store.

    Locally the key comes from `.env`. On Streamlit Community Cloud there is no
    `.env`: the key is pasted into the app's Secrets box, which Streamlit
    normally also exports as an environment variable. Reading `st.secrets` as a
    fallback keeps the app working if it doesn't. The import is guarded so the
    CLI never needs Streamlit loaded.
    """
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit

        return str(streamlit.secrets[name])
    except Exception:
        return default


GROQ_API_KEY = _read_secret("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")

# --------------------------------------------------------------------------
# Escalation copy (FR-11) - kept in one place so support ops can reword it
# --------------------------------------------------------------------------
ESCALATION_TEXT = (
    "call 611 from your NovaCell mobile or use the MyTelecom app"
)


def require_api_key() -> str:
    """Return the Groq API key or raise a clear, actionable error."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key from "
            "https://console.groq.com/keys"
        )
    return GROQ_API_KEY
