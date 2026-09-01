"""Shared local embedding function.

Loaded once per process and reused by every ingest script and by the retriever,
so the ~90 MB all-MiniLM-L6-v2 model is downloaded once and held in memory once
(FR-09, NFR-02).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the process-wide local embedding model."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
