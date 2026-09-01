"""Chroma persistence helpers shared by the ingest scripts and the retriever.

All collections live in a single on-disk Chroma store (`chroma_store/`) so the
app never re-ingests at start-up (NFR-05).
"""

from __future__ import annotations

from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from embeddings import get_embeddings


def get_store(collection_name: str) -> Chroma:
    """Open (or create) a persisted Chroma collection."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def rebuild_collection(
    collection_name: str, documents: Iterable[Document], ids: Iterable[str]
) -> int:
    """Replace a collection's contents with `documents`.

    Dropping and rewriting makes every ingest script idempotent (FR-17): a
    re-run after editing, adding or deleting source rows leaves the collection
    exactly matching the source, with no duplicates and no stale documents.
    """
    documents = list(documents)
    ids = list(ids)

    store = get_store(collection_name)
    try:
        store.delete_collection()
    except Exception:  # collection did not exist yet - nothing to drop
        pass

    store = get_store(collection_name)
    if documents:
        store.add_documents(documents=documents, ids=ids)
    return len(documents)


def collection_count(collection_name: str) -> int:
    """Number of documents currently stored in a collection (0 if missing)."""
    try:
        return get_store(collection_name)._collection.count()
    except Exception:
        return 0
