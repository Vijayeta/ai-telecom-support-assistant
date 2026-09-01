"""Ingest the telecom PDF guide (FR-16).

    data/telecom_guide.pdf  ->  chroma_store/  collection "guides"

The PDF is split into 600-character chunks with 100 characters of overlap so a
procedure that straddles a chunk boundary is still retrievable in full.
Re-running rebuilds the collection (FR-17).

Usage:  python ingest_guides.py
"""

from __future__ import annotations

import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from vectorstore import rebuild_collection


def load_guide_documents() -> tuple[list[Document], list[str]]:
    if not config.GUIDE_PDF.exists():
        raise FileNotFoundError(f"Guide PDF not found: {config.GUIDE_PDF}")

    pages = PyPDFLoader(str(config.GUIDE_PDF)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    documents: list[Document] = []
    ids: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        text = chunk.page_content.strip()
        if not text:
            continue

        # PyPDF page numbers are 0-based; show the human page number.
        page = int(chunk.metadata.get("page", 0)) + 1
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_type": "GUIDE",
                    "identifier": f"{config.GUIDE_PDF.name} p.{page} #{index}",
                    "page": page,
                    "chunk": index,
                    "origin": config.GUIDE_PDF.name,
                },
            )
        )
        ids.append(f"guide-{index}")

    return documents, ids


def main() -> int:
    documents, ids = load_guide_documents()
    count = rebuild_collection(config.GUIDES_COLLECTION, documents, ids)
    print(
        f"[guides] ingested {count} chunks "
        f"({config.CHUNK_SIZE}c / {config.CHUNK_OVERLAP}c overlap) into "
        f"'{config.GUIDES_COLLECTION}'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
