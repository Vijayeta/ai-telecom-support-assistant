"""Ingest the FAQ knowledge source (FR-14).

    data/faq.csv  ->  chroma_store/  collection "faq"

One CSV row becomes exactly one vector document. Re-running is safe: the
collection is rebuilt from scratch each time (FR-17), so support ops can edit
faq.csv and re-run this script without redeploying anything (US-06).

Usage:  python ingest_faq.py
"""

from __future__ import annotations

import csv
import sys

from langchain_core.documents import Document

import config
from vectorstore import rebuild_collection


def load_faq_documents() -> tuple[list[Document], list[str]]:
    if not config.FAQ_CSV.exists():
        raise FileNotFoundError(f"FAQ source not found: {config.FAQ_CSV}")

    documents: list[Document] = []
    ids: list[str] = []

    with config.FAQ_CSV.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=1):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue

            faq_id = (row.get("id") or str(row_number)).strip()
            category = (row.get("category") or "general").strip()

            # Embed question + answer together: customers phrase queries like
            # the question, but answer wording carries the matching keywords.
            documents.append(
                Document(
                    page_content=f"Q: {question}\nA: {answer}",
                    metadata={
                        "source_type": "FAQ",
                        "identifier": f"FAQ-{faq_id}",
                        "question": question,
                        "category": category,
                        "origin": config.FAQ_CSV.name,
                    },
                )
            )
            ids.append(f"faq-{faq_id}")

    return documents, ids


def main() -> int:
    documents, ids = load_faq_documents()
    count = rebuild_collection(config.FAQ_COLLECTION, documents, ids)
    print(f"[faq] ingested {count} FAQ entries into '{config.FAQ_COLLECTION}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
