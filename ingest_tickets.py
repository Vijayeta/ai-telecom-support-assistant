"""Ingest resolved support tickets (FR-15).

    data/tickets.db  ->  chroma_store/  collection "tickets"

One resolved ticket becomes exactly one vector document. Only tickets with
status 'resolved' are indexed - an open ticket has no verified resolution to
ground an answer in. Re-running rebuilds the collection (FR-17), so newly
seeded tickets show up immediately (US-07).

Usage:  python ingest_tickets.py
"""

from __future__ import annotations

import sqlite3
import sys

from langchain_core.documents import Document

import config
from vectorstore import rebuild_collection


def load_ticket_documents() -> tuple[list[Document], list[str]]:
    if not config.TICKETS_DB.exists():
        raise FileNotFoundError(f"Ticket database not found: {config.TICKETS_DB}")

    connection = sqlite3.connect(config.TICKETS_DB)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT ticket_id, category, issue_type, description, resolution
            FROM tickets
            WHERE LOWER(status) = 'resolved'
            ORDER BY ticket_id
            """
        ).fetchall()
    finally:
        connection.close()

    documents: list[Document] = []
    ids: list[str] = []

    for row in rows:
        ticket_id = row["ticket_id"]
        documents.append(
            Document(
                page_content=(
                    f"Issue: {row['issue_type']}\n"
                    f"Category: {row['category']}\n"
                    f"Customer reported: {row['description']}\n"
                    f"Resolution: {row['resolution']}"
                ),
                metadata={
                    "source_type": "TICKET",
                    "identifier": row["ticket_id"],
                    "issue_type": row["issue_type"],
                    "category": row["category"],
                    "origin": config.TICKETS_DB.name,
                },
            )
        )
        ids.append(f"ticket-{ticket_id}")

    return documents, ids


def main() -> int:
    documents, ids = load_ticket_documents()
    count = rebuild_collection(config.TICKETS_COLLECTION, documents, ids)
    print(
        f"[tickets] ingested {count} resolved tickets into "
        f"'{config.TICKETS_COLLECTION}'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
