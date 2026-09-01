"""Run every ingest script in sequence (FR-14, FR-15, FR-16).

Usage:  python ingest_all.py
"""

from __future__ import annotations

import sys

import ingest_faq
import ingest_guides
import ingest_tickets


def main() -> int:
    print("Building knowledge base -> chroma_store/ ...")
    for module in (ingest_faq, ingest_tickets, ingest_guides):
        module.main()
    print("Done. Run 'streamlit run app.py' or 'python main.py'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
