"""Interactive CLI for the NovaCell RAG telecom assistant (FR-18, FR-19).

Run:  python main.py

Type a question and the answer streams back token by token, followed by the
sources it was grounded in. Type `quit` (or `exit`) to leave.
"""

from __future__ import annotations

import sys
import time
import uuid

import config
import interaction_log
from rag_chain import Answer, TelecomRAG
from retriever import describe, is_knowledge_base_ready, knowledge_base_status
from sample_questions import SAMPLE_QUESTIONS

EXIT_WORDS = {"quit", "exit", ":q"}

BANNER = """
==========================================================
  NovaCell Support Assistant (CLI)
  Grounded in FAQ + resolved tickets + user guides.
  Type your question, or 'quit' to exit.
==========================================================
"""


def preflight() -> bool:
    """Fail fast with actionable messages instead of a stack trace."""
    if not is_knowledge_base_ready():
        print("The knowledge base is empty or incomplete:")
        for spec, count in knowledge_base_status():
            state = "ok" if count else "MISSING - run the ingest script"
            print(f"  - {spec.label:<8} {count:>4} docs  ({state})")
        print("\nBuild it with:  python ingest_all.py")
        return False

    if not config.GROQ_API_KEY:
        print(
            "GROQ_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key from "
            "https://console.groq.com/keys"
        )
        return False

    return True


def main() -> int:
    print(BANNER)
    if not preflight():
        return 1

    print("Loading the local embedding model...")
    rag = TelecomRAG()
    session_id = uuid.uuid4().hex[:12]

    print("\nTry one of these:")
    for question in SAMPLE_QUESTIONS[:4]:
        print(f"  - {question}")
    print()

    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not question:
            continue
        if question.lower() in EXIT_WORDS:  # FR-19
            print("Goodbye.")
            return 0

        answer = Answer()
        started = time.perf_counter()
        print("\nBot > ", end="", flush=True)
        try:
            for token in rag.stream(question, answer):
                print(token, end="", flush=True)
        except Exception as error:
            print(
                "\nSorry - I couldn't reach the answering service just now. "
                f"Please try again, or {config.ESCALATION_TEXT}."
                f"\n(details: {error})"
            )
            continue

        latency = time.perf_counter() - started
        sources = [describe(document) for document in answer.documents]

        print(f"\n\nSources ({len(sources)}):")
        for source in sources:
            print(f"  - {source}")
        print(f"[{latency:.1f}s]\n")

        interaction_log.log_answer(
            interaction_id=interaction_log.new_interaction_id(),
            session_id=session_id,
            question=question,
            answer=answer.text,
            sources=sources,
            latency_seconds=round(latency, 2),
            channel="cli",
        )


if __name__ == "__main__":
    sys.exit(main())
