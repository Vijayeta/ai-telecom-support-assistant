"""Append-only interaction log (FR-05a).

Every question/answer pair and every thumbs-up / thumbs-down is written as one
JSON object per line to `logs/interactions.jsonl`, giving support ops a record
of which answers landed and which sources were used.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import config


def _append(record: dict[str, Any]) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with config.INTERACTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def new_interaction_id() -> str:
    return uuid.uuid4().hex[:12]


def log_answer(
    interaction_id: str,
    session_id: str,
    question: str,
    answer: str,
    sources: list[str],
    latency_seconds: float | None = None,
    channel: str = "streamlit",
) -> None:
    _append(
        {
            "event": "answer",
            "interaction_id": interaction_id,
            "session_id": session_id,
            "channel": channel,
            "question": question,
            "answer": answer,
            "sources": sources,
            "latency_seconds": latency_seconds,
        }
    )


def log_feedback(
    interaction_id: str, session_id: str, rating: str, channel: str = "streamlit"
) -> None:
    """rating is 'up' or 'down'."""
    _append(
        {
            "event": "feedback",
            "interaction_id": interaction_id,
            "session_id": session_id,
            "channel": channel,
            "rating": rating,
        }
    )
