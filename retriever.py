"""Merged retriever across the three knowledge collections (FR-06 - FR-08).

The three Chroma collections are queried in parallel, each returning its own
top-3 (FR-07), and the results are merged into one source-labelled context
block (FR-08) that is injected into the prompt.

Anything scoring below `config.RELEVANCE_THRESHOLD` is dropped first, so the
merged block holds at most 9 documents and an off-topic question yields none.

To add a knowledge source (NFR-06): write an `ingest_<source>.py` that writes a
new collection, then append one `CollectionSpec` to COLLECTIONS below. Nothing
else in the app needs to change.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

import config
from vectorstore import collection_count, get_store

# Chroma's l2 relevance function maps our normalised embeddings to roughly
# [-0.41, 1], so LangChain warns on every query that scores fall outside [0, 1].
# The scale is consistent, which is all a threshold needs. Filtered once at
# import rather than per call: RunnableParallel runs the collection branches in
# threads, and `warnings.catch_warnings()` swaps a global, so one branch leaving
# the block would restore the warning underneath the others mid-query.
warnings.filterwarnings(
    "ignore", message="Relevance scores must be between", category=UserWarning
)


@dataclass(frozen=True)
class CollectionSpec:
    """A registered knowledge collection."""

    name: str          # Chroma collection name
    label: str         # how the source is labelled in prompt context + UI
    description: str   # shown in the Streamlit knowledge-base panel


COLLECTIONS: list[CollectionSpec] = [
    CollectionSpec(
        name=config.FAQ_COLLECTION,
        label="FAQ",
        description="Published FAQ entries (data/faq.csv)",
    ),
    CollectionSpec(
        name=config.TICKETS_COLLECTION,
        label="TICKETS",
        description="Resolved support tickets (data/tickets.db)",
    ),
    CollectionSpec(
        name=config.GUIDES_COLLECTION,
        label="GUIDES",
        description="Telecom user guide chunks (data/telecom_guide.pdf)",
    ),
]


def knowledge_base_status() -> list[tuple[CollectionSpec, int]]:
    """(collection, document count) for each registered source."""
    return [(spec, collection_count(spec.name)) for spec in COLLECTIONS]


def is_knowledge_base_ready() -> bool:
    """True when every registered collection holds at least one document."""
    return all(count > 0 for _, count in knowledge_base_status())


def _scored_search(
    spec: CollectionSpec, k: int, threshold: float
) -> Runnable[str, list[Document]]:
    """One collection's top-`k`, dropping anything below `threshold`.

    Without the threshold every question retrieved a full `k` documents per
    collection whatever it asked about, so an off-topic question still arrived
    at the prompt wrapped in nine irrelevant telecom chunks and the refusal
    rested entirely on the model's judgement. Filtering on relevance makes the
    refusal structural: nothing clears the bar, `format_context` reports no
    context, and the answer cannot be built on noise.
    """
    store = get_store(spec.name)

    def run(question: str) -> list[Document]:
        scored = store.similarity_search_with_relevance_scores(question, k=k)

        kept = []
        for document, score in scored:
            if score < threshold:
                continue
            document.metadata["relevance"] = round(float(score), 3)
            kept.append(document)
        return kept

    return RunnableLambda(run)


def build_merged_retriever(
    k: int = config.TOP_K, threshold: float = config.RELEVANCE_THRESHOLD
) -> Runnable[str, list[Document]]:
    """Question -> up to `k` sufficiently relevant documents per collection.

    Branches of a RunnableParallel are executed concurrently, so total
    retrieval latency is roughly that of the slowest collection rather than
    the sum of all three (NFR-01).
    """
    branches = {
        spec.name: _scored_search(spec, k=k, threshold=threshold)
        for spec in COLLECTIONS
    }

    def merge(results: dict[str, list[Document]]) -> list[Document]:
        merged: list[Document] = []
        # Iterate COLLECTIONS (not the dict) to keep source ordering stable.
        for spec in COLLECTIONS:
            for document in results.get(spec.name, []):
                document.metadata.setdefault("source_type", spec.label)
                document.metadata["collection"] = spec.name
                document.metadata["source_label"] = spec.label
                merged.append(document)
        return merged

    return RunnableParallel(branches) | RunnableLambda(merge)


def describe(document: Document) -> str:
    """Short human-readable citation, e.g. 'FAQ | FAQ-12' (FR-13a)."""
    label = document.metadata.get("source_label") or document.metadata.get(
        "source_type", "SOURCE"
    )
    identifier = document.metadata.get("identifier", "unknown")
    return f"{label} | {identifier}"


def format_context(documents: list[Document]) -> str:
    """Render retrieved documents as the source-labelled context block (FR-08).

    Every block is tagged so the model can cite its source and so an answer can
    never silently blend in un-retrieved knowledge.
    """
    if not documents:
        return "NO CONTEXT FOUND"

    blocks = []
    for position, document in enumerate(documents, start=1):
        label = document.metadata.get("source_label", "SOURCE")
        identifier = document.metadata.get("identifier", "unknown")
        blocks.append(
            f"[{position}] SOURCE: {label} | ID: {identifier}\n"
            f"{document.page_content.strip()}"
        )
    return "\n\n---\n\n".join(blocks)
