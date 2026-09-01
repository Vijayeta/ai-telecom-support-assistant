"""Streamlit UI for the NovaCell RAG telecom customer-care chatbot.

Run:  streamlit run app.py

Covers FR-01 to FR-05a and FR-13a: free-text chat, one-click sample questions,
session history, clear-conversation, token streaming, per-answer feedback and
an expandable Sources section.
"""

from __future__ import annotations

import time
import uuid

import streamlit as st

import config
import interaction_log
from rag_chain import Answer, TelecomRAG
from retriever import describe, is_knowledge_base_ready, knowledge_base_status
from sample_questions import SAMPLE_QUESTIONS

st.set_page_config(
    page_title="NovaCell Support Assistant",
    page_icon="📶",
    layout="centered",
    initial_sidebar_state="expanded",
)

GREETING = (
    "Hi! I'm the NovaCell support assistant. Ask me about data speeds, billing, "
    "roaming, SIM or eSIM setup, call quality or the MyTelecom app - I answer "
    "from NovaCell's own FAQ, resolved support tickets and user guides."
)


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the knowledge base...")
def get_rag() -> TelecomRAG:
    """Built once per server process - the embedding model load is expensive."""
    return TelecomRAG()


# --------------------------------------------------------------------------
# Session state (FR-03)
# --------------------------------------------------------------------------
def reset_conversation() -> None:
    """FR-04 - a fresh session with no prior context (US-05)."""
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.feedback = {}


if "messages" not in st.session_state:
    reset_conversation()
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def render_sources(sources: list[str], key: str) -> None:
    """Expandable Sources section for one answer (FR-13a)."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            st.markdown(f"- `{source}`")


def render_feedback(interaction_id: str, key: str) -> None:
    """Thumbs up / down, written to the interaction log (FR-05a)."""
    existing = st.session_state.feedback.get(interaction_id)
    if existing == "up":
        st.caption("Thanks for the feedback.")
        return
    if existing == "down":
        st.caption("Thanks - this answer has been flagged for review.")
        return

    up, down, _ = st.columns([1, 1, 10])
    if up.button("👍", key=f"up-{key}", help="This answer helped"):
        st.session_state.feedback[interaction_id] = "up"
        interaction_log.log_feedback(
            interaction_id, st.session_state.session_id, "up"
        )
        st.rerun()
    if down.button("👎", key=f"down-{key}", help="This answer did not help"):
        st.session_state.feedback[interaction_id] = "down"
        interaction_log.log_feedback(
            interaction_id, st.session_state.session_id, "down"
        )
        st.rerun()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("NovaCell Support")
    st.caption("Grounded in NovaCell's FAQ, resolved tickets and user guides.")

    st.subheader("Sample questions")
    for index, question in enumerate(SAMPLE_QUESTIONS):
        # FR-02 - one click sends the question straight through.
        if st.button(question, key=f"sample-{index}", use_container_width=True):
            st.session_state.pending_question = question

    st.divider()
    if st.button("Clear conversation", use_container_width=True):  # FR-04
        reset_conversation()
        st.rerun()

    st.divider()
    with st.expander("Knowledge base"):
        status = knowledge_base_status()
        for spec, count in status:
            st.markdown(f"**{spec.label}** - {count} docs  \n{spec.description}")
        st.caption(
            f"Retrieving top-{config.TOP_K} per source - "
            f"{len(status) * config.TOP_K} context documents per question"
        )
    with st.expander("Model"):
        st.markdown(
            f"**LLM** `{config.GROQ_MODEL}` (Groq, temperature "
            f"{config.LLM_TEMPERATURE:g})  \n"
            f"**Embeddings** `{config.EMBEDDING_MODEL}` (local)"
        )

    st.caption(
        "This assistant cannot see your account, balance or bill. "
        f"For account-specific help, {config.ESCALATION_TEXT}."
    )


# --------------------------------------------------------------------------
# Start-up checks
# --------------------------------------------------------------------------
st.title("NovaCell Support Assistant")


@st.cache_resource(show_spinner="First run: building the knowledge base...")
def ensure_knowledge_base() -> bool:
    """Build the Chroma store once per container if it is missing.

    `chroma_store/` is deliberately gitignored, so a fresh deployment (Streamlit
    Community Cloud, or any new clone) starts with no vectors while the source
    files in `data/` are committed. Ingesting on first boot keeps the binary
    store out of the repo without leaving the deployed app dead on arrival.
    A local run that has already ingested skips straight through.
    """
    if is_knowledge_base_ready():
        return True

    import ingest_all

    ingest_all.main()
    return is_knowledge_base_ready()


if not ensure_knowledge_base():
    st.error(
        "The knowledge base could not be built. Check that `data/faq.csv`, "
        "`data/tickets.db` and `data/telecom_guide.pdf` are present, or run "
        "`python ingest_all.py` locally."
    )
    st.stop()

if not config.GROQ_API_KEY:
    st.error(
        "`GROQ_API_KEY` is not set. Copy `.env.example` to `.env` and add your "
        "key from https://console.groq.com/keys, then restart the app."
    )
    st.stop()


# --------------------------------------------------------------------------
# Conversation history (FR-03)
# --------------------------------------------------------------------------
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("interaction_id"):
            render_sources(message.get("sources", []), key=str(index))
            render_feedback(message["interaction_id"], key=str(index))


# --------------------------------------------------------------------------
# Input -> answer (FR-01, FR-05)
# --------------------------------------------------------------------------
typed = st.chat_input("Ask about your connection, bill, SIM, roaming or calls...")
question = typed or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer = Answer()
        started = time.perf_counter()
        try:
            # st.write_stream renders tokens as they arrive (FR-05).
            text = st.write_stream(get_rag().stream(question, answer))
        except Exception as error:  # network / API failure - stay graceful
            text = (
                "Sorry - I couldn't reach the answering service just now. "
                f"Please try again, or {config.ESCALATION_TEXT}."
            )
            st.markdown(text)
            st.caption(f"Details: {error}")
            answer.documents = []

        latency = time.perf_counter() - started
        sources = [describe(document) for document in answer.documents]
        interaction_id = interaction_log.new_interaction_id()
        interaction_log.log_answer(
            interaction_id=interaction_id,
            session_id=st.session_state.session_id,
            question=question,
            answer=text,
            sources=sources,
            latency_seconds=round(latency, 2),
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": text,
                "sources": sources,
                "interaction_id": interaction_id,
            }
        )
        # Re-run so the new answer's Sources + feedback controls render with
        # stable widget keys alongside the rest of the history.
        st.rerun()
