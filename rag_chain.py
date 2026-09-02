"""The RAG chain: retrieve -> prompt -> Groq LLM -> streamed text (LCEL).

Grounding rules (FR-10, FR-11) live in the system prompt: the model may use the
retrieved context and nothing else, and when the context does not cover the
question it must say so and point the customer at a human channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

import config
from retriever import build_merged_retriever, format_context

SYSTEM_PROMPT = """You are NovaCell's telecom customer-care assistant.

You answer Tier-1 support questions for mobile subscribers: connectivity, data,
roaming, SIM/eSIM, billing, voice calls, devices and account/app issues.

GROUNDING RULES - these override everything else:
1. Answer ONLY from the CONTEXT below. The context is NovaCell's verified
   knowledge: published FAQ entries, resolved support tickets and official user
   guides.
2. Never use knowledge from your own training. Never invent or estimate prices,
   plan names, phone numbers, codes, timeframes or policies. If a number is not
   in the context, do not state a number.
3. When the context contains NOTHING that answers the question, your entire
   reply is this one sentence:
   "I don't have information about that in NovaCell's support material - please
   {escalation}."
   That sentence replaces the whole answer. Never append it to an answer you
   have already given, and never combine it with other text. If the context
   covers only part of the question, answer that part and stop - do not add a
   refusal for the rest. Do not guess and do not pad with generic advice.
4. You have no access to the customer's account, balance, bill or usage. For any
   personal-account question, explain that you cannot see individual account
   data and direct them to {escalation}.

STYLE:
- Plain language for a non-technical customer. No jargon unless the context uses it.
- Lead with the direct answer, then numbered steps if the context gives a procedure.
- Keep it under about 150 words.
- Do not mention "context", "documents", "sources" or "retrieval" in your answer.

CONTEXT:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def build_llm() -> ChatGroq:
    """Deterministic, non-reasoning Groq chat model (FR-12, FR-13)."""
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.require_api_key(),
        temperature=config.LLM_TEMPERATURE,
        reasoning_effort=config.REASONING_EFFORT,
    )


@dataclass
class Answer:
    """A streamed answer plus the documents it was grounded in (FR-13a)."""

    documents: list[Document] = field(default_factory=list)
    text: str = ""


class TelecomRAG:
    """Retrieval + generation for one question.

    Retrieval is run explicitly (rather than inside the LCEL chain) so the UI
    can render the Sources section for the exact documents used, and so the
    CLI and Streamlit paths share one code path.
    """

    def __init__(self, k: int = config.TOP_K) -> None:
        self.retriever = build_merged_retriever(k=k)
        self.chain: Runnable = PROMPT | build_llm() | StrOutputParser()

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def stream(self, question: str, answer: Answer | None = None) -> Iterator[str]:
        """Yield answer tokens as they arrive (FR-05).

        Pass an `Answer` to collect the retrieved documents and the full text
        as a side effect of consuming the stream.
        """
        answer = answer if answer is not None else Answer()
        answer.documents = self.retrieve(question)

        payload = {
            "context": format_context(answer.documents),
            "question": question,
            "escalation": config.ESCALATION_TEXT,
        }

        for token in self.chain.stream(payload):
            answer.text += token
            yield token

    def invoke(self, question: str) -> Answer:
        """Non-streaming convenience wrapper."""
        answer = Answer()
        for _ in self.stream(question, answer):
            pass
        return answer
