# NovaCell Support Assistant

**A retrieval-augmented support assistant that answers Tier-1 telecom questions from three disconnected knowledge sources — and refuses to answer anything they don't cover.**

---

## One-line versions

**Short — for a project card or tile**

A RAG chatbot that resolves Tier-1 telecom support questions from an FAQ, a resolved-ticket database and a PDF user guide — with every answer cited and nothing invented.

**Medium — for a project intro**

NovaCell Support Assistant is a grounded question-answering system for mobile customer care. It fans a customer's question out across three separately indexed knowledge sources in parallel, assembles nine source-labelled passages into the prompt, and answers only from what it retrieved. When the sources don't cover the question, it says so and hands off to a human instead of guessing.

---

## The problem

NovaCell is a mobile operator with roughly four million subscribers. Support is one of its largest costs and one of its lowest satisfaction scores, and most of the volume is Tier-1: slow data, a confusing charge, a SIM that won't activate, roaming setup before a trip. These are questions whose answers already exist inside the company.

The answers just don't exist in one place. They're split across a public FAQ, a database of past resolved tickets, and official PDF user guides — three surfaces a customer can't search together and an agent has to reconcile by hand. So customers wait on hold for things an FAQ could solve, agents burn out on repetitive tickets, and the same question gets three different answers depending on who picks it up.

The constraint that shaped the whole build: in telecom support, a confidently wrong answer about a price, a policy or a timeframe is worse than no answer at all. Deflection was never the hard part. Trustworthy deflection was.

---

## What it does

Ask it a plain-language question — *"Why is my mobile internet so slow?"*, *"How do I activate an eSIM?"*, *"I was charged for roaming I didn't use"* — and it returns a short, non-technical answer with numbered steps where the source material gives a procedure, streamed token by token.

Every answer ships with an expandable **Sources** list naming each document it drew on: `FAQ | FAQ-12`, `TICKETS | TK-004`, `GUIDES | telecom_guide.pdf p.3`. Nothing is asserted that can't be traced back to a document a support operator owns.

It covers eight support categories — connectivity, data, roaming, SIM/eSIM, billing, voice, device and account — and ships as both a browser chat UI and a terminal REPL running over one shared retrieval path.

---

## How it works

**1 — Fan out.** The question is embedded locally and dispatched to three ChromaDB collections *concurrently*, not as one blended index. FAQ, tickets and guides each return their own top-3.

**2 — Merge and label.** The nine results are merged in a fixed source order and tagged with their origin and identifier, so the model can attribute a claim and never silently blend an un-retrieved fact into a retrieved one.

**3 — Ground.** A system prompt injects the labelled context block under rules that override everything else: answer only from this context; never use training knowledge; never state a number that isn't here; if the context doesn't cover the question, say so in one sentence and hand off.

**4 — Answer.** Qwen3.6-27B on Groq at temperature 0 with reasoning disabled — deterministic and fast, because a support answer should be the same answer every time it's asked.

Retrieval runs outside the generation chain rather than inside it, so the exact documents behind an answer are available to the UI for citation and to the log for later analysis.

---

## The refusal is the product

The most important thing this system does is decline. Three real transcripts, unedited:

> **"What is my current account balance?"**
>
> I cannot see your individual account data, including your current balance. Please call 611 from your NovaCell mobile or use the MyTelecom app to check your specific account details.

> **"Who won the 2024 cricket world cup?"**
>
> I cannot answer questions about sports or general knowledge. I am NovaCell's telecom customer-care assistant and can only help with mobile connectivity, data, roaming, SIM/eSIM, billing, voice calls, devices, and account/app issues.

> **"Airtel and Vodafone have better prices."**
>
> I cannot see individual account data or compare our prices with other providers. Please call 611 from your NovaCell mobile or use the MyTelecom app to speak with a representative about your specific plan options.

Each of those still retrieved nine documents. The model had context in front of it and correctly judged that none of it answered the question. That behaviour — holding the line between *retrieved* and *known* — is what separates a support assistant from a liability.

The v1 boundary is deliberate: no CRM, no billing, no authentication. A question about *your* bill is routed to a human. A question about *how billing works* is answered from the guide.

---

## Decisions worth explaining

**Three collections, not one index.** Blending an FAQ answer, a ticket resolution and a PDF chunk into a single index lets one verbose source crowd out the others on a close match. Querying them separately guarantees every answer sees the canonical FAQ line, the way the issue was actually fixed in practice, and the official procedure — the three perspectives an agent would consult. Running them concurrently means the cost is the slowest collection, not the sum of all three.

**Embeddings run locally.** `all-MiniLM-L6-v2` on the machine, not an embedding API. Re-indexing the entire knowledge base costs nothing, so support ops can re-run it as often as they like — and no customer question is sent to a second vendor.

**Ingestion is idempotent by construction.** Each script drops and rebuilds its own collection from source. Edits, additions and deletions all take effect with no duplicates and no stale documents left behind, and the other two collections are untouched. Support ops update the FAQ file and re-run one script — no engineering release, no migration.

**Only resolved tickets are indexed.** An open ticket has no verified resolution to ground an answer in; indexing one would let an unconfirmed guess become a cited source.

**Adding a fourth source takes two changes.** Write an ingest script, register one collection entry. The retriever, prompt context, Sources list and admin panel all pick it up automatically.

---

## What's in the index

| Collection | Source | Documents |
|---|---|---|
| `faq` | Published FAQ (CSV) | 25 — one per entry |
| `tickets` | Resolved support tickets (SQLite) | 19 of 20 — resolved only |
| `guides` | Official user guide (9-page PDF) | 37 chunks — 600 chars, 100 overlap |
| | **Total** | **81 documents · 9 retrieved per question** |

---

## Stack

| Layer | Choice |
|---|---|
| LLM | `qwen/qwen3.6-27b` via Groq · temperature 0 · reasoning off |
| Orchestration | LangChain (LCEL), parallel retrieval branches |
| Vector store | ChromaDB, persisted to disk — no re-ingest on start |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local |
| Ingestion | CSV, SQLite and PDF loaders — one script per source |
| Interfaces | Streamlit chat UI + CLI REPL |
| Config | `.env`-driven; no credentials in code |

---

## How it's measured

Every interaction is appended to a JSONL log: question, answer, the exact source identifiers retrieved, end-to-end latency, and a thumbs-up/down rating joined back on interaction ID. That makes two questions answerable from data rather than impression — *which answers are landing*, and *which retrieved sources are behind the ones that aren't*. Bad retrieval and bad generation look identical from the outside; the log tells them apart.

Across 11 logged runs on a local machine, end-to-end responses landed between 0.4 and 1.6 seconds, median 0.8 — comfortably inside the 10-second target, though at that sample size it's a smoke test rather than a benchmark.

---

## What v2 needs

Cross-encoder re-ranking before generation. Hybrid dense + BM25 search, since exact codes and plan names are where pure vector search is weakest. A RAGAS-style evaluation harness, so retrieval changes can be judged instead of eyeballed. Multi-turn memory feeding retrieval. Authenticated CRM and billing lookups, to convert today's escalations into answers. Multilingual support.

---

## Credits

Built to a self-authored PRD covering scope, functional and non-functional requirements, user stories, success metrics and out-of-scope boundaries — the product spec and the implementation are the same piece of work.
