# NovaCell Support Assistant — RAG Telecom Customer-Care Chatbot

A Retrieval-Augmented Generation chatbot that answers Tier-1 telecom support
questions (connectivity, data, billing, roaming, SIM/eSIM, voice, account)
grounded **only** in NovaCell's own knowledge: a published FAQ, a database of
resolved support tickets, and the official PDF user guide.

Built to [prd/PRD.md](prd/PRD.md).

---

## Architecture

```
User question
     │
     ▼
Merged retriever (three collections queried in parallel)
  ├── ChromaDB · faq       top-3 FAQ entries
  ├── ChromaDB · tickets   top-3 resolved ticket resolutions
  └── ChromaDB · guides    top-3 PDF guide chunks
     │
     ▼  9 source-labelled context documents
ChatPromptTemplate  (grounding rules + context injection)
     │
     ▼
Qwen3.6-27B on Groq   (temperature 0, reasoning_effort "none")
     │
     ▼
StrOutputParser → streamed to the UI
```

| Layer | Choice |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, **local** — no embedding API cost |
| Vector store | ChromaDB, persisted to `chroma_store/` |
| LLM | `qwen/qwen3.6-27b` via the Groq API |
| Framework | LangChain (LCEL) |
| UI | Streamlit + a CLI REPL |

---

## Setup

Everything installs into a **project-local `.venv`** — never a global interpreter.

```bash
# 1. Create the environment (Python 3.11+; tested on 3.13)
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

#    …or with plain pip:
#    python -m venv .venv
#    .venv/Scripts/python -m pip install -r requirements.txt

# 2. Add your Groq API key (free at https://console.groq.com/keys)
cp .env.example .env      # then edit .env and set GROQ_API_KEY=...

# 3. Build the vector store (downloads the ~90 MB embedding model once)
.venv/Scripts/python ingest_all.py
```

## Run

```bash
# Browser UI
.venv/Scripts/streamlit run app.py

# Terminal REPL
.venv/Scripts/python main.py
```

On Linux/macOS use `.venv/bin/...` instead of `.venv/Scripts/...`.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | All paths, model names, retrieval sizes; loads `.env` |
| `embeddings.py` | The single, cached local embedding model |
| `vectorstore.py` | Chroma persistence + idempotent collection rebuilds |
| `ingest_faq.py` | `data/faq.csv` → collection `faq` (1 row = 1 document) |
| `ingest_tickets.py` | `data/tickets.db` → collection `tickets` (1 resolved ticket = 1 document) |
| `ingest_guides.py` | `data/telecom_guide.pdf` → collection `guides` (600-char chunks, 100-char overlap) |
| `ingest_all.py` | Runs all three ingest scripts |
| `retriever.py` | Collection registry + parallel merged retriever + context formatting |
| `rag_chain.py` | Prompt, Groq LLM, LCEL chain, streaming |
| `app.py` | Streamlit chat UI |
| `main.py` | CLI REPL |
| `interaction_log.py` | Appends Q/A and 👍/👎 to `logs/interactions.jsonl` |
| `sample_questions.py` | The sidebar's one-click sample questions |

---

## Keeping the knowledge base current

Support ops can update knowledge without an engineering release (US-06, US-07):

```bash
# edited data/faq.csv?
.venv/Scripts/python ingest_faq.py

# seeded new resolved tickets into data/tickets.db?
.venv/Scripts/python ingest_tickets.py

# replaced data/telecom_guide.pdf?
.venv/Scripts/python ingest_guides.py
```

Each script rebuilds its own collection from the source, so re-runs are
idempotent: edits, additions and deletions all take effect, with no duplicates
and no stale documents left behind. The other collections are untouched.

## Adding a new knowledge source

1. Write `ingest_<source>.py` that builds `Document`s with `source_type` and
   `identifier` metadata and calls `vectorstore.rebuild_collection(...)`.
2. Append one `CollectionSpec` to `COLLECTIONS` in [retriever.py](retriever.py).

The retriever, prompt context, Sources section and knowledge-base panel all pick
it up automatically.

---

## Grounding & escalation

- The system prompt forbids answering from the model's own training data; if the
  retrieved context doesn't cover the question, the bot says so and points the
  customer to **611 / the MyTelecom app**.
- The bot has no CRM or billing access, so personal-account questions ("what's
  my balance?") are explicitly handed off rather than guessed at.
- Every answer ships with an expandable **Sources** list naming each retrieved
  document (`FAQ | FAQ-12`, `TICKETS | TK-004`, `GUIDES | telecom_guide.pdf p.3`).

## Interaction log

`logs/interactions.jsonl` — one JSON object per line:

```json
{"event":"answer","interaction_id":"…","question":"…","answer":"…","sources":["FAQ | FAQ-2"],"latency_seconds":1.8,"timestamp":"…"}
{"event":"feedback","interaction_id":"…","rating":"down","timestamp":"…"}
```

Join `feedback` to `answer` on `interaction_id` to see which answers, and which
retrieved sources, are landing.

---

## Configuration

All optional, via `.env` (see `.env.example`):

| Variable | Default |
|---|---|
| `GROQ_API_KEY` | *(required)* |
| `GROQ_MODEL` | `qwen/qwen3.6-27b` |
| `LLM_TEMPERATURE` | `0` |
| `REASONING_EFFORT` | `none` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `TOP_K` | `3` (per collection) |
