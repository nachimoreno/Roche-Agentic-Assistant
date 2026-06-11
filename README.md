# Roche Scientist Assistant

A conversational AI assistant for Roche scientists — one place to ask about protocols, onboarding, equipment, samples, and to report issues, instead of hunting across dozens of internal apps.

## Why

Roche scientists have many specialized in-house tools but no single entry point. They often don't know which app to use, don't read email, and work standing up while moving between devices in a lab. The result is wasted time, miscommunication with IT, and slow onboarding.

## What it does

- **Answers questions** grounded in Roche documentation **ingested live from Google Drive** (onboarding, access, instrument booking and calibration, sample stock, ordering consumables, cleaning, decontamination, waste management, campus facilities, and more).
- **Hybrid retrieval** — fuses dense semantic embeddings with BM25 keyword search, so it handles both paraphrased questions and exact tokens (SOP codes, part numbers, app names).
- **Points scientists to the right internal app** when it can't act directly.
- **Collects feedback** for IT — distinguishing questions from feedback and detecting sentiment (frustrated, confused, happy, angry).
- **Speaks the user's language** — English, German, Italian, French — translating from source docs on the fly.
- **Per-user accounts** with hashed passwords and cookie sessions; **chat history is persisted server-side**, so the conversation resumes on any device.
- **Web + desktop UI** — a streaming chat interface (PWA), launchable as its own desktop app window.
- **Creates ServiceNow incidents** from the conversation — *planned, not yet built*.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Configure `.env` (production ingests documents from Google Drive):

```bash
GROQ_API_KEY=gsk_...
# Google Drive source (default). Service account needs read access to the folder.
GOOGLE_SERVICE_ACCOUNT_JSON=secrets/<service-account-key>.json
DRIVE_FOLDER_ID=<folder id from the Drive URL>
```

> No Drive yet? Set `DOCUMENT_SOURCE=local` to ingest a local markdown corpus from `docs_path` instead (defaults to the bundled `tests/fixtures/docs/`). `DOCUMENT_SOURCE=all` ingests from both.

Run the web/desktop server:

```bash
python src/api.py            # or: uvicorn src.api:app --reload
# then open http://127.0.0.1:8000/  (register an account, then chat)
```

On Windows, double-click **`Roche Scientific AI.bat`** to start the server (if needed) and open the chat in its own app window.

Prefer a terminal? The CLI REPL still works:

```bash
python src/main.py
```

On first run, the configured documents are embedded into a local ChromaDB store (`.chroma/`), a BM25 index is built in memory, and SQLite is initialized at `data/app.db`. Subsequent runs reuse both — only changed docs are re-embedded. On startup the server logs whether the Google Drive connection succeeded (or why it failed) to the console.

Try:
- *"How do I clean the centrifuge?"* → grounded English answer with citations
- *"Wie kann ich den Probenbestand prüfen?"* → grounded German answer
- *"This onboarding doc is really confusing."* → recorded as feedback with sentiment

## Tests

```bash
pytest             # fast suite: retrieval, chunking, auth, sessions, API, repositories
pytest -m live     # adds classification + end-to-end tests against real Groq
```

The live Groq tests are auto-skipped when `GROQ_API_KEY` is unset; the live Google Drive tests are auto-skipped unless `DRIVE_FOLDER_ID` and an existing credentials file are configured. Tests use a frozen markdown corpus in `tests/fixtures/docs/` so they run fully offline.

## Stack

Framework-light by design — every external dependency sits behind a small `Protocol` so it's a one-class swap later.

| Layer | Default impl | Interface (swap point) |
|---|---|---|
| LLM | Groq (Llama 3.3 70B, free tier) | `LLMClient` in [src/llm.py](src/llm.py) |
| Embeddings | `fastembed` (multilingual, local, ONNX — no PyTorch) | `EmbeddingProvider` in [src/embeddings.py](src/embeddings.py) |
| Vector store | ChromaDB (persistent, local) | `VectorStore` in [src/vector_store.py](src/vector_store.py) |
| Lexical index | In-memory Okapi BM25 (dependency-free) | `LexicalIndex` in [src/lexical_index.py](src/lexical_index.py) |
| Documents | Google Drive (Docs/Sheets/PDF/DOCX/md) | `DocumentSource` in [src/document_source.py](src/document_source.py) |
| HTTP / UI | FastAPI + streaming + static PWA | [src/api.py](src/api.py), [src/static/](src/static/) |
| Auth | Local accounts, Argon2 hashes, cookie sessions | [src/auth.py](src/auth.py) |
| Persistence | SQLite via SQLModel | `DATABASE_URL` env var → swap to PostgreSQL |
| Config | `pydantic-settings` | [src/settings.py](src/settings.py) |

No LangChain, no LangGraph. Adopted only when a specific need (e.g. multi-step agent state machines for ServiceNow) actually appears.

## Retrieval

Documents are ingested from the configured source, chunked, and embedded into ChromaDB; the same chunks are indexed into BM25. At query time:

1. **Dense** — the query is embedded and matched by cosine similarity (ANN).
2. **Lexical** — the query is scored by Okapi BM25 over the chunk corpus.
3. **Fusion** — the two ranked lists are blended with Reciprocal Rank Fusion, so a chunk surfaced strongly by *either* retriever makes the final top-k.

Set `RETRIEVAL_MODE=dense` to disable BM25 and use embeddings only.

## Project structure

```
src/
  settings.py                 # typed Settings (pydantic-settings)
  logging_setup.py            # structured logging + correlation IDs
  llm.py                      # LLMClient + GroqClient
  embeddings.py               # EmbeddingProvider + FastEmbedProvider
  vector_store.py             # VectorStore + ChromaVectorStore
  lexical_index.py            # LexicalIndex + BM25Index
  document_source.py          # DocumentSource + LocalMarkdownSource + CompositeSource
  google_drive_source.py      # GoogleDriveSource (Docs/Sheets/PDF/DOCX/md)
  conversation_layer.py       # language + question/feedback + emotion
  retrieval.py                # DocumentStore — chunking, hybrid retrieval, RRF
  agent.py                    # RAGAgent — grounded answers with citations
  db.py                       # SQLModel schema (UUIDv7, tenant_id, soft-delete)
  repositories.py             # User / Session / Feedback repositories
  auth.py                     # password hashing + session helpers
  orchestrator.py             # Assistant — wires everything together
  api.py                      # FastAPI server (auth, streaming chat, static UI)
  main.py                     # CLI entry point + composition root (build_assistant)
  static/                     # web UI (index.html) + PWA manifest + icons
tests/
  fixtures/docs/              # offline markdown corpus used by tests
  test_retrieval.py           # semantic + multilingual + hybrid retrieval
  test_chunking.py            # chunker (handles DOCX/PDF text without markdown)
  test_lexical_index.py       # BM25 + RRF fusion
  test_google_drive_source.py # Drive source (mocked + live)
  test_composition.py         # DocumentSource selection
  test_drive_status.py        # startup Drive connectivity reporting
  test_api.py / test_auth.py / test_sessions.py   # HTTP, auth, history
  test_orchestrator.py        # FakeLLMClient + live Groq end-to-end
  ...                         # conversation layer, repositories, db, llm
Roche Scientific AI.bat       # Windows desktop launcher
```

## Production-readiness hedges

The MVP is built so promotion to production is incremental, not a rewrite:

- **`DATABASE_URL` env var** — sqlite → postgres is a one-line change.
- **UUIDv7 primary keys** — time-ordered, globally unique, distribution-friendly.
- **Nullable `tenant_id`** on every table — adding multi-tenancy later doesn't require a backfill.
- **Soft-delete (`deleted_at`)** on every persistent row — compliance-friendly, repository filters it by default.
- **Repository pattern** — orchestrator never touches SQLAlchemy; ORM swaps are contained.
- **Provider interfaces** — LLM / vector store / embeddings / lexical index / document source are all swap points.
- **Document source seam** — Google Drive today; a `SharePointSource` with the same interface is a drop-in for the planned migration.
- **Typed Settings + structured logging** — env-driven config, JSON logs with per-turn correlation IDs.

## Scope

**Built:** Q&A with hybrid (dense + BM25) RAG, Google Drive ingestion, multilingual responses, question/feedback classification, sentiment detection, per-user accounts, persisted server-side sessions + chat history, FastAPI streaming web UI + desktop launcher, schema-first SQL persistence, CLI demo.

**Planned next:** ServiceNow incident creation, SharePoint source (migration from Drive), macro-analytics dashboards, public-web search fallback.

**Out of scope:** Full ServiceNow workflow (routing, assignment, email follow-up), direct integration with every Roche internal app, real confidential Roche data.

## Documents

- [Architecture, Features & Requirements](architecture/Architecture_Features_Requirements.md) — full spec derived from the kickoff meeting
- [Meeting summary 1.pdf](.misc/Meeting%20summary%201.pdf) — original meeting notes

## Success looks like

Scientists save time, onboard faster, hit fewer dead ends, and IT gets real signal on which docs and tools are failing them.
