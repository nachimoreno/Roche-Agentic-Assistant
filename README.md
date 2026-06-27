---
title: Roche Scientist Assistant
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Roche Scientist Assistant

A conversational AI assistant for Roche scientists — one place to ask about protocols, onboarding, equipment, samples, and to report issues, instead of hunting across dozens of internal apps.

**Live demo:** https://nachimoreno-roche-agentic-assistant.hf.space (register an account, then chat)

## Why

Roche scientists have many specialized in-house tools but no single entry point. They often don't know which app to use, don't read email, and work standing up while moving between devices in a lab. The result is wasted time, miscommunication with IT, and slow onboarding.

## What it does

- **Answers questions** grounded in Roche documentation **ingested live from Google Drive** (onboarding, access, instrument booking and calibration, sample stock, ordering consumables, cleaning, decontamination, waste management, campus facilities, and more).
- **Hybrid retrieval** — fuses dense semantic embeddings with BM25 keyword search, so it handles both paraphrased questions and exact tokens (SOP codes, part numbers, app names).
- **Opens ServiceNow incidents** from the conversation — classifies an incident report (broken/inaccessible device, instrument, system, or access issue), confirms the details, then files the ticket and returns its number. Ships with a **mock ServiceNow client by default** so it's fully demoable without credentials; a one-line swap points it at a live instance.
- **Surfaces contradictions** — version-aware retrieval collapses superseded SOP versions (read from document front-matter), and when two *equally-current* documents materially disagree the answer carries a **"sources disagree"** flag (red banner) and logs the conflicting pair for the documentation team.
- **Refuses to guess** — an off-domain guardrail withholds an answer (rather than inventing one) when retrieval confidence is low, and a low-confidence "verify this" badge marks borderline answers. Every answer carries clickable citations to its source.
- **Points scientists to the right internal app** when it can't act directly.
- **Collects feedback** for IT — distinguishing questions, feedback, and incidents, and detecting sentiment (frustrated, confused, happy, angry).
- **Admin analytics dashboard** (`/admin`) for IT and documentation teams — feedback sentiment trends, documentation-gap clusters, an onboarding funnel, contradicting-documents, and most-asked questions; plus a site-wide **announcement banner**.
- **Speaks the user's language** — English, German, Italian, French — translating from source docs on the fly.
- **Hands-free** — voice input (tap-to-speak) and answer read-aloud, plus context-aware follow-up suggestion chips, for scientists working at the bench.
- **Per-user accounts** with hashed passwords and cookie sessions; **chat history is persisted server-side**, so the conversation resumes on any device.
- **Web + desktop UI** — a streaming chat interface (PWA), launchable as its own desktop app window.
- **Knows itself** — what the assistant can and cannot do lives in code (never retrieved), so questions about the assistant answer deterministically.

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

# ServiceNow incident creation is ON by default using a built-in MOCK client
# (no credentials needed). To file against a real instance, set:
#   SERVICENOW_USE_MOCK=false
#   SERVICENOW_INSTANCE=https://devXXXXXX.service-now.com
#   SERVICENOW_USERNAME=admin
#   SERVICENOW_PASSWORD=...
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
- *"The -80 freezer in Lab 4B is alarming."* → confirms the details, then opens a ServiceNow incident and returns its number

## Tests

```bash
pytest             # fast suite: retrieval, chunking, auth, sessions, API, repositories,
                   #             contradiction handling, incident intake + routing
pytest -m live     # adds classification + end-to-end tests against real Groq
```

~340 tests run fully offline against a frozen markdown corpus in `tests/fixtures/docs/`. The live Groq tests are auto-skipped when `GROQ_API_KEY` is unset; the live Google Drive tests are auto-skipped unless `DRIVE_FOLDER_ID` and an existing credentials file are configured; the live ServiceNow tests are auto-skipped unless `SERVICENOW_INSTANCE`/`USERNAME`/`PASSWORD` are set (the mock client is exercised regardless).

## Stack

Framework-light by design — every external dependency sits behind a small `Protocol` so it's a one-class swap later.

| Layer | Default impl | Interface (swap point) |
|---|---|---|
| LLM | Groq (Llama 3.3 70B, free tier) | `LLMClient` in [src/llm.py](src/llm.py) |
| Embeddings | `fastembed` (multilingual, local, ONNX — no PyTorch) | `EmbeddingProvider` in [src/embeddings.py](src/embeddings.py) |
| Vector store | ChromaDB (persistent, local) | `VectorStore` in [src/vector_store.py](src/vector_store.py) |
| Lexical index | In-memory Okapi BM25 (dependency-free) | `LexicalIndex` in [src/lexical_index.py](src/lexical_index.py) |
| Documents | Google Drive (Docs/Sheets/PDF/DOCX/md) | `DocumentSource` in [src/document_source.py](src/document_source.py) |
| Incidents | ServiceNow (mock by default) | `ServiceNowClient` in [src/servicenow_tool.py](src/servicenow_tool.py) + [src/mock_servicenow_client.py](src/mock_servicenow_client.py) |
| HTTP / UI | FastAPI + streaming + static PWA | [src/api.py](src/api.py), [src/static/](src/static/) |
| Auth | Local accounts, Argon2 hashes, cookie sessions | [src/auth.py](src/auth.py) |
| Persistence | SQLite via SQLModel | `DATABASE_URL` env var → swap to PostgreSQL |
| Config | `pydantic-settings` | [src/settings.py](src/settings.py) |

No LangChain, no LangGraph — even the ServiceNow incident flow is a plain confirm-then-file step behind the `LLMClient` seam. A framework gets adopted only when a specific need actually appears.

## Retrieval

Documents are ingested from the configured source, chunked, and embedded into ChromaDB; the same chunks are indexed into BM25. At query time:

1. **Dense** — the query is embedded and matched by cosine similarity (ANN).
2. **Lexical** — the query is scored by Okapi BM25 over the chunk corpus.
3. **Fusion** — the two ranked lists are blended with Reciprocal Rank Fusion, so a chunk surfaced strongly by *either* retriever makes the final top-k.
4. **Version collapse** — fused results are de-duplicated by version: when an explicit `supersedes` chain or a confident same-process/same-title match identifies a superseded document, only the current version survives, so a retired SOP never competes with its replacement.
5. **Guardrail** — if the fused confidence is below a calibrated threshold, the off-domain guardrail returns "not enough information" instead of generating from model priors. Above it, the chunks are passed to the LLM, which answers in the user's language with citations — and flags a **conflict** if two equally-current sources disagree.

Set `RETRIEVAL_MODE=dense` to disable BM25 and use embeddings only.

Version metadata is read from optional document front-matter (`version`, `effective_date`, `status`, `owner`, `supersedes`) — see [architecture/Contradiction_Handling_Design.md](architecture/Contradiction_Handling_Design.md).

## Project structure

```
src/
  settings.py                 # typed Settings (pydantic-settings)
  logging_setup.py            # structured logging + correlation IDs
  llm.py                      # LLMClient + GroqClient
  embeddings.py               # EmbeddingProvider + FastEmbedProvider
  vector_store.py             # VectorStore + ChromaVectorStore
  lexical_index.py            # LexicalIndex + BM25Index
  document_source.py          # DocumentSource + LocalMarkdownSource + CompositeSource (+ front-matter)
  google_drive_source.py      # GoogleDriveSource (Docs/Sheets/PDF/DOCX/md)
  conversation_layer.py       # language + question/feedback/incident + emotion
  retrieval.py                # DocumentStore — chunking, hybrid retrieval, RRF, version collapse
  agent.py                    # RAGAgent — grounded answers, citations, conflict flag, guardrail
  capabilities.py             # CAPABILITIES — the assistant's self-knowledge (can/cannot do)
  incident_intake.py          # IncidentIntake — confirm-then-file gate for incidents
  servicenow_tool.py          # create_servicenow_incident + ServiceNowClient
  mock_servicenow_client.py   # MockServiceNowClient (default — no instance required)
  attribution.py              # resolve which document(s) a piece of feedback concerns
  db.py                       # SQLModel schema (UUIDv7, tenant_id, soft-delete)
  repositories.py             # User / Session / Feedback / QuestionGap + analytics
  demo_seed.py                # synthetic feedback generator for the /admin dashboard
  auth.py                     # password hashing + session helpers
  orchestrator.py             # Assistant — routes question / feedback / incident; wires everything
  api.py                      # FastAPI server (auth, streaming chat, /admin analytics, static UI)
  main.py                     # CLI entry point + composition root (build_assistant)
  pages/admin.html            # admin analytics dashboard
  static/                     # web UI (index.html) + PWA manifest + service worker + icons
tests/                        # ~340 tests (see `pytest`), incl.:
  fixtures/docs/              # offline markdown corpus used by tests
  fixtures/conflict_docs/     # conflicting/superseded docs for contradiction tests
  test_retrieval.py           # semantic + multilingual + hybrid retrieval
  test_contradiction.py       # version collapse + "sources disagree" detection
  test_incident_intake.py / test_incident_routing.py / test_servicenow_tool.py  # incidents
  test_capabilities.py / test_attribution.py / test_front_matter.py / test_question_gap.py
  test_api.py / test_auth.py / test_sessions.py   # HTTP, auth, history
  test_orchestrator.py        # FakeLLMClient + live Groq end-to-end
  ...                         # chunking, lexical index, Drive source, composition, db, llm
scripts/
  seed_synthetic_feedback.py  # seed demo feedback/analytics for /admin
  calibrate_guardrail.py      # tune the off-domain guardrail threshold
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

**Built:** Q&A with hybrid (dense + BM25) RAG; Google Drive ingestion; live document upload to the knowledge base (uploads to Drive + ingests into the index immediately; the attach control greys out automatically when the Drive service account is read-only or unconfigured); **ServiceNow incident creation** (confirm-then-file, mock client by default); **contradiction handling** (version-aware retrieval + "sources disagree" flag); off-domain guardrail, citations, and low-confidence badge; multilingual responses; question/feedback/incident classification and sentiment detection; **admin analytics dashboard** (sentiment trends, documentation-gap clusters, onboarding funnel, contradicting documents, most-asked questions) + announcement banner; voice input/readback and follow-up chips; per-user accounts, persisted server-side sessions + chat history; FastAPI streaming web UI, PWA + desktop launcher; schema-first SQL persistence; CLI demo.

**Planned next:** live-instance ServiceNow validation (the flow is built; remaining work is field mapping + routing against a real instance); enterprise SSO / identity-provider integration; SharePoint source (migration from Drive); public-web search fallback; voice *input* via Whisper; multi-tenancy (assign each user a `tenant_id` at registration, then scope announcements / feedback / sessions by it — the `tenant_id` column already exists on every table, so this is wiring rather than a groups system; a flat tenant-per-user covers per-site banners, with a groups abstraction needed only to target sub-segments); **per-scientist upload siloing** (today an uploaded document joins the shared corpus and becomes answerable for everyone; a future version should scope each upload to its uploader — e.g. tag chunks with the uploader's `user_id`/`tenant_id` and filter retrieval by it — so personal documents stay private until explicitly promoted to the shared knowledge base).

**Out of scope:** Full ServiceNow workflow (routing, assignment, email follow-up), direct integration with every Roche internal app, real confidential Roche data.

## Documents

- [Architecture, Features & Requirements](architecture/Architecture_Features_Requirements.md) — full spec derived from the kickoff meeting
- [Contradiction Handling Design](architecture/Contradiction_Handling_Design.md) — version-aware retrieval + "sources disagree"
- [Feedback Analytics Design](architecture/Feedback_Analytics_Design.md) — the admin dashboard's signals
- [Preprocessing Plan](architecture/Preprocessing_Plan.md) — ingestion/chunking design
- [DEPLOY.md](DEPLOY.md) — Hugging Face Spaces + Neon Postgres deployment (€0/month)

## Success looks like

Scientists save time, onboard faster, hit fewer dead ends, and IT gets real signal on which docs and tools are failing them.
