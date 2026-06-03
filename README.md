# Roche Scientist Assistant

A conversational AI assistant for Roche scientists — one place to ask about protocols, onboarding, equipment, samples, and to report issues, instead of hunting across dozens of internal apps.

## Why

Roche scientists have many specialized in-house tools but no single entry point. They often don't know which app to use, don't read email, and work standing up while moving between devices in a lab. The result is wasted time, miscommunication with IT, and slow onboarding.

## What it does

- **Answers questions** grounded in Roche documentation (onboarding, access requests, cleaning procedures, sample stock, instrument booking, equipment use).
- **Points scientists to the right internal app** when it can't act directly.
- **Collects feedback** for IT — distinguishing questions from feedback and detecting sentiment (frustrated, confused, happy, angry).
- **Speaks the user's language** — English, German, Italian, French — translating from English source docs on the fly.
- **Follows the scientist across devices**, resuming the conversation wherever they log in (chat history is persisted, not in-memory).
- **Creates ServiceNow incidents** from the conversation — *planned, not yet built*.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure: copy your Groq API key into .env
echo 'GROQ_API_KEY=gsk_...' > .env

# Run the REPL
python src/main.py
```

On first run, [data/docs/](data/docs/) is embedded into a local ChromaDB store (`.chroma/`) and SQLite is initialized at `data/app.db`. Subsequent runs reuse both — only modified docs get re-embedded.

Try:
- *"How do I clean the centrifuge?"* → grounded English answer with citations
- *"Wie kann ich den Probenbestand prüfen?"* → grounded German answer
- *"This onboarding doc is really confusing."* → recorded as feedback with sentiment

## Tests

```bash
pytest             # fast suite: retrieval + repository tests
pytest -m live     # adds classification + end-to-end tests against real Groq
```

The live tests are auto-skipped when `GROQ_API_KEY` is unset.

## Stack

Framework-light by design — every external dependency sits behind a small `Protocol` so it's a one-class swap later.

| Layer | Default impl | Interface (swap point) |
|---|---|---|
| LLM | Groq (Llama 3.3 70B, free tier) | `LLMClient` in [src/llm.py](src/llm.py) |
| Embeddings | `fastembed` (multilingual, local, ONNX — no PyTorch) | `EmbeddingProvider` in [src/embeddings.py](src/embeddings.py) |
| Vector store | ChromaDB (persistent, local) | `VectorStore` in [src/vector_store.py](src/vector_store.py) |
| Documents | Local markdown corpus | `DocumentSource` in [src/document_source.py](src/document_source.py) |
| Persistence | SQLite via SQLModel | `DATABASE_URL` env var → swap to PostgreSQL |
| Config | `pydantic-settings` | [src/settings.py](src/settings.py) |
| Schemas | Pydantic + `Literal` types | shared across API + DB |

No LangChain, no LangGraph. Adopted only when a specific need (e.g. multi-step agent state machines for ServiceNow) actually appears.

## Project structure

```
data/
  docs/                       # mock markdown corpus (8 docs)
  app.db                      # SQLite, created on first run
src/
  settings.py                 # typed Settings (pydantic-settings)
  logging_setup.py            # structured logging + correlation IDs
  llm.py                      # LLMClient + GroqClient
  embeddings.py               # EmbeddingProvider + FastEmbedProvider
  vector_store.py             # VectorStore + ChromaVectorStore
  document_source.py          # DocumentSource + LocalMarkdownSource
  conversation_layer.py       # language + question/feedback + emotion
  retrieval.py                # DocumentStore (composes the three above)
  agent.py                    # RAGAgent — grounded answers with citations
  db.py                       # SQLModel schema (UUIDv7, tenant_id, soft-delete)
  repositories.py             # FeedbackRepository + SessionRepository
  orchestrator.py             # Assistant — wires everything together
  main.py                     # CLI entry point
tests/
  test_conversation_layer.py  # live: classification matrix
  test_retrieval.py           # fast: semantic + multilingual retrieval
  test_repositories.py        # fast: DB schema, soft-delete, tenant filter
  test_orchestrator.py        # fast (FakeLLMClient) + live (Groq) end-to-end
```

## Production-readiness hedges

The MVP is built so promotion to production is incremental, not a rewrite:

- **`DATABASE_URL` env var** — sqlite → postgres is a one-line change.
- **UUIDv7 primary keys** — time-ordered, globally unique, distribution-friendly.
- **Nullable `tenant_id`** on every table — adding multi-tenancy later doesn't require a backfill.
- **Soft-delete (`deleted_at`)** on every persistent row — compliance-friendly, repository filters it by default.
- **Repository pattern** — orchestrator never touches SQLAlchemy; ORM swaps are contained.
- **Provider interfaces** — Groq/Chroma/sentence-transformers/markdown are all swap points.
- **Typed Settings + structured logging** — env-driven config, JSON logs with per-turn correlation IDs.

## Scope

**Built:** Q&A with RAG, multilingual responses, question/feedback classification, sentiment detection, persisted sessions + chat history, mock document corpus, schema-first SQL persistence, CLI demo.

**Planned next:** ServiceNow incident creation, real Google Drive ingestion, web/HTTP UI, macro-analytics dashboards, public-web search fallback.

**Out of scope:** Full ServiceNow workflow (routing, assignment, email follow-up), direct integration with every Roche internal app, real confidential Roche data.

## Documents

- [Architecture, Features & Requirements](architecture/Architecture_Features_Requirements.md) — full spec derived from the kickoff meeting
- [Meeting summary 1.pdf](.misc/Meeting%20summary%201.pdf) — original meeting notes

## Success looks like

Scientists save time, onboard faster, hit fewer dead ends, and IT gets real signal on which docs and tools are failing them.
