# Feedback Pipeline & Analytics Dashboard — Design

> Status: **approved design, not yet built**. This document is the single
> source of truth for the next build phase. It is self-contained: it does not
> depend on chat context and can be read/implemented directly.
>
> Last updated: 2026-06-11.

## 1. Goal

Add a **feedback pipeline** with an **admin analytics dashboard** so IT gets
real signal on which documents, processes, and departments are failing Roche
scientists — and how they feel about it.

This builds on the ~60% of the pipeline that already exists (classification +
storage + routing). It adds an **explicit rating signal**, **doc/process/
department attribution**, **aggregation**, and an **admin-only dashboard**,
all inside the existing FastAPI app (no new infra).

## 2. What already exists (do not rebuild)

| Capability | Where |
|---|---|
| Classify message `question` vs `feedback`, detect 1-of-18 emotions | `src/conversation_layer.py` (`ConversationLayer.analyze`) |
| `FeedbackEntry` table (language, emotion, message, `session_id`, `turn_id`, `tenant_id`, soft-delete) | `src/db.py` |
| Route feedback → store + ack in user's language | `src/orchestrator.py` (`handle` / `handle_stream`) |
| Query feedback (`list(language, emotion, since, tenant_id)`) | `src/repositories.py` (`FeedbackRepository`) |
| Answers carry `Citation(source, section)` | `src/agent.py` |

> Note: `FeedbackEntry.turn_id` already exists in the schema but is **never
> populated today** — wiring it up is part of this work.

## 3. Decisions (locked)

1. **Signal:** explicit ratings (thumbs up/down + optional comment) **and** keep
   the existing NLP emotion classification. Two streams, one table,
   distinguished by a `source` column.
2. **Access:** add an **admin `role`** to users; dashboard is admin-gated.
3. **Delivery:** **built into the FastAPI app** — JSON analytics endpoints + a
   static admin page (Chart.js). No external BI tool, no new service.
4. **Analytics focus:** sentiment/volume trends **+** doc-quality gaps **+**
   process/department hotspots.
5. **Rating granularity:** **thumbs up/down + optional free-text comment**
   (comment box appears on a down-vote).
6. **Process/department labels:** **front-matter in each doc**
   (`process:` / `department:`). Non-markdown sources are handled upstream by
   preprocessing the corpus into markdown + front-matter on a shared drive
   (see §6); no in-app fallback parser for now.
7. **Multi-doc blame:** **split** across all cited docs (weighted `1/N`), not
   concentrated on the top citation (see §5.4, §8).

## 4. Architecture overview

```
answer turn ──┐  (persist cited sources on the assistant turn) ← LINCHPIN
              ├─────────────► TurnCitation(turn_id, source, section, process, department, rank)
user rates ───┘                         │
   POST /turns/{id}/feedback            │
        │                               │
        ▼                               ▼
   FeedbackEntry ───────────► FeedbackAttribution (one row per implicated doc, weighted):
   (rating, source,             1. via turn_id → TurnCitation, split 1/N ("citation")
    emotion?, comment?,         2. else embed comment → nearest Chroma chunk ("embedding")
    attribution_method)         3. else none
        │
        ▼
   FeedbackRepository aggregations (plain SQL, on read)
        │
        ▼
   /api/analytics/*  (admin-gated)  ──►  static/admin.html + Chart.js
```

The **linchpin** is persisting which documents each answer used. Without it,
a rating cannot be traced to a process. Everything else hangs off that.

## 5. Data model changes

All additive and nullable-friendly, consistent with the existing UUIDv7 /
`tenant_id` / soft-delete conventions in `src/db.py`.

### 5.1 `User` — add role
```
role: Optional[str] = Field(default="user", index=True)   # "user" | "admin"
```
- `register` must **never** accept a `role` field (privilege-escalation guard).
- First admin is seeded out-of-band (see §9).

### 5.2 New table `TurnCitation`
Records the docs an assistant turn cited. One row per cited source.
```
id: UUID (uuid7, pk)
turn_id: UUID (fk turn.id, index)
tenant_id: Optional[UUID] (index)
source: str                 # Citation.source (filename/id)
section: str                # Citation.section
process: Optional[str] (index)      # resolved from doc front-matter at write time
department: Optional[str] (index)   # resolved from doc front-matter at write time
rank: int                   # citation order (0 = top-ranked); kept for traceability
created_at, deleted_at      # standard
```

### 5.3 `FeedbackEntry` — extend
```
rating: Optional[str] (index)        # "up" | "down" ; null for pure-NLP feedback
source: str (index)                  # "explicit" | "nlp"
comment: Optional[str]               # free text on a down-vote (PII — see §10)
emotion: Optional[str] (index)       # NOW NULLABLE (explicit thumb may have none)
turn_id: Optional[UUID]              # NOW POPULATED for explicit ratings
attribution_method: Optional[str] (index)  # "citation" | "embedding" | "none"
```
- `attribution_method` is the per-feedback summary of *how* it was attributed.
  The actual process/department blame is **split across docs** and lives in the
  `FeedbackAttribution` table (§5.4) — a single feedback can implicate several
  processes, so it cannot be one denormalized column.
- `message` (existing) keeps the volunteered-feedback text; `comment` holds the
  explicit-rating note. Keep them distinct so analytics can tell them apart.

### 5.4 New table `FeedbackAttribution` (weighted, split blame)
One feedback fans out to one row per implicated document. Decision: **blame is
split** — over enough observations, genuinely problematic docs/processes rise
above the noise instead of a single top citation absorbing all blame.
```
id: UUID (uuid7, pk)
feedback_id: UUID (fk feedbackentry.id, index)
tenant_id: Optional[UUID] (index)
source: str (index)                  # the implicated document (Citation.source)
section: Optional[str]
process: Optional[str] (index)       # from doc front-matter
department: Optional[str] (index)    # from doc front-matter
weight: float                        # citation split: 1/N per cited doc; embedding: 1.0
method: str (index)                  # "citation" | "embedding"
distance: Optional[float]            # embedding distance when method="embedding"
created_at, deleted_at               # standard
```
- **Citation path:** one row per cited doc on the turn, each `weight = 1/N`
  (N = number of citations). Aggregating by `process`/`department` sums the
  weights, so a process cited more heavily in a bad answer earns more blame.
- **Embedding path:** a single nearest-doc row, `weight = 1.0`.
- Attribution at **document** granularity (`source`) means hotspots can be
  rolled up by doc, process, or department — the doc-level view is what surfaces
  "really problematic documents."

### 5.5 Migration note (important)
`db._add_missing_columns` only **adds nullable columns** on SQLite — it does
**not** relax an existing `NOT NULL`. Making `FeedbackEntry.emotion` nullable
is therefore a no-op on existing SQLite dev DBs (loose enforcement, existing
rows already have values) but a **real migration on Postgres**. New columns and
the new `TurnCitation` / `FeedbackAttribution` tables are handled by
`create_all` + the additive helper. Treat Postgres promotion as needing a
proper migration step here.

## 6. Doc front-matter & ingestion

Each document declares its owning process and department:
```yaml
---
process: instrument-booking
department: lab-operations
---
```
- **Markdown** (local + Drive `.md`): parse and strip the YAML block; keep
  `process`/`department` as chunk metadata into Chroma **and** a
  `source → {process, department}` lookup used at citation-write time.
- **Strip front-matter from the embedded body** so it does not pollute
  retrieval; it is metadata only.
- **One-time re-embed:** adding front-matter changes each doc's content hash,
  so the "only changed docs re-embed" optimization will re-embed the whole
  corpus once. Expected and acceptable.
- **Non-markdown Drive files** (Google Docs/Sheets/PDF/DOCX) have no YAML block.
  **Parked for now (OD-2):** the corpus is being moved to a shared drive with
  edit permissions so documents can be preprocessed into a markdown-with-
  front-matter form ahead of ingestion. Until that lands, non-markdown sources
  resolve `process=None` and rely on the embedding fallback — no fallback parser
  is being built yet.

## 7. Capture flow

### 7.1 Persist citations (Phase 1)
When an assistant turn is persisted (`orchestrator.handle` and the non-aborted
path of `handle_stream`), also write one `TurnCitation` per `Citation`,
resolving `process`/`department` from the doc lookup, with `rank` from citation
order (`rank=0` = primary).

- **Aborted answers carry no citations.** On Stop/disconnect the server drops
  the answer on `GeneratorExit` and the client re-persists partial text via
  `POST /api/sessions/{id}/messages` with no citations. Such turns get no
  `TurnCitation`; feedback on them falls back to embedding. Citation coverage
  is < 100% by design.

### 7.2 Surface `turn_id` to the client (Phase 1, required)
The UI must know the assistant turn id to rate it. Add `turn_id` to:
- `StreamDone` event → `done` SSE frame in `api.chat_stream`.
- `ChatResponse` (non-stream `chat`).
- `MessageOut` (so historical messages from `GET /messages` are ratable).

### 7.3 Rating endpoint (Phase 1)
```
POST /api/sessions/{session_id}/turns/{turn_id}/feedback
body: { rating: "up" | "down", comment?: string }
```
- Validates the turn is an **assistant** turn in a session the caller owns.
- **Idempotent / upsert:** one rating per `(turn_id, user)`. Re-rating updates
  the existing row (prevents double-click inflating the negative rate).
- If `comment` present, run it through `ConversationLayer` for `emotion`
  (explicit down-votes still get sentiment).
- On write, resolve and denormalize `process`/`department` +
  `attribution_method` (citation → embedding → none).

### 7.4 Thumbs UI (Phase 1)
Thumbs up/down under each assistant message in `src/static/index.html`;
down-vote reveals the optional comment box; POST to the endpoint above.

## 8. Process/department attribution

Priority order, recorded per-feedback in `FeedbackEntry.attribution_method`,
with the actual weighted blame in `FeedbackAttribution`:
1. **`citation`** (deterministic) — `turn_id → TurnCitation`: write one
   `FeedbackAttribution` row per cited doc, each `weight = 1/N`. **Blame is
   split** across all cited docs (and thus their processes/departments), not
   concentrated on the top citation. Rationale: over enough observations the
   consistently-implicated docs accumulate weight and rise above noise, while a
   doc that happened to be cited once in an otherwise-bad answer stays low.
2. **`embedding`** (inferred) — orphan/volunteered feedback with no usable
   citation: embed the text via `fastembed`, nearest-neighbour against the
   existing Chroma chunks, write a single attribution row `weight = 1.0` with
   the nearest chunk's process; store the `distance`.
3. **`none`** — no signal; the feedback is counted in totals but produces no
   attribution rows, so it is excluded from hotspot blame.

The dashboard must visually separate **`citation` (known)** from
**`embedding` (inferred)** so guessed hotspots are never shown as fact.

## 9. Access control

- `require_admin` FastAPI dependency reads `User.role`; non-admins get 404
  (not 403) to avoid revealing endpoint existence, matching the existing
  `_require_owned` convention.
- **Seed first admin** via an env allowlist (`ADMIN_EMAILS`) that promotes on
  login, or a one-off script. `register` never accepts `role`.
- **Tenant scoping:** all analytics queries filter by the admin's `tenant_id`
  when set, so a future tenant A cannot read tenant B's feedback.

## 10. Analytics layer (plain SQL, on read)

At MVP scale, aggregate on read — no materialized rollups, no warehouse. New
methods on `FeedbackRepository`, new admin-gated endpoints:

- `GET /api/analytics/summary?since=` → total volume, negative-rate,
  emotion distribution, language split.
- `GET /api/analytics/hotspots?dimension=document|process|department&since=` →
  top-N by **summed attribution weight** of negative feedback (join
  `FeedbackAttribution`, `GROUP BY` the dimension, `SUM(weight)`), with a
  `citation` vs `embedding` breakdown so inferred blame is distinguishable.
- `GET /api/analytics/trend?bucket=day&since=` → time series of volume +
  negative-rate.

All scoped by tenant (§9) and excluding soft-deleted rows by default.

## 11. Dashboard UI

`src/static/admin.html` + Chart.js, fed by the §10 endpoints, behind
`require_admin`. **Vendor Chart.js** (do not rely on a CDN) — the Roche
environment may block external CDNs.

## 12. Governance / PII

- `comment` and `message` are user free text in a regulated context — may
  contain names, sample IDs, or complaints about colleagues. Stored
  unredacted, admin-visible.
- Soft-delete already supports removal. Add a **retention note** and consider
  redaction/aggregation-only views before any wider rollout. Flagged, not
  blocking the MVP.

## 13. Testing

Follow the existing pattern — offline fixtures + `FakeLLMClient`, no live Groq
needed for the analytics math:
- Repository aggregation unit tests (counts, negative-rate, grouping) on a
  seeded in-memory DB.
- Attribution tests: citation split (N cited docs each get `weight = 1/N`;
  per-process sums are correct), embedding-fallback path (single `weight = 1.0`
  row), none path (no attribution rows).
- Endpoint tests: admin gating (admin vs non-admin), tenant scoping, rating
  upsert/idempotency, `turn_id` surfaced in chat/stream/messages.
- Front-matter parser tests (markdown strip + metadata extraction).

## 14. Phasing

1. **Capture** — `TurnCitation` + persist citations; surface `turn_id`;
   `rating`/`comment`/`source` columns; rating endpoint; thumbs UI;
   `User.role`. **✅ DONE** — `db.py` (schema + `TurnCitation`), `repositories.py`
   (`upsert_rating`, `add_citations`, `get_turn`, `set_role`), `orchestrator.py`
   (`record_rating`, citations persisted, `turn_id` on `Response`/`StreamDone`),
   `api.py` (`POST /sessions/{sid}/turns/{tid}/feedback`, `turn_id` in chat/
   stream/`MessageOut`), `static/index.html` (live thumbs + down-vote comment
   box). Tests in `test_repositories.py`, `test_orchestrator.py`, `test_api.py`.
   `User.role` column lands here; admin *gating* is wired in Phase 3.
2. **Attribute** — front-matter parsing; citation→process join; embedding
   fallback + `attribution_method`. **✅ DONE (against fixtures)** —
   `document_source.py` (`parse_front_matter`, `LocalMarkdownSource` strips the
   block + surfaces `process`/`department`), `retrieval.py` (`DocumentStore`
   builds a `source_id → {process, department, title}` lookup via
   `doc_metadata`; chunks carry the labels), `db.py` (`FeedbackAttribution`
   weighted-split table), `attribution.py` (new `AttributionResolver` —
   `resolve_from_citations` 1/N split, `resolve_from_text` nearest-doc),
   `repositories.py` (`add_citations` now carries labels, `citations_for_turn`,
   `replace_attributions`, `attributions_for`), `orchestrator.py` (citation
   join at answer time; embedding attribution for NLP feedback and
   citationless ratings), `main.py` (wires the resolver). Fixture docs now carry
   `process:`/`department:` front-matter. Tests: `test_front_matter.py`,
   `test_attribution.py`, plus additions to `test_retrieval.py`,
   `test_orchestrator.py`, `test_repositories.py`.
   **Gap:** only `LocalMarkdownSource` parses front-matter — `GoogleDriveSource`
   does not yet, so real Drive docs resolve `process=None` and lean on the
   embedding fallback until the shared-drive migration lands and the Drive
   source adopts the same `parse_front_matter` step (a small follow-up).
3. **Surface** — analytics endpoints + admin dashboard + `require_admin` +
   admin seeding + tenant scoping. **✅ DONE** — `repositories.py`
   (`summary`/`hotspots`/`trend` aggregations + `NEGATIVE_EMOTIONS` taxonomy;
   negative = explicit down-vote OR unrated NLP feedback with a negative
   emotion — an up-vote is never negative even if its comment classifies
   negatively), `auth.py` (`require_admin`, 404 for non-admins, role read
   fresh per request), `settings.py` (`ADMIN_EMAILS` allowlist),
   `api.py` (promote-on-register/login; `GET /api/analytics/summary|hotspots|
   trend?days=&dimension=`; `GET /admin` page route; `role` on `UserOut`;
   all queries scoped to the admin's `tenant_id`), `pages/admin.html`
   (dashboard — outside the `/static` mount so the page itself is gated;
   Chart.js 4.4.9 vendored at `static/vendor/`, no CDN; hotspot bars render
   citation blame solid and embedding blame hatched so inferred attribution
   is visually distinct), `static/index.html` (admin-only header link).
   Tests: analytics math + scoping in `test_repositories.py`; gate, seeding,
   and forged-role rejection in `test_api.py` (the admin tests run the real
   cookie path, since `require_admin` bypasses dependency overrides).
   Verified end-to-end against a live server: register→promote→chat→rate→
   analytics→dashboard, including non-admin 404s.
   Note: the `days` query param realises the doc's `since=` semantically
   (cutoff = now − days; absent = all time).
4. **Discover (later)** — embedding clustering (HDBSCAN/KMeans) over feedback
   vectors for emergent themes the taxonomy misses. Deliberately deferred:
   unstable at low volume; the taxonomy join carries the dashboard until
   enough feedback accrues.

## 15. Resolved decisions (formerly open)

- **OD-1 — Multi-doc blame: RESOLVED → split.** Blame is split across all cited
  docs (`weight = 1/N`), aggregated by summed weight. Over enough observations,
  consistently-problematic docs/processes rise above the noise. See §5.4, §8.
- **OD-2 — Non-markdown front-matter: PARKED.** The corpus is being migrated to
  a shared drive with edit permissions to preprocess docs into markdown +
  front-matter, so a non-markdown fallback parser is intentionally **not** being
  built now. Revisit only if some sources can't be preprocessed. See §6.

## 16. Known limitations (accepted)

- **Citation coverage < 100%** — aborted/partial answers store no citations.
- **Historical sparsity** — pre-existing turns have no `TurnCitation` and old
  feedback rows have no process; no backfill is possible for citations.
- **Mixed question+feedback** — the XOR classifier still drops the feedback
  half of a mixed message; explicit thumbs make this low-priority, deferred.
- **SQLite write contention** — fine at MVP; Postgres is the known prod path
  (`DATABASE_URL` swap).
