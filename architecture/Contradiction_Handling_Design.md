# Contradicting Documents — Detection, Resolution & Surfacing — Design

> Status: **proposed design, not yet built**. This document is the single
> source of truth for the build phase. It is self-contained: it does not depend
> on chat context and can be read/implemented directly.
>
> Last updated: 2026-06-25.

## 1. Goal

Answer the question the company raised: **what happens when two documents give
contradicting answers to the same question?**

Today the assistant retrieves the top chunks, drops them all into one context
block, and asks the LLM to answer. If two chunks disagree, the model silently
picks one — with **no signal to the scientist and no signal to whoever owns the
documents.** In a lab / scientific-safety domain that silent pick is the
dangerous failure mode.

This design adds: (a) deterministic **version-dedup** so stale copies never
compete, (b) **conflict detection + a `conflict` flag** on answers, (c) a
**surface-both-and-flag** UX so a scientist is told when sources disagree, and
(d) **conflict analytics** so a contradiction becomes a documentation-quality
defect routed to doc owners — reusing the documentation-gap machinery that
already exists.

## 2. Two distinct problems (do not conflate)

"Two documents contradict" is really two cases needing different handling:

| Case | Example | Correct behavior |
|---|---|---|
| **A. Stale-version contradiction** | Same logical doc exists as v1 and v2; v1 says "4 working hours", v2 says "2 working hours" | **Silently prefer the newest, drop the stale one.** This is the Google Drive "multiple versions" problem called out in `Architecture_Features_Requirements.md` §2.2. |
| **B. Genuine source conflict** | Two *distinct, current* docs: cleaning doc A says "70% ethanol", doc B says "90% isopropyl" | **Never silently pick.** Surface the disagreement, cite both, flag it to the scientist and to the doc owner. |

Case A is resolved deterministically (Phase 2) and removes most of the noise so
that what remains and reaches detection (Phase 1) is genuine Case-B conflict.

## 3. What already exists (do not rebuild)

| Capability | Where |
|---|---|
| Hybrid retrieval (dense + BM25 + RRF), `top_k=4` | `src/retrieval.py` (`DocumentStore.retrieve_scored`) |
| Per-chunk context block `[source="…" document="…" section="…"]` fed to the LLM | `src/agent.py` (`_format_context`) |
| Answer schema with server-set flags `low_confidence`, `declined`, retrieval scores | `src/agent.py` (`AnswerResult`, `AnswerComplete`) |
| Off-domain decline + low-confidence "verify" badge plumbing | `src/agent.py` (`_off_domain`, `_low_confidence`) → `orchestrator.py` → `api.py` → `static/index.html` |
| Doc metadata lookup `source_id → {process, department, title, url}` + `modified_at` | `src/retrieval.py` (`doc_metadata`, `_doc_meta`), `document_source.py` (`SourceDocument.modified_at`) |
| Front-matter parser (`process:` / `department:`) | `src/document_source.py` (`parse_front_matter`) |
| Documentation-gap capture (declined/low-confidence) + admin dashboard | `src/repositories.py` (`QuestionGapRepository`), `orchestrator.py` (`_log_question_gap`), `pages/admin.html` |

The key reuse insight: a conflict is **the same shape of signal as a
documentation gap** — a turn the assistant couldn't answer cleanly, worth
showing an admin. The gap pipeline is the template for the conflict pipeline.

## 4. Decisions (proposed — confirm before build)

1. **Resolution policy:** **surface-both-and-flag for genuine (Case B)
   conflicts; auto-resolve version (Case A) conflicts by recency.** Rationale:
   in a lab/safety domain, confidently serving a wrong-but-newer answer is worse
   than telling the scientist "the documentation disagrees, here are both —
   verify." Auto-pick-by-authority is the rejected alternative (revisit once
   trustworthy authority metadata exists — see OD-1).
2. **Detection placement:** **in-prompt first (Phase 1)**, reusing the existing
   single LLM call — no extra latency or cost. A dedicated verifier pass
   (Phase 3) is optional and gated behind a setting.
3. **Version identity:** docs are "the same logical document" when they share an
   explicit `supersedes` chain, else a `(process, normalized-title)` key.
   Recency is decided by `effective_date` front-matter, falling back to
   `SourceDocument.modified_at` (already populated).
4. **Metadata source:** **doc front-matter** (consistent with the feedback-
   analytics decision). New optional keys: `version`, `effective_date`,
   `status`, `owner`, `supersedes`. No new infra.
5. **Surfacing:** a server-set **`conflict: bool`** on the answer, parallel to
   the existing `low_confidence` / `declined` flags, threaded to a UI badge.
   Both conflicting docs are always cited.
6. **Analytics:** log conflicts through the **existing gap repository** (new
   `kind="conflict"`) so they appear on the admin dashboard as a ranked list of
   "documents that contradict each other," routed to owners for reconciliation.

## 5. Metadata foundation (Phase 0)

Resolution needs signals to resolve *by*. Today the only ones are
`modified_at` and `process`/`department`. Extend the front-matter the corpus
already uses:

```yaml
---
process: cleaning-lab-devices
department: lab-operations
version: 3
effective_date: 2026-05-01
status: current          # current | deprecated
owner: lab-ops-docs@roche
supersedes: 06_cleaning_lab_devices_v2.md   # optional, explicit chain
---
```

- `parse_front_matter` (`document_source.py`) is already generic, so this is
  only **adding keys to the extraction loop** in
  `LocalMarkdownSource.list_documents` (currently `("process", "department")`).
- Carry the new keys into `_doc_meta` in `DocumentStore.ingest`
  (`retrieval.py`, alongside `process`/`department`/`title`/`url`) **and** onto
  chunk metadata in `_chunk_document` (currently copies `process`/`department`),
  so both the deterministic resolver and the LLM can see recency/version.
- **One-time re-embed:** adding front-matter changes each doc's content hash, so
  the "only changed docs re-embed" optimization re-embeds the corpus once.
  Expected and acceptable (same note as the feedback-analytics front-matter
  rollout).
- **Drive parity gap:** as with the existing front-matter work, only
  `LocalMarkdownSource` parses front-matter today; `GoogleDriveSource` does not.
  Until the shared-drive migration lands, Drive-sourced docs resolve these keys
  as `None` and rely on `modified_at` for recency. Acceptable; same known gap.

## 6. Phase 1 — In-prompt conflict detection (MVP core)

The LLM already sees every retrieved chunk. Teach it to flag disagreement
instead of papering over it. No extra LLM call.

### 6.1 Expose recency/version in context
Add `modified="…"` (and `version="…"`/`status="…"` when present) to the
per-chunk header built in `_format_context` (`agent.py`), so the model can
reason about which side is newer:
```
[source="06_cleaning_v3.md" document="Cleaning Lab Devices" section="Solvents" modified="2026-05-01" version="3"]
```

### 6.2 New system-prompt rule
Add a rule to `_SYSTEM_PROMPT_HEAD` (`agent.py`), after the existing
ambiguity rule (1a) — and note it is **distinct** from ambiguity (1a = "which
thing did you mean?"; this = "the docs themselves disagree"):

> **Conflicting sources.** If two context chunks give *materially different*
> answers to the same question (different values, steps, or thresholds — not
> mere wording), do NOT silently choose one. State plainly that the
> documentation disagrees, give each position with its own citation, and — only
> if one source is clearly more recent or marked current — note which appears
> authoritative while still advising the scientist to verify. Set
> `conflict=true`.

### 6.3 Schema + flag plumbing
Add a server-/model-set `conflict: bool` (default `False`) field, mirroring
`low_confidence`:
- `AnswerResult` (non-stream) and `AnswerComplete` (stream) in `agent.py`.
- The streaming tail JSON contract: add `"conflict"` to `_STREAM_OUTPUT` and
  parse it in `_parse_tail` (defaults to `False` on a malformed tail, like the
  existing fields).
- The non-stream `_JSON_OUTPUT` schema + `model_validate` path.
- Keep both conflicting docs through `_dedupe_citations` (it already keeps one
  row per `source`, so two distinct sources both survive — good).

**Weakness (accepted for MVP):** in-prompt detection won't catch every
conflict. Phase 3 hardens precision; Phase 2 removes the version-noise that
would otherwise dominate.

## 7. Phase 2 — Deterministic version-dedup (answers arch §2.2)

Before building context, collapse superseded versions so a stale copy never
reaches the LLM at all. This resolves **Case A** silently and correctly, and
shrinks Phase-1 detection down to genuine Case-B conflicts.

- New pure helper in `retrieval.py`, e.g.
  `_collapse_superseded(chunks, doc_meta) -> list[Chunk]`, called inside
  `retrieve_scored` after RRF fusion (so it operates on the final ranked list).
- Logic: group chunks by **logical-document identity** — explicit `supersedes`
  chain if present, else `(process, normalized-title)`. Within a group keep only
  the chunks of the newest doc (`effective_date` → `modified_at`), drop the
  rest. A doc explicitly `status: deprecated` is always dropped when a
  non-deprecated sibling exists.
- Pure function over already-fetched data → unit-testable with no LLM/Chroma.
- **Guard:** only collapse when identity is *confident* (shared `supersedes` or
  identical `process` + high title similarity). When unsure, keep both and let
  Phase 1 treat them as a potential conflict — never silently drop a doc we
  can't prove is a stale version.

## 8. Phase 3 — Optional verifier pass (precision, gated)

For higher-stakes reliability, a dedicated lightweight LLM check over the
retrieved chunks: *"do any of these disagree on a factual claim? return the
conflicting source pairs and the claim."* Runs via the existing `LLMClient`
seam, so it stays provider-agnostic (currently Groq).

- Gated behind a `Settings` flag (e.g. `conflict_verifier_enabled`, default
  off) so it can be demoed on/off and its latency cost measured.
- Output feeds the same `conflict` flag and the same analytics row as Phase 1,
  so nothing downstream changes.
- Deliberately deferred past the MVP: it doubles the LLM round-trips on the
  answer path.

## 9. Phase 4 — Surface & route the conflict

A contradiction is a documentation-quality defect. Surface it to the scientist
**and** to the doc owners.

### 9.1 Scientist-facing (UX)
Thread `conflict` through the same path the existing flags already travel:
`AnswerResult`/`AnswerComplete` → `Response`/`StreamDone` (`orchestrator.py`) →
the SSE `done` frame + non-stream response in `api.py` → a badge in
`static/index.html`, reusing the `low_confidence` "verify" badge plumbing.
Copy: **"⚠️ Sources disagree — verify."** Always render *both* citations.

### 9.2 Owner-facing (analytics) — reuse the gap pipeline
Mirror `_log_question_gap` (`orchestrator.py`) with a `_log_conflict` step on
the answer path:
- Extend `QuestionGapRepository.add` with `kind="conflict"` (alongside the
  existing `"declined"` / `"low_confidence"`), recording the query, language,
  retrieval scores, **and the conflicting source ids** (the two+ cited docs).
  A small additive column (e.g. `conflict_sources: Optional[str]`, comma-joined)
  keeps it in the existing table; a sibling table is the alternative if a
  normalized doc-pair view is wanted later.
- Best-effort and non-fatal, exactly like `_log_question_gap` (a logging hiccup
  must never break the turn).
- Add an admin-dashboard panel (`pages/admin.html`) ranking
  **document pairs that most often contradict**, so owners get a concrete
  reconcile-this worklist. This is the real long-term payoff: chat failures
  become a maintenance signal, the same way the gap dashboard works.

## 10. Testing

Follow the existing pattern — offline fixtures + `FakeLLMClient`, no live Groq.
New `tests/test_contradiction.py` plus additions to existing suites:

- **Fixtures:** add deliberately-conflicting docs under `tests/fixtures/docs/` —
  (a) a v1/v2 pair of the same doc (Case A) with different `effective_date`/
  `version`; (b) two distinct current docs with a genuinely different value
  (Case B, e.g. cleaning solvent %).
- **Phase 2 (deterministic, no LLM):** `_collapse_superseded` drops the stale
  version and keeps only the newest; only the newest is cited. Confident-
  identity guard: dissimilar docs are *not* collapsed.
- **Phase 1 (with `FakeLLMClient`):** for a Case-B context, `conflict=True` is
  set and both docs survive `_dedupe_citations`; malformed stream tail defaults
  `conflict=False`.
- **Phase 4:** `conflict` threads through `Response`/`StreamDone` and the API
  payload; `_log_conflict` writes a `kind="conflict"` gap row with both source
  ids; admin endpoint/panel surfaces the pair. Best-effort logging never raises.
- Run under the `roche` conda env (project convention).

## 11. Phasing

1. **Metadata foundation** — front-matter keys (`version`, `effective_date`,
   `status`, `owner`, `supersedes`) into `document_source.py`, `_doc_meta`,
   chunk metadata. Small; unblocks 2–4.
2. **Version-dedup** — `_collapse_superseded` in `retrieval.py` + confident-
   identity guard. The concrete arch §2.2 win; demoable on its own.
3. **Conflict flag** — context recency headers, system-prompt rule, `conflict`
   field through the schema + stream tail.
4. **Surface (UX)** — `conflict` badge through orchestrator/api/UI; both docs
   cited.
5. **Surface (analytics)** — `_log_conflict` + `kind="conflict"` +
   conflicting-source ids + admin panel.
6. **Verifier pass (later)** — optional gated LLM contradiction check.

**Recommended MVP for the capstone demo:** phases 1 → 2 → 3 → 4. That already
gives a defensible end-to-end answer to the company's question (stale versions
auto-resolved, genuine conflicts surfaced + cited + badged). Phases 5–6 are the
production-grade stretch.

## 12. Open decisions

- **OD-1 — Resolution policy for genuine (Case B) conflicts.** Proposed:
  surface-both-and-flag, never silently pick. The rejected alternative is
  auto-pick by recency/authority. Revisit toward auto-pick only once `status`/
  `owner`/`authority` metadata is trusted enough that "most authoritative" is
  reliable. **Needs sign-off (§4.1).**
- **OD-2 — Conflict store shape.** Proposed: reuse `QuestionGapRepository` with
  `kind="conflict"` + a `conflict_sources` column. Alternative: a normalized
  doc-pair table if pair-level analytics grow. Start with the column.

## 13. Known limitations (accepted)

- **Detection recall < 100%** — in-prompt detection (Phase 1) misses subtle or
  cross-section conflicts; the verifier pass (Phase 3) is the mitigation, off by
  default.
- **Retrieval-bounded** — only conflicts *within the retrieved `top_k=4`* can be
  detected; a contradicting doc that never gets retrieved is invisible. Raising
  `top_k` for detection is a tuning lever, not a guarantee.
- **Drive front-matter gap** — until the shared-drive migration, Drive docs lack
  `version`/`status`/`effective_date` and fall back to `modified_at` only (same
  known gap as the feedback-analytics work).
- **Identity heuristic** — `(process, normalized-title)` version-grouping can
  mis-group docs that share a process but are genuinely different; the
  confident-identity guard errs toward *not* collapsing, so the failure mode is
  "treated as a conflict" (safe) rather than "silently dropped" (unsafe).
