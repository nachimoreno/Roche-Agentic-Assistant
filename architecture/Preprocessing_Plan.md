# Document Preprocessing Module — Scope & Plan

> Status: **proposed scope, not yet built**. This document is the single
> source of truth for the preprocessing build phase. It is self-contained and
> can be read/implemented without chat context.
>
> Last updated: 2026-06-25.

## 1. Goal

Add a **document preprocessing module** that improves retrieval quality and
metadata coverage of the corpus *before* it is chunked and embedded — and
**persists** the expensive parts of that work back to Google Drive so they are
computed once, not on every ingest.

Two outcomes:

1. **Better chunks** — documents arrive at the chunker with real structure and
   without boilerplate noise, so semantic retrieval and BM25 both improve.
2. **Richer metadata** — every Drive document gains `process` / `department` /
   `language` / summary tags (today only local markdown has these), which feeds
   citation→process attribution and the documentation-gap analytics.

## 2. Decisions already made

| Decision | Choice | Why |
|---|---|---|
| Persistence surface | **Shadow folder** (`/_processed`), not mutation of originals | Originals are human-owned, edited live; they must never be touched. |
| Drive scope | **`drive.file`** | Least-privilege: the app can manage *only files it created*. It is structurally incapable of altering originals. |
| Request-path impact | **None** | Preprocessing is an offline job + an in-memory ingest decorator. Never runs during a user turn. |
| Default behavior | **Off** | New settings default off, so existing tests and the local-markdown path are unchanged until explicitly enabled. |

> Consequence of `drive.file`: the app **cannot** write `appProperties` onto
> original files it did not create. All persisted enrichment therefore lives in
> the shadow folder (a cleaned text file + a JSON sidecar per document), not as
> metadata on the originals.

## 3. What already exists (do not rebuild)

| Capability | Where |
|---|---|
| `DocumentSource` Protocol + `SourceDocument` shape | `src/document_source.py` |
| Google Drive ingestion (Docs/Sheets/PDF/DOCX/md), name-based dedup | `src/google_drive_source.py` |
| Front-matter `process`/`department` parsing (local md only) | `src/document_source.py` (`parse_front_matter`) |
| Chunking on `## ` H2 headings, greedy repack to 1200 chars | `src/retrieval.py` (`_chunk_document`, `_split_by_h2`, `_split_long`) |
| Content-hash manifest for incremental re-embed | `src/retrieval.py` (`ingest`, `manifest.json`) |
| `CompositeSource` (chain multiple sources) | `src/document_source.py` |
| Composition root that selects the source | `src/main.py` (`build_source`, `build_assistant`) |
| Citation→process attribution | `src/attribution.py`, `DocumentStore.doc_metadata` |

## 4. Problems this addresses (grounded in current code)

1. **Chunking collapses for Google Docs.** `_chunk_document` splits on `## `
   markdown headings (`retrieval.py:314`), but Google Docs are exported as
   `text/plain` (`google_drive_source.py:77`), which strips heading structure.
   Each Doc becomes one headingless section, hard-sliced on a 1200-char
   boundary by `_split_long` — semantically blind chunks. **Biggest single
   retrieval-quality issue.**
2. **Drive docs carry no `process`/`department`.** Local markdown gets these
   from front-matter (`document_source.py:84`); the Drive source never sets
   them. Attribution and documentation-gap analytics run half-blind on the real
   corpus.
3. **Spreadsheets become raw CSV** (`google_drive_source.py:78`) and PDFs get
   naive `extract_text()` with no header/footer/page-number stripping — noisy
   chunks that dilute embeddings.
4. **Version drift is only name-based.** `_deduplicate` keeps the
   newest-by-name file (`google_drive_source.py:360`), but the architecture
   spec (lines 49, 147, 254, 298) calls for handling true near-duplicates with
   different filenames.

## 5. Scope, tiered by ROI

### Tier 1 — high value, low effort, no Drive writes

| Item | What it does | Where |
|---|---|---|
| **Markdown export** | Export Google Docs as `text/markdown` instead of `text/plain`. Restores `## ` headings → fixes chunking immediately. | `google_drive_source.py` (`_GDOC_EXPORT_MIME`) |
| **Deterministic cleaning** | Normalize whitespace + Unicode (NFC) + smart quotes; de-hyphenate PDF line-wraps; strip page numbers, repeated headers/footers, confidentiality stamps, tables of contents, revision-history tables. | new `src/preprocessing.py` |
| **Heading promotion** | Detect implicit structure (ALL-CAPS lines, `1.2 Section` numbering) and promote to `## ` so the existing chunker works on docs that still lack markdown headings. | `src/preprocessing.py` |

Applied **in memory at ingest time** via a `NormalizingSource` decorator — no
Drive access beyond what exists. This delivers most of the retrieval-quality
win on its own.

### Tier 2 — high value, persisted to the shadow folder

| Item | What it does | Persisted as |
|---|---|---|
| **process/department classification** | One LLM pass per doc to tag `process`/`department`, giving Drive docs the metadata model local docs already have. | sidecar JSON |
| **Summary + keyword/synonym header** | Generate a short summary; extract SOP codes, part numbers, app names, and DE/FR/IT synonyms; prepend as a retrieval-boost header. | cleaned text + sidecar |
| **Language detection** | Tag source language; flag docs needing translation. | sidecar JSON |
| **Semantic near-dup / canonical tagging** | Embedding-similarity dedup beyond filename; mark the canonical version and record `superseded_by`. Satisfies the version-drift requirement. | sidecar JSON |

These are **expensive** (LLM + embedding calls), which is exactly why they are
computed once by an offline job and persisted. Idempotency via a stored source
content-hash, mirroring the existing `manifest.json` pattern.

### Tier 3 — deferred / out of scope

- LLM-assisted full re-sectioning of structureless docs (write restructured
  text to the shadow). Only if Tier-1 heading promotion proves insufficient.
- Splitting mega-docs / merging fragments — high risk of altering meaning; keep
  human-in-the-loop.
- OCR for scanned PDFs (`pytesseract`) — only if the real corpus contains
  image-only PDFs.

## 6. Architecture

All additions sit on existing seams; nothing downstream of the source changes.

- **`src/preprocessing.py`** — pure, composable `(str) -> str` steps and a
  `Preprocessor([...])` pipeline. Fully offline-unit-testable, matching the
  repo's test ethos.
- **`NormalizingSource(DocumentSource)`** — a decorator wrapping any source,
  applying Tier-1 cleaning in memory. Wired in `build_source` (`main.py:54`);
  downstream is unchanged.
- **`scripts/preprocess_drive.py`** — the offline Tier-2 write-back job. Reads
  originals, runs the full pipeline incl. LLM enrichment, writes
  `/_processed/<id>.md` + `<id>.json` sidecar. Idempotent via content-hash
  manifest. Never in the request path; run manually or scheduled.
- **`GoogleDriveSource` extensions** — (a) markdown export; (b) when a fresh
  shadow exists (sidecar hash matches the original's current hash), prefer the
  shadow text and merge sidecar metadata into `SourceDocument.metadata`; else
  fall back to the live original.
- **`Settings`** — `preprocess_enabled`, `preprocess_persist`,
  `processed_folder_id`, all defaulting off/empty.

### Shadow read-back, staleness, and fallback

```
original (Drive)  --hash-->  sidecar.hash ?
   match    -> use /_processed/<id>.md  + sidecar metadata   (preprocessed)
   mismatch -> use original text, log stale-shadow            (safe fallback)
   absent   -> use original text                              (not yet processed)
```

Because the shadow stores the *source* hash it was derived from, an edited
original automatically invalidates its shadow until the job re-runs — the
system can never silently serve preprocessed text that no longer matches its
source.

## 7. Phasing

1. **Phase 1 — Tier 1, no Drive writes.** `preprocessing.py` +
   `NormalizingSource` + markdown export. Measurable retrieval lift, zero risk.
   Re-run `scripts/calibrate_guardrail.py` afterward (cleaner text shifts the
   dense-similarity distribution that the off-domain guardrail is tuned to).
2. **Phase 2 — Tier 2, persisted.** `preprocess_drive.py` writing
   process/department/summary/language to the shadow folder under `drive.file`.
   Feeds attribution + documentation-gap analytics.
3. **Phase 3 — optional.** Semantic dedup / canonical tagging; LLM re-sectioning
   only if needed.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Corrupting canonical Roche docs | `drive.file` scope + shadow folder; originals are never writable by the app. |
| Serving stale preprocessed text | Sidecar stores source content-hash; mismatch falls back to the live original. |
| LLM mis-classifying process/department | Treat as a hint, not ground truth; keep the value inspectable in the sidecar; low blast radius (affects attribution, not answer correctness). |
| Guardrail drift after cleaning | Recalibrate `retrieval_min_dense`/`warn`/`lexical` post-Phase-1. |
| Cost of LLM enrichment | Offline, idempotent, hash-gated — only re-runs on changed docs. |
| Hidden behavior change in tests | All new settings default off; local-markdown path and existing fixtures untouched. |

## 9. Testing

- **Unit (offline):** each `preprocessing.py` step in isolation — boilerplate
  stripping, de-hyphenation, heading promotion, Unicode normalization.
- **Decorator:** `NormalizingSource` yields the same `SourceDocument` shape and
  preserves ids/metadata.
- **Chunking regression:** a headingless Drive-style fixture chunks into
  multiple sections *after* heading promotion (vs. one blob before).
- **Shadow read-back:** fresh shadow is used; stale shadow (hash mismatch) falls
  back to the original; absent shadow falls back to the original.
- **Live (opt-in, skipped without creds):** the `preprocess_drive.py` job
  against a real Drive folder, gated like the existing live Drive tests.

## 10. Open questions for the team

- Confirm Roche IT will grant the service account `drive.file` write scope.
- Confirm a dedicated `/_processed` subfolder is acceptable inside the corpus
  folder (vs. a separate Drive folder).
- Decide whether Phase 2 enrichment should be reviewable by a human before the
  ingest trusts it, or trusted automatically.
