"""
preprocessing_source.py
-----------------------
A `DocumentSource` decorator that enriches another source's documents with
`process`/`department` topic labels at ingest time.

WHY THIS EXISTS:
  Isabella's `document_preprocessor` produces clean markdown + a metadata
  header for each Drive document, but as a standalone CLI that writes files to
  local disk. On a Hugging Face Space the filesystem is ephemeral, so writing
  files buys nothing — the corpus is re-ingested from Drive on every cold start
  anyway. Rather than write-then-read a `data/processed/` folder, this wraps the
  live source and applies the *valuable* part of preprocessing — the keyword
  tagging — in-memory as documents stream into `DocumentStore.ingest()`.

  The other half of preprocessing, structured DOCX/PDF -> markdown conversion,
  must happen on the raw bytes before they are flattened, so it lives in
  `GoogleDriveSource(markdown=True)` (its download step), not here. Together
  they give the full preprocessing benefit with no disk round-trip and no
  second Drive read.

SCOPE — METADATA ONLY:
  This decorator never changes a document's content; it only adds metadata.
  `process`/`department` feed the existing attribution + contradiction layer
  (see `attribution.py` and `retrieval.py`). Inference fills a key only when the
  inner source did not already provide it, so explicit front-matter labels
  (e.g. from `LocalMarkdownSource`) always win over a keyword guess.
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator

from document_source import DocumentSource, SourceDocument

logger = logging.getLogger(__name__)


class PreprocessingSource:
    """Wrap a `DocumentSource`, tagging each document with `process`/`department`.

    The inner source supplies the documents; this only augments their metadata.
    `infer_department_and_process` returns `(department, process_type)`; the
    `process_type` is stored under the existing `process` metadata key so it
    plugs straight into attribution/contradiction handling rather than adding a
    parallel field.
    """

    def __init__(self, inner: DocumentSource) -> None:
        self._inner = inner

    def list_documents(self) -> Iterator[SourceDocument]:
        # Imported lazily so importing this module never pulls in the preprocessor
        # (and its optional deps) unless tagging actually runs.
        from document_preprocessor import infer_department_and_process

        for doc in self._inner.list_documents():
            has_process = bool(doc.metadata.get("process"))
            has_department = bool(doc.metadata.get("department"))
            if has_process and has_department:
                # Source already labelled this doc — respect it, don't second-guess.
                yield doc
                continue

            # Prefer the original filename for keyword matching (it often names
            # the process, e.g. "...Onboarding Guide.docx"); fall back to title.
            name = doc.metadata.get("drive_name") or doc.title
            department, process_type = infer_department_and_process(name, doc.content)

            if not has_process:
                doc.metadata["process"] = process_type
            if not has_department:
                doc.metadata["department"] = department

            logger.debug(
                "preprocess.tagged",
                extra={
                    "doc_id": doc.id,
                    "process": doc.metadata.get("process"),
                    "department": doc.metadata.get("department"),
                },
            )
            yield doc


def wrap_if_enabled(
    source: DocumentSource, *, enabled: bool
) -> DocumentSource:
    """Return `source` wrapped in `PreprocessingSource` when `enabled`, else as-is.

    A tiny helper so `main.build_source` stays declarative.
    """
    return PreprocessingSource(source) if enabled else source
