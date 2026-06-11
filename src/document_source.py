"""
document_source.py
------------------
Document source seam.

`DocumentSource` is the interface; `LocalMarkdownSource` is the default
implementation. `GoogleDriveSource` and `SharePointSource` (architecture
§2.2) become future impls that produce the same `SourceDocument` shape.
The retrieval layer is source-agnostic by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


logger = logging.getLogger(__name__)


@dataclass
class SourceDocument:
    """A document pulled from a source, normalized for the ingestion pipeline."""

    id: str                                   # stable across loads
    title: str
    content: str
    modified_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentSource(Protocol):
    def list_documents(self) -> Iterable[SourceDocument]: ...


class CompositeSource:
    """A `DocumentSource` that chains several sources into one stream.

    Used for `document_source="all"` — e.g. local markdown *and* Google Drive
    ingested together. Each child yields the same `SourceDocument` shape, so
    downstream consumers stay source-agnostic.

    `SourceDocument.id` is namespaced per source (filename vs. Drive file id),
    so documents from different sources never collide in the vector store. A
    failure in one source is logged and skipped rather than aborting the whole
    ingest — a Drive outage shouldn't take local docs offline.
    """

    def __init__(self, sources: Iterable[DocumentSource]) -> None:
        self._sources = list(sources)

    def list_documents(self) -> Iterator[SourceDocument]:
        for source in self._sources:
            name = type(source).__name__
            try:
                yield from source.list_documents()
            except Exception:
                logger.exception("source.failed", extra={"source": name})


class LocalMarkdownSource:
    """Default `DocumentSource` reading `.md` files from a local directory."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def list_documents(self) -> Iterator[SourceDocument]:
        if not self._path.exists():
            logger.warning("docs.path.missing", extra={"path": str(self._path)})
            return

        for md_path in sorted(self._path.glob("*.md")):
            stat = md_path.stat()
            raw = md_path.read_text(encoding="utf-8")
            # Split off YAML-ish front-matter (process/department labels). The
            # body is what gets embedded — front-matter is metadata only and
            # must not pollute retrieval.
            front, content = parse_front_matter(raw)
            title = _first_h1(content) or md_path.stem
            metadata: dict[str, Any] = {"path": str(md_path)}
            for key in ("process", "department"):
                if front.get(key):
                    metadata[key] = front[key]
            yield SourceDocument(
                id=md_path.name,
                title=title,
                content=content,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                metadata=metadata,
            )


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a leading `--- ... ---` front-matter block from the body.

    Returns `(fields, body)`. Intentionally a minimal `key: value` parser (no
    nesting, no external YAML dependency) — enough for `process:`/`department:`
    labels. When there is no well-formed block, returns `({}, text)` unchanged.
    """
    if not text.lstrip().startswith("---"):
        return {}, text
    # Work from the first '---' line so a leading blank line is tolerated.
    stripped = text.lstrip("﻿")
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    fields: dict[str, str] = {}
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return fields, body
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            if key:
                fields[key] = value.strip().strip("'\"")
    # No closing delimiter — treat the whole text as body, don't lose content.
    return {}, text


def _first_h1(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None
