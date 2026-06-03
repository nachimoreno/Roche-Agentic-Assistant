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
            content = md_path.read_text(encoding="utf-8")
            title = _first_h1(content) or md_path.stem
            yield SourceDocument(
                id=md_path.name,
                title=title,
                content=content,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                metadata={"path": str(md_path)},
            )


def _first_h1(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None
