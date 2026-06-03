"""
logging_setup.py
----------------
Structured logging with per-turn correlation IDs.

Pattern:
    from logging_setup import setup_logging, new_correlation_id
    setup_logging(level="INFO", fmt="json")
    cid = new_correlation_id()      # at the start of each turn / request
    logger.info("turn.start", extra={"session_id": str(sid)})

The correlation ID is attached to every log record automatically via a
ContextVar + LogFilter, so consumers don't need to pass it everywhere.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from contextvars import ContextVar
from pathlib import Path
from uuid import UUID

from uuid_utils import uuid7


# Suppress known-benign UserWarnings before any consumer imports run.
# - fastembed: informational pooling-change notice; queries and documents
#   both use the new pooling, so retrieval ordering is unaffected.
# - huggingface_hub: confirms it is honouring our progress-bar env var.
warnings.filterwarnings(
    "ignore",
    message=r".*now uses mean pooling instead of CLS embedding.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Cannot enable progress bars.*",
    category=UserWarning,
)


# Libraries that log at INFO/DEBUG and clutter the console. We always
# clamp them to WARNING so the demo stays clean.
_NOISY_LOGGERS = (
    "chromadb",
    "chromadb.telemetry",
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "fastembed",
    "sentence_transformers",
    "posthog",
    "asyncio",
)


# fastembed and huggingface_hub emit tqdm progress bars directly to
# stderr on first model download. Suppress them globally before either
# is imported.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("FASTEMBED_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


def new_correlation_id() -> str:
    """Generate a fresh UUIDv7 correlation ID and bind it to the current context."""
    cid = str(UUID(bytes=uuid7().bytes))
    _correlation_id.set(cid)
    return cid


def current_correlation_id() -> str:
    return _correlation_id.get()


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_") or key == "correlation_id":
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", "-")
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<5} [{cid[:8]}] {record.name}: {record.getMessage()}"
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _JsonFormatter._RESERVED
            and not k.startswith("_")
            and k != "correlation_id"
        }
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(
    level: str = "INFO",
    fmt: str = "text",
    log_file: str | Path | None = None,
) -> None:
    """Configure root logging exactly once.

    Parameters
    ----------
    level : str
        Standard logging level name (DEBUG, INFO, WARNING, ERROR).
    fmt : str
        "json" for production (machine-readable) or "text" for local dev.
    log_file : str | Path | None
        If set, logs are written to this file and the console stays
        clean. If unset, logs go to stderr (handy for development).
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler: logging.Handler
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.addFilter(_CorrelationFilter())
    handler.setFormatter(_JsonFormatter() if fmt == "json" else _TextFormatter())

    root.addHandler(handler)
    root.setLevel(level.upper())

    # Always tame third-party loggers so they don't leak to stderr.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
