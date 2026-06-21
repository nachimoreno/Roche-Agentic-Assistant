"""
db.py
-----
Schema and engine for the Roche Scientist Assistant persistence layer.

Production-shaped from day one:
- UUIDv7 primary keys (time-ordered, globally unique, index-friendly).
- Nullable `tenant_id` indexed on every table — cheap insurance against a
  multi-table backfill when multi-tenancy lands.
- Nullable `deleted_at` indexed on every table — soft-delete pattern.
- All scalar columns; nothing that wouldn't translate cleanly to
  PostgreSQL.
- Engine driven by `Settings.database_url`. Production swap is one env-var
  change.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import Engine, inspect, text
from sqlmodel import Field, SQLModel, create_engine
from uuid_utils import uuid7


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_id() -> UUID:
    """Generate a UUIDv7 (time-ordered) as a stdlib `uuid.UUID` instance."""
    return UUID(bytes=uuid7().bytes)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class User(SQLModel, table=True):
    id: UUID = Field(default_factory=new_id, primary_key=True)
    email: str = Field(index=True, unique=True)         # stored lowercased
    display_name: Optional[str] = Field(default=None)
    password_hash: str
    # "user" | "admin". Nullable so the additive migration helper can add it to
    # existing dev DBs; treated as "user" when absent. `register` never sets it.
    role: Optional[str] = Field(default="user", index=True)
    tenant_id: Optional[UUID] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)
    deleted_at: Optional[datetime] = Field(default=None, index=True)


class Session(SQLModel, table=True):
    id: UUID = Field(default_factory=new_id, primary_key=True)
    tenant_id: Optional[UUID] = Field(default=None, index=True)
    # Opaque owner id (str(User.id) today; a future SSO subject drops in unchanged).
    user_id: Optional[str] = Field(default=None, index=True)
    title: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow, index=True)
    deleted_at: Optional[datetime] = Field(default=None, index=True)


class Turn(SQLModel, table=True):
    id: UUID = Field(default_factory=new_id, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id", index=True)
    tenant_id: Optional[UUID] = Field(default=None, index=True)
    role: str                            # "user" | "assistant"; enforced in repo
    content: str
    language: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)
    deleted_at: Optional[datetime] = Field(default=None, index=True)


class FeedbackEntry(SQLModel, table=True):
    id: UUID = Field(default_factory=new_id, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id", index=True)
    turn_id: Optional[UUID] = Field(default=None, foreign_key="turn.id", index=True)
    tenant_id: Optional[UUID] = Field(default=None, index=True)
    language: str = Field(index=True)
    # NULLABLE going forward: an explicit thumb may carry no emotion. Phase 1
    # still always writes a concrete value (e.g. "neutral") so existing dev DBs,
    # whose column predates this change as NOT NULL, never see a NULL insert.
    emotion: Optional[str] = Field(default=None, index=True)
    message: str
    # Two feedback streams in one table, distinguished by `source`:
    #   "nlp"      — volunteered feedback classified from a chat message
    #   "explicit" — a thumb up/down on a specific assistant answer
    source: Optional[str] = Field(default="nlp", index=True)
    rating: Optional[str] = Field(default=None, index=True)   # "up" | "down"
    comment: Optional[str] = Field(default=None)              # free text on a down-vote
    # How process/department blame was resolved: "citation" | "embedding" | "none".
    # Populated in Phase 2 (attribution); left None at capture time.
    attribution_method: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)
    deleted_at: Optional[datetime] = Field(default=None, index=True)


class FeedbackAttribution(SQLModel, table=True):
    """One weighted row per document a piece of feedback implicates.

    Blame is *split*: a feedback citing N docs writes N rows of weight 1/N, so
    aggregating `SUM(weight)` by process/department/source lets genuinely
    problematic docs rise above the noise. An embedding-resolved orphan writes a
    single row of weight 1.0. Method is "citation" or "embedding".
    """
    id: UUID = Field(default_factory=new_id, primary_key=True)
    feedback_id: UUID = Field(foreign_key="feedbackentry.id", index=True)
    tenant_id: Optional[UUID] = Field(default=None, index=True)
    source: str = Field(index=True)              # the implicated document
    section: Optional[str] = Field(default=None)
    process: Optional[str] = Field(default=None, index=True)
    department: Optional[str] = Field(default=None, index=True)
    weight: float = Field(default=1.0)
    method: str = Field(index=True)              # "citation" | "embedding"
    distance: Optional[float] = Field(default=None)   # embedding distance, if any
    created_at: datetime = Field(default_factory=utcnow, index=True)
    deleted_at: Optional[datetime] = Field(default=None, index=True)


class TurnCitation(SQLModel, table=True):
    """The documents an assistant turn cited — the link that lets a rating
    reach the doc, and therefore the process/department, it concerns.

    Written at answer time. `process`/`department` are resolved from doc
    front-matter in Phase 2; they are nullable and left None at capture time.
    """
    id: UUID = Field(default_factory=new_id, primary_key=True)
    turn_id: UUID = Field(foreign_key="turn.id", index=True)
    tenant_id: Optional[UUID] = Field(default=None, index=True)
    source: str = Field(index=True)              # Citation.source (filename/id)
    section: Optional[str] = Field(default=None)
    process: Optional[str] = Field(default=None, index=True)
    department: Optional[str] = Field(default=None, index=True)
    rank: int = Field(default=0)                 # citation order; 0 = top-ranked
    created_at: datetime = Field(default_factory=utcnow, index=True)
    deleted_at: Optional[datetime] = Field(default=None, index=True)


class Announcement(SQLModel, table=True):
    """An admin-published message shown as a banner to all scientists.

    Only one is "active" at a time: publishing a new one deactivates the
    previous. Kept as rows (not a single mutable record) so there is an audit
    trail of what was shown and when.
    """
    id: UUID = Field(default_factory=new_id, primary_key=True)
    tenant_id: Optional[UUID] = Field(default=None, index=True)
    message: str
    active: bool = Field(default=True, index=True)
    created_by: Optional[str] = Field(default=None)   # admin email that published it
    created_at: datetime = Field(default_factory=utcnow, index=True)
    deleted_at: Optional[datetime] = Field(default=None, index=True)


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def make_engine(database_url: str, *, echo: bool = False) -> Engine:
    connect_args: dict = {}
    if database_url.startswith("sqlite"):
        # Same engine across threads; the orchestrator is single-process for now.
        connect_args["check_same_thread"] = False
    return create_engine(database_url, echo=echo, connect_args=connect_args)


def create_all(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """Dev convenience: additively add new nullable columns to existing tables.

    `create_all` creates missing tables but never ALTERs existing ones, so a
    dev SQLite database created before a new column was added would be missing
    it. This adds any missing *nullable* columns for SQLite only. Production
    (Postgres) should use real migrations.
    """
    if engine.dialect.name != "sqlite":
        return
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    for table_name, table in SQLModel.metadata.tables.items():
        if table_name not in tables:
            continue
        existing = {c["name"] for c in insp.get_columns(table_name)}
        for col in table.columns:
            if col.name in existing or not col.nullable:
                continue
            ddl_type = col.type.compile(dialect=engine.dialect)
            with engine.begin() as conn:
                conn.execute(
                    text(f'ALTER TABLE "{table_name}" ADD COLUMN "{col.name}" {ddl_type}')
                )
            logger.info(
                "schema.column.added",
                extra={"table": table_name, "column": col.name},
            )
