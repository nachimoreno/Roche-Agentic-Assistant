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

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import Engine
from sqlmodel import Field, SQLModel, create_engine
from uuid_utils import uuid7


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
    emotion: str = Field(index=True)
    message: str
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
