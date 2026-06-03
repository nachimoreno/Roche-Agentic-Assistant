"""
repositories.py
---------------
Repository layer — the only code in the system that talks to SQLAlchemy.

The orchestrator never imports SQLModel or sessionmaker directly; it talks
to these two classes. Swapping ORMs (or going async) is contained here.

`include_deleted=False` by default everywhere — soft-deleted rows are
invisible to normal code paths but recoverable for compliance.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Engine
from sqlmodel import Session as DbSession, select

from db import FeedbackEntry, Session, Turn, utcnow


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeedbackRepository
# ---------------------------------------------------------------------------

class FeedbackRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def add(self, entry: FeedbackEntry) -> FeedbackEntry:
        with DbSession(self._engine) as db:
            db.add(entry)
            db.commit()
            db.refresh(entry)
        logger.info(
            "feedback.added",
            extra={
                "id": str(entry.id),
                "session_id": str(entry.session_id),
                "language": entry.language,
                "emotion": entry.emotion,
            },
        )
        return entry

    def list(
        self,
        *,
        language: Optional[str] = None,
        emotion: Optional[str] = None,
        since: Optional[datetime] = None,
        tenant_id: Optional[UUID] = None,
        include_deleted: bool = False,
    ) -> list[FeedbackEntry]:
        with DbSession(self._engine) as db:
            stmt = select(FeedbackEntry)
            if not include_deleted:
                stmt = stmt.where(FeedbackEntry.deleted_at.is_(None))  # type: ignore[union-attr]
            if language is not None:
                stmt = stmt.where(FeedbackEntry.language == language)
            if emotion is not None:
                stmt = stmt.where(FeedbackEntry.emotion == emotion)
            if since is not None:
                stmt = stmt.where(FeedbackEntry.created_at >= since)
            if tenant_id is not None:
                stmt = stmt.where(FeedbackEntry.tenant_id == tenant_id)
            stmt = stmt.order_by(FeedbackEntry.created_at)
            return list(db.exec(stmt).all())

    def soft_delete(self, id: UUID) -> None:
        with DbSession(self._engine) as db:
            entry = db.get(FeedbackEntry, id)
            if entry is None or entry.deleted_at is not None:
                return
            entry.deleted_at = utcnow()
            db.add(entry)
            db.commit()


# ---------------------------------------------------------------------------
# SessionRepository
# ---------------------------------------------------------------------------

_VALID_ROLES = {"user", "assistant"}


class SessionRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_or_create(
        self,
        session_id: UUID,
        *,
        tenant_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
    ) -> Session:
        with DbSession(self._engine) as db:
            session = db.get(Session, session_id)
            if session is not None:
                return session
            session = Session(
                id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            logger.info("session.created", extra={"session_id": str(session_id)})
            return session

    def append_turn(
        self,
        session_id: UUID,
        role: str,
        content: str,
        language: Optional[str] = None,
        *,
        tenant_id: Optional[UUID] = None,
    ) -> Turn:
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {_VALID_ROLES}, got {role!r}")
        turn = Turn(
            session_id=session_id,
            tenant_id=tenant_id,
            role=role,
            content=content,
            language=language,
        )
        with DbSession(self._engine) as db:
            db.add(turn)
            db.commit()
            db.refresh(turn)
        return turn

    def recent_turns(
        self,
        session_id: UUID,
        n: int = 10,
        *,
        include_deleted: bool = False,
    ) -> list[Turn]:
        with DbSession(self._engine) as db:
            stmt = select(Turn).where(Turn.session_id == session_id)
            if not include_deleted:
                stmt = stmt.where(Turn.deleted_at.is_(None))  # type: ignore[union-attr]
            stmt = stmt.order_by(Turn.created_at.desc()).limit(n)  # type: ignore[union-attr]
            turns = list(db.exec(stmt).all())
        # Return oldest-first so callers can append directly to chat history.
        return list(reversed(turns))

    def soft_delete_session(self, id: UUID) -> None:
        with DbSession(self._engine) as db:
            session = db.get(Session, id)
            if session is None or session.deleted_at is not None:
                return
            session.deleted_at = utcnow()
            db.add(session)
            db.commit()
