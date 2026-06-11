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
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session as DbSession, select

from db import FeedbackEntry, Session, Turn, TurnCitation, User, utcnow


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UserRepository
# ---------------------------------------------------------------------------

class EmailTakenError(ValueError):
    """Raised when creating a user whose email already exists."""


class UserRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @staticmethod
    def _norm(email: str) -> str:
        return email.strip().lower()

    def create(
        self,
        *,
        email: str,
        password_hash: str,
        display_name: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
    ) -> User:
        user = User(
            email=self._norm(email),
            password_hash=password_hash,
            display_name=display_name,
            tenant_id=tenant_id,
        )
        try:
            with DbSession(self._engine) as db:
                db.add(user)
                db.commit()
                db.refresh(user)
        except IntegrityError as exc:
            raise EmailTakenError(email) from exc
        logger.info("user.created", extra={"user_id": str(user.id)})
        return user

    def get(self, id: UUID) -> Optional[User]:
        with DbSession(self._engine) as db:
            return db.get(User, id)

    def get_by_email(self, email: str) -> Optional[User]:
        with DbSession(self._engine) as db:
            stmt = select(User).where(
                User.email == self._norm(email),
                User.deleted_at.is_(None),  # type: ignore[union-attr]
            )
            return db.exec(stmt).first()

    def set_role(self, id: UUID, role: str) -> Optional[User]:
        """Set a user's role (e.g. promote to "admin"). Used by admin seeding."""
        with DbSession(self._engine) as db:
            user = db.get(User, id)
            if user is None:
                return None
            user.role = role
            db.add(user)
            db.commit()
            db.refresh(user)
        logger.info("user.role.set", extra={"user_id": str(id), "role": role})
        return user


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

    def upsert_rating(
        self,
        *,
        turn_id: UUID,
        session_id: UUID,
        rating: str,
        comment: Optional[str] = None,
        emotion: Optional[str] = None,
        language: str,
        tenant_id: Optional[UUID] = None,
    ) -> FeedbackEntry:
        """Record an explicit thumb on an assistant turn, idempotently.

        One explicit rating per `turn_id` (a turn belongs to one session, owned
        by one user, so this is effectively one rating per user). Re-rating the
        same turn updates the existing row rather than stacking duplicates that
        would inflate the negative rate.
        """
        with DbSession(self._engine) as db:
            stmt = select(FeedbackEntry).where(
                FeedbackEntry.turn_id == turn_id,
                FeedbackEntry.source == "explicit",
                FeedbackEntry.deleted_at.is_(None),  # type: ignore[union-attr]
            )
            entry = db.exec(stmt).first()
            if entry is None:
                entry = FeedbackEntry(
                    session_id=session_id,
                    turn_id=turn_id,
                    tenant_id=tenant_id,
                    language=language,
                    emotion=emotion,
                    message=comment or "",
                    comment=comment,
                    rating=rating,
                    source="explicit",
                )
            else:
                entry.rating = rating
                entry.comment = comment
                entry.message = comment or ""
                entry.emotion = emotion
                entry.language = language
            db.add(entry)
            db.commit()
            db.refresh(entry)
        logger.info(
            "feedback.rating",
            extra={
                "id": str(entry.id),
                "turn_id": str(turn_id),
                "rating": rating,
                "has_comment": bool(comment),
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

    def create(self, *, user_id: str, title: Optional[str] = None) -> Session:
        """Create a fresh empty session owned by `user_id`."""
        session = Session(user_id=user_id, title=title)
        with DbSession(self._engine) as db:
            db.add(session)
            db.commit()
            db.refresh(session)
        logger.info("session.created", extra={"session_id": str(session.id)})
        return session

    def get(self, session_id: UUID, *, include_deleted: bool = False) -> Optional[Session]:
        with DbSession(self._engine) as db:
            session = db.get(Session, session_id)
            if session is None:
                return None
            if session.deleted_at is not None and not include_deleted:
                return None
            return session

    def list_sessions(self, user_id: str) -> list[Session]:
        """A user's sessions, newest first, excluding soft-deleted."""
        with DbSession(self._engine) as db:
            stmt = (
                select(Session)
                .where(Session.user_id == user_id, Session.deleted_at.is_(None))  # type: ignore[union-attr]
                .order_by(Session.created_at.desc())  # type: ignore[union-attr]
            )
            return list(db.exec(stmt).all())

    def get_turn(self, turn_id: UUID, *, include_deleted: bool = False) -> Optional[Turn]:
        """Fetch a single turn by id (for rating validation)."""
        with DbSession(self._engine) as db:
            turn = db.get(Turn, turn_id)
            if turn is None:
                return None
            if turn.deleted_at is not None and not include_deleted:
                return None
            return turn

    def add_citations(
        self,
        turn_id: UUID,
        citations: list[tuple[str, Optional[str]]],
        *,
        tenant_id: Optional[UUID] = None,
    ) -> None:
        """Persist the documents an assistant turn cited.

        `citations` is an ordered list of `(source, section)` tuples — kept as
        plain tuples so this layer stays decoupled from the agent's `Citation`.
        `rank` follows the order (0 = top-ranked). `process`/`department` are
        resolved in Phase 2; left None here.
        """
        if not citations:
            return
        with DbSession(self._engine) as db:
            for rank, (source, section) in enumerate(citations):
                db.add(
                    TurnCitation(
                        turn_id=turn_id,
                        tenant_id=tenant_id,
                        source=source,
                        section=section,
                        rank=rank,
                    )
                )
            db.commit()

    def messages(self, session_id: UUID, *, include_deleted: bool = False) -> list[Turn]:
        """All turns for a session, oldest first."""
        with DbSession(self._engine) as db:
            stmt = select(Turn).where(Turn.session_id == session_id)
            if not include_deleted:
                stmt = stmt.where(Turn.deleted_at.is_(None))  # type: ignore[union-attr]
            stmt = stmt.order_by(Turn.created_at)  # type: ignore[union-attr]
            return list(db.exec(stmt).all())

    def set_title_if_unset(self, session_id: UUID, title: str) -> None:
        """Set the session title only if it has none yet (first message)."""
        with DbSession(self._engine) as db:
            session = db.get(Session, session_id)
            if session is not None and session.title is None:
                session.title = title
                db.add(session)
                db.commit()

    def rename(self, session_id: UUID, title: str) -> None:
        with DbSession(self._engine) as db:
            session = db.get(Session, session_id)
            if session is None:
                return
            session.title = title
            db.add(session)
            db.commit()

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
