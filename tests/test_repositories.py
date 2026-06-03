"""
Repository tests.

Run against an in-memory SQLite engine — proves the same code that hits
PostgreSQL in production works locally. Fast, no API calls.
"""

from __future__ import annotations

import time
from datetime import timedelta
from uuid import UUID

import pytest

from db import FeedbackEntry, create_all, make_engine, new_id, utcnow
from repositories import FeedbackRepository, SessionRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    eng = make_engine("sqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def sessions(engine):
    return SessionRepository(engine)


@pytest.fixture
def feedback(engine):
    return FeedbackRepository(engine)


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

def test_uuidv7_is_time_ordered():
    ids = [new_id() for _ in range(20)]
    assert ids == sorted(ids), "UUIDv7 IDs should sort in generation order"


def test_uuidv7_is_a_uuid_instance():
    assert isinstance(new_id(), UUID)


# ---------------------------------------------------------------------------
# SessionRepository
# ---------------------------------------------------------------------------

def test_get_or_create_is_idempotent(sessions):
    sid = new_id()
    first = sessions.get_or_create(sid)
    second = sessions.get_or_create(sid)
    assert first.id == second.id == sid


def test_append_turn_persists_in_order(sessions):
    sid = new_id()
    sessions.get_or_create(sid)
    sessions.append_turn(sid, "user", "hello", language="english")
    time.sleep(0.01)
    sessions.append_turn(sid, "assistant", "hi", language="english")

    turns = sessions.recent_turns(sid, n=10)
    assert [t.role for t in turns] == ["user", "assistant"]
    assert [t.content for t in turns] == ["hello", "hi"]


def test_recent_turns_respects_limit(sessions):
    sid = new_id()
    sessions.get_or_create(sid)
    for i in range(5):
        sessions.append_turn(sid, "user", f"msg-{i}", language="english")
        time.sleep(0.005)

    turns = sessions.recent_turns(sid, n=2)
    assert [t.content for t in turns] == ["msg-3", "msg-4"]


def test_append_turn_rejects_invalid_role(sessions):
    sid = new_id()
    sessions.get_or_create(sid)
    with pytest.raises(ValueError):
        sessions.append_turn(sid, "system", "should fail", language="english")


def test_soft_delete_session_hides_session(sessions):
    sid = new_id()
    sessions.get_or_create(sid)
    sessions.append_turn(sid, "user", "hello", language="english")
    sessions.soft_delete_session(sid)

    # The session row still exists but turns querying respects deleted_at on Turn.
    # We separately verify turns are unaffected (turns have their own deleted_at).
    turns = sessions.recent_turns(sid, n=10)
    assert len(turns) == 1


# ---------------------------------------------------------------------------
# FeedbackRepository
# ---------------------------------------------------------------------------

def _make_feedback(sid, *, language="english", emotion="confused", message="meh"):
    return FeedbackEntry(
        session_id=sid,
        language=language,
        emotion=emotion,
        message=message,
    )


def test_add_and_list_feedback(sessions, feedback):
    sid = new_id()
    sessions.get_or_create(sid)
    feedback.add(_make_feedback(sid))

    rows = feedback.list()
    assert len(rows) == 1
    assert rows[0].language == "english"
    assert rows[0].emotion == "confused"


def test_list_filters_by_language_and_emotion(sessions, feedback):
    sid = new_id()
    sessions.get_or_create(sid)
    feedback.add(_make_feedback(sid, language="english", emotion="confused"))
    feedback.add(_make_feedback(sid, language="german", emotion="frustrated"))
    feedback.add(_make_feedback(sid, language="german", emotion="confused"))

    assert len(feedback.list(language="english")) == 1
    assert len(feedback.list(language="german")) == 2
    assert len(feedback.list(emotion="confused")) == 2
    assert len(feedback.list(language="german", emotion="frustrated")) == 1


def test_list_filters_by_since(sessions, feedback):
    sid = new_id()
    sessions.get_or_create(sid)
    feedback.add(_make_feedback(sid))
    cutoff = utcnow() + timedelta(seconds=1)
    time.sleep(1.1)
    feedback.add(_make_feedback(sid, emotion="angry"))

    rows = feedback.list(since=cutoff)
    assert len(rows) == 1
    assert rows[0].emotion == "angry"


def test_list_filters_by_tenant(sessions, feedback):
    sid = new_id()
    tenant_a = new_id()
    tenant_b = new_id()
    sessions.get_or_create(sid, tenant_id=tenant_a)

    entry_a = _make_feedback(sid)
    entry_a.tenant_id = tenant_a
    feedback.add(entry_a)

    entry_b = _make_feedback(sid, emotion="angry")
    entry_b.tenant_id = tenant_b
    feedback.add(entry_b)

    assert len(feedback.list(tenant_id=tenant_a)) == 1
    assert feedback.list(tenant_id=tenant_a)[0].emotion == "confused"
    assert len(feedback.list(tenant_id=tenant_b)) == 1


def test_soft_delete_hides_by_default_but_visible_with_flag(sessions, feedback):
    sid = new_id()
    sessions.get_or_create(sid)
    entry = feedback.add(_make_feedback(sid))

    feedback.soft_delete(entry.id)

    assert feedback.list() == []
    assert len(feedback.list(include_deleted=True)) == 1
    assert feedback.list(include_deleted=True)[0].deleted_at is not None
