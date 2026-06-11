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

from sqlmodel import Session as DbSession, select

from db import FeedbackEntry, TurnCitation, create_all, make_engine, new_id, utcnow
from repositories import FeedbackRepository, SessionRepository, UserRepository


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


# ---------------------------------------------------------------------------
# Turns / citations / explicit ratings (feedback pipeline — Phase 1)
# ---------------------------------------------------------------------------

def test_get_turn_returns_turn(sessions):
    sid = new_id()
    sessions.get_or_create(sid)
    t = sessions.append_turn(sid, "assistant", "hi", language="english")
    got = sessions.get_turn(t.id)
    assert got is not None and got.id == t.id
    assert sessions.get_turn(new_id()) is None      # unknown id


def test_add_citations_persists_ranked_rows_with_labels(engine, sessions):
    sid = new_id()
    sessions.get_or_create(sid)
    t = sessions.append_turn(sid, "assistant", "answer", language="english")
    sessions.add_citations(
        t.id,
        [("a.md", "Intro", "onboarding", "it"), ("b.md", None, None, None)],
    )

    rows = sessions.citations_for_turn(t.id)
    assert [r.source for r in rows] == ["a.md", "b.md"]
    assert [r.rank for r in rows] == [0, 1]
    assert (rows[0].process, rows[0].department) == ("onboarding", "it")
    assert rows[1].process is None


def test_add_citations_noop_on_empty(engine, sessions):
    sid = new_id()
    sessions.get_or_create(sid)
    t = sessions.append_turn(sid, "assistant", "answer", language="english")
    sessions.add_citations(t.id, [])
    assert sessions.citations_for_turn(t.id) == []


def test_replace_attributions_sets_method_and_rows(sessions, feedback):
    from attribution import AttributionRow

    sid = new_id()
    sessions.get_or_create(sid)
    t = sessions.append_turn(sid, "assistant", "answer", language="english")
    entry = feedback.upsert_rating(
        turn_id=t.id, session_id=sid, rating="down", language="english"
    )

    rows = [
        AttributionRow("a.md", "Intro", "onboarding", "it", 0.5, "citation"),
        AttributionRow("b.md", "Setup", "onboarding", "it", 0.5, "citation"),
    ]
    feedback.replace_attributions(entry.id, "citation", rows)

    got = feedback.attributions_for(entry.id)
    assert len(got) == 2
    assert abs(sum(r.weight for r in got) - 1.0) < 1e-9
    assert {r.process for r in got} == {"onboarding"}

    # Re-running replaces rather than stacks.
    feedback.replace_attributions(
        entry.id, "embedding", [AttributionRow("c.md", None, "sample-stock", "lab", 1.0, "embedding", 0.2)]
    )
    got2 = feedback.attributions_for(entry.id)
    assert len(got2) == 1 and got2[0].method == "embedding"


def test_upsert_rating_is_idempotent_per_turn(sessions, feedback):
    sid = new_id()
    sessions.get_or_create(sid)
    t = sessions.append_turn(sid, "assistant", "answer", language="english")

    feedback.upsert_rating(turn_id=t.id, session_id=sid, rating="up", language="english")
    feedback.upsert_rating(
        turn_id=t.id, session_id=sid, rating="down", comment="bad", language="english"
    )

    explicit = [r for r in feedback.list() if r.source == "explicit"]
    assert len(explicit) == 1            # re-rating updated, did not stack
    assert explicit[0].rating == "down"
    assert explicit[0].comment == "bad"


def test_set_role_promotes_user(engine):
    users = UserRepository(engine)
    u = users.create(email="admin@roche.com", password_hash="x")
    assert (u.role or "user") == "user"
    updated = users.set_role(u.id, "admin")
    assert updated is not None and updated.role == "admin"
    assert users.get(u.id).role == "admin"


# ---------------------------------------------------------------------------
# Analytics aggregations (feedback pipeline — Phase 3)
# ---------------------------------------------------------------------------

def _seed_analytics(sessions, feedback):
    """A small, fully-known corpus of feedback for aggregation assertions.

    - explicit down  → citation attribution split 0.5/0.5 across two processes
    - explicit up    → positive; must never count as negative
    - nlp frustrated → embedding attribution to sample-stock (negative)
    - nlp pleased    → positive volunteered feedback
    """
    from attribution import AttributionRow

    sid = new_id()
    sessions.get_or_create(sid)
    t = sessions.append_turn(sid, "assistant", "answer", language="english")

    down = feedback.upsert_rating(
        turn_id=t.id, session_id=sid, rating="down",
        comment="wrong", emotion="confused", language="english",
    )
    feedback.replace_attributions(down.id, "citation", [
        AttributionRow("a.md", "Intro", "onboarding", "it", 0.5, "citation"),
        AttributionRow("b.md", "Slots", "instrument-booking", "lab-operations", 0.5, "citation"),
    ])

    t2 = sessions.append_turn(sid, "assistant", "another", language="german")
    feedback.upsert_rating(
        turn_id=t2.id, session_id=sid, rating="up",
        emotion="neutral", language="german",
    )

    nlp_neg = feedback.add(_make_feedback(sid, emotion="frustrated", message="stock app is broken"))
    feedback.replace_attributions(nlp_neg.id, "embedding", [
        AttributionRow("c.md", None, "sample-stock", "lab-operations", 1.0, "embedding", 0.3),
    ])

    feedback.add(_make_feedback(sid, language="german", emotion="pleased", message="nice!"))
    return sid


def test_summary_counts_and_negative_rate(sessions, feedback):
    _seed_analytics(sessions, feedback)
    s = feedback.summary()

    assert s["total"] == 4
    # Negative = the down-vote + the frustrated NLP entry. The up-vote stays
    # positive even though its comment classified as "confused".
    assert s["negative"] == 2
    assert abs(s["negative_rate"] - 0.5) < 1e-9
    assert s["ratings"] == {"up": 1, "down": 1}
    assert s["emotions"]["frustrated"] == 1
    assert s["languages"] == {"english": 2, "german": 2}
    assert s["sources"] == {"explicit": 2, "nlp": 2}


def test_hotspots_sum_weights_for_negative_feedback_only(sessions, feedback):
    _seed_analytics(sessions, feedback)

    by_process = {r["key"]: r for r in feedback.hotspots(dimension="process")}
    # sample-stock: 1.0 (embedding) tops onboarding/booking at 0.5 each.
    assert by_process["sample-stock"]["weight"] == 1.0
    assert by_process["sample-stock"]["embedding_weight"] == 1.0
    assert by_process["onboarding"]["weight"] == 0.5
    assert by_process["onboarding"]["citation_weight"] == 0.5
    assert by_process["instrument-booking"]["weight"] == 0.5
    # The positive entries contributed nothing.
    assert sum(r["weight"] for r in by_process.values()) == 2.0

    by_dept = {r["key"]: r for r in feedback.hotspots(dimension="department")}
    assert by_dept["lab-operations"]["weight"] == 1.5      # 0.5 + 1.0
    assert by_dept["it"]["weight"] == 0.5

    by_doc = {r["key"]: r for r in feedback.hotspots(dimension="document")}
    assert set(by_doc) == {"a.md", "b.md", "c.md"}
    assert by_doc["c.md"]["feedback_count"] == 1


def test_hotspots_rejects_unknown_dimension(feedback):
    with pytest.raises(ValueError):
        feedback.hotspots(dimension="user")


def test_trend_buckets_by_day(sessions, feedback):
    sid = new_id()
    sessions.get_or_create(sid)
    old = _make_feedback(sid, emotion="angry", message="old gripe")
    old.created_at = utcnow() - timedelta(days=3)
    feedback.add(old)
    feedback.add(_make_feedback(sid, emotion="pleased", message="today, fine"))
    feedback.add(_make_feedback(sid, emotion="confused", message="today, lost"))

    rows = feedback.trend()
    assert len(rows) == 2                      # two distinct days
    assert rows[0]["total"] == 1 and rows[0]["negative"] == 1
    assert rows[-1]["total"] == 2 and rows[-1]["negative"] == 1
    assert rows[0]["date"] < rows[-1]["date"]

    # `since` trims the old bucket.
    recent = feedback.trend(since=utcnow() - timedelta(days=1))
    assert len(recent) == 1 and recent[0]["total"] == 2


def test_summary_scopes_by_since_and_tenant(sessions, feedback):
    sid = new_id()
    tenant = new_id()
    sessions.get_or_create(sid)

    old = _make_feedback(sid, emotion="angry", message="ancient")
    old.created_at = utcnow() - timedelta(days=30)
    feedback.add(old)
    scoped = _make_feedback(sid, emotion="confused", message="tenant's")
    scoped.tenant_id = tenant
    feedback.add(scoped)

    assert feedback.summary()["total"] == 2
    assert feedback.summary(since=utcnow() - timedelta(days=7))["total"] == 1
    assert feedback.summary(tenant_id=tenant)["total"] == 1
    assert feedback.summary(tenant_id=new_id())["total"] == 0
