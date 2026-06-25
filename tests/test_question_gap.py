"""
Documentation-gap tests — QuestionGapRepository, the demo-gap seed, and the
orchestrator's gap-logging hook.

All offline: an in-memory SQLite engine plus a deterministic fake embedder, so
clustering is exercised without loading the real ONNX model or hitting Groq.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from agent import AnswerResult
from conversation_layer import AnalysisResult
from db import create_all, make_engine, QuestionGap
from demo_seed import (
    DEMO_TENANT_ID,
    demo_gap_count,
    ensure_demo_gaps,
    reset,
    seed_gaps,
)
from orchestrator import Assistant
from repositories import FeedbackRepository, QuestionGapRepository, SessionRepository


# ---------------------------------------------------------------------------
# Fixtures & fakes
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    eng = make_engine("sqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


class FakeEmbedder:
    """Deterministic 3-D embeddings keyed on a topic word, so clustering is
    fully controllable: questions sharing a keyword get identical vectors."""

    _topics = {"vpn": [1, 0, 0], "book": [0, 1, 0], "access": [0, 0, 1]}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.array([0.3, 0.3, 0.3], dtype=float)
            for kw, base in self._topics.items():
                if kw in t.lower():
                    v = np.array(base, dtype=float)
            n = np.linalg.norm(v)
            out.append((v / n).tolist() if n else v.tolist())
        return out


# ---------------------------------------------------------------------------
# Repository: clustering + aggregation
# ---------------------------------------------------------------------------

def test_paraphrases_cluster_together(engine):
    repo = QuestionGapRepository(engine, embedder=FakeEmbedder())
    sid = uuid4()
    for q in ("how do I set up the vpn", "vpn not working", "connect to vpn"):
        repo.add(session_id=sid, query=q, kind="declined", language="english")

    clusters = repo.clusters()
    assert len(clusters) == 1
    assert clusters[0]["count"] == 3
    # every exact query is preserved for drill-down
    assert len(clusters[0]["examples"]) == 3


def test_distinct_topics_stay_separate(engine):
    repo = QuestionGapRepository(engine, embedder=FakeEmbedder())
    sid = uuid4()
    repo.add(session_id=sid, query="connect to vpn", kind="declined")
    repo.add(session_id=sid, query="book the microscope", kind="low_confidence")
    repo.add(session_id=sid, query="request access", kind="declined")

    clusters = repo.clusters()
    assert len(clusters) == 3
    assert all(c["count"] == 1 for c in clusters)


def test_clusters_ranked_by_size_with_kind_split(engine):
    repo = QuestionGapRepository(engine, embedder=FakeEmbedder())
    sid = uuid4()
    repo.add(session_id=sid, query="vpn setup", kind="declined")
    repo.add(session_id=sid, query="vpn broken", kind="low_confidence")
    repo.add(session_id=sid, query="book microscope", kind="declined")

    clusters = repo.clusters()
    # biggest cluster first
    assert clusters[0]["count"] == 2
    assert clusters[0]["kinds"] == {"low_confidence": 1, "declined": 1}
    assert repo.count() == 3


def test_no_embedder_falls_back_to_one_cluster_per_row(engine):
    repo = QuestionGapRepository(engine)  # no embedder
    sid = uuid4()
    for q in ("vpn setup", "vpn setup", "vpn setup"):
        repo.add(session_id=sid, query=q, kind="declined")
    # identical text, but without an embedder each row seeds its own cluster
    assert len(repo.clusters()) == 3


def test_tenant_scoping_isolates_clusters(engine):
    repo = QuestionGapRepository(engine, embedder=FakeEmbedder())
    t1, t2 = uuid4(), uuid4()
    sid = uuid4()
    repo.add(session_id=sid, query="vpn setup", kind="declined", tenant_id=t1)
    repo.add(session_id=sid, query="vpn setup", kind="declined", tenant_id=t2)
    # each tenant sees only its own row; clustering never crosses the boundary
    assert repo.count(tenant_id=t1) == 1
    assert repo.count(tenant_id=t2) == 1
    assert len(repo.clusters(tenant_id=t1)) == 1


def test_threshold_controls_merging(engine):
    # A keyword query ([1,0,0]) and a no-keyword one ([.577,.577,.577]) sit at
    # cosine ~0.577 — between the two thresholds below. So a looser bar merges
    # them into one topic; a stricter bar keeps them apart. This is the knob the
    # user tunes (GAP_CLUSTER_SIMILARITY). Separate tenants isolate the two runs.
    pair = ("vpn setup", "an unrelated everyday request")
    sid = uuid4()
    t_loose, t_strict = uuid4(), uuid4()

    loose = QuestionGapRepository(engine, embedder=FakeEmbedder(), similarity=0.5)
    strict = QuestionGapRepository(engine, embedder=FakeEmbedder(), similarity=0.6)
    for q in pair:
        loose.add(session_id=sid, query=q, kind="declined", tenant_id=t_loose)
        strict.add(session_id=sid, query=q, kind="declined", tenant_id=t_strict)

    assert len(loose.clusters(tenant_id=t_loose)) == 1     # merged by the looser bar
    assert len(strict.clusters(tenant_id=t_strict)) == 2   # kept apart by the stricter bar


# ---------------------------------------------------------------------------
# Demo seed
# ---------------------------------------------------------------------------

def test_seed_gaps_is_idempotent_and_deterministic(engine):
    ensure_demo_gaps(engine)
    first = demo_gap_count(engine)
    ensure_demo_gaps(engine)             # second call must no-op
    assert demo_gap_count(engine) == first > 0

    reset(engine)
    ensure_demo_gaps(engine)             # reseed from scratch
    assert demo_gap_count(engine) == first   # same deterministic total


def test_seed_gaps_top_cluster_and_shape(engine):
    seed_gaps(engine=engine)
    repo = QuestionGapRepository(engine)
    clusters = repo.clusters(tenant_id=DEMO_TENANT_ID, limit=20)
    assert clusters, "seed should produce clusters"
    top = clusters[0]
    assert top["label"] == "Booking instruments after hours"
    assert top["count"] == 14
    # endpoint contract: each cluster carries these fields
    for key in ("cluster_id", "label", "count", "kinds", "examples"):
        assert key in top


def test_reset_removes_gap_rows(engine):
    seed_gaps(engine=engine)
    assert demo_gap_count(engine) > 0
    reset(engine)
    assert demo_gap_count(engine) == 0


# ---------------------------------------------------------------------------
# Orchestrator hook — weak/declined turns get logged, confident ones don't
# ---------------------------------------------------------------------------

class _FakeCL:
    """Minimal ConversationLayer: always a question, echoes a corrected query."""

    def analyze(self, message, history=()):
        return AnalysisResult(
            language="english", type="question", corrected_query=message
        )


class _FakeAgent:
    """RAGAgent stand-in returning a pre-baked AnswerResult."""

    def __init__(self, result: AnswerResult):
        self._result = result

    def answer(self, *, message, language, history=(), retrieval_query=None):
        return self._result


def _assistant(engine, answer: AnswerResult) -> Assistant:
    return Assistant(
        conversation_layer=_FakeCL(),
        rag_agent=_FakeAgent(answer),
        session_repo=SessionRepository(engine),
        feedback_repo=FeedbackRepository(engine),
        question_gap_repo=QuestionGapRepository(engine),  # no embedder needed
    )


def test_declined_turn_logs_a_gap(engine):
    answer = AnswerResult(text="off domain", citations=[], declined=True,
                          retrieval_max_dense=0.2, retrieval_max_lexical=0.1)
    a = _assistant(engine, answer)
    a.handle(uuid4(), "how do I bake bread?")

    gaps = QuestionGapRepository(engine).clusters()
    assert len(gaps) == 1
    assert gaps[0]["kinds"]["declined"] == 1


def test_low_confidence_turn_logs_a_gap(engine):
    answer = AnswerResult(text="weak answer", citations=[], low_confidence=True)
    a = _assistant(engine, answer)
    a.handle(uuid4(), "something only loosely covered")

    gaps = QuestionGapRepository(engine).clusters()
    assert len(gaps) == 1
    assert gaps[0]["kinds"]["low_confidence"] == 1


def test_confident_answer_logs_no_gap(engine):
    answer = AnswerResult(text="a solid grounded answer", citations=[])
    a = _assistant(engine, answer)
    a.handle(uuid4(), "how do I clean the centrifuge?")

    assert QuestionGapRepository(engine).count() == 0


# ---------------------------------------------------------------------------
# Onboarding funnel — repository aggregation
# ---------------------------------------------------------------------------

def test_onboarding_excludes_veterans_and_groups_by_topic(engine):
    repo = QuestionGapRepository(engine)
    sid = uuid4()
    repo.add(session_id=sid, query="reset password", kind="declined",
             topic="access", tenure_days=2)
    repo.add(session_id=sid, query="vpn from home", kind="low_confidence",
             topic="access", tenure_days=6)
    repo.add(session_id=sid, query="book the scope", kind="declined",
             topic="booking", tenure_days=40)        # veteran -> excluded
    repo.add(session_id=sid, query="totally off topic", kind="declined",
             topic=None, tenure_days=1)               # newcomer, unclassified

    funnel = repo.onboarding(newcomer_days=14)
    assert funnel["newcomer_days"] == 14
    assert funnel["total"] == 3                        # veteran dropped
    topics = {t["topic"]: t for t in funnel["topics"]}
    assert topics["access"]["count"] == 2
    assert topics["access"]["kinds"] == {"low_confidence": 1, "declined": 1}
    assert "(unclassified)" in topics
    assert "booking" not in topics


def test_onboarding_window_is_tunable(engine):
    repo = QuestionGapRepository(engine)
    sid = uuid4()
    repo.add(session_id=sid, query="q1", kind="declined", topic="access", tenure_days=5)
    repo.add(session_id=sid, query="q2", kind="declined", topic="access", tenure_days=25)
    assert repo.onboarding(newcomer_days=7)["total"] == 1     # only the day-5 one
    assert repo.onboarding(newcomer_days=30)["total"] == 2    # both


def test_gaps_without_tenure_never_enter_funnel(engine):
    repo = QuestionGapRepository(engine)
    # A gap with no tenure recorded (e.g. anonymous/unresolved user) is a valid
    # documentation gap but must not appear in the onboarding funnel.
    repo.add(session_id=uuid4(), query="no tenure", kind="declined", topic="access")
    assert repo.count() == 1
    assert repo.onboarding(newcomer_days=14)["total"] == 0


# ---------------------------------------------------------------------------
# Onboarding funnel — orchestrator enrichment (topic + tenure on the row)
# ---------------------------------------------------------------------------

class _FakeAttr:
    """AttributionResolver stand-in: maps any question to a fixed topic."""

    def __init__(self, process="access", department="IT-Onboarding"):
        self._p, self._d = process, department

    def resolve_from_text(self, text):
        from attribution import AttributionResult, AttributionRow
        return AttributionResult(method="embedding", rows=[AttributionRow(
            source="01_onboarding.md", section=None,
            process=self._p, department=self._d, weight=1.0, method="embedding")])


def test_orchestrator_enriches_gap_with_topic_and_tenure(engine):
    from repositories import UserRepository
    users = UserRepository(engine)
    user = users.create(email="newbie@roche.com", password_hash="x")  # created now -> newcomer
    gaps = QuestionGapRepository(engine)
    a = Assistant(
        conversation_layer=_FakeCL(),
        rag_agent=_FakeAgent(AnswerResult(text="off domain", citations=[], declined=True)),
        session_repo=SessionRepository(engine),
        feedback_repo=FeedbackRepository(engine),
        attribution=_FakeAttr(process="access"),
        question_gap_repo=gaps,
        user_repo=users,
    )
    a.handle(uuid4(), "how do I get access?", user_id=str(user.id))

    funnel = gaps.onboarding(newcomer_days=14)
    assert funnel["total"] == 1
    assert funnel["topics"][0]["topic"] == "access"     # topic resolved via attribution
    # tenure was resolved (newcomer), so the row entered the funnel at all


# ---------------------------------------------------------------------------
# Onboarding funnel — demo seed
# ---------------------------------------------------------------------------

def test_seed_populates_onboarding_funnel(engine):
    ensure_demo_gaps(engine)
    funnel = QuestionGapRepository(engine).onboarding(
        newcomer_days=14, tenant_id=DEMO_TENANT_ID
    )
    assert funnel["total"] > 0
    topics = {t["topic"] for t in funnel["topics"]}
    # access (passwords/VPN) is seeded newcomer-heavy, so it must surface
    assert "access" in topics
