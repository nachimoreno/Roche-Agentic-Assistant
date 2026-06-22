"""
Orchestrator routing for ServiceNow incidents (end-to-end, no Groq).

A fake `LLMClient` drives both the classifier (returns type="incident") and the
intake step (returns a chosen action), and a fake RAG agent stands in for the
answer path. Together they prove the wiring in `Assistant.handle` /
`handle_stream`:

  file         -> a (mock) ServiceNow ticket is created and confirmed
  clarify      -> the proposal/question is returned, NO ticket filed
  cancel       -> the acknowledgement is returned, NO ticket filed
  not_incident -> falls through to the normal RAG answer
  (unwired)    -> an incident turn with no intake also falls through to RAG
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from agent import AnswerComplete, AnswerResult, TextDelta
from conversation_layer import ConversationLayer
from db import create_all, make_engine, new_id
from incident_intake import IncidentIntake
from mock_servicenow_client import MockServiceNowClient
from orchestrator import Assistant, StreamDone, StreamMeta, StreamToken
from repositories import FeedbackRepository, SessionRepository
from servicenow_tool import ServiceNowConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class RoutingFakeLLM:
    """Serves the classifier and the intake from canned payloads, keyed by the
    schema each consumer passes (AnalysisResult vs. IncidentDecision)."""

    def __init__(self, *, analysis: dict[str, Any], decision: dict[str, Any]) -> None:
        self._analysis = analysis
        self._decision = decision

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        history: Sequence[dict[str, str]] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        title = schema.get("title", "")
        if "AnalysisResult" in title:
            return self._analysis
        if "IncidentDecision" in title:
            return self._decision
        raise AssertionError(f"unexpected schema in routing test: {title!r}")

    def stream_text(self, *args, **kwargs):  # pragma: no cover - unused here
        raise NotImplementedError

    def check_auth(self) -> None:  # pragma: no cover - unused here
        pass


class FakeAgent:
    """Stand-in RAG agent — records whether the answer path was taken."""

    RAG_TEXT = "To report a broken device, open a ticket via the IT portal."

    def __init__(self) -> None:
        self.answer_calls = 0

    def answer(self, *, message, language, history=(), retrieval_query=None):
        self.answer_calls += 1
        return AnswerResult(text=self.RAG_TEXT)

    def answer_stream(self, *, message, language, history=(), retrieval_query=None):
        self.answer_calls += 1
        yield TextDelta(text=self.RAG_TEXT)
        yield AnswerComplete(text=self.RAG_TEXT, citations=[])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    eng = make_engine("sqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def reset_mock_servicenow():
    """Fresh ticket counter/history per test so INC numbers are deterministic."""
    MockServiceNowClient.reset()
    yield
    MockServiceNowClient.reset()


@pytest.fixture
def make_assistant(engine):
    def _make(*, analysis: dict, decision: dict, wire_incident: bool = True):
        llm = RoutingFakeLLM(analysis=analysis, decision=decision)
        agent = FakeAgent()
        assistant = Assistant(
            conversation_layer=ConversationLayer(llm=llm),
            rag_agent=agent,
            session_repo=SessionRepository(engine),
            feedback_repo=FeedbackRepository(engine),
            incident_intake=IncidentIntake(llm=llm) if wire_incident else None,
            servicenow_config=ServiceNowConfig(use_mock=True),
        )
        return assistant, agent
    return _make


INCIDENT_ANALYSIS = {"language": "english", "type": "incident"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_file_creates_mock_ticket_and_confirms(make_assistant):
    assistant, agent = make_assistant(
        analysis=INCIDENT_ANALYSIS,
        decision={
            "action": "file",
            "short_description": "Centrifuge not responding in Lab 4B",
            "category": "hardware",
            "urgency": "1",
        },
    )
    resp = assistant.handle(new_id(), "yes, go ahead and file it")

    assert resp.analysis.type == "incident"
    assert "✅" in resp.text
    assert "INC0010001" in resp.text          # first mock ticket
    assert "simulated" in resp.text           # mock-mode note
    assert agent.answer_calls == 0            # RAG path NOT taken
    assert len(MockServiceNowClient.get_all_tickets()) == 1


def test_clarify_proposes_without_filing(make_assistant):
    reply = "I can open a ticket for the Lab 4B centrifuge — shall I file it?"
    assistant, agent = make_assistant(
        analysis=INCIDENT_ANALYSIS,
        decision={"action": "clarify", "reply": reply},
    )
    resp = assistant.handle(new_id(), "the centrifuge in lab 4b is dead")

    assert resp.text == reply
    assert agent.answer_calls == 0
    assert MockServiceNowClient.get_all_tickets() == []


def test_cancel_acknowledges_without_filing(make_assistant):
    reply = "No problem — I won't open a ticket."
    assistant, agent = make_assistant(
        analysis=INCIDENT_ANALYSIS,
        decision={"action": "cancel", "reply": reply},
    )
    resp = assistant.handle(new_id(), "no, don't bother")

    assert resp.text == reply
    assert MockServiceNowClient.get_all_tickets() == []


def test_not_incident_falls_back_to_rag(make_assistant):
    assistant, agent = make_assistant(
        analysis=INCIDENT_ANALYSIS,
        decision={"action": "not_incident"},
    )
    resp = assistant.handle(new_id(), "how do I report a broken centrifuge?")

    assert agent.answer_calls == 1
    assert resp.text == FakeAgent.RAG_TEXT
    assert MockServiceNowClient.get_all_tickets() == []


def test_unwired_incident_intake_falls_back_to_rag(make_assistant):
    # Defensive: if intake is not wired, an "incident" classification must still
    # produce a useful answer (the manual steps) rather than dropping the turn.
    assistant, agent = make_assistant(
        analysis=INCIDENT_ANALYSIS,
        decision={"action": "file"},   # ignored — no intake to consult
        wire_incident=False,
    )
    resp = assistant.handle(new_id(), "the freezer is alarming")

    assert agent.answer_calls == 1
    assert resp.text == FakeAgent.RAG_TEXT
    assert MockServiceNowClient.get_all_tickets() == []


def test_streaming_file_emits_confirmation_and_persists(make_assistant):
    assistant, agent = make_assistant(
        analysis=INCIDENT_ANALYSIS,
        decision={
            "action": "file",
            "short_description": "Cannot log into BioLIMS",
            "category": "access",
            "urgency": "2",
        },
    )
    events = list(assistant.handle_stream(new_id(), "yes file it"))

    assert isinstance(events[0], StreamMeta)
    assert events[0].analysis.type == "incident"
    assert any(isinstance(e, StreamToken) and "INC0010001" in e.text for e in events)

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert "✅" in done.text
    assert done.turn_id is not None
    assert agent.answer_calls == 0
    assert len(MockServiceNowClient.get_all_tickets()) == 1
