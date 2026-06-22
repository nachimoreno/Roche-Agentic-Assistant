"""
Tests for the ServiceNow incident-intake step (incident_intake.py).

A fake `LLMClient` returns canned decisions so these run with no Groq
dependency — they verify the intake validates the structured payload into an
`IncidentDecision` and threads the conversation + reply language into the
prompt it sends the model.
"""

from __future__ import annotations

from typing import Any, Sequence

from incident_intake import IncidentDecision, IncidentIntake


class FakeLLMClient:
    """Returns a fixed structured payload and records the prompts it received."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"system": system, "user": user, "schema": schema})
        return self._payload

    def stream_text(self, *args, **kwargs):  # pragma: no cover - unused here
        raise NotImplementedError

    def check_auth(self) -> None:  # pragma: no cover - unused here
        pass


class _Turn:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


def test_decide_returns_clarify_with_reply():
    llm = FakeLLMClient({
        "action": "clarify",
        "short_description": "Centrifuge not responding in Lab 4B",
        "category": "hardware",
        "urgency": "2",
        "reply": "I can open a ticket for the Lab 4B centrifuge — shall I file it?",
    })
    decision = IncidentIntake(llm=llm).decide("the centrifuge in lab 4b won't turn on")

    assert isinstance(decision, IncidentDecision)
    assert decision.action == "clarify"
    assert "centrifuge" in decision.reply.lower()


def test_decide_file_carries_extracted_fields():
    llm = FakeLLMClient({
        "action": "file",
        "short_description": "Cannot log into BioLIMS",
        "description": "Password rejected since this morning",
        "category": "access",
        "urgency": "1",
    })
    decision = IncidentIntake(llm=llm).decide(
        "yes please file it",
        history=[_Turn("assistant", "Shall I open a ServiceNow ticket?")],
    )

    assert decision.action == "file"
    assert decision.short_description == "Cannot log into BioLIMS"
    assert decision.category == "access"
    assert decision.urgency == "1"


def test_decide_applies_schema_defaults_when_fields_omitted():
    # "not_incident" / "cancel" need no incident fields — the schema defaults
    # (hardware / urgency 3 / empty reply) must fill in cleanly.
    llm = FakeLLMClient({"action": "not_incident"})
    decision = IncidentIntake(llm=llm).decide("how do I report a broken centrifuge?")

    assert decision.action == "not_incident"
    assert decision.category == "hardware"
    assert decision.urgency == "3"
    assert decision.reply == ""


def test_prompt_includes_history_and_reply_language():
    llm = FakeLLMClient({"action": "cancel", "reply": "No problem."})
    IncidentIntake(llm=llm).decide(
        "no don't bother",
        history=[_Turn("assistant", "Shall I open a ticket for the freezer?")],
        language="german",
    )

    user_prompt = llm.calls[0]["user"]
    assert "REPLY LANGUAGE: german" in user_prompt
    assert "freezer" in user_prompt          # history threaded in
    assert "no don't bother" in user_prompt  # latest message labelled
