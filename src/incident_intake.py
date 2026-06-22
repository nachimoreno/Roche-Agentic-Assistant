"""
incident_intake.py
------------------
The ServiceNow incident "skill" — the conversational layer between a scientist
reporting a problem and `servicenow_tool.create_servicenow_incident`.

WHERE IT FITS:
  The conversation classifier (conversation_layer.py) tags a turn as
  type="incident" when the scientist reports an IT problem they want acted on.
  The orchestrator then hands that turn to `IncidentIntake.decide`, which uses
  one structured LLM call to choose what to do next:

    file         -> the user has CONFIRMED; orchestrator files the ticket
    clarify      -> propose/confirm or ask for a missing detail (no ticket yet)
    cancel       -> the user declined; acknowledge, file nothing
    not_incident -> not really a fileable incident; orchestrator falls back to
                    the normal RAG answer path (so a misroute is harmless)

WHY A SECOND LLM STEP (not native tool-calling):
  The repo's GroqClient has no native function-calling, and we never want to
  file a ticket the scientist did not confirm. A dedicated extract-and-decide
  step gives us a deterministic confirmation gate and clean field extraction,
  mirroring how ConversationLayer already turns a message into a typed result.
  The provider seam (LLMClient) is unchanged.

CONFIRM-BEFORE-FILE:
  First mention of a problem -> "clarify" (the assistant proposes the ticket it
  would open and asks the scientist to confirm). Only once the scientist agrees
  does the next turn come back as "file". The model is told never to jump
  straight to "file" on the first mention.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Protocol, Sequence

from pydantic import BaseModel

from llm import LLMClient


class _HistoryTurn(Protocol):
    """Minimal shape the intake needs from a prior chat turn."""

    role: str       # "user" | "assistant"
    content: str


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision schema — what the LLM returns
# ---------------------------------------------------------------------------

IncidentAction = Literal["file", "clarify", "cancel", "not_incident"]


class IncidentDecision(BaseModel):
    """Structured outcome of one intake turn.

    `short_description` / `description` / `category` / `urgency` are only
    meaningful for action == "file"; they mirror the fields in
    servicenow_tool.SERVICENOW_TOOL_DEFINITION. `reply` carries the message to
    show the scientist for "clarify" and "cancel" (it is ignored for "file",
    where the tool produces the confirmation, and for "not_incident", where the
    RAG agent answers instead).
    """

    action: IncidentAction
    short_description: str = ""
    description: str = ""
    category: Literal["hardware", "software", "access", "network"] = "hardware"
    urgency: Literal["1", "2", "3"] = "3"
    reply: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the incident-intake step of an enterprise assistant for Roche
scientists. A previous classifier already decided the latest message is about
an IT incident (a broken / failing / inaccessible device, instrument, system,
application, network, or access issue). Your job is to decide how to handle it
and to extract the fields needed to open a ServiceNow incident.

You return ONE action:

- "file": choose this ONLY when the scientist has clearly CONFIRMED they want
  the ticket opened — typically because the assistant's previous turn proposed
  opening one and the latest message agrees ("yes", "go ahead", "please file").
  NEVER choose "file" on the first mention of a problem.

- "clarify": choose this on the FIRST clear report of a problem, or when a
  needed detail is missing. Put in "reply" a short, friendly message that
  states the incident you would open (summary, category, urgency) and asks the
  scientist to confirm, OR asks the one piece of missing information (e.g. which
  instrument, or whether it is blocking their work). Do not file anything yet.

- "cancel": the scientist declined to open a ticket ("no", "don't bother",
  "never mind"). Put a brief acknowledgement in "reply". File nothing.

- "not_incident": the message is actually a general question or not a fileable
  problem (for example "how do I report a broken centrifuge?" is a question, not
  an incident). The main assistant will answer it instead.

## FIELD EXTRACTION (for "clarify" and "file")

- short_description: a concise one-line summary IN ENGLISH (ServiceNow content
  is English), e.g. "Centrifuge not responding in Lab 4B". Include the location
  or asset if the scientist gave one.
- description: any fuller detail the scientist provided, in their words. May be
  empty.
- category: one of hardware, software, access, network. Pick the best fit
  (a physical instrument/device = hardware; an application or software error =
  software; login/permission problems = access; connectivity = network).
- urgency: "1" High (work is blocked / safety or sample risk), "2" Medium,
  "3" Low. If unclear, use "3" and consider asking in "clarify".

## REPLY LANGUAGE

Write "reply" in the scientist's language (given below). Keep it to one or two
sentences.

## OUTPUT

Return ONLY a JSON object. No prose, no markdown.
"""


def _build_user_prompt(
    message: str,
    history: Optional[Sequence[_HistoryTurn]],
    language: str,
) -> str:
    """Render recent history + the latest message + the target reply language."""
    lines: list[str] = []
    if history:
        lines.append("CONVERSATION SO FAR:")
        for turn in history:
            speaker = "Scientist" if turn.role == "user" else "Assistant"
            lines.append(f"{speaker}: {turn.content}")
        lines.append("")
    lines.append(f"REPLY LANGUAGE: {language}")
    lines.append(f"LATEST MESSAGE (handle this one):\n{message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class IncidentIntake:
    """Stateless incident-intake engine.

    Like ConversationLayer, the LLM provider is injected via `LLMClient`; this
    class has no knowledge of Groq or any specific vendor, and does no filing
    itself — it only decides and extracts. The orchestrator performs the actual
    ServiceNow call so persistence and tool config stay in one place.
    """

    def __init__(self, llm: LLMClient, max_tokens: int = 384) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def decide(
        self,
        message: str,
        *,
        history: Optional[Sequence[_HistoryTurn]] = None,
        language: str = "english",
    ) -> IncidentDecision:
        payload = self._llm.complete_structured(
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(message, history, language),
            schema=IncidentDecision.model_json_schema(),
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        decision = IncidentDecision.model_validate(payload)
        logger.info(
            "incident.intake.decided",
            extra={
                "action": decision.action,
                "category": decision.category,
                "urgency": decision.urgency,
            },
        )
        return decision
