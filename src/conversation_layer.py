"""
conversation_layer.py
---------------------
Conversation Layer for the Roche Scientist Assistant pipeline.

Responsibilities:
  - Detect the language of an incoming scientist message
  - Classify the message as "question" or "feedback"
  - Detect the dominant emotion when the message is feedback

Output is a Pydantic `AnalysisResult` — the single source of truth for
language / message-type / emotion across the rest of the system.

Usage
-----
    from llm import GroqClient
    from conversation_layer import ConversationLayer

    layer = ConversationLayer(llm=GroqClient(api_key=..., model=...))
    result = layer.analyze("How do I clean the centrifuge?")
    print(result.model_dump())
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Protocol, Sequence

from pydantic import BaseModel

from llm import LLMClient


class _HistoryTurn(Protocol):
    """Minimal shape the classifier needs from a prior chat turn."""

    role: str       # "user" | "assistant"
    content: str


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema — single source of truth for downstream consumers
# ---------------------------------------------------------------------------

Language = Literal[
    "english",
    "german",
    "french",
    "italian",
    "spanish",
    "other",
]

MessageType = Literal["question", "feedback"]

Emotion = Literal[
    "frustrated",
    "confused",
    "satisfied",
    "annoyed",
    "angry",
    "disappointed",
    "pleased",
    "overwhelmed",
    "concerned",
    "anxious",
    "happy",
    "neutral",
    "skeptical",
    "uncertain",
    "impressed",
    "stressed",
    "irritated",
    "appreciative",
]


class AnalysisResult(BaseModel):
    language: Language
    type: MessageType
    emotion: Optional[Emotion] = None
    # Only meaningful when type == "feedback": True if the message ALSO contains
    # an answerable operational question (e.g. "this is confusing — how do I
    # request access?"). Lets the orchestrator answer and log feedback at once.
    contains_question: bool = False
    # The latest message with spelling mistakes / typos fixed (meaning, language
    # and intent preserved). Used as the retrieval query so heavy typos don't
    # starve retrieval and wrongly trip the off-domain guardrail. Downstream
    # falls back to the original message when this is absent.
    corrected_query: Optional[str] = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are part of an enterprise conversational AI pipeline for a scientific
assistant system used by Roche scientists.

Your role is NOT to behave like a chatbot. Your role is to behave as a
structured conversational analysis engine inside a larger AI architecture.

For every incoming user message, you must:

1. Detect the primary language of the message.
2. Classify the message as "question" or "feedback".
3. If the type is "feedback", detect the dominant emotion.

## MESSAGE TYPE CLASSIFICATION

A message is a "question" if the user asks for information, requests
guidance, asks where or how to do something, asks operational or
scientific questions, or seeks clarification.

A message is "feedback" if the user expresses an opinion, reports
frustration or confusion, comments on a tool / process / workflow /
document, praises or criticizes something, suggests improvements,
reports usability issues, or expresses emotions about an experience.

If uncertain between "question" and "feedback", classify as "feedback"
only if emotional or opinionated language is present.

## CONVERSATION CONTEXT

You may be given the recent conversation under "CONVERSATION SO FAR".
Use it to interpret the LATEST message — short replies often only make
sense in context.

If the assistant's previous turn asked something or offered a
suggestion and the latest message confirms, corrects, answers, or
follows up on it (e.g. "yes", "that's the one", "no, I meant the other
one", "what about the rotor?"), classify the latest message as a
"question" — it continues the information request.

Still classify the latest message as "feedback" when it is genuinely an
opinion, emotion, praise, complaint, or a comment about a tool /
document / process — even mid-conversation. A continuation does not
override clearly opinionated or emotional language.

When the conversation is absent or empty, classify the message on its
own.

## EMOTION DETECTION

Only perform emotion detection when type = "feedback".
Use a single dominant emotion from the allowed list. If the signal is
weak or unclear, use "neutral".

## EMBEDDED QUESTION DETECTION

Set "contains_question" to true ONLY when type = "feedback" AND the same
message also asks an answerable operational question — for example
"this onboarding is so confusing, how do I request access?" (a complaint
plus a real question). Set it to false for pure feedback with no question
("this tool is terrible"), and leave it false whenever type = "question".

## LANGUAGE DETECTION

Return the full language name in lowercase: english, german, french,
italian, spanish, or "other" if none of the above.

## SPELL CORRECTION

Also return "corrected_query": the LATEST message rewritten with spelling
mistakes and typos fixed, preserving the original meaning, language, and
intent. Only fix clear typos — do NOT answer it, translate it, rephrase it,
expand abbreviations, or add or remove words. If there are no mistakes, return
the message unchanged.

## OUTPUT

Return ONLY a JSON object. No prose, no markdown.
"""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _build_user_prompt(
    message: str,
    history: Optional[Sequence[_HistoryTurn]],
) -> str:
    """Render the recent history (if any) plus the latest message.

    The latest message is labelled explicitly so the classifier always
    knows which turn it must classify, even with history present.
    """
    if not history:
        return message

    lines = ["CONVERSATION SO FAR:"]
    for turn in history:
        speaker = "Scientist" if turn.role == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.content}")
    lines.append("")
    lines.append(f"LATEST MESSAGE (classify this one):\n{message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class ConversationLayer:
    """Stateless conversation analysis engine.

    The LLM provider is injected via the `LLMClient` interface — this class
    has no knowledge of Groq, Anthropic, or any specific vendor.
    """

    def __init__(self, llm: LLMClient, max_tokens: int = 256) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def analyze(
        self,
        message: str,
        history: Optional[Sequence[_HistoryTurn]] = None,
    ) -> AnalysisResult:
        payload = self._llm.complete_structured(
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(message, history),
            schema=AnalysisResult.model_json_schema(),
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        result = AnalysisResult.model_validate(payload)
        logger.info(
            "classification.done",
            extra={
                "language": result.language,
                "type": result.type,
                "emotion": result.emotion,
            },
        )
        return result


# ---------------------------------------------------------------------------
# CLI smoke-test  (python -m conversation_layer)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    from logging_setup import setup_logging
    from llm import GroqClient
    from settings import Settings

    load_dotenv()
    settings = Settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    layer = ConversationLayer(
        llm=GroqClient(api_key=settings.groq_api_key, model=settings.model_name),
    )

    print(f"Roche Scientist Assistant — Conversation Layer")
    print(f"Model : {settings.model_name}")
    print(f"Type a message and press Enter. Type 'exit' to quit.")
    print("-" * 70)

    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not msg:
            continue
        if msg.lower() == "exit":
            print("Goodbye.")
            break
        try:
            r = layer.analyze(msg)
            print(f"OUT : {r.model_dump_json()}")
        except Exception as exc:
            print(f"ERR : {exc}", file=sys.stderr)
        print("-" * 70)
