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
from typing import Literal, Optional

from pydantic import BaseModel

from llm import LLMClient


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

## EMOTION DETECTION

Only perform emotion detection when type = "feedback".
Use a single dominant emotion from the allowed list. If the signal is
weak or unclear, use "neutral".

## LANGUAGE DETECTION

Return the full language name in lowercase: english, german, french,
italian, spanish, or "other" if none of the above.

## OUTPUT

Return ONLY a JSON object. No prose, no markdown.
"""


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

    def analyze(self, message: str) -> AnalysisResult:
        payload = self._llm.complete_structured(
            system=_SYSTEM_PROMPT,
            user=message,
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
