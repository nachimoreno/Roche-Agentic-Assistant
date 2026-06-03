"""
conversation_layer.py
---------------------
Conversation Layer for the Roche Scientist Assistant pipeline.
Uses Groq (free tier) with Llama 3 — no payment required.

Get a free API key at: https://console.groq.com

Responsibilities:
  - Detect the language of an incoming scientist message
  - Classify the message as "question" or "feedback"
  - Detect the dominant emotion when the message is feedback

Usage
-----
    export GROQ_API_KEY="gsk_..."          # set once in your terminal

    from conversation_layer import ConversationLayer

    layer = ConversationLayer()
    result = layer.analyze("How do I clean the centrifuge?")
    print(result.to_json())
    # {"language": "english", "type": "question"}

    result = layer.analyze("This onboarding process is extremely frustrating.")
    print(result.to_json())
    # {"language": "english", "type": "feedback", "emotion": "frustrated"}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "llama-3.3-70b-versatile"   # Free on Groq — fast and accurate
MAX_TOKENS = 256                      # Classification output is always short

_SYSTEM_PROMPT = """\
You are part of an enterprise conversational AI pipeline for a scientific assistant system used by Roche scientists.

Your role is NOT to behave like a chatbot assistant.

Your role is to behave as a structured conversational analysis engine inside a larger AI architecture.

---

## SYSTEM CONTEXT

The architecture works as follows:

1. A scientist sends a message through a conversational interface.
2. The message is forwarded through an API call to an LLM-based conversation layer.
3. Your responsibility inside this layer is ONLY to:
   * detect the message language
   * classify the message type
   * detect emotion when applicable

4. Your JSON response will be consumed programmatically by downstream services, including:
   * routing systems
   * feedback pipelines
   * analytics dashboards
   * orchestration layers
   * future standalone models

5. Your output must therefore be:
   * deterministic
   * machine-readable
   * strictly structured
   * consistent across calls

6. You are NOT responsible for:
   * answering the user's question
   * generating conversational replies
   * giving explanations
   * adding commentary
   * performing reasoning outside classification

You are ONLY responsible for conversational analysis and structured output generation.

---

## OBJECTIVES

For every incoming user message, you must:

1. Detect the language of the message.
2. Classify the message type:
   * "question"
   * "feedback"
3. If the type is "feedback":
   * Detect the primary emotion/sentiment expressed.

---

## PIPELINE LOGIC

STEP 1: Analyze the user message.
STEP 2: Detect the primary language.
STEP 3: Classify whether the message is a question OR feedback.
STEP 4: If "feedback" — detect the dominant emotion.
STEP 5: Return a STRICT JSON response.

---

## MESSAGE TYPE CLASSIFICATION

A message is a "question" if the user:
* asks for information
* requests guidance
* asks where or how to do something
* asks operational or scientific questions
* seeks clarification

A message is "feedback" if the user:
* expresses an opinion
* reports frustration or confusion
* comments on a tool, process, workflow, or document
* praises or criticizes something
* suggests improvements
* reports usability issues
* expresses emotions regarding an experience

---

## EMOTION DETECTION

ONLY perform emotion detection when "type" = "feedback".

The emotion label must be:
* concise
* lowercase
* a single dominant emotional state

Valid examples: frustrated, confused, satisfied, annoyed, angry, disappointed,
pleased, overwhelmed, concerned, anxious, happy, neutral, skeptical, uncertain,
impressed, stressed, irritated, appreciative.

If the emotional signal is weak or unclear, use: "neutral"

---

## LANGUAGE DETECTION

Return the FULL language name in lowercase.
Examples: english, german, french, italian, spanish.
If another language appears, return its full lowercase name.

---

## OUTPUT FORMAT

Return ONLY valid JSON. No markdown, no comments, no extra fields.

If type = "question":
{"language": "<full_language_name>", "type": "question"}

If type = "feedback":
{"language": "<full_language_name>", "type": "feedback", "emotion": "<detected_emotion>"}

---

## IMPORTANT CONSTRAINTS

* Output must always be deterministic.
* Never invent fields.
* Never return null values.
* Never explain reasoning.
* Never output text outside JSON.
* Use lowercase labels only.
* If uncertain between "question" and "feedback":
  classify as "feedback" only if emotional or opinionated language is present.
* If emotion is unclear: use "neutral"

---

## EXAMPLES

Input: "How do I clean the centrifuge?"
Output: {"language": "english", "type": "question"}

Input: "This documentation is extremely confusing."
Output: {"language": "english", "type": "feedback", "emotion": "confused"}

Input: "Me gusta mucho el nuevo sistema."
Output: {"language": "spanish", "type": "feedback", "emotion": "satisfied"}

Input: "Das System funktioniert hervorragend."
Output: {"language": "german", "type": "feedback", "emotion": "pleased"}

Input: "Je ne comprends pas cette procédure."
Output: {"language": "french", "type": "feedback", "emotion": "confused"}

Input: "This process is stressing me out."
Output: {"language": "english", "type": "feedback", "emotion": "stressed"}
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Structured result produced by the conversation layer for one message."""

    language: str
    type: str                       # "question" | "feedback"
    emotion: Optional[str] = None   # present only when type == "feedback"

    def to_dict(self) -> dict:
        """Return the pipeline-ready JSON payload."""
        payload = {"language": self.language, "type": self.type}
        if self.emotion is not None:
            payload["emotion"] = self.emotion
        return payload

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string ready for downstream consumption."""
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class ConversationLayer:
    """
    Stateless conversation analysis engine for the Roche Scientist Assistant.

    Uses Groq's free API with Llama 3.3 70B.
    Get a free key at: https://console.groq.com

    Parameters
    ----------
    api_key : str, optional
        Groq API key. Falls back to the GROQ_API_KEY environment variable.
    model : str
        Groq model to use. Defaults to llama-3.3-70b-versatile (free tier).
    max_tokens : int
        Maximum tokens for the model response.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = Groq(
            api_key=api_key or os.environ.get("GROQ_API_KEY")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, message: str) -> AnalysisResult:
        """
        Classify a scientist message.

        Parameters
        ----------
        message : str
            Raw text sent by the scientist.

        Returns
        -------
        AnalysisResult
            Typed result with language, type, and optional emotion.

        Raises
        ------
        ValueError
            If the model returns unparseable or schema-invalid output.
        groq.APIError
            Propagated on any non-recoverable API error.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,          # deterministic output
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": message},
            ],
        )

        raw_text = (response.choices[0].message.content or "").strip()

        payload = self._parse(raw_text)
        self._validate(payload)

        return AnalysisResult(
            language=payload["language"],
            type=payload["type"],
            emotion=payload.get("emotion"),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse(self, raw_text: str) -> dict:
        """Parse model output as JSON, tolerating accidental markdown fences."""
        text = raw_text
        if text.startswith("```"):
            lines = text.splitlines()
            inner = []
            for line in lines[1:]:
                if line.strip() == "```":
                    break
                inner.append(line)
            text = "\n".join(inner)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"ConversationLayer received non-JSON output.\n"
                f"Raw: {raw_text!r}\n"
                f"Error: {exc}"
            ) from exc

    def _validate(self, payload: dict) -> None:
        """Raise ValueError if required fields are absent or values are invalid."""
        missing = [f for f in ("language", "type") if f not in payload]
        if missing:
            raise ValueError(
                f"ConversationLayer: missing required fields {missing} — got {payload}"
            )
        if payload["type"] not in ("question", "feedback"):
            raise ValueError(
                f"ConversationLayer: invalid type {payload['type']!r}; "
                "expected 'question' or 'feedback'"
            )
        if payload["type"] == "feedback" and "emotion" not in payload:
            raise ValueError(
                f"ConversationLayer: 'emotion' missing for feedback message — got {payload}"
            )


# ---------------------------------------------------------------------------
# CLI smoke-test  (python conversation_layer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    layer = ConversationLayer(API_KEY)
    print(f"Roche Scientist Assistant - Conversation Layer")
    print(f"Model : {layer.model}")
    print(f"Type your message and press Enter. Type 'exit' to quit.")
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
            print(f"OUT : {r.to_json()}")
        except ValueError as exc:
            print(f"ERR : {exc}", file=sys.stderr)
        print("-" * 70)
