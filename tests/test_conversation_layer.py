"""
Conversation-layer behavior tests.

These hit the real Groq API — marked `live` so they're skipped by default
and only run with `pytest -m live`. Conftest also auto-skips if
GROQ_API_KEY is unset.
"""

from __future__ import annotations

import pytest

from conversation_layer import ConversationLayer
from llm import GroqClient
from settings import Settings


pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def layer():
    settings = Settings()
    return ConversationLayer(
        llm=GroqClient(api_key=settings.groq_api_key, model=settings.model_name)
    )


# ---------------------------------------------------------------------------
# Questions — language + type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message, expected_language",
    [
        ("How do I clean the centrifuge?", "english"),
        ("Wie kann ich den Probenbestand prüfen?", "german"),
        ("Comment puis-je réserver un instrument?", "french"),
        ("Come posso richiedere l'accesso a un'applicazione?", "italian"),
    ],
)
def test_question_classification(layer, message, expected_language):
    result = layer.analyze(message)
    assert result.type == "question"
    assert result.language == expected_language
    assert result.emotion is None


def test_spell_correction_fixes_typos(layer):
    # Heavy typos should be corrected so retrieval gets a clean query.
    result = layer.analyze("How do I repot an incidnet?")
    assert result.corrected_query
    low = result.corrected_query.lower()
    assert "report" in low and "incident" in low


# ---------------------------------------------------------------------------
# Feedback — language + type + emotion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message, expected_language, plausible_emotions",
    [
        (
            "This onboarding documentation is extremely confusing.",
            "english",
            {"confused", "frustrated", "annoyed", "irritated"},
        ),
        (
            "Das System funktioniert hervorragend, ich bin sehr zufrieden.",
            "german",
            {"pleased", "satisfied", "happy", "impressed", "appreciative"},
        ),
        (
            "Je ne comprends pas du tout cette procédure, c'est très frustrant.",
            "french",
            {"confused", "frustrated", "irritated", "annoyed"},
        ),
        (
            "Sono molto preoccupato per i continui errori di accesso.",
            "italian",
            {"concerned", "anxious", "frustrated", "stressed"},
        ),
    ],
)
def test_feedback_classification(
    layer, message, expected_language, plausible_emotions
):
    result = layer.analyze(message)
    assert result.type == "feedback"
    assert result.language == expected_language
    assert result.emotion in plausible_emotions, (
        f"unexpected emotion {result.emotion!r} for {message!r}"
    )
