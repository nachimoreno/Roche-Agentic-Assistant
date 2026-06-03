"""
orchestrator.py
---------------
The `Assistant` ties every component together:

    user message
        -> ConversationLayer.analyze()        (language, type, emotion)
        -> route on type:
             feedback -> FeedbackRepository.add(...)
             question -> RAGAgent.answer(...)
        -> persist both user + assistant turns

Chat history lives in the DB (not in memory), so cross-device session
continuity already works in this MVP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from agent import AnswerResult, Citation, RAGAgent, Turn as AgentTurn
from conversation_layer import AnalysisResult, ConversationLayer
from db import FeedbackEntry
from logging_setup import new_correlation_id
from repositories import FeedbackRepository, SessionRepository


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response shape returned to the CLI / future HTTP layer.
# ---------------------------------------------------------------------------

@dataclass
class Response:
    text: str
    analysis: AnalysisResult
    citations: list[Citation]


# ---------------------------------------------------------------------------
# Acknowledgements for feedback turns, in the user's language.
# ---------------------------------------------------------------------------

_FEEDBACK_ACK = {
    "english": "Thank you for the feedback. I've passed it on to the IT team.",
    "german":  "Vielen Dank für das Feedback. Ich habe es an das IT-Team weitergeleitet.",
    "french":  "Merci pour votre retour. Je l'ai transmis à l'équipe informatique.",
    "italian": "Grazie per il feedback. L'ho inoltrato al team IT.",
    "spanish": "Gracias por tus comentarios. Los he enviado al equipo de TI.",
}


def _ack_for(language: str) -> str:
    return _FEEDBACK_ACK.get(language, _FEEDBACK_ACK["english"])


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------

class Assistant:
    def __init__(
        self,
        *,
        conversation_layer: ConversationLayer,
        rag_agent: RAGAgent,
        session_repo: SessionRepository,
        feedback_repo: FeedbackRepository,
        history_turns: int = 10,
    ) -> None:
        self._cl = conversation_layer
        self._agent = rag_agent
        self._sessions = session_repo
        self._feedback = feedback_repo
        self._history_turns = history_turns

    def handle(
        self,
        session_id: UUID,
        message: str,
        *,
        tenant_id: Optional[UUID] = None,
    ) -> Response:
        new_correlation_id()
        logger.info(
            "turn.start",
            extra={"session_id": str(session_id), "msg_chars": len(message)},
        )

        self._sessions.get_or_create(session_id, tenant_id=tenant_id)

        analysis = self._cl.analyze(message)

        self._sessions.append_turn(
            session_id,
            role="user",
            content=message,
            language=analysis.language,
            tenant_id=tenant_id,
        )

        if analysis.type == "feedback":
            self._feedback.add(
                FeedbackEntry(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    language=analysis.language,
                    emotion=analysis.emotion or "neutral",
                    message=message,
                )
            )
            ack = _ack_for(analysis.language)
            self._sessions.append_turn(
                session_id,
                role="assistant",
                content=ack,
                language=analysis.language,
                tenant_id=tenant_id,
            )
            return Response(text=ack, analysis=analysis, citations=[])

        # Question path — load recent history (excluding the user turn we
        # just wrote, which is the same as `message`) and run RAG.
        history_rows = self._sessions.recent_turns(session_id, n=self._history_turns)
        history = [
            AgentTurn(role=t.role, content=t.content)
            for t in history_rows[:-1]   # drop the just-appended user turn
        ]

        answer: AnswerResult = self._agent.answer(
            message=message,
            language=analysis.language,
            history=history,
        )

        self._sessions.append_turn(
            session_id,
            role="assistant",
            content=answer.text,
            language=analysis.language,
            tenant_id=tenant_id,
        )

        return Response(
            text=answer.text,
            analysis=analysis,
            citations=answer.citations,
        )
