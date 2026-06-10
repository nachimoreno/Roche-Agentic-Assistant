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
from typing import Iterator, Optional, Union
from uuid import UUID

from agent import (
    AnswerComplete,
    AnswerResult,
    Citation,
    RAGAgent,
    TextDelta,
    Turn as AgentTurn,
)
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
# Streaming events — what `handle_stream` yields, transport-agnostic.
# ---------------------------------------------------------------------------

@dataclass
class StreamMeta:
    """First event: classification of the incoming message."""

    analysis: AnalysisResult


@dataclass
class StreamToken:
    """A delta of assistant text to append in the UI."""

    text: str


@dataclass
class StreamDone:
    """Terminal event: the full text (already persisted) and citations."""

    text: str
    citations: list[Citation]


StreamEvent = Union[StreamMeta, StreamToken, StreamDone]


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


def _titleize(message: str, *, max_len: int = 60) -> str:
    """Derive a short session title from the first user message."""
    text = " ".join(message.split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


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
        user_id: Optional[str] = None,
    ) -> Response:
        new_correlation_id()
        logger.info(
            "turn.start",
            extra={"session_id": str(session_id), "msg_chars": len(message)},
        )

        self._sessions.get_or_create(session_id, tenant_id=tenant_id, user_id=user_id)

        analysis = self._cl.analyze(message)

        self._sessions.append_turn(
            session_id,
            role="user",
            content=message,
            language=analysis.language,
            tenant_id=tenant_id,
        )
        self._sessions.set_title_if_unset(session_id, _titleize(message))

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

    def handle_stream(
        self,
        session_id: UUID,
        message: str,
        *,
        tenant_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
    ) -> Iterator[StreamEvent]:
        """Streaming counterpart of `handle`.

        Yields a `StreamMeta`, then `StreamToken`s, then a terminal
        `StreamDone`. The user turn is persisted up front; the assistant turn
        is persisted once the full answer has streamed (before `StreamDone`),
        so persistence semantics match the non-streaming path.
        """
        new_correlation_id()
        logger.info(
            "turn.stream.start",
            extra={"session_id": str(session_id), "msg_chars": len(message)},
        )

        self._sessions.get_or_create(session_id, tenant_id=tenant_id, user_id=user_id)

        analysis = self._cl.analyze(message)
        self._sessions.append_turn(
            session_id,
            role="user",
            content=message,
            language=analysis.language,
            tenant_id=tenant_id,
        )
        self._sessions.set_title_if_unset(session_id, _titleize(message))
        yield StreamMeta(analysis=analysis)

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
            yield StreamToken(text=ack)
            self._sessions.append_turn(
                session_id,
                role="assistant",
                content=ack,
                language=analysis.language,
                tenant_id=tenant_id,
            )
            yield StreamDone(text=ack, citations=[])
            return

        history_rows = self._sessions.recent_turns(session_id, n=self._history_turns)
        history = [
            AgentTurn(role=t.role, content=t.content)
            for t in history_rows[:-1]   # drop the just-appended user turn
        ]

        full_text = ""
        citations: list[Citation] = []
        for piece in self._agent.answer_stream(
            message=message,
            language=analysis.language,
            history=history,
        ):
            if isinstance(piece, TextDelta):
                full_text += piece.text
                yield StreamToken(text=piece.text)
            elif isinstance(piece, AnswerComplete):
                full_text = piece.text
                citations = piece.citations

        self._sessions.append_turn(
            session_id,
            role="assistant",
            content=full_text,
            language=analysis.language,
            tenant_id=tenant_id,
        )
        yield StreamDone(text=full_text, citations=citations)
