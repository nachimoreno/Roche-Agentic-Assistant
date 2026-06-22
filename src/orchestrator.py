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
from dataclasses import dataclass, field
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
from attribution import AttributionResolver
from conversation_layer import AnalysisResult, ConversationLayer
from db import FeedbackEntry
from incident_intake import IncidentDecision, IncidentIntake
from logging_setup import new_correlation_id
from repositories import FeedbackRepository, SessionRepository
from servicenow_tool import ServiceNowConfig, create_servicenow_incident


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response shape returned to the CLI / future HTTP layer.
# ---------------------------------------------------------------------------

@dataclass
class Response:
    text: str
    analysis: AnalysisResult
    citations: list[Citation]
    turn_id: Optional[UUID] = None      # the assistant turn, for rating
    follow_ups: list[str] = field(default_factory=list)
    low_confidence: bool = False        # weak grounding -> show a verify badge


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
    turn_id: Optional[UUID] = None      # the assistant turn, for rating
    follow_ups: list[str] = field(default_factory=list)
    low_confidence: bool = False        # weak grounding -> show a verify badge


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
        attribution: Optional[AttributionResolver] = None,
        incident_intake: Optional[IncidentIntake] = None,
        servicenow_config: Optional[ServiceNowConfig] = None,
        history_turns: int = 10,
    ) -> None:
        self._cl = conversation_layer
        self._agent = rag_agent
        self._sessions = session_repo
        self._feedback = feedback_repo
        # Resolves which doc/process a piece of feedback concerns. Optional so
        # the orchestrator still works (attribution simply skipped) if unwired.
        self._attribution = attribution
        # ServiceNow incident skill. Optional so the orchestrator still works
        # without it — an "incident"-classified turn then falls through to the
        # normal RAG answer (which explains the manual steps).
        self._incident = incident_intake
        self._sn_config = servicenow_config
        self._history_turns = history_turns

    def _history_for(self, session_id: UUID) -> list[AgentTurn]:
        """Recent turns as agent `Turn`s, for both classification and RAG.

        Call this BEFORE persisting the current user message so the result
        contains only prior turns. Both the conversation layer (routing) and
        the RAG agent consume the same window.
        """
        rows = self._sessions.recent_turns(session_id, n=self._history_turns)
        return [AgentTurn(role=t.role, content=t.content) for t in rows]

    # ------------------------------------------------------------------
    # Attribution helpers
    # ------------------------------------------------------------------

    def _cited_tuples(self, citations: list[Citation]):
        """(source, section, process, department) for each citation."""
        out = []
        for c in citations:
            process = department = None
            if self._attribution is not None:
                process, department = self._attribution.doc_meta(c.source)
            out.append((c.source, c.section, process, department))
        return out

    def _attribute_text_feedback(
        self, feedback_id: UUID, text: str, tenant_id: Optional[UUID]
    ) -> None:
        """Attribute volunteered (no-citation) feedback via the nearest doc."""
        if self._attribution is None:
            return
        result = self._attribution.resolve_from_text(text)
        self._feedback.replace_attributions(
            feedback_id, result.method, result.rows, tenant_id=tenant_id
        )

    # ------------------------------------------------------------------
    # Incident handling
    # ------------------------------------------------------------------

    def _render_incident(self, decision: IncidentDecision) -> str:
        """Turn an intake decision into the assistant's reply text.

        "file" actually creates the ServiceNow ticket (the tool returns the
        confirmation); "clarify"/"cancel" surface the model's reply unchanged.
        "not_incident" never reaches here — the caller falls back to RAG.
        """
        if decision.action == "file":
            return create_servicenow_incident(
                decision.short_description,
                description=decision.description,
                category=decision.category,
                urgency=decision.urgency,
                config=self._sn_config,
            )
        return decision.reply

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

        # Load prior turns BEFORE classifying so routing is context-aware:
        # a short reply like "yes, that's the one" only makes sense as a
        # continuation of the previous turn. Loaded here (before the current
        # user turn is written) so it already excludes the new message.
        history = self._history_for(session_id)
        analysis = self._cl.analyze(message, history=history)

        self._sessions.append_turn(
            session_id,
            role="user",
            content=message,
            language=analysis.language,
            tenant_id=tenant_id,
        )
        self._sessions.set_title_if_unset(session_id, _titleize(message))

        if analysis.type == "feedback":
            fb = self._feedback.add(
                FeedbackEntry(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    language=analysis.language,
                    emotion=analysis.emotion or "neutral",
                    message=message,
                )
            )
            self._attribute_text_feedback(fb.id, message, tenant_id)
            # If the feedback also embeds a real question, answer it (the
            # feedback is still logged above) instead of only acknowledging.
            if not analysis.contains_question:
                ack = _ack_for(analysis.language)
                ack_turn = self._sessions.append_turn(
                    session_id,
                    role="assistant",
                    content=ack,
                    language=analysis.language,
                    tenant_id=tenant_id,
                )
                return Response(
                    text=ack, analysis=analysis, citations=[], turn_id=ack_turn.id
                )

        # Incident path — the scientist is reporting an IT problem to log. The
        # intake step gates filing behind a confirmation; "not_incident" means
        # the classifier over-fired, so fall through to the normal answer.
        if analysis.type == "incident" and self._incident is not None:
            decision = self._incident.decide(
                message, history=history, language=analysis.language
            )
            if decision.action != "not_incident":
                text = self._render_incident(decision)
                incident_turn = self._sessions.append_turn(
                    session_id,
                    role="assistant",
                    content=text,
                    language=analysis.language,
                    tenant_id=tenant_id,
                )
                return Response(
                    text=text,
                    analysis=analysis,
                    citations=[],
                    turn_id=incident_turn.id,
                )

        # Question path — also handles feedback that embeds a question, and an
        # incident turn the intake declined to act on. Reuse the history already
        # loaded for classification.
        answer: AnswerResult = self._agent.answer(
            message=message,
            language=analysis.language,
            history=history,
            retrieval_query=analysis.corrected_query,
        )

        assistant_turn = self._sessions.append_turn(
            session_id,
            role="assistant",
            content=answer.text,
            language=analysis.language,
            tenant_id=tenant_id,
        )
        self._sessions.add_citations(
            assistant_turn.id,
            self._cited_tuples(answer.citations),
            tenant_id=tenant_id,
        )

        return Response(
            text=answer.text,
            analysis=analysis,
            citations=answer.citations,
            turn_id=assistant_turn.id,
            follow_ups=answer.follow_ups,
            low_confidence=answer.low_confidence,
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

        # Context-aware classification (see `handle`): load prior turns first.
        history = self._history_for(session_id)
        analysis = self._cl.analyze(message, history=history)
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
            fb = self._feedback.add(
                FeedbackEntry(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    language=analysis.language,
                    emotion=analysis.emotion or "neutral",
                    message=message,
                )
            )
            self._attribute_text_feedback(fb.id, message, tenant_id)
            # Pure feedback gets an acknowledgement; feedback that also embeds
            # a real question falls through to the streaming answer below (the
            # feedback is still logged above).
            if not analysis.contains_question:
                ack = _ack_for(analysis.language)
                yield StreamToken(text=ack)
                ack_turn = self._sessions.append_turn(
                    session_id,
                    role="assistant",
                    content=ack,
                    language=analysis.language,
                    tenant_id=tenant_id,
                )
                yield StreamDone(text=ack, citations=[], turn_id=ack_turn.id)
                return

        # Incident path (see `handle`). The reply is computed whole (a tool call
        # or a short confirmation question), so it streams as one token then the
        # terminal event — same shape as the feedback acknowledgement above.
        if analysis.type == "incident" and self._incident is not None:
            decision = self._incident.decide(
                message, history=history, language=analysis.language
            )
            if decision.action != "not_incident":
                text = self._render_incident(decision)
                yield StreamToken(text=text)
                incident_turn = self._sessions.append_turn(
                    session_id,
                    role="assistant",
                    content=text,
                    language=analysis.language,
                    tenant_id=tenant_id,
                )
                yield StreamDone(text=text, citations=[], turn_id=incident_turn.id)
                return

        full_text = ""
        citations: list[Citation] = []
        follow_ups: list[str] = []
        low_confidence = False
        try:
            for piece in self._agent.answer_stream(
                message=message,
                language=analysis.language,
                history=history,
                retrieval_query=analysis.corrected_query,
            ):
                if isinstance(piece, TextDelta):
                    full_text += piece.text
                    yield StreamToken(text=piece.text)
                elif isinstance(piece, AnswerComplete):
                    full_text = piece.text
                    citations = piece.citations
                    follow_ups = piece.follow_ups
                    low_confidence = piece.low_confidence
        except GeneratorExit:
            # Client disconnected mid-stream (Stop button, closed tab). The
            # client persists the partial answer it kept on screen via
            # POST /api/sessions/{id}/messages — persisting here too would
            # risk duplicates, so just log.
            logger.info(
                "turn.stream.aborted",
                extra={"session_id": str(session_id), "chars": len(full_text)},
            )
            raise

        assistant_turn = self._sessions.append_turn(
            session_id,
            role="assistant",
            content=full_text,
            language=analysis.language,
            tenant_id=tenant_id,
        )
        self._sessions.add_citations(
            assistant_turn.id,
            self._cited_tuples(citations),
            tenant_id=tenant_id,
        )
        yield StreamDone(
            text=full_text,
            citations=citations,
            turn_id=assistant_turn.id,
            follow_ups=follow_ups,
            low_confidence=low_confidence,
        )

    def record_rating(
        self,
        *,
        session_id: UUID,
        turn_id: UUID,
        rating: str,
        comment: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
    ) -> FeedbackEntry:
        """Record an explicit thumb up/down on an assistant answer.

        Validates the turn is an assistant turn in `session_id` (raises
        ValueError otherwise — the API maps that to 404). When a comment is
        present it is classified for emotion so explicit down-votes still carry
        sentiment; a classification hiccup degrades to "neutral" rather than
        dropping the rating. The write is idempotent per turn (see
        `FeedbackRepository.upsert_rating`).
        """
        if rating not in ("up", "down"):
            raise ValueError(f"rating must be 'up' or 'down', got {rating!r}")

        turn = self._sessions.get_turn(turn_id)
        if turn is None or turn.session_id != session_id or turn.role != "assistant":
            raise ValueError("no assistant turn to rate in this session")

        comment = (comment or "").strip() or None
        language = turn.language or "english"
        emotion = "neutral"
        if comment:
            try:
                analysis = self._cl.analyze(comment)
                emotion = analysis.emotion or "neutral"
            except Exception:
                logger.warning(
                    "feedback.rating.classify_failed", extra={"turn_id": str(turn_id)}
                )

        entry = self._feedback.upsert_rating(
            turn_id=turn_id,
            session_id=session_id,
            rating=rating,
            comment=comment,
            emotion=emotion,
            language=language,
            tenant_id=tenant_id,
        )

        # Attribute the rating to the doc(s)/process it concerns. Prefer the
        # deterministic citation join; fall back to embedding the comment (or
        # the answer itself) when the rated turn has no citations.
        if self._attribution is not None:
            cits = self._sessions.citations_for_turn(turn_id)
            if cits:
                result = self._attribution.resolve_from_citations(
                    [(c.source, c.section, c.process, c.department) for c in cits]
                )
            else:
                result = self._attribution.resolve_from_text(comment or turn.content)
            self._feedback.replace_attributions(
                entry.id, result.method, result.rows, tenant_id=tenant_id
            )
            entry.attribution_method = result.method   # reflect on the returned object

        return entry
