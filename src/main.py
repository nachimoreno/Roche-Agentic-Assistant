"""
main.py
-------
CLI entry point — a REPL that drives the `Assistant` for one session.

Wires together every component using `Settings` for configuration, then
loops on stdin until the user types `exit`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import Engine

from agent import RAGAgent
from attribution import AttributionResolver
from conversation_layer import ConversationLayer
from db import create_all, make_engine, new_id
from document_source import CompositeSource, DocumentSource, LocalMarkdownSource
from google_drive_source import GoogleDriveSource
from embeddings import FastEmbedProvider
from incident_intake import IncidentIntake
from lexical_index import BM25Index
from llm import GroqClient
from logging_setup import setup_logging
from orchestrator import Assistant
from repositories import (
    FeedbackRepository,
    QuestionGapRepository,
    SessionRepository,
    UserRepository,
)
from retrieval import DocumentStore
from servicenow_tool import ServiceNowConfig
from settings import Settings
from vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)


def build_drive_source(settings: Settings) -> GoogleDriveSource:
    """Construct the Google Drive source from settings (caller checks config)."""
    return GoogleDriveSource(
        folder_id=settings.drive_folder_id,
        credentials_path=(
            settings.google_service_account_json
            or settings.google_oauth_credentials
        ),
        recursive=settings.drive_recursive,
    )


def build_source(settings: Settings) -> DocumentSource:
    """Select the DocumentSource(s) to ingest from based on settings.

    Every implementation yields the same `SourceDocument` shape, so nothing
    downstream of here cares which one is used. "all" combines local markdown
    with Google Drive via `CompositeSource`, including Drive only when it's
    configured so the default degrades cleanly to local-only.
    """
    local = LocalMarkdownSource(path=settings.docs_path)

    if settings.document_source == "local":
        return local

    if settings.document_source == "google_drive":
        if not settings.drive_folder_id:
            raise ValueError(
                "document_source='google_drive' requires drive_folder_id "
                "(set DRIVE_FOLDER_ID in .env)."
            )
        return build_drive_source(settings)

    # "all": local always, plus Drive when configured.
    if settings.drive_folder_id:
        return CompositeSource([local, build_drive_source(settings)])

    logger.warning(
        "source.all.drive_unconfigured",
        extra={"detail": "document_source='all' but no drive_folder_id; using local only"},
    )
    return local


def build_engine(settings: Settings) -> Engine:
    """Create the DB engine and ensure the schema exists.

    Exposed so the HTTP layer can share one engine across the assistant and
    the auth/user repositories rather than opening a second connection pool.
    """
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = make_engine(settings.database_url)
    create_all(engine)
    return engine


def build_assistant(settings: Settings, engine: Optional[Engine] = None) -> Assistant:
    engine = engine or build_engine(settings)

    llm = GroqClient(api_key=settings.groq_api_key, model=settings.model_name)
    # Fail fast: reject a bad/missing key now, before the slow ingest below,
    # rather than on the user's first message.
    llm.check_auth()
    logger.info("startup.auth.ok", extra={"model": settings.model_name})

    embedder = FastEmbedProvider(model_name=settings.embedding_model)
    vector_store = ChromaVectorStore(
        path=settings.chroma_path,
        collection_name=f"roche_{embedder.name.replace('/', '_')}",
    )
    source = build_source(settings)

    lexical = BM25Index() if settings.retrieval_mode == "hybrid" else None
    docs = DocumentStore(
        source=source,
        embedder=embedder,
        vector_store=vector_store,
        manifest_path=f"{settings.chroma_path}/manifest.json",
        lexical_index=lexical,
    )
    report = docs.ingest()
    logger.info(
        "startup.ingest",
        extra={
            "documents_seen": report.documents_seen,
            "documents_reindexed": report.documents_reindexed,
            "chunks_written": report.chunks_written,
        },
    )

    cl = ConversationLayer(llm=llm)
    agent = RAGAgent(
        document_store=docs,
        llm=llm,
        top_k=settings.top_k,
        min_dense=settings.retrieval_min_dense,
        min_lexical=settings.retrieval_min_lexical,
        warn_dense=settings.retrieval_warn_dense,
    )
    session_repo = SessionRepository(engine)
    feedback_repo = FeedbackRepository(engine)
    user_repo = UserRepository(engine)
    attribution = AttributionResolver(docs)
    # Reuse the same embedder that powers retrieval to cluster gap questions —
    # no second model load, and clustering shares the corpus's vector space.
    question_gap_repo = QuestionGapRepository(engine, embedder=embedder)

    incident_intake = IncidentIntake(llm=llm)
    servicenow_config = ServiceNowConfig(
        use_mock=settings.servicenow_use_mock,
        instance=settings.servicenow_instance,
        username=settings.servicenow_username,
        password=settings.servicenow_password,
    )

    return Assistant(
        conversation_layer=cl,
        rag_agent=agent,
        session_repo=session_repo,
        feedback_repo=feedback_repo,
        attribution=attribution,
        incident_intake=incident_intake,
        servicenow_config=servicenow_config,
        question_gap_repo=question_gap_repo,
        user_repo=user_repo,
    )


def _print_response(resp) -> None:
    print(f"\nAssistant ({resp.analysis.language}): {resp.text}")
    if resp.citations:
        print("\nSources:")
        for cite in resp.citations:
            print(f"  - {cite.source} §{cite.section}")
    if resp.analysis.type == "feedback":
        print(f"\n[recorded feedback — emotion: {resp.analysis.emotion}]")
    print("-" * 70)


def main() -> int:
    load_dotenv()
    settings = Settings()
    setup_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        log_file=settings.log_file,
    )

    assistant = build_assistant(settings)
    session_id = new_id()

    print(f"Roche Scientist Assistant")
    print(f"Model    : {settings.model_name}")
    print(f"Session  : {session_id}")
    print(f"Database : {settings.database_url}")
    print(f"Type a message and press Enter. Type 'exit' to quit.")
    print("-" * 70)

    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not msg:
            continue
        if msg.lower() == "exit":
            print("Goodbye.")
            return 0
        try:
            resp = assistant.handle(session_id, msg)
            _print_response(resp)
        except Exception as exc:
            print(f"ERR: {exc}", file=sys.stderr)
            logger.exception("turn.error")


if __name__ == "__main__":
    raise SystemExit(main())
