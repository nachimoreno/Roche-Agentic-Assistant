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

from dotenv import load_dotenv

from agent import RAGAgent
from conversation_layer import ConversationLayer
from db import create_all, make_engine, new_id
from document_source import LocalMarkdownSource
from embeddings import SentenceTransformersProvider
from llm import GroqClient
from logging_setup import setup_logging
from orchestrator import Assistant
from repositories import FeedbackRepository, SessionRepository
from retrieval import DocumentStore
from settings import Settings
from vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)


def build_assistant(settings: Settings) -> Assistant:
    # Ensure data directory exists for SQLite.
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = make_engine(settings.database_url)
    create_all(engine)

    llm = GroqClient(api_key=settings.groq_api_key, model=settings.model_name)
    embedder = SentenceTransformersProvider(model_name=settings.embedding_model)
    vector_store = ChromaVectorStore(
        path=settings.chroma_path,
        collection_name=f"roche_{embedder.name.replace('/', '_')}",
    )
    source = LocalMarkdownSource(path=settings.docs_path)

    docs = DocumentStore(
        source=source,
        embedder=embedder,
        vector_store=vector_store,
        manifest_path=f"{settings.chroma_path}/manifest.json",
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
    agent = RAGAgent(document_store=docs, llm=llm, top_k=settings.top_k)
    session_repo = SessionRepository(engine)
    feedback_repo = FeedbackRepository(engine)

    return Assistant(
        conversation_layer=cl,
        rag_agent=agent,
        session_repo=session_repo,
        feedback_repo=feedback_repo,
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
    setup_logging(level=settings.log_level, fmt=settings.log_format)

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
