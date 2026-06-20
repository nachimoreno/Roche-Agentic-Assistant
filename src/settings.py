"""
settings.py
-----------
Typed configuration for the Roche Scientist Assistant.

Single source of config truth. Every module that needs a value takes a
`Settings` (or one of its fields) via constructor injection — no module
reads `os.environ` directly.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM ----------------------------------------------------------
    groq_api_key: str
    model_name: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # ---- Persistence --------------------------------------------------
    database_url: str = "sqlite:///data/app.db"

    # ---- Retrieval ----------------------------------------------------
    chroma_path: str = ".chroma"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 4
    # "hybrid" fuses dense embeddings with BM25 keyword search (better on exact
    # tokens — codes, part numbers, app names). "dense" uses embeddings only.
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    # Off-domain guardrail. A query is declined deterministically (no LLM call,
    # no citations) only when the top dense cosine < retrieval_min_dense AND the
    # top BM25 score < retrieval_min_lexical.
    #
    # The two signals play different roles (see scripts/calibrate_guardrail.py,
    # which produced these defaults against the lab-ops corpus):
    #   * Dense cosine is the discriminator — it separates in/out of corpus with
    #     a clean gap (measured: in-domain >= 0.51, off-domain <= 0.26), so 0.40
    #     sits safely in the middle.
    #   * BM25 is a poor off-domain signal (non-English in-domain queries score
    #     ~0 against the English corpus; off-domain English queries pick up a few
    #     points from common tokens, measured max ~4.7). Its only job is to
    #     RESCUE a low-dense query with a strong exact-keyword hit (a part number
    #     / SOP code), so 8.0 is a bar set a clear margin above that ~4.7 noise.
    # Both must be below to decline, so dense effectively carries the gate and
    # lexical only ever rescues. In dense-only mode max lexical is 0.0, so the
    # lexical check is inert and only the dense threshold applies.
    # NOTE: re-run the calibration script against the real Drive corpus before
    # production — BM25 scores in particular scale with corpus size. Override
    # either value via .env (RETRIEVAL_MIN_DENSE / RETRIEVAL_MIN_LEXICAL).
    retrieval_min_dense: float = 0.40
    retrieval_min_lexical: float = 8.0

    # ---- Document source ----------------------------------------------
    # Which DocumentSource to ingest from. "google_drive" (default) pulls from
    # a Drive folder — the production source of truth. "local" reads markdown
    # from docs_path; "all" ingests from both (only adds Drive when
    # drive_folder_id is set, so it degrades to local-only if unconfigured).
    document_source: Literal["local", "google_drive", "all"] = "google_drive"
    docs_path: str = "tests/fixtures/docs"
    # Google Drive (only used when document_source == "google_drive").
    drive_folder_id: Optional[str] = None
    drive_recursive: bool = True
    google_service_account_json: Optional[str] = None
    google_oauth_credentials: Optional[str] = None

    # ---- HTTP server -----------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    # ---- Auth ---------------------------------------------------------
    # Signs the session cookie. MUST be overridden in production.
    session_secret: str = "dev-insecure-change-me"
    session_cookie: str = "roche_session"
    # Only send the cookie over HTTPS — set true in production.
    session_https_only: bool = False
    min_password_length: int = 8
    # Comma-separated emails promoted to the "admin" role on login (analytics
    # dashboard access). Kept as a plain string so .env stays simple; use
    # `admin_email_set` to consume it.
    admin_emails: str = ""

    @property
    def admin_email_set(self) -> set[str]:
        return {
            e.strip().lower() for e in self.admin_emails.split(",") if e.strip()
        }

    # ---- Observability ------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"
    # If set, logs are written to this file and the console stays clean.
    # If unset, logs go to stderr (handy for development).
    log_file: Optional[str] = "logs/app.log"
