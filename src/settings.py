"""
settings.py
-----------
Typed configuration for the Roche Scientist Assistant.

Single source of config truth. Every module that needs a value takes a
`Settings` (or one of its fields) via constructor injection — no module
reads `os.environ` directly.
"""

from __future__ import annotations

from typing import Literal

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
    docs_path: str = "data/docs"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 4

    # ---- Observability ------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"
