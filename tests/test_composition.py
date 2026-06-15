"""
Tests for `main.build_source` — the DocumentSource factory that selects the
ingestion backend from Settings. Pure selection logic; no Drive/network calls
(GoogleDriveSource initializes its API client lazily).
"""

from __future__ import annotations

import pytest

from document_source import LocalMarkdownSource
from google_drive_source import GoogleDriveSource
from main import build_source
from settings import Settings


def _settings(**overrides) -> Settings:
    # Pin every field this suite asserts on. conftest's load_dotenv() pushes
    # the developer's real .env into os.environ, so without explicit values
    # here the tests would inherit live config (e.g. DOCUMENT_SOURCE) and fail.
    # Init kwargs outrank env vars in pydantic-settings, so this is hermetic.
    base = {
        "groq_api_key": "test-key",
        "document_source": "local",
        "docs_path": "data/docs",
        "drive_folder_id": None,
        "google_service_account_json": None,
        "google_oauth_credentials": None,
    }
    base.update(overrides)
    return Settings(**base)


def test_defaults_to_local_markdown_source():
    src = build_source(_settings())
    assert isinstance(src, LocalMarkdownSource)


def test_local_source_uses_configured_docs_path():
    src = build_source(_settings(document_source="local", docs_path="some/dir"))
    assert isinstance(src, LocalMarkdownSource)
    # as_posix() so the assertion holds on Windows (backslash) too.
    assert src._path.as_posix() == "some/dir"


def test_google_drive_source_is_selected_and_configured():
    src = build_source(
        _settings(
            document_source="google_drive",
            drive_folder_id="folder-123",
            drive_recursive=False,
            google_service_account_json="sa.json",
        )
    )
    assert isinstance(src, GoogleDriveSource)
    assert src.folder_id == "folder-123"
    assert src.recursive is False
    assert src._creds_path == "sa.json"


def test_google_drive_prefers_service_account_over_oauth():
    src = build_source(
        _settings(
            document_source="google_drive",
            drive_folder_id="folder-123",
            google_service_account_json="sa.json",
            google_oauth_credentials="oauth.json",
        )
    )
    assert src._creds_path == "sa.json"


def test_google_drive_falls_back_to_oauth_credentials():
    src = build_source(
        _settings(
            document_source="google_drive",
            drive_folder_id="folder-123",
            google_oauth_credentials="oauth.json",
        )
    )
    assert src._creds_path == "oauth.json"


def test_google_drive_without_folder_id_raises():
    with pytest.raises(ValueError, match="drive_folder_id"):
        build_source(_settings(document_source="google_drive", drive_folder_id=None))
