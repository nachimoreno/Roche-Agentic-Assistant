"""
Tests for `main.build_source` — the DocumentSource factory that selects the
ingestion backend from Settings. Pure selection logic; no Drive/network calls
(GoogleDriveSource initializes its API client lazily).
"""

from __future__ import annotations

import pytest

from document_source import CompositeSource, LocalMarkdownSource
from google_drive_source import GoogleDriveSource
from main import build_source
from preprocessing_source import PreprocessingSource
from settings import Settings


def _drive(src):
    """Unwrap the preprocessing decorator (on by default) to the Drive source.

    These tests exercise *source selection* — backend, credentials, recursion —
    which is orthogonal to whether documents get preprocessed, so they look
    through the `PreprocessingSource` wrapper to the underlying source.
    """
    return src._inner if isinstance(src, PreprocessingSource) else src


@pytest.fixture(autouse=True)
def _isolate_source_env(monkeypatch):
    # conftest's load_dotenv() pushes the dev .env into os.environ; pydantic
    # reads it even with _env_file=None, so a local DOCUMENT_SOURCE=local would
    # break the default-selection tests. Clear source vars for a clean baseline.
    for var in (
        "DOCUMENT_SOURCE",
        "DRIVE_FOLDER_ID",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_OAUTH_CREDENTIALS",
    ):
        monkeypatch.delenv(var, raising=False)


def _settings(**overrides) -> Settings:
    # Establish a clean, fully-specified baseline so selection logic is
    # deterministic regardless of the developer's real .env — conftest calls
    # load_dotenv(), so Drive credentials there would otherwise bleed into
    # os.environ and these unit tests. Each test overrides only what it needs.
    base = {
        "groq_api_key": "test-key",
        "drive_folder_id": None,
        "google_service_account_json": None,
        "google_oauth_credentials": None,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_defaults_to_google_drive():
    # Default is now Drive-only — production ingests from Google Drive — and
    # preprocessing is on by default, so the Drive source comes back wrapped.
    src = build_source(
        _settings(drive_folder_id="folder-123", google_service_account_json="sa.json")
    )
    assert isinstance(src, PreprocessingSource)
    assert isinstance(_drive(src), GoogleDriveSource)
    assert _drive(src).folder_id == "folder-123"


def test_preprocess_disabled_returns_raw_drive_source():
    # With preprocessing off, build_source returns the bare GoogleDriveSource.
    src = build_source(
        _settings(
            drive_folder_id="folder-123",
            google_service_account_json="sa.json",
            preprocess_documents=False,
        )
    )
    assert isinstance(src, GoogleDriveSource)
    assert src.folder_id == "folder-123"


def test_all_combines_local_and_drive():
    # "all" still combines local + Drive when Drive is configured; only the
    # Drive child is preprocessed (local markdown is already structured).
    src = build_source(
        _settings(
            document_source="all",
            drive_folder_id="folder-123",
            google_service_account_json="sa.json",
        )
    )
    assert isinstance(src, CompositeSource)
    kinds = [type(s).__name__ for s in src._sources]
    assert kinds == ["LocalMarkdownSource", "PreprocessingSource"]
    assert isinstance(_drive(src._sources[1]), GoogleDriveSource)


def test_all_without_drive_falls_back_to_local():
    # "all" but no drive_folder_id → degrades to local-only, no error.
    src = build_source(_settings(document_source="all", drive_folder_id=None))
    assert isinstance(src, LocalMarkdownSource)


def test_explicit_local_source_selected():
    src = build_source(_settings(document_source="local"))
    assert isinstance(src, LocalMarkdownSource)


def test_local_source_uses_configured_docs_path():
    src = build_source(_settings(document_source="local", docs_path="some/dir"))
    assert isinstance(src, LocalMarkdownSource)
    assert str(src._path) == "some/dir"


def test_google_drive_source_is_selected_and_configured():
    src = build_source(
        _settings(
            document_source="google_drive",
            drive_folder_id="folder-123",
            drive_recursive=False,
            google_service_account_json="sa.json",
        )
    )
    assert isinstance(_drive(src), GoogleDriveSource)
    assert _drive(src).folder_id == "folder-123"
    assert _drive(src).recursive is False
    assert _drive(src)._creds_path == "sa.json"


def test_google_drive_prefers_service_account_over_oauth():
    src = build_source(
        _settings(
            document_source="google_drive",
            drive_folder_id="folder-123",
            google_service_account_json="sa.json",
            google_oauth_credentials="oauth.json",
        )
    )
    assert _drive(src)._creds_path == "sa.json"


def test_google_drive_falls_back_to_oauth_credentials():
    src = build_source(
        _settings(
            document_source="google_drive",
            drive_folder_id="folder-123",
            google_oauth_credentials="oauth.json",
        )
    )
    assert _drive(src)._creds_path == "oauth.json"


def test_google_drive_without_folder_id_raises():
    with pytest.raises(ValueError, match="drive_folder_id"):
        build_source(_settings(document_source="google_drive", drive_folder_id=None))
