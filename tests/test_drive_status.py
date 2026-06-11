"""
tests/test_drive_status.py
--------------------------
Tests for `api._report_drive_status` — the startup probe that tells the
operator on the console whether the Google Drive integration came up, and
why it didn't. No real Drive calls; `build_drive_source` is stubbed.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import api
from settings import Settings


def _settings(**overrides) -> Settings:
    # Clean baseline isolated from the developer's real .env (conftest calls
    # load_dotenv()), so each test declares exactly the Drive config it needs.
    base = {
        "groq_api_key": "test-key",
        "drive_folder_id": None,
        "google_service_account_json": None,
        "google_oauth_credentials": None,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _messages(caplog) -> str:
    return " ".join(r.getMessage() for r in caplog.records)


def test_local_source_reports_disabled(caplog):
    caplog.set_level(logging.INFO, logger="roche.startup")
    api._report_drive_status(_settings(document_source="local"))
    assert "disabled" in _messages(caplog)


def test_all_without_folder_reports_skipped(caplog):
    caplog.set_level(logging.INFO, logger="roche.startup")
    api._report_drive_status(_settings(document_source="all", drive_folder_id=None))
    assert "skipped" in _messages(caplog)


def test_success_reports_ok_with_folder_and_count(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="roche.startup")
    fake = MagicMock()
    fake.check_connection.return_value = 5
    monkeypatch.setattr(api, "build_drive_source", lambda s: fake)

    api._report_drive_status(
        _settings(
            document_source="all",
            drive_folder_id="folder-1",
            google_service_account_json="sa.json",
        )
    )

    msgs = _messages(caplog)
    assert "OK" in msgs
    assert "folder-1" in msgs
    assert "5" in msgs


def test_failure_reports_reason(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="roche.startup")
    fake = MagicMock()
    fake.check_connection.side_effect = FileNotFoundError("secrets/key.json")
    monkeypatch.setattr(api, "build_drive_source", lambda s: fake)

    api._report_drive_status(
        _settings(
            document_source="google_drive",
            drive_folder_id="folder-1",
            google_service_account_json="sa.json",
        )
    )

    msgs = _messages(caplog)
    assert "FAILED" in msgs
    assert "FileNotFoundError" in msgs   # the exception type
    assert "secrets/key.json" in msgs    # the underlying reason
