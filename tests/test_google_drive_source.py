"""
tests/test_google_drive_source.py
----------------------------------
Fast tests (no real Drive calls) + live tests against the real API.

Run fast suite:
    pytest tests/test_google_drive_source.py

Run live tests (requires GOOGLE_SERVICE_ACCOUNT_JSON + DRIVE_FOLDER_ID in .env):
    pytest tests/test_google_drive_source.py -m live
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Adjust import path if needed when placed inside the repo
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google_drive_source import (
    GoogleDriveSource,
    _deduplicate,
    _infer_title,
    _parse_time,
)


# ---------------------------------------------------------------------------
# Fast unit tests — no Drive API calls
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_keeps_single_file(self):
        files = [{"name": "SOP.md", "modifiedTime": "2024-01-01T00:00:00Z", "id": "1"}]
        assert _deduplicate(files) == files

    def test_keeps_newest_when_duplicate_names(self):
        files = [
            {"name": "SOP.md", "modifiedTime": "2024-01-01T00:00:00Z", "id": "old"},
            {"name": "SOP.md", "modifiedTime": "2024-06-01T00:00:00Z", "id": "new"},
        ]
        result = _deduplicate(files)
        assert len(result) == 1
        assert result[0]["id"] == "new"

    def test_case_insensitive_name_matching(self):
        files = [
            {"name": "SOP.md",   "modifiedTime": "2024-01-01T00:00:00Z", "id": "1"},
            {"name": "sop.md",   "modifiedTime": "2024-06-01T00:00:00Z", "id": "2"},
        ]
        result = _deduplicate(files)
        assert len(result) == 1

    def test_different_names_both_kept(self):
        files = [
            {"name": "SOP.md",       "modifiedTime": "2024-01-01T00:00:00Z", "id": "1"},
            {"name": "Onboarding.md","modifiedTime": "2024-01-01T00:00:00Z", "id": "2"},
        ]
        assert len(_deduplicate(files)) == 2


class TestInferTitle:
    def test_uses_h1_from_content(self):
        content = "# Lab Cleaning Procedure\n\nSome text."
        assert _infer_title("doc.md", content) == "Lab Cleaning Procedure"

    def test_falls_back_to_filename_without_extension(self):
        assert _infer_title("onboarding_guide.md", "no heading here") == "onboarding_guide"

    def test_strips_pdf_extension(self):
        assert _infer_title("procedure.pdf", "") == "procedure"

    def test_no_extension(self):
        assert _infer_title("readme", "content") == "readme"


class TestParseTime:
    def test_parses_drive_iso_format(self):
        dt = _parse_time("2024-03-15T10:30:00.000Z")
        assert dt.year == 2024
        assert dt.month == 3

    def test_empty_string_returns_now(self):
        dt = _parse_time("")
        assert isinstance(dt, datetime)

    def test_result_is_timezone_aware(self):
        dt = _parse_time("2024-01-01T00:00:00Z")
        assert dt.tzinfo is not None


class TestGoogleDriveSourceListDocuments:
    """Mock the Drive API service to test list_documents without real calls."""

    def _make_source(self, folder_id="test-folder"):
        src = GoogleDriveSource(folder_id=folder_id, credentials_path=None)
        return src

    def _mock_service(self, files: list[dict]):
        """Return a mock Drive service that yields the given file list."""
        service = MagicMock()
        service.files().list().execute.return_value = {
            "files": files,
            "nextPageToken": None,
        }
        # Make chained .list(q=..., ...) also return the mock
        service.files().list.return_value.execute.return_value = {
            "files": files,
            "nextPageToken": None,
        }
        return service

    def test_yields_source_documents(self):
        src = self._make_source()
        src._service = self._mock_service([
            {
                "id": "abc123",
                "name": "onboarding.md",
                "mimeType": "text/plain",
                "modifiedTime": "2024-06-01T00:00:00Z",
                "subfolder": "",
            }
        ])
        src._download_text = lambda meta: "# Onboarding Guide\n\nStep 1: ..."

        docs = list(src.list_documents())
        assert len(docs) == 1
        assert docs[0].id == "abc123"
        assert docs[0].title == "Onboarding Guide"
        assert "onboarding" in docs[0].content.lower()
        assert docs[0].metadata["source"] == "google_drive"

    def test_subfolder_tracked_in_metadata(self):
        src = self._make_source()
        # Mock _list_files directly since the service mock bypasses subfolder injection
        src._list_files = lambda folder_id, subfolder_name="": [
            {
                "id": "xyz",
                "name": "cleaning_sop.md",
                "mimeType": "text/plain",
                "modifiedTime": "2024-06-01T00:00:00Z",
                "subfolder": "Lab Procedures",
            }
        ]
        src._download_text = lambda meta: "# Cleaning SOP\n\nUse 70% ethanol."

        docs = list(src.list_documents())
        assert docs[0].metadata["subfolder"] == "Lab Procedures"

    def test_skips_empty_documents(self):
        src = self._make_source()
        src._service = self._mock_service([
            {"id": "empty", "name": "empty.md", "mimeType": "text/plain", "modifiedTime": "2024-01-01T00:00:00Z", "subfolder": ""},
        ])
        src._download_text = lambda meta: "   "

        docs = list(src.list_documents())
        assert docs == []

    def test_deduplication_applied(self):
        src = self._make_source()
        src._service = self._mock_service([
            {"id": "old", "name": "SOP.md", "mimeType": "text/plain", "modifiedTime": "2024-01-01T00:00:00Z", "subfolder": ""},
            {"id": "new", "name": "SOP.md", "mimeType": "text/plain", "modifiedTime": "2024-12-01T00:00:00Z", "subfolder": ""},
        ])
        src._download_text = lambda meta: "# SOP\n\nContent."

        docs = list(src.list_documents())
        assert len(docs) == 1
        assert docs[0].id == "new"


# ---------------------------------------------------------------------------
# Live integration tests — skipped unless credentials present
# ---------------------------------------------------------------------------

LIVE = pytest.mark.skipif(
    not os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or not os.getenv("DRIVE_FOLDER_ID"),
    reason="Live Drive tests require GOOGLE_SERVICE_ACCOUNT_JSON + DRIVE_FOLDER_ID env vars",
)


@LIVE
def test_live_list_documents_returns_results():
    src = GoogleDriveSource(folder_id=os.environ["DRIVE_FOLDER_ID"])
    docs = list(src.list_documents())
    assert len(docs) > 0, "Expected at least one document in the test Drive folder"
    print(f"\nLive test: found {len(docs)} documents")
    for d in docs:
        print(f"  - {d.title} ({len(d.content)} chars, modified {d.modified_at.date()})")


@LIVE
def test_live_documents_have_content():
    src = GoogleDriveSource(folder_id=os.environ["DRIVE_FOLDER_ID"])
    docs = list(src.list_documents())
    for doc in docs:
        assert doc.content.strip(), f"Document '{doc.title}' has no content"
        assert doc.id, "Document missing id"
        assert doc.title, "Document missing title"


@LIVE
def test_live_no_duplicates():
    src = GoogleDriveSource(folder_id=os.environ["DRIVE_FOLDER_ID"])
    docs = list(src.list_documents())
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids)), "Duplicate document IDs found after dedup"
