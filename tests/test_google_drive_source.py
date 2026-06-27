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

import google_drive_source as gds
from google_drive_source import (
    GoogleDriveSource,
    _deduplicate,
    _infer_title,
    _parse_time,
    extract_text,
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


class TestCheckConnection:
    """check_connection() is the startup health probe used by api.py."""

    def test_returns_file_count_on_success(self):
        src = GoogleDriveSource(folder_id="f", credentials_path=None)
        service = MagicMock()
        service.files().list().execute.return_value = {"files": [{"id": "1"}, {"id": "2"}]}
        src._service = service

        assert src.check_connection() == 2

    def test_propagates_failure_reason(self):
        src = GoogleDriveSource(folder_id="f", credentials_path=None)
        service = MagicMock()
        service.files().list().execute.side_effect = RuntimeError("403 forbidden")
        src._service = service

        with pytest.raises(RuntimeError, match="403 forbidden"):
            src.check_connection()


class TestCanWrite:
    """can_write() is the non-destructive upload-permission probe behind the
    UI's grey-out. It must never raise — a viewer-only account returns False."""

    def _source_with_capability(self, can_add):
        src = GoogleDriveSource(folder_id="f", credentials_path=None)
        service = MagicMock()
        service.files().get().execute.return_value = {
            "capabilities": {"canAddChildren": can_add}
        }
        src._writer_service = service
        return src

    def test_true_when_folder_allows_adding_children(self):
        assert self._source_with_capability(True).can_write() is True

    def test_false_when_viewer_only(self):
        assert self._source_with_capability(False).can_write() is False

    def test_false_and_swallows_errors(self):
        src = GoogleDriveSource(folder_id="f", credentials_path=None)
        service = MagicMock()
        service.files().get().execute.side_effect = RuntimeError("403 forbidden")
        src._writer_service = service
        assert src.can_write() is False


class TestExtractText:
    """extract_text() turns uploaded bytes into plain text, dispatching on type."""

    def test_txt_decodes_utf8(self):
        assert extract_text("notes.txt", "héllo".encode("utf-8")) == "héllo"

    def test_md_decodes(self):
        assert "# Title" in extract_text("doc.md", b"# Title\n\nbody")

    def test_unknown_extension_falls_back_to_text_decode(self):
        assert extract_text("data.csv", b"a,b,c") == "a,b,c"

    def test_pdf_dispatches_to_pdf_extractor(self, monkeypatch):
        monkeypatch.setattr(gds, "_pdf_to_text", lambda data: "PDF TEXT")
        assert extract_text("f.pdf", b"%PDF-1.4 ...") == "PDF TEXT"

    def test_docx_dispatches_to_docx_extractor(self, monkeypatch):
        monkeypatch.setattr(gds, "_docx_to_text", lambda data: "DOCX TEXT")
        assert extract_text("f.docx", b"PK\x03\x04 ...") == "DOCX TEXT"

    def test_dispatches_on_mime_when_extension_missing(self, monkeypatch):
        monkeypatch.setattr(gds, "_pdf_to_text", lambda data: "PDF TEXT")
        assert extract_text("noext", b"bytes", "application/pdf") == "PDF TEXT"


_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class TestMarkdownMode:
    """markdown=True routes DOCX/PDF through the structured converters; the
    default keeps the historical flat-text extraction. _download_text dispatch
    only — the converters themselves are covered in test_document_preprocessor."""

    def _source(self, markdown):
        src = GoogleDriveSource(folder_id="f", credentials_path="{}", markdown=markdown)
        src._get_service = lambda: MagicMock()  # no real Drive
        return src

    def test_markdown_mode_uses_structured_converter(self, monkeypatch):
        monkeypatch.setattr(gds, "_download_bytes", lambda service, fid: b"raw")
        monkeypatch.setattr(gds, "_docx_to_markdown", lambda data: "## Heading")
        monkeypatch.setattr(gds, "_docx_to_text", lambda data: "flat")
        out = self._source(markdown=True)._download_text(
            {"id": "1", "name": "d.docx", "mimeType": _DOCX_MIME}
        )
        assert out == "## Heading"

    def test_default_mode_uses_flat_text(self, monkeypatch):
        monkeypatch.setattr(gds, "_download_bytes", lambda service, fid: b"raw")
        monkeypatch.setattr(gds, "_docx_to_markdown", lambda data: "## Heading")
        monkeypatch.setattr(gds, "_docx_to_text", lambda data: "flat")
        out = self._source(markdown=False)._download_text(
            {"id": "1", "name": "d.docx", "mimeType": _DOCX_MIME}
        )
        assert out == "flat"

    def test_docx_markdown_falls_back_when_converter_unavailable(self, monkeypatch):
        import document_preprocessor as dp
        monkeypatch.setattr(dp, "docx_to_markdown", lambda data: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(gds, "_docx_to_text", lambda data: "flat fallback")
        assert gds._docx_to_markdown(b"raw") == "flat fallback"


# ---------------------------------------------------------------------------
# Live integration tests — skipped unless credentials present
# ---------------------------------------------------------------------------

def _live_creds_ready() -> bool:
    """True only when Drive is fully configured *and* the key file exists.

    Checking the file exists (not just the env var) means a path pointing at a
    missing key skips these tests instead of erroring with FileNotFoundError.
    """
    creds = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_OAUTH_CREDENTIALS")
    if not creds or not os.getenv("DRIVE_FOLDER_ID"):
        return False
    return os.path.isfile(creds)


LIVE = pytest.mark.skipif(
    not _live_creds_ready(),
    reason="Live Drive tests require DRIVE_FOLDER_ID + an existing credentials file",
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
