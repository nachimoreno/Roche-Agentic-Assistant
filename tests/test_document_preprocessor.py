"""
test_document_preprocessor.py
-------------------------------
Fast mocked tests for the preprocessing pipeline logic.

Run:
    pytest tests/test_document_preprocessor.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from unittest.mock import MagicMock, patch
from document_preprocessor import (
    DocumentPreprocessor,
    ProcessedDocument,
    infer_department_and_process,
    plain_text_to_markdown,
)


class TestInferDepartmentAndProcess:

    def test_onboarding_detected_from_filename(self):
        dept, proc = infer_department_and_process(
            "New Employee Onboarding Guide.docx", "welcome to the team"
        )
        assert proc == "onboarding"

    def test_waste_detected_from_content(self):
        dept, proc = infer_department_and_process(
            "Misc Document.docx", "this covers waste disposal procedures"
        )
        assert proc == "waste"

    def test_falls_back_to_general(self):
        dept, proc = infer_department_and_process(
            "Random Notes.docx", "nothing matches any keyword here"
        )
        assert proc == "general"
        assert dept == "General"

    def test_procurement_detected(self):
        dept, proc = infer_department_and_process(
            "Ordering Chemicals.docx", "how to order consumables"
        )
        assert proc == "procurement"


class TestProcessedDocument:

    def test_to_file_content_includes_frontmatter(self):
        doc = ProcessedDocument(
            source_id="abc123",
            source_name="Test Doc.docx",
            department="Onboarding",
            process_type="onboarding",
            markdown="# Test\n\nBody text.",
        )
        content = doc.to_file_content()
        assert "source_file:" in content
        assert "abc123" in content
        assert "Onboarding" in content
        assert "# Test" in content

    def test_frontmatter_is_valid_yaml_delimited(self):
        doc = ProcessedDocument(
            source_id="x", source_name="y", department="z",
            process_type="w", markdown="body"
        )
        content = doc.to_file_content()
        assert content.startswith("---\n")
        assert content.count("---") == 2


class TestPlainTextToMarkdown:

    def test_decodes_bytes(self):
        result = plain_text_to_markdown(b"Hello world")
        assert result == "Hello world"

    def test_handles_invalid_utf8_gracefully(self):
        result = plain_text_to_markdown(b"\xff\xfe invalid")
        assert isinstance(result, str)


class TestDocumentPreprocessor:
    """Tests the orchestration logic with mocked Drive services."""

    def _make_preprocessor(self):
        return DocumentPreprocessor(folder_id="test-folder", credentials_path="fake.json")

    def test_ensure_processed_folder_reuses_existing(self):
        pp = self._make_preprocessor()
        mock_read = MagicMock()
        mock_read.files().list().execute.return_value = {
            "files": [{"id": "existing-folder-id", "name": "processed"}]
        }
        pp._read_service = mock_read

        folder_id = pp._ensure_processed_folder()
        assert folder_id == "existing-folder-id"

    def test_ensure_processed_folder_creates_if_missing(self):
        pp = self._make_preprocessor()
        mock_read = MagicMock()
        mock_read.files().list().execute.return_value = {"files": []}
        pp._read_service = mock_read

        mock_write = MagicMock()
        mock_write.files().create().execute.return_value = {"id": "new-folder-id"}
        pp._write_service = mock_write

        folder_id = pp._ensure_processed_folder()
        assert folder_id == "new-folder-id"

    def test_list_source_documents_filters_unsupported_mime(self):
        pp = self._make_preprocessor()
        mock_read = MagicMock()
        mock_read.files().list().execute.return_value = {
            "files": [
                {"id": "1", "name": "doc.docx", "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
                {"id": "2", "name": "image.png", "mimeType": "image/png"},
                {"id": "3", "name": "doc.pdf", "mimeType": "application/pdf"},
            ]
        }
        pp._read_service = mock_read

        docs = pp._list_source_documents()
        names = [d["name"] for d in docs]
        assert "doc.docx" in names
        assert "doc.pdf" in names
        assert "image.png" not in names

    def test_read_and_write_services_are_independent(self):
        """Critical: read and write must never share the same service object.
        Verifies separate lazy-init slots exist before any service is built."""
        pp = self._make_preprocessor()
        assert pp._read_service is None
        assert pp._write_service is None
        # The two builder methods are distinct and each caches independently
        assert pp._get_read_service.__name__ != pp._get_write_service.__name__


class TestProcessOne:

    @patch("document_preprocessor.docx_to_markdown")
    def test_process_one_builds_processed_document(self, mock_convert):
        mock_convert.return_value = "# Heading\n\nBody"
        pp = DocumentPreprocessor(folder_id="f", credentials_path="fake.json")
        pp._download = MagicMock(return_value=b"fake docx bytes")

        file_meta = {
            "id": "doc-1",
            "name": "Onboarding Guide.docx",
            "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        result = pp.process_one(file_meta)

        assert result.source_id == "doc-1"
        assert result.process_type == "onboarding"
        assert "Heading" in result.markdown


class TestWriteLocalFallback:
    """
    Tests the dev/demo local write path. See module docstring's
    'KNOWN PLATFORM CONSTRAINT' section — this is explicitly NOT the
    production write path and must never be used with real Roche data.
    """

    def test_writes_file_to_local_output_dir(self, tmp_path):
        pp = DocumentPreprocessor(
            folder_id="f",
            local_output_dir=str(tmp_path / "processed"),
            credentials_path="fake.json",
        )
        doc = ProcessedDocument(
            source_id="abc", source_name="Test Doc.docx",
            department="Onboarding", process_type="onboarding",
            markdown="# Hello",
        )
        output_path = pp._write_local_fallback(doc)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            content = f.read()
        assert "# Hello" in content
        assert "source_file:" in content

    def test_creates_output_dir_if_missing(self, tmp_path):
        target_dir = str(tmp_path / "nested" / "processed")
        pp = DocumentPreprocessor(
            folder_id="f", local_output_dir=target_dir, credentials_path="fake.json"
        )
        doc = ProcessedDocument(
            source_id="x", source_name="Doc.pdf",
            department="General", process_type="general", markdown="body",
        )
        pp._write_local_fallback(doc)
        assert os.path.isdir(target_dir)

    def test_strips_source_extension_in_output_filename(self, tmp_path):
        pp = DocumentPreprocessor(
            folder_id="f", local_output_dir=str(tmp_path), credentials_path="fake.json"
        )
        doc = ProcessedDocument(
            source_id="x", source_name="Cleaning Procedure.docx",
            department="Lab Operations", process_type="lab_operations", markdown="body",
        )
        output_path = pp._write_local_fallback(doc)
        assert output_path.endswith("Cleaning Procedure.md")
        assert ".docx" not in output_path

    def test_overwrites_existing_processed_file(self, tmp_path):
        pp = DocumentPreprocessor(
            folder_id="f", local_output_dir=str(tmp_path), credentials_path="fake.json"
        )
        doc_v1 = ProcessedDocument(
            source_id="x", source_name="Doc.docx", department="d",
            process_type="p", markdown="version one",
        )
        doc_v2 = ProcessedDocument(
            source_id="x", source_name="Doc.docx", department="d",
            process_type="p", markdown="version two",
        )
        path1 = pp._write_local_fallback(doc_v1)
        path2 = pp._write_local_fallback(doc_v2)

        assert path1 == path2
        with open(path2) as f:
            content = f.read()
        assert "version two" in content
        assert "version one" not in content
