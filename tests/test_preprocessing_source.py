"""
tests/test_preprocessing_source.py
----------------------------------
Unit tests for the PreprocessingSource decorator — pure, no Drive calls.
Verifies it tags process/department from keywords, maps process_type onto the
existing `process` key, and never overrides labels the inner source provided.

Run:
    pytest tests/test_preprocessing_source.py
"""

import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from document_source import SourceDocument
from preprocessing_source import PreprocessingSource, wrap_if_enabled


def _doc(title, content, metadata=None):
    return SourceDocument(
        id=title,
        title=title,
        content=content,
        modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata=metadata if metadata is not None else {},
    )


class _FakeSource:
    """Minimal DocumentSource yielding a fixed list of docs."""

    def __init__(self, docs):
        self._docs = docs

    def list_documents(self):
        yield from self._docs


class TestTagging:
    def test_tags_untagged_doc_from_filename(self):
        src = _FakeSource([
            _doc(
                "New Employee Onboarding Guide",
                "Welcome to the team.",
                {"drive_name": "New Employee Onboarding Guide.docx"},
            )
        ])
        out = list(PreprocessingSource(src).list_documents())
        assert out[0].metadata["process"] == "onboarding"
        assert out[0].metadata["department"] == "Onboarding"

    def test_tags_from_content_when_filename_neutral(self):
        src = _FakeSource([
            _doc("Doc 1", "Procedures for chemical waste disposal and decontamination.")
        ])
        out = list(PreprocessingSource(src).list_documents())
        assert out[0].metadata["process"] == "waste"
        assert out[0].metadata["department"] == "Waste"

    def test_falls_back_to_general(self):
        src = _FakeSource([_doc("Misc", "Nothing matches here.")])
        out = list(PreprocessingSource(src).list_documents())
        assert out[0].metadata["process"] == "general"
        assert out[0].metadata["department"] == "General"

    def test_maps_process_type_to_process_key_not_process_type(self):
        # The existing attribution layer reads `process`, not `process_type`.
        src = _FakeSource([_doc("Onboarding", "digital access guide")])
        out = list(PreprocessingSource(src).list_documents())
        assert "process" in out[0].metadata
        assert "process_type" not in out[0].metadata


class TestRespectsExistingMetadata:
    def test_does_not_override_existing_labels(self):
        src = _FakeSource([
            _doc(
                "Onboarding Guide",  # would infer "onboarding"
                "digital access guide",
                {"process": "lab_operations", "department": "Lab Operations"},
            )
        ])
        out = list(PreprocessingSource(src).list_documents())
        assert out[0].metadata["process"] == "lab_operations"
        assert out[0].metadata["department"] == "Lab Operations"

    def test_fills_only_the_missing_label(self):
        src = _FakeSource([
            _doc("Onboarding Guide", "digital access guide", {"department": "HR"})
        ])
        out = list(PreprocessingSource(src).list_documents())
        assert out[0].metadata["department"] == "HR"        # kept
        assert out[0].metadata["process"] == "onboarding"   # filled


class TestPassthrough:
    def test_content_and_identity_unchanged(self):
        original = _doc("Doc", "waste disposal steps", {"drive_id": "abc"})
        out = list(PreprocessingSource(_FakeSource([original])).list_documents())
        assert out[0].id == "Doc"
        assert out[0].title == "Doc"
        assert out[0].content == "waste disposal steps"
        assert out[0].metadata["drive_id"] == "abc"


class TestWrapIfEnabled:
    def test_wraps_when_enabled(self):
        src = _FakeSource([])
        assert isinstance(wrap_if_enabled(src, enabled=True), PreprocessingSource)

    def test_returns_same_object_when_disabled(self):
        src = _FakeSource([])
        assert wrap_if_enabled(src, enabled=False) is src
