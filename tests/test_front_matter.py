"""
Front-matter parsing tests.

Covers the minimal `--- key: value ---` parser and that LocalMarkdownSource
strips the block from the embedded body while surfacing process/department.
"""

from __future__ import annotations

from document_source import LocalMarkdownSource, parse_front_matter


def test_parses_keys_and_strips_block():
    text = "---\nprocess: onboarding\ndepartment: it\n---\n\n# Title\n\nBody."
    fields, body = parse_front_matter(text)
    assert fields == {"process": "onboarding", "department": "it"}
    assert body == "# Title\n\nBody."


def test_strips_surrounding_quotes_and_lowercases_keys():
    text = "---\nProcess: 'sample-stock'\nDepartment: \"lab-operations\"\n---\nx"
    fields, _ = parse_front_matter(text)
    assert fields == {"process": "sample-stock", "department": "lab-operations"}


def test_no_front_matter_returns_text_unchanged():
    text = "# Just a heading\n\nNo front-matter here."
    fields, body = parse_front_matter(text)
    assert fields == {}
    assert body == text


def test_unterminated_block_does_not_lose_content():
    # Missing closing '---' → treat the whole thing as body, drop nothing.
    text = "---\nprocess: onboarding\n# oops no close\nmore body"
    fields, body = parse_front_matter(text)
    assert fields == {}
    assert body == text


def test_local_source_surfaces_labels_and_clean_body(tmp_path):
    doc = tmp_path / "10_widget.md"
    doc.write_text(
        "---\nprocess: widgetry\ndepartment: lab-operations\n---\n\n# Widget\n\nHow to widget.",
        encoding="utf-8",
    )
    [sd] = list(LocalMarkdownSource(tmp_path).list_documents())
    assert sd.id == "10_widget.md"
    assert sd.title == "Widget"
    assert sd.metadata["process"] == "widgetry"
    assert sd.metadata["department"] == "lab-operations"
    # Front-matter must not remain in the content that gets embedded.
    assert "process:" not in sd.content
    assert sd.content.startswith("# Widget")
