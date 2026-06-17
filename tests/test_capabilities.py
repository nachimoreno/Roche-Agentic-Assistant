"""
Self-knowledge constant — content and rendering contract.

`CAPABILITIES` is the single source of truth for what the assistant can do. It
is never ingested or retrieved, so these tests guard the content directly: the
two sections render, the supported languages are named, Drive ingestion reads
as a current capability (the bug doc 00 used to contradict), and ServiceNow
incident creation is described as not-yet rather than claimed.
"""

from __future__ import annotations

from capabilities import CAPABILITIES


def _split(block: str) -> tuple[str, str]:
    can_part, sep, cannot_part = block.partition("What you cannot do yet:")
    assert sep, "block must have a 'cannot do yet' section"
    return can_part, cannot_part


def test_block_renders_both_sections_with_date():
    block = CAPABILITIES.as_prompt_block()
    assert "What you can do today" in block
    assert CAPABILITIES.as_of in block
    assert "What you cannot do yet" in block


def test_supported_languages_named():
    block = CAPABILITIES.as_prompt_block()
    for lang in ("English", "German", "French", "Italian"):
        assert lang in block


def test_drive_ingestion_is_a_current_capability():
    # Drive is built (google_drive_source.py); it must read as can-do. This is
    # the contradiction the old doc 00 introduced by calling it not-yet.
    can_part, cannot_part = _split(CAPABILITIES.as_prompt_block())
    assert "Google Drive" in can_part
    assert "Google Drive" not in cannot_part


def test_servicenow_is_not_yet_and_never_claimed():
    can_part, cannot_part = _split(CAPABILITIES.as_prompt_block())
    assert "ServiceNow" in cannot_part
    assert "ServiceNow" not in can_part
