"""
Test-wide fixtures and harness setup.

`pytest.ini` adds `src/` to `pythonpath`, so test modules can import
project modules directly (e.g. `from db import ...`).
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_collection_modifyitems(config, items):
    """Skip `live` tests when no Groq API key is available."""
    if os.environ.get("GROQ_API_KEY"):
        return
    skip_live = pytest.mark.skip(reason="GROQ_API_KEY not set")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
