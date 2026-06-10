"""
Tests for GroqClient.check_auth — the startup credential check.

No network: the Groq SDK client is swapped for a fake whose models.list()
either succeeds or raises a real groq.AuthenticationError.
"""

from __future__ import annotations

import httpx
import pytest
from groq import AuthenticationError

from llm import GroqClient, LLMAuthError


def _auth_error() -> AuthenticationError:
    req = httpx.Request("GET", "https://api.groq.com/openai/v1/models")
    resp = httpx.Response(401, request=req)
    return AuthenticationError("Invalid API Key", response=resp, body=None)


class _FakeModels:
    def __init__(self, exc=None):
        self._exc = exc

    def list(self):
        if self._exc:
            raise self._exc
        return ["llama-3.3-70b-versatile"]


class _FakeClient:
    def __init__(self, exc=None):
        self.models = _FakeModels(exc)


def _client_with(exc=None) -> GroqClient:
    gc = GroqClient(api_key="x", model="m")  # no network at construction
    gc._client = _FakeClient(exc)
    return gc


def test_check_auth_passes_when_credentials_accepted():
    _client_with().check_auth()  # should not raise


def test_check_auth_raises_llmautherror_on_401():
    with pytest.raises(LLMAuthError):
        _client_with(_auth_error()).check_auth()


def test_check_auth_propagates_non_auth_errors():
    # A connectivity/other error is not an auth problem and must not be
    # masquerade as LLMAuthError.
    boom = RuntimeError("network down")
    with pytest.raises(RuntimeError) as ei:
        _client_with(boom).check_auth()
    assert not isinstance(ei.value, LLMAuthError)
