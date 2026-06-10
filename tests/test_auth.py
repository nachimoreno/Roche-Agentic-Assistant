"""
Auth tests: password hashing and the register/login/logout/me HTTP flow.

A file-backed SQLite DB is injected into app.state.users (a temp file, not
:memory:, because TestClient runs sync endpoints in a worker thread and
:memory: gives each connection its own empty database). Lifespan is not
triggered, so no Groq/ingest.
"""

from __future__ import annotations

import os

os.environ.setdefault("GROQ_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient

import api
from auth import hash_password, verify_password
from db import create_all, make_engine
from repositories import UserRepository


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def test_hash_password_round_trip():
    h = hash_password("correct horse battery staple")
    assert h != "correct horse battery staple"      # not plaintext
    assert verify_password(h, "correct horse battery staple")
    assert not verify_password(h, "wrong password")


def test_hashes_are_salted_unique():
    assert hash_password("same") != hash_password("same")


# ---------------------------------------------------------------------------
# HTTP flow
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/auth.db")
    create_all(engine)
    api.app.state.users = UserRepository(engine)
    return TestClient(api.app)


def _register(client, email="sci@roche.com", password="supersecret", name="Dr Sci"):
    return client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "display_name": name},
    )


def test_register_creates_account_and_logs_in(client):
    r = _register(client)
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == "sci@roche.com"
    assert body["display_name"] == "Dr Sci"
    assert "password" not in body and "password_hash" not in body
    # Cookie set -> /me works without re-auth.
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "sci@roche.com"


def test_register_normalizes_email_and_rejects_duplicates(client):
    assert _register(client, email="Sci@Roche.com").status_code == 201
    dup = _register(client, email="sci@roche.com")
    assert dup.status_code == 409


def test_register_validates_email_and_password(client):
    assert client.post("/api/auth/register", json={"email": "nope", "password": "longenough"}).status_code == 422
    assert client.post("/api/auth/register", json={"email": "a@b.co", "password": "short"}).status_code == 422


def test_me_requires_authentication(client):
    assert client.get("/api/auth/me").status_code == 401


def test_login_success_and_failure(client):
    _register(client, email="lab@roche.com", password="supersecret")
    client.post("/api/auth/logout")

    bad = client.post("/api/auth/login", json={"email": "lab@roche.com", "password": "nope"})
    assert bad.status_code == 401
    # Unknown email returns the same generic error (no user enumeration).
    unknown = client.post("/api/auth/login", json={"email": "ghost@roche.com", "password": "whatever"})
    assert unknown.status_code == 401
    assert bad.json()["detail"] == unknown.json()["detail"]

    ok = client.post("/api/auth/login", json={"email": "lab@roche.com", "password": "supersecret"})
    assert ok.status_code == 200
    assert client.get("/api/auth/me").status_code == 200


def test_logout_clears_session(client):
    _register(client)
    assert client.get("/api/auth/me").status_code == 200
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401
