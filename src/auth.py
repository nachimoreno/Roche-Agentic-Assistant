"""
auth.py
-------
Authentication seam.

Password hashing (argon2id) and the `get_current_user` FastAPI dependency
live here. The orchestrator and repositories never learn *how* a user is
authenticated — they receive an opaque `user_id` string. Swapping this local
password provider for SSO/OIDC later means reimplementing `get_current_user`
and the login endpoints; nothing downstream changes.
"""

from __future__ import annotations

import logging
from uuid import UUID

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error
from fastapi import HTTPException, Request, status

from db import User

logger = logging.getLogger(__name__)

_ph = PasswordHasher()


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """True if `password` matches `password_hash`; False on any mismatch."""
    try:
        return _ph.verify(password_hash, password)
    except Argon2Error:
        return False


def require_admin(request: Request) -> User:
    """Resolve the session to a live admin `User`, or 404.

    404 (not 403) so non-admins can't tell the admin surface exists — same
    convention as `_require_owned` in the API layer. Role is read fresh from
    the DB via `get_current_user`, so a revoked admin loses access on the
    next request, not the next login.
    """
    user = get_current_user(request)
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    return user


def get_current_user(request: Request) -> User:
    """Resolve the signed-cookie session to a live `User`, or 401.

    Reads `user_id` from the SessionMiddleware-backed cookie and loads the
    user via the `UserRepository` on `app.state.users`.
    """
    uid = request.session.get("user_id")
    if not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    users = request.app.state.users
    try:
        user = users.get(UUID(uid))
    except (ValueError, TypeError):
        user = None

    if user is None or user.deleted_at is not None:
        request.session.clear()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    return user
