"""
Schema bootstrap tests — create_all additively patches an older SQLite DB.
"""

from __future__ import annotations

import sqlalchemy as sa

from db import create_all, make_engine


def test_create_all_adds_missing_nullable_column(tmp_path):
    url = f"sqlite:///{tmp_path}/old.db"
    engine = make_engine(url)
    # Simulate a pre-`title` schema: a session table without the column.
    with engine.begin() as conn:
        conn.execute(sa.text(
            "CREATE TABLE session ("
            "id VARCHAR PRIMARY KEY, tenant_id VARCHAR, user_id VARCHAR, "
            "created_at DATETIME, deleted_at DATETIME)"
        ))

    create_all(engine)  # should ALTER session to add `title`, create other tables

    cols = {c["name"] for c in sa.inspect(engine).get_columns("session")}
    assert "title" in cols
    # And the brand-new tables were created too.
    assert "user" in sa.inspect(engine).get_table_names()


def test_create_all_is_idempotent(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/fresh.db")
    create_all(engine)
    create_all(engine)  # second run must not raise (column already present)
    cols = {c["name"] for c in sa.inspect(engine).get_columns("session")}
    assert "title" in cols
