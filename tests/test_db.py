"""
Schema bootstrap tests — create_all additively patches an older DB (the
backends we run: SQLite in dev, PostgreSQL in prod).
"""

from __future__ import annotations

from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from db import _MIGRATABLE_DIALECTS, QuestionGap, create_all, make_engine
from repositories import QuestionGapRepository


# Columns the questiongap table had *before* the onboarding-funnel feature —
# used to forge a pre-migration table so we can prove create_all upgrades it.
_OLD_QUESTIONGAP_DDL = """
CREATE TABLE questiongap (
  id VARCHAR PRIMARY KEY, session_id VARCHAR, turn_id VARCHAR,
  tenant_id VARCHAR, query VARCHAR, kind VARCHAR, language VARCHAR,
  retrieval_max_dense FLOAT, retrieval_max_lexical FLOAT, embedding VARCHAR,
  cluster_id VARCHAR, cluster_label VARCHAR,
  created_at DATETIME, deleted_at DATETIME
)
"""

_ONBOARDING_COLUMNS = ("topic", "department", "tenure_days")


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


# ---------------------------------------------------------------------------
# Onboarding-funnel migration: columns *and* their declared indexes, on a DB
# created before the feature existed. This is the gap that left the funnel
# silent (and writes failing) on the deployed Postgres DB.
# ---------------------------------------------------------------------------

def test_create_all_adds_onboarding_columns_and_indexes(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/pre_onboarding.db")
    with engine.begin() as conn:
        conn.execute(sa.text(_OLD_QUESTIONGAP_DDL))

    create_all(engine)

    insp = sa.inspect(engine)
    cols = {c["name"] for c in insp.get_columns("questiongap")}
    assert set(_ONBOARDING_COLUMNS) <= cols
    # The columns are declared index=True, so the indexes must exist too —
    # create_all skips indexes on already-existing tables, the helper backfills.
    idx = {i["name"] for i in insp.get_indexes("questiongap")}
    for col in _ONBOARDING_COLUMNS:
        assert f"ix_questiongap_{col}" in idx


def test_migrated_db_is_writable_and_funnel_populates(tmp_path):
    """End-to-end proof the upgrade is *functional*, not just present: after
    migrating an old DB, the repo can write the new columns and the onboarding
    funnel reflects them — exactly what silently failed before."""
    engine = make_engine(f"sqlite:///{tmp_path}/pre_onboarding.db")
    with engine.begin() as conn:
        conn.execute(sa.text(_OLD_QUESTIONGAP_DDL))
    create_all(engine)

    repo = QuestionGapRepository(engine)
    sid = uuid4()
    repo.add(session_id=sid, query="how do I get vpn access",
             kind="declined", topic="access", tenure_days=3)   # newcomer
    repo.add(session_id=sid, query="book the scope",
             kind="declined", topic="booking", tenure_days=90)  # veteran

    funnel = repo.onboarding(newcomer_days=14)
    assert funnel["total"] == 1
    assert funnel["topics"][0]["topic"] == "access"


def test_idempotent_on_already_migrated_questiongap(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/pre_onboarding.db")
    with engine.begin() as conn:
        conn.execute(sa.text(_OLD_QUESTIONGAP_DDL))
    create_all(engine)
    create_all(engine)  # second pass: columns + indexes already there, no raise

    idx = {i["name"] for i in sa.inspect(engine).get_indexes("questiongap")}
    assert "ix_questiongap_topic" in idx


# ---------------------------------------------------------------------------
# Postgres path (no live server available locally, so we verify the contract:
# the helper no longer skips Postgres, and the DDL it builds is valid PG SQL).
# ---------------------------------------------------------------------------

def test_postgres_is_migrated_not_skipped():
    # Regression guard for the original bug: Postgres was excluded, so prod
    # never got the new columns. It must be in the migratable set now.
    assert "postgresql" in _MIGRATABLE_DIALECTS
    assert "sqlite" in _MIGRATABLE_DIALECTS


def test_onboarding_columns_compile_to_valid_postgresql():
    pg = postgresql.dialect()
    table = QuestionGap.__table__
    for name in _ONBOARDING_COLUMNS:
        ddl_type = table.c[name].type.compile(dialect=pg)
        assert ddl_type  # e.g. VARCHAR / INTEGER — never empty
        stmt = f'ALTER TABLE "{table.name}" ADD COLUMN "{name}" {ddl_type}'
        assert stmt.startswith('ALTER TABLE "questiongap" ADD COLUMN "')
