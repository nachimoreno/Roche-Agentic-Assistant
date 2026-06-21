"""
seed_synthetic_feedback.py
--------------------------
CLI for manually (re)seeding or tearing down the synthetic feedback dataset
that powers the admin analytics dashboard (``/admin``).

The generator itself lives in ``src/demo_seed.py`` (shared with the app, which
seeds automatically on startup via ``ensure_demo_feedback``). This script is the
hands-on entry point: reseed at a different volume, spread, or seed; write the
full Session/Turn/citation chain; or tear the demo data down.

Run (conda env ``roche``)::

    python scripts/seed_synthetic_feedback.py --reset                 # teardown only
    python scripts/seed_synthetic_feedback.py --reset --count 1500    # crisper trend
    python scripts/seed_synthetic_feedback.py --reset --count 800 --full

All rows live under the demo tenant; ``--reset`` hard-deletes exactly that
tenant's rows and nothing else.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from db import create_all, make_engine                    # noqa: E402
from demo_seed import (                                    # noqa: E402
    DEFAULT_COUNT,
    DEFAULT_DAYS,
    DEFAULT_SEED,
    DEMO_TENANT_ID,
    PROCESS_CONFIG,
    reset,
    seed,
)
from repositories import FeedbackRepository               # noqa: E402

DEFAULT_DATABASE_URL = "sqlite:///data/app.db"


def _report(engine, seeded: dict) -> None:
    feedback = FeedbackRepository(engine)
    summary = feedback.summary(tenant_id=DEMO_TENANT_ID)
    hot_proc = feedback.hotspots(dimension="process", tenant_id=DEMO_TENANT_ID)
    hot_dept = feedback.hotspots(dimension="department", tenant_id=DEMO_TENANT_ID)
    hot_doc = feedback.hotspots(dimension="document", tenant_id=DEMO_TENANT_ID)
    trend = feedback.trend(tenant_id=DEMO_TENANT_ID)

    print("\n" + "=" * 68)
    print(f"SEEDED  {seeded['total']} entries across {seeded['sessions']} sessions")
    print(f"  negative: {seeded['negative']}  "
          f"(rate {seeded['negative_rate']:.1%})  "
          f"+{seeded['spike_extra']} spike entries "
          f"in {seeded['spike_window'][0]}..{seeded['spike_window'][1]}")
    print("\n  per process (expected neg_rate → seeded volume):")
    for p, cfg in sorted(PROCESS_CONFIG.items(),
                         key=lambda kv: -seeded["per_process"][kv[0]]):
        print(f"    {p:<20} ~{cfg['neg_rate']:>4.0%} neg   "
              f"{seeded['per_process'][p]:>4} entries")

    print("\n--- /api/analytics/summary --------------------------------------")
    print(f"  total={summary['total']}  negative={summary['negative']}  "
          f"rate={summary['negative_rate']:.1%}")
    print(f"  ratings={summary['ratings']}  sources={summary['sources']}")
    print(f"  languages={summary['languages']}")
    print(f"  emotions={summary['emotions']}")

    print("\n--- /api/analytics/hotspots?dimension=process -------------------")
    for r in hot_proc:
        print(f"  {r['key']:<20} weight={r['weight']:>6.2f}  "
              f"(cite {r['citation_weight']:.2f} / embed {r['embedding_weight']:.2f})  "
              f"n={r['feedback_count']}")
    print("\n--- hotspots?dimension=department ------------------------------")
    for r in hot_dept:
        print(f"  {r['key']:<20} weight={r['weight']:>6.2f}  n={r['feedback_count']}")
    print("\n--- hotspots?dimension=document (top 5) ------------------------")
    for r in hot_doc[:5]:
        print(f"  {r['key']:<34} weight={r['weight']:>6.2f}  n={r['feedback_count']}")

    print("\n--- /api/analytics/trend (per day) -----------------------------")
    for row in trend:
        bar = "#" * row["negative"]
        print(f"  {row['date']}  total={row['total']:>3}  neg={row['negative']:>3}  {bar}")
    print("=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT,
                    help=f"number of feedback entries to generate (default {DEFAULT_COUNT})")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"spread feedback over the last N days (default {DEFAULT_DAYS})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"RNG seed for deterministic output (default {DEFAULT_SEED})")
    ap.add_argument("--reset", action="store_true",
                    help="delete all synthetic (demo-tenant) rows first")
    ap.add_argument("--full", action="store_true",
                    help="also write real Session/Turn/TurnCitation rows")
    ap.add_argument("--database-url", default=DEFAULT_DATABASE_URL,
                    help=f"SQLAlchemy URL (default {DEFAULT_DATABASE_URL})")
    args = ap.parse_args()

    # Ensure the sqlite parent dir exists (mirrors main.build_engine).
    if args.database_url.startswith("sqlite:///"):
        db_path = Path(args.database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = make_engine(args.database_url)
    create_all(engine)

    print(f"tenant   = {DEMO_TENANT_ID}")
    print(f"database = {args.database_url}")

    if args.reset:
        deleted = reset(engine)
        print("reset    = " + ", ".join(f"{k}:{v}" for k, v in deleted.items()))

    if args.count <= 0:
        print("count <= 0: nothing to seed (reset-only run).")
        return

    seeded = seed(
        engine=engine, count=args.count, days=args.days,
        seed=args.seed, full=args.full, now=datetime.now(timezone.utc),
    )
    _report(engine, seeded)


if __name__ == "__main__":
    main()
