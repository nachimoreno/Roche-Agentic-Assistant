"""
demo_seed.py
------------
Synthetic feedback generator for the admin analytics dashboard (``/admin``).

The dashboard reads only two tables — ``FeedbackEntry`` and
``FeedbackAttribution`` — through ``FeedbackRepository``. There is no LLM or
retrieval in the read path, so a grounded, hand-shaped dataset is enough to make
every chart light up. This module writes *production-shaped* rows through the
same repository methods the live app uses, so analytics can't tell synthetic
feedback from real feedback.

It is a story, not noise. The per-process knobs below encode a narrative:

  * lab-operations carries more pain than IT;
  * instrument-booking is the worst hotspot (high volume, ~55% negative);
  * sample-stock is the runner-up (~45% negative);
  * virtual-sessions has a one-week negative SPIKE (an "outage") visible on the
    trend chart;
  * equipment-cleaning / decontamination are mostly positive (~10% negative).

Isolation & teardown
--------------------
All rows are written under a dedicated **demo tenant** (``DEMO_TENANT_ID``), so
``reset()`` can hard-delete exactly the synthetic data with a tenant-scoped
DELETE and nothing else. An admin promoted via ``ADMIN_EMAILS`` has
``tenant_id = None``, and the analytics repo applies *no* tenant filter when the
viewer's tenant is ``None`` — so any such admin sees this data immediately at
``/admin`` without a special demo login.

Two entry points:

* ``ensure_demo_feedback(engine, ...)`` — idempotent; called on app startup.
  Seeds once if the demo tenant has no feedback yet, otherwise does nothing.
* ``seed(...)`` / ``reset(...)`` — the building blocks, also driven by the
  ``scripts/seed_synthetic_feedback.py`` CLI for manual (re)seeding/teardown.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import Engine, delete
from sqlmodel import Session as DbSession

from attribution import AttributionRow
from db import (
    FeedbackAttribution,
    FeedbackEntry,
    Session,
    Turn,
    TurnCitation,
    new_id,
    utcnow,
)
from repositories import FeedbackRepository, SessionRepository


logger = logging.getLogger(__name__)


# A fixed sentinel tenant for all synthetic demo data. Recognisable in the DB
# and the unit by which `reset()` tears everything down.
DEMO_TENANT_ID = UUID("d3000000-0000-4000-8000-000000000001")

# Defaults shared by the CLI and the startup hook. ~2000 keeps the per-day
# trend line smooth while seeding in a couple of seconds on SQLite.
DEFAULT_COUNT = 2000
DEFAULT_DAYS = 60
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Narrative configuration — the "story" the demo tells
# ---------------------------------------------------------------------------

# process -> knobs. `doc`/`dept` mirror the corpus front-matter exactly (see
# tests/fixtures/docs/*.md). `volume` is a relative weight (entries are drawn
# in proportion). `neg_rate` is the baseline share of negative feedback. The
# emotion pools are drawn from to keep sentiment coherent across both streams.
PROCESS_CONFIG: dict[str, dict] = {
    "instrument-booking": dict(
        doc="04_booking_instruments.md", dept="lab-operations",
        volume=22, neg_rate=0.55,
        neg=["frustrated", "annoyed", "irritated"], pos=["satisfied", "pleased"],
    ),
    "sample-stock": dict(
        doc="05_checking_sample_stock.md", dept="lab-operations",
        volume=16, neg_rate=0.45,
        neg=["confused", "frustrated", "concerned"], pos=["satisfied", "neutral"],
    ),
    "virtual-sessions": dict(
        doc="08_virtual_session_troubleshooting.md", dept="it",
        volume=14, neg_rate=0.30, spike=True,
        neg=["frustrated", "anxious", "stressed"], pos=["pleased", "impressed"],
    ),
    "incident-reporting": dict(
        doc="03_incident_reporting.md", dept="it",
        volume=10, neg_rate=0.25,
        neg=["confused", "uncertain"], pos=["satisfied", "appreciative"],
    ),
    "internal-apps": dict(
        doc="02_navigating_internal_apps.md", dept="it",
        volume=9, neg_rate=0.22,
        neg=["confused", "annoyed"], pos=["neutral", "satisfied"],
    ),
    "onboarding": dict(
        doc="01_onboarding_access_requests.md", dept="it",
        volume=9, neg_rate=0.20,
        neg=["confused", "overwhelmed"], pos=["appreciative", "pleased"],
    ),
    "equipment-cleaning": dict(
        doc="06_cleaning_lab_devices.md", dept="lab-operations",
        volume=10, neg_rate=0.10,
        neg=["uncertain"], pos=["satisfied", "pleased", "impressed"],
    ),
    "decontamination": dict(
        doc="07_decontamination.md", dept="lab-operations",
        volume=8, neg_rate=0.10,
        neg=["concerned"], pos=["satisfied", "appreciative"],
    ),
}

# Realistic co-citation: a complaint sometimes blames a second, related doc, so
# blame spreads 0.5/0.5 across two processes. (secondary_doc, process, dept).
SECONDARY_DOC: dict[str, tuple[str, str, str]] = {
    "virtual-sessions": ("03_incident_reporting.md", "incident-reporting", "it"),
    "incident-reporting": ("08_virtual_session_troubleshooting.md", "virtual-sessions", "it"),
    "instrument-booking": ("05_checking_sample_stock.md", "sample-stock", "lab-operations"),
    "sample-stock": ("04_booking_instruments.md", "instrument-booking", "lab-operations"),
    "onboarding": ("02_navigating_internal_apps.md", "internal-apps", "it"),
    "internal-apps": ("01_onboarding_access_requests.md", "onboarding", "it"),
    "equipment-cleaning": ("07_decontamination.md", "decontamination", "lab-operations"),
    "decontamination": ("06_cleaning_lab_devices.md", "equipment-cleaning", "lab-operations"),
}

# english messages are process-specific (the story); other languages use a
# compact generic bank — enough to populate the language distribution.
NEG_MESSAGES: dict[str, list[str]] = {
    "instrument-booking": [
        "The instrument scheduler won't let me release my booking.",
        "I keep getting double-booked slots, this is infuriating.",
        "Booking confirmation never arrived and my slot is gone.",
    ],
    "sample-stock": [
        "Stock counts are wrong again — the app shows zero for a full freezer.",
        "I can't tell what's reserved versus available in Sample Stock.",
        "My restock request just disappeared, no confirmation at all.",
    ],
    "virtual-sessions": [
        "My virtual session keeps disconnecting mid-analysis.",
        "The remote desktop froze and I lost my work.",
        "I can't log into the virtual session at all this morning.",
    ],
    "incident-reporting": [
        "ServiceNow won't accept my incident — the form errors out.",
        "I have no idea which category to pick for this incident.",
    ],
    "internal-apps": [
        "I can't find the right internal app for this task.",
        "The link in the app catalog is broken.",
    ],
    "onboarding": [
        "Still waiting on access three days after onboarding.",
        "The onboarding steps don't match what I actually see on screen.",
    ],
    "equipment-cleaning": [
        "I'm not sure this cleaning step is correct for the rotor.",
    ],
    "decontamination": [
        "The decontamination steps are unclear for this kind of spill.",
    ],
}
POS_MESSAGES: dict[str, list[str]] = {
    "instrument-booking": ["Booking an instrument was quick and clear, thanks!",
                           "Released my slot easily — great."],
    "sample-stock": ["Found the stock levels right away, very handy."],
    "virtual-sessions": ["Reconnected in seconds, worked perfectly."],
    "incident-reporting": ["Filing the incident was straightforward, thank you."],
    "internal-apps": ["The app catalog pointed me to exactly the right tool."],
    "onboarding": ["Onboarding access came through fast, much appreciated."],
    "equipment-cleaning": ["The cleaning guidance was clear and easy to follow.",
                           "Exactly the procedure I needed, thanks."],
    "decontamination": ["The spill response steps were clear and reassuring."],
}
NEG_GENERIC: dict[str, list[str]] = {
    "german": ["Das funktioniert leider nicht, sehr frustrierend.",
               "Ich komme hier einfach nicht weiter."],
    "french": ["Ça ne marche pas du tout, très frustrant.",
               "Je n'arrive pas à m'en sortir."],
    "italian": ["Non funziona affatto, davvero frustrante.",
                "Non riesco proprio a procedere."],
}
POS_GENERIC: dict[str, list[str]] = {
    "german": ["Hat super geklappt, danke!", "Sehr hilfreich, vielen Dank."],
    "french": ["Ça a très bien marché, merci !", "Très utile, merci."],
    "italian": ["Ha funzionato benissimo, grazie!", "Molto utile, grazie."],
}

# language -> share. Majority english; the rest spread across the supported set.
LANGUAGE_WEIGHTS = {"english": 0.70, "german": 0.13, "french": 0.10, "italian": 0.07}

# explicit thumbs vs volunteered NLP feedback.
SOURCE_WEIGHTS = {"explicit": 0.60, "nlp": 0.40}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _weighted_choice(rng: random.Random, weights: dict) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _spike_window(now: datetime, days: int) -> tuple[datetime, datetime]:
    """A short, ~5-day window roughly 60% of the way back, for the
    virtual-sessions "outage". Kept tight so the extra negatives concentrate
    into a sharp, unmistakable spike on the trend rather than a low plateau.
    For days=60 / now=today this lands ~38..33 days ago."""
    center = now - timedelta(days=days * 0.6)
    return center - timedelta(days=2.5), center + timedelta(days=2.5)


def _attribution_rows(rng: random.Random, process: str, cfg: dict):
    """Return (method, rows) for a negative entry, mirroring what the live
    AttributionResolver would write — but assigned directly (deterministic, no
    vector store needed). ~70% citation (blame split 1/N), ~30% embedding."""
    doc, dept = cfg["doc"], cfg["dept"]
    if rng.random() < 0.30:
        # embedding: orphan/volunteered feedback resolved to its nearest doc.
        distance = round(rng.uniform(0.18, 0.42), 3)
        return "embedding", [
            AttributionRow(doc, None, process, dept, 1.0, "embedding", distance)
        ]
    # citation: the rated answer's cited docs, blame split evenly.
    docs = [(doc, process, dept)]
    if rng.random() < 0.30 and process in SECONDARY_DOC:
        sdoc, sproc, sdept = SECONDARY_DOC[process]
        docs.append((sdoc, sproc, sdept))
    weight = round(1.0 / len(docs), 4)
    rows = [AttributionRow(d, None, p, dp, weight, "citation") for d, p, dp in docs]
    return "citation", rows


def _message(rng: random.Random, process: str, lang: str, negative: bool) -> str:
    if lang == "english":
        bank = (NEG_MESSAGES if negative else POS_MESSAGES)[process]
    else:
        bank = (NEG_GENERIC if negative else POS_GENERIC)[lang]
    return rng.choice(bank)


def _backdate(rng: random.Random, now: datetime, days: int) -> datetime:
    """A uniformly random instant within the [now-days, now] window."""
    secs = rng.uniform(0, days * 86400)
    return now - timedelta(seconds=secs)


def _make_entry(
    rng: random.Random,
    *,
    process: str,
    cfg: dict,
    now: datetime,
    days: int,
    sessions: list[UUID],
    spike: tuple[datetime, datetime],
    force_negative: bool = False,
    force_date: datetime | None = None,
) -> tuple[FeedbackEntry, bool]:
    """Build one (un-persisted) FeedbackEntry plus its negativity flag."""
    created_at = force_date or _backdate(rng, now, days)

    neg_rate = cfg["neg_rate"]
    if cfg.get("spike") and spike[0] <= created_at <= spike[1]:
        neg_rate = 0.85          # the outage week
    negative = force_negative or rng.random() < neg_rate

    lang = _weighted_choice(rng, LANGUAGE_WEIGHTS)
    source = _weighted_choice(rng, SOURCE_WEIGHTS)
    emotion = rng.choice(cfg["neg"] if negative else cfg["pos"])
    text = _message(rng, process, lang, negative)
    sid = rng.choice(sessions)

    if source == "explicit":
        rating = "down" if negative else "up"
        # A down-vote carries a comment ~60% of the time; an up-vote rarely does.
        comment = text if (negative and rng.random() < 0.6) else None
        entry = FeedbackEntry(
            session_id=sid,
            turn_id=new_id(),           # synthetic answer id (real Turn in --full)
            tenant_id=DEMO_TENANT_ID,
            language=lang,
            emotion=emotion,
            message=comment or "",
            source="explicit",
            rating=rating,
            comment=comment,
            created_at=created_at,
        )
    else:
        entry = FeedbackEntry(
            session_id=sid,
            turn_id=None,               # volunteered, not tied to one answer
            tenant_id=DEMO_TENANT_ID,
            language=lang,
            emotion=emotion,
            message=text,
            source="nlp",
            rating=None,
            comment=None,
            created_at=created_at,
        )
    return entry, negative


def _persist_full(sessions_repo: SessionRepository, entry: FeedbackEntry, attr_rows) -> None:
    """--full: back the entry with a real assistant Turn (and citations for
    negatives) so the citation→attribution chain exists end to end. The charts
    don't read any of this; it's for end-to-end realism / drill-down demos."""
    turn = sessions_repo.append_turn(
        entry.session_id, "assistant", entry.message or "(answer)",
        language=entry.language, tenant_id=DEMO_TENANT_ID,
    )
    with DbSession(sessions_repo._engine) as db:
        stored = db.get(FeedbackEntry, entry.id)
        if stored is not None:
            stored.turn_id = turn.id
            db.add(stored)
            db.commit()
    if attr_rows:
        sessions_repo.add_citations(
            turn.id,
            [(r.source, r.section, r.process, r.department) for r in attr_rows],
            tenant_id=DEMO_TENANT_ID,
        )


# ---------------------------------------------------------------------------
# Seed / reset / ensure
# ---------------------------------------------------------------------------

def seed(
    *,
    engine: Engine,
    count: int = DEFAULT_COUNT,
    days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
    full: bool = False,
    now: datetime | None = None,
) -> dict:
    """Generate `count` feedback entries (+ attribution for the negatives).

    Deterministic for a fixed `seed`. Dates are relative to `now` (default: the
    current UTC instant), so the dataset is always "the last `days` days".
    """
    now = now or utcnow()
    rng = random.Random(seed)
    sessions_repo = SessionRepository(engine)
    feedback = FeedbackRepository(engine)

    # 1) Pool of demo sessions (always — FeedbackEntry.session_id references one).
    n_sessions = max(20, count // 8)
    session_ids: list[UUID] = []
    for i in range(n_sessions):
        sid = new_id()
        sessions_repo.get_or_create(
            sid, tenant_id=DEMO_TENANT_ID, user_id=f"demo-user-{i % 12:02d}"
        )
        session_ids.append(sid)

    spike = _spike_window(now, days)
    processes = list(PROCESS_CONFIG)
    p_weights = [PROCESS_CONFIG[p]["volume"] for p in processes]

    seeded_neg = 0
    per_process: dict[str, int] = {p: 0 for p in processes}

    def _persist(entry: FeedbackEntry, negative: bool, process: str, cfg: dict) -> None:
        nonlocal seeded_neg
        stored = feedback.add(entry)
        per_process[process] += 1
        rows = None
        if negative:
            seeded_neg += 1
            method, rows = _attribution_rows(rng, process, cfg)
            feedback.replace_attributions(
                stored.id, method, rows, tenant_id=DEMO_TENANT_ID
            )
        if full:
            _persist_full(sessions_repo, stored, rows)

    # 2) Main draw: pick a process ∝ volume, build + persist.
    for _ in range(count):
        process = rng.choices(processes, weights=p_weights, k=1)[0]
        cfg = PROCESS_CONFIG[process]
        entry, negative = _make_entry(
            rng, process=process, cfg=cfg, now=now, days=days,
            sessions=session_ids, spike=spike,
        )
        _persist(entry, negative, process, cfg)

    # 3) Inject extra negatives into the virtual-sessions outage window so it is
    #    unmistakable on the trend chart. Scales with volume (so it stays a sharp
    #    spike, not a ripple, as `count` grows) but is capped so instrument-
    #    booking remains the clear #1 chronic hotspot.
    spike_extra = max(8, min(count // 25, 80))
    cfg = PROCESS_CONFIG["virtual-sessions"]
    for _ in range(spike_extra):
        when = spike[0] + timedelta(seconds=rng.uniform(0, (spike[1] - spike[0]).total_seconds()))
        entry, negative = _make_entry(
            rng, process="virtual-sessions", cfg=cfg, now=now, days=days,
            sessions=session_ids, spike=spike, force_negative=True, force_date=when,
        )
        _persist(entry, negative, "virtual-sessions", cfg)

    total = sum(per_process.values())
    return {
        "total": total,
        "negative": seeded_neg,
        "negative_rate": seeded_neg / total if total else 0.0,
        "per_process": per_process,
        "sessions": len(session_ids),
        "spike_extra": spike_extra,
        "spike_window": (spike[0].date().isoformat(), spike[1].date().isoformat()),
    }


def reset(engine: Engine) -> dict:
    """Hard-delete every synthetic row (anything under the demo tenant)."""
    counts: dict[str, int] = {}
    # children first (FKs aren't enforced on sqlite, but keep it correct anyway).
    for model in (FeedbackAttribution, FeedbackEntry, TurnCitation, Turn, Session):
        with DbSession(engine) as db:
            result = db.execute(
                delete(model).where(model.tenant_id == DEMO_TENANT_ID)
            )
            counts[model.__name__] = result.rowcount or 0
            db.commit()
    return counts


def demo_feedback_count(engine: Engine) -> int:
    """How many feedback entries currently exist under the demo tenant."""
    return FeedbackRepository(engine).summary(tenant_id=DEMO_TENANT_ID)["total"]


def ensure_demo_feedback(
    engine: Engine,
    *,
    enabled: bool = True,
    count: int = DEFAULT_COUNT,
    days: int = DEFAULT_DAYS,
    seed_value: int = DEFAULT_SEED,
) -> None:
    """Idempotently make sure the demo feedback exists (called on startup).

    Seeds once when the demo tenant has no feedback yet; otherwise no-op. This
    is what keeps the ``/admin`` dashboard populated on any fresh database —
    the data is regenerated deterministically the first time the app boots
    against an empty DB. Re-seeding from scratch is a manual op (the CLI's
    ``--reset``); this never duplicates.
    """
    if not enabled:
        return
    try:
        existing = demo_feedback_count(engine)
    except Exception:
        # A schema not yet migrated, etc. Never let demo seeding break startup.
        logger.exception("demo_seed.precheck_failed")
        return
    if existing:
        logger.info("demo_seed.skip", extra={"existing": existing})
        return
    try:
        result = seed(engine=engine, count=count, days=days, seed=seed_value)
    except Exception:
        logger.exception("demo_seed.failed")
        return
    logger.info(
        "demo_seed.seeded",
        extra={
            "total": result["total"],
            "negative": result["negative"],
            "negative_rate": round(result["negative_rate"], 3),
        },
    )
