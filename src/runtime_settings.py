"""
runtime_settings.py
-------------------
Live, admin-tunable configuration for the running assistant.

`settings.py` is the *boot* configuration (env + defaults, read once at start).
This module is its runtime complement: a small set of knobs an admin can change
from the /settings surface and have take effect on the *already-built* assistant
— no restart, persisted across restarts.

It does that by holding references to the live components it tunes (the
`RAGAgent`, the `GroqClient`, the `DocumentStore`) and mutating them in place.
A few knobs (demo seeding, the onboarding window) aren't held on a component;
those live in `_extra` and are read by the relevant code paths via `get()`.

All component coupling is centralised in `_build_accessors` so there is exactly
one place that knows which attribute backs each key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent import RAGAgent
from llm import GroqClient
from repositories import SettingsRepository
from retrieval import DocumentStore
from settings import Settings


logger = logging.getLogger(__name__)

# Default onboarding window (days) for the analytics "newcomer" funnel. Mirrors
# the analytics endpoint's own default; overridable from /settings.
DEFAULT_NEWCOMER_DAYS = 14


@dataclass(frozen=True)
class ParamSpec:
    """Metadata for one tunable knob — drives validation and the /settings UI."""
    key: str
    group: str
    label: str
    help: str
    type: str                                  # float | int | bool | enum | string
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[tuple[str, ...]] = None
    unit: Optional[str] = None
    # True: takes effect on the running assistant immediately. False: persisted,
    # but only read on the next startup (e.g. demo seeding happens at boot).
    live: bool = True


# The groups the UI renders, in order. (id, title, description)
GROUPS: tuple[tuple[str, str, str], ...] = (
    ("confidence", "Confidence thresholds",
     "When the assistant answers, warns, or declines based on retrieval strength."),
    ("retrieval", "Retrieval",
     "How much context is pulled per question, and how it's ranked."),
    ("llm", "Answer generation",
     "How the language model turns retrieved context into an answer."),
    ("demo", "Demo & onboarding",
     "Synthetic-data seeding and the analytics newcomer window."),
)


PARAM_SPECS: tuple[ParamSpec, ...] = (
    # ---- Confidence thresholds (all on a [0, 1] retrieval-score scale) -------
    ParamSpec(
        "retrieval_min_dense", "confidence", "Decline floor (dense)",
        "Top dense cosine below this (and below the lexical rescue bar) means the "
        "query is treated as off-domain and declined — no answer, no citations. "
        "Lower = answers more questions; higher = declines more readily.",
        "float", minimum=0.0, maximum=1.0, step=0.01,
    ),
    ParamSpec(
        "retrieval_warn_dense", "confidence", "Verify-this warning",
        "Answers given with a top dense cosine below this get a low-confidence "
        '"verify against the source" caution badge. Sits above the decline floor.',
        "float", minimum=0.0, maximum=1.0, step=0.01,
    ),
    ParamSpec(
        "retrieval_min_lexical", "confidence", "Lexical rescue bar",
        "A low-dense query is still answered if its normalised BM25 keyword match "
        "is at least this high (a near-exact code / part-number hit). Only used in "
        "hybrid mode.",
        "float", minimum=0.0, maximum=1.0, step=0.01,
    ),
    # ---- Retrieval ----------------------------------------------------------
    ParamSpec(
        "top_k", "retrieval", "Chunks per query",
        "How many document chunks are retrieved and fed to the model as context. "
        "More = broader grounding but a longer, noisier prompt.",
        "int", minimum=1, maximum=20, step=1, unit="chunks",
    ),
    ParamSpec(
        "retrieval_mode", "retrieval", "Retrieval mode",
        "Hybrid fuses dense embeddings with BM25 keyword search (better on exact "
        "tokens — codes, part numbers, app names). Dense uses embeddings only.",
        "enum", choices=("hybrid", "dense"),
    ),
    # ---- Answer generation --------------------------------------------------
    ParamSpec(
        "llm_temperature", "llm", "Temperature",
        "Sampling randomness. 0.0 is deterministic and tightly grounded; higher "
        "values make wording more varied (and answers less repeatable).",
        "float", minimum=0.0, maximum=1.5, step=0.05,
    ),
    ParamSpec(
        "llm_max_tokens", "llm", "Max answer tokens",
        "Upper bound on answer length. Lower keeps answers terse; higher allows "
        "longer, more detailed responses.",
        "int", minimum=128, maximum=4096, step=64, unit="tokens",
    ),
    ParamSpec(
        "model_name", "llm", "Model",
        "The Groq model id used to generate answers. Changing it swaps the model "
        "on the running client.",
        "string",
    ),
    # ---- Demo & onboarding --------------------------------------------------
    ParamSpec(
        "seed_demo_feedback", "demo", "Seed demo feedback",
        "On a fresh database, populate the analytics dashboard with a synthetic "
        "feedback dataset so it's always demoable. Takes effect on next restart.",
        "bool", live=False,
    ),
    ParamSpec(
        "demo_feedback_count", "demo", "Demo feedback volume",
        "How many synthetic feedback rows to seed. Higher smooths the trend line. "
        "Takes effect on next restart.",
        "int", minimum=0, maximum=20000, step=100, unit="rows", live=False,
    ),
    ParamSpec(
        "newcomer_days", "demo", "Newcomer window",
        "Default tenure (days) that counts as a 'newcomer' in the onboarding "
        "funnel analytics.",
        "int", minimum=1, maximum=365, step=1, unit="days",
    ),
)

_SPECS_BY_KEY: dict[str, ParamSpec] = {s.key: s for s in PARAM_SPECS}


def _clamp(value: float, spec: ParamSpec) -> float:
    if spec.minimum is not None:
        value = max(spec.minimum, value)
    if spec.maximum is not None:
        value = min(spec.maximum, value)
    return value


def _coerce(spec: ParamSpec, raw: Any) -> Any:
    """Validate/normalise a raw input for `spec`. Numbers clamp to range;
    enums must match a choice; raises ValueError on anything uncoercible."""
    if spec.type == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        raise ValueError("expected a boolean")
    if spec.type == "enum":
        text = str(raw)
        if spec.choices and text not in spec.choices:
            raise ValueError(f"must be one of {spec.choices}")
        return text
    if spec.type == "int":
        return int(_clamp(int(round(float(raw))), spec))
    if spec.type == "float":
        return float(_clamp(float(raw), spec))
    # string
    text = str(raw).strip()
    if not text:
        raise ValueError("must not be empty")
    return text


class RuntimeSettings:
    """Holds the live components and applies tunable overrides to them.

    Construct after the assistant is built, then call `load_persisted()` to
    replay any stored overrides onto the running components.
    """

    def __init__(
        self,
        *,
        agent: RAGAgent,
        llm: GroqClient,
        docs: DocumentStore,
        repo: SettingsRepository,
        settings: Settings,
    ) -> None:
        self._agent = agent
        self._llm = llm
        self._docs = docs
        self._repo = repo
        # Knobs not backed by a live component (read by their code path via get).
        self._extra: dict[str, Any] = {
            "seed_demo_feedback": settings.seed_demo_feedback,
            "demo_feedback_count": settings.demo_feedback_count,
            "newcomer_days": DEFAULT_NEWCOMER_DAYS,
        }
        self._accessors = self._build_accessors()

    def _build_accessors(
        self,
    ) -> dict[str, tuple[Callable[[], Any], Callable[[Any], None]]]:
        """key -> (getter, setter). The single point that couples a tunable key
        to the component attribute that backs it."""
        a, llm, docs, extra = self._agent, self._llm, self._docs, self._extra

        def ex_get(k: str) -> Callable[[], Any]:
            return lambda: extra[k]

        def ex_set(k: str) -> Callable[[Any], None]:
            return lambda v: extra.__setitem__(k, v)

        return {
            "retrieval_min_dense": (
                lambda: a.min_dense, lambda v: setattr(a, "min_dense", v)),
            "retrieval_warn_dense": (
                lambda: a.warn_dense, lambda v: setattr(a, "warn_dense", v)),
            "retrieval_min_lexical": (
                lambda: a.min_lexical, lambda v: setattr(a, "min_lexical", v)),
            "top_k": (lambda: a.top_k, lambda v: setattr(a, "top_k", v)),
            "retrieval_mode": (
                lambda: "hybrid" if docs.hybrid else "dense",
                lambda v: setattr(docs, "hybrid", v == "hybrid")),
            "llm_temperature": (
                lambda: a.temperature, lambda v: setattr(a, "temperature", v)),
            "llm_max_tokens": (
                lambda: a.max_tokens, lambda v: setattr(a, "max_tokens", v)),
            "model_name": (lambda: llm.model, lambda v: setattr(llm, "model", v)),
            "seed_demo_feedback": (
                ex_get("seed_demo_feedback"), ex_set("seed_demo_feedback")),
            "demo_feedback_count": (
                ex_get("demo_feedback_count"), ex_set("demo_feedback_count")),
            "newcomer_days": (ex_get("newcomer_days"), ex_set("newcomer_days")),
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """The current effective value for one key (component or extra)."""
        acc = self._accessors.get(key)
        return acc[0]() if acc else default

    def current(self) -> dict[str, Any]:
        """Every tunable's current effective value."""
        return {k: get() for k, (get, _set) in self._accessors.items()}

    def schema(self) -> dict[str, Any]:
        """Grouped param metadata + current values, for the /settings UI."""
        cur = self.current()
        groups = []
        for gid, title, desc in GROUPS:
            params = []
            for spec in PARAM_SPECS:
                if spec.group != gid:
                    continue
                params.append({
                    "key": spec.key,
                    "label": spec.label,
                    "help": spec.help,
                    "type": spec.type,
                    "min": spec.minimum,
                    "max": spec.maximum,
                    "step": spec.step,
                    "choices": list(spec.choices) if spec.choices else None,
                    "unit": spec.unit,
                    "live": spec.live,
                    "value": cur.get(spec.key),
                })
            groups.append(
                {"id": gid, "title": title, "description": desc, "params": params}
            )
        return {"groups": groups}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def apply(
        self,
        updates: dict[str, Any],
        *,
        updated_by: Optional[str] = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Validate, apply to the live components, and (optionally) persist.

        Unknown keys are ignored; numbers are clamped to range. Raises ValueError
        if any known key carries an uncoercible value (nothing is applied then).
        Returns the coerced values that were applied.
        """
        coerced: dict[str, Any] = {}
        for key, raw in updates.items():
            spec = _SPECS_BY_KEY.get(key)
            if spec is None:
                continue
            try:
                coerced[key] = _coerce(spec, raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid value for {key!r}: {raw!r} ({exc})")

        for key, value in coerced.items():
            self._accessors[key][1](value)

        if persist and coerced:
            self._repo.set_many(coerced, updated_by=updated_by)
        if coerced:
            logger.info("settings.applied", extra={"keys": sorted(coerced)})
        return coerced

    def load_persisted(self) -> None:
        """Replay stored overrides onto the running components (no re-persist)."""
        try:
            stored = self._repo.get_all()
        except Exception:                                  # pragma: no cover
            logger.exception("settings.load_failed")
            return
        if stored:
            self.apply(stored, persist=False)
            logger.info("settings.restored", extra={"keys": sorted(stored)})
