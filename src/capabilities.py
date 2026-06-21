"""
capabilities.py
---------------
The single source of truth for what the Roche Scientist Assistant can and
cannot do.

Self-knowledge lives here in code — it is never ingested into the document
corpus or retrieved — so a capability question answers correctly and
deterministically regardless of what retrieval returns or how a message is
classified. The agent injects ``CAPABILITIES.as_prompt_block()`` into its
system prompt on **every** answer turn, so the model always has its own
description to hand, even when a self-question is misrouted as operational.

Keep this tight (a dozen bullets): it is added to every answer turn's tokens.
When a planned feature ships (for example ServiceNow incident creation, later
in the agent-skills set), move its bullet from ``cannot_do`` to ``can_do`` and
bump ``as_of``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    """A dated, two-part description of the assistant's abilities."""

    as_of: str
    can_do: tuple[str, ...]
    cannot_do: tuple[str, ...]

    def as_prompt_block(self) -> str:
        """Render the capabilities as a bullet block for the system prompt."""
        can = "\n".join(f"- {item}" for item in self.can_do)
        cannot = "\n".join(f"- {item}" for item in self.cannot_do)
        return (
            f"What you can do today (as of {self.as_of}):\n"
            f"{can}\n\n"
            f"What you cannot do yet:\n"
            f"{cannot}"
        )


# Reflects code reality as of this date. Drive ingestion is built
# (google_drive_source.py, default document_source="google_drive"), so it is a
# can-do; ServiceNow incident creation is not built yet, so it stays in
# cannot-do until the ServiceNow skill lands.
CAPABILITIES = Capabilities(
    as_of="2026-06-21",
    can_do=(
        "Answer operational questions grounded in internal lab documentation "
        "ingested from the team's Google Drive — onboarding and access, "
        "navigating internal applications, incident reporting, instrument "
        "booking and calibration, sample stock, ordering chemicals and "
        "consumables, cleaning, decontamination and disinfection, waste "
        "management, lab sharing, campus and facilities, and virtual session "
        "troubleshooting. You cite the document and section each answer comes "
        "from.",
        "Explain how scientists work with you: they can type or tap the "
        "microphone to speak a question, tap \"Listen\" under an answer to "
        "hear it read aloud, click a citation to open its source document, and "
        "tap a suggested follow-up to continue. A Help (?) button in the top "
        "bar explains all of this. When asked how to use you or a specific "
        "feature, explain it from this.",
        "Point scientists to the right internal application when the action "
        "lives there — you explain where to go, but do not act inside that app.",
        "Capture feedback for the IT and documentation teams and detect its "
        "sentiment, so the most painful spots can be prioritized.",
        "Understand and reply in English, German, French, and Italian. The "
        "source documentation is English; you translate answers naturally.",
        "Continue the conversation across devices — chat history is tied to "
        "the scientist's account, not a single machine.",
    ),
    cannot_do=(
        "Create ServiceNow incidents directly from the conversation — that is "
        "planned; for now, explain the manual steps to open one.",
        "Perform actions inside other internal Roche applications on a "
        "scientist's behalf, such as booking an instrument or reserving stock.",
        "Look things up on the public web — answers come only from the "
        "internal corpus.",
    ),
)
