"""
servicenow_tool.py
------------------
ServiceNow incident creation tool for the Roche Agentic Assistant.

HOW IT FITS IN:
  The agent already has a tool-calling pattern in agent.py.
  This file adds one new tool: `create_servicenow_incident`.
  The agent calls it automatically when it detects the scientist
  has an incident to report (broken device, access issue, etc.).

SETUP:
  1. Create a free developer instance at developer.servicenow.com
  2. Add to .env:
       SERVICENOW_INSTANCE=https://devXXXXXX.service-now.com
       SERVICENOW_USERNAME=admin
       SERVICENOW_PASSWORD=your_password

FLOW:
  Scientist: "My centrifuge isn't working"
      ↓
  Agent detects incident intent
      ↓
  Agent collects: description, device/system, urgency
      ↓
  create_servicenow_incident() called
      ↓
  Returns: "Ticket INC0012345 created successfully"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

from mock_servicenow_client import MockServiceNowClient

logger = logging.getLogger(__name__)

# Toggle between mock and real ServiceNow.
# Set SERVICENOW_USE_MOCK=false in .env once your real dev instance is ready —
# everything else (tool definition, agent wiring, tests) stays the same.
#
# This module-level default is the back-compat path used when no explicit
# ServiceNowConfig is passed (e.g. the unit tests). The running app injects a
# ServiceNowConfig built from Settings instead, so config flows through the
# normal settings seam rather than being read from os.environ here.
USE_MOCK = os.getenv("SERVICENOW_USE_MOCK", "true").lower() == "true"


@dataclass
class ServiceNowConfig:
    """Resolved ServiceNow connection config, injected by the caller.

    Built from `Settings` in the composition root and passed into
    `create_servicenow_incident`. When it is absent the function falls back to
    the module-level `USE_MOCK` toggle and env-var credentials, preserving the
    original standalone behaviour the unit tests rely on.
    """

    use_mock: bool = True
    instance: str = ""
    username: str = ""
    password: str = ""

# ---------------------------------------------------------------------------
# Incident dataclass — what we collect from the scientist
# ---------------------------------------------------------------------------

@dataclass
class IncidentPayload:
    """
    Fields collected from the scientist before creating the ticket.
    Only short_description is truly required — everything else has a default.
    """
    short_description: str                          # e.g. "Centrifuge not responding in Lab 4B"
    description: str        = ""                    # longer detail the scientist provides
    category: str           = "hardware"            # hardware | software | access | network
    urgency: str            = "3"                   # 1=High, 2=Medium, 3=Low
    caller_id: str          = "scientist"           # in prod: link to Roche user account

    def to_api_payload(self) -> dict:
        return {
            "short_description": self.short_description,
            "description":       self.description,
            "category":          self.category,
            "urgency":           self.urgency,
            "caller_id":         self.caller_id,
            "impact":            self.urgency,      # mirror urgency as impact
        }


# ---------------------------------------------------------------------------
# ServiceNow client
# ---------------------------------------------------------------------------

class ServiceNowClient:
    """
    Thin wrapper around the ServiceNow Table API.
    Uses basic auth — fine for a dev/demo instance.

    Reads credentials from env vars:
        SERVICENOW_INSTANCE  — full URL e.g. https://devXXXXXX.service-now.com
        SERVICENOW_USERNAME  — usually 'admin' on dev instances
        SERVICENOW_PASSWORD  — your dev instance password
    """

    INCIDENTS_ENDPOINT = "/api/now/table/incident"

    def __init__(
        self,
        instance_url: Optional[str] = None,
        username:     Optional[str] = None,
        password:     Optional[str] = None,
    ):
        self.instance_url = (instance_url or os.getenv("SERVICENOW_INSTANCE", "")).rstrip("/")
        self.username     = username or os.getenv("SERVICENOW_USERNAME", "")
        self.password     = password or os.getenv("SERVICENOW_PASSWORD", "")

        if not all([self.instance_url, self.username, self.password]):
            raise ValueError(
                "ServiceNow credentials missing. Set SERVICENOW_INSTANCE, "
                "SERVICENOW_USERNAME, SERVICENOW_PASSWORD in your .env"
            )

    def create_incident(self, payload: IncidentPayload) -> dict:
        """
        POST to ServiceNow Table API to create an incident.
        Returns the created incident record dict.
        Raises on HTTP error.
        """
        url = self.instance_url + self.INCIDENTS_ENDPOINT

        response = requests.post(
            url,
            json    = payload.to_api_payload(),
            auth    = HTTPBasicAuth(self.username, self.password),
            headers = {
                "Content-Type": "application/json",
                "Accept":       "application/json",
            },
            timeout = 15,
        )

        response.raise_for_status()
        result = response.json().get("result", {})

        logger.info(
            "servicenow.incident.created",
            extra={
                "number":            result.get("number"),
                "short_description": payload.short_description,
            },
        )

        return result


# ---------------------------------------------------------------------------
# Tool function — this is what the agent calls
# ---------------------------------------------------------------------------

def create_servicenow_incident(
    short_description: str,
    description:       str = "",
    category:          str = "hardware",
    urgency:           str = "3",
    *,
    config: Optional[ServiceNowConfig] = None,
) -> str:
    """
    Create a ServiceNow incident from the agent.

    This is the function the orchestrator calls when an incident is confirmed.
    Returns a human-readable confirmation string for the scientist.

    Parameters
    ----------
    short_description : str
        One-line summary of the issue (collected from the scientist).
    description : str
        Full details the scientist provided.
    category : str
        "hardware" | "software" | "access" | "network"
    urgency : str
        "1" = High, "2" = Medium, "3" = Low
    config : ServiceNowConfig, optional
        Injected connection config. When omitted, falls back to the module-level
        `USE_MOCK` toggle and env-var credentials (the standalone/test path).

    Returns
    -------
    str
        Confirmation message with ticket number, or error message.
    """
    use_mock = config.use_mock if config is not None else USE_MOCK
    try:
        if use_mock:
            client = MockServiceNowClient()
        elif config is not None:
            client = ServiceNowClient(config.instance, config.username, config.password)
        else:
            client = ServiceNowClient()
        payload = IncidentPayload(
            short_description = short_description,
            description       = description,
            category          = category,
            urgency           = urgency,
        )
        result = client.create_incident(payload)

        number  = result.get("number", "unknown")

        mode_note = " _(simulated — demo mode)_" if use_mock else ""

        return (
            f"✅ Incident created successfully.{mode_note}\n"
            f"**Ticket number:** {number}\n"
            f"**Summary:** {short_description}\n"
            f"**Category:** {category}\n"
            f"**Urgency:** {'High' if urgency=='1' else 'Medium' if urgency=='2' else 'Low'}\n\n"
            f"The IT team has been notified and will follow up. "
            f"You can reference ticket **{number}** in any follow-up communications."
        )

    except ValueError as e:
        # Missing credentials
        logger.error("servicenow.credentials.missing", extra={"error": str(e)})
        return (
            "⚠️ ServiceNow is not configured. "
            "Please contact IT directly to report this issue."
        )

    except requests.HTTPError as e:
        logger.error("servicenow.http.error", extra={"error": str(e)})
        return (
            f"⚠️ Could not create ticket (HTTP {e.response.status_code}). "
            "Please try again or contact IT directly."
        )

    except requests.ConnectionError:
        logger.error("servicenow.connection.error")
        return (
            "⚠️ Could not reach ServiceNow. "
            "Please check your connection or contact IT directly."
        )

    except Exception as e:
        logger.error("servicenow.unexpected.error", extra={"error": str(e)})
        return f"⚠️ Unexpected error: {e}. Please contact IT directly."


# ---------------------------------------------------------------------------
# Tool definition (OpenAI/Anthropic function-calling schema)
# ---------------------------------------------------------------------------

# This repo's GroqClient does not use native function-calling — incidents are
# detected by the conversation classifier ("incident" type) and routed through
# incident_intake.IncidentIntake, which extracts these same fields. This schema
# is kept as the canonical parameter contract (and for a future provider that
# does support native tool-calling); incident_intake mirrors its field set.

SERVICENOW_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "create_servicenow_incident",
        "description": (
            "Creates a ServiceNow IT support ticket when a scientist reports a problem "
            "with a device, system, software, or access issue. "
            "Use this when the scientist describes something broken, not working, "
            "inaccessible, or needs IT support. "
            "Do NOT use this for general questions or document lookups."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "short_description": {
                    "type": "string",
                    "description": "One-line summary of the issue, e.g. 'Centrifuge not responding in Lab 4B'"
                },
                "description": {
                    "type": "string",
                    "description": "Full details provided by the scientist about the issue"
                },
                "category": {
                    "type": "string",
                    "enum": ["hardware", "software", "access", "network"],
                    "description": "Type of issue"
                },
                "urgency": {
                    "type": "string",
                    "enum": ["1", "2", "3"],
                    "description": "1=High (blocking work), 2=Medium, 3=Low"
                },
            },
            "required": ["short_description"],
        },
    },
}
