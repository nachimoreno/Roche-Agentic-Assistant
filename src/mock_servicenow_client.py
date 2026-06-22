"""
mock_servicenow_client.py
--------------------------
A drop-in replacement for ServiceNowClient that simulates incident creation
without needing a real ServiceNow instance.

WHY THIS EXISTS:
  Real ServiceNow developer instances are currently waitlisted. This mock
  has the EXACT same method signature as ServiceNowClient, so the agent's
  tool-calling code (create_servicenow_incident in servicenow_tool.py)
  doesn't need to change at all — just swap which client gets instantiated.

WHEN YOUR REAL INSTANCE ARRIVES:
  In servicenow_tool.py, change one line:
      client = MockServiceNowClient()
  back to:
      client = ServiceNowClient()
  Nothing else changes — same payload shape, same return shape, same tests.

WHAT IT SIMULATES:
  - Auto-incrementing ticket numbers (INC0010001, INC0010002, ...)
  - Realistic response shape matching ServiceNow's Table API
  - In-memory storage so you can show "all tickets created this session"
  - Small artificial delay (200ms) to feel like a real network call in demos
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import ClassVar

logger = logging.getLogger(__name__)


class MockServiceNowClient:
    """
    Drop-in replacement for ServiceNowClient. Same `create_incident()`
    method signature and return shape — the rest of the codebase doesn't
    need to know it's talking to a mock instead of the real API.

    Tickets are stored in-memory at the CLASS level so they persist across
    multiple calls within the same demo/process (but reset between runs).
    """

    # Class-level storage so all instances share the same ticket counter
    # and history — mimics a real shared ServiceNow instance during a demo.
    _tickets:        ClassVar[list[dict]] = []
    _ticket_counter: ClassVar[int]        = 10000  # starts at INC0010001

    # Matches ServiceNowClient's __init__ signature so callers don't need
    # to change anything when swapping clients in/out.
    def __init__(self, instance_url: str = "", username: str = "", password: str = ""):
        self.instance_url = instance_url or "https://mock-roche-instance.service-now.com"
        logger.info("mock_servicenow.initialized", extra={"instance": self.instance_url})

    def create_incident(self, payload) -> dict:
        """
        Simulates POST /api/now/table/incident.
        `payload` is an IncidentPayload (same object servicenow_tool.py builds).
        Returns a dict shaped like ServiceNow's real API response.
        """
        time.sleep(0.2)  # simulate network latency for a more realistic demo

        MockServiceNowClient._ticket_counter += 1
        number = f"INC00{MockServiceNowClient._ticket_counter}"
        sys_id = str(uuid.uuid4()).replace("-", "")

        record = {
            "number":            number,
            "sys_id":            sys_id,
            "short_description": payload.short_description,
            "description":       payload.description,
            "category":          payload.category,
            "urgency":           payload.urgency,
            "impact":            payload.urgency,
            "state":             {"value": "1", "display_value": "New"},
            "caller_id":         payload.caller_id,
            "sys_created_on":    datetime.now(timezone.utc).isoformat(),
        }

        MockServiceNowClient._tickets.append(record)

        logger.info(
            "mock_servicenow.incident.created",
            extra={"number": number, "short_description": payload.short_description},
        )

        return record

    @classmethod
    def get_all_tickets(cls) -> list[dict]:
        """Returns every ticket created this session — useful for a demo
        'here's everything the agent has filed so far' moment."""
        return cls._tickets

    @classmethod
    def reset(cls):
        """Clears ticket history and resets the counter. Useful between
        test runs or demo rehearsals so numbers start fresh."""
        cls._tickets = []
        cls._ticket_counter = 10000
