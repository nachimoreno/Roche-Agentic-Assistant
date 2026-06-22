"""
test_mock_servicenow_client.py
--------------------------------
Tests for the MockServiceNowClient — verifies it behaves like a drop-in
replacement for the real ServiceNowClient.

Run:
    pytest tests/test_mock_servicenow_client.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from mock_servicenow_client import MockServiceNowClient
from servicenow_tool import IncidentPayload


@pytest.fixture(autouse=True)
def reset_mock_client():
    """Reset ticket counter/history before every test for isolation."""
    MockServiceNowClient.reset()
    yield
    MockServiceNowClient.reset()


class TestMockServiceNowClient:

    def test_initializes_without_credentials(self):
        # Unlike the real client, mock should never require credentials
        client = MockServiceNowClient()
        assert client.instance_url

    def test_create_incident_returns_ticket_number(self):
        client = MockServiceNowClient()
        payload = IncidentPayload(short_description="Centrifuge broken")
        result = client.create_incident(payload)
        assert result["number"].startswith("INC00")

    def test_ticket_numbers_increment(self):
        client = MockServiceNowClient()
        p1 = IncidentPayload(short_description="Issue 1")
        p2 = IncidentPayload(short_description="Issue 2")

        r1 = client.create_incident(p1)
        r2 = client.create_incident(p2)

        num1 = int(r1["number"].replace("INC00", ""))
        num2 = int(r2["number"].replace("INC00", ""))
        assert num2 == num1 + 1

    def test_response_shape_matches_real_api(self):
        client = MockServiceNowClient()
        payload = IncidentPayload(short_description="Test issue", urgency="1")
        result = client.create_incident(payload)

        # Same keys a real ServiceNow Table API response would have
        for key in ["number", "sys_id", "short_description", "category", "urgency", "state"]:
            assert key in result

    def test_short_description_preserved(self):
        client = MockServiceNowClient()
        payload = IncidentPayload(short_description="PCR machine overheating")
        result = client.create_incident(payload)
        assert result["short_description"] == "PCR machine overheating"

    def test_get_all_tickets_returns_history(self):
        client = MockServiceNowClient()
        client.create_incident(IncidentPayload(short_description="Issue A"))
        client.create_incident(IncidentPayload(short_description="Issue B"))

        tickets = MockServiceNowClient.get_all_tickets()
        assert len(tickets) == 2
        descriptions = [t["short_description"] for t in tickets]
        assert "Issue A" in descriptions
        assert "Issue B" in descriptions

    def test_reset_clears_history_and_counter(self):
        client = MockServiceNowClient()
        client.create_incident(IncidentPayload(short_description="Issue A"))
        assert len(MockServiceNowClient.get_all_tickets()) == 1

        MockServiceNowClient.reset()
        assert len(MockServiceNowClient.get_all_tickets()) == 0

        result = client.create_incident(IncidentPayload(short_description="Fresh start"))
        assert result["number"] == "INC0010001"

    def test_shared_state_across_instances(self):
        """Class-level storage means separate instantiations share history —
        mimics how multiple agent calls would hit the same ServiceNow instance."""
        client_a = MockServiceNowClient()
        client_b = MockServiceNowClient()

        client_a.create_incident(IncidentPayload(short_description="From client A"))
        client_b.create_incident(IncidentPayload(short_description="From client B"))

        assert len(MockServiceNowClient.get_all_tickets()) == 2
