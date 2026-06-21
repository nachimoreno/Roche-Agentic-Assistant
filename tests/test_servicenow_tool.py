"""
test_servicenow_tool.py
-----------------------
Tests for the ServiceNow tool — both fast mocked tests and a live test
against a real developer instance.

Run fast tests (no credentials needed):
    pytest tests/test_servicenow_tool.py -v

Run live test (requires .env with ServiceNow credentials):
    pytest tests/test_servicenow_tool.py -m live -v -s
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

load_dotenv()

from servicenow_tool import (
    IncidentPayload,
    ServiceNowClient,
    create_servicenow_incident,
    SERVICENOW_TOOL_DEFINITION,
)


# ---------------------------------------------------------------------------
# IncidentPayload tests
# ---------------------------------------------------------------------------

class TestIncidentPayload:

    def test_required_field_only(self):
        p = IncidentPayload(short_description="Centrifuge broken")
        assert p.short_description == "Centrifuge broken"
        assert p.urgency == "3"
        assert p.category == "hardware"

    def test_to_api_payload_has_required_keys(self):
        p = IncidentPayload(short_description="BioLIMS not accessible")
        payload = p.to_api_payload()
        assert "short_description" in payload
        assert "category" in payload
        assert "urgency" in payload

    def test_custom_urgency(self):
        p = IncidentPayload(short_description="Critical failure", urgency="1")
        assert p.to_api_payload()["urgency"] == "1"

    def test_category_access(self):
        p = IncidentPayload(short_description="Cannot log in", category="access")
        assert p.to_api_payload()["category"] == "access"


# ---------------------------------------------------------------------------
# ServiceNowClient tests
# ---------------------------------------------------------------------------

class TestServiceNowClient:

    def test_raises_without_credentials(self):
        with pytest.raises(ValueError, match="credentials missing"):
            ServiceNowClient(instance_url="", username="", password="")

    def test_initializes_with_credentials(self):
        client = ServiceNowClient(
            instance_url="https://dev12345.service-now.com",
            username="admin",
            password="password",
        )
        assert "dev12345" in client.instance_url


# ---------------------------------------------------------------------------
# create_servicenow_incident tool function tests
# ---------------------------------------------------------------------------

class TestCreateServiceNowIncident:
    """All tests in this class exercise the REAL ServiceNowClient code path
    (mocked at the requests.post level) by forcing USE_MOCK=False."""

    @pytest.fixture(autouse=True)
    def force_real_client(self):
        import servicenow_tool
        original = servicenow_tool.USE_MOCK
        servicenow_tool.USE_MOCK = False
        yield
        servicenow_tool.USE_MOCK = original

    def _mock_response(self, number="INC0012345"):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": {
                "number":            number,
                "sys_id":            "abc123",
                "short_description": "Test incident",
                "state":             {"value": "1"},
            }
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @patch("servicenow_tool.requests.post")
    def test_returns_ticket_number_on_success(self, mock_post):
        mock_post.return_value = self._mock_response("INC0099999")
        with patch.dict(os.environ, {
            "SERVICENOW_INSTANCE": "https://dev12345.service-now.com",
            "SERVICENOW_USERNAME": "admin",
            "SERVICENOW_PASSWORD": "pass",
        }):
            result = create_servicenow_incident(
                short_description="PCR machine not responding"
            )
        assert "INC0099999" in result
        assert "✅" in result

    @patch("servicenow_tool.requests.post")
    def test_confirmation_includes_summary(self, mock_post):
        mock_post.return_value = self._mock_response()
        with patch.dict(os.environ, {
            "SERVICENOW_INSTANCE": "https://dev12345.service-now.com",
            "SERVICENOW_USERNAME": "admin",
            "SERVICENOW_PASSWORD": "pass",
        }):
            result = create_servicenow_incident(
                short_description="Scanner not connecting",
                category="hardware",
                urgency="2",
            )
        assert "Scanner not connecting" in result
        assert "Medium" in result

    def test_returns_warning_without_credentials(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in ["SERVICENOW_INSTANCE","SERVICENOW_USERNAME","SERVICENOW_PASSWORD"]:
                os.environ.pop(k, None)
            result = create_servicenow_incident("Test issue")
        assert "⚠️" in result
        assert "not configured" in result

    @patch("servicenow_tool.requests.post")
    def test_handles_http_error_gracefully(self, mock_post):
        import requests as req
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_post.return_value.raise_for_status.side_effect = req.HTTPError(
            response=mock_resp
        )
        with patch.dict(os.environ, {
            "SERVICENOW_INSTANCE": "https://dev12345.service-now.com",
            "SERVICENOW_USERNAME": "admin",
            "SERVICENOW_PASSWORD": "wrongpassword",
        }):
            result = create_servicenow_incident("Test issue")
        assert "⚠️" in result

    @patch("servicenow_tool.requests.post")
    def test_handles_connection_error_gracefully(self, mock_post):
        import requests as req
        mock_post.side_effect = req.ConnectionError()
        with patch.dict(os.environ, {
            "SERVICENOW_INSTANCE": "https://dev12345.service-now.com",
            "SERVICENOW_USERNAME": "admin",
            "SERVICENOW_PASSWORD": "pass",
        }):
            result = create_servicenow_incident("Test issue")
        assert "⚠️" in result
        assert "connection" in result.lower()


# ---------------------------------------------------------------------------
# Tool definition shape test
# ---------------------------------------------------------------------------

class TestToolDefinition:

    def test_has_required_keys(self):
        assert SERVICENOW_TOOL_DEFINITION["type"] == "function"
        fn = SERVICENOW_TOOL_DEFINITION["function"]
        assert fn["name"] == "create_servicenow_incident"
        assert "description" in fn
        assert "parameters" in fn

    def test_short_description_is_required(self):
        required = SERVICENOW_TOOL_DEFINITION["function"]["parameters"]["required"]
        assert "short_description" in required

    def test_category_enum_values(self):
        props = SERVICENOW_TOOL_DEFINITION["function"]["parameters"]["properties"]
        assert set(props["category"]["enum"]) == {"hardware", "software", "access", "network"}

    def test_urgency_enum_values(self):
        props = SERVICENOW_TOOL_DEFINITION["function"]["parameters"]["properties"]
        assert set(props["urgency"]["enum"]) == {"1", "2", "3"}


# ---------------------------------------------------------------------------
# Live test — skipped unless credentials present
# ---------------------------------------------------------------------------

LIVE = pytest.mark.skipif(
    not all([
        os.getenv("SERVICENOW_INSTANCE"),
        os.getenv("SERVICENOW_USERNAME"),
        os.getenv("SERVICENOW_PASSWORD"),
    ]),
    reason="Live ServiceNow tests require SERVICENOW_INSTANCE, SERVICENOW_USERNAME, SERVICENOW_PASSWORD",
)


@LIVE
def test_live_create_incident():
    """Creates a real incident on your ServiceNow dev instance."""
    result = create_servicenow_incident(
        short_description = "[TEST] Roche Assistant integration test — please ignore",
        description       = "Automated test from the Roche Agentic Assistant capstone project.",
        category          = "software",
        urgency           = "3",
    )
    print(f"\nLive result:\n{result}")
    assert "✅" in result
    assert "INC" in result


@LIVE
def test_live_incident_has_ticket_number():
    result = create_servicenow_incident(
        short_description="[TEST] Second integration test — please ignore",
        urgency="3",
    )
    # Extract ticket number from response
    lines = result.split("\n")
    ticket_line = [l for l in lines if "Ticket number" in l]
    assert len(ticket_line) == 1
    assert "INC" in ticket_line[0]
