"""Tests for the v2 assistant protocol adapter (protocol/adapters.py)."""

import json
import pytest
from protocol.types import AssistantRequest, WorkspaceContext, SUPPORTED_DIAGRAM_TYPES
from protocol.adapters import (
    safe_json_loads,
    normalize_diagram_type,
    strip_diagram_prefix,
    parse_v2_payload,
    _unwrap_v2_envelope,
    _derive_diagram_summaries_from_snapshot,
)

from tests.conftest import make_v2_payload


# ---------------------------------------------------------------------------
# safe_json_loads
# ---------------------------------------------------------------------------

class TestSafeJsonLoads:
    def test_valid_json(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_returns_none_for_non_string(self):
        assert safe_json_loads(123) is None
        assert safe_json_loads(None) is None

    def test_returns_none_for_non_object(self):
        assert safe_json_loads("[1, 2]") is None

    def test_returns_none_for_invalid_json(self):
        assert safe_json_loads("{invalid}") is None

    def test_returns_none_for_empty_string(self):
        assert safe_json_loads("") is None
        assert safe_json_loads("  ") is None


# ---------------------------------------------------------------------------
# normalize_diagram_type
# ---------------------------------------------------------------------------

class TestNormalizeDiagramType:
    @pytest.mark.parametrize("dt", sorted(SUPPORTED_DIAGRAM_TYPES))
    def test_valid_types(self, dt):
        assert normalize_diagram_type(dt) == dt

    def test_invalid_type_returns_default(self):
        assert normalize_diagram_type("Bogus") == "ClassDiagram"

    def test_custom_default(self):
        assert normalize_diagram_type("Bogus", default="StateMachineDiagram") == "StateMachineDiagram"

    def test_non_string_returns_default(self):
        assert normalize_diagram_type(42) == "ClassDiagram"
        assert normalize_diagram_type(None) == "ClassDiagram"


# ---------------------------------------------------------------------------
# strip_diagram_prefix
# ---------------------------------------------------------------------------

class TestStripDiagramPrefix:
    def test_no_prefix(self):
        msg, dt = strip_diagram_prefix("Create a User class")
        assert msg == "Create a User class"
        assert dt is None

    def test_with_prefix(self):
        msg, dt = strip_diagram_prefix("[DIAGRAM_TYPE:ObjectDiagram] Create an instance")
        assert msg == "Create an instance"
        assert dt == "ObjectDiagram"

    def test_empty_string(self):
        msg, dt = strip_diagram_prefix("")
        assert msg == ""
        assert dt is None

    def test_non_string(self):
        msg, dt = strip_diagram_prefix(None)  # type: ignore[arg-type]
        assert msg == ""
        assert dt is None


# ---------------------------------------------------------------------------
# _unwrap_v2_envelope
# ---------------------------------------------------------------------------

class TestUnwrapV2Envelope:
    def test_unwraps_nested_v2_payload(self):
        inner = {
            "action": "user_message",
            "protocolVersion": "2.0",
            "message": "hello",
            "context": {"activeDiagramType": "ClassDiagram"},
        }
        raw = {"action": "user_message", "message": json.dumps(inner)}
        result = _unwrap_v2_envelope(raw)
        assert result["protocolVersion"] == "2.0"
        assert result["message"] == "hello"

    def test_passes_through_non_v2(self):
        raw = {"action": "user_message", "message": "plain text"}
        result = _unwrap_v2_envelope(raw)
        assert result["message"] == "plain text"

    def test_handles_non_dict(self):
        assert _unwrap_v2_envelope("not a dict") == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _derive_diagram_summaries_from_snapshot
# ---------------------------------------------------------------------------

class TestDeriveSummaries:
    def test_with_diagrams(self):
        snapshot = {
            "diagrams": {
                "ClassDiagram": {"id": "cd-1", "title": "My Classes"},
                "ObjectDiagram": {"id": "od-1"},
            }
        }
        summaries = _derive_diagram_summaries_from_snapshot(snapshot)
        assert len(summaries) == 2
        assert summaries[0]["diagramType"] == "ClassDiagram"
        assert summaries[0]["title"] == "My Classes"

    def test_no_diagrams(self):
        assert _derive_diagram_summaries_from_snapshot({}) == []

    def test_non_dict(self):
        assert _derive_diagram_summaries_from_snapshot(None) == []


# ---------------------------------------------------------------------------
# parse_v2_payload (integration)
# ---------------------------------------------------------------------------

class TestParseV2Payload:
    def test_basic_message(self):
        raw = make_v2_payload("Create a User class", "ClassDiagram")
        request = parse_v2_payload(raw)
        assert request.message == "Create a User class"
        assert request.diagram_type == "ClassDiagram"
        assert request.is_v2

    def test_with_active_model(self):
        model = {"elements": {"cls-1": {"name": "Foo"}}}
        raw = make_v2_payload("Modify User", "ClassDiagram", active_model=model)
        request = parse_v2_payload(raw)
        assert request.current_model == model

    def test_frontend_event_action(self):
        raw = make_v2_payload("", action="frontend_event")
        request = parse_v2_payload(raw)
        assert request.action == "frontend_event"

    def test_unknown_diagram_falls_back(self):
        inner = {
            "action": "user_message",
            "protocolVersion": "2.0",
            "message": "hello",
            "context": {"activeDiagramType": "FooDiagram"},
        }
        raw = {"action": "user_message", "message": json.dumps(inner)}
        request = parse_v2_payload(raw)
        assert request.diagram_type == "ClassDiagram"
