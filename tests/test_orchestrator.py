"""Tests for the workspace orchestrator (orchestrator/workspace_orchestrator.py)."""

import pytest
from orchestrator.workspace_orchestrator import (
    determine_target_diagram_type,
    determine_target_diagram_types,
    resolve_diagram_id,
    build_switch_diagram_action,
    _collect_explicit_targets,
    _rank_implicit_targets,
)
from protocol.types import AssistantRequest, WorkspaceContext


def _make_request(
    message: str,
    active_diagram_type: str = "ClassDiagram",
    project_snapshot=None,
) -> AssistantRequest:
    return AssistantRequest(
        message=message,
        diagram_type=active_diagram_type,
        context=WorkspaceContext(
            active_diagram_type=active_diagram_type,
            project_snapshot=project_snapshot,
        ),
    )


# ---------------------------------------------------------------------------
# _collect_explicit_targets
# ---------------------------------------------------------------------------

class TestCollectExplicitTargets:
    def test_single_target(self):
        targets = _collect_explicit_targets("create a class diagram for a hotel")
        assert targets == ["ClassDiagram"]

    def test_multiple_targets(self):
        targets = _collect_explicit_targets(
            "create a class diagram and a state machine diagram"
        )
        assert "ClassDiagram" in targets
        assert "StateMachineDiagram" in targets

    def test_no_target(self):
        assert _collect_explicit_targets("hello world") == []

    def test_quantum(self):
        targets = _collect_explicit_targets("create a quantum circuit")
        assert targets == ["QuantumCircuitDiagram"]

    def test_gui(self):
        targets = _collect_explicit_targets("create a gui diagram")
        assert targets == ["GUINoCodeDiagram"]


# ---------------------------------------------------------------------------
# _rank_implicit_targets
# ---------------------------------------------------------------------------

class TestRankImplicitTargets:
    def test_class_keywords(self):
        targets = _rank_implicit_targets("create entities with attributes and methods")
        assert targets[0] == "ClassDiagram"

    def test_state_keywords(self):
        targets = _rank_implicit_targets("model the lifecycle with transitions and events")
        assert targets[0] == "StateMachineDiagram"

    def test_agent_keywords(self):
        targets = _rank_implicit_targets("build a conversational agent with intents")
        assert targets[0] == "AgentDiagram"

    def test_no_match(self):
        assert _rank_implicit_targets("") == []


# ---------------------------------------------------------------------------
# determine_target_diagram_type / determine_target_diagram_types
# ---------------------------------------------------------------------------

class TestDetermineTargetDiagramType:
    def test_explicit_class_diagram(self):
        request = _make_request("create a class diagram for hotel booking")
        assert determine_target_diagram_type(request) == "ClassDiagram"

    def test_explicit_state_machine(self):
        request = _make_request("create a state machine diagram for order lifecycle")
        assert determine_target_diagram_type(request) == "StateMachineDiagram"

    def test_implicit_fallback_to_active(self):
        request = _make_request("add an attribute", active_diagram_type="ObjectDiagram")
        result = determine_target_diagram_type(request)
        # Should fall back to active diagram when no explicit or strong implicit match
        assert result in {"ObjectDiagram", "ClassDiagram"}

    def test_modify_stays_on_active(self):
        request = _make_request("rename user to customer", active_diagram_type="StateMachineDiagram")
        result = determine_target_diagram_type(request, last_intent="modify_model_intent")
        # Modify intent should prefer staying on active diagram
        assert result == "StateMachineDiagram"

    def test_multiple_targets(self):
        request = _make_request("create a class diagram and object diagram and state machine")
        targets = determine_target_diagram_types(request, max_targets=3)
        assert len(targets) >= 2
        assert "ClassDiagram" in targets


# ---------------------------------------------------------------------------
# resolve_diagram_id
# ---------------------------------------------------------------------------

class TestResolveDiagramId:
    def test_active_diagram_id(self):
        request = AssistantRequest(
            context=WorkspaceContext(
                active_diagram_type="ClassDiagram",
                active_diagram_id="diag-123",
            ),
        )
        assert resolve_diagram_id(request, "ClassDiagram") == "diag-123"

    def test_from_snapshot(self):
        request = _make_request(
            "",
            project_snapshot={
                "diagrams": {
                    "ObjectDiagram": {"id": "od-456", "title": "Objects"},
                }
            },
        )
        assert resolve_diagram_id(request, "ObjectDiagram") == "od-456"

    def test_missing(self):
        request = _make_request("")
        assert resolve_diagram_id(request, "ObjectDiagram") is None


# ---------------------------------------------------------------------------
# build_switch_diagram_action
# ---------------------------------------------------------------------------

class TestBuildSwitchDiagramAction:
    def test_structure(self):
        action = build_switch_diagram_action("StateMachineDiagram", reason="User asked")
        assert action["action"] == "switch_diagram"
        assert action["diagramType"] == "StateMachineDiagram"
        assert action["reason"] == "User asked"

    def test_default_reason(self):
        action = build_switch_diagram_action("ClassDiagram")
        assert "ClassDiagram" in action["reason"]
