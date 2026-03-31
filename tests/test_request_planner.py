"""Tests for the request planner (orchestrator/request_planner.py)."""

import json
import pytest
from orchestrator.request_planner import (
    plan_assistant_operations,
    _split_message_segments,
    _match_segment_target,
    _should_use_llm_planner,
    _normalize_operations,
    _fallback_operations,
)
from protocol.types import AssistantRequest, WorkspaceContext


def _make_request(message: str, active_diagram_type: str = "ClassDiagram") -> AssistantRequest:
    return AssistantRequest(
        message=message,
        diagram_type=active_diagram_type,
        context=WorkspaceContext(active_diagram_type=active_diagram_type),
    )


def _noop_predict(prompt: str) -> str:
    """Fake LLM predict that returns an empty plan."""
    return json.dumps({"operations": []})


# ---------------------------------------------------------------------------
# _split_message_segments
# ---------------------------------------------------------------------------

class TestSplitMessageSegments:
    def test_simple(self):
        segments = _split_message_segments("create a class diagram")
        assert segments == ["create a class diagram"]

    def test_with_connectors(self):
        segments = _split_message_segments("create a class diagram and then generate django code")
        assert len(segments) == 2

    def test_with_semicolons(self):
        segments = _split_message_segments("create a class; generate sql")
        assert len(segments) == 2

    def test_empty(self):
        assert _split_message_segments("") == []

    def test_non_string(self):
        assert _split_message_segments(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _match_segment_target
# ---------------------------------------------------------------------------

class TestMatchSegmentTarget:
    def test_class_diagram(self):
        assert _match_segment_target("create a class diagram for hotel") == "ClassDiagram"

    def test_state_machine(self):
        assert _match_segment_target("model the state machine for orders") == "StateMachineDiagram"

    def test_no_match(self):
        assert _match_segment_target("say hello") is None


# ---------------------------------------------------------------------------
# _should_use_llm_planner
# ---------------------------------------------------------------------------

class TestShouldUseLlmPlanner:
    def test_simple_request_no_planner(self):
        assert not _should_use_llm_planner("create a User class", 1, False)

    def test_multi_target_with_connector(self):
        assert _should_use_llm_planner(
            "create a class diagram and a state machine diagram", 2, False
        )

    def test_generation_with_connector(self):
        assert _should_use_llm_planner(
            "create a class diagram and then generate django code", 1, True
        )

    def test_long_complex_no_diagram_keywords(self):
        msg = "I want to build something complex with multiple entities and relationships and attributes " * 2
        assert _should_use_llm_planner(msg, 1, False)


# ---------------------------------------------------------------------------
# _normalize_operations
# ---------------------------------------------------------------------------

class TestNormalizeOperations:
    def test_valid_model_operation(self):
        ops = [{"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system", "request": "create a hotel"}]
        request = _make_request("create a hotel")
        result = _normalize_operations(ops, request, "complete_system")
        assert len(result) == 1
        assert result[0]["type"] == "model"

    def test_invalid_diagram_type_skipped(self):
        ops = [{"type": "model", "diagramType": "FooDiagram", "mode": "complete_system", "request": "create"}]
        request = _make_request("")
        result = _normalize_operations(ops, request, "complete_system")
        assert len(result) == 0

    def test_deduplication(self):
        ops = [
            {"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system", "request": "create a hotel"},
            {"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system", "request": "create a hotel"},
        ]
        request = _make_request("create a hotel")
        result = _normalize_operations(ops, request, "complete_system")
        assert len(result) == 1

    def test_generation_operation(self):
        ops = [{"type": "generation", "generatorType": "django", "config": {}}]
        request = _make_request("generate django code")
        result = _normalize_operations(ops, request, "complete_system")
        assert len(result) == 1
        assert result[0]["generatorType"] == "django"

    def test_invalid_generator_type_inferred(self):
        ops = [{"type": "generation", "generatorType": "invalid_gen", "request": "generate django"}]
        request = _make_request("generate django code")
        result = _normalize_operations(ops, request, "complete_system")
        # Should infer django from the request message
        assert len(result) == 1
        assert result[0]["generatorType"] == "django"

    def test_non_list_returns_empty(self):
        request = _make_request("")
        assert _normalize_operations("not a list", request, "complete_system") == []


# ---------------------------------------------------------------------------
# _fallback_operations
# ---------------------------------------------------------------------------

class TestFallbackOperations:
    def test_single_target(self):
        request = _make_request("create a class diagram for hotel")
        ops = _fallback_operations(request, "complete_system", None)
        assert len(ops) >= 1
        assert ops[0]["type"] == "model"
        assert ops[0]["diagramType"] == "ClassDiagram"

    def test_with_generation(self):
        request = _make_request("create a class diagram and generate django code")
        ops = _fallback_operations(request, "complete_system", None)
        gen_ops = [o for o in ops if o["type"] == "generation"]
        assert len(gen_ops) == 1
        assert gen_ops[0]["generatorType"] == "django"


# ---------------------------------------------------------------------------
# plan_assistant_operations (integration with noop LLM)
# ---------------------------------------------------------------------------

class TestPlanAssistantOperations:
    def test_simple_request(self):
        request = _make_request("create a User class")
        ops = plan_assistant_operations(request, "single_element", "create_single_element_intent", _noop_predict)
        assert len(ops) >= 1
        assert ops[0]["type"] == "model"

    def test_empty_message(self):
        request = _make_request("")
        ops = plan_assistant_operations(request, "complete_system", None, _noop_predict)
        # Even with empty message, should produce at least a fallback
        assert len(ops) >= 1
