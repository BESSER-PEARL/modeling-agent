"""Tests for the request planner (orchestrator/request_planner.py)."""

import json
import pytest
from orchestrator.request_planner import (
    GENERATOR_PREREQUISITES,
    plan_assistant_operations,
    _split_message_segments,
    _match_segment_target,
    _should_use_llm_planner,
    _normalize_operations,
    _fallback_operations,
    _validate_and_fix_plan,
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


def test_api_and_semantic_generators_share_class_diagram_prerequisite():
    assert GENERATOR_PREREQUISITES["rest_api"] == ["ClassDiagram"]
    assert GENERATOR_PREREQUISITES["rdf"] == ["ClassDiagram"]


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


# ---------------------------------------------------------------------------
# GUI builds need a class diagram first
# ---------------------------------------------------------------------------

class TestGuiNeedsClassDiagram:
    """Live report: "Build a complete library management platform with models
    and UI" on an empty workspace. The classifier targeted GUINoCodeDiagram,
    the plan was a lone GUI op, and the user got 9 invented screens with no
    class diagram on the canvas (and no Basic/AI screen choice, which is only
    asked when a class diagram exists)."""

    MSG = "Build a complete library management platform with models and UI"

    @staticmethod
    def _shape(ops):
        return [(o["type"], o.get("diagramType") or o.get("generatorType")) for o in ops]

    @pytest.mark.parametrize("target", ["GUINoCodeDiagram", "ClassDiagram", None])
    def test_ui_request_on_empty_workspace_takes_the_web_app_route(self, target):
        # The classifier's target flips between runs; live, ClassDiagram gave
        # a model with no screens and GUINoCodeDiagram gave screens with no model.
        request = _make_request(self.MSG)
        ops = plan_assistant_operations(
            request, "complete_system", "create_complete_system_intent",
            _noop_predict, llm_target_type=target,
        )
        assert self._shape(ops) == [
            ("model", "ClassDiagram"),
            ("model", "GUINoCodeDiagram"),
            ("generation", "web_app"),
        ]
        assert ops[0]["mode"] == "complete_system"
        assert self.MSG in ops[0]["request"]

    def test_lone_gui_op_on_empty_workspace_gets_class_diagram_first(self):
        # Prerequisite pass, independent of the web-app route.
        request = _make_request(self.MSG)
        ops = _validate_and_fix_plan(
            [{"type": "model", "diagramType": "GUINoCodeDiagram",
              "mode": "complete_system", "request": self.MSG}],
            request,
        )
        assert self._shape(ops) == [
            ("model", "ClassDiagram"),
            ("model", "GUINoCodeDiagram"),
        ]

    def test_create_without_ui_words_stays_a_class_diagram(self):
        request = _make_request("Build a complete library management platform")
        ops = plan_assistant_operations(
            request, "complete_system", "create_complete_system_intent",
            _noop_predict, llm_target_type="ClassDiagram",
        )
        assert self._shape(ops) == [("model", "ClassDiagram")]

    def test_existing_class_diagram_is_not_rebuilt(self):
        request = _make_request("create a GUI for my model")
        request.context.project_snapshot = {
            "diagrams": {"ClassDiagram": [{"model": {"elements": {"c1": {"name": "Book"}}}}]},
        }
        ops = plan_assistant_operations(
            request, "complete_system", "create_complete_system_intent",
            _noop_predict, llm_target_type="GUINoCodeDiagram",
        )
        assert self._shape(ops) == [("model", "GUINoCodeDiagram")]

    def test_gui_modify_does_not_add_class_diagram(self):
        request = _make_request("add a footer to the home page", "GUINoCodeDiagram")
        ops = plan_assistant_operations(
            request, "modify_model", "modify_model_intent",
            _noop_predict, llm_target_type="GUINoCodeDiagram",
        )
        assert ("model", "ClassDiagram") not in self._shape(ops)
