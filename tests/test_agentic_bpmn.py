"""Agentic BPMN: mode selection, prompt composition, the validated generation
path, and project-scoped agentDiagramRef / reflectionReviewerLaneId."""

import json
from pathlib import Path

import pytest

from diagram_handlers.types.bpmn_diagram_handler import (
    AGENTIC_DESIGN_RULES,
    MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN,
    MODIFY_SYSTEM_PROMPT_BPMN,
    BPMNDiagramHandler,
)
from protocol.types import AssistantRequest, WorkspaceContext
from schemas import BPMNModificationResponse, SystemBPMNSpec
from utilities.model_resolution import resolve_agent_diagram_refs

_TEMPLATES = Path(__file__).resolve().parent / "fixtures" / "wme_templates"
is_agentic = BPMNDiagramHandler._is_agentic_bpmn_request


def _template(name):
    data = json.loads((_TEMPLATES / f"{name}.json").read_text(encoding="utf-8"))
    return data.get("model", data)


AGENTIC_MODEL = {"elements": {
    "pool-1": {"type": "BPMNPool", "name": "Review Swarm"},
    "lane-w": {"type": "BPMNSwimlane", "name": "Writer", "owner": "pool-1", "isAgentic": True, "role": "solution"},
    "lane-s": {"type": "BPMNSwimlane", "name": "Supervisor", "owner": "pool-1", "isAgentic": True,
               "role": "supervision"},
    "t-draft": {"type": "BPMNTask", "name": "Draft Answer", "owner": "lane-w"},
}}


class TestModeSelection:
    @pytest.mark.parametrize("message", [
        "Create a carpool booking process",
        "Model the order process between the customer and supplier participants",
        "Orchestrate the parcel delivery from warehouse to customer",
        "Create a BPMN process with a pool for the customer and one for the bank",
        "Model the swimlanes of the hiring process",
        "Add a review task after Draft",
    ])
    def test_ordinary_collaboration_wording_is_not_agentic(self, message):
        assert is_agentic(message, None) is False

    @pytest.mark.parametrize("template", ["pizza_store", "car_wash"])
    def test_wme_templates_with_pools_and_lanes_are_not_agentic(self, template):
        model = _template(template)
        assert any(el.get("type") == "BPMNPool" for el in model["elements"].values())
        assert is_agentic("add a task to check the order", model) is False

    @pytest.mark.parametrize("message", [
        "Create an Agentic BPMN swarm for document review",
        "Model a multi-agent review process",
        "Build an agent swarm that triages support tickets",
        "Add an AI agent lane for the fact checker",
        "Change the lane role of Reviewer to supervision",
        "Add Governance DSL to the merging gateway",
        "Set the trust score of the writer to 80",
    ])
    def test_explicit_agentic_intent_is_agentic(self, message):
        assert is_agentic(message, None) is True

    def test_existing_agentic_lane_selects_agentic_mode(self):
        assert is_agentic("rename Draft Answer to Draft Reply", AGENTIC_MODEL) is True


def _spec_with_gaps():
    """An agentic spec with an orphaned end event and invented references."""
    return SystemBPMNSpec.model_validate({
        "systemName": "Review Swarm",
        "pools": [{"id": "swarm", "name": "Swarm", "lanes": [
            {"id": "writer", "name": "Writer", "isAgentic": True, "role": "solution",
             "multiplicity": 3, "agentDiagramRef": "agent-real"},
            {"id": "supervisor", "name": "Supervisor", "isAgentic": True, "role": "supervision",
             "agentDiagramRef": "made-up-uuid"},
        ]}],
        "nodes": [
            {"id": "start", "name": "Request", "type": "startEvent", "poolId": "swarm", "laneId": "writer"},
            {"id": "draft", "name": "Draft", "type": "task", "poolId": "swarm", "laneId": "writer",
             "isAgentic": True, "reflectionMode": "cross", "reflectionReviewerLaneId": "supervisor",
             "agentDiagramRef": "invented"},
            {"id": "check", "name": "Check", "type": "task", "poolId": "swarm", "laneId": "supervisor",
             "isAgentic": True, "reflectionMode": "self", "reflectionReviewerLaneId": "writer"},
            {"id": "done", "name": "Done", "type": "endEvent", "poolId": "swarm", "laneId": "supervisor"},
        ],
        "flows": [
            {"source": "start", "target": "draft"},
            {"source": "draft", "target": "check"},
        ],
    })


class TestAgenticGeneration:
    def _run(self, monkeypatch, message, raw_request=None, refs=None):
        calls = []

        def fake_two_pass(self, user_request, system_prompt, reasoning_prompt, response_schema, **kwargs):
            calls.append({"user_request": user_request, "system_prompt": system_prompt})
            return _spec_with_gaps()

        monkeypatch.setattr(BPMNDiagramHandler, "predict_two_pass_structured", fake_two_pass)
        result = BPMNDiagramHandler(None).generate_complete_system(
            message, raw_request=raw_request, agent_diagram_refs=refs if refs is not None else {"agent-real": "Writer"},
        )
        return result, calls[0]

    def test_agentic_path_runs_validate_and_refine(self, monkeypatch):
        refined = []
        original = BPMNDiagramHandler._validate_and_refine

        def spy(self, spec):
            refined.append(True)
            return original(self, spec)

        monkeypatch.setattr(BPMNDiagramHandler, "_validate_and_refine", spy)
        result, _call = self._run(monkeypatch, "Create an agentic swarm for document review")
        assert refined == [True]
        flows = {(f["source"], f["target"]) for f in result["systemSpec"]["flows"]}
        assert ("check", "done") in flows, "orphaned end event must be reconnected"

    def test_agentic_prompt_composes_base_and_agentic_rules(self, monkeypatch):
        _result, call = self._run(monkeypatch, "Create an agentic swarm for document review")
        system_prompt = call["system_prompt"]
        assert system_prompt.startswith(BPMNDiagramHandler(None).get_system_prompt())
        assert system_prompt.endswith(AGENTIC_DESIGN_RULES)
        assert "'collaboration'" not in AGENTIC_DESIGN_RULES and "consensus" not in AGENTIC_DESIGN_RULES

    def test_project_agent_diagrams_are_listed_and_invented_refs_dropped(self, monkeypatch):
        result, call = self._run(monkeypatch, "Create an agentic swarm for document review")
        assert "- agent-real: Writer" in call["user_request"]
        spec = result["systemSpec"]
        lanes = {lane["id"]: lane for lane in spec["pools"][0]["lanes"]}
        nodes = {node["id"]: node for node in spec["nodes"]}
        assert lanes["writer"]["agentDiagramRef"] == "agent-real"
        assert lanes["supervisor"]["agentDiagramRef"] is None
        assert nodes["draft"]["agentDiagramRef"] is None
        assert lanes["writer"]["multiplicity"] == 3

    def test_reviewer_lane_kept_only_for_cross_reflection_on_a_declared_lane(self, monkeypatch):
        result, _call = self._run(monkeypatch, "Create an agentic swarm for document review")
        nodes = {node["id"]: node for node in result["systemSpec"]["nodes"]}
        assert nodes["draft"]["reflectionReviewerLaneId"] == "supervisor"
        assert nodes["check"]["reflectionReviewerLaneId"] is None

    def test_no_agent_diagrams_means_no_refs(self, monkeypatch):
        result, call = self._run(monkeypatch, "Create an agentic swarm", refs={})
        assert "none — leave every agentDiagramRef null" in call["user_request"]
        lanes = result["systemSpec"]["pools"][0]["lanes"]
        assert all(lane["agentDiagramRef"] is None for lane in lanes)

    def test_detection_reads_the_raw_message_not_the_enriched_prompt(self, monkeypatch):
        enriched = (
            "Conversation so far: the user built an agentic swarm earlier.\n\n"
            "User request: create a pizza ordering process"
        )
        _result, call = self._run(monkeypatch, enriched, raw_request="create a pizza ordering process")
        assert AGENTIC_DESIGN_RULES not in call["system_prompt"]


class TestAgenticModification:
    def test_agentic_modify_prompt_extends_the_base_prompt(self):
        assert MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN.startswith(MODIFY_SYSTEM_PROMPT_BPMN)
        for base_rule in (
            "MUST reference by id",
            "Partial matches are valid",
            "add exactly ONE new task",
            "Never put flow endpoints inside add_task/add_gateway/add_event",
        ):
            assert base_rule in MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN
        assert "add_swimlane" in MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN

    def _modify(self, monkeypatch, model, mods, message="update it", refs=None):
        seen = {}

        def fake_predict(self, prompt, schema, **kwargs):
            seen["system_prompt"] = kwargs.get("system_prompt")
            seen["prompt"] = prompt
            return BPMNModificationResponse.model_validate({"modifications": mods, "message": "ok"})

        monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", fake_predict)
        result = BPMNDiagramHandler(None).generate_modification(
            message, current_model=model, agent_diagram_refs=refs or {},
        )
        return result, seen

    def test_template_edit_uses_the_base_prompt(self, monkeypatch):
        mods = [{"action": "add_task", "target": {"nodeName": "Bake Pizza"}}]
        _result, seen = self._modify(monkeypatch, _template("pizza_store"), mods, "add a Bake Pizza task")
        assert seen["system_prompt"] == MODIFY_SYSTEM_PROMPT_BPMN

    def test_agentic_model_edit_uses_the_agentic_prompt_and_lists_refs(self, monkeypatch):
        mods = [{"action": "modify_node", "target": {"nodeName": "Draft Answer"}, "changes": {"trustScore": 70}}]
        _result, seen = self._modify(monkeypatch, AGENTIC_MODEL, mods, refs={"agent-1": "Writer Agent"})
        assert seen["system_prompt"] == MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN
        assert "- agent-1: Writer Agent" in seen["prompt"]

    def test_invented_agent_diagram_ref_is_cleared(self, monkeypatch):
        mods = [
            {"action": "modify_swimlane", "target": {"swimlaneName": "Writer"},
             "changes": {"agentDiagramRef": "not-a-project-diagram"}},
            {"action": "modify_node", "target": {"nodeName": "Draft Answer"},
             "changes": {"agentDiagramRef": "agent-1"}},
        ]
        result, _seen = self._modify(monkeypatch, AGENTIC_MODEL, mods, refs={"agent-1": "Writer Agent"})
        changes = [m["changes"] for m in result["modifications"]]
        assert changes[0]["agentDiagramRef"] is None
        assert changes[1]["agentDiagramRef"] == "agent-1"

    def test_reviewer_lane_is_resolved_to_an_existing_lane_id(self, monkeypatch):
        mods = [
            {"action": "modify_node", "target": {"nodeName": "Draft Answer"},
             "changes": {"reflectionMode": "cross", "reflectionReviewerLaneId": "supervisor"}},
            {"action": "modify_node", "target": {"nodeName": "Draft Answer"},
             "changes": {"reflectionMode": "cross", "reflectionReviewerLaneId": "Ghost Lane"}},
        ]
        result, _seen = self._modify(monkeypatch, AGENTIC_MODEL, mods)
        changes = [m["changes"] for m in result["modifications"]]
        assert changes[0]["reflectionReviewerLaneId"] == "lane-s"
        assert changes[1]["reflectionReviewerLaneId"] is None


def test_resolve_agent_diagram_refs_reads_the_project_snapshot():
    request = AssistantRequest(message="x", context=WorkspaceContext(project_snapshot={"diagrams": {
        "AgentDiagram": [
            {"id": "a-1", "title": "Writer Agent", "model": {}},
            {"id": "a-2", "model": {}},
            {"title": "no id"},
        ],
        "ClassDiagram": [{"id": "c-1", "title": "Domain"}],
    }}))
    assert resolve_agent_diagram_refs(request) == {"a-1": "Writer Agent", "a-2": ""}
    assert resolve_agent_diagram_refs(AssistantRequest(message="x", context=WorkspaceContext())) == {}
