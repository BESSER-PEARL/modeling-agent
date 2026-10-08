"""The shared reference guardrail (BaseDiagramHandler._validate_mod_refs).

Covers the Component / Deployment same-batch preview, per-request element
passing (handlers are singletons shared by every session, so no request state
may live on ``self``) and the shared prompt fragments.
"""

import re

import pytest

from diagram_handlers.core.prompt_fragments import EXACT_NAMES_RULE, POSITION_DISCLAIMER, REMOVE_ELEMENT_RULE
from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
from diagram_handlers.types.component_diagram_handler import (
    MODIFY_SYSTEM_PROMPT_COMPONENT,
    ComponentDiagramHandler,
)
from diagram_handlers.types.deployment_diagram_handler import (
    MODIFY_SYSTEM_PROMPT_DEPLOYMENT,
    DeploymentDiagramHandler,
)
from schemas import ComponentModificationResponse, DeploymentModificationResponse

COMPONENT_MODEL = {"elements": {
    "c-planner": {"type": "Component", "name": "Planner", "stereotype": "solution"},
    "s-core": {"type": "Subsystem", "name": "Core", "stereotype": "subsystem"},
}}
DEPLOYMENT_MODEL = {"elements": {
    "n-cloud": {"type": "DeploymentNode", "name": "Cloud", "stereotype": "cloud"},
    "a-api": {"type": "DeploymentArtifact", "name": "API", "owner": "n-cloud"},
}}


def _patch_llm(monkeypatch, handler_cls, schema, mods):
    response = schema.model_validate({"modifications": mods, "message": "ok"})
    monkeypatch.setattr(handler_cls, "predict_structured", lambda self, *a, **k: response)


def _actions(result):
    mods = result.get("modifications") or [result["modification"]]
    return [(m["action"], (m.get("target") or {}).get("elementName"), (m.get("changes") or {}).get("target"))
            for m in mods]


class TestSameBatchPreview:
    def test_component_dependency_to_component_added_earlier_is_kept(self, monkeypatch):
        _patch_llm(monkeypatch, ComponentDiagramHandler, ComponentModificationResponse, [
            {"action": "add_component", "target": {"elementName": "Writer"}, "changes": {"stereotype": "solution"}},
            {"action": "add_dependency", "target": {},
             "changes": {"source": "Planner", "target": "Writer", "dependencyStereotype": "delegates"}},
            {"action": "add_dependency", "target": {},
             "changes": {"source": "Planner", "target": "Ghost", "dependencyStereotype": "uses"}},
        ])
        result = ComponentDiagramHandler(None).generate_modification("add a writer", current_model=COMPONENT_MODEL)
        assert _actions(result) == [("add_component", "Writer", None), ("add_dependency", None, "Writer")]
        assert "Ghost" not in result["message"]

    def test_component_rename_then_reference_new_name(self, monkeypatch):
        _patch_llm(monkeypatch, ComponentDiagramHandler, ComponentModificationResponse, [
            {"action": "modify_element", "target": {"elementName": "Planner"}, "changes": {"name": "Orchestrator"}},
            {"action": "add_component", "target": {"elementName": "Memory"}, "changes": {"stereotype": "db"}},
            {"action": "add_dependency", "target": {},
             "changes": {"source": "Orchestrator", "target": "Memory", "dependencyStereotype": "has"}},
        ])
        result = ComponentDiagramHandler(None).generate_modification("rename", current_model=COMPONENT_MODEL)
        assert len(result["modifications"]) == 3

    def test_component_removed_element_cannot_be_referenced_later(self, monkeypatch):
        _patch_llm(monkeypatch, ComponentDiagramHandler, ComponentModificationResponse, [
            {"action": "remove_element", "target": {"elementName": "Planner"}},
            {"action": "add_dependency", "target": {},
             "changes": {"source": "Planner", "target": "Core", "dependencyStereotype": "uses"}},
        ])
        result = ComponentDiagramHandler(None).generate_modification("remove", current_model=COMPONENT_MODEL)
        assert [m["action"] for m in result["modifications"]] == ["remove_element"]
        assert result["message"] == "Removed **Planner**."

    def test_deployment_artifact_added_earlier_can_be_connected(self, monkeypatch):
        _patch_llm(monkeypatch, DeploymentDiagramHandler, DeploymentModificationResponse, [
            {"action": "add_artifact", "target": {"elementName": "Worker"}, "changes": {"owner": "Cloud"}},
            {"action": "add_dependency", "target": {}, "changes": {"source": "API", "target": "Worker"}},
            {"action": "add_dependency", "target": {}, "changes": {"source": "API", "target": "Nowhere"}},
        ])
        result = DeploymentDiagramHandler(None).generate_modification("add worker", current_model=DEPLOYMENT_MODEL)
        assert _actions(result) == [("add_artifact", "Worker", None), ("add_dependency", None, "Worker")]


@pytest.mark.parametrize(("handler_cls", "schema", "model", "mod"), [
    (ComponentDiagramHandler, ComponentModificationResponse, COMPONENT_MODEL,
     {"action": "modify_element", "target": {"elementName": "Planner"}, "changes": {"name": "Lead"}}),
    (DeploymentDiagramHandler, DeploymentModificationResponse, DEPLOYMENT_MODEL,
     {"action": "modify_element", "target": {"elementName": "API"}, "changes": {"name": "Gateway"}}),
])
def test_handlers_keep_no_request_state(monkeypatch, handler_cls, schema, model, mod):
    _patch_llm(monkeypatch, handler_cls, schema, [mod])
    handler = handler_cls(None)
    handler.generate_modification("rename it", current_model=model)
    assert "_elements" not in vars(handler)


def test_bpmn_keeps_no_request_state(monkeypatch):
    from schemas import BPMNModificationResponse

    response = BPMNModificationResponse.model_validate({
        "modifications": [{"action": "modify_node", "target": {"nodeName": "Ship"}, "changes": {"name": "Send"}}],
        "message": "ok",
    })
    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", lambda self, *a, **k: response)
    handler = BPMNDiagramHandler(None)
    handler.generate_modification("rename", current_model={"elements": {"t": {"type": "BPMNTask", "name": "Ship"}}})
    assert "_elements" not in vars(handler)


def test_concurrent_requests_on_the_shared_handler_do_not_mix_models(monkeypatch):
    """Session B's request runs while session A's LLM call is in flight (the
    factory hands both sessions the SAME handler instance). A's guardrail must
    still validate against A's model, not the one B left behind."""
    handler = ComponentDiagramHandler(None)
    model_a = {"elements": {"a1": {"type": "Component", "name": "Alpha"}}}
    model_b = {"elements": {"b1": {"type": "Component", "name": "Beta"}}}
    responses = {
        "Alpha": ComponentModificationResponse.model_validate({"modifications": [
            {"action": "modify_element", "target": {"elementName": "Alpha"}, "changes": {"name": "Alpha2"}},
        ], "message": "ok"}),
        "Beta": ComponentModificationResponse.model_validate({"modifications": [
            {"action": "modify_element", "target": {"elementName": "Beta"}, "changes": {"name": "Beta2"}},
        ], "message": "ok"}),
    }
    results = {}

    def fake_predict(self, user_prompt, *args, **kwargs):
        if "Alpha" in user_prompt and "B" not in results:
            results["B"] = handler.generate_modification("rename Beta", current_model=model_b)
        return responses["Alpha" if "Alpha" in user_prompt else "Beta"]

    monkeypatch.setattr(ComponentDiagramHandler, "predict_structured", fake_predict)
    results["A"] = handler.generate_modification("rename Alpha", current_model=model_a)

    assert results["A"]["action"] == "modify_model", results["A"]
    assert results["A"]["modification"]["changes"]["name"] == "Alpha2"
    assert results["B"]["modification"]["changes"]["name"] == "Beta2"


def test_element_id_is_named_from_the_request_model():
    handler = ComponentDiagramHandler(None)
    name = handler._build_mod_target_name(
        "remove_element", {"elementId": "c-planner"}, elements=COMPONENT_MODEL["elements"],
    )
    assert name == "Planner"
    assert handler._build_mod_target_name("remove_element", {"elementId": "c-planner"}) == "c-planner"


class TestSharedPromptFragments:
    @pytest.mark.parametrize(
        "prompt", [MODIFY_SYSTEM_PROMPT_COMPONENT, MODIFY_SYSTEM_PROMPT_DEPLOYMENT], ids=["component", "deployment"],
    )
    def test_modify_prompts_use_the_shared_rules_verbatim(self, prompt):
        assert EXACT_NAMES_RULE in prompt
        assert REMOVE_ELEMENT_RULE in prompt
        assert "elementFound: false" in prompt and "Ctrl+Z" in prompt

    @pytest.mark.parametrize("handler_cls", [ComponentDiagramHandler, DeploymentDiagramHandler])
    def test_system_prompt_uses_position_disclaimer(self, handler_cls):
        prompt = handler_cls(None).get_system_prompt()
        assert POSITION_DISCLAIMER in prompt
        assert not re.search(r"\{[A-Za-z_]+\}", prompt)
