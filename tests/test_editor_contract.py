"""Shared contract between the agent's structured output and the editor.

``tests/fixtures/contract/editor_tokens.json`` is a copy of the editor-side
vocabulary (``bpmn/common/types.ts``, ``agentic-tokens.ts``) and of the field
names the assistant converters read. The spec / modification fixtures are
payloads the editor accepts; the agent's schemas must accept them and must
offer exactly the editor's literals — no more, no fewer.
"""

import copy
import json
from pathlib import Path

import pytest

from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
from schemas import (
    BPMNModificationResponse,
    ComponentModificationResponse,
    DeploymentModificationResponse,
    SystemBPMNSpec,
    SystemComponentSpec,
    SystemDeploymentSpec,
)

_CONTRACT = Path(__file__).resolve().parent / "fixtures" / "contract"


def _load(name):
    return json.loads((_CONTRACT / f"{name}.json").read_text(encoding="utf-8"))


TOKENS = _load("editor_tokens")


def _defs(model):
    schema = model.model_json_schema()
    return schema, schema.get("$defs", {})


def _props(schema_or_def):
    return schema_or_def["properties"]


def _enum(prop):
    """All literal values a (possibly Optional) property allows."""
    if "enum" in prop:
        return list(prop["enum"])
    if "const" in prop:
        return [prop["const"]]
    values = []
    for option in prop.get("anyOf", []):
        values.extend(_enum(option))
    return values


# ---------------------------------------------------------------------------
# Fixtures validate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("fixture", "model"), [
    ("agentic_bpmn_spec", SystemBPMNSpec),
    ("component_spec", SystemComponentSpec),
    ("deployment_spec", SystemDeploymentSpec),
    ("bpmn_modifications", BPMNModificationResponse),
    ("component_modifications", ComponentModificationResponse),
    ("deployment_modifications", DeploymentModificationResponse),
])
def test_contract_fixture_validates(fixture, model):
    payload = _load(fixture)
    parsed = model.model_validate(payload).model_dump()
    # Nothing the editor relies on is silently dropped or coerced.
    for key, value in payload.items():
        if isinstance(value, list):
            assert len(parsed[key]) == len(value), key


def test_agentic_bpmn_fixture_survives_the_handler_repair_pass():
    spec = SystemBPMNSpec.model_validate(_load("agentic_bpmn_spec")).model_dump()
    handler = BPMNDiagramHandler(None)
    refined = handler._validate_and_refine(copy.deepcopy(spec))
    handler._sanitize_agentic_refs(refined, {})
    lanes = refined["pools"][0]["lanes"]
    assert [(lane["role"], lane["multiplicity"]) for lane in lanes] == [
        ("solution", 1), ("solution", 3), ("supervision", 1),
    ]
    nodes = {node["id"]: node for node in refined["nodes"]}
    assert nodes["draft"]["reflectionReviewerLaneId"] == "supervisor"
    assert nodes["merge"]["governanceDsl"] == "Policy: MajorityPolicy"
    assert nodes["review"]["owner"] == "reviewers"


# ---------------------------------------------------------------------------
# Literals and field names match the editor
# ---------------------------------------------------------------------------

class TestBPMNContract:
    bpmn = TOKENS["bpmn"]

    def test_lane_vocabulary(self):
        _schema, defs = _defs(SystemBPMNSpec)
        lane = _props(defs["BPMNLaneSpec"])
        assert set(lane) == set(self.bpmn["laneFields"])
        assert _enum(lane["role"]) == self.bpmn["BPMNAgentProfile"]
        assert {"type": "integer", "minimum": 1} in lane["multiplicity"]["anyOf"]

    def test_node_vocabulary(self):
        _schema, defs = _defs(SystemBPMNSpec)
        node = _props(defs["BPMNNodeSpec"])
        assert set(self.bpmn["taskFields"]) | set(self.bpmn["gatewayFields"]) <= set(node)
        assert _enum(node["reflectionMode"]) == self.bpmn["BPMNReflectionMode"]
        assert _enum(node["gatewayRole"]) == self.bpmn["BPMNGatewayRole"]
        assert _enum(node["taskType"]) == self.bpmn["BPMNTaskType"]
        assert set(_enum(node["gatewayType"])) == set(self.bpmn["BPMNGatewayType"])

    def test_pool_and_flow_fields(self):
        _schema, defs = _defs(SystemBPMNSpec)
        assert set(_props(defs["BPMNPoolSpec"])) == set(self.bpmn["poolFields"])
        assert set(_props(defs["BPMNFlowSpec"])) == set(self.bpmn["flowFields"])

    def test_modification_vocabulary(self):
        _schema, defs = _defs(BPMNModificationResponse)
        changes = _props(defs["BPMNModificationChanges"])
        assert _enum(changes["role"]) == self.bpmn["BPMNAgentProfile"]
        assert _enum(changes["reflectionMode"]) == self.bpmn["BPMNReflectionMode"]
        assert _enum(changes["gatewayRole"]) == self.bpmn["BPMNGatewayRole"]
        assert set(self.bpmn["taskFields"]) | set(self.bpmn["gatewayFields"]) <= set(changes)


class TestComponentContract:
    component = TOKENS["component"]

    def test_spec_shape(self):
        schema, defs = _defs(SystemComponentSpec)
        assert set(_props(schema)) == set(self.component["specKeys"])
        assert set(_props(defs["ComponentSpec"])) == set(self.component["componentFields"])
        assert set(_props(defs["ComponentDependencySpec"])) == set(self.component["dependencyFields"])

    def test_stereotypes_are_the_agentic_tokens(self):
        _schema, defs = _defs(SystemComponentSpec)
        expected = self.component["AGENT_CATEGORY_TOKENS"] + self.component["CAPABILITY_TOKENS"]
        assert set(_enum(_props(defs["ComponentSpec"])["stereotype"])) == set(expected)
        edges = _enum(_props(defs["ComponentDependencySpec"])["stereotype"])
        assert edges == self.component["AGENTIC_EDGE_KIND_TOKENS"]

    def test_modification_literals(self):
        _schema, defs = _defs(ComponentModificationResponse)
        changes = _props(defs["ComponentModificationChanges"])
        expected = self.component["AGENT_CATEGORY_TOKENS"] + self.component["CAPABILITY_TOKENS"] + ["subsystem"]
        assert set(_enum(changes["stereotype"])) == set(expected)
        assert _enum(changes["dependencyStereotype"]) == self.component["AGENTIC_EDGE_KIND_TOKENS"]


class TestDeploymentContract:
    deployment = TOKENS["deployment"]

    def test_spec_shape(self):
        schema, defs = _defs(SystemDeploymentSpec)
        assert set(_props(schema)) == set(self.deployment["specKeys"])
        assert set(_props(defs["DeploymentNodeSpec"])) == set(self.deployment["nodeFields"])
        assert set(_props(defs["DeploymentArtifactSpec"])) == set(self.deployment["artifactFields"])
        assert set(_props(defs["DeploymentComponentSpec"])) == set(self.deployment["componentFields"])
        assert set(_props(defs["DeploymentDependencySpec"])) == set(self.deployment["dependencyFields"])

    def test_component_stereotypes_reuse_the_component_presets(self):
        # The editor's DeploymentComponent shares the UML component popup,
        # whose presets are the agentic component tokens.
        _schema, defs = _defs(SystemDeploymentSpec)
        expected = TOKENS["component"]["AGENT_CATEGORY_TOKENS"] + TOKENS["component"]["CAPABILITY_TOKENS"]
        assert set(_enum(_props(defs["DeploymentComponentSpec"])["stereotype"])) == set(expected)
