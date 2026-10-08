"""Schema guarantees for agentic BPMN, Component and Deployment structured output.

OpenAI Structured Outputs feed every Field description to the model, so a
field without a description is a field the model has to guess.
"""

import pytest
from pydantic import ValidationError

from schemas import (
    BPMNFlowSpec,
    BPMNLaneSpec,
    BPMNModificationChanges,
    BPMNModificationTarget,
    BPMNNodeSpec,
    BPMNPoolSpec,
    ComponentDependencySpec,
    ComponentModificationChanges,
    ComponentSpec,
    DeploymentComponentSpec,
    DeploymentModificationChanges,
    DeploymentNodeSpec,
    SystemBPMNSpec,
)

_ALL_DESCRIBED = [
    BPMNNodeSpec, BPMNLaneSpec, BPMNPoolSpec, BPMNFlowSpec, SystemBPMNSpec,
    BPMNModificationTarget, BPMNModificationChanges,
    ComponentSpec, ComponentDependencySpec, ComponentModificationChanges,
    DeploymentNodeSpec, DeploymentComponentSpec, DeploymentModificationChanges,
]


@pytest.mark.parametrize("model", _ALL_DESCRIBED, ids=lambda m: m.__name__)
def test_every_field_has_a_description(model):
    missing = [name for name, field in model.model_fields.items() if not (field.description or "").strip()]
    assert missing == []


def test_bpmn_process_needs_at_least_one_node():
    with pytest.raises(ValidationError):
        SystemBPMNSpec(nodes=[])
    with pytest.raises(ValidationError):
        SystemBPMNSpec()


@pytest.mark.parametrize("field", ["type", "taskType", "gatewayType"])
def test_restored_node_descriptions_explain_the_values(field):
    description = BPMNNodeSpec.model_fields[field].description
    assert "'" in description and len(description) > 60


def test_flow_name_description_restored():
    assert "gateway" in BPMNFlowSpec.model_fields["name"].description


class TestLaneMultiplicity:
    def test_is_an_integer_like_the_editor(self):
        lane = BPMNLaneSpec(id="reviewers", isAgentic=True, role="solution", multiplicity=3)
        assert lane.multiplicity == 3
        schema = BPMNLaneSpec.model_json_schema()["properties"]["multiplicity"]
        assert {"type": "integer", "minimum": 1} in schema["anyOf"]

    @pytest.mark.parametrize("bad", [0, -2, "1..*", "many"])
    def test_rejects_non_positive_or_ranges(self, bad):
        with pytest.raises(ValidationError):
            BPMNLaneSpec(id="x", multiplicity=bad)

    def test_matches_modification_schema(self):
        gen = BPMNLaneSpec.model_fields["multiplicity"]
        mod = BPMNModificationChanges.model_fields["multiplicity"]
        assert gen.annotation == mod.annotation
        assert gen.metadata == mod.metadata


@pytest.mark.parametrize("role", ["collaboration", "consensus", "worker", "manager"])
def test_lane_roles_are_limited_to_the_editor_profiles(role):
    with pytest.raises(ValidationError):
        BPMNLaneSpec(id="x", isAgentic=True, role=role)
    with pytest.raises(ValidationError):
        BPMNModificationChanges(role=role)


def test_supervision_component_and_new_edge_kinds_are_accepted():
    assert ComponentSpec(id="sup", name="Supervisor", stereotype="supervision").stereotype == "supervision"
    for kind in ("has", "granted", "implements"):
        assert ComponentDependencySpec(source="a", target="b", stereotype=kind).stereotype == kind
        assert ComponentModificationChanges(dependencyStereotype=kind).dependencyStereotype == kind


@pytest.mark.parametrize("bad", ["database", "service", "agent"])
def test_unknown_component_stereotypes_are_rejected(bad):
    with pytest.raises(ValidationError):
        ComponentSpec(id="c", name="C", stereotype=bad)
    with pytest.raises(ValidationError):
        DeploymentComponentSpec(id="c", name="C", stereotype=bad)


def test_reflection_reviewer_lane_is_part_of_the_task_schema():
    node = BPMNNodeSpec(
        id="draft", name="Draft", type="task", isAgentic=True,
        reflectionMode="cross", reflectionReviewerLaneId="reviewer",
    )
    assert node.reflectionReviewerLaneId == "reviewer"
    assert BPMNModificationChanges(reflectionReviewerLaneId="lane-1").reflectionReviewerLaneId == "lane-1"
