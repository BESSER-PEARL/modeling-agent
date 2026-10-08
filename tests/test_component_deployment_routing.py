"""Layer-2 routing for Component / Deployment diagrams.

Regression for the PR #21 review: generic architecture vocabulary
("system architecture", "software architecture", "subsystem", "service
architecture") used to route class-diagram requests to ComponentDiagram, or
append a ComponentDiagram as a second target next to the class diagram.
"""

import json

import pytest

from orchestrator.request_planner import plan_assistant_operations
from orchestrator.workspace_orchestrator import (
    determine_target_diagram_type,
    determine_target_diagram_types,
)
from protocol.types import AssistantRequest, WorkspaceContext

CREATE = "create_complete_system_intent"


def _request(message: str, active: str = "ClassDiagram") -> AssistantRequest:
    return AssistantRequest(
        message=message, diagram_type=active, context=WorkspaceContext(active_diagram_type=active),
    )


def _noop_predict(prompt: str) -> str:
    return json.dumps({"operations": []})


GENERIC_ARCHITECTURE_MESSAGES = [
    "Model the system architecture of a library with classes Book and Author",
    "Create a class diagram for the system architecture of a hotel",
    "Design the software architecture of an online shop with Customer and Order classes",
    "Create a class diagram where Billing is a subsystem of the hospital",
    "Model the service architecture of a ride sharing app with Driver and Trip entities",
]


@pytest.mark.parametrize("message", GENERIC_ARCHITECTURE_MESSAGES)
def test_generic_architecture_words_stay_on_the_class_diagram(message):
    targets = determine_target_diagram_types(_request(message), last_intent=CREATE, max_targets=3)
    assert targets == ["ClassDiagram"]


@pytest.mark.parametrize("message", GENERIC_ARCHITECTURE_MESSAGES)
def test_classifier_verdict_gets_no_component_extra(message):
    targets = determine_target_diagram_types(
        _request(message), last_intent=CREATE, max_targets=3, llm_target_type="ClassDiagram",
    )
    assert targets == ["ClassDiagram"]


def test_hotel_request_plans_a_single_class_diagram():
    message = "Create a class diagram for the system architecture of a hotel"
    for target in ("ClassDiagram", None):
        ops = plan_assistant_operations(
            _request(message), "complete_system", CREATE, _noop_predict, llm_target_type=target,
        )
        assert [(op["type"], op.get("diagramType")) for op in ops] == [("model", "ClassDiagram")]


def test_implicit_pattern_hit_cannot_add_a_second_diagram_to_the_verdict():
    # "lifecycle" is a StateMachineDiagram pattern hit, not a diagram name.
    message = "Create a library system where each book has a lifecycle"
    targets = determine_target_diagram_types(
        _request(message), last_intent=CREATE, max_targets=3, llm_target_type="ClassDiagram",
    )
    assert targets == ["ClassDiagram"]


@pytest.mark.parametrize(("message", "expected"), [
    ("create a component diagram for a RAG agent swarm", "ComponentDiagram"),
    ("create a UML component model for the payment platform", "ComponentDiagram"),
    ("draw the components diagram of the shop", "ComponentDiagram"),
    ("create a deployment diagram for the agent services", "DeploymentDiagram"),
    ("build the deployment model for a web shop", "DeploymentDiagram"),
])
def test_explicit_diagram_names_still_route(message, expected):
    assert determine_target_diagram_type(_request(message), last_intent=CREATE) == expected


def test_explicitly_named_second_diagram_is_kept():
    message = "create a class diagram and a component diagram for a library"
    assert determine_target_diagram_types(_request(message), last_intent=CREATE, max_targets=3) == [
        "ClassDiagram", "ComponentDiagram",
    ]
    assert determine_target_diagram_types(
        _request(message), last_intent=CREATE, max_targets=3, llm_target_type="ClassDiagram",
    ) == ["ClassDiagram", "ComponentDiagram"]


def test_classifier_component_verdict_leads():
    message = "lay out the subsystems of the agent platform"
    assert determine_target_diagram_type(
        _request(message), last_intent=CREATE, llm_target_type="ComponentDiagram",
    ) == "ComponentDiagram"


# ---------------------------------------------------------------------------
# Example / suggestion wording must not look like a code-generation request
# ---------------------------------------------------------------------------

def test_deploy_wording_would_trigger_the_deploy_generator():
    """Why the wording below matters: the deploy fuzzy pattern fires on this."""
    from handlers.generation_handler import detect_generator_type

    assert detect_generator_type("deploy a component to a cloud node") == "deploy"


def test_classifier_modify_example_is_not_a_generator_request():
    from handlers.generation_handler import detect_generator_type
    from unified_classifier import _SYSTEM_PROMPT

    example = "place a component on a cloud node"
    assert f"'{example}'" in _SYSTEM_PROMPT
    assert detect_generator_type(example) is None


@pytest.mark.parametrize("diagram_type", ["ComponentDiagram", "DeploymentDiagram"])
def test_suggestion_chips_are_not_generator_requests(diagram_type):
    from handlers.generation_handler import detect_generator_type
    from suggestions import get_suggested_actions

    for action in get_suggested_actions(diagram_type, "complete_system", []):
        assert detect_generator_type(action["prompt"]) is None, action
