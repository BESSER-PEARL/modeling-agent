"""ComponentDiagram / DeploymentDiagram must be registered in every routing and
capability point, not only in the handler factory (see CLAUDE.md checklist)."""

import ast
from pathlib import Path
from typing import get_args

import pytest

from diagram_handlers.core.base_handler import BaseDiagramHandler
from orchestrator.request_planner import _try_heuristic_decomposition
from protocol.types import AssistantRequest, SUPPORTED_DIAGRAM_TYPES, WorkspaceContext
from conversation.reply_copy import META_ANSWER, OUT_OF_SCOPE_REDIRECT
from prompts.classifier import SYSTEM_PROMPT
from routing.unified_classifier import (
    _RELEVANT_DIAGRAM_TYPES,
    _TARGET_DIAGRAM_TYPES,
    UnifiedClassification,
    _workspace_summary_lines,
)

NEW_TYPES = ("ComponentDiagram", "DeploymentDiagram")
_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("diagram_type", NEW_TYPES)
def test_classifier_can_name_the_type(diagram_type):
    assert diagram_type in get_args(_TARGET_DIAGRAM_TYPES)
    verdict = UnifiedClassification(
        intent="create_complete_system_intent",
        target_diagram_type=diagram_type,
        reason="test",
    )
    assert verdict.target_diagram_type == diagram_type
    assert f"target_diagram_type='{diagram_type}'" in SYSTEM_PROMPT


@pytest.mark.parametrize("diagram_type", NEW_TYPES)
def test_type_is_supported_and_surfaced_to_classifier(diagram_type):
    assert diagram_type in SUPPORTED_DIAGRAM_TYPES
    assert diagram_type in {dtype for dtype, _unit in _RELEVANT_DIAGRAM_TYPES}


def test_workspace_summary_reports_component_and_deployment_tabs():
    ctx = WorkspaceContext(project_snapshot={"diagrams": {
        "ComponentDiagram": [{"id": "c1", "model": {"elements": {
            "a": {"type": "Component", "name": "Planner"},
        }}}],
    }})
    lines = _workspace_summary_lines(ctx)
    assert "- ComponentDiagram: 1 component element(s) (Planner)" in lines
    assert "- DeploymentDiagram: not present" in lines


def test_classifier_prompt_keeps_diagram_requests_out_of_generation():
    lower = SYSTEM_PROMPT.lower()
    assert "generate a deployment diagram for agent services" in lower
    assert "create a component diagram for a rag agent" in lower
    assert "agentic bpmn" in lower


@pytest.mark.parametrize("schema_name", ["ComponentModificationResponse", "DeploymentModificationResponse"])
def test_modification_schemas_use_small_output_budget(schema_name):
    assert schema_name in BaseDiagramHandler._SMALL_OUTPUT_SCHEMAS


@pytest.mark.parametrize(("message", "diagram_type"), [
    ("create a component diagram for a RAG agent swarm", "ComponentDiagram"),
    ("design a UML component model", "ComponentDiagram"),
    ("create a deployment diagram for the agent services", "DeploymentDiagram"),
    ("build the deployment model for a web shop", "DeploymentDiagram"),
])
def test_planner_single_diagram_heuristic(message, diagram_type):
    ops = _try_heuristic_decomposition(
        message, AssistantRequest(message=message, context=WorkspaceContext()), "complete_system",
    )
    assert ops == [{
        "type": "model", "diagramType": diagram_type, "mode": "complete_system", "request": message,
    }]


def test_capability_copy_mentions_new_types():
    for text in (OUT_OF_SCOPE_REDIRECT, META_ANSWER):
        assert "component and" in text and "deployment diagrams" in text


def test_fallback_and_quick_responses_list_new_types():
    from app import state_bodies

    help_text = state_bodies._QUICK_RESPONSES["help"]
    assert "10 diagram types" in help_text and "Component, and Deployment" in help_text
    capabilities = state_bodies._QUICK_RESPONSES["what_can_you_do"]
    assert "**Component Diagrams**" in capabilities and "**Deployment Diagrams**" in capabilities
    from prompts.fallback import fallback_prompt

    assert "component diagrams, deployment diagrams" in fallback_prompt("hi")


def test_quick_capability_list_is_one_markdown_list():
    """A stray blank line inside the bullet list splits it into two lists."""
    from app import state_bodies

    text = state_bodies._QUICK_RESPONSES["what_can_you_do"]
    block = text[text.index("**Create diagrams:**"):text.index("**Modify diagrams:**")]
    bullets = block.strip().split("\n")[1:]
    assert all(line.startswith("- **") for line in bullets), bullets


def test_training_sentences_cover_new_types():
    tree = ast.parse((_ROOT / "modeling_agent.py").read_text(encoding="utf-8"))
    sentences = [
        node.value.lower() for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert any("component diagram" in s for s in sentences)
    assert any("deployment diagram" in s for s in sentences)


def test_readme_table_lists_new_types():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    for diagram_type in NEW_TYPES:
        assert f"| `{diagram_type}` | Yes | Yes | Yes |" in readme
