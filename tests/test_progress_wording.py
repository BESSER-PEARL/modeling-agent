"""Progress lines for a complete build name the diagram being built.

Every type except the GUI used to get the class-diagram wording, so a BPMN
request showed "Generating classes and relationships..." (BESSER #653).
"""
from __future__ import annotations

import pytest

from src.diagram_handlers.registry.metadata import get_diagram_type_info
from src.execution.model_operations import _complete_system_progress_steps


def _lines(diagram_type: str) -> list:
    label = get_diagram_type_info(diagram_type).get("name", diagram_type)
    return [msg for _delay, msg in _complete_system_progress_steps(diagram_type, label)]


@pytest.mark.parametrize(
    "diagram_type",
    ["BPMN", "StateMachineDiagram", "ObjectDiagram", "AgentDiagram", "QuantumCircuitDiagram",
     "ComponentDiagram", "DeploymentDiagram", "UserDiagram"],
)
def test_non_class_builds_do_not_mention_classes(diagram_type):
    text = " ".join(_lines(diagram_type)).lower()
    assert "class" not in text
    assert "attributes and methods" not in text


def test_bpmn_build_names_the_bpmn_diagram():
    assert _lines("BPMN")[0] == "Building your BPMN Diagram..."


def test_class_and_gui_wording_is_unchanged():
    assert _lines("ClassDiagram")[0] == "Generating classes and relationships..."
    assert _lines("GUINoCodeDiagram")[0].startswith("Designing your screens")
