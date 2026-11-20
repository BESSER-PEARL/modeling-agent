"""Progress lines for a complete build name the diagram being built.

Every type except the GUI used to get the class-diagram wording, so a BPMN
request showed "Generating classes and relationships..." (BESSER #653).
"""
from __future__ import annotations

import pytest

from diagram_handlers.registry.metadata import get_diagram_type_info
from execution.model_operations import _complete_system_progress_steps
from protocol.types import SUPPORTED_DIAGRAM_TYPES


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


# ---------------------------------------------------------------------------
# "Step n/m" lines of a multi-step plan
# ---------------------------------------------------------------------------

def _step_line(monkeypatch, operation) -> str:
    from execution import progress

    sent = []
    monkeypatch.setattr(
        progress, "reply_progress",
        lambda _session, message, **_kwargs: sent.append(message),
    )
    progress._report_progress(None, 1, 3, operation)
    (line,) = sent
    return line


@pytest.mark.parametrize("diagram_type", sorted(SUPPORTED_DIAGRAM_TYPES))
def test_step_line_names_the_diagram_as_the_user_knows_it(monkeypatch, diagram_type):
    line = _step_line(monkeypatch, {"type": "model", "diagramType": diagram_type})
    name = get_diagram_type_info(diagram_type)["name"]
    assert line == f"Step 2/3: Creating {name}…"


def test_step_line_of_an_unknown_diagram_type_shows_its_own_name(monkeypatch):
    line = _step_line(monkeypatch, {"type": "model", "diagramType": "KnowledgeGraphDiagram"})
    assert line == "Step 2/3: Creating KnowledgeGraphDiagram…"


@pytest.mark.parametrize("generator_type, expected", [
    ("web_app", "Step 2/3: Generating web app…"),
    ("rest_api", "Step 2/3: Generating REST API…"),
    ("jsonschema", "Step 2/3: Generating JSON Schema…"),
])
def test_step_line_names_what_is_generated(monkeypatch, generator_type, expected):
    line = _step_line(monkeypatch, {"type": "generation", "generatorType": generator_type})
    assert line == expected
