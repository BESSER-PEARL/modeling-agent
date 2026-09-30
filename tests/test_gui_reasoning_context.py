"""The design-reasoning pass must plan against the real class diagram.

It was told to "identify the 2-4 core entities" and invent their fields even
when a class diagram existed, so the plan drifted from the classes the
widgets bind to and never mentioned the class methods.
"""

from diagram_handlers.types.gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from schemas import AuthoredSystemGUISpec

TASK_METADATA = [{
    "id": "cls-task",
    "name": "Task",
    "attributes": [{"id": "a-hours", "name": "estimate_hours", "type": "float",
                    "isNumeric": True, "isString": False}],
    "methods": [{"id": "m-complete", "name": "complete", "isInstanceMethod": True, "params": []}],
}]


def _reasoning_prompt(class_metadata):
    captured = {}
    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)

    def _stub(**kw):
        captured.update(kw)
        return AuthoredSystemGUISpec(pages=[{"name": "Home", "sections": [
            {"html": "<section class='s'><h2>H</h2></section>"}]}])

    handler.predict_two_pass_structured = _stub
    handler.generate_complete_system("team task tracker", class_metadata=class_metadata)
    return captured["reasoning_prompt"]


def test_reasoning_pass_sees_the_class_diagram():
    prompt = _reasoning_prompt(TASK_METADATA)
    assert "Task" in prompt and "complete" in prompt and "estimate_hours" in prompt
    assert "Identify the 2-4 core entities" not in prompt


def test_reasoning_pass_without_class_diagram_still_plans_entities():
    assert "Identify the 2-4 core entities" in _reasoning_prompt(None)
