"""Web-app generation gate in execute_planned_operations (Phase 2).

When a plan built a GUI (a GUINoCodeDiagram model op) and the GUI op did NOT halt
for a choice, execution falls through to the direct generation loop. That loop
must DEFER code generation (ask the user to review/generate), like the
class-diagram flow — NOT auto-run web_app generation.

This fall-through path previously had ZERO coverage, which let a NameError
(`model_ops` undefined) ship to production. These tests exercise it directly.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import execution.planning as planning  # noqa: E402


def _run(plan):
    """Drive execute_planned_operations with a fixed plan (GUI op does not halt).

    Returns (handle_generation_request_mock, list_of_reply_payloads).
    """
    captured = []
    gen = MagicMock(return_value={"action": "trigger_generator"})
    session = MagicMock()
    session.get.return_value = None
    with patch.object(planning, "plan_assistant_operations", return_value=plan), \
         patch.object(planning, "execute_model_operation", return_value="Diagram"), \
         patch.object(planning, "reply_payload", side_effect=lambda s, p: captured.append(p)), \
         patch.object(planning, "reply_message"), \
         patch.object(planning, "handle_generation_request", gen), \
         patch.object(planning, "_report_progress"), \
         patch.object(planning, "build_request_for_target", side_effect=lambda r, t: r):
        planning.execute_planned_operations(
            session, MagicMock(), "complete_system", "create_complete_system_intent"
        )
    return gen, captured


def test_webapp_plan_defers_generation_when_gui_op_does_not_halt():
    plan = [
        {"type": "model", "diagramType": "GUINoCodeDiagram",
         "mode": "complete_system", "request": "create a GUI"},
        {"type": "generation", "generatorType": "web_app", "config": {}},
    ]
    gen, captured = _run(plan)
    # web_app generation must NOT auto-run
    assert not gen.called
    # a defer prompt with the Generate suggestion was sent (no NameError crash)
    assert captured, "expected a defer message"
    defer = captured[-1]
    assert "generate the web app" in (defer.get("message") or "").lower()
    labels = [a.get("label") for a in defer.get("suggestedActions", [])]
    assert any("Generate the web app" in (label or "") for label in labels)


def test_explicit_non_gui_generation_still_runs():
    # A class-diagram-only plan with an explicit generation op (no GUI) must NOT
    # be gated — the user explicitly asked to generate.
    plan = [
        {"type": "model", "diagramType": "ClassDiagram",
         "mode": "complete_system", "request": "create classes"},
        {"type": "generation", "generatorType": "django", "config": {}},
    ]
    gen, _captured = _run(plan)
    assert gen.called  # django generation runs (not gated)
