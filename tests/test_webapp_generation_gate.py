"""Bulletproof web-app pause in execute_planned_operations.

A "create a web app" plan builds the model + GUI and would then auto-run web_app
code generation. The plan's generation op is STRIPPED at the source (so nothing
can auto-run on any path), and a session flag drives the "generate the web app?"
prompt. Explicit non-GUI generations ("create X and generate django") are
untouched and still run.

These tests exercise the strip + prompt directly (the path that previously let a
rare auto-generation slip through).
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import execution.planning as planning  # noqa: E402


class _FakeSession:
    """Minimal stateful session so set()/get() work for the web-app flag."""
    def __init__(self):
        self._d = {}

    def get(self, key):
        return self._d.get(key)

    def set(self, key, value):
        self._d[key] = value


def _run(plan):
    """Drive execute_planned_operations with a fixed plan (GUI op does not halt).

    Returns (handle_generation_request_mock, emit_webapp_generate_prompt_mock).
    """
    gen = MagicMock(return_value={"action": "trigger_generator"})
    prompt = MagicMock()
    session = _FakeSession()
    with patch.object(planning, "plan_assistant_operations", return_value=plan), \
         patch.object(planning, "execute_model_operation", return_value="Diagram"), \
         patch.object(planning, "reply_payload"), \
         patch.object(planning, "reply_message"), \
         patch.object(planning, "emit_webapp_generate_prompt", prompt), \
         patch.object(planning, "handle_generation_request", gen), \
         patch.object(planning, "_report_progress"), \
         patch.object(planning, "build_request_for_target", side_effect=lambda r, t: r):
        planning.execute_planned_operations(
            session, MagicMock(), "complete_system", "create_complete_system_intent"
        )
    return gen, prompt


def test_webapp_plan_strips_generation_and_shows_prompt():
    plan = [
        {"type": "model", "diagramType": "GUINoCodeDiagram",
         "mode": "complete_system", "request": "create a GUI"},
        {"type": "generation", "generatorType": "web_app", "config": {}},
    ]
    gen, prompt = _run(plan)
    # web_app generation was stripped — it must NEVER auto-run
    assert not gen.called
    # and the user is shown the "generate the web app?" prompt
    assert prompt.called


def test_explicit_non_gui_generation_still_runs():
    # A class-diagram-only plan with an explicit generation op (no GUI) must NOT
    # be gated — the user explicitly asked to generate.
    plan = [
        {"type": "model", "diagramType": "ClassDiagram",
         "mode": "complete_system", "request": "create classes"},
        {"type": "generation", "generatorType": "django", "config": {}},
    ]
    gen, prompt = _run(plan)
    assert gen.called          # django generation runs (not gated)
    assert not prompt.called   # no web-app pause


# ---------------------------------------------------------------------------
# GUI-choice path: the pause nudge must survive the "auto"/"llm" answer.
#
# Because the generation op is stripped at the plan source, the GUI-choice's
# stored ``remaining_operations`` is EMPTY — so ``_resume_remaining_ops`` (which
# also emits the prompt) never runs. The choice handler must still show the
# "generate the web app?" nudge from its own tail. This is the exact gap a live
# probe caught: the GUI built, then went silent.
# ---------------------------------------------------------------------------

import confirmation  # noqa: E402
from session_keys import PENDING_GUI_CHOICE, PENDING_WEBAPP_GENERATE  # noqa: E402


def _run_gui_choice(answer):
    """Drive handle_pending_gui_choice with a web-app pause flag already set.

    Simulates the post-strip state: a GUI choice is pending with EMPTY
    remaining_operations (generation was stripped) and PENDING_WEBAPP_GENERATE
    is set. Returns the emit_webapp_generate_prompt mock and the session.
    """
    session = _FakeSession()
    session.set(PENDING_GUI_CHOICE, {"remaining_operations": [], "operation_request": "create a web app"})
    session.set(PENDING_WEBAPP_GENERATE, True)
    req = MagicMock()
    req.message = answer
    emit = MagicMock()
    with patch.object(confirmation, "parse_assistant_request", return_value=req), \
         patch.object(confirmation, "reply_payload"), \
         patch.object(confirmation, "reply_message"), \
         patch.object(confirmation, "replace", lambda obj, **kw: obj), \
         patch.object(confirmation, "execute_model_operation", return_value="Diagram"), \
         patch.object(confirmation, "_build_auto_gui_message", return_value="screens ready"), \
         patch.object(confirmation, "emit_webapp_generate_prompt", emit):
        handled = confirmation.handle_pending_gui_choice(session)
    return handled, emit, session


def test_gui_choice_auto_still_shows_generate_prompt():
    handled, emit, session = _run_gui_choice("auto")
    assert handled                                       # the choice was consumed
    assert emit.called                                   # the generate nudge fired
    assert session.get(PENDING_WEBAPP_GENERATE) is None   # flag cleared (no re-emit)


def test_gui_choice_llm_still_shows_generate_prompt():
    handled, emit, session = _run_gui_choice("llm")
    assert handled
    assert emit.called
    assert session.get(PENDING_WEBAPP_GENERATE) is None
