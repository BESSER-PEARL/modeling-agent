"""Bulletproof web-app pause in execute_planned_operations.

A "create a web app" plan builds the model + GUI and would then auto-run web_app
code generation. The plan's generation op is STRIPPED at the source (so nothing
can auto-run on any path), and a session flag drives the "generate the web app?"
prompt. Mixed plans for every OTHER generator ("create X and generate django")
now pause the same way — the generation op is stashed in the pending-generation
state awaiting the user's explicit go-ahead (see test_mixed_plan_generation_pause
for that flow); only DIRECT generation-only plans still run immediately.

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

    Returns (handle_generation_request_mock, emit_webapp_generate_prompt_mock,
    session).
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
    return gen, prompt, session


def test_webapp_plan_strips_generation_and_shows_prompt():
    plan = [
        {"type": "model", "diagramType": "GUINoCodeDiagram",
         "mode": "complete_system", "request": "create a GUI"},
        {"type": "generation", "generatorType": "web_app", "config": {}},
    ]
    gen, prompt, session = _run(plan)
    # web_app generation was stripped — it must NEVER auto-run
    assert not gen.called
    # and the user is shown the "generate the web app?" prompt
    assert prompt.called
    # the web-app pause keeps its own mechanism — no generic stash armed
    from session_keys import PENDING_GENERATOR_TYPE
    assert session.get(PENDING_GENERATOR_TYPE) is None


def test_explicit_non_gui_mixed_plan_pauses_generation():
    # A class-diagram plan with an explicit generation op (no GUI) now pauses
    # exactly like the web-app flow: the model is built, the generation op is
    # stashed, and the user decides via the injection message's question.
    from session_keys import (
        PENDING_GENERATOR_CONFIG,
        PENDING_GENERATOR_TYPE,
        PLAN_GENERATION_CONFIRM_FLAG,
    )
    plan = [
        {"type": "model", "diagramType": "ClassDiagram",
         "mode": "complete_system", "request": "create classes"},
        {"type": "generation", "generatorType": "django", "config": {}},
    ]
    gen, prompt, session = _run(plan)
    assert not gen.called      # django generation must NOT auto-run
    assert not prompt.called   # and the web-app prompt stays web-app-only
    assert session.get(PENDING_GENERATOR_TYPE) == "django"
    stash = session.get(PENDING_GENERATOR_CONFIG)
    assert isinstance(stash, dict) and stash.get(PLAN_GENERATION_CONFIRM_FLAG) is True


def test_direct_generation_only_plan_still_runs():
    # A DIRECT generation request (no modeling step in the plan) keeps its
    # current behavior: the generator runs immediately, nothing is stashed.
    from session_keys import PENDING_GENERATOR_TYPE
    plan = [
        {"type": "generation", "generatorType": "django", "config": {}},
    ]
    gen, prompt, session = _run(plan)
    assert gen.called
    assert not prompt.called
    assert session.get(PENDING_GENERATOR_TYPE) is None


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
    payloads = []
    with patch.object(confirmation, "parse_assistant_request", return_value=req), \
         patch.object(confirmation, "reply_payload", side_effect=lambda _s, p: payloads.append(p)), \
         patch.object(confirmation, "reply_message"), \
         patch.object(confirmation, "replace", lambda obj, **kw: obj), \
         patch.object(confirmation, "execute_model_operation", return_value="Diagram"), \
         patch.object(confirmation, "_build_auto_gui_message", return_value="screens ready"), \
         patch.object(confirmation, "emit_webapp_generate_prompt", emit):
        handled = confirmation.handle_pending_gui_choice(session)
    return handled, emit, session, payloads


def _nudges_generation(payloads, emit):
    """True when the user was prompted to generate — either the standalone
    emit_webapp_generate_prompt fired (LLM path) or the reply carries a
    'Generate …' suggested action (auto path embeds the nudge in-message)."""
    if emit.called:
        return True
    for p in payloads:
        labels = " ".join(a.get("label", "") for a in (p.get("suggestedActions") or []))
        if "Generate" in labels:
            return True
    return False


def _auto_ran_generation(payloads):
    return any(p.get("action") in ("trigger_generator", "trigger_smart_generator")
              for p in payloads)


def test_gui_choice_auto_still_shows_generate_prompt():
    handled, emit, session, payloads = _run_gui_choice("auto")
    assert handled                              # the choice was consumed
    assert _nudges_generation(payloads, emit)   # the generate nudge is shown
    assert not _auto_ran_generation(payloads)   # and code-gen did NOT auto-run


def test_gui_choice_llm_still_shows_generate_prompt():
    handled, emit, session, payloads = _run_gui_choice("llm")
    assert handled
    assert _nudges_generation(payloads, emit)
    assert not _auto_ran_generation(payloads)
