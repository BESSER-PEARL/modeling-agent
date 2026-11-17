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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import execution.planning as planning
from common import pending_dialogs
from common.pending_dialogs import PENDING_GUI_CHOICE, PENDING_WEBAPP_GENERATE
from common.session_keys import UNIFIED_CLASSIFICATION


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
    from common.pending_dialogs import PENDING_GENERATOR_TYPE
    assert pending_dialogs.get(session, PENDING_GENERATOR_TYPE) is None


def test_explicit_non_gui_mixed_plan_pauses_generation():
    # A class-diagram plan with an explicit generation op (no GUI) now pauses
    # exactly like the web-app flow: the model is built, the generation op is
    # stashed, and the user decides via the injection message's question.
    from common.pending_dialogs import (
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
    assert pending_dialogs.get(session, PENDING_GENERATOR_TYPE) == "django"
    stash = pending_dialogs.get(session, PENDING_GENERATOR_CONFIG)
    assert isinstance(stash, dict) and stash.get(PLAN_GENERATION_CONFIRM_FLAG) is True


def test_direct_generation_only_plan_still_runs():
    # A DIRECT generation request (no modeling step in the plan) keeps its
    # current behavior: the generator runs immediately, nothing is stashed.
    from common.pending_dialogs import PENDING_GENERATOR_TYPE
    plan = [
        {"type": "generation", "generatorType": "django", "config": {}},
    ]
    gen, prompt, session = _run(plan)
    assert gen.called
    assert not prompt.called
    assert pending_dialogs.get(session, PENDING_GENERATOR_TYPE) is None


# ---------------------------------------------------------------------------
# GUI-choice path: the pause nudge must survive the "auto"/"llm" answer.
#
# Because the generation op is stripped at the plan source, the GUI-choice's
# stored ``remaining_operations`` is EMPTY — so ``_resume_remaining_ops`` (which
# also emits the prompt) never runs. The choice handler must still show the
# "generate the web app?" nudge from its own tail. Otherwise the GUI builds,
# then goes silent.
# ---------------------------------------------------------------------------

from app import confirmation  # noqa: E402


def _run_gui_choice(answer):
    """Drive handle_pending_gui_choice with a web-app pause flag already set.

    Simulates the post-strip state: a GUI choice is pending with EMPTY
    remaining_operations (generation was stripped) and PENDING_WEBAPP_GENERATE
    is set. Returns the emit_webapp_generate_prompt mock and the session.
    """
    session = _FakeSession()
    pending_dialogs.put(session, PENDING_GUI_CHOICE, {"remaining_operations": [], "operation_request": "create a web app"})
    pending_dialogs.put(session, PENDING_WEBAPP_GENERATE, True)
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


# ---------------------------------------------------------------------------
# A screen build that fails must not be announced as ready.
#
# execute_model_operation returns None after it has sent the failure text (the
# handler produced no diagram), and may raise before the handler runs.
# ---------------------------------------------------------------------------

_WEBAPP_PLAN = [
    {"type": "model", "diagramType": "ClassDiagram",
     "mode": "complete_system", "request": "create a class diagram"},
    {"type": "model", "diagramType": "GUINoCodeDiagram",
     "mode": "complete_system", "request": "create a GUI"},
    {"type": "generation", "generatorType": "web_app", "config": {}},
]


def _run_plan(plan, execute, session=None):
    """Drive execute_planned_operations; *execute* stands in for
    execute_model_operation. Returns (prompt mock, session)."""
    prompt = MagicMock()
    session = session or _FakeSession()
    with patch.object(planning, "plan_assistant_operations", return_value=plan), \
         patch.object(planning, "execute_model_operation", side_effect=execute), \
         patch.object(planning, "reply_payload"), \
         patch.object(planning, "reply_message"), \
         patch.object(planning, "emit_webapp_generate_prompt", prompt), \
         patch.object(planning, "handle_generation_request", MagicMock()), \
         patch.object(planning, "_report_progress"), \
         patch.object(planning, "build_request_for_target", side_effect=lambda r, t: r):
        planning.execute_planned_operations(
            session, MagicMock(), "complete_system", "create_complete_system_intent"
        )
    return prompt, session


def _built(session, request, operation, default_mode):
    return operation["diagramType"]


def test_screens_ready_prompt_needs_a_built_screen_step():
    def execute(session, request, operation, default_mode):
        if operation["diagramType"] == "GUINoCodeDiagram":
            raise RuntimeError("screen build crashed")
        return operation["diagramType"]

    prompt, session = _run_plan(_WEBAPP_PLAN, execute)
    assert not prompt.called
    assert pending_dialogs.get(session, PENDING_WEBAPP_GENERATE) is None


def test_webapp_plan_with_built_screens_shows_the_prompt_once():
    prompt, session = _run_plan(_WEBAPP_PLAN, _built)
    assert prompt.call_count == 1
    assert pending_dialogs.get(session, PENDING_WEBAPP_GENERATE) is None


def test_failed_webapp_plan_is_not_announced_by_a_later_plan():
    # The class diagram step fails: the failure text is sent, None comes back
    # and no question is open.
    prompt, session = _run_plan(_WEBAPP_PLAN, lambda *a, **k: None)
    assert not prompt.called

    later_plan = [{"type": "model", "diagramType": "ClassDiagram",
                   "mode": "complete_system", "request": "create a class diagram"}]
    prompt, session = _run_plan(later_plan, _built, session)
    assert not prompt.called
    assert pending_dialogs.get(session, PENDING_WEBAPP_GENERATE) is None


def _answer_gui_choice(session, answer, execute, verdict=None):
    """Answer the open screen question on *session*; *execute* stands in for
    execute_model_operation and *verdict* for the classifier's verdict on the
    message. Returns (handled, emit mock, reply payloads, reply messages)."""
    req = MagicMock()
    req.message = answer
    session.set(UNIFIED_CLASSIFICATION, verdict)
    emit = MagicMock()
    payloads, messages = [], []
    with patch.object(confirmation, "parse_assistant_request", return_value=req), \
         patch.object(confirmation, "reply_payload", side_effect=lambda _s, p: payloads.append(p)), \
         patch.object(confirmation, "reply_message", side_effect=lambda _s, m: messages.append(m)), \
         patch.object(confirmation, "replace", lambda obj, **kw: obj), \
         patch.object(confirmation, "execute_model_operation", execute), \
         patch.object(confirmation, "_build_auto_gui_message", return_value="screens ready"), \
         patch.object(confirmation, "emit_webapp_generate_prompt", emit):
        handled = confirmation.handle_pending_gui_choice(session)
    return handled, emit, payloads, messages


def _open_webapp_gui_choice():
    session = _FakeSession()
    pending_dialogs.put(session, PENDING_GUI_CHOICE, {
        "remaining_operations": [], "operation_request": "create a web app",
        "operation": {"diagramType": "GUINoCodeDiagram"}, "default_mode": "complete_system",
    })
    pending_dialogs.put(session, PENDING_WEBAPP_GENERATE, True)
    return session


def _choice_labels(payloads):
    return [
        [a.get("label") for a in (p.get("suggestedActions") or [])]
        for p in payloads
    ]


@pytest.mark.parametrize("execute", [
    MagicMock(return_value=None),                      # handler produced no diagram
    MagicMock(side_effect=RuntimeError("crashed")),    # crashed before the handler
])
def test_failed_ai_design_keeps_the_screen_question_open(execute):
    session = _open_webapp_gui_choice()
    handled, emit, payloads, _ = _answer_gui_choice(session, "Experimental AI design", execute)

    assert handled
    assert not emit.called                                           # no "screens are ready"
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE)          # still answerable
    assert pending_dialogs.get(session, PENDING_WEBAPP_GENERATE)     # the plan is still on
    # The two choices are offered again, and nothing else.
    assert _choice_labels(payloads) == [["Basic CRUD pages", "Experimental AI design"]]
    assert payloads[0]["action"] == "assistant_message"


def test_ai_design_can_be_retried_after_a_failure():
    session = _open_webapp_gui_choice()
    _answer_gui_choice(session, "Experimental AI design", MagicMock(return_value=None))

    handled, emit, payloads, _ = _answer_gui_choice(
        session, "Experimental AI design", MagicMock(return_value="GUINoCodeDiagram"),
    )
    assert handled
    assert emit.call_count == 1
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None
    assert pending_dialogs.get(session, PENDING_WEBAPP_GENERATE) is None


def test_basic_pages_can_be_picked_after_a_failed_ai_design():
    session = _open_webapp_gui_choice()
    _answer_gui_choice(session, "Experimental AI design", MagicMock(return_value=None))

    handled, emit, payloads, _ = _answer_gui_choice(session, "Basic CRUD pages", MagicMock())
    assert handled
    assert [p["action"] for p in payloads] == ["auto_generate_gui"]
    assert _nudges_generation(payloads, emit)
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None


# ---------------------------------------------------------------------------
# The open screen question does not swallow the user's next request.
# ---------------------------------------------------------------------------

def _new_request_verdict(intent="modify_model_intent"):
    return SimpleNamespace(pending_flow_action="new_request", pending_flow_answer=None, intent=intent)


@pytest.mark.parametrize("verdict", [_new_request_verdict(), None])   # None: classifier outage
@pytest.mark.parametrize("after_failed_ai_design", [True, False])
@pytest.mark.parametrize("message", [
    "design a state machine for orders instead",       # "design" is an AI-design keyword
    "add a custom domain attribute to Shop",           # so is "custom"
    "set the default currency of Order to EUR",        # "default" is a basic-pages keyword
    "ok forget the screens for now and add a Payment class",   # "forget" is a cancel keyword
    "what went wrong?",
    "design a Payment class",                          # short, but a request
    "add an AI agent",
])
def test_new_request_leaves_the_screen_question(message, after_failed_ai_design, verdict):
    session = _open_webapp_gui_choice()
    if after_failed_ai_design:
        _answer_gui_choice(session, "Experimental AI design", MagicMock(return_value=None))
        assert pending_dialogs.get(session, PENDING_GUI_CHOICE)

    execute = MagicMock(return_value="GUINoCodeDiagram")
    handled, emit, payloads, messages = _answer_gui_choice(
        session, message, execute, verdict=verdict,
    )
    assert not handled              # normal routing takes the message
    assert not execute.called       # no screen build
    assert payloads == [] and messages == []
    assert not emit.called
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None
    assert pending_dialogs.get(session, PENDING_WEBAPP_GENERATE) is None


@pytest.mark.parametrize("label", ["Experimental AI design", "Basic CRUD pages"])
def test_a_choice_button_is_an_answer_whatever_the_classifier_says(label):
    session = _open_webapp_gui_choice()
    handled, _emit, _payloads, _messages = _answer_gui_choice(
        session, label, MagicMock(return_value="GUINoCodeDiagram"), verdict=_new_request_verdict(),
    )
    assert handled
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None


@pytest.mark.parametrize("verdict", [None, _new_request_verdict()])
@pytest.mark.parametrize("answer", ["try again", "Retry", "again", "Try again."])
def test_retry_words_after_a_failed_ai_design_run_it_again(answer, verdict):
    session = _open_webapp_gui_choice()
    _answer_gui_choice(session, "Experimental AI design", MagicMock(return_value=None))

    execute = MagicMock(return_value="GUINoCodeDiagram")
    handled, emit, payloads, _ = _answer_gui_choice(session, answer, execute, verdict=verdict)
    assert handled
    assert execute.called and execute.call_args.kwargs.get("_skip_gui_choice") is True
    assert all(p.get("action") != "auto_generate_gui" for p in payloads)
    assert emit.call_count == 1
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None


def _choice_made(handled, execute, payloads, messages):
    """Which answer the screen question took: llm, auto, cancel or None."""
    if not handled:
        return None
    if execute.called:
        return "llm"
    if [p.get("action") for p in payloads] == ["auto_generate_gui"]:
        return "auto"
    if messages == ["Cancelled. No GUI was generated."]:
        return "cancel"
    return "other"


@pytest.mark.parametrize("verdict", [_new_request_verdict(), None])
@pytest.mark.parametrize("after_failed_ai_design", [False, True])
@pytest.mark.parametrize("message, expected", [
    ("Experimental AI design please", "llm"),
    ("Basic CRUD pages, thanks", "auto"),
    ("AI please", "llm"),
    ("ai", "llm"),
    ("basic is fine", "auto"),
    ("go with the default", "auto"),
    ("the basic ones", "auto"),
    ("1", "auto"),
    ("2", "llm"),
    ("option 2", "llm"),
    ("crud", "auto"),
    ("cancel", "cancel"),
    ("never mind", "cancel"),
])
def test_a_short_answer_is_an_answer_whatever_the_classifier_says(
    message, expected, after_failed_ai_design, verdict,
):
    session = _open_webapp_gui_choice()
    if after_failed_ai_design:
        _answer_gui_choice(session, "Experimental AI design", MagicMock(return_value=None))

    execute = MagicMock(return_value="GUINoCodeDiagram")
    handled, _emit, payloads, messages = _answer_gui_choice(session, message, execute, verdict=verdict)
    assert _choice_made(handled, execute, payloads, messages) == expected
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None


@pytest.mark.parametrize("verdict", [_new_request_verdict(), None])
@pytest.mark.parametrize("message", ["try again please", "retry the AI design"])
def test_a_short_retry_after_a_failed_ai_design_runs_it_again(message, verdict):
    session = _open_webapp_gui_choice()
    _answer_gui_choice(session, "Experimental AI design", MagicMock(return_value=None))

    execute = MagicMock(return_value="GUINoCodeDiagram")
    handled, _emit, payloads, messages = _answer_gui_choice(session, message, execute, verdict=verdict)
    assert _choice_made(handled, execute, payloads, messages) == "llm"


def test_retry_words_mean_nothing_before_a_failed_ai_design():
    session = _open_webapp_gui_choice()
    execute = MagicMock(return_value="GUINoCodeDiagram")
    handled, _emit, _payloads, _messages = _answer_gui_choice(session, "try again", execute)
    assert not handled
    assert not execute.called


_GUI_OP = {"type": "model", "diagramType": "GUINoCodeDiagram",
           "mode": "complete_system", "request": "create a GUI"}


def _resume_with_screen_step(execute):
    """Resume a web-app plan whose class diagram step waited for replace/keep:
    the screen step is what is left. Returns (emit mock, session)."""
    session = _FakeSession()
    pending_dialogs.put(session, PENDING_WEBAPP_GENERATE, True)
    emit = MagicMock()
    with patch.object(confirmation, "execute_model_operation", side_effect=execute), \
         patch.object(confirmation, "reply_message"), \
         patch.object(confirmation, "emit_webapp_generate_prompt", emit), \
         patch("orchestrator.request_builders.build_request_for_target", side_effect=lambda r, t: r):
        confirmation._resume_remaining_ops(
            session, [dict(_GUI_OP)], MagicMock(), "ClassDiagram", "complete_system",
            "create a web app", {},
        )
    return emit, session


def test_resumed_plan_asks_the_screen_question_without_announcing_screens():
    def asks_how_to_create_the_screens(session, request, operation, default_mode):
        pending_dialogs.put(session, PENDING_GUI_CHOICE, {"operation": operation})
        return None

    emit, session = _resume_with_screen_step(asks_how_to_create_the_screens)
    assert not emit.called
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE)
    assert pending_dialogs.get(session, PENDING_WEBAPP_GENERATE)  # for the answer's follow-up


def _raises(**_kwargs):
    raise RuntimeError("screen build crashed")


@pytest.mark.parametrize("execute", [lambda **_kwargs: None, _raises])
def test_resumed_plan_does_not_announce_screens_that_failed(execute):
    emit, _session = _resume_with_screen_step(execute)
    assert not emit.called


def test_resumed_plan_announces_screens_it_built():
    emit, session = _resume_with_screen_step(lambda **_kwargs: "GUINoCodeDiagram")
    assert emit.call_count == 1
    assert pending_dialogs.get(session, PENDING_WEBAPP_GENERATE) is None


def test_cancelling_the_screen_question_ends_the_webapp_plan():
    session = _open_webapp_gui_choice()
    handled, emit, payloads, messages = _answer_gui_choice(session, "cancel", MagicMock())

    assert handled
    assert not emit.called
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None
    assert pending_dialogs.get(session, PENDING_WEBAPP_GENERATE) is None

    # A later plan on the same session does not announce screens.
    later_plan = [{"type": "model", "diagramType": "ClassDiagram",
                   "mode": "complete_system", "request": "create a class diagram"}]
    prompt, _ = _run_plan(later_plan, _built, session)
    assert not prompt.called
