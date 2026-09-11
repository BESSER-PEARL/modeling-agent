"""Focused tests for the pilot UX fixes (#3 new-diagram-tab wording,
#4 AI-Generated screen-creation suggestion, #5 clickable API-key link).

These exercise the exact strings and routing the pilot feedback called out,
without requiring a live LLM (the classifier path is bypassed; the keyword
fallback and the message builders are pure).
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time  # noqa: E402

import execution.model_operations as mo  # noqa: E402
import confirmation  # noqa: E402
from session_keys import (  # noqa: E402
    LAST_SMART_GEN_AT,
    LAST_SMART_GEN_PROJECT_ID,
    PENDING_GUI_CHOICE,
    SMART_GEN_ARMED_PROJECT_ID,
)
from handlers.generation_handler import _build_smart_gen_confirmation  # noqa: E402
from tests.conftest import FakeSession, make_session  # noqa: E402


# ---------------------------------------------------------------------------
# #4 — the AI-Generated screen-creation suggestion
# ---------------------------------------------------------------------------

_CLASS_DIAGRAM = {"elements": {"c1": {"type": "Class", "name": "Product"}}}


def _emit_gui_choice():
    """Drive execute_model_operation to the GUI generation-mode choice and
    return the payload it emits (message + suggestedActions)."""
    session = FakeSession()
    request = MagicMock()
    request.message = "create a web app for my classes"
    operation = {
        "diagramType": "GUINoCodeDiagram",
        "mode": "complete_system",
        "request": "create a web app for my classes",
    }
    captured = []
    with patch.object(mo, "resolve_class_diagram", return_value=_CLASS_DIAGRAM), \
         patch.object(mo, "reply_payload", side_effect=lambda _s, p: captured.append(p)):
        result = mo.execute_model_operation(
            session=session,
            request=request,
            operation=operation,
            default_mode="complete_system",
            _skip_existing_check=True,
        )
    assert result is None  # a GUI-choice prompt was stored
    assert len(captured) == 1
    return captured[0]


def test_gui_choice_ai_generated_suggestion_sends_human_phrase():
    payload = _emit_gui_choice()
    actions = payload.get("suggestedActions") or []
    labels = {a["label"]: a["prompt"] for a in actions}

    # The AI option is surfaced as "AI-Generated (experimental)" and its
    # prompt is the SAME human-meaningful phrase — never the opaque "llm".
    assert "AI-Generated (experimental)" in labels
    assert labels["AI-Generated (experimental)"] == "AI-Generated (experimental)"
    assert all(a["prompt"] != "llm" for a in actions)

    # The deterministic option is preserved.
    assert labels.get("Fast & deterministic") == "Fast & deterministic"


def test_gui_choice_message_has_no_preselection():
    payload = _emit_gui_choice()
    # Exactly two options, presented as an equal choice — no default/preselected
    # flag on either one.
    actions = payload.get("suggestedActions") or []
    assert len(actions) == 2
    for a in actions:
        assert set(a.keys()) == {"label", "prompt"}


def _route_gui_choice(answer: str):
    """Drive handle_pending_gui_choice with a pending GUI choice and a given
    answer (no UNIFIED_CLASSIFICATION set → keyword-fallback path)."""
    session = FakeSession()
    session.set(PENDING_GUI_CHOICE, {
        "operation_request": "create a web app",
        "operation": {"diagramType": "GUINoCodeDiagram"},
        "default_mode": "complete_system",
    })
    request = MagicMock()
    request.message = answer
    exec_mock = MagicMock(return_value="GUINoCodeDiagram")
    payloads = []
    with patch.object(confirmation, "parse_assistant_request", return_value=request), \
         patch.object(confirmation, "reply_payload", side_effect=lambda _s, p: payloads.append(p)), \
         patch.object(confirmation, "reply_message"), \
         patch.object(confirmation, "replace", lambda obj, **kw: obj), \
         patch.object(confirmation, "execute_model_operation", exec_mock), \
         patch.object(confirmation, "emit_webapp_generate_prompt"):
        handled = confirmation.handle_pending_gui_choice(session)
    return handled, exec_mock, payloads


def test_ai_generated_phrase_reaches_ai_gui_generation():
    handled, exec_mock, payloads = _route_gui_choice("AI-Generated (experimental)")
    assert handled
    # LLM/AI-GUI path: execute_model_operation is invoked with the
    # skip-gui-choice flag (the same trigger "llm" used to fire).
    assert exec_mock.called
    assert exec_mock.call_args.kwargs.get("_skip_gui_choice") is True
    # It did NOT take the deterministic auto path.
    assert all(p.get("action") != "auto_generate_gui" for p in payloads)


def test_legacy_llm_token_still_routes_to_ai_gui():
    # Back-compat: a literal "llm" (older clients / typed answers) still works.
    handled, exec_mock, _ = _route_gui_choice("llm")
    assert handled
    assert exec_mock.called
    assert exec_mock.call_args.kwargs.get("_skip_gui_choice") is True


def test_fast_deterministic_still_takes_auto_path():
    handled, exec_mock, payloads = _route_gui_choice("Fast & deterministic")
    assert handled
    # Deterministic path emits auto_generate_gui and does NOT call the LLM path.
    assert any(p.get("action") == "auto_generate_gui" for p in payloads)
    assert not exec_mock.called


# ---------------------------------------------------------------------------
# #3 — "new diagram tab" wording in the existing-model confirmation
# ---------------------------------------------------------------------------

def test_existing_model_confirmation_says_new_diagram_tab():
    session = FakeSession()
    request = MagicMock()
    # get_all_diagrams_of_type drives the tab count / can_add_tab flag.
    request.context.get_all_diagrams_of_type.return_value = ["tab-1"]
    captured = []
    with patch.object(mo, "reply_payload", side_effect=lambda _s, p: captured.append(p)):
        mo._build_existing_model_confirmation(
            session=session,
            request=request,
            target_diagram_type="ClassDiagram",
            existing_summary="2 classes",
            pending_data={},
            source_description="I can create a new ClassDiagram",
        )
    assert len(captured) == 1
    payload = captured[0]
    # The user-visible copy disambiguates from a browser tab.
    assert "new diagram tab" in payload["message"]
    assert "new tab**?" not in payload["message"]
    labels = [a["label"] for a in payload.get("suggestedActions", [])]
    assert "Create in a new diagram tab" in labels
    # The routing prompt is unchanged so NEW_TAB_KEYWORDS still matches.
    new_tab_action = next(
        a for a in payload["suggestedActions"] if a["label"] == "Create in a new diagram tab"
    )
    assert new_tab_action["prompt"] == "new tab"


# ---------------------------------------------------------------------------
# #5 — clickable "set up your own API key" link in the pre-gen confirmation
# ---------------------------------------------------------------------------

def test_smart_gen_confirmation_has_clickable_key_link():
    session = FakeSession()
    payload = _build_smart_gen_confirmation(
        session, "build a blog app", "anthropic",
    )
    msg = payload["message"]
    # The key mention is a markdown link the frontend intercepts (wme:add-key)
    # to open the BYOK dialog — see markdown-renderer.tsx.
    assert "[set up your own API key](wme:add-key)" in msg
    # The generate/continue action is preserved.
    labels = [a["label"] for a in payload.get("suggestedActions", [])]
    assert "Continue" in labels


# ---------------------------------------------------------------------------
# Context-aware confirmation: FIX/MODIFY vs FIRST generation
# ---------------------------------------------------------------------------

_CONFIRM_PROMPT = "generate anyway with my current model"

_PROJECT_ID = "proj-todo-0001"
_OTHER_PROJECT_ID = "proj-shop-0002"


def _project_session(message: str, project_id: str = _PROJECT_ID) -> FakeSession:
    """A session whose workspace context carries a real project id, exactly as
    the frontend sends it (the whole project object as ``projectSnapshot``;
    ``BesserProject.id`` survives its context compaction)."""
    return make_session(
        message,
        project_snapshot={
            "id": project_id,
            "type": "Project",
            "name": "Todo",
            "diagrams": {},
        },
    )


def _record_generated_app(session, project_id: str, *, minutes_ago: float = 0.0):
    """Session state left behind by a SUCCESSFUL smart-gen run for a project —
    what ``_handle_smart_generator_result`` writes."""
    session.set(LAST_SMART_GEN_AT, time.time() - minutes_ago * 60)
    session.set(LAST_SMART_GEN_PROJECT_ID, project_id)


def test_confirmation_first_generation_uses_from_scratch_copy():
    session = FakeSession()
    payload = _build_smart_gen_confirmation(
        session, "a blog with posts and comments", "anthropic",
        user_message="create a blog app",
    )
    msg = payload["message"]
    assert "BESSER will generate your application" in msg
    assert "update your existing app" not in msg
    actions = payload["suggestedActions"]
    assert actions[0]["label"] == "Continue"
    assert actions[0]["prompt"] == _CONFIRM_PROMPT
    # Clickable key link present in the first-gen branch.
    assert "[set up your own API key](wme:add-key)" in msg


def test_confirmation_fix_message_uses_fix_framing():
    # Fix words AND an app already generated for this project — the only
    # combination that may claim "your existing app".
    session = _project_session("I get a 500 error when I submit the form, please fix it")
    _record_generated_app(session, _PROJECT_ID)
    payload = _build_smart_gen_confirmation(
        session, "the endpoint 500s", "anthropic",
        user_message="I get a 500 error when I submit the form, please fix it",
    )
    msg = payload["message"]
    assert "update your existing app" in msg
    assert "BESSER will generate your application" not in msg
    actions = payload["suggestedActions"]
    # The label changes to "Fix it" but the confirm token is UNCHANGED so the
    # existing confirm gate still fires.
    assert actions[0]["label"] == "Fix it"
    assert actions[0]["prompt"] == _CONFIRM_PROMPT
    # Clickable key link present in the fix branch too.
    assert "[set up your own API key](wme:add-key)" in msg


def test_confirmation_fix_framing_from_recent_run_signal():
    # No fix words in the message, but a smart-gen run already completed FOR
    # THIS PROJECT → a follow-up is a modify of that app.
    session = _project_session("add a search bar")
    _record_generated_app(session, _PROJECT_ID)
    payload = _build_smart_gen_confirmation(
        session, "add a search bar to the product list", "anthropic",
        user_message="add a search bar",
    )
    actions = payload["suggestedActions"]
    assert actions[0]["label"] == "Fix it"
    assert actions[0]["prompt"] == _CONFIRM_PROMPT


def test_confirmation_rebuild_resume_stays_from_scratch():
    # A mismatch/rebuild resume carries a reason_prefix and is a fresh build —
    # even with a recent prior run, it must NOT be fix-framed.
    session = _project_session("fix it")
    _record_generated_app(session, _PROJECT_ID)
    payload = _build_smart_gen_confirmation(
        session, "a hotel booking system", "anthropic",
        reason_prefix="Model rebuilt and ready.",
        user_message="fix it",
    )
    msg = payload["message"]
    assert "Model rebuilt and ready." in msg
    assert "BESSER will generate your application" in msg
    assert payload["suggestedActions"][0]["label"] == "Continue"


# ---------------------------------------------------------------------------
# The fix copy is PROJECT-scoped (live bug 2026-09-11)
#
# Reported sequence, all inside ONE brand-new project:
#   1. "I want a todo app"                  → "I'll update your existing app…" [Fix it]
#   2. "I want you to model a new todoapp"  → model built (no code generated)
#   3. "generate the application"           → "I'll update your existing app…" [Fix it]
# Nothing in that project had ever been generated. The recency signal was
# session-scoped (the BAF session survives a project switch) so a run in an
# EARLIER project still satisfied it, and the vocabulary check also read the
# classifier's machine-written refined instructions.
# ---------------------------------------------------------------------------

def test_fresh_project_first_message_is_from_scratch():
    """Step 1 of the report: first message in a brand-new project."""
    session = _project_session("I want a todo app")
    payload = _build_smart_gen_confirmation(
        session, "Build a todo app with tasks the user can complete", "anthropic",
        user_message="I want a todo app",
    )
    msg = payload["message"]
    assert "update your existing app" not in msg
    assert "BESSER will generate your application" in msg
    assert payload["suggestedActions"][0]["label"] == "Continue"


def test_generate_after_modelling_only_is_from_scratch():
    """Step 3 of the report: the project has a MODEL but has never produced an
    app, and a run from a previous project is still fresh in the session."""
    session = _project_session("generate the application")
    _record_generated_app(session, _OTHER_PROJECT_ID)
    payload = _build_smart_gen_confirmation(
        session, "Generate the TodoApp from the class diagram", "anthropic",
        user_message="generate the application",
    )
    msg = payload["message"]
    assert "update your existing app" not in msg
    assert "BESSER will generate your application" in msg
    assert payload["suggestedActions"][0]["label"] == "Continue"


def test_project_switch_after_recent_run_goes_back_to_from_scratch():
    """A recent run in project A must not fix-frame a request in project B."""
    session = _project_session("add a search bar", project_id=_OTHER_PROJECT_ID)
    _record_generated_app(session, _PROJECT_ID)  # the run belongs to A, we are in B
    payload = _build_smart_gen_confirmation(
        session, "add a search bar to the list", "anthropic",
        user_message="add a search bar",
    )
    assert payload["suggestedActions"][0]["label"] == "Continue"
    # …and switching back to A restores the fix framing.
    back = _project_session("add a search bar", project_id=_PROJECT_ID)
    _record_generated_app(back, _PROJECT_ID)
    payload_back = _build_smart_gen_confirmation(
        back, "add a search bar to the list", "anthropic",
        user_message="add a search bar",
    )
    assert payload_back["suggestedActions"][0]["label"] == "Fix it"


def test_recent_run_signal_expires_after_30_minutes():
    """Same project, but the run is older than the window: a neutral request
    falls back to the from-scratch copy."""
    session = _project_session("add a search bar")
    _record_generated_app(session, _PROJECT_ID, minutes_ago=31)
    payload = _build_smart_gen_confirmation(
        session, "add a search bar to the list", "anthropic",
        user_message="add a search bar",
    )
    assert payload["suggestedActions"][0]["label"] == "Continue"


def test_expired_run_plus_explicit_fix_words_stays_fix_framed():
    """The app still exists in this project, so the user's own fix vocabulary
    is honored beyond the 30-minute window."""
    session = _project_session("the login page is broken, please fix it")
    _record_generated_app(session, _PROJECT_ID, minutes_ago=120)
    payload = _build_smart_gen_confirmation(
        session, "repair the login page", "anthropic",
        user_message="the login page is broken, please fix it",
    )
    assert payload["suggestedActions"][0]["label"] == "Fix it"


def test_fix_words_without_an_app_in_this_project_are_not_fix_framed():
    """Fix vocabulary alone can no longer reach the fix copy: there is nothing
    to fix in a project that never generated anything."""
    session = _project_session("fix the checkout page, it crashes")
    payload = _build_smart_gen_confirmation(
        session, "repair the checkout page", "anthropic",
        user_message="fix the checkout page, it crashes",
    )
    assert payload["suggestedActions"][0]["label"] == "Continue"


def test_error_words_in_refined_instructions_do_not_flip_the_copy():
    """The classifier's refined instructions are machine-written feature prose:
    an ordinary first build mentions error messages / 404s, which used to match
    the fix vocabulary and flip the copy."""
    session = _project_session("I want a todo app")
    payload = _build_smart_gen_confirmation(
        session,
        "Build a todo app with form validation that shows clear error messages, "
        "and a REST API returning 404 for unknown tasks",
        "anthropic",
        user_message="I want a todo app",
    )
    assert payload["suggestedActions"][0]["label"] == "Continue"


def test_unknown_project_id_fails_closed_to_from_scratch():
    """No resolvable project id (older frontend / widget mode): the signal is
    unusable, so the neutral copy wins rather than claiming an existing app."""
    session = FakeSession()  # no workspace context at all
    _record_generated_app(session, _PROJECT_ID)
    payload = _build_smart_gen_confirmation(
        session, "add a search bar to the list", "anthropic",
        user_message="add a search bar",
    )
    assert payload["suggestedActions"][0]["label"] == "Continue"


def test_armed_project_is_recorded_and_promoted_on_success():
    """The run's project is captured when it is ARMED (the completion callback
    is a context-free frontend_event) and promoted once the run succeeds."""
    import types as _types

    from handlers.generation_handler import _handle_frontend_event

    session = _project_session("I want a todo app")
    _build_smart_gen_confirmation(
        session, "Build a todo app", "anthropic", user_message="I want a todo app",
    )
    assert session.get(SMART_GEN_ARMED_PROJECT_ID) == _PROJECT_ID
    assert session.get(LAST_SMART_GEN_PROJECT_ID) is None  # armed is not finished

    result = _types.SimpleNamespace(
        action="frontend_event", message="",
        raw_payload={
            "eventType": "generator_result", "ok": True,
            "metadata": {"smart": True},
        },
    )
    _handle_frontend_event(result, session)
    assert session.get(LAST_SMART_GEN_PROJECT_ID) == _PROJECT_ID

    # …and the follow-up in that same project is now fix-framed.
    follow_up = _project_session("add a search bar")
    follow_up.set(LAST_SMART_GEN_AT, session.get(LAST_SMART_GEN_AT))
    follow_up.set(LAST_SMART_GEN_PROJECT_ID, session.get(LAST_SMART_GEN_PROJECT_ID))
    payload = _build_smart_gen_confirmation(
        follow_up, "add a search bar to the list", "anthropic",
        user_message="add a search bar",
    )
    assert payload["suggestedActions"][0]["label"] == "Fix it"


def test_failed_run_does_not_mark_the_project_as_generated():
    """Only SUCCESSFUL runs count — a failed run leaves no app to fix."""
    import types as _types

    from handlers.generation_handler import _handle_frontend_event

    session = _project_session("I want a todo app")
    _build_smart_gen_confirmation(
        session, "Build a todo app", "anthropic", user_message="I want a todo app",
    )
    failure = _types.SimpleNamespace(
        action="frontend_event", message="",
        raw_payload={
            "eventType": "generator_result", "ok": False,
            "metadata": {"smart": True, "errorCode": "COST_CAP"},
        },
    )
    _handle_frontend_event(failure, session)
    assert session.get(LAST_SMART_GEN_PROJECT_ID) is None
