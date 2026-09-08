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
from session_keys import LAST_SMART_GEN_AT, PENDING_GUI_CHOICE  # noqa: E402
from handlers.generation_handler import _build_smart_gen_confirmation  # noqa: E402
from tests.conftest import FakeSession  # noqa: E402


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
    session = FakeSession()
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
    # No fix words in the message, but a smart-gen run already completed this
    # session → a follow-up is a modify of that app.
    session = FakeSession()
    session.set(LAST_SMART_GEN_AT, time.time())
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
    session = FakeSession()
    session.set(LAST_SMART_GEN_AT, time.time())
    payload = _build_smart_gen_confirmation(
        session, "a hotel booking system", "anthropic",
        reason_prefix="Model rebuilt and ready.",
        user_message="fix it",
    )
    msg = payload["message"]
    assert "Model rebuilt and ready." in msg
    assert "BESSER will generate your application" in msg
    assert payload["suggestedActions"][0]["label"] == "Continue"
