"""The modeling assistant models; it does not push code generation.

BESSER-PEARL/BESSER#747: after building or editing a diagram the assistant
closed with "Want to keep refining, or continue with generating your
application?" (or "...your database", read from the word "database" in a
modeling request) and offered a "Generate application" button, as if the
editor were the agentic interface. A modeling reply now offers modeling next
steps only. Generation is still there when the user asks for it: a
"design X and generate Y" request (a paused plan) keeps its "continue with
generating?" question, and "generate the application" still routes to it.
Every closing line is English here and translated at the reply choke point
(``conversation.reply_language``), so these checks cover every reply language.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

import execution.model_operations as mo
from app import confirmation
from common import pending_dialogs
from common.pending_dialogs import (
    PENDING_GENERATOR_CONFIG,
    PENDING_GENERATOR_TYPE,
    PENDING_GUI_CHOICE,
    PENDING_WEBAPP_GENERATE,
    PLAN_GENERATION_CONFIRM_FLAG,
)
from conversation.reply_copy import REFINE_MODEL_PROMPT, continue_generating_prompt
from conversation.suggestions import get_suggested_actions
from protocol.types import AssistantRequest, WorkspaceContext
from tests.conftest import MINIMAL_CLASS_MODEL, FakeSession


def _offers_generation(payload):
    """True when a reply proposes generating code: in its text or its buttons."""
    message = (payload.get("message") or "").lower()
    if "generating your" in message or "generate the" in message:
        return True
    for action in payload.get("suggestedActions") or []:
        label = (action.get("label") or "").lower()
        prompt = (action.get("prompt") or "").lower()
        if label.startswith("generate") or prompt.startswith("generate"):
            return True
    return False


def _run(monkeypatch, message, mode, diagram_type="ClassDiagram", session=None):
    """Run execute_model_operation with a stub handler; return the sent payload."""
    session = session or FakeSession()
    handler = MagicMock()
    if mode == "modify_model":
        handler.generate_modification.return_value = {
            "action": "modify_model",
            "message": "Removed attribute **id** from **User**.",
            "modifications": [],
        }
    else:
        handler.generate_complete_system.return_value = {
            "action": "inject_complete_system",
            "message": "I created the model.",
            "systemSpec": {"classes": [{"className": "Brand"}, {"className": "Keyword"}]},
        }
    factory = MagicMock()
    factory.get_handler.return_value = handler
    monkeypatch.setattr(mo.ctx, "diagram_factory", factory)
    monkeypatch.setattr(mo, "reply_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(mo, "reply_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mo, "resolve_target_model",
        lambda *args, **kwargs: copy.deepcopy(MINIMAL_CLASS_MODEL) if mode == "modify_model" else None,
    )
    monkeypatch.setattr(mo, "build_workspace_context_block", lambda *args, **kwargs: "")
    sent = []
    monkeypatch.setattr(mo, "reply_payload", lambda _s, payload: sent.append(payload))
    request = AssistantRequest(
        message=message,
        context=WorkspaceContext(active_diagram_type=diagram_type),
    )
    mo.execute_model_operation(
        session, request,
        {"diagramType": diagram_type, "mode": mode, "request": message},
        mode, _skip_existing_check=True,
    )
    assert sent, "no reply was sent"
    return sent[-1]


# The two screenshots in the issue, plus requests that name an artifact.
_MODELING_REQUESTS = [
    "create a database to track brands, keywords and search results",
    "design a library management system",
    "create a REST API model for a shop",
    "build a django app model for a hotel",
]


@pytest.mark.parametrize("message", _MODELING_REQUESTS)
def test_a_built_model_offers_modeling_steps_only(monkeypatch, message):
    payload = _run(monkeypatch, message, "complete_system")
    assert not _offers_generation(payload), payload
    assert payload["message"].endswith(REFINE_MODEL_PROMPT)
    labels = [a["label"] for a in payload["suggestedActions"]]
    assert "Review the model" in labels and "Explain the specs" in labels


@pytest.mark.parametrize("message", [
    "remove the id attributes from every class",
    "remove the brandId field from the database",
])
def test_an_edit_does_not_ask_to_continue_with_generation(monkeypatch, message):
    payload = _run(monkeypatch, message, "modify_model")
    assert not _offers_generation(payload), payload
    assert "keep refining" not in payload["message"].lower()
    assert payload["message"] == "Removed attribute **id** from **User**."


@pytest.mark.parametrize("diagram_type", ["StateMachineDiagram", "BPMN", "AgentDiagram"])
def test_other_diagram_types_do_not_offer_generation(monkeypatch, diagram_type):
    payload = _run(monkeypatch, "create an order process", "complete_system", diagram_type)
    assert not _offers_generation(payload), payload


def test_a_request_that_asked_to_generate_still_gets_its_question(monkeypatch):
    """'design X and generate Y': planning paused the generation and this reply
    is where the user confirms it, so the question and the button stay."""
    session = FakeSession()
    pending_dialogs.put(session, PENDING_GENERATOR_TYPE, "sql")
    pending_dialogs.put(session, PENDING_GENERATOR_CONFIG, {PLAN_GENERATION_CONFIRM_FLAG: True})
    payload = _run(
        monkeypatch, "create a database for a library and generate the sql",
        "complete_system", session=session,
    )
    assert payload["message"].endswith(continue_generating_prompt("database"))
    assert payload["suggestedActions"][0] == {"label": "Generate database", "prompt": "generate sql"}


def test_an_edit_inside_a_generate_request_keeps_its_question(monkeypatch):
    session = FakeSession()
    pending_dialogs.put(session, PENDING_GENERATOR_TYPE, "django")
    pending_dialogs.put(session, PENDING_GENERATOR_CONFIG, {PLAN_GENERATION_CONFIRM_FLAG: True})
    payload = _run(monkeypatch, "rename User to Member and generate django", "modify_model", session=session)
    assert "continue with generating your Django app?" in payload["message"]


# ----------------------------------------------------------------------
# The quick-action pools
# ----------------------------------------------------------------------

_DIAGRAM_TYPES = [
    "ClassDiagram", "StateMachineDiagram", "GUINoCodeDiagram", "AgentDiagram",
    "ObjectDiagram", "QuantumCircuitDiagram", "BPMN", "UserDiagram",
    "ComponentDiagram", "DeploymentDiagram", "UnknownDiagram",
]
_MODES = ["complete_system", "single_element", "modify_model"]


@pytest.mark.parametrize("diagram_type", _DIAGRAM_TYPES)
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("available", [None, ["ClassDiagram"]])
def test_no_modeling_suggestion_proposes_generation(diagram_type, mode, available):
    actions = get_suggested_actions(diagram_type, mode, available_diagrams=available)
    assert actions
    assert not _offers_generation({"suggestedActions": actions}), actions


@pytest.mark.parametrize("diagram_type", ["ClassDiagram", "StateMachineDiagram"])
@pytest.mark.parametrize("mode", _MODES)
def test_context_aware_suggestions_do_not_propose_generation(diagram_type, mode):
    summary = {"element_names": ["Brand", "Keyword"], "element_count": 2, "relationship_count": 0}
    actions = get_suggested_actions(diagram_type, mode, model_summary=summary)
    assert actions
    assert not _offers_generation({"suggestedActions": actions}), actions


def test_after_a_requested_generation_the_suggestions_may_still_mention_it():
    """Post-generation chips follow a generation the user ran; out of scope."""
    actions = get_suggested_actions("ClassDiagram", "generation", generator_type="python")
    assert actions[0]["label"] == "Generate another format"


# ----------------------------------------------------------------------
# Screens laid out without a web-app request
# ----------------------------------------------------------------------

def _auto_gui(webapp_requested):
    session = FakeSession()
    pending_dialogs.put(session, PENDING_GUI_CHOICE, {
        "remaining_operations": [], "operation_request": "create screens for my classes",
    })
    if webapp_requested:
        pending_dialogs.put(session, PENDING_WEBAPP_GENERATE, True)
    request = MagicMock()
    request.message = "auto"
    payloads = []
    with patch.object(confirmation, "parse_assistant_request", return_value=request), \
         patch.object(confirmation, "reply_payload", side_effect=lambda _s, p: payloads.append(p)), \
         patch.object(confirmation, "reply_message"), \
         patch.object(confirmation, "emit_webapp_generate_prompt") as emit:
        assert confirmation.handle_pending_gui_choice(session)
    return session, payloads, emit


def test_screens_alone_get_a_modeling_follow_up():
    session, payloads, emit = _auto_gui(webapp_requested=False)
    (payload,) = payloads
    assert payload["action"] == "auto_generate_gui"
    assert not _offers_generation(payload), payload
    assert payload["message"].endswith(REFINE_MODEL_PROMPT)
    assert not emit.called


def test_screens_for_a_requested_web_app_still_ask_to_generate_it_once():
    session, payloads, emit = _auto_gui(webapp_requested=True)
    (payload,) = payloads
    assert payload["message"].endswith(continue_generating_prompt("web app"))
    assert payload["suggestedActions"][0]["label"] == "Generate web app"
    assert not emit.called
    # Consumed: no later path repeats the prompt.
    assert not pending_dialogs.get(session, PENDING_WEBAPP_GENERATE)


def test_the_greeting_leads_with_modeling():
    """The greeting promised "the model, the screens, and the code" and listed app examples first."""
    from app.state_bodies import GREETING_MESSAGE

    opening, examples = GREETING_MESSAGE.split("\n\n")[:2]
    assert "model" in opening.lower()
    for pitch in ("code", "screens", " app"):
        assert pitch not in opening.lower(), pitch
        assert pitch not in examples.lower(), pitch
    assert "When you ask for it, I generate code" in GREETING_MESSAGE
    assert GREETING_MESSAGE.endswith("What would you like to model today?")


def test_the_fallback_reply_does_not_offer_generation_unprompted():
    from prompts.fallback import fallback_prompt

    prompt = fallback_prompt("hello")
    identity = prompt.split("The user said:")[0]
    assert "code generation" not in identity.lower()
    assert "do not offer it unprompted" in prompt
