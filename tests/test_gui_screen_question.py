"""The "Basic CRUD pages or Experimental AI design" question for new screens.

Basic CRUD pages build one page per class, so a request that already lists
its screens lost every screen it named when that answer was picked; such a
request is built by the AI design without asking. A typed answer with no
question open (the screens were already built) is applied, not asked again.
"""

from unittest.mock import MagicMock, patch

import pytest

import execution.model_operations as modeling
from app import confirmation
from app import state_bodies as sb
from common import pending_dialogs
from common.pending_dialogs import PENDING_GUI_CHOICE
from protocol.actions import Action
from protocol.types import AssistantRequest, WorkspaceContext
from utilities.original_request import remember_original_request

from tests.conftest import MINIMAL_CLASS_MODEL, FakeSession

GUI_OP = {"diagramType": "GUINoCodeDiagram", "mode": "complete_system", "request": "Create the GUI for the shop"}


@pytest.fixture
def run_gui_op(monkeypatch):
    handler = MagicMock()
    handler.generate_complete_system.return_value = {"action": "assistant_message", "message": "ok"}
    factory = MagicMock()
    factory.get_handler.return_value = handler
    monkeypatch.setattr(modeling.ctx, "diagram_factory", factory)
    payloads = []
    monkeypatch.setattr(modeling, "reply_payload", lambda _s, payload: payloads.append(payload))
    for name in ("reply_progress", "reply_message"):
        monkeypatch.setattr(modeling, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(modeling, "resolve_target_model", lambda *args: None)
    monkeypatch.setattr(modeling, "resolve_class_diagram", lambda request: MINIMAL_CLASS_MODEL)
    monkeypatch.setattr(modeling, "build_workspace_context_block", lambda *args: "")

    def run(message, session=None, project_id=None):
        session = session or FakeSession()
        context = WorkspaceContext(project_snapshot={"id": project_id} if project_id else None)
        request = AssistantRequest(message=message, context=context)
        modeling.execute_model_operation(session, request, dict(GUI_OP), "complete_system")
        asked = any("How would you like me to create your screens" in str(p) for p in payloads)
        return session, handler, asked

    return run


@pytest.mark.parametrize("message", [
    "Create a web app for a book shop with three screens: Catalogue, Basket and Checkout.",
    "An order tracker. Exactly 4 screens, linked by a menu.",
    "Build the GUI. Screens: Customers, Orders, New order",
    "Create the following pages - Authors, Books",
])
def test_a_request_that_lists_its_screens_is_designed_without_asking(run_gui_op, message):
    session, handler, asked = run_gui_op(message)
    assert not asked
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None
    handler.generate_complete_system.assert_called_once()


def test_the_app_description_listing_screens_counts_for_a_later_screen_step(run_gui_op):
    session = FakeSession()
    remember_original_request(session, "A library app with two screens: Books and Authors.", "p1")
    _, handler, asked = run_gui_op("Create the GUI", session=session, project_id="p1")
    assert not asked
    handler.generate_complete_system.assert_called_once()


@pytest.mark.parametrize("message", [
    "Build screens for my app",
    "Create one page per class",
    "Add a page that lists the orders",
])
def test_a_request_without_its_screens_is_still_asked(run_gui_op, message):
    session, handler, asked = run_gui_op(message)
    assert asked
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is not None
    handler.generate_complete_system.assert_not_called()


@pytest.mark.parametrize("message,answer", [
    ("Basic CRUD pages", "auto"),
    ("Experimental AI design", "llm"),
])
def test_a_typed_answer_with_no_question_open_is_not_asked_again(run_gui_op, message, answer):
    session, handler, asked = run_gui_op(message)
    assert not asked
    pending = pending_dialogs.get(session, PENDING_GUI_CHOICE)
    assert pending and pending.get(modeling.GUI_CHOICE_IN_REQUEST)
    assert modeling.gui_choice_from_label(message) == answer
    handler.generate_complete_system.assert_not_called()


def test_the_modeling_body_settles_a_question_its_own_message_answers():
    session = FakeSession()

    def plan(**kwargs):
        pending_dialogs.put(session, PENDING_GUI_CHOICE, {
            "operation_request": "Create the GUI", "operation": dict(GUI_OP),
            modeling.GUI_CHOICE_IN_REQUEST: True,
        })

    request = AssistantRequest(message="Basic CRUD pages", context=WorkspaceContext())
    payloads = []
    with patch.object(sb, "_common_preamble", return_value=request), \
         patch.object(sb, "execute_planned_operations", side_effect=plan), \
         patch.object(confirmation, "parse_assistant_request", return_value=request), \
         patch.object(confirmation, "reply_payload", side_effect=lambda _s, p: payloads.append(p)):
        sb._modeling_state_body(session, "create_complete_system_intent", "complete_system", "EMPTY")
    assert [p["action"] for p in payloads] == [Action.AUTO_GENERATE_GUI]
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None


def test_a_typed_answer_wins_over_screens_the_app_description_lists(run_gui_op):
    session = FakeSession()
    remember_original_request(session, "A library app with two screens: Books and Authors.", "p1")
    _, handler, asked = run_gui_op("Basic CRUD pages", session=session, project_id="p1")
    assert not asked
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE).get(modeling.GUI_CHOICE_IN_REQUEST)
    handler.generate_complete_system.assert_not_called()
