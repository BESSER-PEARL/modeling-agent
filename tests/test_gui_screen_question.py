"""The "Basic CRUD pages or Experimental AI design" question for new screens.

Basic CRUD pages build one page per class, so a request that already lists
its screens lost every screen it named when that answer was picked; the screen
step of such a request (as the classifier judges it, in any language) is built
by the AI design without asking. Later GUI requests are asked again, and an
explicit choice always wins. A typed answer with no question open (the screens
were already built) is applied, not asked again.
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
from common.session_keys import UNIFIED_CLASSIFICATION
from routing import unified_classifier
from routing.unified_classifier import UnifiedClassification
from utilities.original_request import remember_original_request

import execution.planning as planning
from tests.conftest import MINIMAL_CLASS_MODEL, FakeSession

GUI_OP = {"diagramType": "GUINoCodeDiagram", "mode": "complete_system", "request": "Create the GUI for the shop"}


@pytest.fixture
def run_gui_op(monkeypatch):
    handler = MagicMock()
    handler.generate_complete_system.return_value = {
        "action": "inject_complete_system", "diagramType": "GUINoCodeDiagram",
        "model": {"pages": []}, "message": "Built",
    }
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

    def run(message, session=None, project_id=None, verdict=None, operation=None):
        session = session or FakeSession()
        session.set(UNIFIED_CLASSIFICATION, verdict)
        context = WorkspaceContext(project_snapshot={"id": project_id} if project_id else None)
        request = AssistantRequest(message=message, context=context)
        modeling.execute_model_operation(session, request, dict(operation or GUI_OP), "complete_system")
        asked = any("How would you like me to create your screens" in str(p) for p in payloads)
        run.payloads = payloads
        return session, handler, asked

    return run


LISTED = UnifiedClassification(intent="create_complete_system_intent", screens_listed=True, reason="r")
NOT_LISTED = UnifiedClassification(intent="create_complete_system_intent", reason="r")
LISTED_OP = {**GUI_OP, modeling.SCREENS_LISTED_OP_KEY: True}
REQUEST = "Bookshop app with a catalogue screen, a basket page and a checkout page"


def _plan(session, message, verdict, operations):
    """The operations execute_planned_operations hands to execute_model_operation."""
    session.set(UNIFIED_CLASSIFICATION, verdict)
    executed = []
    request = AssistantRequest(message=message, context=WorkspaceContext(project_snapshot={"id": "p1"}))
    with patch.object(planning, "plan_assistant_operations", return_value=[dict(op) for op in operations]), \
         patch.object(planning, "execute_model_operation",
                      side_effect=lambda _s, _r, op, **_kw: executed.append(op) or op["diagramType"]), \
         patch.object(planning, "_report_progress"), \
         patch.object(planning, "reply_payload"), \
         patch.object(planning, "reply_message"):
        planning.execute_planned_operations(session, request, "complete_system", "create_complete_system_intent")
    return executed


def test_a_plan_marks_its_screen_step_when_its_request_lists_the_screens():
    class_op = {"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system", "request": "r"}
    gui_op = {"type": "model", **GUI_OP}
    executed = _plan(FakeSession(), REQUEST, LISTED, [class_op, gui_op])
    assert [op.get(modeling.SCREENS_LISTED_OP_KEY) for op in executed] == [None, True]
    executed = _plan(FakeSession(), "Create the GUI", NOT_LISTED, [gui_op])
    assert executed[0].get(modeling.SCREENS_LISTED_OP_KEY) is None


def test_the_screen_step_of_a_request_listing_its_screens_is_designed_without_asking(run_gui_op):
    session, handler, asked = run_gui_op(REQUEST, verdict=LISTED, operation=LISTED_OP)
    assert not asked
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None
    handler.generate_complete_system.assert_called_once()


@pytest.mark.parametrize("message", [
    "Use basic CRUD pages for the GUI",
    "I'd prefer the basic CRUD pages",
    "Regenerate the GUI",
])
def test_a_later_gui_request_in_the_same_project_is_asked_again(run_gui_op, message):
    session = FakeSession()
    first = AssistantRequest(message=REQUEST, context=WorkspaceContext(project_snapshot={"id": "p1"}))
    session.set(UNIFIED_CLASSIFICATION, LISTED)
    unified_classifier._remember_stated_target_and_language(session, first, LISTED, "p1")
    remember_original_request(session, REQUEST, "p1")
    _, handler, asked = run_gui_op(message, session=session, project_id="p1", verdict=NOT_LISTED)
    assert asked
    handler.generate_complete_system.assert_not_called()


def test_a_failed_design_of_listed_screens_offers_a_retry_or_crud_pages(run_gui_op):
    # The same re-offer the question's own AI-design answer gets on failure.
    handler_failure = {"action": "assistant_message", "message": "The design failed."}
    session = FakeSession()
    modeling.ctx.diagram_factory.get_handler.return_value.generate_complete_system.return_value = handler_failure
    run_gui_op("Three screens: A, B, C", session=session, verdict=LISTED, operation=LISTED_OP)
    pending = pending_dialogs.get(session, PENDING_GUI_CHOICE)
    assert pending and pending.get(modeling.AI_DESIGN_FAILED)
    offer = run_gui_op.payloads[-1]
    assert offer["suggestedActions"] == modeling.gui_choice_actions()
    assert "try the AI design again" in offer["message"]


def test_a_screen_step_resumed_after_another_question_keeps_its_requests_listed_screens(run_gui_op):
    # The step runs after a replace/keep answer, which itself lists nothing.
    _, handler, asked = run_gui_op("Replace", verdict=NOT_LISTED, operation=LISTED_OP)
    assert not asked
    handler.generate_complete_system.assert_called_once()


def test_a_screen_step_parked_behind_replace_keep_keeps_its_requests_listed_screens(run_gui_op, monkeypatch):
    session = FakeSession()
    session.set(UNIFIED_CLASSIFICATION, LISTED)
    monkeypatch.setattr(modeling, "resolve_target_model", lambda *args: {"pages": [{"name": "Home"}]})
    monkeypatch.setattr(modeling, "model_has_elements", lambda model: bool(model))
    questions, payloads = [], []
    monkeypatch.setattr(modeling, "_build_existing_model_confirmation", lambda **kw: questions.append(kw))
    monkeypatch.setattr(modeling, "reply_payload", lambda _s, payload: payloads.append(payload))
    request = AssistantRequest(message=REQUEST, context=WorkspaceContext())
    modeling.execute_model_operation(session, request, dict(LISTED_OP), "complete_system")
    parked = questions[0]["pending_data"]["operation"]
    session.set(UNIFIED_CLASSIFICATION, NOT_LISTED)
    handler = modeling.ctx.diagram_factory.get_handler.return_value
    modeling.execute_model_operation(session, AssistantRequest(message="Replace", context=WorkspaceContext()),
                                     parked, "complete_system", _skip_existing_check=True)
    assert not any("How would you like me to create your screens" in str(p) for p in payloads)
    handler.generate_complete_system.assert_called_once()


def test_a_failed_design_of_listed_screens_waits_for_the_plans_open_dialog(run_gui_op):
    failure = {"action": "assistant_message", "message": "The design failed."}
    modeling.ctx.diagram_factory.get_handler.return_value.generate_complete_system.return_value = failure
    session = FakeSession()
    session.set(UNIFIED_CLASSIFICATION, LISTED)
    request = AssistantRequest(message="Three screens: A, B, C", context=WorkspaceContext())
    with pending_dialogs.one_per_plan():
        assert pending_dialogs.may_open("another dialog")
        queued = modeling.execute_model_operation(session, request, dict(LISTED_OP), "complete_system")
    assert isinstance(queued, modeling.Deferred)
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None
    # Asked later without designing again.
    handler = modeling.ctx.diagram_factory.get_handler.return_value
    handler.generate_complete_system.reset_mock()
    assert modeling.execute_model_operation(session, request, queued.operation, "complete_system") is None
    handler.generate_complete_system.assert_not_called()
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE).get(modeling.AI_DESIGN_FAILED)

@pytest.mark.parametrize("message", [
    "Build screens for my app",
    "Create one page per class",
    # Pages of a document, a 'pages' attribute and pagination are not screens.
    "A Book has a title, an author and 300 pages",
    "Book attributes - title, pages: int",
    "Each report is at most 10 pages long",
    "show me 3 pages of results",
    "Paginate the table, 5 pages max",
    "the website has 404 pages",
    "Pages: 120",
    "one landing page",
])
def test_without_the_classifier_finding_listed_screens_the_question_is_asked(run_gui_op, message):
    session, handler, asked = run_gui_op(message)
    assert asked
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is not None
    handler.generate_complete_system.assert_not_called()


def test_classifier_field_names_what_is_not_a_screen_and_guards_keep_it():
    description = UnifiedClassification.model_fields["screens_listed"].description
    for trap in ("document", "attribute", "pagination", "one page per class", "basic CRUD pages"):
        assert trap in description
    assert unified_classifier._rebuilt(LISTED, intent="modify_model_intent", reason="g").screens_listed


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


def test_a_typed_answer_wins_over_listed_screens(run_gui_op):
    session, handler, asked = run_gui_op("Basic CRUD pages", verdict=LISTED, operation=LISTED_OP)
    assert not asked
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE).get(modeling.GUI_CHOICE_IN_REQUEST)
    handler.generate_complete_system.assert_not_called()
