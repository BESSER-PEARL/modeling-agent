"""The "Basic CRUD pages or Experimental AI design" question for new screens.

Basic CRUD pages build one page per class, so a request that already lists
its screens lost every screen it named when that answer was picked; such a
request (as the classifier judges it, in any language) is built by the AI
design without asking. A typed answer with no
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
from common.session_keys import UNIFIED_CLASSIFICATION
from routing import unified_classifier
from routing.unified_classifier import UnifiedClassification

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

    def run(message, session=None, project_id=None, verdict=None):
        session = session or FakeSession()
        session.set(UNIFIED_CLASSIFICATION, verdict)
        context = WorkspaceContext(project_snapshot={"id": project_id} if project_id else None)
        request = AssistantRequest(message=message, context=context)
        modeling.execute_model_operation(session, request, dict(GUI_OP), "complete_system")
        asked = any("How would you like me to create your screens" in str(p) for p in payloads)
        return session, handler, asked

    return run


LISTED = UnifiedClassification(intent="create_complete_system_intent", screens_listed=True, reason="r")


def test_a_request_the_classifier_finds_listing_its_screens_is_designed_without_asking(run_gui_op):
    session, handler, asked = run_gui_op("Three screens: Catalogue, Basket and Checkout.", verdict=LISTED)
    assert not asked
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE) is None
    handler.generate_complete_system.assert_called_once()


def test_listed_screens_are_remembered_for_the_project_only(run_gui_op):
    session = FakeSession()
    request = AssistantRequest(message="x", context=WorkspaceContext(project_snapshot={"id": "p1"}))
    unified_classifier._remember_stated_target_and_language(session, request, LISTED, "p1")
    _, handler, asked = run_gui_op("Create the GUI", session=session, project_id="p1")
    assert not asked
    handler.generate_complete_system.assert_called_once()
    _, _, asked = run_gui_op("Create the GUI", session=session, project_id="p2")
    assert asked


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
    for trap in ("document", "attribute", "pagination", "one page per class"):
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
    session, handler, asked = run_gui_op("Basic CRUD pages", verdict=LISTED)
    assert not asked
    assert pending_dialogs.get(session, PENDING_GUI_CHOICE).get(modeling.GUI_CHOICE_IN_REQUEST)
    handler.generate_complete_system.assert_not_called()
