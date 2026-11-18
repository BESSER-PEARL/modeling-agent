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
from utilities.original_request import remember_original_request

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
        run.payloads = payloads
        return session, handler, asked

    return run


LISTED = UnifiedClassification(intent="create_complete_system_intent", screens_listed=True, reason="r")
NOT_LISTED = UnifiedClassification(intent="create_complete_system_intent", reason="r")
LIBRARY = "Library app with three screens: Books, Members, Loans"


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



def test_a_failed_design_of_listed_screens_offers_a_retry_or_crud_pages(run_gui_op):
    # The same re-offer the question's own AI-design answer gets on failure.
    handler_failure = {"action": "assistant_message", "message": "The design failed."}
    session = FakeSession()
    modeling.ctx.diagram_factory.get_handler.return_value.generate_complete_system.return_value = handler_failure
    run_gui_op("Three screens: A, B, C", session=session, verdict=LISTED)
    pending = pending_dialogs.get(session, PENDING_GUI_CHOICE)
    assert pending and pending.get(modeling.AI_DESIGN_FAILED)
    offer = run_gui_op.payloads[-1]
    assert offer["suggestedActions"] == modeling.gui_choice_actions()
    assert "try the AI design again" in offer["message"]


def _describe(session, description, verdict, project_id="p1"):
    """One turn that describes the app: the classifier's verdict, then the stored description."""
    request = AssistantRequest(message=description, context=WorkspaceContext(project_snapshot={"id": project_id}))
    session.set(UNIFIED_CLASSIFICATION, verdict)
    unified_classifier._remember_stated_target_and_language(session, request, verdict, project_id)
    remember_original_request(session, description, project_id)


def test_a_new_app_in_the_same_project_drops_the_screens_the_old_one_listed(run_gui_op):
    session = FakeSession()
    _describe(session, LIBRARY, LISTED)
    _describe(session, "Create a hotel booking system", NOT_LISTED)
    _, handler, asked = run_gui_op("Create the GUI", session=session, project_id="p1")
    assert asked
    handler.generate_complete_system.assert_not_called()


def test_a_follow_up_added_to_the_description_keeps_its_listed_screens(run_gui_op):
    session = FakeSession()
    _describe(session, LIBRARY, LISTED)
    _describe(session, LIBRARY + "\n\n## Subsequent user request\n\nadd fines", NOT_LISTED)
    _, _, asked = run_gui_op("Create the GUI", session=session, project_id="p1")
    assert not asked


def test_a_build_resumed_after_replace_keep_keeps_the_screens_its_request_listed(run_gui_op, monkeypatch):
    session = FakeSession()
    _describe(session, "Create a hotel booking system", NOT_LISTED)
    # The new app's request lists screens, but the project already has a model.
    context = WorkspaceContext(project_snapshot={"id": "p1"})
    session.set(UNIFIED_CLASSIFICATION, LISTED)
    unified_classifier._remember_stated_target_and_language(
        session, AssistantRequest(message=LIBRARY, context=context), LISTED, "p1")
    monkeypatch.setattr(modeling, "resolve_target_model", lambda *args: MINIMAL_CLASS_MODEL)
    questions = []
    monkeypatch.setattr(modeling, "_build_existing_model_confirmation", lambda **kw: questions.append(kw))
    class_op = {"diagramType": "ClassDiagram", "mode": "complete_system", "request": LIBRARY}
    modeling.execute_model_operation(session, AssistantRequest(message=LIBRARY, context=context),
                                     class_op, "complete_system")
    parked = questions[0]["pending_data"]["operation"]
    # "Replace": the answer lists nothing, the build it resumes did.
    session.set(UNIFIED_CLASSIFICATION, NOT_LISTED)
    modeling.execute_model_operation(session, AssistantRequest(message="Replace", context=context),
                                     parked, "complete_system", _skip_existing_check=True)
    monkeypatch.setattr(modeling, "resolve_target_model", lambda *args: None)  # no GUI yet
    handler = modeling.ctx.diagram_factory.get_handler.return_value
    handler.generate_complete_system.reset_mock()
    _, _, asked = run_gui_op("Create the GUI", session=session, project_id="p1")
    assert not asked
    handler.generate_complete_system.assert_called_once()


def test_a_failed_design_of_listed_screens_waits_for_the_plans_open_dialog(run_gui_op):
    failure = {"action": "assistant_message", "message": "The design failed."}
    modeling.ctx.diagram_factory.get_handler.return_value.generate_complete_system.return_value = failure
    session = FakeSession()
    session.set(UNIFIED_CLASSIFICATION, LISTED)
    request = AssistantRequest(message="Three screens: A, B, C", context=WorkspaceContext())
    with pending_dialogs.one_per_plan():
        assert pending_dialogs.may_open("another dialog")
        queued = modeling.execute_model_operation(session, request, dict(GUI_OP), "complete_system")
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
