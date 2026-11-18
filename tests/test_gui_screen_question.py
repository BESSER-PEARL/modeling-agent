"""The "Basic CRUD pages or Experimental AI design" question for new screens.

Basic CRUD pages build one page per class, so a request that already lists
its screens lost every screen it named when that answer was picked; such a
request is built by the AI design without asking.
"""

from unittest.mock import MagicMock

import pytest

import execution.model_operations as modeling
from common import pending_dialogs
from common.pending_dialogs import PENDING_GUI_CHOICE
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

