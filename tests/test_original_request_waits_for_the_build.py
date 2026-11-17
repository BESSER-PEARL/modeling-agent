"""A request is remembered as the app's original description only once its model is built.

Reproduced on production: "Design a dashboard for inventory management" with a model
already on the canvas asked replace/keep, and the user's mistyped answer "trplace"
matched no answer, so it was treated as a new request, built nothing (it asked
replace/keep again) and still became the remembered original request. "generate the
code" then started a Spec-Driven run whose whole request was "trplace", on the old
model. The request is now remembered when the build goes ahead: at once with nothing
to replace, or when the user answers the replace/keep question it raised.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import execution.model_operations as mo
from common import pending_dialogs
from common.pending_dialogs import PENDING_COMPLETE_SYSTEM
from protocol.types import AssistantRequest, WorkspaceContext
from tests.conftest import MINIMAL_CLASS_MODEL, FakeSession
from utilities.original_request import original_request_for_project, remember_original_request

_PROJECT = "proj-inventory"
_LIBRARY = "Create a library app with books, authors and loans."
_DASHBOARD = "Design a dashboard for inventory management"


def _execute(monkeypatch, session, message, *, canvas=MINIMAL_CLASS_MODEL, operation=None, resume=False):
    handler = MagicMock()
    handler.generate_complete_system.return_value = {
        "action": "inject_complete_system", "message": "I created the model.",
        "systemSpec": {"classes": [{"className": "Item"}]},
    }
    factory = MagicMock()
    factory.get_handler.return_value = handler
    monkeypatch.setattr(mo.ctx, "diagram_factory", factory)
    for name in ("reply_progress", "reply_message", "reply_payload"):
        monkeypatch.setattr(mo, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(mo, "resolve_target_model", lambda *args, **kwargs: copy.deepcopy(canvas) if canvas else None)
    monkeypatch.setattr(mo, "build_workspace_context_block", lambda *args, **kwargs: "")
    request = AssistantRequest(message=message, context=WorkspaceContext(
        active_diagram_type="ClassDiagram", project_snapshot={"id": _PROJECT}))
    operation = operation or {"diagramType": "ClassDiagram", "mode": "complete_system", "request": message}
    mo.execute_model_operation(session, request, operation, "complete_system",
                               _skip_existing_check=resume, _replace_existing=resume or None)
    return handler


def _abandon_question(session):
    """What app.confirmation does with a reply that matches no answer: drop the question."""
    pending_dialogs.put(session, PENDING_COMPLETE_SYSTEM, None)


def test_a_reply_that_built_nothing_does_not_replace_the_remembered_request(monkeypatch):
    session = FakeSession()
    remember_original_request(session, _LIBRARY, _PROJECT)

    _execute(monkeypatch, session, _DASHBOARD)
    assert pending_dialogs.get(session, PENDING_COMPLETE_SYSTEM)  # replace/keep was asked
    _abandon_question(session)
    _execute(monkeypatch, session, "trplace")
    assert pending_dialogs.get(session, PENDING_COMPLETE_SYSTEM)  # asked again, nothing built

    assert original_request_for_project(session, _PROJECT) == _LIBRARY


def test_answering_the_question_remembers_the_request_that_raised_it(monkeypatch):
    session = FakeSession()
    remember_original_request(session, _LIBRARY, _PROJECT)

    _execute(monkeypatch, session, _DASHBOARD)
    stored = pending_dialogs.get(session, PENDING_COMPLETE_SYSTEM)
    pending_dialogs.put(session, PENDING_COMPLETE_SYSTEM, None)
    # app.confirmation resumes the stored operation on "replace", with the stored step request.
    handler = _execute(monkeypatch, session, stored["message"], operation=stored["operation"], resume=True)

    handler.generate_complete_system.assert_called_once()
    assert original_request_for_project(session, _PROJECT) == _DASHBOARD


def test_with_nothing_to_replace_the_request_is_remembered_at_once(monkeypatch):
    session = FakeSession()
    _execute(monkeypatch, session, _DASHBOARD, canvas=None)
    assert original_request_for_project(session, _PROJECT) == _DASHBOARD
