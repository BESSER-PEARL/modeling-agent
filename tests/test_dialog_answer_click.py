"""A click on a quick action that answers the open question by its exact words
is decided by that question, without a classifier call.

Live, each such click (the run confirmation's "continue with the generation",
a screen-choice label) paid a full classifier call of about 11k prompt tokens
and 3.5-4.3 s before the reply, and the dialog's exact reading then overrode the
verdict. A click the dialog cannot decide, and anything typed, is still classified.
"""

import uuid

import pytest

from app.confirmation import dialog_answer_verdict
from common import pending_dialogs
from common.session_keys import LAST_SUGGESTED_PROMPTS
from routing.unified_classifier import UnifiedClassification
from tests.conftest import FakeSession


def _session(message):
    return FakeSession({"action": "user_message", "protocolVersion": "2.0", "message": message,
                        "context": {"activeDiagramType": "ClassDiagram"}, "sessionId": uuid.uuid4().hex})



@pytest.mark.parametrize("dialog_slot, prompt, answer", [
    (pending_dialogs.PENDING_SMART_GEN_INSTRUCTIONS, "continue with the generation", "confirm"),
    (pending_dialogs.PENDING_GUI_CHOICE, "Basic CRUD pages", "auto"),
])
def test_a_click_that_answers_the_open_question_is_decided_by_it(dialog_slot, prompt, answer):
    session = _session(prompt)
    pending_dialogs.put(session, dialog_slot, {"operation_request": "x"} if "gui" in dialog_slot else "x")
    session.set(LAST_SUGGESTED_PROMPTS, [prompt])

    verdict = dialog_answer_verdict(session, prompt)

    assert (verdict["pending_flow_action"], verdict["pending_flow_answer"]) == ("answer", answer)
    assert UnifiedClassification(**verdict)


@pytest.mark.parametrize("message, suggested", [
    ("yes", []),                                                  # typed, not a click
    ("create a class diagram for a recipe app",                   # a click the dialog cannot decide
     ["create a class diagram for a recipe app", "generate anyway with my current model"]),
])
def test_anything_else_is_classified(message, suggested):
    session = _session(message)
    pending_dialogs.put(session, pending_dialogs.PENDING_SMART_GEN_INSTRUCTIONS, "x")
    session.set(LAST_SUGGESTED_PROMPTS, suggested)

    assert dialog_answer_verdict(session, message) is None


def test_without_an_open_question_a_click_is_classified():
    session = _session("continue with the generation")
    session.set(LAST_SUGGESTED_PROMPTS, ["continue with the generation"])

    assert dialog_answer_verdict(session, "continue with the generation") is None
