"""A full app description must survive the replace/keep question.

Live prod session 3eda187f, 2026-11-20 10:01-10:02 UTC (gpt-6 on the user's
own OpenAI key), project already holding a 1-class model:

1. Full "Recommendations Tracker" prompt -> replace/keep question, prompt stored.
2. The same prompt sent again -> classifier: new_request + needs_clarification.
   The confirmation was dropped and only the clarifying question went out, so
   the prompt was stored nowhere.
3. "replace" -> no open question, so "replace" became a create request.
4. "replace" -> the class diagram was built from the word "replace":
   RecommendationSystem with User, Item, Recommendation.
"""
from unittest.mock import patch

from app import confirmation
from app import state_bodies as sb
from common import pending_dialogs
from common.pending_dialogs import PENDING_COMPLETE_SYSTEM
from common.session_keys import UNIFIED_CLASSIFICATION
from execution import model_operations as mo
from protocol.types import AssistantRequest, WorkspaceContext
from routing.unified_classifier import UnifiedClassification
from utilities.original_request import original_request_for_project

from tests.conftest import FakeSession

PROMPT = (
    'Create a web app called "Recommendations Tracker" to keep all the '
    "recommendations in one place, organised by type. Class diagram: one class "
    "named Recommendation, no other classes or associations."
)


def _req(message):
    return AssistantRequest(
        message=message,
        context=WorkspaceContext(active_diagram_type="ClassDiagram", project_snapshot={"id": "p1"}),
    )


def _preamble(session, request, verdict):
    replies = []
    session.set(UNIFIED_CLASSIFICATION, verdict)
    with patch.object(sb, "parse_assistant_request", return_value=request), \
         patch.object(confirmation, "parse_assistant_request", return_value=request), \
         patch.object(sb, "remember_repo_recap"), \
         patch.object(sb, "handle_file_attachments", return_value=False), \
         patch.object(sb, "_conversation_memory"), \
         patch.object(sb, "get_or_classify", return_value=verdict), \
         patch.object(sb, "reply_message", side_effect=lambda _s, m: replies.append(m)):
        result = sb._common_preamble(session)
    return result, replies


def test_resent_prompt_at_replace_question_is_built_not_dropped():
    session = FakeSession()
    pending_dialogs.put(session, PENDING_COMPLETE_SYSTEM, {"operation_request": PROMPT})
    verdict = UnifiedClassification(
        intent="create_complete_system_intent", reason="stub",
        pending_flow_action="new_request",
        needs_clarification=True,
        clarifying_question="Should I replace your existing model or keep it?",
    )
    result, replies = _preamble(session, _req(PROMPT), verdict)
    # The request goes on to the create path, which asks replace/keep again
    # with THIS prompt attached, instead of a question that stores nothing.
    assert result is not None and result.message == PROMPT
    assert replies == []


def test_clarifying_question_still_asked_without_an_open_question():
    session = FakeSession()
    verdict = UnifiedClassification(
        intent="modify_model_intent", reason="stub",
        needs_clarification=True,
        clarifying_question="Which class should get the email attribute?",
    )
    result, replies = _preamble(session, _req("add an email attribute"), verdict)
    assert result is None
    assert replies == ["Which class should get the email attribute?"]


def test_bare_answer_is_never_built_as_a_system():
    session = FakeSession()
    replies = []
    with patch.object(mo, "reply_message", side_effect=lambda _s, m: replies.append(m)), \
         patch.object(mo, "resolve_target_model",
                      side_effect=AssertionError("must not reach the build")):
        result = mo.execute_model_operation(
            session=session,
            request=_req("replace"),
            operation={"diagramType": "ClassDiagram", "mode": "complete_system", "request": "replace"},
            default_mode="complete_system",
        )
    assert result is None
    assert len(replies) == 1 and "describe" in replies[0].lower()
    assert original_request_for_project(session, "p1") == ""
