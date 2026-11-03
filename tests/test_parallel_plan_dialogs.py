"""A parallel plan opens one dialog at a time.

Model groups of different diagram types run in threads, and each op may stop on
a question (replace/keep, the GUI mode). Two questions at once overwrote one
slot (two replace/keep) or buried one (an answer to the replace/keep question
cleared the GUI choice), so one diagram was never built. Driven through the
real plan, model-operation and confirmation code with a fake handler.
"""
import itertools
import json
from unittest.mock import patch

import pytest

from app import confirmation
from common import pending_dialogs
from common.pending_dialogs import PENDING_COMPLETE_SYSTEM, PENDING_GUI_CHOICE
from common.session_keys import UNIFIED_CLASSIFICATION, UNIFIED_CLASSIFICATION_EVENT_ID
from conversation import agent_context as ctx
from execution import planning
from protocol.adapters import parse_assistant_request
from routing.unified_classifier import UnifiedClassification
from tests.conftest import FakeSession, _FakeEvent, make_v2_payload


def _model(element_type, *names):
    return {
        "elements": {f"e{i}": {"id": f"e{i}", "type": element_type, "name": n, "owner": None}
                     for i, n in enumerate(names)},
        "relationships": {},
    }


SNAPSHOT = {"id": "p1", "name": "Library", "diagrams": {
    "ClassDiagram": [{"model": _model("Class", "Book", "Member")}],
    "StateMachineDiagram": [{"model": _model("State", "Open", "Closed")}],
    "AgentDiagram": [{"model": _model("AgentState", "Greet")}],
}}


class _Handler:
    def __init__(self, dtype):
        self.dtype = dtype

    def generate_complete_system(self, prompt, **kw):
        return {"action": "inject_complete_system", "diagramType": self.dtype,
                "systemSpec": {}, "message": f"Built {self.dtype}."}


class _Factory:
    def get_handler(self, dtype):
        return _Handler(dtype)


_ids = itertools.count()


def _say(session, message, answer=None):
    session.event = _FakeEvent(make_v2_payload(message, "ClassDiagram", project_snapshot=SNAPSHOT))
    session.event.id = f"ev{next(_ids)}"
    if answer:
        session.set(UNIFIED_CLASSIFICATION, UnifiedClassification(
            intent="decline_intent", reason="test",
            pending_flow_action="answer", pending_flow_answer=answer))
        session.set(UNIFIED_CLASSIFICATION_EVENT_ID, session.event.id)
    before = len(session._replies)
    return before


def _frames(session, since):
    return [json.loads(r) for r in session._replies[since:]]


def _questions(frames):
    return [f for f in frames
            if any(a.get("prompt") in ("replace", "Basic CRUD pages")
                   for a in f.get("suggestedActions") or [])]


def _built(frames):
    return [f["diagramType"] for f in frames
            if f.get("action") in ("inject_complete_system", "auto_generate_gui")]


@pytest.fixture
def run_plan(monkeypatch):
    monkeypatch.setattr(ctx, "diagram_factory", _Factory())

    def _run(plan):
        session = FakeSession()
        since = _say(session, "build it all")
        with patch.object(planning, "plan_assistant_operations", return_value=plan):
            planning.execute_planned_operations(
                session, parse_assistant_request(session),
                "complete_system", "create_complete_system_intent",
            )
        return session, _frames(session, since)

    return _run


def _answer(session, message, answer):
    since = _say(session, message, answer)
    assert confirmation.resume_pending(session)
    return _frames(session, since)


def _op(dtype):
    return {"type": "model", "diagramType": dtype, "mode": "complete_system", "request": f"a {dtype}"}


def test_two_replace_keep_questions_are_asked_one_after_the_other(run_plan):
    session, frames = run_plan([_op("StateMachineDiagram"), _op("AgentDiagram")])
    assert len(_questions(frames)) == 1

    built = []
    for _ in range(2):
        frames = _answer(session, "replace", "replace")
        built += _built(frames)
    assert sorted(built) == ["AgentDiagram", "StateMachineDiagram"]
    assert pending_dialogs.current(session) is None


def test_a_gui_choice_and_a_replace_question_are_asked_one_after_the_other(run_plan):
    session, frames = run_plan([_op("StateMachineDiagram"), _op("GUINoCodeDiagram")])
    asked = _questions(frames)
    assert len(asked) == 1

    built = []
    for _ in range(2):
        first = pending_dialogs.current(session).kind
        if first == pending_dialogs.GUI_CHOICE:
            built += _built(_answer(session, "Basic CRUD pages", "auto"))
        else:
            built += _built(_answer(session, "replace", "replace"))
    assert sorted(built) == ["GUINoCodeDiagram", "StateMachineDiagram"]
    assert pending_dialogs.current(session) is None


def test_a_parallel_plan_stores_its_leftover_operations_on_one_dialog(run_plan):
    """A generation op never rides along: the web-app and mixed-plan pauses strip
    every runnable generation op from a plan that has a model op, so nothing
    is left to run twice."""
    plan = [_op("StateMachineDiagram"), _op("GUINoCodeDiagram"),
            {"type": "generation", "generatorType": "web_app", "config": {}}]
    session, _ = run_plan(plan)

    carriers = [slot for slot in (PENDING_COMPLETE_SYSTEM, PENDING_GUI_CHOICE)
                if (pending_dialogs.get(session, slot) or {}).get("remaining_operations")]
    assert len(carriers) <= 1
    for slot in carriers:
        leftovers = pending_dialogs.get(session, slot)["remaining_operations"]
        assert all(op["type"] == "model" for op in leftovers)
