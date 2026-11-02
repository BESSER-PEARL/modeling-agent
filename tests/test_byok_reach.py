"""A user's API key reaches every LLM call of their request.

The planner ran independent model groups on pool threads, which start with an
empty context and so without the request's BYOK contextvar.
"""

import threading
from unittest.mock import MagicMock, patch

import execution.planning as planning
from llm import byok
from tests.conftest import FakeSession


def test_parallel_plan_groups_run_on_the_users_key():
    """Two independent model groups run on pool threads; each still sees the key."""
    seen = []

    def _model_op(session, request, op, default_mode):
        seen.append((op["diagramType"], threading.current_thread().name, byok.is_active()))
        return op["diagramType"]

    plan = [
        {"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system", "request": "a library"},
        {"type": "model", "diagramType": "StateMachineDiagram", "mode": "complete_system", "request": "a loan"},
    ]
    token = byok.set_current("openai", "sk-user-not-real")
    try:
        with patch.object(planning, "plan_assistant_operations", return_value=plan), \
             patch.object(planning, "execute_model_operation", _model_op), \
             patch.object(planning, "reply_payload"), \
             patch.object(planning, "reply_message"), \
             patch.object(planning, "_report_progress"), \
             patch.object(planning, "build_request_for_target", side_effect=lambda r, t: r):
            planning.execute_planned_operations(
                FakeSession(), MagicMock(), "complete_system", "create_complete_system_intent",
            )
    finally:
        byok.reset_current(token)

    assert sorted(op for op, _, _ in seen) == ["ClassDiagram", "StateMachineDiagram"]
    assert all(thread != threading.current_thread().name for _, thread, _ in seen)
    assert all(active for _, _, active in seen)
    assert not byok.is_active()
