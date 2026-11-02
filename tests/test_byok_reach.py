"""A user's API key reaches every LLM call of their request.

Three paths used to fall back to the server key while a key was active: the
memory summarizer and the no-RAG UML answer called BAF's ``predict`` directly,
and the planner ran independent model groups on pool threads, which start with
an empty context and so without the request's BYOK contextvar.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import execution.planning as planning
from app import state_bodies
from conversation import agent_context as ctx
from ingestion import file_handling
from llm import byok
from tests.conftest import FakeSession


class _ServerLLM:
    """BAF's ``LLMOpenAI`` stand-in: records ``predict``, has no usable client."""

    name = "gpt-4o-mini"
    client = None

    def __init__(self):
        self.prompts = []

    def predict(self, prompt, parameters=None):
        self.prompts.append(prompt)
        return "server answer"


@pytest.fixture
def anthropic_key(monkeypatch):
    """An active Anthropic BYOK key whose SDK records the prompts it gets."""
    import anthropic

    prompts = []

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="user answer")],
                usage=None, stop_reason="end_turn",
            )

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    token = byok.set_current("anthropic", "sk-ant-user")
    yield prompts
    byok.reset_current(token)


def test_the_memory_summarizer_runs_on_the_users_key(monkeypatch, anthropic_key):
    captured = {}

    def _get_memory(session_id, summarizer=None):
        captured["summarizer"] = summarizer
        return MagicMock()

    server = _ServerLLM()
    monkeypatch.setattr(ctx, "gpt_text", server)
    request = SimpleNamespace(action="user_message", message="add a Book class")
    monkeypatch.setattr(state_bodies, "parse_assistant_request", lambda s: request)
    monkeypatch.setattr(state_bodies, "resume_pending", lambda s: False)
    monkeypatch.setattr(state_bodies, "handle_file_attachments", lambda s, r: False)
    monkeypatch.setattr(state_bodies, "memory_session_key", lambda s, r: "s-1")
    monkeypatch.setattr(state_bodies, "get_memory", _get_memory)
    monkeypatch.setattr(state_bodies, "get_or_classify", lambda *a: SimpleNamespace())

    assert state_bodies._common_preamble(FakeSession()) is request
    summary = captured["summarizer"]("summarize: user asked for a Book class")

    assert summary == "user answer"
    assert server.prompts == []
    assert "summarize: user asked for a Book class" in str(anthropic_key)


@pytest.mark.parametrize("rag_fails", [False, True])
def test_the_uml_answer_without_rag_runs_on_the_users_key(monkeypatch, anthropic_key, rag_fails):
    server = _ServerLLM()
    replies = []
    session = FakeSession()
    session.run_rag = MagicMock(side_effect=RuntimeError("index down"))
    monkeypatch.setattr(ctx, "gpt_text", server)
    monkeypatch.setattr(ctx, "uml_rag", object() if rag_fails else None)
    monkeypatch.setattr(
        state_bodies, "_common_preamble",
        lambda s: SimpleNamespace(message="what is an association class?"),
    )
    monkeypatch.setattr(state_bodies, "reply_message", lambda s, text: replies.append(text))

    state_bodies.uml_rag_body(session)

    assert server.prompts == []
    assert len(anthropic_key) == 1
    assert replies and replies[-1].endswith("user answer")


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


@pytest.mark.parametrize("provider, server_key, served", [
    ("nebius", None, False),
    ("nebius", "sk-server", True),
    ("anthropic", None, True),
])
def test_vision_is_offered_only_when_a_key_can_read_images(monkeypatch, provider, server_key, served):
    """A key whose provider has no image model leaves vision to the server
    key; with no server key either, conversion says so instead of failing."""
    monkeypatch.setattr(ctx, "openai_api_key", server_key)
    monkeypatch.setattr("llm.provider.get_provider", lambda: SimpleNamespace(vision="vision-call"))
    token = byok.set_current(provider, "user-key")
    try:
        vision = file_handling._vision_predict()
    finally:
        byok.reset_current(token)

    assert (vision == "vision-call") is served
