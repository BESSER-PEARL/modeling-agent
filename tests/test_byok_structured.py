"""Structured generation must run on the user's own key when they saved one.

``predict_structured`` (most element and modification paths of every diagram
type) used the server client unconditionally, so a user with their own key
still spent the server's quota, and modeling failed when that quota ran out.
"""
import os
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import byok  # noqa: E402
from diagram_handlers.types.agent_diagram_handler import AgentDiagramHandler  # noqa: E402


class _Answer(BaseModel):
    name: str


def _completion(parsed):
    message = SimpleNamespace(parsed=parsed, content=parsed.model_dump_json(), refusal=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None,
    )


class _ParseClient:
    """An OpenAI-shaped client that records ``beta.chat.completions.parse`` calls."""

    def __init__(self, answer="from-parse"):
        self.calls = []
        answer_model = _Answer(name=answer)

        def parse(**kwargs):
            self.calls.append(kwargs)
            return _completion(answer_model)

        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))


class _ServerLLM:
    name = "gpt-5.5"

    def __init__(self):
        self.client = _ParseClient(answer="server")
        self.predict_calls = 0

    def predict(self, prompt):
        self.predict_calls += 1
        return '{"name": "server"}'


@pytest.fixture
def server():
    return _ServerLLM()


@pytest.fixture
def handler(server):
    return AgentDiagramHandler(server)


def _with_key(monkeypatch, provider, client):
    token = byok.set_current(provider, "sk-user", None)
    monkeypatch.setattr(byok, "get_active_client", lambda: client)
    return token


def test_without_a_user_key_the_server_client_is_used(handler, server):
    result = handler.predict_structured("make it", _Answer, max_retries=0)

    assert result.name == "server"
    assert len(server.client.calls) == 1


def test_an_openai_key_runs_the_structured_call_on_the_users_client(monkeypatch, handler, server):
    user_client = _ParseClient(answer="user")
    stub = SimpleNamespace(provider="openai", openai_client=user_client)
    token = _with_key(monkeypatch, "openai", stub)
    try:
        result = handler.predict_structured("make it", _Answer, max_retries=0, model="gpt-5.5")
    finally:
        byok.reset_current(token)

    assert result.name == "user"
    assert server.client.calls == [], "the server key must not be used"
    assert len(user_client.calls) == 1


@pytest.mark.parametrize("provider", ["anthropic", "mistral"])
def test_other_providers_use_the_users_key_through_json_mode(monkeypatch, handler, server, provider):
    seen = []

    def predict_raw(prompt, **kwargs):
        seen.append(prompt)
        return '{"name": "user"}'

    stub = SimpleNamespace(provider=provider, openai_client=None, predict_raw=predict_raw)
    token = _with_key(monkeypatch, provider, stub)
    try:
        result = handler.predict_structured("make it", _Answer, max_retries=0)
    finally:
        byok.reset_current(token)

    assert result.name == "user"
    assert server.client.calls == [] and server.predict_calls == 0
    assert seen, "the call must go through the user's client"


# ---------------------------------------------------------------------------
# The intent classifier goes through LLMProvider.parse
# ---------------------------------------------------------------------------

def _provider(server):
    from llm.provider import LLMProvider
    return LLMProvider(server, model_name="gpt-4o-mini")


def test_classifier_parse_uses_the_users_openai_client(monkeypatch, server):
    user_client = _ParseClient(answer="user")
    token = _with_key(monkeypatch, "openai", SimpleNamespace(provider="openai", openai_client=user_client))
    try:
        result = _provider(server).parse([{"role": "user", "content": "hi"}], schema=_Answer)
    finally:
        byok.reset_current(token)

    assert result.name == "user"
    assert server.client.calls == []
    assert len(user_client.calls) == 1


def test_classifier_parse_uses_json_mode_for_other_providers(monkeypatch, server):
    seen = {}

    def predict_raw(prompt, **kwargs):
        seen.update(kwargs)
        return '{"name": "user"}'

    stub = SimpleNamespace(provider="anthropic", openai_client=None, predict_raw=predict_raw)
    token = _with_key(monkeypatch, "anthropic", stub)
    try:
        result = _provider(server).parse([{"role": "user", "content": "hi"}], schema=_Answer)
    finally:
        byok.reset_current(token)

    assert result.name == "user"
    assert server.client.calls == [] and server.predict_calls == 0
    assert seen.get("json_mode") is True


# ---------------------------------------------------------------------------
# File / image attachments -> diagram
# ---------------------------------------------------------------------------

def test_json_prediction_runs_on_the_users_key(monkeypatch):
    seen = {}

    def predict_raw(prompt, **kwargs):
        seen.update(kwargs)
        return '{"ok": true}'

    token = _with_key(monkeypatch, "mistral", SimpleNamespace(provider="mistral", predict_raw=predict_raw))
    try:
        assert byok.predict_json("convert this", model="gpt-4o") == '{"ok": true}'
    finally:
        byok.reset_current(token)
    assert seen == {"model": "gpt-4o", "json_mode": True}


def test_json_prediction_without_a_user_key_defers_to_the_server():
    assert byok.predict_json("convert this") is None


@pytest.mark.parametrize(("provider", "base_url", "expected"), [
    ("openai", None, "sk-user"),        # official OpenAI: the user's key pays for vision
    ("anthropic", None, None),          # cannot call OpenAI vision -> server key
    ("mistral", None, None),
])
def test_vision_uses_the_users_openai_key_when_it_can(provider, base_url, expected):
    token = byok.set_current(provider, "sk-user", None, base_url)
    try:
        assert byok.user_openai_key() == expected
    finally:
        byok.reset_current(token)


def test_vision_without_a_user_key_uses_the_server_key():
    assert byok.user_openai_key() is None


# ---------------------------------------------------------------------------
# Nebius keys (OpenAI-compatible Token Factory endpoint)
# ---------------------------------------------------------------------------

def test_a_nebius_key_is_accepted_and_calls_the_nebius_endpoint(monkeypatch):
    import openai

    created = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)
            completions = SimpleNamespace(create=self._create)
            self.chat = SimpleNamespace(completions=completions)

        def _create(self, **kwargs):
            created["call"] = kwargs
            message = SimpleNamespace(content='{"ok": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    token = byok.set_current("nebius", "neb-key", None)
    try:
        assert byok.get_current() is not None, "nebius must not fall back to the server key"
        client = byok.get_active_client()
        assert client.openai_client is None  # structured calls use JSON mode
        assert client.predict_raw("hi", json_mode=True) == '{"ok": true}'
    finally:
        byok.reset_current(token)

    assert created["base_url"] == byok.NEBIUS_BASE_URL
    assert created["api_key"] == "neb-key"
    assert created["call"]["model"].startswith("Qwen/")
    assert "max_tokens" in created["call"] and "max_completion_tokens" not in created["call"]


def test_the_per_call_timeout_outlasts_a_long_reasoning_pass():
    """Live (2026-09-23): the hotel spec's reasoning pass on Nebius Qwen was cut
    at exactly 120 s and redone, costing ~2 minutes; the retry took ~110 s."""
    assert byok._SDK_TIMEOUT_SECONDS >= 300
