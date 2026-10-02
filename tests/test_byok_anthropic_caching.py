"""Anthropic BYOK structured calls put the static prefix in a cached system block.

Before the fix ``LLMProvider.parse`` merged the system prompt and the user
block into one user message and appended the schema after the dynamic
content, with no ``cache_control``: the ~9k-token classifier prefix was billed
in full on every message.
"""

from types import SimpleNamespace

import anthropic
import openai
import pytest
from pydantic import BaseModel

import byok
from llm.provider import LLMProvider

SYSTEM = "You are the router. " * 50


class _Out(BaseModel):
    intent: str


class _FakeAnthropic:
    calls = []

    def __init__(self, **kwargs):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        _FakeAnthropic.calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=50, output_tokens=5,
                                cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"intent": "x"}')],
                               usage=usage, stop_reason="end_turn")


class _FakeOpenAI:
    calls = []

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _FakeOpenAI.calls.append(kwargs)
        message = SimpleNamespace(content='{"intent": "x"}', refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=None)


def _parse(provider_name, user_text):
    token = byok.set_current(provider_name, "test-key")
    try:
        return LLMProvider(SimpleNamespace(client=None)).parse(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}], _Out,
        )
    finally:
        byok.reset_current(token)


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    _FakeAnthropic.calls.clear()
    _FakeOpenAI.calls.clear()
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)


def test_static_prefix_is_a_cached_system_block_before_the_dynamic_content():
    _parse("anthropic", "first message")
    _parse("anthropic", "second message")

    first, second = _FakeAnthropic.calls
    block = first["system"][0]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert block["text"].startswith(SYSTEM)
    assert '"intent"' in block["text"]  # the schema is part of the static prefix
    assert second["system"] == first["system"]  # byte-identical prefix across messages
    user = first["messages"][0]["content"]
    assert user.startswith("first message")
    assert SYSTEM not in user


def test_openai_compatible_byok_prompt_is_unchanged():
    _parse("mistral", "first message")

    (call,) = _FakeOpenAI.calls
    assert [m["role"] for m in call["messages"]] == ["user"]
    content = call["messages"][0]["content"]
    assert content.startswith(SYSTEM + "\nfirst message\n\nReturn ONLY a JSON object")
