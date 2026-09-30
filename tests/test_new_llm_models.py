"""GPT-6 and the current Claude models get valid request params and real prices.

GPT-6 facts are from a live probe (2026-09-28, chat completions) of
gpt-6-astra / gpt-6-sol / gpt-6-luna:

* ``temperature`` other than 1 -> 400 at the default effort (like gpt-5*),
* ``max_tokens`` -> 400, only ``max_completion_tokens`` is accepted,
* ``reasoning_effort="low"`` + json_schema / json_object / streaming -> OK on
  all three; ``"minimal"`` -> 400 on all three, ``"none"`` -> 400 on astra.

Before this change the gpt-5-prefix rule treated gpt-6 as a gpt-4o-style
model, so every gpt-6 call sent a temperature (and the vision path
``max_tokens``) and got a 400. Unpriced models report placeholder costs.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import byok  # noqa: E402
import model_config  # noqa: E402
from model_config import reasoning_effort_for, supports_custom_temperature  # noqa: E402

GPT6 = ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"]
CLAUDE_NO_SAMPLING = [
    "claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5",
    "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7",
]

@pytest.mark.parametrize("model", GPT6)
def test_gpt6_gets_reasoning_effort_not_temperature(model):
    assert supports_custom_temperature(model) is False
    assert reasoning_effort_for(model) == model_config.MODEL_REASONING_EFFORT


@pytest.mark.parametrize("model", GPT6)
def test_gpt6_is_a_large_tier_request_for_byok(model):
    assert byok._tier_of(model) == "large"


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-oss-120b"])
def test_pre_gpt5_models_keep_their_temperature(model):
    assert supports_custom_temperature(model) is True
    assert reasoning_effort_for(model) is None


class _Completions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content='{"ok": true}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


@pytest.mark.parametrize("model", GPT6)
def test_an_openai_key_on_gpt6_sends_only_params_gpt6_accepts(model, monkeypatch):
    monkeypatch.setattr(model_config, "MODEL_REASONING_EFFORT", "low")
    client = byok.BYOKClient.__new__(byok.BYOKClient)
    client.provider, client._user_model, client._base_url = "openai", model, None
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    client.predict_raw("make a model", json_mode=True)
    sent = client._client.chat.completions.calls[0]
    assert "temperature" not in sent and "max_tokens" not in sent
    assert sent["max_completion_tokens"] > 0
    assert sent["reasoning_effort"] == "low"


@pytest.mark.parametrize("model", GPT6)
def test_gpt6_vision_uses_max_completion_tokens(model, monkeypatch):
    from handlers import file_conversion_handler

    monkeypatch.setattr(file_conversion_handler, "MODEL_VISION", model)
    params = file_conversion_handler._vision_sampling_params()
    assert "max_tokens" not in params and "temperature" not in params
    assert params["max_completion_tokens"] > 0


def test_gpt6_luna_streaming_uses_configured_low_effort(monkeypatch):
    import session_helpers

    monkeypatch.setattr(model_config, "MODEL_REASONING_EFFORT", "low")
    monkeypatch.setattr(session_helpers, "reply_stream_chunk", lambda *_: None)
    completions = _Completions()

    def create(**kwargs):
        completions.calls.append(kwargs)
        return [SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", delta=SimpleNamespace(
                content="OK", refusal=None))], usage=None,
        )]

    completions.create = create
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assert session_helpers._stream_openai(None, client, "describe", "", "stream", "gpt-6-luna") == "OK"
    sent = completions.calls[0]
    assert sent["reasoning_effort"] == "low"
    assert "temperature" not in sent


@pytest.mark.parametrize("model", CLAUDE_NO_SAMPLING + [
    "us.anthropic.claude-opus-5-5", "anthropic.claude-fable-5-1",
    # A later generation or point release needs no code change.
    "claude-haiku-5", "claude-sonnet-5-5",
])
def test_current_claude_models_get_no_sampling_params(model):
    assert supports_custom_temperature(model) is False
    assert reasoning_effort_for(model) is None


@pytest.mark.parametrize("model", [
    "claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5", "claude-haiku-4-5-20251001",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
])
def test_claude_models_that_take_sampling_keep_it(model):
    assert supports_custom_temperature(model) is True


class _Messages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"ok": true}')], usage=None)


@pytest.mark.parametrize("model", CLAUDE_NO_SAMPLING)
def test_an_anthropic_key_sends_nothing_the_current_claude_models_reject(model):
    """Fable 5.1 / Opus 5.5 400 on forced tool_choice, disabled thinking,
    budget_tokens and assistant prefill; all of these 400 on sampling params.
    The Anthropic path gets JSON from a prompt instruction, so it must send
    none of them."""
    client = byok.BYOKClient.__new__(byok.BYOKClient)
    client.provider, client._user_model, client._base_url = "anthropic", model, None
    client._client = SimpleNamespace(messages=_Messages())
    client.predict_raw("make a model", json_mode=True)
    sent = client._client.messages.calls[0]
    for rejected in ("temperature", "top_p", "top_k", "tool_choice", "tools", "thinking"):
        assert rejected not in sent
    assert [m["role"] for m in sent["messages"]] == ["user"]
