"""Claude models that reject sampling params never receive them.

Sonnet 5 (and Opus 4.7+, Fable, Mythos) answer any temperature/top_p/top_k
with a 400. The Anthropic key path sent temperature on every call, so a user
with their own claude-sonnet-5 key got no modeling at all; PIA runs went
through the OpenAI-compatible path and never showed it.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import byok  # noqa: E402
import model_config  # noqa: E402
from model_config import anthropic_effort, reasoning_effort_for, supports_custom_temperature  # noqa: E402


@pytest.mark.parametrize("model", [
    "claude-sonnet-5", "us.anthropic.claude-sonnet-5", "anthropic/claude-sonnet-5",
    "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-opus-5-5",
    "claude-fable-5-1", "claude-mythos-5-1",
])
def test_claude_models_without_sampling_get_no_temperature_and_no_reasoning_effort(model):
    assert supports_custom_temperature(model) is False
    # reasoning_effort through a gateway can become budget_tokens, another 400.
    assert reasoning_effort_for(model) is None


@pytest.mark.parametrize("model", [
    "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6", "gpt-4o",
    "gpt-4.1-mini", "Qwen/Qwen3-30B-A3B-Instruct-2507", "mistral-large-latest",
])
def test_models_that_take_a_temperature_still_get_one(model):
    assert supports_custom_temperature(model) is True
    assert anthropic_effort(model) is None


def test_openai_reasoning_models_are_unchanged():
    assert supports_custom_temperature("gpt-5.6-terra") is False
    assert reasoning_effort_for("gpt-5.6-terra") == model_config.MODEL_REASONING_EFFORT


def test_claude_effort_follows_the_configured_level_and_skips_non_claude_levels(monkeypatch):
    monkeypatch.setattr(model_config, "MODEL_REASONING_EFFORT", "low")
    assert anthropic_effort("claude-sonnet-5") == "low"
    for not_a_claude_level in ("none", "minimal", ""):
        monkeypatch.setattr(model_config, "MODEL_REASONING_EFFORT", not_a_claude_level)
        assert anthropic_effort("claude-sonnet-5") is None


class _Messages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"ok": true}')], usage=None)


def _anthropic_client(model):
    client = byok.BYOKClient.__new__(byok.BYOKClient)
    client.provider, client._user_model, client._base_url = "anthropic", model, None
    client._client = SimpleNamespace(messages=_Messages())
    return client


def test_a_sonnet_5_key_sends_effort_instead_of_temperature(monkeypatch):
    monkeypatch.setattr(model_config, "MODEL_REASONING_EFFORT", "low")
    client = _anthropic_client("claude-sonnet-5")
    client.predict_raw("make a model", json_mode=True)
    sent = client._client.messages.calls[0]
    assert sent["model"] == "claude-sonnet-5"
    assert "temperature" not in sent and "top_p" not in sent and "top_k" not in sent
    assert sent["extra_body"] == {"output_config": {"effort": "low"}}


def test_an_older_claude_key_keeps_its_temperature():
    client = _anthropic_client("claude-haiku-4-5-20251001")
    client.predict_raw("make a model", json_mode=True)
    sent = client._client.messages.calls[0]
    assert 0.0 <= sent["temperature"] <= 1.0
    assert "extra_body" not in sent


def test_the_classifier_sizes_its_budget_for_the_users_model(monkeypatch):
    """A user key sends the classifier call to the user's model; an 800-token
    budget sized for the server's gpt-4o would starve Sonnet 5's thinking."""
    import unified_classifier
    from protocol.types import AssistantRequest, WorkspaceContext

    seen = {}

    class _Provider:
        model_name = "gpt-4o-mini"

        def parse(self, messages, schema, temperature, max_tokens):
            seen["max_tokens"] = max_tokens
            return None

    token = byok.set_current("anthropic", "sk-ant-user", "claude-sonnet-5")
    request = AssistantRequest(message="hello", context=WorkspaceContext(active_diagram_type="ClassDiagram"))
    try:
        unified_classifier.classify_message(request, _Provider())
    finally:
        byok.reset_current(token)
    assert seen.get("max_tokens") == 4000
