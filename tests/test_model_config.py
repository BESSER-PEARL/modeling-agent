"""Tests for the env-driven model routing table (``model_config``).

Constants are read from ``BESSER_AGENT_MODEL_*`` env vars at import time,
so the tests reload the module under controlled environments.
"""

import importlib

import pytest

import model_config

_ALL_ENV_VARS = [
    "BESSER_AGENT_MODEL_CLASSIFIER",
    "BESSER_AGENT_MODEL_GENERATION_LARGE",
    "BESSER_AGENT_MODEL_GENERATION_SMALL",
    "BESSER_AGENT_MODEL_REASONING",
    "BESSER_AGENT_MODEL_VISION",
    "BESSER_AGENT_MODEL_EMBEDDINGS",
]


@pytest.fixture(autouse=True)
def _restore_module():
    """Re-import with the real environment after each test so other tests
    (and constant importers) see the genuine values again."""
    yield
    importlib.reload(model_config)


def test_defaults_without_env(monkeypatch):
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    mod = importlib.reload(model_config)
    assert mod.MODEL_CLASSIFIER == "gpt-4o-mini"
    assert mod.MODEL_GENERATION_LARGE == "gpt-5.5"
    assert mod.MODEL_GENERATION_SMALL == "gpt-4o"
    assert mod.MODEL_REASONING == "gpt-5"
    assert mod.MODEL_VISION == "gpt-5"
    assert mod.MODEL_EMBEDDINGS == "text-embedding-3-small"


def test_env_overrides_take_effect(monkeypatch):
    monkeypatch.setenv("BESSER_AGENT_MODEL_CLASSIFIER", "my-router")
    monkeypatch.setenv("BESSER_AGENT_MODEL_GENERATION_LARGE", "my-frontier")
    monkeypatch.setenv("BESSER_AGENT_MODEL_GENERATION_SMALL", "my-small")
    monkeypatch.setenv("BESSER_AGENT_MODEL_REASONING", "my-reasoner")
    monkeypatch.setenv("BESSER_AGENT_MODEL_VISION", "my-vision")
    monkeypatch.setenv("BESSER_AGENT_MODEL_EMBEDDINGS", "my-embeddings")
    mod = importlib.reload(model_config)
    assert mod.MODEL_CLASSIFIER == "my-router"
    assert mod.MODEL_GENERATION_LARGE == "my-frontier"
    assert mod.MODEL_GENERATION_SMALL == "my-small"
    assert mod.MODEL_REASONING == "my-reasoner"
    assert mod.MODEL_VISION == "my-vision"
    assert mod.MODEL_EMBEDDINGS == "my-embeddings"


def test_blank_env_value_falls_back_to_default(monkeypatch):
    # An empty/whitespace env var must not silently route to model "".
    monkeypatch.setenv("BESSER_AGENT_MODEL_CLASSIFIER", "   ")
    mod = importlib.reload(model_config)
    assert mod.MODEL_CLASSIFIER == "gpt-4o-mini"


def test_routed_models_have_cost_table_entries():
    """Every routed model must have a price entry â€” unknown models fall
    back to ``_DEFAULT_COST`` and silently skew cost reporting."""
    from tracking.token_tracker import _COST_PER_1K

    for model in {
        model_config.MODEL_CLASSIFIER,
        model_config.MODEL_GENERATION_LARGE,
        model_config.MODEL_GENERATION_SMALL,
        model_config.MODEL_REASONING,
        model_config.MODEL_VISION,
    }:
        assert model in _COST_PER_1K, f"no cost entry for routed model {model}"


def test_supports_custom_temperature_families():
    """gpt-5* / o-series reject explicit temperature; older tiers accept it."""
    from model_config import supports_custom_temperature

    assert supports_custom_temperature("gpt-4o")
    assert supports_custom_temperature("gpt-4o-mini")
    assert supports_custom_temperature("gpt-4.1-mini")
    assert not supports_custom_temperature("gpt-5")
    assert not supports_custom_temperature("gpt-5.5")
    assert not supports_custom_temperature("o3-mini")
    # Unknown/empty model ids keep the old behavior (temperature sent) —
    # call paths resolve the effective model before consulting the guard.
    assert supports_custom_temperature("")


def test_reasoning_effort_only_for_reasoning_models(monkeypatch):
    """reasoning_effort is sent for gpt-5*/o-series only — gpt-4o & friends
    reject the parameter with a 400."""
    from model_config import reasoning_effort_for

    assert reasoning_effort_for("gpt-4o") is None
    assert reasoning_effort_for("gpt-4o-mini") is None
    assert reasoning_effort_for("") is None
    assert reasoning_effort_for("gpt-5.5") == "low"
    assert reasoning_effort_for("gpt-5") == "low"
    assert reasoning_effort_for("o3-mini") == "low"

    monkeypatch.setenv("BESSER_AGENT_MODEL_REASONING_EFFORT", "medium")
    mod = importlib.reload(model_config)
    assert mod.reasoning_effort_for("gpt-5.5") == "medium"
    assert mod.reasoning_effort_for("gpt-4o") is None

