"""The deployment's model catalog: gateway endpoints and rates, shared with BESSER.

Before it, the frontend sent the internal PIA gateway's URL with every key
(the host was public in the bundle), and a gateway's own rates had no place to
live, so its calls were priced at the vendor's list price. The gateway here is
fictional: a deployment's catalog file is the only place a real one belongs.
"""

import json

import pytest

import byok
import model_catalog
from model_catalog import load_model_catalog
from tracking.token_tracker import TokenTracker

GATEWAY_URL = "https://gateway.example.internal/v1"
CATALOG = {
    "providers": {
        "pia": {
            "label": "Example gateway",
            "base_url": GATEWAY_URL,
            "models": [
                {"id": "claude-sonnet-5-5", "default": True,
                 "price": {"input": 2.5, "output": 12.0}},
                {"id": "gpt-6-luna", "price": {"input": 0.3, "output": 0.7, "cache_read": 0.03}},
            ],
        },
        "openai": {"label": "OpenAI", "models": [
            {"id": "gpt-9-nova", "price": {"input": 7.0, "output": 21.0}},
        ]},
        "mistral": None,
    },
}


@pytest.fixture(autouse=True)
def _fresh_catalog(monkeypatch):
    monkeypatch.delenv(model_catalog.CATALOG_ENV_VAR, raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", raising=False)
    load_model_catalog.cache_clear()
    yield
    load_model_catalog.cache_clear()


@pytest.fixture
def use_catalog(tmp_path, monkeypatch):
    def _use(data) -> None:
        path = tmp_path / "catalog.json"
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        monkeypatch.setenv(model_catalog.CATALOG_ENV_VAR, str(path))
        load_model_catalog.cache_clear()
    return _use


def _byok(provider, base_url=None, model="claude-sonnet-5-5"):
    return byok.set_current(provider, "sk-test", model, base_url)


def _cost_of_a_million_each(model):
    tracker = TokenTracker()
    tracker.record(prompt_tokens=1_000_000, completion_tokens=1_000_000, model=model)
    return tracker.summary()["estimated_cost_usd"]


def test_without_a_catalog_there_is_no_gateway():
    token = _byok("pia")
    try:
        assert byok.get_current() is None
    finally:
        byok.reset_current(token)


def test_the_gateway_endpoint_comes_from_the_catalog(use_catalog):
    use_catalog(CATALOG)
    # The gate stays off: the endpoint is the operator's, not the request's.
    token = _byok("pia", base_url="http://169.254.169.254/")
    try:
        cfg = byok.get_current()
        assert (cfg.provider, cfg.base_url, cfg.model) == ("openai", GATEWAY_URL, "claude-sonnet-5-5")
    finally:
        byok.reset_current(token)


def test_a_custom_base_url_is_still_gated(use_catalog):
    use_catalog(CATALOG)
    token = _byok("openai", base_url="http://localhost:11434/v1")
    try:
        assert byok.get_current() is None
    finally:
        byok.reset_current(token)


@pytest.mark.parametrize("model, expected", [
    ("claude-sonnet-5-5", 2.5 + 12.0),
    ("gpt-6-luna", 0.3 + 0.7),
])
def test_gateway_calls_are_priced_at_the_catalog_rates(use_catalog, model, expected):
    use_catalog(CATALOG)
    token = _byok("pia", model=model)
    try:
        assert _cost_of_a_million_each(model) == pytest.approx(expected)
    finally:
        byok.reset_current(token)


def test_the_shared_client_on_a_gateway_is_priced_there_too(use_catalog, monkeypatch):
    use_catalog(CATALOG)
    monkeypatch.setenv("OPENAI_BASE_URL", GATEWAY_URL)
    assert _cost_of_a_million_each("gpt-6-luna") == pytest.approx(0.3 + 0.7)


def test_the_same_id_off_the_gateway_keeps_its_vendor_price(use_catalog):
    use_catalog(CATALOG)
    assert _cost_of_a_million_each("gpt-6-luna") == pytest.approx(0.1 + 0.5)


def test_a_catalog_price_on_a_vendor_model_is_used(use_catalog):
    use_catalog(CATALOG)
    assert _cost_of_a_million_each("gpt-9-nova") == pytest.approx(7.0 + 21.0)


def test_cache_reads_and_writes_use_the_catalog_rates(use_catalog):
    use_catalog(CATALOG)
    token = _byok("pia", model="gpt-6-luna")
    try:
        tracker = TokenTracker()
        tracker.record(prompt_tokens=2_000_000, completion_tokens=0, model="gpt-6-luna",
                       cached_prompt_tokens=1_000_000)
        # 1M cached at 0.03 + 1M uncached at 0.3.
        assert tracker.summary()["estimated_cost_usd"] == pytest.approx(0.33)
    finally:
        byok.reset_current(token)


@pytest.mark.parametrize("bad", [
    "{not json",
    {"providers": {"pia": {"label": "P", "base_url": "gateway", "models": []}}},
    {"providers": {"x": {"label": "X", "models": [{"id": "m", "price": {"input": 1}}]}}},
])
def test_a_malformed_catalog_fails_loudly(use_catalog, bad):
    use_catalog(bad)
    with pytest.raises(ValueError, match="Model catalog"):
        load_model_catalog()
