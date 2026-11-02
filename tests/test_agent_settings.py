"""AgentSettings reads every variable exactly as the code it replaced did.

Each row pairs the expression the agent used before the settings class
(``old``, kept verbatim, reading ``os.environ``) with how the agent reads the
same value now (``new``). Every row is checked unset and with a list of edge
values, so a change to a variable's name, default or parsing fails here.
"""

import importlib
import os
import sys

import pytest

from common.settings import AgentSettings


def _get(name, *default):
    return os.environ.get(name, *default)


def _old_model(tier, default):
    """llm.model_config._env before the settings class."""
    return lambda: os.getenv("BESSER_AGENT_MODEL_" + tier, "").strip() or default


def _model(field):
    return lambda s: getattr(s, field)


def _current_route(_settings):
    from llm.byok_config import current_byok
    from llm.tracking import token_tracker

    token = current_byok.set(None)
    try:
        return token_tracker._current_route()
    finally:
        current_byok.reset(token)


def _allow_custom_base_url(_settings):
    from llm import byok

    return byok._allow_custom_base_url()


def _telemetry_base(_settings):
    from conversation import telemetry

    return telemetry._telemetry_url().removesuffix("/besser_api/telemetry/event")


# (field, variable, default, old read, new read)
ROWS = [
    # The Dockerfile's ${OPENAI_API_KEY:-} and ${BESSER_AGENT_WS_ORIGIN:-https://...}.
    ("openai_api_key", "OPENAI_API_KEY", "",
     lambda: _get("OPENAI_API_KEY") or "", lambda s: s.openai_api_key.get_secret_value()),
    ("ws_origin", "BESSER_AGENT_WS_ORIGIN", "https://editor.besser-pearl.org",
     lambda: _get("BESSER_AGENT_WS_ORIGIN") or "https://editor.besser-pearl.org", lambda s: s.ws_origin),
    ("ws_origin_alt", "BESSER_AGENT_WS_ORIGIN_ALT", "https://experimental.besser-pearl.org",
     lambda: _get("BESSER_AGENT_WS_ORIGIN_ALT") or "https://experimental.besser-pearl.org",
     lambda s: s.ws_origin_alt),
    ("model_classifier", "BESSER_AGENT_MODEL_CLASSIFIER", "gpt-6-luna",
     _old_model("CLASSIFIER", "gpt-6-luna"), _model("model_classifier")),
    ("model_generation_large", "BESSER_AGENT_MODEL_GENERATION_LARGE", "gpt-5-mini",
     _old_model("GENERATION_LARGE", "gpt-5-mini"), _model("model_generation_large")),
    ("model_generation_gui", "BESSER_AGENT_MODEL_GENERATION_GUI", "gpt-6-sol",
     _old_model("GENERATION_GUI", "gpt-6-sol"), _model("model_generation_gui")),
    ("model_generation_small", "BESSER_AGENT_MODEL_GENERATION_SMALL", "gpt-6-luna",
     _old_model("GENERATION_SMALL", "gpt-6-luna"), _model("model_generation_small")),
    ("model_reasoning", "BESSER_AGENT_MODEL_REASONING", "gpt-5-mini",
     _old_model("REASONING", "gpt-5-mini"), _model("model_reasoning")),
    ("model_vision", "BESSER_AGENT_MODEL_VISION", "gpt-4o",
     _old_model("VISION", "gpt-4o"), _model("model_vision")),
    ("model_embeddings", "BESSER_AGENT_MODEL_EMBEDDINGS", "text-embedding-3-small",
     _old_model("EMBEDDINGS", "text-embedding-3-small"), _model("model_embeddings")),
    ("model_reasoning_effort", "BESSER_AGENT_MODEL_REASONING_EFFORT", "low",
     _old_model("REASONING_EFFORT", "low"), _model("model_reasoning_effort")),
    ("model_catalog", "BESSER_MODEL_CATALOG", "",
     lambda: os.getenv("BESSER_MODEL_CATALOG", "").strip(), lambda s: s.model_catalog),
    ("allow_custom_base_url", "BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", False,
     lambda: os.getenv("BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", "").strip().lower() in ("1", "true", "yes", "on"),
     _allow_custom_base_url),
    ("openai_base_url", "OPENAI_BASE_URL", None,
     lambda: os.getenv("OPENAI_BASE_URL") or None, _current_route),
    ("cost_log_interval", "BESSER_AGENT_COST_LOG_INTERVAL", "600",
     lambda: float(os.getenv("BESSER_AGENT_COST_LOG_INTERVAL", "600") or 0),
     lambda s: float(s.cost_log_interval or 0)),
    ("compact_spec", "BESSER_AGENT_COMPACT_SPEC", True,
     lambda: os.environ.get("BESSER_AGENT_COMPACT_SPEC", "1") != "0", lambda s: s.compact_spec),
    ("stt_language", "BESSER_AGENT_STT_LANGUAGE", None,
     lambda: os.getenv("BESSER_AGENT_STT_LANGUAGE", "").strip() or None, lambda s: s.stt_language),
    ("backend_url", "BESSER_BACKEND_URL", "http://localhost:3001",
     lambda: os.environ.get("BESSER_BACKEND_URL", "http://localhost:3001").rstrip("/"), _telemetry_base),
    ("log_prompts", "LOG_PROMPTS", False,
     lambda: os.getenv("LOG_PROMPTS", "").strip().lower() in {"1", "true", "yes"}, lambda s: s.log_prompts),
]

EDGE_VALUES = [
    "", " ", "0", " 0 ", "-1", "1", " 1", "7", " 7 ", "2.5", "1e3", "1_000", "nan", "inf", "abc",
    "true", "TRUE", " yes ", "on", "ON", "no", "false", "False", "off", "gpt-x", " gpt-x ",
    "https://a.example/", "/srv/catalog.json", "$HOME", 'a"b', "a\\b", "é",
]

ALL_VARIABLES = [row[1] for row in ROWS]


@pytest.fixture
def clean_env(monkeypatch):
    """No agent variable set."""
    for name in ALL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _outcome(read):
    """The value read, typed (True is not 1, NaN equals NaN), or the error raised."""
    try:
        value = read()
    except ValueError as exc:
        return type(exc)
    return type(value), repr(value)


def test_every_field_has_a_row():
    assert sorted(row[0] for row in ROWS) == sorted(AgentSettings.model_fields)


@pytest.mark.parametrize("field, variable, default", [row[:3] for row in ROWS], ids=[row[0] for row in ROWS])
def test_field_names_its_variable_and_default(field, variable, default):
    info = AgentSettings.model_fields[field]
    assert info.validation_alias == variable
    value = info.default.get_secret_value() if hasattr(info.default, "get_secret_value") else info.default
    assert value == default
    assert info.description


@pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
def test_unset_reads_as_before(clean_env, row):
    _, _, _, old, new = row
    assert _outcome(lambda: new(AgentSettings())) == _outcome(old)


@pytest.mark.parametrize("value", EDGE_VALUES, ids=repr)
@pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
def test_edge_values_read_as_before(clean_env, row, value):
    _, variable, _, old, new = row
    clean_env.setenv(variable, value)
    assert _outcome(lambda: new(AgentSettings())) == _outcome(old)


def test_a_malformed_value_never_stops_the_other_settings(clean_env):
    for name in ALL_VARIABLES:
        clean_env.setenv(name, "é not a value ;")
    AgentSettings()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows environment names ignore case")
def test_names_are_case_sensitive(clean_env):
    clean_env.setenv("log_prompts", "1")
    assert AgentSettings().log_prompts is False


def test_a_variable_named_like_the_field_is_ignored(clean_env):
    """Only BESSER_AGENT_WS_ORIGIN sets ws_origin. The field name differs from the
    variable beyond case, so this holds on Windows too, unlike the test above."""
    clean_env.setenv("ws_origin", "https://elsewhere.example")
    assert AgentSettings().ws_origin == "https://editor.besser-pearl.org"


def test_module_level_values_are_read_at_import(clean_env):
    """A reload sees the new environment, as before; the per-call reads see it at once."""
    from llm import byok, model_config

    clean_env.setenv("BESSER_AGENT_MODEL_REASONING_EFFORT", " medium ")
    clean_env.setenv("BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", "on")
    try:
        assert model_config.MODEL_REASONING_EFFORT != "medium"
        assert importlib.reload(model_config).MODEL_REASONING_EFFORT == "medium"
        assert byok._allow_custom_base_url() is True
    finally:
        clean_env.delenv("BESSER_AGENT_MODEL_REASONING_EFFORT")
        importlib.reload(model_config)


def test_import_time_constants_come_from_the_settings(clean_env):
    from diagram_handlers.core import base_handler
    from diagram_handlers.types import class_diagram_handler
    from llm.tracking import token_tracker

    settings = AgentSettings()
    assert base_handler._LOG_PROMPTS is settings.log_prompts
    assert class_diagram_handler.COMPACT_SPEC_ENABLED is settings.compact_spec
    assert token_tracker._SUMMARY_INTERVAL_SECONDS == float(settings.cost_log_interval or 0)
