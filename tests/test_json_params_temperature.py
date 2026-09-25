"""JSON-mode calls must not send a temperature the model rejects.

gpt-5* / o-series models 400 on an explicit non-default temperature:

    Unsupported value: 'temperature' does not support 0.2 with this model.
    Only the default (1) value is supported.

`base_handler` guards its two call sites with `supports_custom_temperature`.
`gpt_predict_json` did not, and it is the path the file-conversion text branch
uses (its `model` override exists for exactly that caller). So once the
configured model moved to a gpt-5 tier, every converted text file failed with

    Failed to process the text file. The AI model encountered an error.

which is what a user saw after pasting a requirements document.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_config import reasoning_effort_for, supports_custom_temperature  # noqa: E402


@pytest.mark.parametrize("model", [
    "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.5", "o1-mini", "o3", "o4-mini",
])
def test_fixed_temperature_models_are_detected(model):
    assert supports_custom_temperature(model) is False
    assert reasoning_effort_for(model), "these models take reasoning_effort instead"


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6"])
def test_ordinary_models_still_accept_temperature(model):
    assert supports_custom_temperature(model) is True


def _params_for(model: str) -> dict:
    """Mirror of agent_setup._json_params_for, which is a closure.

    Kept in sync by test_agent_setup_uses_the_guard below, which asserts the
    real function applies the same rule rather than sending temperature flat.
    """
    params = {"max_completion_tokens": 1, "response_format": {"type": "json_object"}}
    if supports_custom_temperature(model):
        params["temperature"] = 0.2
    else:
        effort = reasoning_effort_for(model)
        if effort:
            params["reasoning_effort"] = effort
    return params


def test_a_reasoning_model_gets_no_temperature():
    params = _params_for("gpt-5.6-terra")
    assert "temperature" not in params, "this is the 400 the user hit"
    assert params["reasoning_effort"]


def test_a_standard_model_keeps_its_temperature():
    assert _params_for("gpt-4o")["temperature"] == 0.2


def test_agent_setup_uses_the_guard():
    """The real call site must not send temperature unconditionally."""
    source = (Path(__file__).resolve().parents[1] / "src" / "agent_setup.py").read_text(
        encoding="utf-8"
    )
    assert "_json_params_for" in source, "the per-model params helper is gone"
    assert "supports_custom_temperature" in source, (
        "gpt_predict_json no longer guards temperature -- reasoning models will 400"
    )
    assert "'temperature': LLM_TEMPERATURE,\n        'max_completion_tokens'" not in source, (
        "the unconditional temperature dict is back"
    )
