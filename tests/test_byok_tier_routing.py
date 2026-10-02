"""BYOK routes each call by the model_config tier it requests.

Before the fix ``_tier_of`` bucketed any reasoning-model id as "large", so a
small edit (MODEL_GENERATION_SMALL = gpt-6-luna) ran on the provider's top
model: gpt-5.5 or claude-sonnet-5 instead of gpt-4o-mini or claude-haiku-4-5.
"""

import pytest

import byok
from model_config import (
    MODEL_CLASSIFIER,
    MODEL_GENERATION_GUI,
    MODEL_GENERATION_LARGE,
    MODEL_GENERATION_SMALL,
    MODEL_REASONING,
)

CHEAP = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5", "mistral": "mistral-small-latest"}
STRONG = {"openai": "gpt-5.5", "anthropic": "claude-sonnet-5", "mistral": "mistral-large-latest"}


@pytest.mark.parametrize("provider", sorted(CHEAP))
@pytest.mark.parametrize("requested", [MODEL_GENERATION_SMALL, MODEL_CLASSIFIER, None])
def test_small_edits_and_classifier_calls_use_the_cheap_sibling(provider, requested):
    assert byok.resolve_model(provider, requested, None) == CHEAP[provider]


@pytest.mark.parametrize("provider", sorted(STRONG))
@pytest.mark.parametrize("requested", [MODEL_GENERATION_LARGE, MODEL_GENERATION_GUI, MODEL_REASONING])
def test_complete_system_generation_stays_on_the_strong_model(provider, requested):
    assert byok.resolve_model(provider, requested, None) == STRONG[provider]


def test_a_chosen_model_still_wins_for_small_edits():
    assert byok.resolve_model("anthropic", MODEL_GENERATION_SMALL, "claude-opus-5") == "claude-opus-5"
