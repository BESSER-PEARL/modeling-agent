"""Unified LLM-based classifier for generation requests.

The modeling agent's unified classifier decides — in its single
per-message LLM call — both the STATE-level intent and the generation
SUB-routing. This module holds the dispatch-shape schema for that
sub-routing verdict:

    route ∈ {"smart", "deterministic", "modeling", "other"}

and — in the same structured-output call — returns any fields the
caller needs so we never re-prompt the LLM later:

  * For ``route == "deterministic"``: which BESSER built-in generator
    (``generator_type``) so the caller dispatches to django / pydantic /
    sql / etc. without any further keyword matching.
  * For ``route == "smart"``: a polished ``refined_instructions`` prompt
    for the smart generator, plus a provider suggestion.

The generation-only classifier prompt that used to live here (a second
rulebook that drifted out of sync with the unified classifier) is
retired — ``generation_handler._classification_to_legacy`` adapts the
unified verdict into this shape instead. This module now only defines
the :class:`GenerationClassification` dispatch schema and assembles the
``trigger_smart_generator`` payload.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------


# The names BESSER's deterministic generator registry understands. If
# the classifier picks ``deterministic`` with a type not in this list,
# the caller falls back to ``None`` and shows the user the generator
# menu. Keep in sync with ``generation_handler.GENERATOR_KEYWORDS``.
_DETERMINISTIC_GENERATOR_TYPES = Literal[
    "django",
    "backend",
    "web_app",
    "sql",
    "sqlalchemy",
    "python",
    "java",
    "pydantic",
    "jsonschema",
    "smartdata",
    "agent",
    "qiskit",
    "rest_api",
    "rdf",
    "export",
    "deploy",
]


class GenerationClassification(BaseModel):
    """Single source of truth for routing + sub-routing of a generation request."""

    route: Literal["smart", "deterministic", "modeling", "other"] = Field(
        ...,
        description=(
            "Pick exactly one:\n"
            "  'smart'         — user wants a custom codebase beyond any single "
            "BESSER built-in generator: ANY non-BESSER language or framework "
            "(rust, kotlin, swift, rails, flask, next.js, spring boot, go, "
            "elixir, php/laravel, c#, c++, angular, svelte, ios app, ...) OR "
            "a compound build ('full-stack fastapi + jwt + postgres', "
            "'dockerized next.js', 'react + node'). This path runs the "
            "LLM-augmented smart generator with the user's BYOK key.\n"
            "  'deterministic' — user wants ONE of BESSER's built-in generators "
            "EXACTLY: django, pydantic, python classes, java classes, sql, "
            "sqlalchemy, jsonschema, smartdata, agent, qiskit, web_app, "
            "backend, rest_api, rdf, export, deploy. If a non-BESSER language "
            "is named, it is 'smart', never 'deterministic'.\n"
            "  'modeling'      — user wants to CREATE or MODIFY a diagram "
            "(class diagram, state machine, GUI, agent diagram), not "
            "generate source code. 'generate a class diagram for X' is "
            "modeling, not deterministic.\n"
            "  'other'         — greeting, capability question, chat, "
            "anything that is not a code-generation or modeling request."
        ),
    )
    generator_type: Optional[_DETERMINISTIC_GENERATOR_TYPES] = Field(
        default=None,
        description=(
            "REQUIRED when route='deterministic'. One of BESSER's built-in "
            "generator names. LEAVE NULL for route != 'deterministic'."
        ),
    )
    refined_instructions: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when route='smart'. A polished, implementation-focused "
            "prompt for the smart generator: 1-3 short paragraphs naming the "
            "stack (e.g. 'Ruby on Rails 7, PostgreSQL via Active Record, "
            "Devise auth'), any explicit non-functional requirements the "
            "user mentioned (JWT, Docker, migrations, tests), and NO "
            "invented requirements. Max 2000 chars. LEAVE NULL for "
            "route != 'smart'."
        ),
    )
    provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description=(
            "Suggested LLM provider for the smart generator when route='smart'. "
            "Default 'anthropic'. Ignored for other routes."
        ),
    )
    reason: str = Field(
        ...,
        description=(
            "One short sentence (max 160 chars) explaining the classification. "
            "Used for logs and surfaced to the user as a hint."
        ),
    )


# ---------------------------------------------------------------------
# Payload assembly for the smart-gen path
# ---------------------------------------------------------------------


_DEFAULT_SMART_GEN_MODEL_BY_PROVIDER: Dict[str, str] = {
    # Match ``besser/generators/llm/llm_client.py::DEFAULT_MODELS`` and
    # the backend's config endpoint. If these diverge, a run with no
    # explicit ``llmModel`` override will fail with an "unknown model"
    # upstream error.
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "mistral": "mistral-large-latest",
}


def build_trigger_smart_generator_payload(
    classification: GenerationClassification,
    reason_prefix: str = "",
) -> Dict[str, Any]:
    """Assemble the WebSocket ``trigger_smart_generator`` action payload.

    Requires a classification whose ``route == 'smart'``.
    """
    if classification.route != "smart":
        raise ValueError(
            "build_trigger_smart_generator_payload called with non-smart classification"
        )
    instructions = (classification.refined_instructions or "").strip()
    if not instructions:
        raise ValueError("smart classification has no refined_instructions")

    provider = classification.provider or "anthropic"
    llm_model = _DEFAULT_SMART_GEN_MODEL_BY_PROVIDER.get(provider, "claude-sonnet-4-6")

    # Short, neutral run banner. The provider/free-tier choice and the BYOK
    # option were already conveyed by the confirmation copy shown before this
    # run (see ``_build_smart_gen_confirmation``), so this mid-run line stays
    # minimal instead of repeating the API-key explanation.
    return {
        "action": "trigger_smart_generator",
        "instructions": instructions,
        "provider": provider,
        "llmModel": llm_model,
        "message": "Generating your application from the spec…",
    }
