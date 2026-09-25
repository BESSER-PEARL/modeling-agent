"""Per-call-site LLM model routing table.

Every LLM call in the agent belongs to one of five tiers. Each tier is
env-overridable so deployments (e.g. a PIA/Bedrock gateway exposing
different model names) can re-point a tier without code changes:

* ``MODEL_CLASSIFIER`` — routing/classification/extraction tier: the
  unified classifier, the legacy generation sub-router, the request
  planner, JSON-repair / self-correction / name-extraction recovery
  calls, the memory summarizer, UML RAG, help/fallback streaming, and
  ``gpt_predict_json``.
* ``MODEL_GENERATION_LARGE`` — complete-system structured diagram
  generation (the one place where output quality is the product).
* ``MODEL_GENERATION_SMALL`` — single-element & modification structured
  calls, ``describe_model`` streaming, and the file-conversion TEXT path
  (latency-sensitive, schema-constrained outputs).
* ``MODEL_REASONING`` — the free-text design-reasoning pass of two-pass
  generation.
* ``MODEL_VISION`` — file-conversion vision calls (image / PDF → diagram).

Embeddings are pinned separately (``MODEL_EMBEDDINGS``) so a silent
OpenAI default change can never alter RAG behavior.
"""

import os

_ENV_PREFIX = "BESSER_AGENT_MODEL_"


def _env(name: str, default: str) -> str:
    """Read ``BESSER_AGENT_MODEL_<name>``, falling back to *default*."""
    value = os.getenv(_ENV_PREFIX + name, "").strip()
    return value or default


MODEL_CLASSIFIER = _env("CLASSIFIER", "gpt-4o-mini")
# gpt-5.6-terra: newer than gpt-5.5 at half the cost, faster, and ~10x cheaper
# cached input for the stable system prompt. reasoning_effort="low" and the
# fixed-temperature handling apply via the "gpt-5" prefix. Vision stays on
# gpt-5. All overridable via BESSER_AGENT_MODEL_* env vars.
MODEL_GENERATION_LARGE = _env("GENERATION_LARGE", "gpt-5.6-terra")
# GUI complete-system generation gets its OWN knob: design quality tracks the
# model's taste far more than diagram generation does, so it can run a
# stronger model without slowing down class-diagram creates. Defaults to the
# LARGE tier when unset (BESSER_AGENT_MODEL_GENERATION_GUI overrides).
MODEL_GENERATION_GUI = _env("GENERATION_GUI", "") or MODEL_GENERATION_LARGE
MODEL_GENERATION_SMALL = _env("GENERATION_SMALL", "gpt-5.6-luna")
MODEL_REASONING = _env("REASONING", "gpt-5.6-terra")
MODEL_VISION = _env("VISION", "gpt-5")

# Pinned explicitly so a langchain/OpenAI default bump never silently
# changes the RAG vector space (existing vectors would stop matching).
MODEL_EMBEDDINGS = _env("EMBEDDINGS", "text-embedding-3-small")


# Model families that reject an explicit ``temperature`` other than the
# default (the OpenAI API returns 400 for gpt-5* / o-series reasoning
# models). Call sites must omit the parameter for these models instead
# of passing their usual 0.0–0.4 values.
_FIXED_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")
# Claude models that 400 on any temperature/top_p/top_k. Substring match so
# gateway ids such as "us.anthropic.claude-sonnet-5" count too.
_NO_SAMPLING_CLAUDE = ("claude-sonnet-5", "claude-opus-4-7", "claude-opus-4-8",
                       "claude-opus-5", "claude-fable", "claude-mythos")


def supports_custom_temperature(model: str) -> bool:
    """True when *model* accepts an explicit ``temperature`` parameter."""
    name = (model or "").lower()
    if any(m in name for m in _NO_SAMPLING_CLAUDE):
        return False
    return not any(name.startswith(p) for p in _FIXED_TEMPERATURE_PREFIXES)


# reasoning_effort for gpt-5* / o-series calls. "low" cuts gpt-5.5's
# hidden reasoning from ~512 to ~50 tokens on diagram generation (42s →
# 26s) with no measurable quality loss — structured diagram specs don't
# need deep chain-of-thought. NOTE: "minimal" is rejected by gpt-5.5.
MODEL_REASONING_EFFORT = _env("REASONING_EFFORT", "low")


def reasoning_effort_for(model: str) -> "str | None":
    """``reasoning_effort`` to pass for *model*, or None for non-reasoning
    models (gpt-4o & friends reject the parameter). Never for Claude: an
    OpenAI-compatible gateway may translate it to ``budget_tokens``, which
    Sonnet 5 rejects; Claude effort goes through ``anthropic_effort``."""
    if supports_custom_temperature(model) or "claude" in (model or "").lower():
        return None
    return MODEL_REASONING_EFFORT


_ANTHROPIC_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def anthropic_effort(model: str) -> "str | None":
    """``output_config.effort`` for a Claude model that takes no sampling
    params, or None when the configured effort is not a Claude level."""
    if supports_custom_temperature(model):
        return None
    effort = (MODEL_REASONING_EFFORT or "").lower()
    return effort if effort in _ANTHROPIC_EFFORTS else None
