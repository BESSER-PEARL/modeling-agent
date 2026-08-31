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
# 2026-07: moved onto the newer gpt-5.6 family. gpt-5.6-terra ($2.50/$15) is a
# generation newer than gpt-5.5 ($5/$30) at HALF the cost and is faster, and its
# cached-input price ($0.25 vs $2.50) makes the stable system prompt ~10x cheaper
# on input. reasoning_effort="low" + fixed-temperature handling apply
# automatically (the "gpt-5" prefix already matches gpt-5.6). A/B terra vs
# gpt-5.6-sol ($5/$30, max quality) on diagram quality and keep the winner.
# Vision stays on gpt-5 until gpt-5.6 image support is confirmed. All
# overridable via BESSER_AGENT_MODEL_* env vars.
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


def supports_custom_temperature(model: str) -> bool:
    """True when *model* accepts an explicit ``temperature`` parameter."""
    name = (model or "").lower()
    return not any(name.startswith(p) for p in _FIXED_TEMPERATURE_PREFIXES)


# reasoning_effort for gpt-5* / o-series calls. "low" cuts gpt-5.5's
# hidden reasoning from ~512 to ~50 tokens on diagram generation (42s →
# 26s) with no measurable quality loss — structured diagram specs don't
# need deep chain-of-thought. NOTE: "minimal" is rejected by gpt-5.5.
MODEL_REASONING_EFFORT = _env("REASONING_EFFORT", "low")


def reasoning_effort_for(model: str) -> "str | None":
    """``reasoning_effort`` to pass for *model*, or None for non-reasoning
    models (gpt-4o & friends reject the parameter)."""
    if supports_custom_temperature(model):
        return None
    return MODEL_REASONING_EFFORT
