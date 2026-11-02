"""Environment configuration of the modeling agent.

Every variable the agent reads is declared once on :class:`AgentSettings` with
its default and meaning. ``AgentSettings()`` reads the environment when it is
built: module-level values build it at import (a reloaded module re-reads
them), the others at each use, so a change reaches the next call.

A variable whose old reading raised on a bad value is kept as text and parsed
where it is used, so it still fails there and never stops the other settings
from loading.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BeforeValidator, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _one_of(*words: str):
    """True when the stripped, lower-cased value is one of *words*."""
    return BeforeValidator(lambda value: str(value).strip().lower() in words)


def _stripped_or(default: str):
    """The stripped value, or *default* when that is blank."""
    return BeforeValidator(lambda value: str(value).strip() or default)


def _model(tier: str, default: str, description: str):
    return Field(default, validation_alias=f"BESSER_AGENT_MODEL_{tier}", description=description)


class AgentSettings(BaseSettings):
    """The modeling agent's environment variables (all optional)."""

    # Defaults are final values, not raw text, so they are not parsed again.
    # No protected namespace: the model tiers are named ``model_*``.
    model_config = SettingsConfigDict(
        case_sensitive=True, populate_by_name=True, extra="ignore", validate_default=False,
        protected_namespaces=())

    # -- Container config.yaml (rendered by app.container_config) ---------------
    openai_api_key: SecretStr = Field(
        SecretStr(""), validation_alias="OPENAI_API_KEY",
        description="Server OpenAI key, written to config.yaml as nlp.openai.api_key.")
    ws_origin: Annotated[str, BeforeValidator(lambda v: v or "https://editor.besser-pearl.org")] = Field(
        "https://editor.besser-pearl.org", validation_alias="BESSER_AGENT_WS_ORIGIN",
        description="First allowed browser origin in config.yaml (blank: the default).")
    ws_origin_alt: Annotated[str, BeforeValidator(lambda v: v or "https://experimental.besser-pearl.org")] = Field(
        "https://experimental.besser-pearl.org", validation_alias="BESSER_AGENT_WS_ORIGIN_ALT",
        description="Second allowed browser origin in config.yaml (blank: the default).")

    # -- Model tiers (read at import by llm.model_config; blank is the default) ---
    model_classifier: Annotated[str, _stripped_or("gpt-6-luna")] = _model(
        "CLASSIFIER", "gpt-6-luna", "Unified classifier, planner, repair, summaries, help.")
    model_generation_large: Annotated[str, _stripped_or("gpt-5-mini")] = _model(
        "GENERATION_LARGE", "gpt-5-mini", "Complete-system generation of the larger diagram types.")
    model_generation_gui: Annotated[str, _stripped_or("gpt-6-sol")] = _model(
        "GENERATION_GUI", "gpt-6-sol", "GUI complete-system generation.")
    model_generation_small: Annotated[str, _stripped_or("gpt-6-luna")] = _model(
        "GENERATION_SMALL", "gpt-6-luna", "Single-element and modification calls.")
    model_reasoning: Annotated[str, _stripped_or("gpt-5-mini")] = _model(
        "REASONING", "gpt-5-mini", "Free-text reasoning pass of two-pass generation.")
    model_vision: Annotated[str, _stripped_or("gpt-4o")] = _model(
        "VISION", "gpt-4o", "File-conversion vision calls.")
    model_embeddings: Annotated[str, _stripped_or("text-embedding-3-small")] = _model(
        "EMBEDDINGS", "text-embedding-3-small", "RAG embeddings.")
    model_reasoning_effort: Annotated[str, _stripped_or("low")] = _model(
        "REASONING_EFFORT", "low", "reasoning_effort sent to reasoning models.")

    # -- LLM routing and cost ------------------------------------------------------
    model_catalog: Annotated[str, BeforeValidator(lambda v: str(v).strip())] = Field(
        "", validation_alias="BESSER_MODEL_CATALOG",
        description="Path to the deployment's model catalog; read once per process.")
    allow_custom_base_url: Annotated[bool, _one_of("1", "true", "yes", "on")] = Field(
        False, validation_alias="BESSER_AGENT_ALLOW_CUSTOM_BASE_URL",
        description="Let a BYOK request name its own API base URL (1, true, yes, on). Read per request.")
    openai_base_url: Annotated[Optional[str], BeforeValidator(lambda v: v or None)] = Field(
        None, validation_alias="OPENAI_BASE_URL",
        description="Endpoint of the shared OpenAI client, for per-route pricing (blank: none). Read per call.")
    # Text: float() at import, as before, so a malformed value fails there.
    cost_log_interval: str = Field(
        "600", validation_alias="BESSER_AGENT_COST_LOG_INTERVAL",
        description="Seconds between the token/cost INFO lines (blank or 0: off). Read at import.")

    # -- Features and diagnostics ----------------------------------------------
    compact_spec: Annotated[bool, BeforeValidator(lambda v: v != "0")] = Field(
        True, validation_alias="BESSER_AGENT_COMPACT_SPEC",
        description="Compact class-diagram spec schema; off only for exactly '0'. Read at import.")
    stt_language: Annotated[Optional[str], BeforeValidator(lambda v: str(v).strip() or None)] = Field(
        None, validation_alias="BESSER_AGENT_STT_LANGUAGE",
        description="Pins speech-to-text to a language (blank: auto-detect). Read at startup.")
    backend_url: str = Field(
        "http://localhost:3001", validation_alias="BESSER_BACKEND_URL",
        description="BESSER backend base URL for the opt-in study telemetry. Read per event.")
    log_prompts: Annotated[bool, _one_of("1", "true", "yes")] = Field(
        False, validation_alias="LOG_PROMPTS",
        description="Log full LLM prompts at INFO (1, true, yes). Read at import.")
