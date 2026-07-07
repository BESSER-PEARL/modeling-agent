"""Bring-your-own-key (BYOK) per-request LLM routing for the modeling agent.

A user can paste their own OpenAI / Anthropic / Mistral API key in the
frontend; it is sent over the WebSocket and stored on the BAF session
(keys ``user_api_key`` / ``user_api_provider`` / ``user_api_model``).
When such a key is present, the agent's conversational + generation LLM
calls run through a *per-request* client built from that key instead of
the shared server LLM.

Concurrency safety
------------------
Routing is driven by a :class:`contextvars.ContextVar` (``current_byok``)
holding the per-request :class:`BYOKConfig`. The websocket request
boundary sets it just before ``agent.receive_event`` and resets it after.
asyncio's ``loop.call_soon_threadsafe`` (used by BAF's
``Session.call_manage_transition``) captures a *copy* of the calling
thread's context, so the value propagates into the session's event-loop
thread for that turn. Every session has its own event loop, so two
concurrent users can never cross keys. We never mutate the shared global
LLM/client.

Scope (the agreed "high-value slice")
-------------------------------------
This routes the **free-text** call shapes the agent uses for generation
and conversation:

* ``base_handler._predict_raw`` / ``predict_with_retry`` (generation),
* ``session_helpers.stream_llm_response`` (conversational reply / help /
  describe).

BAF-internal intent classification, RAG embeddings, and OpenAI
*structured-output* ``.parse()`` calls (``predict_structured``) stay on
the shared server key — see the module that wires routing for details.

Errors
------
SDK call exceptions (auth / rate-limit / bad-request) are **not** swallowed
here; they propagate so the existing ``errors.classify_error`` taxonomy
and ``base_handler.predict_with_retry`` can detect and surface them. Only
configuration problems (unknown provider, missing ``anthropic`` SDK) raise
the local :class:`BYOKError`.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from typing import Optional

from agent_config import (
    LLM_MAX_TOKENS_LARGE,
    LLM_TEMPERATURE,
    LLM_TEXT_TEMPERATURE,
)
from model_config import reasoning_effort_for, supports_custom_temperature

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("openai", "anthropic", "mistral")

# Mistral speaks the OpenAI Chat Completions protocol at this endpoint.
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# Per-request SDK timeout. Without it the SDKs default to several minutes,
# which would let a hung BYOK call stall a whole turn.
_SDK_TIMEOUT_SECONDS = 120.0


class BYOKError(RuntimeError):
    """Raised for BYOK *configuration* problems (unknown provider, missing
    SDK). Provider *call* errors (auth/rate-limit) are NOT wrapped — they
    propagate from the SDK so the existing error taxonomy classifies them."""


# ---------------------------------------------------------------------------
# Tier -> per-provider canonical model mapping
# ---------------------------------------------------------------------------
# The agent requests OpenAI-canonical model names per call site (see
# ``model_config``): gpt-4o-mini (classifier), gpt-4o (small generation),
# gpt-5.5 (large generation), gpt-5 (reasoning / vision). BYOK bypasses any
# PIA/Bedrock gateway, so we collapse those into two tiers — "large"
# (quality / heavy) and "small" (cheap / latency-sensitive) — and map each
# tier to the chosen provider's canonical equivalent.
#
# ``large`` honours the user's explicitly chosen model (``user_api_model``)
# when supplied; ``small`` always uses the provider's cheap sibling to keep
# routing / repair / classifier-tier calls inexpensive on the user's key.
_PROVIDER_TIER_MODELS = {
    "openai":    {"large": "gpt-5.5",              "small": "gpt-4o-mini"},
    "anthropic": {"large": "claude-sonnet-4-6",    "small": "claude-haiku-4-5"},
    "mistral":   {"large": "mistral-large-latest", "small": "mistral-small-latest"},
}


def _tier_of(requested_model: Optional[str]) -> str:
    """Bucket a requested OpenAI-canonical model name into a BYOK tier.

    ``None``/empty means the call site used the instance default, which is
    the cheap CLASSIFIER tier -> ``"small"``. gpt-5* / o-series reasoning
    models are heavy -> ``"large"``. Everything else (gpt-4o, gpt-4o-mini)
    -> ``"small"``.
    """
    m = (requested_model or "").strip().lower()
    if not m:
        return "small"
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return "large"
    return "small"


def resolve_model(
    provider: str,
    requested_model: Optional[str],
    user_model: Optional[str],
) -> str:
    """Map the agent's per-call (OpenAI) model request to a concrete model
    name for *provider*."""
    table = _PROVIDER_TIER_MODELS.get(provider, _PROVIDER_TIER_MODELS["openai"])
    if _tier_of(requested_model) == "large":
        return (user_model or "").strip() or table["large"]
    return table["small"]


# ---------------------------------------------------------------------------
# Per-request config + context variable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BYOKConfig:
    """The user's key/provider for a single request. ``model`` is the user's
    explicitly chosen model (used for the ``large`` tier), if any."""

    provider: str
    api_key: str
    model: Optional[str] = None

    def redacted(self) -> str:
        """A log-safe description that never reveals the key."""
        return f"BYOK(provider={self.provider}, model={self.model or '<default>'}, key=(redacted))"


# Holds the active per-request BYOK config, or ``None`` when the request
# uses the shared server LLM. Read by the routing call sites.
current_byok: "contextvars.ContextVar[Optional[BYOKConfig]]" = contextvars.ContextVar(
    "current_byok", default=None
)


def set_current(
    provider: Optional[str],
    api_key: Optional[str],
    model: Optional[str] = None,
) -> "contextvars.Token":
    """Set the per-request BYOK config from raw session values.

    A missing key/provider (or an unsupported provider) clears BYOK so the
    request falls back to the shared server LLM. Returns the token to pass
    to :func:`reset_current`.
    """
    provider_norm = (provider or "").strip().lower()
    key = (api_key or "").strip()
    if not key or provider_norm not in SUPPORTED_PROVIDERS:
        if key and provider_norm and provider_norm not in SUPPORTED_PROVIDERS:
            logger.warning("BYOK: ignoring unsupported provider %r", provider_norm)
        return current_byok.set(None)
    cfg = BYOKConfig(provider=provider_norm, api_key=key, model=(model or "").strip() or None)
    logger.info("BYOK active for this request: %s", cfg.redacted())
    return current_byok.set(cfg)


def reset_current(token: "contextvars.Token") -> None:
    """Restore the previous BYOK config (best effort)."""
    try:
        current_byok.reset(token)
    except Exception:  # pragma: no cover - defensive
        pass


def get_current() -> Optional[BYOKConfig]:
    """Return the active :class:`BYOKConfig`, or ``None``."""
    return current_byok.get()


def is_active() -> bool:
    """True when a BYOK key is active for the current request/context.

    Downstream callers (e.g. the shared-limit message) can use this to know
    whether the user's own key is in play. The user's key/provider/model
    are stored on the BAF session under ``user_api_key`` /
    ``user_api_provider`` / ``user_api_model``.
    """
    return current_byok.get() is not None


# ---------------------------------------------------------------------------
# Per-request multi-provider client
# ---------------------------------------------------------------------------

class BYOKClient:
    """A per-request LLM client for one of the three supported providers.

    Mirrors the provider handling of
    ``besser/generators/llm/llm_client.py``: OpenAI and Mistral go through
    the ``openai`` SDK (Mistral via ``base_url``); Anthropic goes through
    the ``anthropic`` SDK's messages API.
    """

    def __init__(self, provider: str, api_key: str, model: Optional[str] = None) -> None:
        self.provider = (provider or "").strip().lower()
        self._user_model = (model or "").strip() or None
        if self.provider not in SUPPORTED_PROVIDERS:
            raise BYOKError(f"Unsupported BYOK provider: {provider!r}")

        if self.provider in ("openai", "mistral"):
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - openai is a hard dep
                raise BYOKError(
                    "The 'openai' SDK is required for OpenAI/Mistral BYOK."
                ) from exc
            client_kwargs = {"api_key": api_key, "timeout": _SDK_TIMEOUT_SECONDS}
            if self.provider == "mistral":
                client_kwargs["base_url"] = MISTRAL_BASE_URL
            self._client = OpenAI(**client_kwargs)
        else:  # anthropic — lazily gated; SDK may not be installed
            try:
                import anthropic
            except ImportError as exc:
                raise BYOKError(
                    "The 'anthropic' SDK is not installed, so Anthropic BYOK is "
                    "unavailable. Install it with `pip install anthropic`, or use "
                    "an OpenAI or Mistral key instead."
                ) from exc
            self._client = anthropic.Anthropic(api_key=api_key, timeout=_SDK_TIMEOUT_SECONDS)

    # -- public call shapes -------------------------------------------------

    def predict_raw(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single free-text chat-completion call, returning the text.

        ``model`` is the agent's per-call (OpenAI-canonical) tier request;
        it is mapped to this provider's equivalent via :func:`resolve_model`.
        ``max_tokens`` overrides the default completion cap (used by the GUI
        complete-system path to keep large multi-page JSON from truncating).
        """
        target = resolve_model(self.provider, model, self._user_model)
        temp = LLM_TEMPERATURE if temperature is None else temperature
        cap = max_tokens or LLM_MAX_TOKENS_LARGE
        if self.provider == "anthropic":
            return self._anthropic_call(prompt, target, json_mode, temp, cap)
        return self._openai_call(prompt, target, json_mode, temp, reasoning_effort, cap)

    def predict_text(self, prompt: str) -> str:
        """Simple free-text path (cheap/small tier, conversational temp)."""
        return self.predict_raw(
            prompt, model=None, json_mode=False, temperature=LLM_TEXT_TEMPERATURE
        )

    # -- provider implementations ------------------------------------------

    def _openai_call(
        self,
        prompt: str,
        model: str,
        json_mode: bool,
        temperature: float,
        reasoning_effort: Optional[str],
        max_tokens: int = LLM_MAX_TOKENS_LARGE,
    ) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Mistral uses ``max_tokens``; OpenAI uses ``max_completion_tokens``.
        if self.provider == "mistral":
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_completion_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # gpt-5* / o-series reject a custom temperature; cap reasoning instead.
        if supports_custom_temperature(model):
            kwargs["temperature"] = temperature
        else:
            effort = reasoning_effort or reasoning_effort_for(model)
            if effort:
                kwargs["reasoning_effort"] = effort
        completion = self._client.chat.completions.create(**kwargs)
        self._track_openai(getattr(completion, "usage", None), model)
        if not completion.choices:
            return ""
        return completion.choices[0].message.content or ""

    def _anthropic_call(
        self,
        prompt: str,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int = LLM_MAX_TOKENS_LARGE,
    ) -> str:
        content = prompt
        if json_mode:
            content = (
                prompt
                + "\n\nReturn ONLY valid JSON. No markdown code fences, no prose."
            )
        # Anthropic accepts temperature in [0, 1]; the agent uses 0.2/0.4.
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=max(0.0, min(1.0, temperature)),
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(message, "content", []) or []
            if getattr(block, "type", None) == "text"
        )
        self._track_anthropic(getattr(message, "usage", None), model)
        if json_mode:
            text = _strip_code_fences(text)
        return text

    # -- best-effort token tracking ----------------------------------------

    @staticmethod
    def _track_openai(usage, model: str) -> None:
        if usage is None:
            return
        try:
            from tracking import get_tracker

            get_tracker().record_from_usage(usage, model=model)
        except Exception as exc:  # pragma: no cover - tracking is best effort
            logger.debug("BYOK token tracking failed (best-effort): %s", exc)

    @staticmethod
    def _track_anthropic(usage, model: str) -> None:
        if usage is None:
            return
        try:
            from tracking import get_tracker

            get_tracker().record(
                prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                model=model,
            )
        except Exception as exc:  # pragma: no cover - tracking is best effort
            logger.debug("BYOK token tracking failed (best-effort): %s", exc)


def _strip_code_fences(text: str) -> str:
    """Strip ```json / ``` fences an LLM may wrap JSON output in."""
    t = (text or "").strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def get_active_client() -> Optional[BYOKClient]:
    """Build a :class:`BYOKClient` from the active context, or ``None``.

    A fresh client is built per call (SDK client construction is local and
    cheap). Returns ``None`` when no BYOK key is active, so callers fall
    back to the shared server LLM.
    """
    cfg = current_byok.get()
    if cfg is None:
        return None
    return BYOKClient(cfg.provider, cfg.api_key, cfg.model)
