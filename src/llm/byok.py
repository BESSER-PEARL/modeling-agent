"""Bring-your-own-key (BYOK) per-request LLM routing for the modeling agent.

A user can paste their own OpenAI / Anthropic / Mistral / Nebius API key in
the frontend; it is sent over the WebSocket and stored on the BAF session
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

Scope
-----
``llm.provider.LLMProvider`` asks :func:`get_active_client` on every call,
so these run on the user's key when one is active: free text and JSON mode,
structured output (``.parse()`` on the user's OpenAI client, JSON mode for the
other providers) and streaming (one chunk). A call with images (file
conversion, a GUI mockup) uses the user's key only when the provider can read
images (the ``vision`` entry of the tier table, ``BYOKClient.vision_model``);
otherwise it runs on the server key.

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
from typing import Optional, Sequence

from common.agent_config import (
    LLM_MAX_TOKENS_LARGE,
    LLM_TEMPERATURE,
)
from common.errors import is_key_rejection
from common.settings import AgentSettings
from llm.model_config import (
    MODEL_CLASSIFIER,
    MODEL_GENERATION_GUI,
    MODEL_GENERATION_LARGE,
    MODEL_GENERATION_SMALL,
    MODEL_REASONING,
    MODEL_VISION,
    anthropic_effort,
    is_openai_reasoning_model,
    reasoning_effort_for,
    supports_custom_temperature,
)
from llm.byok_config import BYOKConfig, current_byok
from llm.model_catalog import load_model_catalog
from utilities.image_inputs import anthropic_content, openai_content

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("openai", "anthropic", "mistral", "nebius")

# Mistral speaks the OpenAI Chat Completions protocol at this endpoint.
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
# Nebius Token Factory is OpenAI-compatible too (same endpoint BESSER uses).
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Per-request SDK timeout. Without it the SDKs default to several minutes,
# which would let a hung BYOK call stall a whole turn. 120 s was too short
# for a long-spec reasoning pass on slower providers.
_SDK_TIMEOUT_SECONDS = 300.0


class BYOKError(RuntimeError):
    """Raised for BYOK *configuration* problems (unknown provider, missing
    SDK). Provider *call* errors (auth/rate-limit) are NOT wrapped — they
    propagate from the SDK so the existing error taxonomy classifies them."""


# ---------------------------------------------------------------------------
# Tier -> per-provider canonical model mapping
# ---------------------------------------------------------------------------
# Each call site requests its ``model_config`` tier model. BYOK collapses the
# tiers into two and maps each to the chosen provider's model: "large"
# (GENERATION_LARGE, GENERATION_GUI, REASONING, VISION) and "small"
# (GENERATION_SMALL edits and the CLASSIFIER tier).
#
# A model the user explicitly chose (``user_api_model``) is used for every call,
# both tiers. Without one, ``small`` uses the provider's cheap sibling to keep
# small edits and routing / repair calls inexpensive on the user's key.
#
# ``vision`` is where a call with images (file conversion, a GUI mockup) runs:
# - USER_MODEL: every current chat model of the provider reads images (OpenAI
#   gpt-4o and later, Claude 4.x/5.x), so the call keeps the user's pick, else
#   its tier model;
# - a model id: the user's pick may be text-only (Mistral's codestral), so the
#   images go to this model, mistral-large-latest (Mistral Large 3 or 4, both
#   multimodal per Mistral's docs);
# - none: no image model is known (Nebius's Qwen3-30B-A3B is text-only), and
#   neither for a catalog gateway or a custom endpoint: the server key runs it.
USER_MODEL = "user"

_PROVIDER_TIER_MODELS = {
    "openai":    {"large": "gpt-6-sol",            "small": "gpt-6-luna",
                  "vision": USER_MODEL},
    "anthropic": {"large": "claude-sonnet-5-5",    "small": "claude-haiku-5-5",
                  "vision": USER_MODEL},
    "mistral":   {"large": "mistral-large-latest", "small": "mistral-small-latest",
                  "vision": "mistral-large-latest"},
    # One small-activation MoE serves both tiers (BESSER's Nebius default).
    "nebius":    {"large": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                  "small": "Qwen/Qwen3-30B-A3B-Instruct-2507"},
}


def _image_model(
    provider: str, base_url: Optional[str], requested_model: Optional[str], user_model: Optional[str],
) -> Optional[str]:
    """The model a call with images requesting *requested_model* runs on, or
    ``None`` when the key cannot take it (see ``vision`` above)."""
    if base_url:
        return None
    vision = _PROVIDER_TIER_MODELS.get(provider, {}).get("vision")
    if vision == USER_MODEL:
        return resolve_model(provider, requested_model, user_model)
    return vision


def _tier_of(requested_model: Optional[str]) -> str:
    """Bucket a requested model into a BYOK tier by the ``model_config`` tier it
    names. ``None``/empty is the instance default, the CLASSIFIER tier. A large
    tier wins when two tiers share a model; a name that is no tier's falls back
    to "large" for gpt-5+ / o-series reasoning models, else "small"."""
    m = (requested_model or "").strip().lower()
    if not m:
        return "small"
    large = (MODEL_GENERATION_LARGE, MODEL_GENERATION_GUI, MODEL_REASONING, MODEL_VISION)
    if m in {t.lower() for t in large}:
        return "large"
    if m in {MODEL_GENERATION_SMALL.lower(), MODEL_CLASSIFIER.lower()}:
        return "small"
    if is_openai_reasoning_model(m):
        return "large"
    return "small"


def resolve_model(
    provider: str,
    requested_model: Optional[str],
    user_model: Optional[str],
) -> str:
    """Map the agent's per-call (OpenAI) model request to a concrete model
    name for *provider*. A model the user picked is used for every call; the
    tier defaults apply only when they picked none."""
    chosen = (user_model or "").strip()
    if chosen:
        return chosen
    table = _PROVIDER_TIER_MODELS.get(provider, _PROVIDER_TIER_MODELS["openai"])
    return table[_tier_of(requested_model)]


# ---------------------------------------------------------------------------
# Per-request config + context variable
# ---------------------------------------------------------------------------

def _allow_custom_base_url() -> bool:
    """Whether a per-request BYOK ``base_url`` (the local provider) is allowed.

    OFF by default: having the agent open an arbitrary user-supplied URL is an
    SSRF surface on a shared host. Local / on-prem deploys opt in via
    ``BESSER_AGENT_ALLOW_CUSTOM_BASE_URL``. A gateway from the model catalog
    (e.g. PIA) does not need it: its endpoint is the operator's.
    """
    return AgentSettings().allow_custom_base_url


def set_current(
    provider: Optional[str],
    api_key: Optional[str],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> "contextvars.Token":
    """Set the per-request BYOK config from raw session values.

    A missing key/provider (or an unsupported provider) clears BYOK so the
    request falls back to the shared server LLM. Returns the token to pass
    to :func:`reset_current`.
    """
    provider_norm = (provider or "").strip().lower()
    key = (api_key or "").strip()
    base = (base_url or "").strip() or None
    gateway = load_model_catalog().base_url(provider_norm)
    if key and gateway:
        # A gateway of the deployment's model catalog (e.g. 'pia'): the
        # operator's endpoint, so no SSRF gate and any request URL is ignored.
        cfg = BYOKConfig(
            provider="openai", api_key=key,
            model=(model or "").strip() or None, base_url=gateway,
        )
        logger.info("BYOK active for this request (gateway %s): %s",
                    provider_norm, cfg.redacted())
        return current_byok.set(cfg)
    if not key or provider_norm not in SUPPORTED_PROVIDERS:
        if key and provider_norm and provider_norm not in SUPPORTED_PROVIDERS:
            logger.warning("BYOK: ignoring unsupported provider %r", provider_norm)
        return current_byok.set(None)
    # SSRF gate: a custom base_url (local) is only honoured when the deploy
    # opts in. Otherwise disable BYOK entirely — a local key against the real
    # OpenAI endpoint would fail anyway, so falling back to the shared LLM is the
    # safe, sensible behaviour on a hosted box.
    if base and not _allow_custom_base_url():
        logger.warning(
            "BYOK: custom base_url disabled on this deployment; falling back to "
            "the shared LLM. Set BESSER_AGENT_ALLOW_CUSTOM_BASE_URL to enable."
        )
        return current_byok.set(None)
    cfg = BYOKConfig(
        provider=provider_norm,
        api_key=key,
        model=(model or "").strip() or None,
        base_url=base,
    )
    logger.info("BYOK active for this request: %s", cfg.redacted())
    return current_byok.set(cfg)


def reset_current(token: "contextvars.Token") -> None:
    """Restore the previous BYOK config (best effort)."""
    try:
        current_byok.reset(token)
    except Exception:  # pragma: no cover - defensive
        pass


def active_vision_model() -> Optional[str]:
    """The model the active key's vision call runs on, or ``None`` (no key, or
    one that cannot read images: vision then runs on the server key)."""
    cfg = current_byok.get()
    return None if cfg is None else _image_model(cfg.provider, cfg.base_url, MODEL_VISION, cfg.model)


def is_active() -> bool:
    """True when a BYOK key is active for the current request/context.

    Downstream callers (e.g. the shared-limit message) can use this to know
    whether the user's own key is in play. The user's key/provider/model
    are stored on the BAF session under ``user_api_key`` /
    ``user_api_provider`` / ``user_api_model``.
    """
    return current_byok.get() is not None


def rejected_user_key(error: Optional[BaseException]) -> bool:
    """True when *error* is a provider refusing this request's BYOK key. With
    no key active, the refused credentials are the server's own: an outage."""
    return is_active() and is_key_rejection(error)


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

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.provider = (provider or "").strip().lower()
        self._user_model = (model or "").strip() or None
        self._base_url = base_url
        if self.provider not in SUPPORTED_PROVIDERS:
            raise BYOKError(f"Unsupported BYOK provider: {provider!r}")

        if self.provider in ("openai", "mistral", "nebius"):
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - openai is a hard dep
                raise BYOKError(
                    "The 'openai' SDK is required for OpenAI/Mistral BYOK."
                ) from exc
            client_kwargs = {"api_key": api_key, "timeout": _SDK_TIMEOUT_SECONDS}
            if base_url:
                # A catalog gateway (e.g. PIA) or the 'local' provider: a
                # custom OpenAI-compatible endpoint.
                client_kwargs["base_url"] = base_url
            elif self.provider == "mistral":
                client_kwargs["base_url"] = MISTRAL_BASE_URL
            elif self.provider == "nebius":
                client_kwargs["base_url"] = NEBIUS_BASE_URL
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

    @property
    def openai_client(self):
        """The OpenAI SDK client for structured ``.parse()`` calls, or ``None``.

        Only the official OpenAI endpoint is known to support Structured
        Outputs; Anthropic, Mistral and custom gateways use JSON mode instead.
        """
        if self.provider == "openai" and not self._base_url:
            return self._client
        return None

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
        system: Optional[str] = None,
        images: Sequence[str] = (),
    ) -> str:
        """Single free-text chat-completion call, returning the text.

        ``model`` is the agent's per-call (OpenAI-canonical) tier request;
        it is mapped to this provider's equivalent via :func:`resolve_model`.
        ``max_tokens`` overrides the default completion cap (used by the GUI
        complete-system path to keep large multi-page JSON from truncating).
        ``system`` is a static instruction prefix: a cached system block on
        Anthropic, prepended to the prompt elsewhere. ``images`` are data
        URLs sent with the prompt (see ``utilities.image_inputs``), on the
        model the ``vision`` rule names (callers check :attr:`vision_model`
        is set).
        """
        if images:
            target = _image_model(self.provider, self._base_url, model, self._user_model)
        else:
            target = resolve_model(self.provider, model, self._user_model)
        return self._call(
            prompt, target, json_mode, temperature, reasoning_effort, max_tokens, system, images,
        )

    @property
    def vision_model(self) -> Optional[str]:
        """The model a file-conversion vision call runs on, or ``None`` when
        this key cannot read images (see :func:`_image_model`)."""
        return _image_model(self.provider, self._base_url, MODEL_VISION, self._user_model)

    def vision(
        self,
        prompt: str,
        images: Sequence[str],
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """One JSON-mode image call on :attr:`vision_model` (callers check it is set)."""
        return self._call(
            prompt, self.vision_model, True, temperature, None, max_tokens, None, images, timeout,
        )

    def _call(
        self,
        prompt: str,
        target: str,
        json_mode: bool,
        temperature: Optional[float],
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        system: Optional[str],
        images: Sequence[str],
        timeout: Optional[float] = None,
    ) -> str:
        temp = LLM_TEMPERATURE if temperature is None else temperature
        cap = max_tokens or LLM_MAX_TOKENS_LARGE
        if self.provider == "anthropic":
            return self._anthropic_call(prompt, target, json_mode, cap, system, images, timeout)
        if system:
            prompt = f"{system}\n{prompt}"
        return self._openai_call(
            prompt, target, json_mode, temp, reasoning_effort, cap, images, timeout,
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
        images: Sequence[str] = (),
        timeout: Optional[float] = None,
    ) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": openai_content(prompt, images)}],
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        # Mistral and Nebius use ``max_tokens``; OpenAI uses ``max_completion_tokens``.
        if self.provider in ("mistral", "nebius"):
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_completion_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # gpt-5* / o-series reject a custom temperature; cap reasoning instead.
        if supports_custom_temperature(model):
            kwargs["temperature"] = temperature
        else:
            # A caller's effort only applies to models that accept the parameter.
            effort = reasoning_effort_for(model) and (reasoning_effort or reasoning_effort_for(model))
            if effort:
                kwargs["reasoning_effort"] = effort
        completion = self._client.chat.completions.create(**kwargs)
        self._track_openai(getattr(completion, "usage", None), model)
        if not completion.choices:
            return ""
        from common.errors import raise_if_openai_refusal
        raise_if_openai_refusal(completion.choices[0])
        return completion.choices[0].message.content or ""

    def _anthropic_call(
        self,
        prompt: str,
        model: str,
        json_mode: bool,
        max_tokens: int = LLM_MAX_TOKENS_LARGE,
        system: Optional[str] = None,
        images: Sequence[str] = (),
        timeout: Optional[float] = None,
    ) -> str:
        content = prompt
        if json_mode:
            content = (
                prompt
                + "\n\nReturn ONLY valid JSON. No markdown code fences, no prose."
            )
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": anthropic_content(content, images)}],
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if system:
            # Static prefix first, cached: later calls bill it at the cache-read rate.
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]
        # No sampling params on any model: the SDK dropped them in 1.0, and API
        # version 2026-10-21 (sent by SDK 2.x) answers them with a 400.
        if anthropic_effort(model):
            # Sonnet 5 & co. reject sampling params; cap adaptive thinking instead.
            kwargs["extra_body"] = {"output_config": {"effort": anthropic_effort(model)}}
        message = self._client.messages.create(**kwargs)
        self._track_anthropic(getattr(message, "usage", None), model)
        from common.errors import raise_if_anthropic_refusal
        raise_if_anthropic_refusal(message)
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(message, "content", []) or []
            if getattr(block, "type", None) == "text"
        )
        if json_mode:
            text = _strip_code_fences(text)
        return text

    # -- best-effort token tracking ----------------------------------------

    @staticmethod
    def _track_openai(usage, model: str) -> None:
        if usage is None:
            return
        try:
            from llm.tracking import get_tracker

            get_tracker().record_from_usage(usage, model=model)
        except Exception as exc:  # pragma: no cover - tracking is best effort
            logger.debug("BYOK token tracking failed (best-effort): %s", exc)

    @staticmethod
    def _track_anthropic(usage, model: str) -> None:
        if usage is None:
            return
        try:
            from llm.tracking import get_tracker

            # input_tokens excludes the cache reads and writes, which bill separately.
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            get_tracker().record(
                prompt_tokens=(getattr(usage, "input_tokens", 0) or 0) + cache_read + cache_write,
                completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                model=model,
                cached_prompt_tokens=cache_read,
                cache_write_tokens=cache_write,
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
    return BYOKClient(cfg.provider, cfg.api_key, cfg.model, cfg.base_url)
