"""
LLM Provider
------------
The agent's one door to the model SDKs. Every LLM call outside BAF's own
RAG / intent classification goes through :class:`LLMProvider`:

* ``complete`` — free text or JSON mode, optionally with images,
* ``parse`` — one structured-output attempt into a Pydantic model,
* ``stream`` — a reply as text chunks,
* ``vision`` — the file-conversion image / PDF call,
* ``predict`` — BAF's ``predict`` on the wrapped instance (its own model and
  constructor parameters).

Every method but ``predict`` resolves the request's BYOK key first
(``byok.get_active_client``) and otherwise uses the wrapped BAF ``LLMOpenAI``
instance's shared client, whose network calls ``app.agent_setup`` patches
with ``llm.retry`` (BYOK clients use the SDK's own retries). Sampling
parameters come from ``model_config``, usage goes to the token tracker with
the model actually called, and refusals raise ``ModelRefusal``.

Usage::

    from llm import LLMProvider

    provider = LLMProvider(gpt)
    text = provider.complete(prompt, model=MODEL_GENERATION_SMALL)
    result = provider.parse(messages, schema=MyPydanticModel)
"""

import json
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type

from openai import LengthFinishReasonError
from pydantic import BaseModel

from common.agent_config import (
    LLM_MAX_TOKENS_LARGE,
    LLM_MAX_TOKENS_TEXT,
    LLM_TEMPERATURE,
    LLM_TEXT_TEMPERATURE,
)
from common.errors import LLMPredictionError, raise_if_openai_refusal
from llm import byok, tracking
from llm.byok_config import get_current
from llm.model_config import (
    MODEL_CLASSIFIER,
    MODEL_VISION,
    reasoning_effort_for,
    sampling_params,
    supports_custom_temperature,
)
from utilities.image_inputs import openai_content
from utilities.json_repair import validate_llm_json

logger = logging.getLogger(__name__)

# Vision replies are whole diagram specs.
_VISION_MAX_TOKENS = 8192
_VISION_TEMPERATURE = 0.1


class OutputTruncated(LLMPredictionError):
    """A structured reply hit the completion cap (the call is still billed).
    The SDK's ``LengthFinishReasonError`` is the ``__cause__``."""


def _json_params(model: str) -> Dict[str, Any]:
    """JSON-mode parameters for BAF's ``predict`` on *model*."""
    return {
        **sampling_params(model, LLM_TEMPERATURE),
        "max_completion_tokens": LLM_MAX_TOKENS_LARGE,
        "response_format": {"type": "json_object"},
    }


def _vision_params() -> Dict[str, Any]:
    """Token/sampling params for ``MODEL_VISION``: older models (gpt-4o,
    gpt-4.1) take ``max_tokens`` and a temperature, reasoning models
    ``max_completion_tokens`` and ``reasoning_effort``."""
    if supports_custom_temperature(MODEL_VISION):
        return {"max_tokens": _VISION_MAX_TOKENS, "temperature": _VISION_TEMPERATURE}
    kwargs: Dict[str, Any] = {"max_completion_tokens": _VISION_MAX_TOKENS}
    if reasoning_effort_for(MODEL_VISION):
        kwargs["reasoning_effort"] = reasoning_effort_for(MODEL_VISION)
    return kwargs


class LLMProvider:
    """The call machinery over one BAF ``LLMOpenAI`` instance (or a test
    double exposing ``predict`` and optionally an OpenAI-shaped ``client``)."""

    def __init__(self, llm_instance: Any, model_name: Optional[str] = None) -> None:
        self._llm = llm_instance
        self._model = model_name or getattr(llm_instance, "name", MODEL_CLASSIFIER)
        self.tracker = tracking.get_tracker()

    @property
    def client(self) -> Any:
        """The shared OpenAI SDK client of the wrapped instance, or None."""
        return getattr(self._llm, 'client', None)

    @property
    def model_name(self) -> str:
        return self._model

    def _record(self, usage: Any, model: str) -> None:
        """Record a completion's real usage (best effort)."""
        if usage is None:
            return
        try:
            self.tracker.record_from_usage(usage, model=model)
        except Exception as exc:
            logger.debug(f"Token tracking failed (best-effort): {exc}")

    def _create(self, client: Any, kwargs: Dict[str, Any]) -> str:
        """One ``chat.completions.create`` call: track, check refusal, return the text."""
        completion = client.chat.completions.create(**kwargs)
        self._record(getattr(completion, "usage", None), kwargs["model"])
        if not completion.choices:
            return ""
        raise_if_openai_refusal(completion.choices[0])
        return completion.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Text and JSON
    # ------------------------------------------------------------------

    def predict(self, prompt: str, **kwargs) -> str:
        """BAF's ``predict`` (the instance's model and constructor
        parameters); BAF exposes no usage, so the tokens are estimated."""
        response = self._llm.predict(prompt, **kwargs)
        self.tracker.record(
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(response) // 4 if response else 0,
            model=self._model,
        )
        return response

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: Optional[int] = None,
        images: Sequence[str] = (),
    ) -> str:
        """One free-text (or JSON-mode) call, returning the text.

        ``model`` is a ``model_config`` tier; without one (and without
        images) the call goes through BAF's ``predict`` on the instance.
        ``images`` are data URLs sent with the prompt.
        """
        byok_client = byok.get_active_client()
        if byok_client is not None:
            return byok_client.predict_raw(
                prompt,
                model=model,
                json_mode=json_mode,
                temperature=temperature,
                reasoning_effort=reasoning_effort_for(model) if model else None,
                max_tokens=max_tokens,
                images=images,
            )
        client = self.client
        if images and not model:
            model = self._model
        if model and client is not None and hasattr(client, 'chat'):
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": openai_content(prompt, images)}],
                "max_completion_tokens": max_tokens or LLM_MAX_TOKENS_LARGE,
                **sampling_params(model, temperature),
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return self._create(client, kwargs)
        if images:
            raise LLMPredictionError("This LLM client cannot send images")
        if json_mode:
            return self.predict(prompt, parameters=_json_params(self._model))
        return self.predict(prompt)

    def vision(self, prompt: str, images: Sequence[str], *, timeout: float) -> str:
        """The file-conversion vision call (``MODEL_VISION``, JSON mode).

        The official OpenAI endpoint (server key or a user's OpenAI key) gets
        the vision request as is; any other BYOK provider or gateway gets it
        through its own client with the BYOK model mapping.
        """
        byok_client = byok.get_active_client()
        if byok_client is not None and byok_client.openai_client is None:
            return byok_client.predict_raw(
                prompt,
                model=MODEL_VISION,
                json_mode=True,
                temperature=_VISION_TEMPERATURE,
                max_tokens=_VISION_MAX_TOKENS,
                images=images,
            )
        client = byok_client.openai_client if byok_client is not None else self.client
        if client is None:
            raise LLMPredictionError("No LLM client is available for vision")
        return self._create(client, {
            "model": MODEL_VISION,
            "messages": [{"role": "user", "content": openai_content(prompt, images)}],
            "response_format": {"type": "json_object"},
            **_vision_params(),
            "timeout": timeout,
        })

    # ------------------------------------------------------------------
    # Structured outputs
    # ------------------------------------------------------------------

    def supports_structured_outputs(self) -> bool:
        """Whether :meth:`parse` uses the OpenAI Structured Outputs API (else JSON mode)."""
        byok_client = byok.get_active_client()
        if byok_client is not None:
            return byok_client.openai_client is not None
        client = self.client
        return client is not None and hasattr(client, 'beta') and hasattr(client.beta, 'chat')

    def parse(
        self,
        messages: List[Dict[str, Any]],
        schema: Type[BaseModel],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        model: Optional[str] = None,
    ) -> BaseModel:
        """One structured-output attempt, validated into *schema*.

        OpenAI's Structured Outputs API when available; otherwise JSON mode
        plus validation (the schema rides in the prompt).

        Raises:
            OutputTruncated: the reply hit ``max_tokens``.
            ModelRefusal: the model declined.
            LLMPredictionError: the reply parsed to nothing.
            ValueError: the reply had no choices.
        """
        effective_model = model or self._model
        # BYOK: a user who saved a key pays for this call too.
        byok_client = byok.get_active_client()
        if byok_client is not None:
            client = byok_client.openai_client
            if client is None:
                # Anthropic / Mistral / custom endpoint: JSON mode + validation.
                schema_note = (
                    "\n\nReturn ONLY a JSON object matching this JSON schema:\n"
                    + json.dumps(schema.model_json_schema())
                )
                system = None
                if byok_client.provider == "anthropic":
                    # Static system prompt + schema form the cached prefix,
                    # ahead of the per-message content.
                    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
                    system += schema_note
                    prompt = "\n".join(m["content"] for m in messages if m["role"] != "system")
                else:
                    prompt = "\n".join(m["content"] for m in messages) + schema_note
                raw = byok_client.predict_raw(
                    prompt, model=effective_model, json_mode=True,
                    temperature=temperature, max_tokens=max_tokens, system=system,
                )
                return validate_llm_json(schema, byok._strip_code_fences(raw))
            cfg = get_current()
            effective_model = byok.resolve_model("openai", effective_model, cfg.model if cfg else None)
        else:
            client = self.client
        if client is None or not hasattr(client, 'beta'):
            prompt = "\n".join(m["content"] for m in messages)
            return validate_llm_json(schema, self.predict(prompt))

        parse_kwargs: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "response_format": schema,
            "max_completion_tokens": max_tokens,
            **sampling_params(effective_model, temperature),
        }
        try:
            completion = client.beta.chat.completions.parse(**parse_kwargs)
        except LengthFinishReasonError as exc:
            # A truncated parse is still billed.
            self._record(exc.completion.usage, effective_model)
            raise OutputTruncated(
                f"Structured output truncated at max_tokens={max_tokens}"
            ) from exc

        usage = getattr(completion, 'usage', None)
        self._record(usage, effective_model)
        if usage is not None:
            logger.debug(
                "Structured call usage: model=%s prompt=%s completion=%s finish_reason=%s",
                effective_model, getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
                completion.choices[0].finish_reason if completion.choices else None,
            )

        if not completion.choices:
            raise ValueError("LLM returned no choices (possible content filter)")

        raise_if_openai_refusal(completion.choices[0])
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise LLMPredictionError("LLM returned empty structured output")
        return parsed

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: float = LLM_TEXT_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS_TEXT,
    ) -> Iterator[str]:
        """A reply as text chunks.

        The request is sent before this returns, so a failed call raises
        here, not mid-iteration. A BYOK reply (whose providers do not share
        OpenAI's chunk shape) and a client-less instance come as one chunk.
        Iteration raises ``ModelRefusal`` at the end when the model declined.
        """
        byok_client = byok.get_active_client()
        if byok_client is not None:
            # The free-text BYOK shape has no system/user split.
            text = byok_client.predict_raw(
                f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                model=model,
                json_mode=False,
                temperature=temperature,
            )
            return iter([text] if text else [])
        client = self.client
        if client is None or not hasattr(client, 'chat'):
            response = self.predict(prompt)
            return iter([response if isinstance(response, str) else str(response)])

        effective_model = model or self._model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        events = client.chat.completions.create(
            model=effective_model,
            messages=messages,
            max_completion_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **sampling_params(effective_model, temperature),
        )
        return self._stream_chunks(events, effective_model)

    def _stream_chunks(self, events: Any, model: str) -> Iterator[str]:
        usage = None
        refusal = ""
        finish_reason = None
        produced = False
        try:
            for event in events:
                if getattr(event, 'usage', None) is not None:
                    usage = event.usage
                if not event.choices:
                    continue
                finish_reason = getattr(event.choices[0], 'finish_reason', None) or finish_reason
                delta = event.choices[0].delta
                delta_refusal = getattr(delta, 'refusal', None)
                if isinstance(delta_refusal, str):
                    refusal += delta_refusal
                content = getattr(delta, 'content', None)
                if content:
                    produced = True
                    yield content
        finally:
            self._record(usage, model)
        if not produced:
            raise_if_openai_refusal({"message": {"refusal": refusal}, "finish_reason": finish_reason})


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_provider: Optional[LLMProvider] = None
_provider_lock = threading.Lock()


def get_provider(
    llm_instance: Any = None,
    model_name: str = MODEL_CLASSIFIER,
) -> Optional[LLMProvider]:
    """Get or create the global LLMProvider singleton (the shared ``gpt``).

    On first call, pass the LLM instance to initialize.  Subsequent calls
    return the existing provider.
    """
    global _provider
    if _provider is not None:
        return _provider
    if llm_instance is None:
        return None
    with _provider_lock:
        if _provider is None:
            _provider = LLMProvider(llm_instance, model_name)
    return _provider
