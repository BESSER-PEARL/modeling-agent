"""
LLM Provider Abstraction
-------------------------
Unified interface for accessing different LLM providers (OpenAI, Anthropic,
local models).  Wraps the BESSER framework's LLMOpenAI instance while
providing direct access to the underlying client for advanced features
like structured outputs and streaming.

Usage::

    from llm import get_provider

    provider = get_provider()
    # Structured output
    result = provider.parse(messages, schema=MyPydanticModel)
    # Streaming
    for chunk in provider.stream(messages):
        print(chunk, end="")
    # Token tracking is automatic
    print(provider.tracker.summary())
"""

import logging
import threading
from typing import Any, Dict, Iterator, List, Optional, Type

from pydantic import BaseModel

from model_config import MODEL_CLASSIFIER, reasoning_effort_for, supports_custom_temperature
from tracking import get_tracker

logger = logging.getLogger(__name__)


class LLMProvider:
    """Unified LLM provider wrapping a BESSER LLMOpenAI instance.

    Provides three access levels:
    1. ``predict(prompt)`` — Simple text prediction (uses BAF's predict)
    2. ``parse(messages, schema)`` — Structured outputs via Pydantic
    3. ``stream(messages)`` — Token-by-token streaming iterator

    All calls are automatically tracked via the TokenTracker singleton.
    """

    def __init__(self, llm_instance: Any, model_name: str = MODEL_CLASSIFIER) -> None:
        self._llm = llm_instance
        self._model = model_name
        self.tracker = get_tracker()

    @property
    def client(self) -> Any:
        """Direct access to the underlying OpenAI client."""
        return getattr(self._llm, 'client', None)

    @property
    def model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Level 1: Simple prediction
    # ------------------------------------------------------------------

    def predict(self, prompt: str, **kwargs) -> str:
        """Simple text prediction via BAF's predict method."""
        response = self._llm.predict(prompt, **kwargs)
        # Estimate tokens for tracking
        self.tracker.record(
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(response) // 4 if response else 0,
            model=self._model,
        )
        return response

    # ------------------------------------------------------------------
    # Level 2: Structured outputs
    # ------------------------------------------------------------------

    def parse(
        self,
        messages: List[Dict[str, str]],
        schema: Type[BaseModel],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        model: Optional[str] = None,
    ) -> BaseModel:
        """Parse LLM response into a validated Pydantic model.

        Uses OpenAI's structured outputs API when available, falls back to
        JSON mode + manual validation otherwise.

        Args:
            model: Optional per-call model override (see ``model_config``);
                defaults to the provider's configured model.
        """
        effective_model = model or self._model
        client = self.client
        if client is None or not hasattr(client, 'beta'):
            # Fallback: predict + parse
            prompt = "\n".join(m["content"] for m in messages)
            raw = self.predict(prompt)
            return schema.model_validate_json(raw)

        parse_kwargs: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "response_format": schema,
            "max_completion_tokens": max_tokens,
        }
        # gpt-5* / o-series models 400 on an explicit non-default
        # temperature — omit the parameter for them; cap their hidden
        # reasoning instead (quality holds, latency drops ~40%).
        if supports_custom_temperature(effective_model):
            parse_kwargs["temperature"] = temperature
        else:
            parse_kwargs["reasoning_effort"] = reasoning_effort_for(effective_model)
        completion = client.beta.chat.completions.parse(**parse_kwargs)

        if hasattr(completion, 'usage') and completion.usage:
            self.tracker.record_from_usage(completion.usage, model=effective_model)

        if not completion.choices:
            raise ValueError("LLM returned no choices (possible content filter)")

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            refusal = getattr(completion.choices[0].message, 'refusal', None)
            raise ValueError(f"LLM refused or returned empty: {refusal}")

        return parsed

    # ------------------------------------------------------------------
    # Level 3: Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream LLM response token by token.

        Yields content strings as they arrive. Tracks usage from the
        final chunk's usage stats.

        Args:
            model: Optional per-call model override (see ``model_config``).
        """
        effective_model = model or self._model
        client = self.client
        if client is None or not hasattr(client, 'chat'):
            # Fallback: predict and yield as single chunk
            prompt = "\n".join(m["content"] for m in messages)
            yield self.predict(prompt)
            return

        stream_kwargs: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # gpt-5* / o-series models 400 on an explicit non-default
        # temperature — omit the parameter for them; cap their hidden
        # reasoning instead (quality holds, latency drops ~40%).
        if supports_custom_temperature(effective_model):
            stream_kwargs["temperature"] = temperature
        else:
            stream_kwargs["reasoning_effort"] = reasoning_effort_for(effective_model)
        stream = client.chat.completions.create(**stream_kwargs)

        for event in stream:
            if hasattr(event, 'usage') and event.usage is not None:
                self.tracker.record_from_usage(event.usage, model=effective_model)

            if not event.choices:
                continue

            content = getattr(event.choices[0].delta, 'content', None)
            if content:
                yield content


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_provider: Optional[LLMProvider] = None
_provider_lock = threading.Lock()


def get_provider(
    llm_instance: Any = None,
    model_name: str = MODEL_CLASSIFIER,
) -> Optional[LLMProvider]:
    """Get or create the global LLMProvider singleton.

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
