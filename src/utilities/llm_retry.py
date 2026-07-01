"""
LLM Retry
---------
Retry-with-exponential-backoff for the *shared* server LLM's raw SDK call.

Background: a 1544-question QA run at high concurrency showed ~26% of
requests failing with a user-facing "AI service temporarily unavailable" /
"couldn't process that modification" error. Almost all of those were
transient upstream failures (OpenAI 429 rate-limits, timeouts, 5xx) under
load, not logic bugs — the handlers were correctly surfacing the first
exception the SDK raised. This module lets the SDK call retry a transient
failure a bounded number of times before any handler ever sees it.

Used exclusively by ``agent_setup.py``, which patches the shared OpenAI
client's ``chat.completions.create`` / ``beta.chat.completions.parse``
methods with :func:`with_retry` right after BESSER's ``LLM.initialize()``
creates the client. That is the single choke point every caller
(BAF's ``.predict``/``.chat``/intent classification, ``llm.provider``'s
structured-output ``.parse()`` and streaming ``.stream()``, and this
module's own ``gpt_predict_json`` model-override branch) goes through, so
none of them need to be touched individually.

Deliberately NOT retried here:
  * Auth errors (401) / bad request (400) / other 4xx — permanent for the
    current request; retrying would only burn the backoff budget and
    delay the (correct) error.
  * BYOK's per-user client (``byok.BYOKClient``) — a different client
    object entirely, never patched by this module. A user's own key
    failing (or the shared key's retries being exhausted) must still
    reach ``model_operations.py``'s existing rate_limit/auth_error
    handling untouched, so the intentional "add your API key" BYOK
    messaging keeps working.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Backoff schedule — bounded so a live chat never stalls for long.
# ---------------------------------------------------------------------------
MAX_ATTEMPTS = 4            # 1 initial try + up to 3 retries
BASE_DELAY_SECONDS = 0.6    # first retry wait
MAX_DELAY_SECONDS = 6.0     # per-attempt delay cap
JITTER_SECONDS = 0.3        # +/- random jitter to avoid a synchronized thundering herd
# Worst case added latency: 0.6 + 1.2 + 2.4 (+ up to 0.9 jitter) ≈ 5s.


def _retryable_status_code(status_code: Any) -> bool:
    """429 (rate limit) and any 5xx are transient; other 4xx are permanent."""
    try:
        code = int(status_code)
    except (TypeError, ValueError):
        return False
    return code == 429 or code >= 500


def _sdk_exception_types() -> Tuple[Type[BaseException], ...]:
    """Best-effort import of the OpenAI/Anthropic SDK's transient exception
    classes. Imported lazily/defensively — neither SDK is guaranteed to be
    importable, and this module must not break on import if one is missing.
    """
    exc_types = []
    try:
        import openai
        exc_types += [
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ]
    except Exception:  # pragma: no cover - defensive, SDK optional
        pass
    try:
        import anthropic
        exc_types += [
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ]
    except Exception:  # pragma: no cover - defensive, SDK optional
        pass
    return tuple(exc_types)


def _is_transient(exc: BaseException) -> bool:
    """True when *exc* looks like a transient upstream failure worth retrying.

    Retries: rate-limit (429 / ``RateLimitError``), timeouts, connection
    errors, and 5xx / ``InternalServerError``.
    Never retries: auth errors (401), bad-request (400), or any other 4xx —
    and never a successful call, since this is only consulted on exception.
    """
    if isinstance(exc, _sdk_exception_types()):
        return True

    # Generic APIStatusError (either SDK) not covered by the explicit list
    # above (e.g. a future subclass): fall back to the HTTP status code.
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return _retryable_status_code(status_code)

    # Bare socket/timeout errors that don't subclass the SDK hierarchy
    # (e.g. raw httpx/requests errors bubbling through).
    err_name = type(exc).__name__.lower()
    if "timeout" in err_name or "connection" in err_name:
        return True

    return False


def with_retry(func: F, *, label: str = "llm_call") -> F:
    """Wrap *func* so transient upstream failures retry with backoff.

    Returns a new callable with the same signature that behaves exactly
    like *func* on success or on a non-transient error (raised immediately,
    first attempt) — the only difference is observable on a transient
    failure, where it is retried up to :data:`MAX_ATTEMPTS` times.
    """

    @functools.wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            attempt += 1
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if attempt >= MAX_ATTEMPTS or not _is_transient(exc):
                    raise
                delay = min(MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
                delay += random.uniform(0, JITTER_SECONDS)
                logger.warning(
                    "%s: transient %s on attempt %d/%d (%s) — retrying in %.2fs",
                    label, type(exc).__name__, attempt, MAX_ATTEMPTS, exc, delay,
                )
                time.sleep(delay)

    return _wrapped  # type: ignore[return-value]


def patch_openai_client_for_retry(client: Any, *, label: str = "llm") -> None:
    """Patch an OpenAI-SDK-compatible ``client``'s network-call methods
    in place so every caller sharing this client benefits transparently.

    Patches ``client.chat.completions.create`` (used by BAF's
    ``predict``/``chat``/``intent_classification``, ``llm.provider``'s
    ``predict``/``stream``, and ``gpt_predict_json``'s model-override
    branch) and ``client.beta.chat.completions.parse`` (used by
    ``llm.provider``'s structured-output ``parse``), when present.

    Idempotent: safe to call more than once on the same client (e.g. if
    the framework re-initializes the LLM) — already-wrapped methods are
    left alone instead of being wrapped again.
    """
    completions = getattr(getattr(client, "chat", None), "completions", None)
    if completions is not None and hasattr(completions, "create"):
        if not getattr(completions.create, "_llm_retry_wrapped", False):
            wrapped = with_retry(completions.create, label=f"{label}.chat.completions.create")
            wrapped._llm_retry_wrapped = True
            completions.create = wrapped

    beta_completions = getattr(getattr(getattr(client, "beta", None), "chat", None), "completions", None)
    if beta_completions is not None and hasattr(beta_completions, "parse"):
        if not getattr(beta_completions.parse, "_llm_retry_wrapped", False):
            wrapped_parse = with_retry(beta_completions.parse, label=f"{label}.beta.chat.completions.parse")
            wrapped_parse._llm_retry_wrapped = True
            beta_completions.parse = wrapped_parse
