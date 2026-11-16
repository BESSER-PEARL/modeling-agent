"""The LLM usage of the turn being handled, reported to the editor as ``llmUsage``.

Every LLM call records its tokens through ``TokenTracker.record``; while a
turn is open in the current context, that also adds them to the turn's
:class:`TurnUsage`. A turn is opened per incoming event
(``session_helpers.begin_turn_usage``, from the first transition hook and the
state preamble) in the context BAF runs that event's transitions and state
body in; the thread pool of a parallel plan copies that context, so its calls
count too. The reply choke point (``session_helpers._send_to_session``) stamps
the running totals on the turn's terminal replies.

Totals are only as exact as what is recorded: a call through BAF's own
``predict`` has no usage, so its tokens are the tracker's estimate.
"""

import contextvars
import threading
from typing import Any, Dict, Optional, Tuple

from llm.byok_config import get_current

# The provider of a call on the server's shared client (the one SDK it uses).
SERVER_PROVIDER = "openai"


class TurnUsage:
    """Token totals of one turn's LLM calls, per (provider, model)."""

    def __init__(self, key: Any) -> None:
        self.key = key
        self._lock = threading.Lock()
        # (provider, model) -> [input tokens, output tokens], in first-call order.
        self._per_model: Dict[Tuple[str, str], list] = {}

    def add(self, input_tokens: int, output_tokens: int, *, model: str, provider: str) -> None:
        with self._lock:
            totals = self._per_model.setdefault((provider, model), [0, 0])
            totals[0] += max(int(input_tokens or 0), 0)
            totals[1] += max(int(output_tokens or 0), 0)

    def snapshot(self) -> Optional[Dict[str, Any]]:
        """``{"model", "provider", "inputTokens", "outputTokens"}``: the totals
        of every call so far, named after the model that used the most tokens
        (the first one on a tie); None before the turn's first call."""
        with self._lock:
            if not self._per_model:
                return None
            (provider, model), _ = max(
                self._per_model.items(), key=lambda item: item[1][0] + item[1][1],
            )
            return {
                "model": model,
                "provider": provider,
                "inputTokens": sum(t[0] for t in self._per_model.values()),
                "outputTokens": sum(t[1] for t in self._per_model.values()),
            }


_current: "contextvars.ContextVar[Optional[TurnUsage]]" = contextvars.ContextVar(
    "turn_usage", default=None,
)


def begin_turn(key: Any) -> TurnUsage:
    """Open the turn *key* in the current context, or return it when it is
    already the open one (the hook and the state body of one event)."""
    usage = _current.get()
    if usage is None or usage.key != key:
        usage = TurnUsage(key)
        _current.set(usage)
    return usage


def current_turn() -> Optional[TurnUsage]:
    return _current.get()


def end_turn() -> None:
    """Close the open turn in the current context (tests)."""
    _current.set(None)


def record_call(input_tokens: int, output_tokens: int, model: str) -> None:
    """Add one call to the open turn, if any: on the user's provider when a
    BYOK key is active, else on the server's."""
    usage = _current.get()
    if usage is None:
        return
    cfg = get_current()
    usage.add(input_tokens, output_tokens, model=model,
              provider=cfg.provider if cfg is not None else SERVER_PROVIDER)
