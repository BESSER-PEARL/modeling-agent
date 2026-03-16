"""
Token & Cost Tracker
--------------------
Thread-safe singleton that accumulates token usage and estimated cost
across all LLM calls.  Supports per-session and global aggregation.

Usage::

    from tracking import get_tracker

    tracker = get_tracker()
    tracker.record(prompt_tokens=120, completion_tokens=80, model="gpt-4.1-mini")

    print(tracker.summary())           # global totals
    print(tracker.session_summary(sid)) # per-session totals
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table (USD per 1 000 tokens) — updated for gpt-4.1-mini pricing
# ---------------------------------------------------------------------------

_COST_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
    "gpt-4.1": {"prompt": 0.002, "completion": 0.008},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
}

# Fallback for unknown models
_DEFAULT_COST = {"prompt": 0.001, "completion": 0.004}


@dataclass
class _UsageBucket:
    """Accumulates token counts and cost for a scope (global or session)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    call_count: int = 0
    cache_hits: int = 0
    first_call_ts: float = field(default_factory=time.time)
    last_call_ts: float = field(default_factory=time.time)


class TokenTracker:
    """Thread-safe token/cost tracker with per-session granularity."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._global = _UsageBucket()
        self._sessions: Dict[str, _UsageBucket] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "gpt-4.1-mini",
        session_id: Optional[str] = None,
        cached: bool = False,
    ) -> None:
        """Record a single LLM call's token usage."""
        total = prompt_tokens + completion_tokens
        cost_table = _COST_PER_1K.get(model, _DEFAULT_COST)
        cost = (
            (prompt_tokens / 1000) * cost_table["prompt"]
            + (completion_tokens / 1000) * cost_table["completion"]
        )

        now = time.time()

        with self._lock:
            # Global bucket
            self._global.prompt_tokens += prompt_tokens
            self._global.completion_tokens += completion_tokens
            self._global.total_tokens += total
            self._global.estimated_cost_usd += cost
            self._global.call_count += 1
            self._global.last_call_ts = now
            if cached:
                self._global.cache_hits += 1

            # Per-session bucket
            if session_id:
                bucket = self._sessions.setdefault(session_id, _UsageBucket())
                bucket.prompt_tokens += prompt_tokens
                bucket.completion_tokens += completion_tokens
                bucket.total_tokens += total
                bucket.estimated_cost_usd += cost
                bucket.call_count += 1
                bucket.last_call_ts = now
                if cached:
                    bucket.cache_hits += 1

        logger.debug(
            f"[TokenTracker] +{total} tokens (p={prompt_tokens}, c={completion_tokens}) "
            f"${cost:.6f} model={model} session={session_id or 'global'}"
        )

    def record_from_usage(
        self,
        usage,
        model: str = "gpt-4.1-mini",
        session_id: Optional[str] = None,
    ) -> None:
        """Record from an OpenAI ``CompletionUsage`` object (or any obj with
        ``prompt_tokens`` and ``completion_tokens`` attributes)."""
        if usage is None:
            return
        self.record(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=model,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return global usage summary."""
        with self._lock:
            g = self._global
            return {
                "prompt_tokens": g.prompt_tokens,
                "completion_tokens": g.completion_tokens,
                "total_tokens": g.total_tokens,
                "estimated_cost_usd": round(g.estimated_cost_usd, 6),
                "call_count": g.call_count,
                "cache_hits": g.cache_hits,
                "active_sessions": len(self._sessions),
            }

    def session_summary(self, session_id: str) -> Dict:
        """Return usage summary for a specific session."""
        with self._lock:
            bucket = self._sessions.get(session_id)
            if not bucket:
                return {"session_id": session_id, "total_tokens": 0, "call_count": 0}
            return {
                "session_id": session_id,
                "prompt_tokens": bucket.prompt_tokens,
                "completion_tokens": bucket.completion_tokens,
                "total_tokens": bucket.total_tokens,
                "estimated_cost_usd": round(bucket.estimated_cost_usd, 6),
                "call_count": bucket.call_count,
                "cache_hits": bucket.cache_hits,
            }

    def reset_session(self, session_id: str) -> None:
        """Clear usage data for a session (e.g. on disconnect)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def reset_all(self) -> None:
        """Clear all usage data."""
        with self._lock:
            self._global = _UsageBucket()
            self._sessions.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker: Optional[TokenTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> TokenTracker:
    """Return the global TokenTracker singleton (created on first call)."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = TokenTracker()
    return _tracker
