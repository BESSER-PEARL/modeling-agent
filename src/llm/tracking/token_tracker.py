"""
Token & Cost Tracker
--------------------
Thread-safe singleton that accumulates token usage and estimated cost
across all LLM calls.  Supports per-session and global aggregation.

Usage::

    from llm.tracking import get_tracker

    tracker = get_tracker()
    tracker.record(prompt_tokens=120, completion_tokens=80, model="gpt-6-luna")

    print(tracker.summary())           # global totals
    print(tracker.session_summary(sid)) # per-session totals
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from common.settings import AgentSettings
from llm.byok_config import get_current
from llm.model_catalog import load_model_catalog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prices
#
# A call is priced from the deployment's model catalog (a gateway bills its own
# rates), else from the vendor price file, else from _UNPUBLISHED. The price
# file (data/, package data) is a copy of the one BESSER vendors
# (besser/llm/data/model_prices.json: litellm's
# model_prices_and_context_window.json reduced to the vendors we call), so the
# agent and the backend price a model the same; refresh both together
# (tests/test_token_tracker.py compares them when ../BESSER is checked out). A model
# none of them prices records tokens at _DEFAULT_COST, a placeholder, and is
# reported once via _warn_unknown_model.
# ---------------------------------------------------------------------------

_PRICES_PATH = Path(__file__).with_name("data") / "model_prices.json"

# BESSER's lookup order: one bare id can be listed under several vendors at
# different rates, so only the vendors we call are tried.
_PRICE_LOOKUP_PREFIXES = ("", "anthropic/", "openai/", "mistral/", "nebius/")

# USD per 1M tokens for models the price file does not list yet, at the rates
# BESSER bills them. Delete a row once a refresh of the price file adds it.
_UNPUBLISHED: Dict[str, Dict[str, float]] = {
    # Cached rates assume the 10 % gpt-6-astra lists.
    "gpt-6-sol": {"input": 2.0, "output": 10.0, "cache_read": 0.2},
    "gpt-6-luna": {"input": 0.1, "output": 0.5, "cache_read": 0.01},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.2, "cache_write": 5.0},
    # BESSER's default model catalog rates (besser/llm/data/model_catalog.json).
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5},
    "claude-haiku-5-5": {"input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125},
    "gpt-6.1-sol": {"input": 2.0, "output": 10.0},
}

# Anthropic bills a 5-minute cache write at 1.25x the input rate.
_CACHE_WRITE_MULTIPLIER = 1.25

# Gateway prefixes in front of a canonical id (Bedrock "us.anthropic.", "openai/").
_GATEWAY_PREFIX = re.compile(r"^(?:[a-z]{2}\.)?(?:anthropic|openai)[./]")
# Snapshot suffixes after a canonical id: a date, a Vertex "@..." tag, a Bedrock "-v1:0".
_SNAPSHOT_SUFFIX = re.compile(r"^(?:-\d{4}-?\d{2}-?\d{2}|@[\w.-]+)?(?:-v\d+(?::\d+)?)?$")


@lru_cache(maxsize=1)
def _price_file() -> Dict[str, dict]:
    """The price file keyed by lower-cased id. Empty if unreadable, so every
    model falls to the placeholder and its warning rather than reading as free."""
    try:
        with open(_PRICES_PATH, encoding="utf-8") as handle:
            return {key.lower(): row for key, row in json.load(handle).items()}
    except (OSError, ValueError):
        logger.warning("Model price file unreadable at %s", _PRICES_PATH)
        return {}


def _per_1k(rates: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Per-1M ``input``/``output``/``cache_read``/``cache_write`` rates as the
    per-1K table ``record`` reads; a missing cache rate keeps its default."""
    table = {"prompt": rates["input"] / 1000, "completion": rates["output"] / 1000}
    for key, name in (("cache_read", "cached"), ("cache_write", "cache_write")):
        if rates.get(key) is not None:
            table[name] = rates[key] / 1000
    return table


def _per_million(per_token: Optional[float]) -> Optional[float]:
    return None if per_token is None else per_token * 1_000_000


def _listed_cost_table(model: str) -> Optional[Dict[str, float]]:
    """Rates for exactly *model* (lower-cased) from the price file or _UNPUBLISHED."""
    table = _price_file()
    for prefix in _PRICE_LOOKUP_PREFIXES:
        row = table.get(prefix + model)
        # Image and embedding rows list no output price, so no chat rate either.
        if row and None not in (row.get("input_cost_per_token"), row.get("output_cost_per_token")):
            return _per_1k({
                "input": _per_million(row["input_cost_per_token"]),
                "output": _per_million(row["output_cost_per_token"]),
                "cache_read": _per_million(row.get("cache_read_input_token_cost")),
                "cache_write": _per_million(row.get("cache_creation_input_token_cost")),
            })
    rates = _UNPUBLISHED.get(model)
    return _per_1k(rates) if rates else None


@lru_cache(maxsize=256)
def _vendor_cost_table(model: str) -> Optional[Dict[str, float]]:
    """Vendor list rates for *model*: the exact id, else the longest listed id
    it extends by a gateway prefix and/or a snapshot suffix
    (claude-opus-5-5-20261101, us.anthropic.claude-opus-5-5). A different
    variant (gpt-6-sol-pro) stays unknown rather than inheriting its base
    model's price."""
    name = (model or "").strip().lower()
    exact = _listed_cost_table(name)
    if exact is not None:
        return exact
    base = _GATEWAY_PREFIX.sub("", name)
    for cut in range(len(base), 0, -1):
        if _SNAPSHOT_SUFFIX.match(base[cut:]):
            table = _listed_cost_table(base[:cut])
            if table is not None:
                return table
    return None


def _current_route() -> Optional[str]:
    """The endpoint this call goes to: the user's BYOK endpoint, else the
    shared client's (``OPENAI_BASE_URL``); None for a vendor's own API."""
    cfg = get_current()
    if cfg is not None:
        return cfg.base_url
    return AgentSettings().openai_base_url


def _catalog_cost_table(model: str) -> Optional[Dict[str, float]]:
    """The model catalog's rate for *model* on the current route, per 1K."""
    rates = load_model_catalog().price(model, _current_route())
    return None if rates is None else _per_1k(rates)


def _cost_table_for(model: str) -> Optional[Dict[str, float]]:
    """Rates for *model*: the model catalog's for the current route, else the
    vendor list rates."""
    catalog = _catalog_cost_table(model)
    if catalog is not None:
        return catalog
    return _vendor_cost_table(model)


# Seconds between INFO lines with the global totals (0 disables them).
_SUMMARY_INTERVAL_SECONDS = float(AgentSettings().cost_log_interval or 0)

# Fallback for unknown models — placeholder pricing, not any real model's rate.
_DEFAULT_COST = {"prompt": 0.001, "completion": 0.004}

# Models already reported as unpriced, so the warning is
# emitted once per model rather than once per LLM call.
_unknown_models_seen: set = set()
_unknown_models_lock = threading.Lock()


def _warn_unknown_model(model: str) -> None:
    """Log once that `model` has no price and is using placeholder pricing."""
    with _unknown_models_lock:
        if model in _unknown_models_seen:
            return
        _unknown_models_seen.add(model)
    logger.warning(
        "No price for model %r; reported costs for it are estimated from placeholder "
        "pricing (%s/1K prompt, %s/1K completion). Price it in the model catalog "
        "(BESSER_MODEL_CATALOG).",
        model, _DEFAULT_COST["prompt"], _DEFAULT_COST["completion"],
    )


@dataclass
class _UsageBucket:
    """Accumulates token counts and cost for a scope (global or session)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
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
        self._last_summary_ts = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        model: str,
        session_id: Optional[str] = None,
        cached: bool = False,
        cached_prompt_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Record a single LLM call's token usage.

        ``prompt_tokens`` counts every input token; ``cached_prompt_tokens`` and
        ``cache_write_tokens`` are the parts of it read from / written to a
        prompt cache, billed at the cached and cache-write rates.
        """
        total = prompt_tokens + completion_tokens
        cost_table = _cost_table_for(model)
        if cost_table is None:
            _warn_unknown_model(model)
            cost_table = _DEFAULT_COST
        uncached = max(prompt_tokens - cached_prompt_tokens - cache_write_tokens, 0)
        cost = (
            (uncached / 1000) * cost_table["prompt"]
            + (cached_prompt_tokens / 1000) * cost_table.get("cached", cost_table["prompt"])
            + (cache_write_tokens / 1000) * cost_table.get(
                "cache_write", cost_table["prompt"] * _CACHE_WRITE_MULTIPLIER)
            + (completion_tokens / 1000) * cost_table["completion"]
        )

        now = time.time()

        with self._lock:
            # Global bucket
            self._global.prompt_tokens += prompt_tokens
            self._global.completion_tokens += completion_tokens
            self._global.total_tokens += total
            self._global.cached_prompt_tokens += cached_prompt_tokens
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
                bucket.cached_prompt_tokens += cached_prompt_tokens
                bucket.estimated_cost_usd += cost
                bucket.call_count += 1
                bucket.last_call_ts = now
                if cached:
                    bucket.cache_hits += 1

            log_summary = (
                _SUMMARY_INTERVAL_SECONDS > 0
                and now - self._last_summary_ts >= _SUMMARY_INTERVAL_SECONDS
            )
            if log_summary:
                self._last_summary_ts = now

        if log_summary:
            g = self.summary()
            logger.info(
                "[TokenTracker] totals since start: calls=%d prompt=%d (cached=%d) "
                "completion=%d est_cost=$%.4f",
                g["call_count"], g["prompt_tokens"], g["cached_prompt_tokens"],
                g["completion_tokens"], g["estimated_cost_usd"],
            )

        logger.debug(
            f"[TokenTracker] +{total} tokens (p={prompt_tokens}, c={completion_tokens}) "
            f"${cost:.6f} model={model} session={session_id or 'global'}"
        )

    def record_from_usage(
        self,
        usage,
        model: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Record from an OpenAI ``CompletionUsage`` object (or any obj with
        ``prompt_tokens`` and ``completion_tokens`` attributes)."""
        if usage is None:
            return
        details = getattr(usage, "prompt_tokens_details", None)
        self.record(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=model,
            session_id=session_id,
            cached_prompt_tokens=getattr(details, "cached_tokens", 0) or 0,
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
                "cached_prompt_tokens": g.cached_prompt_tokens,
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
