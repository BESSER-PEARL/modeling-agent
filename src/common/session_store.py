"""The registry of the per-session stores that expire their own entries.

A module-level store keyed by a session key is either bounded (an LRU, like the
reply buffers in ``conversation.session_helpers``) or registers its expiry
here; the session reaper runs every registered expiry through
``sessions.expire()``.

The stores key on the conversation key (``memory.memory_session_key``: the
payload ``sessionId``, else the BAF session id), which outlives a BAF session,
so a BAF session close drops nothing from them. BAF's own session state (its
session variables, including every key in ``common.session_keys``) belongs to
BAF and is freed by ``Agent.close_session``; it is not registered here.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


class SessionStore:
    """Per-session store expiries by name; one ``expire`` runs them all."""

    def __init__(self) -> None:
        self._expiries: Dict[str, Callable[[float], object]] = {}

    def register(self, name: str, *, expire: Callable[[float], object]) -> None:
        """Register a store's expiry; registering a name again replaces it (module reload)."""
        self._expiries[name] = expire

    @property
    def names(self) -> List[str]:
        return sorted(self._expiries)

    def expire(self, max_age_seconds: float) -> None:
        """Run each store's expiry; a failing store is logged and skipped."""
        for name, expire in self._expiries.items():
            try:
                expire(max_age_seconds)
            except Exception as exc:
                logger.warning("[SessionStore] %s: expiry failed: %s", name, exc)


sessions = SessionStore()
