"""The registry of every in-process store that holds per-session state.

Each module-level store keyed by a session key registers here with how to drop
one key, list its keys and, if it expires entries itself, how to run that.
``sessions.close(key)`` then drops a key from every store, and the session
reaper runs every store's own expiry through ``sessions.expire()``.

The stores key on the conversation key (``memory.memory_session_key``: the
payload ``sessionId``, else the BAF session id). BAF's own session state (its
session variables, including every key in ``common.session_keys``) belongs to
BAF and is freed by ``Agent.close_session``; it is not registered here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Store:
    discard: Callable[[str], None]
    keys: Callable[[], Iterable[str]]
    expire: Optional[Callable[[float], object]] = None


class SessionStore:
    """Per-session stores by name; one ``close`` drops a key from all."""

    def __init__(self) -> None:
        self._stores: Dict[str, _Store] = {}

    def register(
        self,
        name: str,
        *,
        discard: Callable[[str], None],
        keys: Callable[[], Iterable[str]],
        expire: Optional[Callable[[float], object]] = None,
    ) -> None:
        """Register a store; registering a name again replaces it (module reload)."""
        self._stores[name] = _Store(discard, keys, expire)

    @property
    def names(self) -> List[str]:
        return sorted(self._stores)

    def holding(self, key: str) -> List[str]:
        """Names of the stores that still hold *key*."""
        return [name for name, store in sorted(self._stores.items()) if key in set(store.keys())]

    def close(self, key: str) -> None:
        """Drop *key* from every registered store."""
        for name, store in self._stores.items():
            try:
                store.discard(key)
            except Exception as exc:
                logger.warning("[SessionStore] %s: closing a session failed: %s", name, exc)

    def expire(self, max_age_seconds: float) -> None:
        """Run each store's own expiry (stores bounded otherwise have none)."""
        for name, store in self._stores.items():
            if store.expire is None:
                continue
            try:
                store.expire(max_age_seconds)
            except Exception as exc:
                logger.debug("[SessionStore] %s: expiry failed: %s", name, exc)


sessions = SessionStore()
