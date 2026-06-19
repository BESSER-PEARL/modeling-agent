"""Conversation memory for multi-turn context."""

import logging
from typing import Any, Optional

from .conversation_memory import ConversationMemory, get_memory

logger = logging.getLogger(__name__)

__all__ = ["ConversationMemory", "get_memory", "memory_session_key"]


def memory_session_key(session: Any, request: Optional[Any] = None) -> str:
    """Return the stable key to use for :func:`get_memory`.

    Prefers the v2 payload's ``sessionId`` (``AssistantRequest.session_id``)
    — it is stable across WebSocket reconnects, whereas the BAF session id
    changes on every reconnect without a stable user query param, which
    would silently drop all conversation context (B-5). Falls back to the
    BAF session id when no parsed request / sessionId is available.

    Args:
        session: The BAF session object.
        request: An already-parsed ``AssistantRequest`` when the caller has
            one handy; otherwise the cached parse is attempted here.
    """
    sid = getattr(request, "session_id", None) if request is not None else None
    if not sid:
        try:
            # Deferred import: memory must stay importable without the
            # protocol/BAF stack (pure unit tests).
            from protocol.adapters import parse_assistant_request
            parsed = parse_assistant_request(session)
            sid = getattr(parsed, "session_id", None)
        except Exception as exc:
            logger.debug(f"memory_session_key: request parse failed (best-effort): {exc}")
            sid = None
    if sid:
        return str(sid)
    baf_id = getattr(session, "id", None)
    return str(baf_id) if baf_id else str(id(session))
