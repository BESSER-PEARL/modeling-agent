"""
Conversation Memory
-------------------
Sliding-window conversation memory with optional LLM summarization.

Each WebSocket session gets its own memory instance.  Recent messages
are stored verbatim; when the window fills up, the oldest messages
are compressed into a summary so the LLM always has relevant context
without exceeding token budgets.

Usage::

    from memory import get_memory

    mem = get_memory(session_id)
    mem.add_user("Create an e-commerce system")
    mem.add_assistant("Created 5 classes: User, Product, Order, Cart, Payment")
    context = mem.build_context()  # returns list of message dicts
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_WINDOW_SIZE = 20  # max recent messages to keep verbatim
_DEFAULT_MAX_SUMMARY_LEN = 500  # max chars in the rolling summary
_SUMMARY_TRIGGER = 16  # summarize when deque reaches this size


@dataclass
class _Message:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """Per-session sliding-window conversation memory."""

    def __init__(
        self,
        session_id: str,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        summarizer: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.session_id = session_id
        self._window_size = window_size
        self._messages: Deque[_Message] = deque(maxlen=window_size)
        self._summary: str = ""
        self._summarizer = summarizer
        self._lock = threading.Lock()
        self._total_messages: int = 0

    # ------------------------------------------------------------------
    # Adding messages
    # ------------------------------------------------------------------

    def add_user(self, content: str, **metadata) -> None:
        """Record a user message."""
        self._add("user", content, metadata)

    def add_assistant(self, content: str, **metadata) -> None:
        """Record an assistant message."""
        self._add("assistant", content, metadata)

    def add_system(self, content: str, **metadata) -> None:
        """Record a system/context message."""
        self._add("system", content, metadata)

    def _add(self, role: str, content: str, metadata: dict) -> None:
        # Check if compression is needed, and if so extract the data to
        # summarize while holding the lock, but do the LLM call outside
        # the lock to avoid blocking other WebSocket threads.
        compress_text: Optional[str] = None
        compress_count: int = 0

        with self._lock:
            if (
                len(self._messages) >= _SUMMARY_TRIGGER
                and self._summarizer
                and len(self._messages) >= self._window_size - 2
            ):
                compress_text, compress_count = self._extract_for_compress()

            self._messages.append(_Message(
                role=role,
                content=content,
                metadata=metadata,
            ))
            self._total_messages += 1

        # Run the summarizer outside the lock so it doesn't block other threads
        if compress_text is not None:
            self._run_compress(compress_text, compress_count)

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Build a list of message dicts suitable for the OpenAI messages array.

        Returns most recent messages (up to *max_messages*), prepended with
        the rolling summary if one exists.
        """
        with self._lock:
            msgs: List[Dict[str, str]] = []

            # Prepend rolling summary as a system message
            if self._summary:
                msgs.append({
                    "role": "system",
                    "content": (
                        f"Previous conversation summary:\n{self._summary}"
                    ),
                })

            limit = max_messages or self._window_size
            recent = list(self._messages)[-limit:]
            for m in recent:
                msgs.append({"role": m.role, "content": m.content})

            return msgs

    def get_last_n(self, n: int = 5) -> List[Dict[str, str]]:
        """Return the last *n* messages as dicts."""
        with self._lock:
            recent = list(self._messages)[-n:]
            return [{"role": m.role, "content": m.content} for m in recent]

    def get_summary(self) -> str:
        """Return the current rolling summary."""
        with self._lock:
            return self._summary

    @property
    def message_count(self) -> int:
        """Total messages added (including those already compressed)."""
        return self._total_messages

    @property
    def window_count(self) -> int:
        """Messages currently in the sliding window."""
        with self._lock:
            return len(self._messages)

    # ------------------------------------------------------------------
    # Compression / summarization
    # ------------------------------------------------------------------

    def _extract_for_compress(self) -> Tuple[str, int]:
        """Extract oldest messages for summarization.  Must hold ``self._lock``.

        Returns ``(text_to_summarize, message_count)`` — the actual LLM call
        happens outside the lock in :meth:`_run_compress`.
        """
        half = len(self._messages) // 2
        old_messages = []
        for _ in range(half):
            old_messages.append(self._messages.popleft())

        text_parts = []
        if self._summary:
            text_parts.append(f"Previous summary: {self._summary}")
        for m in old_messages:
            text_parts.append(f"{m.role}: {m.content}")

        return "\n".join(text_parts), len(old_messages)

    def _run_compress(self, text_to_summarize: str, message_count: int) -> None:
        """Run the summarizer LLM call **outside** the lock, then store the result.

        This prevents the (potentially slow) LLM round-trip from blocking all
        other threads that need to add messages to the memory.
        """
        try:
            prompt = (
                "Summarize this conversation history in 2-3 sentences. "
                "Focus on: what diagrams were created/modified, key decisions, "
                "and the user's current goal. Be concise.\n\n"
                f"{text_to_summarize[:3000]}"
            )
            new_summary = self._summarizer(prompt)[:_DEFAULT_MAX_SUMMARY_LEN]
        except Exception as exc:
            logger.warning(f"[Memory:{self.session_id}] Summarization failed: {exc}")
            new_summary = text_to_summarize[:_DEFAULT_MAX_SUMMARY_LEN]

        # Store under lock (fast operation)
        with self._lock:
            self._summary = new_summary
        logger.info(
            f"[Memory:{self.session_id}] Compressed {message_count} messages "
            f"into summary ({len(new_summary)} chars)"
        )

    def clear(self) -> None:
        """Clear all messages and summary."""
        with self._lock:
            self._messages.clear()
            self._summary = ""
            self._total_messages = 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory state for inspection/debugging."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "total_messages": self._total_messages,
                "window_count": len(self._messages),
                "has_summary": bool(self._summary),
                "summary": self._summary[:200] if self._summary else "",
                "recent_messages": [
                    {"role": m.role, "content": m.content[:100]}
                    for m in list(self._messages)[-5:]
                ],
            }


# ---------------------------------------------------------------------------
# Module-level memory registry
# ---------------------------------------------------------------------------

_memories: Dict[str, ConversationMemory] = {}
_memories_lock = threading.Lock()


def get_memory(
    session_id: str,
    summarizer: Optional[Callable[[str], str]] = None,
) -> ConversationMemory:
    """Get or create a ConversationMemory for the given session.

    Args:
        session_id: Unique session identifier.
        summarizer: Optional callable ``(text) -> summary_text`` used to
            compress old messages.  Typically ``llm.predict``.
    """
    with _memories_lock:
        mem = _memories.get(session_id)
        if mem is None:
            mem = ConversationMemory(session_id, summarizer=summarizer)
            _memories[session_id] = mem
        return mem


def remove_memory(session_id: str) -> None:
    """Remove a session's memory (e.g. on disconnect)."""
    with _memories_lock:
        _memories.pop(session_id, None)


def cleanup_stale_memories(max_age_seconds: int = 86400) -> int:
    """Remove conversation memories that haven't been updated in max_age_seconds.

    Call this periodically (e.g., every hour) to prevent memory leaks
    from disconnected sessions.
    """
    now = time.time()
    stale_ids = []
    with _memories_lock:
        for session_id, mem in _memories.items():
            with mem._lock:
                if not mem._messages:
                    stale_ids.append(session_id)
                    continue
                last_msg = mem._messages[-1]
                if now - last_msg.timestamp > max_age_seconds:
                    stale_ids.append(session_id)
        for sid in stale_ids:
            _memories.pop(sid, None)
    if stale_ids:
        logger.info(f"[Memory] Cleaned up {len(stale_ids)} stale session(s)")
    return len(stale_ids)
