"""The per-request BYOK config and the context variable that holds it.

Set by ``byok.set_current`` at the request boundary and read by the LLM call
sites and by ``tracking`` (cost per route). ``byok`` records usage into
``tracking``, so this state sits below both.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BYOKConfig:
    """The user's key/provider for a single request. ``model`` is the user's
    explicitly chosen model (used for every call), if any. ``base_url``
    is set for an OpenAI-compatible endpoint: the 'local' provider's (it
    arrives as provider='openai' + this URL), or a catalog gateway's."""

    provider: str
    api_key: str
    model: Optional[str] = None
    base_url: Optional[str] = None

    def redacted(self) -> str:
        """A log-safe description that never reveals the key."""
        base = f", base_url={self.base_url}" if self.base_url else ""
        return (
            f"BYOK(provider={self.provider}, model={self.model or '<default>'}"
            f"{base}, key=(redacted))"
        )


# Holds the active per-request BYOK config, or ``None`` when the request
# uses the shared server LLM. Read by the routing call sites.
current_byok: "contextvars.ContextVar[Optional[BYOKConfig]]" = contextvars.ContextVar(
    "current_byok", default=None
)


def get_current() -> Optional[BYOKConfig]:
    """Return the active :class:`BYOKConfig`, or ``None``."""
    return current_byok.get()
