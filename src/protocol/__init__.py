"""Protocol utilities for assistant request parsing and compatibility."""

from __future__ import annotations

from typing import Any

from .types import AssistantRequest, WorkspaceContext


def parse_assistant_request(session: Any) -> AssistantRequest:
    """Parse a session payload into a normalized assistant request.

    Imported lazily to keep ``protocol.types`` lightweight and avoid pulling
    runtime dependencies when only protocol dataclasses are needed.
    """
    from .adapters import parse_assistant_request as _parse_assistant_request

    return _parse_assistant_request(session)


__all__ = [
    "AssistantRequest",
    "WorkspaceContext",
    "parse_assistant_request",
]
