"""Protocol utilities: re-export the assistant-request dataclasses.

Callers import ``parse_assistant_request`` directly from ``protocol.adapters``;
this package only re-exports the lightweight dataclasses.
"""

from __future__ import annotations

from .types import AssistantRequest, WorkspaceContext

__all__ = [
    "AssistantRequest",
    "WorkspaceContext",
]
