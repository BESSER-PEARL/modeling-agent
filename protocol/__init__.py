"""Protocol utilities for assistant request parsing and compatibility."""

from .types import AssistantRequest, WorkspaceContext
from .adapters import parse_assistant_request

__all__ = ["AssistantRequest", "WorkspaceContext", "parse_assistant_request"]
