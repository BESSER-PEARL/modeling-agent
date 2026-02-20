"""Backward-compatible exports for handler factory and metadata APIs."""

from .registry.factory import DiagramHandlerFactory
from .registry.metadata import DIAGRAM_TYPE_METADATA, get_diagram_type_info

__all__ = [
    "DiagramHandlerFactory",
    "DIAGRAM_TYPE_METADATA",
    "get_diagram_type_info",
]
