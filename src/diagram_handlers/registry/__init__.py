"""Registration APIs for diagram handlers."""

from .factory import DiagramHandlerFactory
from .metadata import DIAGRAM_TYPE_METADATA, get_diagram_type_info

__all__ = [
    "DiagramHandlerFactory",
    "DIAGRAM_TYPE_METADATA",
    "get_diagram_type_info",
]

