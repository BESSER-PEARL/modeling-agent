"""
Diagram Handler Factory
Creates and manages diagram type handlers
"""

from __future__ import annotations

from typing import Dict, Optional

from ..core.base_handler import BaseDiagramHandler
from ..types.agent_diagram_handler import AgentDiagramHandler
from ..types.class_diagram_handler import ClassDiagramHandler
from ..types.gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from ..types.object_diagram_handler import ObjectDiagramHandler
from ..types.quantum_circuit_diagram_handler import QuantumCircuitDiagramHandler
from ..types.state_machine_handler import StateMachineHandler

HANDLER_CLASSES = (
    ClassDiagramHandler,
    ObjectDiagramHandler,
    StateMachineHandler,
    AgentDiagramHandler,
    GUINoCodeDiagramHandler,
    QuantumCircuitDiagramHandler,
)


class DiagramHandlerFactory:
    """Factory for creating diagram handlers."""

    def __init__(self, llm):
        """Initialize factory with LLM instance."""
        self.llm = llm
        self._handlers: Dict[str, BaseDiagramHandler] = {}
        self._initialize_handlers()

    def _initialize_handlers(self) -> None:
        """Create all diagram handlers."""
        for handler_class in HANDLER_CLASSES:
            handler = handler_class(self.llm)
            self._handlers[handler.get_diagram_type()] = handler

    def get_handler(self, diagram_type: str) -> Optional[BaseDiagramHandler]:
        """Get handler for specific diagram type."""
        return self._handlers.get(diagram_type)

    def get_supported_types(self) -> list[str]:
        """Get list of supported diagram types."""
        return list(self._handlers.keys())

    def is_supported(self, diagram_type: str) -> bool:
        """Check if diagram type is supported."""
        return diagram_type in self._handlers
