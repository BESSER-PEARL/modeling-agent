"""
Diagram Handler Factory
Creates and manages diagram type handlers
"""

from typing import Dict, Optional

from .agent_diagram_handler import AgentDiagramHandler
from .base_handler import BaseDiagramHandler
from .class_diagram_handler import ClassDiagramHandler
from .gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from .object_diagram_handler import ObjectDiagramHandler
from .quantum_circuit_diagram_handler import QuantumCircuitDiagramHandler
from .state_machine_handler import StateMachineHandler


class DiagramHandlerFactory:
    """Factory for creating diagram handlers."""

    def __init__(self, llm):
        """Initialize factory with LLM instance."""
        self.llm = llm
        self._handlers: Dict[str, BaseDiagramHandler] = {}
        self._initialize_handlers()

    def _initialize_handlers(self):
        """Create all diagram handlers."""
        handlers = [
            ClassDiagramHandler(self.llm),
            ObjectDiagramHandler(self.llm),
            StateMachineHandler(self.llm),
            AgentDiagramHandler(self.llm),
            GUINoCodeDiagramHandler(self.llm),
            QuantumCircuitDiagramHandler(self.llm),
        ]

        for handler in handlers:
            self._handlers[handler.get_diagram_type()] = handler

    def get_handler(self, diagram_type: str) -> Optional[BaseDiagramHandler]:
        """Get handler for specific diagram type."""
        return self._handlers.get(diagram_type)

    def get_supported_types(self) -> list:
        """Get list of supported diagram types."""
        return list(self._handlers.keys())

    def is_supported(self, diagram_type: str) -> bool:
        """Check if diagram type is supported."""
        return diagram_type in self._handlers


# Diagram type metadata
DIAGRAM_TYPE_METADATA = {
    "ClassDiagram": {
        "name": "Class Diagram",
        "icon": "class",
        "elements": ["Class", "Interface", "Relationship"],
        "description": "Model classes, attributes, methods, and relationships",
        "keywords": ["class", "interface", "inheritance", "association"],
    },
    "ObjectDiagram": {
        "name": "Object Diagram",
        "icon": "object",
        "elements": ["Object", "Link"],
        "description": "Model object instances and their relationships",
        "keywords": ["object", "instance", "link"],
    },
    "StateMachineDiagram": {
        "name": "State Machine Diagram",
        "icon": "state",
        "elements": ["State", "Transition", "InitialState", "FinalState"],
        "description": "Model state transitions and behaviors",
        "keywords": ["state", "transition", "event", "trigger"],
    },
    "AgentDiagram": {
        "name": "Agent Diagram",
        "icon": "agent",
        "elements": ["Agent", "Message", "Environment"],
        "description": "Model agent systems and interactions",
        "keywords": ["agent", "message", "environment", "multi-agent"],
    },
    "GUINoCodeDiagram": {
        "name": "GUI No-Code Diagram",
        "icon": "gui",
        "elements": ["Page", "Section", "Component"],
        "description": "Model GUI layouts and page structures for no-code editing",
        "keywords": ["gui", "web ui", "page", "layout", "grapesjs"],
    },
    "QuantumCircuitDiagram": {
        "name": "Quantum Circuit Diagram",
        "icon": "quantum",
        "elements": ["Qubit", "Gate", "Measurement"],
        "description": "Model quantum circuits with gates and measurements",
        "keywords": ["quantum", "qiskit", "gate", "qubit", "circuit"],
    },
}


def get_diagram_type_info(diagram_type: str) -> dict:
    """Get metadata for a diagram type."""
    return DIAGRAM_TYPE_METADATA.get(diagram_type, DIAGRAM_TYPE_METADATA["ClassDiagram"])
