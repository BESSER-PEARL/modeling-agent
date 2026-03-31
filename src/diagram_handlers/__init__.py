"""
Diagram Handlers Package
Provides specialized handlers for different UML diagram types.

Positions are computed deterministically by the :pymod:`layout_engine`
after the LLM returns semantic content.
"""

from .core.base_handler import BaseDiagramHandler, validate_spec
from .core.layout_engine import apply_layout
from .registry.factory import DiagramHandlerFactory
from .registry.metadata import DIAGRAM_TYPE_METADATA, get_diagram_type_info
from .types.agent_diagram_handler import AgentDiagramHandler
from .types.class_diagram_handler import ClassDiagramHandler
from .types.gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from .types.object_diagram_handler import ObjectDiagramHandler
from .types.quantum_circuit_diagram_handler import QuantumCircuitDiagramHandler
from .types.state_machine_handler import StateMachineHandler

__all__ = [
    'BaseDiagramHandler',
    'DiagramHandlerFactory',
    'ClassDiagramHandler',
    'ObjectDiagramHandler',
    'StateMachineHandler',
    'AgentDiagramHandler',
    'GUINoCodeDiagramHandler',
    'QuantumCircuitDiagramHandler',
    'DIAGRAM_TYPE_METADATA',
    'get_diagram_type_info',
    'apply_layout',
    'validate_spec',
]
