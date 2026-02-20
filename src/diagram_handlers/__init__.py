"""
Diagram Handlers Package
Provides specialized handlers for different UML diagram types.

Positions are computed deterministically by the :pymod:`layout_engine`
after the LLM returns semantic content.
"""

from .base_handler import BaseDiagramHandler, validate_spec
from .class_diagram_handler import ClassDiagramHandler
from .object_diagram_handler import ObjectDiagramHandler
from .state_machine_handler import StateMachineHandler
from .agent_diagram_handler import AgentDiagramHandler
from .gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from .quantum_circuit_diagram_handler import QuantumCircuitDiagramHandler
from .layout_engine import apply_layout

__all__ = [
    'BaseDiagramHandler',
    'ClassDiagramHandler',
    'ObjectDiagramHandler',
    'StateMachineHandler',
    'AgentDiagramHandler',
    'GUINoCodeDiagramHandler',
    'QuantumCircuitDiagramHandler',
    'apply_layout',
    'validate_spec',
]
