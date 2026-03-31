"""Diagram type metadata and lookup helpers."""

from __future__ import annotations

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

