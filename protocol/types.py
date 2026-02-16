from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SUPPORTED_DIAGRAM_TYPES = {
    "ClassDiagram",
    "ObjectDiagram",
    "StateMachineDiagram",
    "AgentDiagram",
    "GUINoCodeDiagram",
    "QuantumCircuitDiagram",
}


@dataclass
class WorkspaceContext:
    """Normalized workspace context used by the v2 assistant protocol."""

    active_diagram_type: str = "ClassDiagram"
    active_diagram_id: Optional[str] = None
    active_model: Optional[Dict[str, Any]] = None
    project_snapshot: Optional[Dict[str, Any]] = None
    diagram_summaries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AssistantRequest:
    """Canonical request object consumed by assistant states."""

    action: str = "user_message"
    protocol_version: str = "2.0"
    client_mode: str = "widget"
    session_id: Optional[str] = None
    message: str = ""
    diagram_type: str = "ClassDiagram"
    diagram_id: Optional[str] = None
    current_model: Optional[Dict[str, Any]] = None
    context: WorkspaceContext = field(default_factory=WorkspaceContext)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_v2(self) -> bool:
        return self.protocol_version == "2.0"
