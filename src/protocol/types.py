from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Streaming actions
ACTION_STREAM_CHUNK = "stream_chunk"
ACTION_STREAM_DONE = "stream_done"
ACTION_PROGRESS = "progress"


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
    current_diagram_indices: Optional[Dict[str, int]] = None

    def get_active_index(self, diagram_type: str) -> int:
        """Return the active tab index for the given diagram type (default 0)."""
        if self.current_diagram_indices and diagram_type in self.current_diagram_indices:
            return self.current_diagram_indices[diagram_type]
        return 0

    def get_diagram_from_snapshot(self, diagram_type: str, index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get a single diagram dict from the project snapshot arrays.

        If index is None, uses the active index for that type.
        """
        if not isinstance(self.project_snapshot, dict):
            return None
        diagrams = self.project_snapshot.get("diagrams")
        if not isinstance(diagrams, dict):
            return None
        target = diagrams.get(diagram_type)
        if isinstance(target, list):
            idx = index if index is not None else self.get_active_index(diagram_type)
            if 0 <= idx < len(target) and isinstance(target[idx], dict):
                return target[idx]
            # Fallback to first non-empty diagram
            for d in target:
                if isinstance(d, dict) and isinstance(d.get("model"), dict):
                    return d
            return target[0] if target and isinstance(target[0], dict) else None
        # Legacy: single dict (backwards compat)
        if isinstance(target, dict):
            return target
        return None

    def get_all_diagrams_of_type(self, diagram_type: str) -> List[Dict[str, Any]]:
        """Return all diagram dicts for a given type from the project snapshot."""
        if not isinstance(self.project_snapshot, dict):
            return []
        diagrams = self.project_snapshot.get("diagrams")
        if not isinstance(diagrams, dict):
            return []
        target = diagrams.get(diagram_type)
        if isinstance(target, list):
            return [d for d in target if isinstance(d, dict)]
        if isinstance(target, dict):
            return [target]
        return []


@dataclass
class FileAttachment:
    """A file uploaded alongside a user message."""

    filename: str = ""
    content_b64: str = ""
    mime_type: str = ""


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
    attachments: List[FileAttachment] = field(default_factory=list)

    @property
    def is_v2(self) -> bool:
        return self.protocol_version == "2.0"

    @property
    def has_attachments(self) -> bool:
        return len(self.attachments) > 0
