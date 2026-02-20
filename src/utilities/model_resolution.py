"""
Model Resolution
-----------------
Functions to resolve the correct model and reference data from the
``AssistantRequest`` context.  Used by the agent orchestrator and
workspace-context builder.
"""

from typing import Any, Dict, List, Optional

from protocol.types import AssistantRequest


def resolve_target_model(
    request: AssistantRequest, target_diagram_type: str,
) -> Optional[Dict[str, Any]]:
    """Find the best available model for *target_diagram_type* in the request context."""
    if target_diagram_type == request.context.active_diagram_type and isinstance(request.current_model, dict):
        return request.current_model

    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            target = diagrams.get(target_diagram_type)
            if isinstance(target, dict) and isinstance(target.get("model"), dict):
                return target.get("model")

    if isinstance(request.current_model, dict):
        return request.current_model
    if isinstance(request.context.active_model, dict):
        return request.context.active_model
    return None


def resolve_object_reference_diagram(
    request: AssistantRequest,
    target_model: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve the best available ClassDiagram model for ObjectDiagram grounding.

    Priority:
    1) Object diagram's own ``referenceDiagramData`` (if already set)
    2) Active in-memory ClassDiagram model from current context
    3) Current model when it is a ClassDiagram
    4) Project snapshot ClassDiagram model
    """
    if isinstance(target_model, dict):
        reference_diagram = target_model.get("referenceDiagramData")
        if isinstance(reference_diagram, dict):
            return reference_diagram

    active_diagram_type = request.context.active_diagram_type or request.diagram_type
    active_model = request.context.active_model if isinstance(request.context.active_model, dict) else None
    if active_diagram_type == "ClassDiagram" and isinstance(active_model, dict):
        if isinstance(active_model.get("elements"), dict):
            return active_model

    if active_diagram_type == "ClassDiagram" and isinstance(request.current_model, dict):
        if isinstance(request.current_model.get("elements"), dict):
            return request.current_model

    if isinstance(request.context.project_snapshot, dict):
        diagrams = request.context.project_snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            class_diagram = diagrams.get("ClassDiagram")
            if isinstance(class_diagram, dict) and isinstance(class_diagram.get("model"), dict):
                return class_diagram["model"]

    return None


def count_reference_classes(reference_diagram: Optional[Dict[str, Any]]) -> int:
    """Count how many Class elements exist in a reference diagram."""
    if not isinstance(reference_diagram, dict):
        return 0
    elements = reference_diagram.get("elements")
    if not isinstance(elements, dict):
        return 0
    return sum(
        1
        for element in elements.values()
        if isinstance(element, dict) and element.get("type") == "Class"
    )


def resolve_class_diagram(request: AssistantRequest) -> Optional[Dict[str, Any]]:
    """Return the ClassDiagram model from the workspace context, if available.

    Checks the project snapshot first, then falls back to the active/current
    model when the active diagram type is ``ClassDiagram``.
    """
    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            cd = diagrams.get("ClassDiagram")
            if isinstance(cd, dict) and isinstance(cd.get("model"), dict):
                return cd["model"]
    if request.context.active_diagram_type == "ClassDiagram" and isinstance(request.current_model, dict):
        return request.current_model
    return None
