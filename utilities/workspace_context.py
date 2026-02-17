"""
Workspace Context Builder
--------------------------
Builds the multi-line workspace context block that is appended to every
LLM prompt, giving the model awareness of the project structure, active
diagram, and existing layout.
"""

from typing import Any, Dict, List, Optional

from protocol.types import AssistantRequest
from .model_context import compact_model_summary, detailed_model_summary
from .model_resolution import resolve_target_model
from .layout_helpers import build_layout_anchor_lines


def build_workspace_context_block(
    request: AssistantRequest,
    target_diagram_type: str,
    target_model: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a multi-line workspace context string to append to LLM prompts."""
    lines: List[str] = []
    lines.append(f"Target diagram type: {target_diagram_type}")
    lines.append(f"Active diagram type: {request.context.active_diagram_type or request.diagram_type}")

    if request.context.active_diagram_id:
        lines.append(f"Active diagram id: {request.context.active_diagram_id}")

    active_model = request.context.active_model or request.current_model
    if active_model is not None:
        active_dt = request.context.active_diagram_type or request.diagram_type
        # Provide full structural detail for the model the LLM will modify
        lines.append(detailed_model_summary(active_model, active_dt))

    if target_model is None:
        target_model = resolve_target_model(request, target_diagram_type)
    layout_anchors = build_layout_anchor_lines(target_model, target_diagram_type)
    if layout_anchors:
        lines.append("Existing layout anchors (avoid overlap with these):")
        lines.extend(layout_anchors)

    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        project_name = snapshot.get("name")
        project_description = snapshot.get("description")
        if isinstance(project_name, str) and project_name.strip():
            lines.append(f"Project name: {project_name.strip()}")
        if isinstance(project_description, str) and project_description.strip():
            lines.append(f"Project description: {project_description.strip()}")

        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            diagram_lines: List[str] = []
            for dt, payload in diagrams.items():
                if not isinstance(payload, dict):
                    continue
                title = payload.get("title")
                model = payload.get("model")
                title_part = f" ({title})" if isinstance(title, str) and title.strip() else ""
                diagram_lines.append(f"- {dt}{title_part}: {compact_model_summary(model, dt)}")
            if diagram_lines:
                lines.append("Project diagrams overview:")
                lines.extend(diagram_lines[:10])

    summaries = request.context.diagram_summaries or []
    if summaries:
        compact_summaries: List[str] = []
        for item in summaries:
            if not isinstance(item, dict):
                continue
            dt = item.get("diagramType")
            title = item.get("title")
            if isinstance(dt, str):
                if isinstance(title, str) and title.strip():
                    compact_summaries.append(f"{dt} ({title.strip()})")
                else:
                    compact_summaries.append(dt)
        if compact_summaries:
            lines.append("Diagram summaries: " + ", ".join(compact_summaries[:10]))

    return "Workspace context:\n" + "\n".join(lines)
