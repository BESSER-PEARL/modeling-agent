"""
Workspace Context Builder
--------------------------
Builds the multi-line workspace context block that is appended to every
LLM prompt, giving the model awareness of the project structure, active
diagram, existing layout, and session history.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from besser.agent.core.session import Session

from protocol.types import AssistantRequest
from .model_context import compact_model_summary, detailed_model_summary
from .model_resolution import resolve_target_model
from .layout_helpers import build_layout_anchor_lines

logger = logging.getLogger(__name__)


def build_workspace_context_block(
    request: AssistantRequest,
    target_diagram_type: str,
    target_model: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a multi-line workspace context string to append to LLM prompts.

    Enhanced with cross-diagram awareness: when generating a StateMachine,
    GUI, or ObjectDiagram, the ClassDiagram summary is included as reference
    so the LLM can produce consistent models.
    """
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
                if isinstance(payload, list):
                    # New format: array of tabs
                    tab_count = len(payload)
                    if tab_count == 0:
                        continue
                    active_idx = request.context.get_active_index(dt)
                    active_tab = payload[active_idx] if 0 <= active_idx < tab_count else payload[0]
                    active_title = (
                        active_tab.get("title")
                        if isinstance(active_tab, dict) and isinstance(active_tab.get("title"), str)
                        else None
                    )
                    active_model = active_tab.get("model") if isinstance(active_tab, dict) else None
                    if tab_count == 1:
                        title_part = f" ({active_title})" if active_title and active_title.strip() else ""
                        diagram_lines.append(f"- {dt}{title_part}: {compact_model_summary(active_model, dt)}")
                    else:
                        active_label = f"active: '{active_title.strip()}'" if active_title and active_title.strip() else f"active tab {active_idx}"
                        diagram_lines.append(
                            f"- {dt} ({tab_count} tabs, {active_label}): {compact_model_summary(active_model, dt)}"
                        )
                elif isinstance(payload, dict):
                    # Legacy format: single dict
                    title = payload.get("title")
                    model = payload.get("model")
                    title_part = f" ({title})" if isinstance(title, str) and title.strip() else ""
                    diagram_lines.append(f"- {dt}{title_part}: {compact_model_summary(model, dt)}")
            if diagram_lines:
                lines.append("Project diagrams overview:")
                lines.extend(diagram_lines[:10])

            # --- Cross-diagram context ---
            # When the target is NOT a ClassDiagram, include the ClassDiagram
            # summary as reference so the LLM can produce consistent models.
            cross_ref = _build_cross_diagram_reference(
                target_diagram_type, diagrams, request.context,
            )
            if cross_ref:
                lines.append(cross_ref)

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


def _build_cross_diagram_reference(
    target_diagram_type: str,
    diagrams: Dict[str, Any],
    context: Optional[Any] = None,
) -> str:
    """Build cross-diagram reference context.

    When the target is a non-ClassDiagram type, include the ClassDiagram
    summary so the LLM can reference existing classes, attributes, and
    relationships for consistency.
    """
    # Diagram types that benefit from ClassDiagram context
    _NEEDS_CLASS_CONTEXT = {
        "StateMachineDiagram",
        "ObjectDiagram",
        "GUINoCodeDiagram",
    }

    if target_diagram_type not in _NEEDS_CLASS_CONTEXT:
        return ""

    class_payload = diagrams.get("ClassDiagram")

    # New format: array of tabs — pick the active one via context
    if isinstance(class_payload, list):
        if not class_payload:
            return ""
        active_idx = context.get_active_index("ClassDiagram") if context is not None else 0
        tab = class_payload[active_idx] if 0 <= active_idx < len(class_payload) else class_payload[0]
        class_model = tab.get("model") if isinstance(tab, dict) else None
    elif isinstance(class_payload, dict):
        class_model = class_payload.get("model")
    else:
        return ""

    if not isinstance(class_model, dict):
        return ""

    summary = detailed_model_summary(class_model, "ClassDiagram")
    if not summary:
        return ""

    hint = ""
    if target_diagram_type == "StateMachineDiagram":
        hint = (
            "\nUse the class diagram above as reference: states should correspond "
            "to real operations and lifecycle stages of the domain entities."
        )
    elif target_diagram_type == "GUINoCodeDiagram":
        hint = (
            "\nUse the class diagram above as reference: GUI pages/components "
            "should display and manage the attributes and relationships defined "
            "in the class diagram."
        )
    elif target_diagram_type == "ObjectDiagram":
        hint = (
            "\nUse the class diagram above as reference: objects must be instances "
            "of the classes defined above, with matching attribute names and types."
        )

    return f"Reference ClassDiagram (for consistency):\n{summary}{hint}"


# ---------------------------------------------------------------------------
# Session history tracking
# ---------------------------------------------------------------------------

_SESSION_HISTORY_KEY = "_session_action_history"
_MAX_HISTORY_ENTRIES = 15


def record_session_action(session: Session, action_summary: str) -> None:
    """Record a completed action in session history for LLM context."""
    history: List[str] = session.get(_SESSION_HISTORY_KEY) or []
    history.append(action_summary)
    if len(history) > _MAX_HISTORY_ENTRIES:
        history = history[-_MAX_HISTORY_ENTRIES:]
    session.set(_SESSION_HISTORY_KEY, history)


def build_session_summary(session: Session) -> str:
    """Build a compact session summary string from recorded actions.

    Returns an empty string if no actions have been recorded yet.
    """
    history: List[str] = session.get(_SESSION_HISTORY_KEY) or []
    if not history:
        return ""
    lines = ["Session history (what you've done so far):"]
    for i, entry in enumerate(history, 1):
        lines.append(f"  {i}. {entry}")
    return "\n".join(lines)
