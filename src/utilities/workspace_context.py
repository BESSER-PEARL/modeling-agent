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
    from baf.core.session import Session

from protocol.types import AssistantRequest
from session_keys import SESSION_ACTION_HISTORY
from .model_context import compact_model_summary, detailed_model_summary

logger = logging.getLogger(__name__)


def build_workspace_context_block(
    request: AssistantRequest,
    target_diagram_type: str,
) -> str:
    """Build a compact workspace context string to append to LLM prompts.

    Only includes information the LLM can actually act on:
    - Target diagram type
    - Detailed model summary (for modify — so the LLM knows what exists)
    - Cross-diagram reference (for StateMachine/Object/GUI — consistency)
    - Non-empty diagram types in the project (so the LLM knows what's available)
    """
    lines: List[str] = []
    lines.append(f"Target diagram type: {target_diagram_type}")

    # Detailed model summary — the LLM needs this to know what's on the canvas
    active_model = request.context.active_model or request.current_model
    if active_model is not None:
        active_dt = request.context.active_diagram_type or request.diagram_type
        lines.append(detailed_model_summary(active_model, active_dt))

    # Cross-diagram reference for non-ClassDiagram targets
    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            cross_ref = _build_cross_diagram_reference(
                target_diagram_type, diagrams, request.context,
            )
            if cross_ref:
                lines.append(cross_ref)

            # Only list non-empty diagrams (empty ones are noise)
            non_empty: List[str] = []
            for dt, payload in diagrams.items():
                summary = ""
                if isinstance(payload, list) and payload:
                    active_idx = request.context.get_active_index(dt)
                    tab = payload[active_idx] if 0 <= active_idx < len(payload) else payload[0]
                    model = tab.get("model") if isinstance(tab, dict) else None
                    summary = compact_model_summary(model, dt)
                elif isinstance(payload, dict):
                    summary = compact_model_summary(payload.get("model"), dt)
                # Skip empty diagrams
                if summary and "0 element" not in summary and "0 circuit" not in summary:
                    non_empty.append(f"- {dt}: {summary}")
            if non_empty:
                lines.append("Other diagrams in project:")
                lines.extend(non_empty[:6])

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

_MAX_HISTORY_ENTRIES = 15


def record_session_action(session: Session, action_summary: str) -> None:
    """Record a completed action in session history for LLM context."""
    history: List[str] = session.get(SESSION_ACTION_HISTORY) or []
    history.append(action_summary)
    if len(history) > _MAX_HISTORY_ENTRIES:
        history = history[-_MAX_HISTORY_ENTRIES:]
    session.set(SESSION_ACTION_HISTORY, history)


def build_session_summary(session: Session) -> str:
    """Build a compact session summary string from recorded actions.

    Returns an empty string if no actions have been recorded yet.
    """
    history: List[str] = session.get(SESSION_ACTION_HISTORY) or []
    if not history:
        return ""
    lines = ["Session history (what you've done so far):"]
    for i, entry in enumerate(history, 1):
        lines.append(f"  {i}. {entry}")
    return "\n".join(lines)
