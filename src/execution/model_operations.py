"""Single model-operation execution and helpers.

Handles the core execute_model_operation flow: existing-model guards,
GUI-choice prompts, handler dispatch, progress threads, and result enrichment.
"""

import logging
import threading
from typing import Any, Dict, Optional

from baf.core.session import Session

import agent_context as ctx
from agent_config import MAX_TABS, CONVERSATION_HISTORY_DEPTH
from protocol.types import AssistantRequest
from session_helpers import reply_message, reply_payload, reply_progress
from model_utils import model_has_elements
from diagram_handlers.registry.metadata import get_diagram_type_info
from orchestrator import determine_target_diagram_type, resolve_diagram_id
from utilities.model_resolution import (
    resolve_target_model,
    resolve_object_reference_diagram,
    count_reference_classes,
    resolve_class_diagram,
)
from utilities.workspace_context import build_workspace_context_block, record_session_action
from utilities.class_metadata import extract_class_metadata
from suggestions import get_suggested_actions
from session_keys import (
    LAST_EXECUTED_DIAGRAM_TYPE,
    LAST_MATCHED_INTENT,
    PENDING_COMPLETE_SYSTEM,
    PENDING_GUI_CHOICE,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared confirmation flow for existing-model guard
# ------------------------------------------------------------------

def _build_existing_model_confirmation(
    session: Session,
    request: AssistantRequest,
    target_diagram_type: str,
    existing_summary: str,
    pending_data: dict,
    source_description: str = "",
) -> None:
    """Build and send confirmation prompt when target diagram already has elements.

    Computes tab count, builds confirmation message with replace/keep/new-tab options,
    stores pending state, and sends the reply.
    """
    all_tabs = request.context.get_all_diagrams_of_type(target_diagram_type)
    tab_count = len(all_tabs) if all_tabs else 1
    max_tabs = MAX_TABS
    can_add_tab = tab_count < max_tabs

    # Build confirmation actions
    confirmation_actions = [
        {"label": "Replace existing", "prompt": "replace"},
        {"label": "Keep and add alongside", "prompt": "keep"},
    ]
    if can_add_tab:
        confirmation_actions.append({"label": "Create in new tab", "prompt": "new tab"})

    # Store pending state
    pending_data['can_add_tab'] = can_add_tab
    session.set(PENDING_COMPLETE_SYSTEM, pending_data)

    # Build message
    if can_add_tab:
        tab_info = f"({tab_count}/{max_tabs} tabs used)"
    else:
        tab_info = f"(All {max_tabs} tabs are in use)"

    # source_description differentiates file-upload vs model-operation path
    confirmation_msg = (
        f"{source_description}, but you already have a model ({existing_summary}). "
        f"Would you like me to **replace** it, **keep** it and add alongside"
        + (f", or create in a **new tab**? {tab_info}" if can_add_tab else f"? {tab_info}")
    )

    reply_payload(session, {
        "action": "assistant_message",
        "message": confirmation_msg,
        "suggestedActions": confirmation_actions,
    })


# ------------------------------------------------------------------
# Single model operation
# ------------------------------------------------------------------

def execute_model_operation(
    session: Session,
    request: AssistantRequest,
    operation: Dict[str, Any],
    default_mode: str,
    _skip_existing_check: bool = False,
    _replace_existing: Optional[bool] = None,
    _skip_gui_choice: bool = False,
    _create_new_tab: bool = False,
) -> Optional[str]:
    """Execute a single model operation (create, modify, etc.).

    Returns the target diagram type on success, ``None`` if a confirmation
    prompt was stored (existing-model guard or GUI-choice prompt) or on failure.
    """
    target_diagram_type = operation.get("diagramType")
    if not isinstance(target_diagram_type, str) or not target_diagram_type:
        target_diagram_type = determine_target_diagram_type(
            request, last_intent=session.get(LAST_MATCHED_INTENT),
        )

    operation_mode = operation.get("mode")
    if not isinstance(operation_mode, str) or not operation_mode:
        operation_mode = default_mode

    operation_request = operation.get("request")
    if not isinstance(operation_request, str) or not operation_request.strip():
        operation_request = request.message
    operation_request = operation_request.strip()

    logger.info(
        f"⚙️ [ModelOp] Executing: diagram={target_diagram_type}, mode={operation_mode}, "
        f"request={operation_request[:120]!r}"
    )

    # ── Existing-model guard for complete_system ─────────────────────────
    if (
        not _skip_existing_check
        and operation_mode == 'complete_system'
    ):
        existing_model = resolve_target_model(request, target_diagram_type)
        if model_has_elements(existing_model):
            from utilities.model_context import compact_model_summary

            summary = compact_model_summary(existing_model, target_diagram_type)
            stored_operation = {**operation, 'mode': operation_mode}

            _build_existing_model_confirmation(
                session=session,
                request=request,
                target_diagram_type=target_diagram_type,
                existing_summary=summary,
                pending_data={
                    'message': operation_request,
                    'diagram_type': target_diagram_type,
                    'operation': stored_operation,
                    'default_mode': default_mode,
                },
                source_description=f"I generated a new {target_diagram_type}",
            )
            logger.info(
                f"[ModelOp] Asked user to confirm replace/keep for existing {target_diagram_type}"
            )
            return None

    # ── GUI generation-mode choice ───────────────────────────────────────
    _CUSTOM_GUI_HINTS = {
        "chart", "dashboard", "custom", "specific", "page for",
        "sidebar", "metric", "kpi", "landing", "hero",
        "form", "layout", "only", "just", "don't include",
        "exclude", "style", "theme", "color", "dark",
        "personali", "unique", "tailored", "bespoke",
    }
    _resolved_class_diagram = None
    if target_diagram_type == "GUINoCodeDiagram" and operation_mode in ("complete_system", None, ""):
        _req_lower = (operation_request or "").lower()
        _wants_custom = any(hint in _req_lower for hint in _CUSTOM_GUI_HINTS)

        _resolved_class_diagram = resolve_class_diagram(request)
        _has_class_diagram = (
            isinstance(_resolved_class_diagram, dict)
            and isinstance(_resolved_class_diagram.get("elements"), dict)
            and len(_resolved_class_diagram["elements"]) > 0
        )

        if _has_class_diagram and _wants_custom:
            logger.info("[ModelOp] Custom GUI request detected — using LLM-driven path")

        elif _has_class_diagram and not _skip_gui_choice:
            session.set(PENDING_GUI_CHOICE, {
                'operation_request': operation_request,
                'operation': operation,
                'default_mode': default_mode,
                'diagram_type': target_diagram_type,
                '_replace_existing': _replace_existing,
            })
            reply_payload(session, {
                "action": "assistant_message",
                "message": (
                    "How would you like me to generate the GUI?\n\n"
                    "1️⃣ **Auto-generate** — Fast & deterministic. Creates one page per class "
                    "with data tables and method buttons.\n"
                    "2️⃣ **LLM-generated** *(experimental)* — AI-designed layout with "
                    "personalized pages, navigation, and styling."
                ),
                "suggestedActions": [
                    {"label": "Auto-generate", "prompt": "auto"},
                    {"label": "LLM-generated (experimental)", "prompt": "llm"},
                ],
            })
            logger.info("[ModelOp] Asked user to choose GUI generation mode")
            return None

    handler = ctx.diagram_factory.get_handler(target_diagram_type)
    if not handler:
        logger.warning(f"⚠️ [ModelOp] No handler for diagram type: {target_diagram_type}")
        reply_message(
            session,
            f"{target_diagram_type} is not supported by the modeling handler yet.",
        )
        return None

    # Send progress feedback
    diagram_info = get_diagram_type_info(target_diagram_type)
    diagram_label = diagram_info.get("name", target_diagram_type)
    if operation_mode == "complete_system":
        reply_progress(session, f"Thinking about your {diagram_label} design...")
    elif operation_mode == "modify_model":
        reply_progress(session, f"Analyzing changes...")

    target_model = resolve_target_model(request, target_diagram_type)

    # Inject conversation context for multi-turn awareness.
    conversation_context = ""
    if not _skip_existing_check:
        try:
            from memory import get_memory
            session_id = getattr(session, 'id', None) or str(id(session))
            mem = get_memory(session_id)
            recent = mem.get_last_n(CONVERSATION_HISTORY_DEPTH)
            if recent and len(recent) > 1:
                history_lines = []
                for msg in recent[:-1]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")[:200]
                    history_lines.append(f"  {role}: {content}")
                if history_lines:
                    conversation_context = (
                        "Recent conversation context (use this to understand what the user has been working on):\n"
                        + "\n".join(history_lines)
                        + "\n\n"
                    )
        except Exception as exc:
            logger.debug(f"Conversation memory retrieval failed (best-effort): {exc}")

    modeling_prompt = (
        f"{conversation_context}"
        f"{operation_request}\n\n"
        f"{build_workspace_context_block(request, target_diagram_type)}"
    )

    # ── Resolve class metadata for GUI diagram ──
    gui_class_metadata = None
    if target_diagram_type == "GUINoCodeDiagram":
        class_diagram_model = _resolved_class_diagram or resolve_class_diagram(request)
        if isinstance(class_diagram_model, dict):
            gui_class_metadata = extract_class_metadata(class_diagram_model)
            if gui_class_metadata:
                logger.info(
                    f"[ModelOp] Resolved {len(gui_class_metadata)} class(es) for GUI chart binding"
                )

    logger.debug(f"[ModelOp] Modeling prompt ({len(modeling_prompt)} chars): {modeling_prompt[:300]!r}")
    logger.debug(f"[ModelOp] Target model present: {target_model is not None}, type: {type(target_model).__name__}")

    # Timed progress updates while the handler runs
    _progress_stop = threading.Event()

    def _timed_progress():
        steps = []
        if operation_mode == "complete_system":
            steps = [
                (8, "Generating classes and relationships..."),
                (20, "Building attributes and methods..."),
                (35, "Almost there..."),
            ]
        elif operation_mode == "modify_model":
            steps = [
                (4, "Updating model..."),
            ]
        for delay, msg in steps:
            if _progress_stop.wait(timeout=delay):
                return
            reply_progress(session, msg)

    if operation_mode in ("complete_system", "modify_model"):
        progress_thread = threading.Thread(target=_timed_progress, daemon=True)
        progress_thread.start()
    else:
        progress_thread = None

    try:
        if operation_mode == "modify_model":
            extra_kwargs: Dict[str, Any] = {"class_metadata": gui_class_metadata}
            if target_diagram_type == "ObjectDiagram":
                reference_diagram = resolve_object_reference_diagram(request, target_model)
                reference_class_count = count_reference_classes(reference_diagram)
                if reference_class_count > 0:
                    logger.info(
                        f"[ModelOp] ObjectDiagram modify reference resolved with {reference_class_count} class(es)."
                    )
                else:
                    logger.warning(
                        "[ModelOp] ObjectDiagram modify reference is missing or empty; output may drift."
                    )
                extra_kwargs["reference_diagram"] = reference_diagram
            result = handler.generate_modification(
                modeling_prompt,
                target_model,
                **extra_kwargs,
            )
        else:
            if target_diagram_type == "ObjectDiagram":
                reference_diagram = resolve_object_reference_diagram(request, target_model)
                reference_class_count = count_reference_classes(reference_diagram)
                if reference_class_count > 0:
                    logger.info(
                        f"[ModelOp] ObjectDiagram reference resolved with {reference_class_count} class(es)."
                    )
                else:
                    logger.warning(
                        "[ModelOp] ObjectDiagram reference is missing or empty."
                    )
                    reply_message(
                        session,
                        "Please create a **Class Diagram** first — Object Diagrams "
                        "need class definitions to instantiate from.",
                    )
                    return None
                result = handler.generate_complete_system(
                    modeling_prompt,
                    reference_diagram=reference_diagram,
                    existing_model=target_model,
                )
            else:
                result = handler.generate_complete_system(
                    modeling_prompt,
                    existing_model=target_model,
                    class_metadata=gui_class_metadata,
                )
    except Exception as exc:
        logger.error(f"❌ [ModelOp] Handler exception: {exc}", exc_info=True)
        reply_message(
            session,
            f"Something went wrong while processing your {diagram_label} request. "
            "Please try again or rephrase.",
        )
        return None
    finally:
        _progress_stop.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1)

    logger.info(
        f"✅ [ModelOp] Handler result: action={result.get('action') if isinstance(result, dict) else 'N/A'}, "
        f"has_message={bool(result.get('message')) if isinstance(result, dict) else False}"
    )
    logger.debug(f"[ModelOp] Full result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")

    if not isinstance(result, dict):
        reply_message(session, f"I could not create a valid {target_diagram_type} response.")
        return None

    if result.get("action") == "assistant_message":
        reply_message(session, result.get("message", "Something went wrong. Please try again."))
        return None

    result["diagramType"] = target_diagram_type
    diagram_id = resolve_diagram_id(request, target_diagram_type)
    if isinstance(diagram_id, str):
        result["diagramId"] = diagram_id

    if _replace_existing is not None:
        result["replaceExisting"] = bool(_replace_existing)
        logger.info(f"[ModelOp] replaceExisting={_replace_existing} (from direct parameter)")

    if _create_new_tab:
        result["createNewTab"] = True

    available_diagrams = _collect_available_diagrams(request)
    model_summary = _get_model_summary(result)
    suggestions = get_suggested_actions(
        diagram_type=target_diagram_type,
        operation_mode=operation_mode,
        available_diagrams=available_diagrams,
        model_summary=model_summary,
    )
    if suggestions:
        result["suggestedActions"] = suggestions

    logger.info(
        f"📤 [ModelOp] Sending result: action={result.get('action')}, "
        f"replaceExisting={result.get('replaceExisting', 'NOT SET')}, "
        f"keys={list(result.keys())}"
    )
    reply_payload(session, result)

    session.set(LAST_EXECUTED_DIAGRAM_TYPE, target_diagram_type)

    action_label = result.get("action", "unknown")
    record_session_action(
        session,
        f"{action_label} on {target_diagram_type} (mode={operation_mode}): "
        f"{operation_request[:80]}",
    )

    return target_diagram_type


def _collect_available_diagrams(request: AssistantRequest) -> list:
    """Collect diagram types that have at least one non-empty diagram."""
    snapshot = request.context.project_snapshot
    if not isinstance(snapshot, dict):
        return []
    diagrams = snapshot.get("diagrams")
    if not isinstance(diagrams, dict):
        return []
    available = []
    for dtype, value in diagrams.items():
        if isinstance(value, list):
            if any(isinstance(d, dict) and d.get("model") for d in value):
                available.append(dtype)
        elif isinstance(value, dict) and value.get("model"):
            available.append(dtype)
    return available


def _get_model_summary(result: dict) -> dict:
    """Summarize what was created/modified from a handler result payload."""
    if not isinstance(result, dict):
        return {}
    summary: Dict[str, Any] = {
        "action": result.get("action", "unknown"),
    }
    elements = result.get("elements")
    if isinstance(elements, dict):
        summary["element_count"] = len(elements)
        names = [
            el.get("name") or el.get("id", "")
            for el in elements.values()
            if isinstance(el, dict)
        ]
        summary["element_names"] = names[:20]
    elif isinstance(elements, list):
        summary["element_count"] = len(elements)

    relationships = result.get("relationships")
    if isinstance(relationships, (list, dict)):
        summary["relationship_count"] = (
            len(relationships) if isinstance(relationships, list)
            else len(relationships.keys()) if isinstance(relationships, dict)
            else 0
        )
    return summary
