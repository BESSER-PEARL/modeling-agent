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
from utilities.model_context import is_diagram_nontrivial
from suggestions import get_suggested_actions, get_artifact_label, get_post_spec_suggestions
from session_keys import (
    LAST_EXECUTED_DIAGRAM_TYPE,
    LAST_MATCHED_INTENT,
    MISMATCH_REGEN_PENDING,
    PENDING_COMPLETE_SYSTEM,
    PENDING_GUI_CHOICE,
    PENDING_SMART_GEN_INSTRUCTIONS,
    PENDING_SMART_GEN_PROVIDER,
    PENDING_SMART_GEN_TIMESTAMP,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# In-turn creation → snapshot bridge (empty-workspace guard fix)
#
# A create/complete-system op pushes the freshly built model STRAIGHT to the
# frontend; it never round-trips through the backend's project snapshot. When a
# single user turn is planned into "create diagram → generate code", the later
# generate step reads the pre-create (often empty) snapshot and wrongly refuses
# with "your workspace looks empty — there's no model to turn into code yet".
#
# These helpers record a lightweight canonical copy of the just-created model
# back into the working request's project snapshot so a later generate step can
# validate its diagram prerequisites. The frontend remains authoritative.
# ------------------------------------------------------------------

def _elements_from_result(
    result_payload: Any,
    diagram_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract a non-empty ``elements`` map from a handler result payload.

    Understands class and agent ``systemSpec`` payloads, an editor-model
    ``model.elements`` shape, and a single-``element`` shape. Falls back to a
    marker for legacy callers; the snapshot bridge validates that marker
    against the requested diagram type before recording it.
    """
    if not isinstance(result_payload, dict):
        return {}

    # Editor-model style: {"model": {"elements": {...}}}
    model = result_payload.get("model")
    if isinstance(model, dict) and isinstance(model.get("elements"), dict):
        return dict(model["elements"])

    # Class-diagram style: {"systemSpec": {"classes": [...], ...}}
    spec = result_payload.get("systemSpec")
    if isinstance(spec, dict):
        elements: Dict[str, Any] = {}
        if diagram_type == "AgentDiagram":
            for index, state in enumerate(spec.get("states") or []):
                if not isinstance(state, dict):
                    continue
                name = state.get("stateName") or state.get("name") or f"state_{index}"
                elements[f"created-state-{index}-{name}"] = {
                    "type": "AgentState",
                    "name": str(name),
                }
            for index, intent in enumerate(spec.get("intents") or []):
                if not isinstance(intent, dict):
                    continue
                name = intent.get("intentName") or intent.get("name") or f"intent_{index}"
                elements[f"created-intent-{index}-{name}"] = {
                    "type": "AgentIntent",
                    "name": str(name),
                }
            return elements

        classes = spec.get("classes")
        if isinstance(classes, list):
            for i, cls in enumerate(classes):
                if not isinstance(cls, dict):
                    continue
                name = cls.get("className") or cls.get("name") or f"class_{i}"
                elements[f"created-{i}-{name}"] = {"type": "Class", "name": str(name)}
        # May be {} when the spec had no classes → guard should still fire.
        return elements

    # Single-element style: {"element": {...}}
    element = result_payload.get("element")
    if isinstance(element, dict):
        name = element.get("className") or element.get("name") or "element"
        return {f"created-{name}": {"type": "Class", "name": str(name)}}

    # Unrecognized but successful create → mark non-empty with a placeholder so
    # a later generate step doesn't think the workspace is empty.
    return {"created-marker": {"type": "Element"}}


def _record_created_model_in_snapshot(
    context: Any, diagram_type: str, result_payload: Dict[str, Any],
) -> bool:
    """Make an in-turn creation visible to a later generate step in the same plan.

    Records a lightweight representation of the just-created model under
    ``context.project_snapshot["diagrams"][diagram_type]`` so a generate op
    planned in the SAME turn passes ``_project_has_any_model``.

    Returns ``True`` when the snapshot was updated. No-op (``False``) when the
    context/type is missing, the payload produced nothing, or a non-empty model
    for that type is already present in the snapshot.
    """
    if context is None or not isinstance(diagram_type, str) or not diagram_type:
        return False

    def _entry_is_nonempty(entry: Any) -> bool:
        return (
            isinstance(entry, dict)
            and isinstance(entry.get("model"), dict)
            and is_diagram_nontrivial(entry["model"], diagram_type)
        )

    snapshot = getattr(context, "project_snapshot", None)
    if not isinstance(snapshot, dict):
        snapshot = {}
        try:
            context.project_snapshot = snapshot
        except Exception:  # pragma: no cover — context without a settable field
            return False

    diagrams = snapshot.get("diagrams")
    if not isinstance(diagrams, dict):
        diagrams = {}
        snapshot["diagrams"] = diagrams

    existing = diagrams.get(diagram_type)
    existing_entries = (
        existing if isinstance(existing, list)
        else ([existing] if existing is not None else [])
    )
    if any(_entry_is_nonempty(e) for e in existing_entries):
        # A non-empty model for this type is already in the snapshot; the guard
        # already passes, so there is nothing to bridge.
        return False

    direct_model = result_payload.get("model")
    if isinstance(direct_model, dict) and is_diagram_nontrivial(
        direct_model, diagram_type,
    ):
        bridged_model = dict(direct_model)
    else:
        elements = _elements_from_result(result_payload, diagram_type)
        if not elements:
            # The create genuinely produced nothing — leave the guard to fire.
            return False
        bridged_model = {"elements": elements}

    if not is_diagram_nontrivial(bridged_model, diagram_type):
        # Do not let a generic placeholder satisfy a typed prerequisite (for
        # example, GUI readiness requires canonical ``pages`` content).
        return False

    diagrams[diagram_type] = [{"model": bridged_model}]
    logger.info(
        f"[ModelOp] Recorded in-turn {diagram_type} creation into snapshot "
        "so a later generate step can validate its prerequisites."
    )
    return True


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
# Destructive modify-model guard
#
# A 1544-question QA run found that the vague correction "that's wrong,
# redo it" on an 11-class CRM class diagram produced a modify_model plan
# with 49 remove_element operations that wiped the ENTIRE model -- applied
# silently, with no confirmation. Every diagram handler's generate_modification
# (see diagram_handlers/core/base_handler.py::_execute_modification) returns
# either a single "modification" dict or a batch "modifications" list of
# {action, target, changes} entries; when that batch's *net effect* is to
# remove most/all of the existing top-level elements, we must ask before
# applying it instead of trusting the LLM's plan blindly.
# ------------------------------------------------------------------

# Target fields that scope a remove_element to a CHILD of a top-level
# element (an attribute, method, relationship, or transition endpoint)
# rather than to the top-level element itself. Deleting ONE class together
# with its relationships requires several remove_element entries -- one per
# relationship, per class_diagram_handler's REMOVE_ELEMENT_RULE -- so these
# must be excluded from the "how many top-level elements would this remove"
# tally below, or a normal single-class deletion would be over-guarded.
_CHILD_SCOPE_TARGET_KEYS = (
    "attributeName", "attributeId",
    "methodName", "methodId",
    "relationshipName", "relationshipId",
    "sourceClass", "targetClass",
    "sourceStateName", "targetStateName",
    "transitionName", "transitionId",
)

# Element "type" values that are always children of another element
# (mirrors the convention used by the layout engine's occupied-rect
# extraction) -- these never count as a top-level element when tallying
# the EXISTING model either.
_CHILD_ELEMENT_TYPES = {
    "ClassAttribute", "ClassMethod",
    "AgentStateBody", "AgentStateFallbackBody", "AgentIntentBody",
}

# Any ONE of these being true means the modify_model plan would destroy a
# large fraction of the existing model:
#   - 3+ top-level elements removed in a single plan, or
#   - the removal would clear the model entirely, or
#   - 2+ removed AND that is at least half of what currently exists.
# The >=2 floor on the fraction rule keeps it from flagging e.g. "remove 1
# of my 2 classes" (a normal single edit) while still catching "remove 1 of
# my 1 class" via the clears-the-model rule.
_DESTRUCTIVE_MIN_REMOVED = 3
_DESTRUCTIVE_MIN_REMOVED_FOR_FRACTION = 2
_DESTRUCTIVE_FRACTION = 0.5


def _is_top_level_removal(target: Any) -> bool:
    """True when a remove_element target identifies a top-level element
    (a class/state/object/intent) rather than one of its children."""
    if not isinstance(target, dict) or not target:
        return False
    if any(target.get(key) for key in _CHILD_SCOPE_TARGET_KEYS):
        return False
    return any(
        isinstance(value, str) and value.strip()
        for value in target.values()
    )


def _count_removed_top_level_elements(result: Dict[str, Any]) -> int:
    """Count how many TOP-LEVEL elements a modify_model result would delete.

    Handles both the single-``modification`` and batch-``modifications``
    shapes emitted by every diagram handler's ``generate_modification``.
    Returns 0 (never crashes) when the result doesn't have either shape --
    e.g. GUINoCodeDiagram's modify path returns an already-applied ``model``
    instead, and only ever performs one edit at a time, so it's out of
    scope for this batch-plan guard.
    """
    if not isinstance(result, dict):
        return 0
    mods = result.get("modifications")
    if not isinstance(mods, list):
        single = result.get("modification")
        mods = [single] if isinstance(single, dict) else []

    return sum(
        1
        for mod in mods
        if isinstance(mod, dict)
        and mod.get("action") == "remove_element"
        and _is_top_level_removal(mod.get("target"))
    )


def _count_existing_top_level_elements(model: Optional[Dict[str, Any]]) -> int:
    """Count top-level (non-child) elements in the CURRENT model."""
    if not isinstance(model, dict):
        return 0
    elements = model.get("elements")
    if not isinstance(elements, dict):
        return 0
    count = 0
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        if element.get("type") in _CHILD_ELEMENT_TYPES:
            continue
        if element.get("owner"):
            continue
        count += 1
    return count


def _is_mass_deletion(removed: int, existing: int) -> bool:
    """True when *removed* top-level elements is a destructive fraction of
    *existing* -- see threshold rationale in the constants above."""
    if removed <= 0:
        return False
    if removed >= _DESTRUCTIVE_MIN_REMOVED:
        return True
    if existing > 0 and removed >= existing:
        return True
    if (
        removed >= _DESTRUCTIVE_MIN_REMOVED_FOR_FRACTION
        and existing > 0
        and (removed / existing) >= _DESTRUCTIVE_FRACTION
    ):
        return True
    return False


def _build_destructive_modify_confirmation(
    session: Session,
    target_diagram_type: str,
    removed_count: int,
    existing_count: int,
    result: Dict[str, Any],
) -> None:
    """Ask the user to confirm a modify_model plan that would wipe out most
    or all of the existing model, instead of silently applying it.

    Stores the ALREADY-COMPUTED ``result`` payload (not an operation to
    re-run): re-invoking the LLM on confirm could produce a different plan
    than the one the user just approved, so ``handle_pending_system_
    confirmation`` in confirmation.py sends this exact payload back on a
    "confirm" answer, or discards it entirely otherwise.
    """
    pending_data = {
        "destructive_modify": True,
        "diagram_type": target_diagram_type,
        "precomputed_payload": result,
    }
    session.set(PENDING_COMPLETE_SYSTEM, pending_data)

    if existing_count > 0:
        detail = f"remove {removed_count} of your {existing_count} existing element(s)"
    else:
        detail = f"remove {removed_count} element(s)"

    message = (
        f"That change would {detail} — most or all of your current "
        f"{target_diagram_type}. Since the instruction was short, I want to "
        "confirm before applying such a large change. **Confirm** to go "
        "ahead, or **cancel** to keep your model as is."
    )
    reply_payload(session, {
        "action": "assistant_message",
        "message": message,
        "suggestedActions": [
            {"label": "Confirm — apply the change", "prompt": "confirm"},
            {"label": "Cancel — keep my model", "prompt": "cancel"},
        ],
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

    # ── Modify-without-target guard ──────────────────────────────────────
    # A modify_model op on a flow-style diagram only makes sense when that
    # diagram already exists with content. When it doesn't (e.g. the user
    # asks to "add an agent/chatbot to the app" while sitting on the
    # class/GUI diagram — so the request resolves to an AgentDiagram that
    # hasn't been created yet), promote the op to complete_system so the
    # diagram is actually generated instead of failing in
    # generate_modification on an empty model.
    #
    # Scoped to AgentDiagram / StateMachineDiagram / QuantumCircuitDiagram:
    # their generate_modification needs an existing structure to edit. The
    # ClassDiagram / ObjectDiagram / GUI handlers already create elements
    # from an empty model on modify (e.g. "create a class called User"), so
    # they are intentionally excluded to avoid regressing single-element
    # creation into a full-system build.
    _PROMOTE_MODIFY_WHEN_EMPTY = {
        "AgentDiagram",
        "StateMachineDiagram",
        "QuantumCircuitDiagram",
    }
    _promoted_modify_to_complete = False
    if operation_mode == "modify_model" and target_diagram_type in _PROMOTE_MODIFY_WHEN_EMPTY:
        _existing = resolve_target_model(request, target_diagram_type)
        if not model_has_elements(_existing):
            logger.info(
                "[ModelOp] No existing %s to modify — promoting modify_model "
                "to complete_system so the diagram is created.",
                target_diagram_type,
            )
            operation_mode = "complete_system"
            _promoted_modify_to_complete = True

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
                source_description=f"I can create a new {target_diagram_type}",
            )
            logger.info(
                f"[ModelOp] Asked user to confirm replace/keep for existing {target_diagram_type}"
            )
            return None

    # ── GUI generation-mode choice ───────────────────────────────────────
    # NOTE: pure scoping/filler words ("only", "just") and over-generic ones
    # ("form", "layout", "style") were removed — they forced the experimental
    # custom-GUI path on plain requests like "create a GUI for just the Product
    # class". Keep only hints that genuinely signal a custom/bespoke GUI.
    _CUSTOM_GUI_HINTS = {
        "chart", "dashboard", "custom", "specific", "page for",
        "sidebar", "metric", "kpi", "landing", "hero",
        "don't include", "exclude", "theme", "color", "dark",
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
                    "2️⃣ **AI-generated** *(experimental)* — AI-designed layout with "
                    "personalized pages, navigation, and styling."
                ),
                "suggestedActions": [
                    {"label": "Auto-generate", "prompt": "auto"},
                    {"label": "AI-generated (experimental)", "prompt": "llm"},
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

    # Inject conversation context for multi-turn awareness: the rolling
    # SUMMARY of older turns (so the agent remembers beyond the recent
    # window — the summary was previously computed but never fed to the
    # LLM) PLUS the last CONVERSATION_HISTORY_DEPTH messages verbatim.
    conversation_context = ""
    if not _skip_existing_check:
        try:
            from memory import get_memory, memory_session_key
            # Stable payload sessionId so memory survives reconnects (B-5).
            session_id = memory_session_key(session, request)
            mem = get_memory(session_id)
            summary = (mem.get_summary() or "").strip()
            recent = mem.get_last_n(CONVERSATION_HISTORY_DEPTH)
            blocks = []
            if summary:
                blocks.append(
                    "Summary of earlier conversation (remember what the user has "
                    f"already created or discussed):\n  {summary}"
                )
            if recent and len(recent) > 1:
                history_lines = []
                for msg in recent[:-1]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")[:200]
                    history_lines.append(f"  {role}: {content}")
                if history_lines:
                    blocks.append(
                        "Recent messages (oldest first):\n" + "\n".join(history_lines)
                    )
            if blocks:
                conversation_context = (
                    "Recent conversation context (use this to understand what the "
                    "user has been working on):\n"
                    + "\n\n".join(blocks)
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
            # ``raw_request`` lets handlers distinguish the user's actual
            # message from the context-enriched modeling prompt (used for the
            # two-pass fast-path length check).
            extra_kwargs: Dict[str, Any] = {
                "class_metadata": gui_class_metadata,
                "raw_request": operation_request,
            }
            if target_diagram_type == "ObjectDiagram":
                reference_diagram = resolve_object_reference_diagram(request, target_model)
                reference_class_count = count_reference_classes(reference_diagram)
                if reference_class_count > 0:
                    logger.info(
                        f"[ModelOp] ObjectDiagram modify reference resolved with {reference_class_count} class(es)."
                    )
                elif not model_has_elements(target_model):
                    # No class diagram to instantiate from AND the object
                    # diagram is empty — this modify would create the first,
                    # unlinked object. Apply the same guard the complete_system
                    # path uses instead of silently producing a dangling object
                    # (#54). Edits to an EXISTING object diagram still proceed.
                    logger.warning(
                        "[ModelOp] ObjectDiagram modify with no reference classes "
                        "and no existing objects — blocking unlinked object creation."
                    )
                    reply_message(
                        session,
                        "Please create a **Class Diagram** first — Object Diagrams "
                        "need class definitions to instantiate from.",
                    )
                    return None
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
                    raw_request=operation_request,
                )
            else:
                result = handler.generate_complete_system(
                    modeling_prompt,
                    existing_model=target_model,
                    class_metadata=gui_class_metadata,
                    raw_request=operation_request,
                )
    except Exception as exc:
        logger.error(f"❌ [ModelOp] Handler exception: {exc}", exc_info=True)
        # Smart message for provider rate-limit / auth failures. When the
        # SHARED server key hits its limit and the user has NOT supplied their
        # own key, prompt them to add one (BYOK); when the user's OWN key
        # fails, tell them to check it. errorCode (rate_limit / auth_error) is
        # carried so the frontend can offer "Add your API key".
        from errors import classify_error, ErrorCode
        try:
            _code = classify_error(exc)
        except Exception:
            _code = ErrorCode.UNKNOWN
        try:
            import byok
            _byok_active = byok.is_active()
        except Exception:
            _byok_active = False
        # Emit action='agent_error' with errorCode so the frontend surfaces an
        # inline "Add your API key" button (it keys on rate_limit/auth_error).
        if _byok_active and _code in (ErrorCode.RATE_LIMIT, ErrorCode.AUTH_ERROR):
            reply_payload(session, {
                "action": "agent_error",
                "errorCode": "auth_error",
                "message": (
                    "Your API key was rejected or hit its rate limit. Check the "
                    "key (the key icon in the assistant) and try again."
                ),
                "retryable": True,
                "suggestedRecovery": "Check your API key",
            })
        elif _code == ErrorCode.RATE_LIMIT:
            reply_payload(session, {
                "action": "agent_error",
                "errorCode": "rate_limit",
                "message": (
                    "We've hit the shared free usage limit for the AI service. "
                    "Add your own API key (the key icon in the assistant) to keep "
                    "going — it stays in your browser and is used only for your "
                    "requests."
                ),
                "retryable": True,
                "suggestedRecovery": "Add your own API key",
            })
        elif _code == ErrorCode.AUTH_ERROR:
            reply_payload(session, {
                "action": "agent_error",
                "errorCode": "auth_error",
                "message": (
                    "The AI service is temporarily unavailable. Please try again "
                    "shortly, or add your own API key in the assistant settings."
                ),
                "retryable": False,
                "suggestedRecovery": "Try again shortly, or add your own API key",
            })
        else:
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

    # When a modify_model op was silently promoted to a full complete_system
    # build (no existing diagram to edit), tell the user the scope changed so
    # they aren't surprised by a whole new diagram instead of a small edit (#60).
    if _promoted_modify_to_complete and isinstance(result.get("message"), str):
        result["message"] = (
            f"There wasn't an existing {diagram_label} to modify, so I created a "
            f"new one instead. " + result["message"]
        )

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

    # For complete-system creations, detect the intended artifact type from the
    # original user message and append an artifact-aware follow-up sentence +
    # replace the generic buttons with artifact-specific ones.
    if operation_mode == "complete_system" and isinstance(result.get("message"), str):
        from handlers.generation_handler import detect_generator_type  # lazy to avoid circular import
        _detected_gen = detect_generator_type(request.message)
        _artifact = get_artifact_label(_detected_gen)
        result["message"] += (
            f"\n\nYou can now review or refine the specification, or continue "
            f"with generating your {_artifact}. What would you like to do?"
        )
        result["suggestedActions"] = get_post_spec_suggestions(_detected_gen)

    # ── Destructive modify-model guard ───────────────────────────────────
    # Block a modify_model plan whose net effect would delete most/all of
    # the existing top-level elements (see the guard section above); ask
    # for confirmation instead of applying it silently.
    if operation_mode == "modify_model":
        _removed_count = _count_removed_top_level_elements(result)
        if _removed_count > 0:
            _existing_count = _count_existing_top_level_elements(target_model)
            if _is_mass_deletion(_removed_count, _existing_count):
                logger.warning(
                    f"[ModelOp] Blocking destructive modify_model on {target_diagram_type}: "
                    f"would remove {_removed_count} top-level element(s) "
                    f"(existing={_existing_count}) — asking for confirmation "
                    "instead of applying silently."
                )
                _build_destructive_modify_confirmation(
                    session=session,
                    target_diagram_type=target_diagram_type,
                    removed_count=_removed_count,
                    existing_count=_existing_count,
                    result=result,
                )
                return None

    logger.info(
        f"📤 [ModelOp] Sending result: action={result.get('action')}, "
        f"replaceExisting={result.get('replaceExisting', 'NOT SET')}, "
        f"keys={list(result.keys())}"
    )
    reply_payload(session, result)

    # Bridge this in-turn creation into the working request's snapshot so a
    # generate op planned later in the SAME turn doesn't read the pre-create
    # (empty) snapshot and wrongly refuse with "your workspace looks empty".
    # Only genuine creations (not modify_model, which implies a pre-existing
    # model) update the snapshot here.
    if result.get("action") in ("inject_complete_system", "inject_element"):
        _record_created_model_in_snapshot(
            getattr(request, "context", None), target_diagram_type, result,
        )

    session.set(LAST_EXECUTED_DIAGRAM_TYPE, target_diagram_type)

    # ── Mismatch "Update model + generate" resume ────────────────────────
    # When this build is the model-rebuild half of the domain-mismatch
    # "Update model + generate" quick action, fire the stashed Spec-Driven
    # handoff now so the "+ generate" half is actually honored — otherwise
    # the user is left to click "Generate application" again (the button
    # over-promises). One-shot: the flag is consumed here regardless, so an
    # ordinary create never triggers this. Still freshness-gated, and skipped
    # on an explicit "keep" (which deliberately preserves the old model).
    if (
        operation_mode == "complete_system"
        and result.get("action") == "inject_complete_system"
        and _replace_existing is not False
        and bool(session.get(MISMATCH_REGEN_PENDING))
    ):
        session.delete(MISMATCH_REGEN_PENDING)  # consume once
        try:
            from handlers.generation_handler import (
                _build_smart_gen_confirmation,
                _smart_gen_stash_is_fresh,
            )

            _stash = session.get(PENDING_SMART_GEN_INSTRUCTIONS)
            _fresh = _smart_gen_stash_is_fresh(session.get(PENDING_SMART_GEN_TIMESTAMP))
            if isinstance(_stash, str) and _stash.strip() and _fresh:
                logger.info("[ModelOp] Resuming stashed smart-gen after mismatch rebuild")
                _payload = _build_smart_gen_confirmation(
                    session,
                    _stash,
                    session.get(PENDING_SMART_GEN_PROVIDER) or "anthropic",
                    reason_prefix="Model rebuilt and ready.",
                )
                if isinstance(_payload, dict):
                    reply_payload(session, _payload)
        except Exception:
            logger.exception("[ModelOp] mismatch smart-gen resume failed")

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
