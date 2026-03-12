"""
Execution Engine
----------------
Core model-operation execution, planned-operation dispatch, and
file-attachment processing.

Functions in this module access shared globals (LLM, diagram factory) via
:mod:`src.agent_context` at **call time**, not import time.
"""

import concurrent.futures
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from besser.agent import nlp
from besser.agent.core.session import Session

import agent_context as ctx
from protocol.adapters import parse_assistant_request
from protocol.types import AssistantRequest
from session_helpers import reply_message, reply_payload
from confirmation import model_has_elements
from diagram_handlers.factory import get_diagram_type_info
from handlers.generation_handler import handle_generation_request
from handlers.file_conversion_handler import convert_file_to_diagram_spec
from orchestrator import (
    plan_assistant_operations,
    determine_target_diagram_type,
    resolve_diagram_id,
)
from utilities.model_helpers import (
    resolve_target_model,
    resolve_object_reference_diagram,
    count_reference_classes,
    build_workspace_context_block,
    build_request_for_target,
    build_generation_request,
    extract_class_metadata,
)
from utilities.model_resolution import resolve_class_diagram
from utilities.workspace_context import record_session_action
from quality_review import review_generated_model  # noqa: F401 – kept for explicit quality-check intent
from suggestions import get_suggested_actions

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# File-attachment handling
# ------------------------------------------------------------------

def handle_file_attachments(session: Session, request: AssistantRequest) -> bool:
    """Process file attachments if present.  Returns True if attachments were handled."""
    if not request.has_attachments:
        return False

    openai_key = ctx.openai_api_key

    for attachment in request.attachments:
        logger.info(
            f"[FileConversion] Processing attachment: {attachment.filename} "
            f"({attachment.mime_type}, {len(attachment.content_b64)} b64 chars)"
        )
        result = convert_file_to_diagram_spec(
            file_content_b64=attachment.content_b64,
            filename=attachment.filename,
            llm_predict=ctx.gpt_predict_json,
            openai_api_key=openai_key,
        )
        reply_payload(session, result)

    return True


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
            request, last_intent=session.get("last_matched_intent"),
        )

    operation_mode = operation.get("mode")
    if not isinstance(operation_mode, str) or not operation_mode:
        operation_mode = default_mode

    operation_request = operation.get("request")
    if not isinstance(operation_request, str) or not operation_request.strip():
        operation_request = request.message
    operation_request = operation_request.strip()

    logger.info(
        f"[ModelOp] Executing: diagram={target_diagram_type}, mode={operation_mode}, "
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
            # Store the resolved mode in the operation so re-execution
            # always uses complete_system, not whatever the LLM planner
            # originally returned (which may have been missing or wrong).
            stored_operation = {**operation, 'mode': operation_mode}

            # Count existing tabs to determine if a new tab can be opened
            all_tabs = request.context.get_all_diagrams_of_type(target_diagram_type)
            tab_count = len(all_tabs) if all_tabs else 1
            max_tabs = 5
            can_add_tab = tab_count < max_tabs

            session.set('pending_complete_system', {
                'message': operation_request,
                'diagram_type': target_diagram_type,
                'operation': stored_operation,
                'default_mode': default_mode,
                'can_add_tab': can_add_tab,
            })

            if can_add_tab:
                reply_message(
                    session,
                    f"You already have a {target_diagram_type} model ({summary}). "
                    "Would you like me to **replace** it, **keep** it and add alongside, "
                    f"or create in a **new tab**? (You have {tab_count}/{max_tabs} tabs used)",
                )
            else:
                reply_message(
                    session,
                    f"You already have a {target_diagram_type} model ({summary}). "
                    "Would you like me to **replace** it, or **keep** it and add alongside? "
                    f"(All {max_tabs} tabs are in use)",
                )
            logger.info(
                f"[ModelOp] Asked user to confirm replace/keep"
                f"{'/new-tab' if can_add_tab else ''} for existing {target_diagram_type} "
                f"({tab_count}/{max_tabs} tabs)"
            )
            return None

    # ── GUI generation-mode choice ───────────────────────────────────────
    # When the user requests a full GUINoCodeDiagram and a class diagram
    # exists, offer them a choice between:
    #   (a) Deterministic auto-generate (one page per class, fast & stable)
    #   (b) LLM-driven generation (personalized, experimental)
    # If the user's request already contains strong customization hints,
    # skip the prompt and go straight to the LLM path.
    _CUSTOM_GUI_HINTS = {
        "chart", "dashboard", "custom", "specific", "page for",
        "sidebar", "metric", "kpi", "landing", "hero",
        "form", "layout", "only", "just", "don't include",
        "exclude", "style", "theme", "color", "dark",
        "personali", "unique", "tailored", "bespoke",
    }
    if target_diagram_type == "GUINoCodeDiagram" and operation_mode in ("complete_system", None, ""):
        _req_lower = (operation_request or "").lower()
        _wants_custom = any(hint in _req_lower for hint in _CUSTOM_GUI_HINTS)

        class_diagram_model = resolve_class_diagram(request)
        _has_class_diagram = (
            isinstance(class_diagram_model, dict)
            and isinstance(class_diagram_model.get("elements"), dict)
            and len(class_diagram_model["elements"]) > 0
        )

        if _has_class_diagram and _wants_custom:
            # Explicit customization hints → skip straight to LLM path
            logger.info("[ModelOp] Custom GUI request detected — using LLM-driven path")
            # Fall through to the normal handler below

        elif _has_class_diagram and not _skip_gui_choice:
            # No explicit customization → ask the user which approach they prefer
            session.set('pending_gui_choice', {
                'operation_request': operation_request,
                'operation': operation,
                'default_mode': default_mode,
                'diagram_type': target_diagram_type,
                '_replace_existing': _replace_existing,
            })
            reply_message(
                session,
                "How would you like me to generate the GUI?\n\n"
                "1️⃣ **Auto-generate** — Fast & deterministic. Creates one page per class "
                "with data tables and method buttons.\n"
                "2️⃣ **LLM-generated** *(experimental)* — AI-designed layout with "
                "personalized pages, navigation, and styling.\n\n"
                "Reply **auto** or **1** for the auto-generated GUI, "
                "or **llm** / **2** / **personalized** for the AI-designed version.",
            )
            logger.info("[ModelOp] Asked user to choose GUI generation mode")
            return None

    handler = ctx.diagram_factory.get_handler(target_diagram_type)
    if not handler:
        logger.warning(f"[ModelOp] No handler for diagram type: {target_diagram_type}")
        reply_message(
            session,
            f"{target_diagram_type} is not supported by the modeling handler yet.",
        )
        return None

    target_model = resolve_target_model(request, target_diagram_type)
    modeling_prompt = (
        f"{operation_request}\n\n"
        f"{build_workspace_context_block(request, target_diagram_type, target_model)}"
    )

    # ── Resolve class metadata for GUI diagram (charts/tables need it) ──
    gui_class_metadata = None
    if target_diagram_type == "GUINoCodeDiagram":
        class_diagram_model = resolve_class_diagram(request)
        if isinstance(class_diagram_model, dict):
            gui_class_metadata = extract_class_metadata(class_diagram_model)
            if gui_class_metadata:
                logger.info(
                    f"[ModelOp] Resolved {len(gui_class_metadata)} class(es) for GUI chart binding"
                )

    logger.debug(f"[ModelOp] Modeling prompt ({len(modeling_prompt)} chars): {modeling_prompt[:300]!r}")
    logger.debug(f"[ModelOp] Target model present: {target_model is not None}, type: {type(target_model).__name__}")

    if operation_mode == "single_element":
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
            result = handler.generate_single_element(
                modeling_prompt,
                reference_diagram=reference_diagram,
                existing_model=target_model,
            )
        else:
            result = handler.generate_single_element(
                modeling_prompt,
                existing_model=target_model,
                class_metadata=gui_class_metadata,
            )
    elif operation_mode == "modify_model":
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

    logger.info(
        f"[ModelOp] Handler result: action={result.get('action') if isinstance(result, dict) else 'N/A'}, "
        f"has_message={bool(result.get('message')) if isinstance(result, dict) else False}"
    )
    logger.debug(f"[ModelOp] Full result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")

    if not isinstance(result, dict):
        reply_message(session, f"I could not create a valid {target_diagram_type} response.")
        return None

    # If the handler returned an error message (e.g. LLM failure), send it
    # directly without injecting diagram metadata.
    if result.get("action") == "assistant_message":
        reply_message(session, result.get("message", "Something went wrong. Please try again."))
        return None

    result["diagramType"] = target_diagram_type
    diagram_id = resolve_diagram_id(request, target_diagram_type)
    if isinstance(diagram_id, str):
        result["diagramId"] = diagram_id

    # Propagate replaceExisting flag
    if _replace_existing is not None:
        result["replaceExisting"] = bool(_replace_existing)
        logger.info(f"[ModelOp] replaceExisting={_replace_existing} (from direct parameter)")
    else:
        replace_flag = session.get('_replace_existing_model')
        if replace_flag is not None:
            result["replaceExisting"] = bool(replace_flag)
            session.set('_replace_existing_model', None)
            logger.info(f"[ModelOp] replaceExisting={replace_flag} (from session variable)")

    # Signal the frontend to create a new tab before injecting
    if _create_new_tab:
        result["createNewTab"] = True

    # Attach contextual suggestions to the payload
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
        f"[ModelOp] Sending result: action={result.get('action')}, "
        f"replaceExisting={result.get('replaceExisting', 'NOT SET')}, "
        f"keys={list(result.keys())}"
    )
    reply_payload(session, result)

    # Track for "what's next" suggestions in state bodies
    session.set('_last_executed_diagram_type', target_diagram_type)

    # Record action in session history for context-aware prompts
    action_label = result.get("action", "unknown")
    record_session_action(
        session,
        f"{action_label} on {target_diagram_type} (mode={operation_mode}): "
        f"{operation_request[:80]}",
    )

    # Quality review is opt-in — only run when user explicitly asks
    # (e.g., "review my model", "check quality").  Removed automatic
    # post-generation review to keep responses fast and focused.

    return target_diagram_type


def _collect_available_diagrams(request: AssistantRequest) -> list:
    """Collect diagram types that have at least one non-empty diagram in the project snapshot.

    Handles both the legacy single-dict format and the multi-tab array format
    where each diagram type maps to a list of ProjectDiagram objects.
    """
    snapshot = request.context.project_snapshot
    if not isinstance(snapshot, dict):
        return []
    diagrams = snapshot.get("diagrams")
    if not isinstance(diagrams, dict):
        return []
    available = []
    for dtype, value in diagrams.items():
        if isinstance(value, list):
            # Multi-tab: include type only when at least one tab has a model
            if any(isinstance(d, dict) and d.get("model") for d in value):
                available.append(dtype)
        elif isinstance(value, dict) and value.get("model"):
            available.append(dtype)
    return available


def _get_model_summary(result: dict) -> dict:
    """Summarize what was created/modified from a handler result payload.

    Extracts high-level statistics (class count, relationship count, element
    names) so downstream suggestion logic can tailor recommendations without
    needing the full result.
    """
    if not isinstance(result, dict):
        return {}
    summary: Dict[str, Any] = {
        "action": result.get("action", "unknown"),
    }
    elements = result.get("elements")
    if isinstance(elements, dict):
        summary["element_count"] = len(elements)
        # Collect top-level element names (classes, states, screens, etc.)
        names = [
            el.get("name") or el.get("id", "")
            for el in elements.values()
            if isinstance(el, dict)
        ]
        summary["element_names"] = names[:20]  # cap to avoid bloated payloads
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


# ------------------------------------------------------------------
# Structured error payloads
# ------------------------------------------------------------------

def _build_error_payload(operation: dict, error: Exception, error_code: str = "unknown") -> dict:
    """Build a structured error payload with recovery hints."""
    return {
        "action": "agent_error",
        "errorCode": error_code,
        "message": str(error),
        "operation": {
            "type": operation.get("type"),
            "diagramType": operation.get("diagramType"),
            "mode": operation.get("mode"),
        },
        "suggestedRecovery": _get_recovery_hint(error_code),
        "retryable": error_code in ("llm_failure", "parse_error", "timeout"),
    }


def _get_recovery_hint(error_code: str) -> str:
    """Return a user-facing hint for recovering from a specific error type."""
    hints = {
        "llm_failure": "try again",
        "parse_error": "try rephrasing more specifically",
        "validation_error": "try simplifying the model",
        "timeout": "try breaking into smaller steps",
        "prerequisite_missing": "create the required diagram first",
        "handler_missing": "this diagram type is not yet supported",
        "generation_handler_error": "try regenerating — if it persists, check the model for issues",
    }
    return hints.get(error_code, "try rephrasing your request")


def _classify_error(error: Exception) -> str:
    """Map an exception to a structured error code."""
    err_name = type(error).__name__.lower()
    err_msg = str(error).lower()

    if "timeout" in err_name or "timeout" in err_msg:
        return "timeout"
    if "parse" in err_name or "json" in err_name or "decode" in err_msg:
        return "parse_error"
    if "validation" in err_name or "invalid" in err_msg:
        return "validation_error"
    if any(kw in err_msg for kw in ("openai", "llm", "rate limit", "api")):
        return "llm_failure"
    return "unknown"


# ------------------------------------------------------------------
# Progress reporting
# ------------------------------------------------------------------

def _report_progress(session: Session, current_idx: int, total: int, operation: dict):
    """Send a progress update to the user for multi-step plans."""
    op_type = operation.get("type", "unknown")
    diagram_type = operation.get("diagramType", "")

    if total > 1:
        progress_msg = f"Step {current_idx + 1}/{total}: "
        if op_type == "model":
            progress_msg += f"Creating {diagram_type}..." if diagram_type else "Processing model..."
        elif op_type == "generation":
            gen_type = operation.get("generatorType", "code")
            progress_msg += f"Generating {gen_type}..."
        else:
            progress_msg += "Processing..."

        # Send as a lightweight status message
        reply_message(session, progress_msg)


# ------------------------------------------------------------------
# Parallel operation dispatch helpers
# ------------------------------------------------------------------

def _can_run_parallel(operations: List[dict]) -> Tuple[List[List[dict]], List[dict]]:
    """Split operations into parallel-safe groups.

    Model operations for DIFFERENT diagram types with no dependencies can run
    in parallel.  Generation ops always run after their prerequisite model ops.

    Returns
    -------
    independent_model_groups : list[list[dict]]
        Each inner list is a group of model ops for one diagram type.
    gen_ops : list[dict]
        Generation operations, to be executed sequentially after all model ops.
    """
    model_ops = [op for op in operations if isinstance(op, dict) and op.get("type") == "model"]
    gen_ops = [op for op in operations if isinstance(op, dict) and op.get("type") == "generation"]

    # Model ops for different diagram types are independent
    independent_model_groups: Dict[str, List[dict]] = {}
    for op in model_ops:
        dt = op.get("diagramType", "unknown")
        independent_model_groups.setdefault(dt, []).append(op)

    return list(independent_model_groups.values()), gen_ops


# ------------------------------------------------------------------
# Planned-operation dispatch
# ------------------------------------------------------------------

def execute_planned_operations(
    session: Session,
    request: AssistantRequest,
    default_mode: str,
    matched_intent: Optional[str],
) -> None:
    """Run the orchestrator planner and dispatch each resulting operation.

    Improvements over sequential-only execution:
    - Independent model operations (different diagram types) run in parallel
    - Structured error payloads with recovery hints replace generic messages
    - Progress messages keep the user informed during multi-step plans
    - Suggestion attachments are enriched with model summaries
    """
    operations = plan_assistant_operations(
        request=request,
        default_mode=default_mode,
        matched_intent=matched_intent,
        llm_predict=ctx.gpt_predict_json,
    )

    if not operations:
        reply_message(session, "I couldn't determine an execution plan from your request.")
        return

    # Split into parallel-safe model groups and sequential generation ops
    model_groups, gen_ops = _can_run_parallel(operations)
    total_steps = sum(len(g) for g in model_groups) + len(gen_ops)

    working_request = request
    step_counter = 0

    # ── Phase 1: Model operations ────────────────────────────────────
    # Multiple independent diagram-type groups can run in parallel.
    # Within each group, operations run sequentially (same diagram type).
    if len(model_groups) > 1:
        # Parallel execution across independent diagram-type groups
        logger.info(
            f"[PlannedOps] Running {len(model_groups)} independent model group(s) in parallel"
        )

        # We need to detect if any group triggers a pending confirmation.
        # If so, we must store remaining ops and halt.
        confirmation_triggered = False
        all_flat_ops = [op for group in model_groups for op in group]

        def _run_model_group(group: List[dict]) -> List[Tuple[dict, Optional[str], Optional[Exception]]]:
            """Execute a group of model ops sequentially, return results."""
            results = []
            for op in group:
                try:
                    executed = execute_model_operation(
                        session, working_request, op, default_mode=default_mode,
                    )
                    results.append((op, executed, None))
                except Exception as exc:
                    results.append((op, None, exc))
            return results

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(model_groups), 4),
        ) as executor:
            futures = {
                executor.submit(_run_model_group, group): group
                for group in model_groups
            }
            for future in concurrent.futures.as_completed(futures):
                group = futures[future]
                try:
                    group_results = future.result()
                except Exception as exc:
                    # Entire group failed — report structured error for each op
                    for op in group:
                        step_counter += 1
                        _report_progress(session, step_counter - 1, total_steps, op)
                        error_code = _classify_error(exc)
                        error_payload = _build_error_payload(op, exc, error_code)
                        reply_payload(session, error_payload)
                        logger.error(f"[PlannedOps] Parallel group error: {exc}")
                    continue

                for op, executed_target, error in group_results:
                    step_counter += 1
                    _report_progress(session, step_counter - 1, total_steps, op)

                    if error is not None:
                        error_code = _classify_error(error)
                        error_payload = _build_error_payload(op, error, error_code)
                        reply_payload(session, error_payload)
                        logger.error(f"[PlannedOps] Model op error: {error}")
                        continue

                    if executed_target is None:
                        # Pending confirmation — store remaining ops
                        remaining_ops = gen_ops[:]  # gen ops always remain
                        _store_remaining_ops(session, remaining_ops, request)
                        confirmation_triggered = True
                        continue

                    if isinstance(executed_target, str) and executed_target:
                        working_request = build_request_for_target(
                            working_request, executed_target,
                        )

        if confirmation_triggered:
            logger.info("[PlannedOps] Pending confirmation stored — halting remaining operations")
            return

    else:
        # Single group (or empty) — sequential execution, no thread overhead
        flat_model_ops = model_groups[0] if model_groups else []
        all_ops_flat = flat_model_ops + gen_ops

        for idx, operation in enumerate(flat_model_ops):
            step_counter += 1
            _report_progress(session, step_counter - 1, total_steps, operation)

            try:
                executed_target = execute_model_operation(
                    session, working_request, operation, default_mode=default_mode,
                )
                if executed_target is None:
                    # Pending confirmation stored — save remaining ops so they
                    # can be resumed after the user confirms.
                    remaining = (
                        [op for op in flat_model_ops[idx + 1:] if isinstance(op, dict)]
                        + gen_ops
                    )
                    _store_remaining_ops(session, remaining, request)
                    logger.info("[PlannedOps] Pending confirmation stored — halting remaining operations")
                    return
                if isinstance(executed_target, str) and executed_target:
                    working_request = build_request_for_target(working_request, executed_target)
            except Exception as error:
                error_code = _classify_error(error)
                error_payload = _build_error_payload(operation, error, error_code)
                reply_payload(session, error_payload)
                logger.error(f"[PlannedOps] Model op error ({error_code}): {error}")
            continue

    # ── Phase 2: Generation operations (always sequential) ───────────
    for operation in gen_ops:
        step_counter += 1
        _report_progress(session, step_counter - 1, total_steps, operation)

        generator_type = operation.get("generatorType")
        if not isinstance(generator_type, str) or not generator_type:
            continue

        generation_message = operation.get("request") if isinstance(operation.get("request"), str) else None
        generation_request = build_generation_request(
            working_request,
            generator_type=generator_type,
            config=operation.get("config") if isinstance(operation.get("config"), dict) else {},
            message_override=generation_message,
        )
        try:
            response_payload = handle_generation_request(session, generation_request)
        except Exception as error:
            error_code = _classify_error(error)
            response_payload = _build_error_payload(operation, error, error_code)
            logger.error(f"[PlannedOps] Generation error ({error_code}): {error}")

        if isinstance(response_payload, dict):
            # Attach suggestions after code generation
            gen_suggestions = get_suggested_actions(
                diagram_type="",
                operation_mode="generation",
                available_diagrams=_collect_available_diagrams(working_request),
                generator_type=generator_type,
            )
            if gen_suggestions:
                response_payload["suggestedActions"] = gen_suggestions
            reply_payload(session, response_payload)
        elif isinstance(response_payload, str):
            reply_message(session, response_payload)


def _store_remaining_ops(
    session: Session, remaining: List[dict], request: AssistantRequest,
) -> None:
    """Persist remaining operations alongside a pending confirmation."""
    remaining = [op for op in remaining if isinstance(op, dict)]
    if remaining:
        pending = session.get('pending_complete_system')
        if isinstance(pending, dict):
            pending['remaining_operations'] = remaining
            pending['original_message'] = request.message
            session.set('pending_complete_system', pending)
            logger.info(
                f"[PlannedOps] Stored {len(remaining)} remaining operation(s) "
                f"alongside pending confirmation"
            )
