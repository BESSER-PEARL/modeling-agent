"""
Execution Engine
----------------
Core model-operation execution, planned-operation dispatch, and
file-attachment processing.

Functions in this module access shared globals (LLM, diagram factory) via
:mod:`src.agent_context` at **call time**, not import time.
"""

import json
import logging
from typing import Any, Dict, Optional

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
) -> Optional[str]:
    """Execute a single model operation (create, modify, etc.).

    Returns the target diagram type on success, ``None`` if a confirmation
    prompt was stored (existing-model guard) or on failure.
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
            session.set('pending_complete_system', {
                'message': operation_request,
                'diagram_type': target_diagram_type,
                'operation': operation,
                'default_mode': default_mode,
            })
            reply_message(
                session,
                f"You already have a {target_diagram_type} model ({summary}). "
                "Would you like me to **replace** it with a new one, or **keep** "
                "the existing model and add alongside it?",
            )
            logger.info(
                f"[ModelOp] Asked user to confirm replace/keep for existing {target_diagram_type}"
            )
            return None

    # ── GUI Auto-Generate shortcut ──────────────────────────────────────
    if target_diagram_type == "GUINoCodeDiagram" and operation_mode in ("complete_system", None, ""):
        class_diagram_model = resolve_class_diagram(request)
        if isinstance(class_diagram_model, dict):
            elements = class_diagram_model.get("elements")
            if isinstance(elements, dict) and len(elements) > 0:
                logger.info("[ModelOp] Routing GUI complete_system to frontend auto-generate")
                reply_payload(session, {
                    "action": "auto_generate_gui",
                    "diagramType": "GUINoCodeDiagram",
                    "message": (
                        "I'll generate the GUI automatically from your Class Diagram. "
                        "Each class will get its own page with a data table and method buttons."
                    ),
                })
                return target_diagram_type

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
                    "[ModelOp] ObjectDiagram reference is missing or empty; output may drift."
                )
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
                    "[ModelOp] ObjectDiagram reference is missing or empty; output may drift."
                )
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

    logger.info(
        f"[ModelOp] Sending result: action={result.get('action')}, "
        f"replaceExisting={result.get('replaceExisting', 'NOT SET')}, "
        f"keys={list(result.keys())}"
    )
    reply_payload(session, result)
    return target_diagram_type


# ------------------------------------------------------------------
# Planned-operation dispatch
# ------------------------------------------------------------------

def execute_planned_operations(
    session: Session,
    request: AssistantRequest,
    default_mode: str,
    matched_intent: Optional[str],
) -> None:
    """Run the orchestrator planner and dispatch each resulting operation."""
    operations = plan_assistant_operations(
        request=request,
        default_mode=default_mode,
        matched_intent=matched_intent,
        llm_predict=ctx.gpt_predict_json,
    )

    if not operations:
        reply_message(session, "I couldn't determine an execution plan from your request.")
        return

    working_request = request

    for operation in operations:
        if not isinstance(operation, dict):
            continue

        operation_type = operation.get("type")
        if operation_type == "model":
            try:
                executed_target = execute_model_operation(
                    session, working_request, operation, default_mode=default_mode,
                )
                if executed_target is None:
                    # Pending confirmation stored — stop processing further ops.
                    logger.info("[PlannedOps] Pending confirmation stored — halting remaining operations")
                    return
                if isinstance(executed_target, str) and executed_target:
                    working_request = build_request_for_target(working_request, executed_target)
            except Exception as error:
                logger.error(f"Error executing model operation {operation}: {error}")
                reply_message(session, "I encountered an issue while applying a modeling step.")
            continue

        if operation_type == "generation":
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
                logger.error(f"Error executing generation operation {operation}: {error}")
                response_payload = {
                    "action": "agent_error",
                    "code": "generation_handler_error",
                    "message": f"Failed to process {generator_type} generation request.",
                    "retryable": True,
                }

            if isinstance(response_payload, dict):
                reply_payload(session, response_payload)
            elif isinstance(response_payload, str):
                reply_message(session, response_payload)
