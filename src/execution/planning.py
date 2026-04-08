"""Orchestrator-driven multi-step dispatch and parallel execution.

Runs the request planner, splits operations into parallel-safe groups,
dispatches model and generation operations, and handles error reporting.
"""

import concurrent.futures
import logging
from typing import Any, Dict, List, Optional, Tuple

from baf.core.session import Session

import agent_context as ctx
from protocol.types import AssistantRequest
from session_helpers import reply_message, reply_payload
from orchestrator import plan_assistant_operations
from handlers.generation_handler import handle_generation_request
from utilities.request_builders import build_request_for_target, build_generation_request
from suggestions import get_suggested_actions
from errors import ErrorCode, classify_error, build_error_response
from session_keys import PENDING_COMPLETE_SYSTEM, PENDING_GUI_CHOICE

from .model_operations import execute_model_operation, _collect_available_diagrams
from .progress import _report_progress

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Structured error payloads
# ------------------------------------------------------------------

def _build_error_payload(operation: dict, error: Exception, error_code: str = "unknown") -> dict:
    """Build a structured error payload with recovery hints."""
    try:
        code_enum = ErrorCode(error_code)
    except ValueError:
        code_enum = ErrorCode.UNKNOWN
    logger.error(f"Operation error: {error}")
    return build_error_response(
        code_enum,
        operation=operation,
    )


def _classify_error(error: Exception) -> str:
    """Map an exception to a structured error code string."""
    return classify_error(error).value


# ------------------------------------------------------------------
# Parallel operation dispatch helpers
# ------------------------------------------------------------------

def _can_run_parallel(operations: List[dict]) -> Tuple[List[List[dict]], List[dict]]:
    """Split operations into parallel-safe groups.

    Model operations for DIFFERENT diagram types with no dependencies can run
    in parallel.  Generation ops always run after their prerequisite model ops.
    """
    model_ops = [op for op in operations if isinstance(op, dict) and op.get("type") == "model"]
    gen_ops = [op for op in operations if isinstance(op, dict) and op.get("type") == "generation"]

    _DIAGRAM_DEPENDENCIES: Dict[str, set] = {
        "GUINoCodeDiagram": {"ClassDiagram"},
        "ObjectDiagram": {"ClassDiagram"},
    }

    independent_model_groups: Dict[str, List[dict]] = {}
    for op in model_ops:
        dt = op.get("diagramType", "unknown")
        independent_model_groups.setdefault(dt, []).append(op)

    all_types = set(independent_model_groups.keys())
    has_dependency = any(
        _DIAGRAM_DEPENDENCIES.get(dtype, set()) & all_types
        for dtype in all_types
    )
    if has_dependency:
        return [model_ops], gen_ops

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

    model_groups, gen_ops = _can_run_parallel(operations)
    total_steps = sum(len(g) for g in model_groups) + len(gen_ops)

    working_request = request
    step_counter = 0

    # ── Phase 1: Model operations ────────────────────────────────────
    if len(model_groups) > 1:
        logger.info(
            f"[PlannedOps] Running {len(model_groups)} independent model group(s) in parallel"
        )

        confirmation_triggered = False

        def _run_model_group(group: List[dict]) -> List[Tuple[dict, Optional[str], Optional[Exception]]]:
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
                    for op in group:
                        step_counter += 1
                        _report_progress(session, step_counter - 1, total_steps, op)
                        error_code = _classify_error(exc)
                        error_payload = _build_error_payload(op, exc, error_code)
                        reply_payload(session, error_payload)
                        logger.error(f"❌ [PlannedOps] Parallel group error: {exc}")
                    continue

                for op, executed_target, error in group_results:
                    step_counter += 1
                    _report_progress(session, step_counter - 1, total_steps, op)

                    if error is not None:
                        error_code = _classify_error(error)
                        error_payload = _build_error_payload(op, error, error_code)
                        reply_payload(session, error_payload)
                        logger.error(f"❌ [PlannedOps] Model op error: {error}")
                        continue

                    if executed_target is None:
                        remaining_ops = gen_ops[:]
                        _store_remaining_ops(session, remaining_ops, request)
                        confirmation_triggered = True
                        continue

                    if isinstance(executed_target, str) and executed_target:
                        working_request = build_request_for_target(
                            working_request, executed_target,
                        )

        if confirmation_triggered:
            logger.info("⏸️ [PlannedOps] Pending confirmation stored — halting remaining operations")
            return

    else:
        flat_model_ops = model_groups[0] if model_groups else []

        for idx, operation in enumerate(flat_model_ops):
            step_counter += 1
            _report_progress(session, step_counter - 1, total_steps, operation)

            try:
                executed_target = execute_model_operation(
                    session, working_request, operation, default_mode=default_mode,
                )
                if executed_target is None:
                    remaining = (
                        [op for op in flat_model_ops[idx + 1:] if isinstance(op, dict)]
                        + gen_ops
                    )
                    _store_remaining_ops(session, remaining, request)
                    logger.info("⏸️ [PlannedOps] Pending confirmation stored — halting remaining operations")
                    return
                if isinstance(executed_target, str) and executed_target:
                    working_request = build_request_for_target(working_request, executed_target)
            except Exception as error:
                error_code = _classify_error(error)
                error_payload = _build_error_payload(operation, error, error_code)
                reply_payload(session, error_payload)
                for key in (PENDING_COMPLETE_SYSTEM, PENDING_GUI_CHOICE):
                    if session.get(key):
                        session.set(key, None)
                logger.error(f"❌ [PlannedOps] Model op error ({error_code}): {error}")
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
            logger.error(f"❌ [PlannedOps] Generation error ({error_code}): {error}")

        if isinstance(response_payload, dict):
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
    if not remaining:
        return

    for key in (PENDING_COMPLETE_SYSTEM, PENDING_GUI_CHOICE):
        pending = session.get(key)
        if isinstance(pending, dict):
            pending['remaining_operations'] = remaining
            pending['original_message'] = request.message
            session.set(key, pending)
            logger.info(
                f"[PlannedOps] Stored {len(remaining)} remaining operation(s) "
                f"alongside {key}"
            )
