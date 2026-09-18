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
from session_helpers import reply_message, reply_payload, emit_webapp_generate_prompt
from orchestrator import plan_assistant_operations
from handlers.generation_handler import handle_generation_request
from utilities.request_builders import build_request_for_target, build_generation_request
from suggestions import get_suggested_actions
from errors import ErrorCode, classify_error, build_error_response
from session_keys import (
    PENDING_COMPLETE_SYSTEM,
    PENDING_GENERATOR_CONFIG,
    PENDING_GENERATOR_TYPE,
    PENDING_GUI_CHOICE,
    PENDING_WEBAPP_GENERATE,
    PLAN_GENERATION_CONFIRM_FLAG,
)

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


def _clear_paused_plan_generation(session: Session) -> None:
    """Drop a plan-paused generation stash after a model-op failure.

    Only clears the pending-generation state when it carries the plan-pause
    marker — a user-driven config flow (no marker) is never touched. A
    broken build must never leave a generator armed to fire on a later,
    unrelated affirmative."""
    config = session.get(PENDING_GENERATOR_CONFIG)
    if isinstance(config, dict) and config.get(PLAN_GENERATION_CONFIRM_FLAG):
        session.set(PENDING_GENERATOR_TYPE, None)
        session.set(PENDING_GENERATOR_CONFIG, None)
        logger.info("[PlannedOps] Cleared paused generation after a model-op failure")


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
    # Consume the unified classifier's diagram-type verdict (it read the full
    # message + workspace) as the PRIMARY diagram target — previously this was
    # produced but discarded, and the type was re-derived from keyword lists.
    _llm_target = None
    try:
        from session_keys import UNIFIED_CLASSIFICATION
        _uc = session.get(UNIFIED_CLASSIFICATION)
        _llm_target = getattr(_uc, "target_diagram_type", None) if _uc is not None else None
    except Exception:
        _llm_target = None
    operations = plan_assistant_operations(
        request=request,
        default_mode=default_mode,
        matched_intent=matched_intent,
        llm_predict=ctx.gpt_predict_json,
        llm_target_type=_llm_target,
    )

    if not operations:
        reply_message(session, "I couldn't determine an execution plan from your request.")
        return

    model_groups, gen_ops = _can_run_parallel(operations)

    # ── BULLETPROOF PAUSE ────────────────────────────────────────────
    # A "create a web app" plan builds the model + GUI and would then auto-run
    # web_app code generation. STRIP that generation op from the plan at the
    # source, so there is nothing to auto-run on ANY execution path (this is the
    # single source of truth — it supersedes the per-path gates). A session flag
    # then drives the "generate the web app?" prompt once the GUI is built; the
    # user triggers generation explicitly afterwards. Explicit non-GUI plans
    # ("create X and generate django") have no GUINoCodeDiagram op, so their
    # generation is untouched and still runs.
    _webapp_build = (
        any(isinstance(op, dict) and op.get("diagramType") == "GUINoCodeDiagram" for op in operations)
        and any(isinstance(op, dict) and op.get("type") == "generation" for op in operations)
    )
    if _webapp_build:
        gen_ops = []  # nothing auto-runs
        session.set(PENDING_WEBAPP_GENERATE, True)
        logger.info("⏸️ [PlannedOps] Web-app build — stripped auto-generation; "
                    "user will be asked to generate after the GUI is built")

    # ── GENERALIZED PAUSE (every other generator) ────────────────────
    # A mixed "design X and generate Y" plan would auto-run generation the
    # moment the model lands — while the injection message is literally
    # asking "review or continue with generating?", self-answering its own
    # question. Extend the web-app pause to ALL generator types: strip the
    # generation op(s) from the plan and stash the first one in the existing
    # pending-generation state (PENDING_GENERATOR_TYPE / _CONFIG), marked as
    # awaiting confirmation. The injected model's follow-up question and
    # quick actions then let the user decide; an affirmative answer fires
    # the stashed generator through handle_generation_request's pending
    # path. A DIRECT generation request (no model op in the plan) is
    # untouched and still runs immediately.
    _mixed_plan = (
        bool(gen_ops)
        and not _webapp_build
        and any(isinstance(op, dict) and op.get("type") == "model" for op in operations)
    )
    if _mixed_plan:
        _paused_op = next(
            (op for op in gen_ops
             if isinstance(op.get("generatorType"), str) and op.get("generatorType")),
            None,
        )
        if _paused_op is not None:
            if len(gen_ops) > 1:
                logger.info(
                    "⏸️ [PlannedOps] Mixed plan carried %d generation ops — "
                    "stashing the first (%s); the user drives the rest explicitly",
                    len(gen_ops), _paused_op.get("generatorType"),
                )
            _paused_config = (
                _paused_op.get("config")
                if isinstance(_paused_op.get("config"), dict) else {}
            )
            session.set(PENDING_GENERATOR_TYPE, _paused_op["generatorType"])
            session.set(
                PENDING_GENERATOR_CONFIG,
                {**_paused_config, PLAN_GENERATION_CONFIRM_FLAG: True},
            )
            gen_ops = []  # nothing auto-runs
            logger.info(
                "⏸️ [PlannedOps] Mixed modeling+generation plan — paused '%s' "
                "generation until the user confirms",
                _paused_op["generatorType"],
            )

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
                    _clear_paused_plan_generation(session)
                    continue

                for op, executed_target, error in group_results:
                    step_counter += 1
                    _report_progress(session, step_counter - 1, total_steps, op)

                    if error is not None:
                        error_code = _classify_error(error)
                        error_payload = _build_error_payload(op, error, error_code)
                        reply_payload(session, error_payload)
                        _clear_paused_plan_generation(session)
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
                _clear_paused_plan_generation(session)
                logger.error(f"❌ [PlannedOps] Model op error ({error_code}): {error}")
            continue

    # ── Phase 2: Generation operations (always sequential) ───────────
    # Web-app builds had their generation stripped above; show the pause prompt
    # once the GUI is built. This is the FALL-THROUGH path (the GUI op did NOT
    # halt for a choice, e.g. a slow/timeout auto-completion, so execution reached
    # here directly). The GUI-choice path emits the same prompt from
    # confirmation._resume_remaining_ops. Non-web-app generations still run below.
    if session.get(PENDING_WEBAPP_GENERATE):
        session.set(PENDING_WEBAPP_GENERATE, None)
        logger.info("⏸️ [PlannedOps] Web-app GUI built — asking the user to generate")
        emit_webapp_generate_prompt(session)
        return

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
