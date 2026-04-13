"""
Confirmation Flow
-----------------
Pending complete-system confirmation logic.

When the user asks to create a new complete system but a non-trivial model
already exists for that diagram type, we store the pending creation and ask
whether to replace or keep the existing one.  The confirmation answer may be
routed to ANY state by the intent classifier ("yes", "replace", "keep" …),
so :func:`handle_pending_system_confirmation` is checked at the top of every
state body.
"""

import logging
import re
from typing import Any

from baf.core.session import Session

from protocol.adapters import parse_assistant_request
from session_helpers import reply_message, reply_payload
from model_utils import model_has_elements  # noqa: F401  (re-export for backward compat)
from session_keys import PENDING_COMPLETE_SYSTEM, PENDING_GUI_CHOICE
from execution import execute_model_operation

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Keyword lists
# ------------------------------------------------------------------

REPLACE_KEYWORDS = [
    'replace', 'yes', 'overwrite', 'new one', 'start fresh',
    'remove', 'clear', 'delete', 'erase', 'fresh',
]
KEEP_KEYWORDS = [
    'keep', 'no', 'add', 'both', 'alongside', 'merge',
    "don't remove", 'do not remove',
]
CANCEL_KEYWORDS = ['cancel', 'never mind', 'forget', 'stop', 'abort']
NEW_TAB_KEYWORDS = [
    'new tab', 'new diagram', 'another tab', 'separate', 'own tab',
    'different tab', 'create new', 'add tab', 'fresh tab',
]

# Short words that must match as whole words to avoid false positives
# (e.g. "no" should not match inside "nothing", "note", "another").
_WHOLE_WORD_KEYWORDS = {'no', 'yes', 'add', 'keep'}


def keyword_matches(keyword: str, text: str) -> bool:
    """Check if *keyword* appears in *text*, using word-boundary matching for short ambiguous words."""
    if keyword in _WHOLE_WORD_KEYWORDS:
        return bool(re.search(rf'\b{re.escape(keyword)}\b', text))
    return keyword in text


# ------------------------------------------------------------------
# GUI generation-mode choice
# ------------------------------------------------------------------

_AUTO_KEYWORDS = ['auto', '1', 'deterministic', 'fast', 'standard', 'default', 'basic']
_LLM_KEYWORDS = ['llm', '2', 'personali', 'ai', 'experimental', 'custom', 'design']


def handle_pending_gui_choice(session: Session) -> bool:
    """Process a pending GUI generation-mode choice, if one exists.

    Returns ``True`` when a pending choice was found **and** handled
    (caller should ``return``).  Returns ``False`` otherwise.
    """
    pending = session.get(PENDING_GUI_CHOICE)
    if not pending:
        return False

    request = parse_assistant_request(session)
    user_msg = (request.message or '').lower().strip()

    wants_cancel = any(keyword_matches(w, user_msg) for w in CANCEL_KEYWORDS)
    if wants_cancel:
        session.set(PENDING_GUI_CHOICE, None)
        reply_message(session, "Cancelled. No GUI was generated.")
        return True

    wants_auto = any(keyword_matches(w, user_msg) for w in _AUTO_KEYWORDS)
    wants_llm = any(keyword_matches(w, user_msg) for w in _LLM_KEYWORDS)

    if not wants_auto and not wants_llm:
        # The user's message doesn't look like a GUI choice answer.
        # Treat it as a brand-new request: clear the pending state and let
        # the normal processing pipeline handle it.
        logger.info(
            "[GUIChoice] Message doesn't match auto/llm/cancel — "
            "treating as new request, clearing pending state"
        )
        session.set(PENDING_GUI_CHOICE, None)
        return False  # Let normal state body handle the new request

    if wants_auto:
        remaining_ops = pending.get('remaining_operations')
        session.set(PENDING_GUI_CHOICE, None)
        logger.info("🔄 [GUIChoice] User chose AUTO-GENERATE (deterministic)")
        reply_payload(session, {
            "action": "auto_generate_gui",
            "diagramType": "GUINoCodeDiagram",
            "message": (
                "Generating GUI from your Class Diagram\u2026\n\n"
                "I'll generate the GUI automatically from your Class Diagram. "
                "Each class will get its own page with a data table and method buttons."
            ),
        })
        # Resume any remaining operations from the original plan
        if isinstance(remaining_ops, list) and remaining_ops:
            logger.info(f"[GUIChoice] Resuming {len(remaining_ops)} remaining op(s) after auto-generate")
            _resume_remaining_ops(
                session, remaining_ops, request,
                'GUINoCodeDiagram', 'complete_system',
                pending.get('operation_request', ''), pending,
            )
        return True

    # LLM-driven path
    logger.info("🔄 [GUIChoice] User chose LLM-GENERATED (experimental)")
    stored_operation = pending.get('operation', {})
    stored_default_mode = pending.get('default_mode', 'complete_system')
    stored_replace = pending.get('_replace_existing')
    remaining_ops = pending.get('remaining_operations')

    # Restore the original request message for the operation
    working_request = request
    working_request.message = pending.get('operation_request', request.message)

    try:
        execute_model_operation(
            session=session,
            request=working_request,
            operation=stored_operation,
            default_mode=stored_default_mode,
            _skip_existing_check=True,
            _replace_existing=stored_replace,
            _skip_gui_choice=True,
        )
        session.set(PENDING_GUI_CHOICE, None)  # Clear only on success
    except Exception as exc:
        logger.error(f"❌ [GUIChoice] Error executing LLM GUI generation: {exc}", exc_info=True)
        reply_message(
            session,
            "Something went wrong. You can try again by saying **auto** or **llm**, or **cancel** to abort.",
        )
        session.set(PENDING_GUI_CHOICE, None)
        return True

    # Resume any remaining operations from the original plan
    if isinstance(remaining_ops, list) and remaining_ops:
        logger.info(f"[GUIChoice] Resuming {len(remaining_ops)} remaining op(s) after LLM generation")
        _resume_remaining_ops(
            session, remaining_ops, working_request,
            'GUINoCodeDiagram', stored_default_mode,
            pending.get('operation_request', ''), pending,
        )

    return True


# ------------------------------------------------------------------
# Pending complete-system confirmation
# ------------------------------------------------------------------


def handle_pending_system_confirmation(session: Session) -> bool:
    """Process a pending complete-system confirmation, if one exists.

    Returns ``True`` when a pending confirmation was found **and** handled
    (the caller should ``return`` immediately).  Returns ``False`` otherwise
    so the normal body logic can proceed.
    """
    pending = session.get(PENDING_COMPLETE_SYSTEM)
    if not pending:
        return False

    request = parse_assistant_request(session)
    user_msg = (request.message or '').lower().strip()

    wants_cancel = any(keyword_matches(w, user_msg) for w in CANCEL_KEYWORDS)
    if wants_cancel:
        session.set(PENDING_COMPLETE_SYSTEM, None)
        reply_message(session, "Cancelled. Your existing model is unchanged.")
        return True

    wants_new_tab = pending.get('can_add_tab', False) and any(keyword_matches(w, user_msg) for w in NEW_TAB_KEYWORDS)
    wants_replace = any(keyword_matches(w, user_msg) for w in REPLACE_KEYWORDS)
    wants_keep = any(keyword_matches(w, user_msg) for w in KEEP_KEYWORDS)

    if not wants_replace and not wants_keep and not wants_new_tab:
        # The user's message doesn't look like a confirmation answer.
        # Treat it as a brand-new request: clear the pending state and let
        # the normal processing pipeline handle it.
        logger.info(
            "[PendingConfirm] Message doesn't match replace/keep/new-tab/cancel — "
            "treating as new request, clearing pending state"
        )
        session.set(PENDING_COMPLETE_SYSTEM, None)
        return False  # Let normal state body handle the new request

    # --- User answered: execute the stored creation -----------------------

    # ── Pre-computed payload path (file uploads) ──────────────────────
    # File conversions (PDF, PlantUML, images) produce a ready-to-send
    # payload.  We just need to stamp replaceExisting / create_new_tab.
    precomputed = pending.get('precomputed_payload')
    if precomputed is not None:
        precomputed = dict(precomputed)  # Shallow copy to avoid mutating stored state
        stored_diagram_type = pending.get('diagram_type', 'ClassDiagram')
        if wants_new_tab:
            logger.info(f"🔄 [PendingConfirm] File upload: user chose NEW TAB for {stored_diagram_type}")
            reply_payload(session, {
                "action": "create_diagram_tab",
                "diagramType": stored_diagram_type,
            })
            precomputed["replaceExisting"] = True
        elif wants_replace:
            logger.info(f"🔄 [PendingConfirm] File upload: user chose REPLACE for {stored_diagram_type}")
            precomputed["replaceExisting"] = True
        else:
            logger.info(f"🔄 [PendingConfirm] File upload: user chose KEEP for {stored_diagram_type}")
            precomputed["replaceExisting"] = False

        reply_payload(session, precomputed)
        session.set(PENDING_COMPLETE_SYSTEM, None)
        return True

    # Re-execute the stored operation with the original parameters.
    stored_message = pending.get('message', '')
    stored_diagram_type = pending.get('diagram_type', 'ClassDiagram')
    stored_operation = pending.get('operation', {})
    stored_default_mode = pending.get('default_mode', 'complete_system')

    # Rebuild a minimal request that carries the stored message.
    working_request = request
    working_request.message = stored_message

    # ── New tab path ──────────────────────────────────────────────────
    if wants_new_tab:
        logger.info(f"🔄 [PendingConfirm] User chose NEW TAB for {stored_diagram_type}")
        try:
            execute_model_operation(
                session=session,
                request=working_request,
                operation=stored_operation,
                default_mode=stored_default_mode,
                _skip_existing_check=True,
                _replace_existing=True,  # New tab is empty, replace is fine
                _create_new_tab=True,
            )
            session.set(PENDING_COMPLETE_SYSTEM, None)
        except Exception as exc:
            logger.error(f"❌ [PendingConfirm] Error after new tab creation: {exc}", exc_info=True)
            reply_message(session, "Something went wrong creating the new tab. Please try again.")
            session.set(PENDING_COMPLETE_SYSTEM, None)
            return True

        # Resume remaining operations same as replace/keep path
        remaining_ops = pending.get('remaining_operations')
        if isinstance(remaining_ops, list) and remaining_ops:
            logger.info(
                f"[PendingConfirm] Resuming {len(remaining_ops)} remaining operation(s) "
                f"after new tab creation"
            )
            _resume_remaining_ops(
                session, remaining_ops, working_request,
                stored_diagram_type, stored_default_mode, stored_message, pending,
            )

        return True

    # ── Replace / Keep path ───────────────────────────────────────────
    replace_existing = wants_replace

    if replace_existing:
        logger.info(f"🔄 [PendingConfirm] User chose REPLACE for {stored_diagram_type}")
    else:
        logger.info(f"🔄 [PendingConfirm] User chose KEEP for {stored_diagram_type}")

    try:
        execute_model_operation(
            session=session,
            request=working_request,
            operation=stored_operation,
            default_mode=stored_default_mode,
            _skip_existing_check=True,
            _replace_existing=replace_existing,
        )
        session.set(PENDING_COMPLETE_SYSTEM, None)  # Clear only on success
    except Exception as exc:
        logger.error(f"❌ [PendingConfirm] Error executing stored operation: {exc}", exc_info=True)
        reply_message(
            session,
            "Something went wrong. You can try again by saying **replace**, **keep**, or **new tab**, or **cancel** to abort.",
        )
        session.set(PENDING_COMPLETE_SYSTEM, None)
        return True

    # ── Resume remaining operations from the original plan ───────────
    remaining_ops = pending.get('remaining_operations')
    if isinstance(remaining_ops, list) and remaining_ops:
        logger.info(
            f"[PendingConfirm] Resuming {len(remaining_ops)} remaining operation(s) "
            f"from original plan"
        )
        _resume_remaining_ops(
            session, remaining_ops, working_request,
            stored_diagram_type, stored_default_mode, stored_message, pending,
        )

    return True


def _resume_remaining_ops(
    session: Session,
    remaining_ops: list,
    working_request: Any,
    stored_diagram_type: str,
    stored_default_mode: str,
    stored_message: str,
    pending: dict,
) -> None:
    """Execute remaining operations that were queued behind a pending confirmation."""
    from utilities.request_builders import build_request_for_target

    # Rebuild the working request so subsequent operations see the
    # just-created diagram in context.
    resume_request = working_request
    if stored_diagram_type:
        resume_request = build_request_for_target(working_request, stored_diagram_type)

    for op_idx, remaining_op in enumerate(remaining_ops):
        if not isinstance(remaining_op, dict):
            continue
        op_type = remaining_op.get('type')
        if op_type == 'model':
            try:
                result = execute_model_operation(
                    session=session,
                    request=resume_request,
                    operation=remaining_op,
                    default_mode=stored_default_mode,
                )
                if result is None:
                    # This operation stored a new pending confirmation.
                    # Save the rest of the remaining ops so they can be
                    # resumed after the user responds to the new prompt.
                    leftover = [
                        op for op in remaining_ops[op_idx + 1:]
                        if isinstance(op, dict)
                    ]
                    if leftover:
                        # Check both pending types — the operation may have
                        # stored either a system confirmation or a GUI choice.
                        for key in (PENDING_COMPLETE_SYSTEM, PENDING_GUI_CHOICE):
                            new_pending = session.get(key)
                            if isinstance(new_pending, dict):
                                new_pending['remaining_operations'] = leftover
                                new_pending['original_message'] = (
                                    pending.get('original_message', stored_message)
                                )
                                session.set(key, new_pending)
                                logger.info(
                                    f"[PendingConfirm] Nested pending stored in {key} with "
                                    f"{len(leftover)} remaining op(s)"
                                )
                                break
                    break
                if isinstance(result, str) and result:
                    resume_request = build_request_for_target(resume_request, result)
            except Exception as exc:
                logger.error(
                    f"[PendingConfirm] Error executing remaining operation "
                    f"{remaining_op}: {exc}",
                    exc_info=True,
                )
                reply_message(
                    session,
                    "Something went wrong while processing a follow-up operation. "
                    "Please try again.",
                )
        elif op_type == 'generation':
            from handlers.generation_handler import handle_generation_request
            from utilities.request_builders import build_generation_request
            from session_helpers import reply_payload

            gen_type = remaining_op.get('generatorType')
            if isinstance(gen_type, str) and gen_type:
                gen_req = build_generation_request(
                    resume_request,
                    generator_type=gen_type,
                    config=remaining_op.get('config') if isinstance(remaining_op.get('config'), dict) else {},
                )
                try:
                    gen_response = handle_generation_request(session, gen_req)
                    if isinstance(gen_response, dict):
                        reply_payload(session, gen_response)
                except Exception as exc:
                    logger.error(
                        f"[PendingConfirm] Error executing remaining generation: {exc}",
                        exc_info=True,
                    )
                    reply_message(
                        session,
                        "Something went wrong while running code generation. "
                        "Please try again.",
                    )
