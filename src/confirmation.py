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
from typing import Any, Dict, Optional

from besser.agent.core.session import Session

from protocol.adapters import parse_assistant_request
from session_helpers import reply_message, reply_payload

logger = logging.getLogger(__name__)


def _flush_pending_suggestions(session: Session) -> None:
    """Send any pending quality suggestions stored by execute_model_operation."""
    quality = session.get('_pending_quality_suggestions')
    if isinstance(quality, str) and quality:
        reply_message(session, quality)
        session.set('_pending_quality_suggestions', None)

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

# Short words that must match as whole words to avoid false positives
# (e.g. "no" should not match inside "nothing", "note", "another").
_WHOLE_WORD_KEYWORDS = {'no', 'yes', 'add', 'keep'}


def keyword_matches(keyword: str, text: str) -> bool:
    """Check if *keyword* appears in *text*, using word-boundary matching for short ambiguous words."""
    if keyword in _WHOLE_WORD_KEYWORDS:
        return bool(re.search(rf'\b{re.escape(keyword)}\b', text))
    return keyword in text


def model_has_elements(model: Optional[Dict[str, Any]]) -> bool:
    """Return True when *model* contains at least one user-visible element."""
    if not isinstance(model, dict):
        return False
    elements = model.get('elements')
    return isinstance(elements, dict) and len(elements) > 0


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
    from execution import execute_model_operation

    pending = session.get('pending_gui_choice')
    if not pending:
        return False

    request = parse_assistant_request(session)
    user_msg = (request.message or '').lower().strip()

    wants_cancel = any(keyword_matches(w, user_msg) for w in CANCEL_KEYWORDS)
    if wants_cancel:
        session.set('pending_gui_choice', None)
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
        session.set('pending_gui_choice', None)
        return False  # Let normal state body handle the new request

    if wants_auto:
        session.set('pending_gui_choice', None)
        logger.info("[GUIChoice] User chose AUTO-GENERATE (deterministic)")
        reply_payload(session, {
            "action": "auto_generate_gui",
            "diagramType": "GUINoCodeDiagram",
            "message": (
                "Generating GUI from your Class Diagram\u2026\n\n"
                "I'll generate the GUI automatically from your Class Diagram. "
                "Each class will get its own page with a data table and method buttons."
            ),
        })
        return True

    # LLM-driven path
    logger.info("[GUIChoice] User chose LLM-GENERATED (experimental)")
    stored_operation = pending.get('operation', {})
    stored_default_mode = pending.get('default_mode', 'complete_system')
    stored_replace = pending.get('_replace_existing')

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
        session.set('pending_gui_choice', None)  # Clear only on success
        _flush_pending_suggestions(session)
    except Exception as exc:
        logger.error(f"[GUIChoice] Error executing LLM GUI generation: {exc}", exc_info=True)
        reply_message(
            session,
            "Something went wrong. You can try again by saying **auto** or **llm**, or **cancel** to abort.",
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
    # Import here to break the circular dependency:
    # confirmation → execution → (no dependency back to confirmation)
    from execution import execute_model_operation

    pending = session.get('pending_complete_system')
    if not pending:
        return False

    request = parse_assistant_request(session)
    user_msg = (request.message or '').lower().strip()

    wants_cancel = any(keyword_matches(w, user_msg) for w in CANCEL_KEYWORDS)
    if wants_cancel:
        session.set('pending_complete_system', None)
        reply_message(session, "Cancelled. Your existing model is unchanged.")
        return True

    wants_replace = any(keyword_matches(w, user_msg) for w in REPLACE_KEYWORDS)
    wants_keep = any(keyword_matches(w, user_msg) for w in KEEP_KEYWORDS)

    if not wants_replace and not wants_keep:
        # The user's message doesn't look like a confirmation answer.
        # Treat it as a brand-new request: clear the pending state and let
        # the normal processing pipeline handle it.
        logger.info(
            "[PendingConfirm] Message doesn't match replace/keep/cancel — "
            "treating as new request, clearing pending state"
        )
        session.set('pending_complete_system', None)
        return False  # Let normal state body handle the new request

    # --- User answered: execute the stored creation -----------------------
    replace_existing = wants_replace

    # Re-execute the stored operation with the original parameters.
    stored_message = pending.get('message', '')
    stored_diagram_type = pending.get('diagram_type', 'ClassDiagram')
    stored_operation = pending.get('operation', {})
    stored_default_mode = pending.get('default_mode', 'complete_system')

    # Rebuild a minimal request that carries the stored message.
    working_request = request
    working_request.message = stored_message

    if replace_existing:
        logger.info(f"[PendingConfirm] User chose REPLACE for {stored_diagram_type}")
    else:
        logger.info(f"[PendingConfirm] User chose KEEP for {stored_diagram_type}")

    try:
        execute_model_operation(
            session=session,
            request=working_request,
            operation=stored_operation,
            default_mode=stored_default_mode,
            _skip_existing_check=True,
            _replace_existing=replace_existing,
        )
        session.set('pending_complete_system', None)  # Clear only on success
    except Exception as exc:
        logger.error(f"[PendingConfirm] Error executing stored operation: {exc}", exc_info=True)
        reply_message(
            session,
            "Something went wrong. You can try again by saying **replace** or **keep**, or **cancel** to abort.",
        )
        return True

    # ── Resume remaining operations from the original plan ───────────
    remaining_ops = pending.get('remaining_operations')
    if isinstance(remaining_ops, list) and remaining_ops:
        logger.info(
            f"[PendingConfirm] Resuming {len(remaining_ops)} remaining operation(s) "
            f"from original plan"
        )
        from utilities.model_helpers import build_request_for_target

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
                        new_pending = session.get('pending_complete_system')
                        if isinstance(new_pending, dict):
                            leftover = [
                                op for op in remaining_ops[op_idx + 1:]
                                if isinstance(op, dict)
                            ]
                            if leftover:
                                new_pending['remaining_operations'] = leftover
                                new_pending['original_message'] = (
                                    pending.get('original_message', stored_message)
                                )
                                session.set('pending_complete_system', new_pending)
                                logger.info(
                                    f"[PendingConfirm] Nested pending stored with "
                                    f"{len(leftover)} remaining op(s)"
                                )
                        break
                    if isinstance(result, str) and result:
                        resume_request = build_request_for_target(resume_request, result)
                except Exception as exc:
                    logger.error(
                        f"[PendingConfirm] Error executing remaining operation "
                        f"{remaining_op}: {exc}",
                        exc_info=True,
                    )
            elif op_type == 'generation':
                from handlers.generation_handler import handle_generation_request
                from utilities.model_helpers import build_generation_request
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

    # Flush any pending quality suggestions stored by execute_model_operation
    _flush_pending_suggestions(session)

    return True
