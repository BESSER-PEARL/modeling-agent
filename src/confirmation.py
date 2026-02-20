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
from session_helpers import reply_message

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

# Short words that must match as whole words to avoid false positives
# (e.g. "no" should not match inside "nothing", "note", "another").
_WHOLE_WORD_KEYWORDS = {'no'}


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
        # The user's message doesn't look like a confirmation — clear the
        # pending state and let the normal body logic handle it as a new request.
        session.set('pending_complete_system', None)
        return False

    # --- User answered: execute the stored creation -----------------------
    session.set('pending_complete_system', None)

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
    except Exception as exc:
        logger.error(f"[PendingConfirm] Error executing stored operation: {exc}", exc_info=True)
        reply_message(session, "Something went wrong while creating the model. Please try again.")

    return True
