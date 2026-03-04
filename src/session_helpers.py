"""
Session Helpers
---------------
Reply / message utilities and intent-matching condition functions.

These are pure functions with no dependency on module-level globals;
they only use the ``session`` object and the protocol adapters.
"""

import json
import logging
from typing import Any, Dict, Optional

from besser.agent.core.session import Session

from protocol.adapters import parse_assistant_request
from handlers.generation_handler import should_route_to_generation

logger = logging.getLogger(__name__)

# Maximum user message length (characters).  Messages beyond this are
# truncated to avoid blowing the LLM context window.  ~12 000 chars ≈
# ~3 000 tokens, leaving plenty of headroom inside the 1M context of
# gpt-4.1-mini while still fitting any realistic request.
MAX_USER_MESSAGE_CHARS = 12_000


# ------------------------------------------------------------------
# Message extraction helpers
# ------------------------------------------------------------------

def get_user_message(session: Session) -> str:
    """Extract normalized message using protocol adapters."""
    request = parse_assistant_request(session)
    message = request.message or ""
    if len(message) > MAX_USER_MESSAGE_CHARS:
        original_len = len(message)
        logger.warning(
            f"User message truncated from {original_len} to {MAX_USER_MESSAGE_CHARS} chars"
        )
        message = message[:MAX_USER_MESSAGE_CHARS] + "\u2026[truncated]"
        reply_message(
            session,
            f"Your message was quite long ({original_len:,} characters) and has been "
            f"trimmed to {MAX_USER_MESSAGE_CHARS:,} characters. If important details "
            "were near the end, consider splitting your request into smaller parts.",
        )
    return message


def get_diagram_type(session: Session, default: str = 'ClassDiagram') -> str:
    """Extract normalized diagram type using protocol adapters."""
    request = parse_assistant_request(session, default_diagram_type=default)
    return request.diagram_type or default


def get_current_model(session: Session) -> Optional[Dict[str, Any]]:
    """Extract normalized current model from protocol adapters."""
    request = parse_assistant_request(session)
    return request.current_model


# ------------------------------------------------------------------
# Intent-matching condition functions for JSON events
# ------------------------------------------------------------------

def json_intent_matches(session: Session, params: Dict[str, Any]) -> bool:
    """Check if the predicted intent matches the target intent for JSON events.

    Skips intent matching when a pending confirmation or selection flow is
    active — the user's reply (e.g. "replace", "auto") should stay in the
    current state so _common_preamble can handle it, instead of being
    misrouted by the intent classifier.
    """
    # If awaiting generator selection, suppress intent matching so the
    # route_to_generation condition (next priority) can capture the reply.
    pending = session.get("pending_generator_type")
    if pending == "_awaiting_selection":
        return False

    # If a pending confirmation or GUI choice is active, suppress intent
    # matching so the message stays in the current state and _common_preamble
    # handles it.  This prevents "replace"/"keep"/"auto"/"llm" from being
    # misclassified as modify_model_intent or fallback_intent.
    if session.get("pending_complete_system"):
        return False
    if session.get("pending_gui_choice"):
        return False

    target_intent_name = params.get('intent_name')
    if not target_intent_name:
        return False

    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        matched_intent = session.event.predicted_intent.intent
        return matched_intent.name == target_intent_name

    return False


def json_no_intent_matched(session: Session) -> bool:
    """Check if no specific intent was matched (fallback).

    Also returns True when a pending confirmation suppressed intent matching,
    so the message stays in the current state for _common_preamble to handle.
    """
    if session.get("pending_complete_system") or session.get("pending_gui_choice"):
        return True
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        matched_intent = session.event.predicted_intent.intent
        return matched_intent.name == 'fallback_intent'
    return True


# ------------------------------------------------------------------
# Reply helpers
# ------------------------------------------------------------------

def reply_message(session: Session, message: str):
    """Send assistant message, wrapped for v2 protocol clients."""
    request = parse_assistant_request(session)
    if request.is_v2:
        session.reply(json.dumps({
            "action": "assistant_message",
            "message": message,
        }))
    else:
        session.reply(message)


def reply_payload(session: Session, payload: Dict[str, Any]):
    """Send JSON payload response for both protocol versions."""
    logger.info(
        f"[Reply] Sending payload: action={payload.get('action')}, "
        f"diagramType={payload.get('diagramType')}, "
        f"replaceExisting={payload.get('replaceExisting', 'NOT SET')}, "
        f"message={str(payload.get('message', ''))[:100]!r}"
    )
    logger.debug(f"[Reply] Full payload keys: {list(payload.keys())}")
    session.reply(json.dumps(payload))


def route_to_generation(session: Session) -> bool:
    """Detect generation workflow requests or frontend callback events."""
    request = parse_assistant_request(session)
    return should_route_to_generation(session, request)
