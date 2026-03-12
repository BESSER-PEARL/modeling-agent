"""
Session Helpers
---------------
Reply / message utilities and intent-matching condition functions.

These are pure functions with no dependency on module-level globals;
they only use the ``session`` object and the protocol adapters.
"""

import json
import logging
import time
import uuid
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


def _send_to_session(session: Session, payload: Dict[str, Any]):
    """Low-level helper: serialize *payload* as JSON and send it via the session.

    This mirrors the mechanism used by :func:`reply_payload` — a single
    ``session.reply(json.dumps(...))`` call — so streaming messages travel
    through the exact same WebSocket path as every other server-initiated
    message.
    """
    session.reply(json.dumps(payload))


# ------------------------------------------------------------------
# Streaming reply helpers
# ------------------------------------------------------------------

def reply_stream_start(session: Session, stream_id: str = None) -> str:
    """Start a streaming response.  Returns the stream ID."""
    if stream_id is None:
        stream_id = str(uuid.uuid4())[:8]
    payload = {
        "action": "stream_start",
        "streamId": stream_id,
    }
    _send_to_session(session, payload)
    return stream_id


def reply_stream_chunk(session: Session, chunk: str, stream_id: str):
    """Send a streaming text chunk to the frontend."""
    payload = {
        "action": "stream_chunk",
        "streamId": stream_id,
        "chunk": chunk,
        "done": False,
    }
    _send_to_session(session, payload)


def reply_stream_done(session: Session, stream_id: str, full_text: str = ""):
    """Signal the end of a streaming response."""
    payload = {
        "action": "stream_done",
        "streamId": stream_id,
        "fullText": full_text,
        "done": True,
    }
    _send_to_session(session, payload)


def reply_progress(session: Session, message: str, step: int = 0, total: int = 0):
    """Send a progress indicator to the frontend."""
    payload = {
        "action": "progress",
        "message": message,
        "step": step,
        "total": total,
    }
    _send_to_session(session, payload)


def stream_llm_response(
    session: Session, llm_instance: Any, prompt: str, system_prompt: str = ""
) -> str:
    """Stream an LLM response chunk by chunk to the frontend.

    Args:
        session: The WebSocket session.
        llm_instance: The OpenAI LLM instance.
        prompt: The user prompt.
        system_prompt: Optional system prompt.

    Returns:
        The full completed text.
    """
    stream_id = reply_stream_start(session)
    full_text = ""

    try:
        # Note: This depends on how the BESSER framework's LLM predict works.
        # If it doesn't support streaming natively, fall back to non-streaming
        # and send the full response as a single chunk.

        # Attempt streaming if available
        response = llm_instance.predict(prompt)

        # If we can't stream, send as one chunk
        full_text = response if isinstance(response, str) else str(response)

        # Send in word-sized chunks to simulate streaming
        words = full_text.split(' ')
        chunk_size = 3  # Send 3 words at a time
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if i > 0:
                chunk = ' ' + chunk
            reply_stream_chunk(session, chunk, stream_id)
            time.sleep(0.03)  # Small delay for natural feel

    except Exception as e:
        full_text = f"I encountered an issue: {str(e)}"
        reply_stream_chunk(session, full_text, stream_id)

    reply_stream_done(session, stream_id, full_text)
    return full_text


# ------------------------------------------------------------------
# Generation routing
# ------------------------------------------------------------------

def route_to_generation(session: Session) -> bool:
    """Detect generation workflow requests or frontend callback events."""
    request = parse_assistant_request(session)
    return should_route_to_generation(session, request)
