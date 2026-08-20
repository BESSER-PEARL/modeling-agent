"""
Session Helpers
---------------
Reply / message utilities and intent-matching condition functions.

These are pure functions with no dependency on module-level globals;
they only use the ``session`` object and the protocol adapters.
"""

import json
import logging
import uuid
from typing import Any, Dict, Optional

from baf.core.session import Session

from protocol.adapters import parse_assistant_request
from handlers.generation_handler import (
    should_route_to_generation,
    detect_generator_type,
)
from agent_config import (
    MAX_USER_MESSAGE_CHARS,
    STREAM_BUFFER_THRESHOLD,
    LLM_TEXT_TEMPERATURE,
    LLM_MAX_TOKENS_TEXT,
)
from model_config import MODEL_CLASSIFIER
from memory import get_memory, memory_session_key
from session_keys import (
    PENDING_COMPLETE_SYSTEM,
    PENDING_GUI_CHOICE,
    PENDING_GENERATOR_TYPE,
    PENDING_SMART_GEN_INSTRUCTIONS,
    UNIFIED_CLASSIFICATION,
)

logger = logging.getLogger(__name__)


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

    Priority:
      1. Unified classifier's non-fallback verdict
         (``UNIFIED_CLASSIFICATION`` in session) — our own LLM call,
         authoritative when it reached a decision.
      2. BAF's built-in ``session.event.predicted_intent`` — fallback
         when the unified classifier wasn't called (e.g. tests, or
         if the ensure-classification hook is ever disabled).

    Skips intent matching when a pending confirmation or selection flow
    is active — the user's reply (e.g. "replace", "auto") should stay
    in the current state so ``_common_preamble`` can handle it, instead
    of being misrouted by the intent classifier.
    """
    # If awaiting generator selection, suppress intent matching so the
    # route_to_generation condition (next priority) can capture the reply.
    pending = session.get(PENDING_GENERATOR_TYPE)
    if pending == "_awaiting_selection":
        return False

    # If a pending confirmation or GUI choice is active, suppress intent
    # matching so the message stays in the current state and _common_preamble
    # handles it.  This prevents "replace"/"keep"/"auto"/"llm" from being
    # misclassified as modify_model_intent or fallback_intent.
    if session.get(PENDING_COMPLETE_SYSTEM):
        return False
    if session.get(PENDING_GUI_CHOICE):
        return False
    if session.get(PENDING_SMART_GEN_INSTRUCTIONS):
        return False

    target_intent_name = params.get('intent_name')
    if not target_intent_name:
        return False

    # Priority 1: the unified classifier, if it ran for this message.
    # See ``ensure_unified_classification`` in state_bodies.py.
    unified = session.get(UNIFIED_CLASSIFICATION)
    if unified is not None and unified.intent != "fallback_intent":
        return unified.intent == target_intent_name

    # Priority 2: BAF's description-based classifier. Fallback only —
    # runs when the unified classifier hook wasn't installed (tests,
    # unusual state machines). The old keyword cross-validation layer
    # (keyword and phrase heuristics) that
    # used to rescue BAF from its own misclassifications has been
    # deleted — if we're in this branch, we trust BAF's answer as-is.
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        matched_intent = session.event.predicted_intent.intent
        return matched_intent.name == target_intent_name

    return False


def json_no_intent_matched(session: Session) -> bool:
    """Check if no specific intent was matched (fallback).

    Also returns True when a pending confirmation suppressed intent matching,
    so the message stays in the current state for _common_preamble to handle.
    """
    if (
        session.get(PENDING_COMPLETE_SYSTEM)
        or session.get(PENDING_GUI_CHOICE)
        or session.get(PENDING_SMART_GEN_INSTRUCTIONS)
    ):
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
    try:
        request = parse_assistant_request(session)
        if request.is_v2:
            session.reply(json.dumps({
                "action": "assistant_message",
                "message": message,
            }))
        else:
            session.reply(message)
    except Exception as exc:
        logger.error(f"❌ [Reply] Failed to send message: {exc}", exc_info=True)
        return

    # Record in conversation memory
    _record_assistant_response(session, message)


def reply_payload(session: Session, payload: Dict[str, Any]):
    """Send JSON payload response for both protocol versions."""
    logger.info(
        f"📤 [Reply] Sending payload: action={payload.get('action')}, "
        f"diagramType={payload.get('diagramType')}, "
        f"replaceExisting={payload.get('replaceExisting', 'NOT SET')}, "
        f"message={str(payload.get('message', ''))[:100]!r}"
    )
    logger.debug(f"[Reply] Full payload keys: {list(payload.keys())}")
    try:
        session.reply(json.dumps(payload))
    except Exception as exc:
        logger.error(f"❌ [Reply] Failed to send payload: {exc}", exc_info=True)
        return

    # Record in conversation memory so follow-up messages have context
    message = payload.get('message', '')
    if message:
        _record_assistant_response(session, message)


def emit_webapp_generate_prompt(session: Session) -> None:
    """The web-app pause prompt: after the model + GUI are built, ask the user to
    review or generate — instead of auto-running code generation. Single source of
    truth for the prompt (called from every path where a web-app GUI completes)."""
    reply_payload(session, {
        "action": "assistant_message",
        "message": (
            "Your app is fully modeled now — the data classes and their screens "
            "are both ready. Take a look, and when you're happy with it just say "
            "**generate the web app** and I'll build the code."
        ),
        "suggestedActions": [
            {"label": "Generate the web app", "prompt": "generate web app"},
            {"label": "Review the spec", "prompt": "describe my diagram"},
            {"label": "Make a change", "prompt": ""},
        ],
    })


def _send_to_session(session: Session, payload: Dict[str, Any]):
    """Low-level helper: serialize *payload* as JSON and send it via the session.

    This mirrors the mechanism used by :func:`reply_payload` — a single
    ``session.reply(json.dumps(...))`` call — so streaming messages travel
    through the exact same WebSocket path as every other server-initiated
    message.
    """
    session.reply(json.dumps(payload))


def _record_assistant_response(session: Session, content: str) -> None:
    """Best-effort recording of assistant response in conversation memory."""
    try:
        if content and len(content) > 5:  # skip trivial messages
            # Keyed on the stable payload sessionId so memory survives
            # WebSocket reconnects (BAF session ids churn) — see B-5.
            mem = get_memory(memory_session_key(session))
            mem.add_assistant(content[:500])  # cap to avoid bloating memory
    except Exception as exc:
        logger.debug(f"Recording assistant response failed (best-effort): {exc}")


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
    session: Session, llm_instance: Any, prompt: str, system_prompt: str = "",
    model: Optional[str] = None,
) -> str:
    """Stream an LLM response token-by-token to the frontend.

    Attempts real OpenAI streaming via ``llm_instance.client``.  Falls back
    to a single-chunk send if the client doesn't support streaming.

    Args:
        session: The WebSocket session.
        llm_instance: The BESSER LLMOpenAI instance (has ``.client``).
        prompt: The user prompt.
        system_prompt: Optional system prompt.
        model: Optional per-call model override (see ``model_config``);
            defaults to the instance's configured model.

    Returns:
        The full completed text.
    """
    stream_id = reply_stream_start(session)
    full_text = ""

    try:
        # BYOK: when the current request carries a user-supplied key, route
        # this conversational/help/describe call through the user's own
        # per-request client. Anthropic/Mistral don't share OpenAI's
        # streaming chunk shape, so BYOK replies are fetched whole and sent
        # as a single chunk through the same stream protocol. SDK
        # auth/rate-limit errors propagate to the handler below.
        from byok import get_active_client
        byok_client = get_active_client()
        if byok_client is not None:
            # No system/user split in the free-text BYOK shape; fold the
            # system prompt into the prompt when present.
            byok_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            full_text = byok_client.predict_raw(
                byok_prompt,
                model=model,
                json_mode=False,
                temperature=LLM_TEXT_TEMPERATURE,
            )
            if full_text:
                reply_stream_chunk(session, full_text, stream_id)
        else:
            client = getattr(llm_instance, 'client', None)
            if client is not None and hasattr(client, 'chat'):
                # Real streaming via OpenAI SDK
                full_text = _stream_openai(
                    session, client, prompt, system_prompt, stream_id,
                    model=model or getattr(llm_instance, 'name', MODEL_CLASSIFIER),
                )
            else:
                # Fallback: single-chunk non-streaming
                response = llm_instance.predict(prompt)
                full_text = response if isinstance(response, str) else str(response)
                reply_stream_chunk(session, full_text, stream_id)

    except Exception as e:
        logger.error(f"❌ [Streaming] Error: {e}")
        if not full_text:
            from errors import classify_error, get_recovery_hint
            error_code = classify_error(e)
            hint = get_recovery_hint(error_code)
            full_text = f"{hint['message']} Please {hint['recovery']}."
            reply_stream_chunk(session, full_text, stream_id)

    reply_stream_done(session, stream_id, full_text)
    _record_assistant_response(session, full_text)
    return full_text


def _stream_openai(
    session: Session,
    client: Any,
    prompt: str,
    system_prompt: str,
    stream_id: str,
    model: str = MODEL_CLASSIFIER,
) -> str:
    """Real token-by-token streaming using the OpenAI SDK.

    Sends each delta chunk to the frontend immediately, giving users
    instant visual feedback instead of waiting for the full response.
    Tracks token usage via stream_options.
    """
    from tracking import get_tracker

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    full_text = ""
    chunk_buffer = ""
    # Buffer threshold — balance between WebSocket flood (too small) and
    # perceived latency (too large).  ~200 chars ≈ 1-2 sentences, giving
    # fluid streaming without overwhelming the connection.
    _BUFFER_THRESHOLD = STREAM_BUFFER_THRESHOLD

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=LLM_TEXT_TEMPERATURE,
        max_completion_tokens=LLM_MAX_TOKENS_TEXT,
        stream=True,
        stream_options={"include_usage": True},
    )

    usage = None
    try:
        for event in stream:
            # Final chunk with usage stats
            if hasattr(event, 'usage') and event.usage is not None:
                usage = event.usage

            if not event.choices:
                continue

            delta = event.choices[0].delta
            content = getattr(delta, 'content', None)
            if content:
                full_text += content
                chunk_buffer += content

                # Flush buffer when it has enough content or hits a natural break
                if (
                    len(chunk_buffer) >= _BUFFER_THRESHOLD
                    or content.endswith(('\n', '.', '!', '?', ':'))
                ):
                    reply_stream_chunk(session, chunk_buffer, stream_id)
                    chunk_buffer = ""
    except Exception as exc:
        logger.error(f"❌ [Streaming] Mid-stream error: {exc}")
        # Partial text already captured in full_text; fall through to flush
    finally:
        # Always flush remaining buffer so no text is silently lost
        if chunk_buffer:
            reply_stream_chunk(session, chunk_buffer, stream_id)

    # Track tokens
    if usage:
        tracker = get_tracker()
        tracker.record_from_usage(usage, model=model)

    return full_text


# ------------------------------------------------------------------
# Generation routing
# ------------------------------------------------------------------

def route_to_generation(session: Session) -> bool:
    """Detect generation workflow requests or frontend callback events."""
    request = parse_assistant_request(session)
    return should_route_to_generation(session, request)
