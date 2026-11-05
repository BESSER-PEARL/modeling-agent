"""
Session Helpers
---------------
Reply / message utilities and intent-matching condition functions.

These are pure functions with no dependency on module-level globals;
they only use the ``session`` object and the protocol adapters.
"""

import json
import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional

from baf.core.session import Session

from protocol.adapters import parse_assistant_request
from common.agent_config import STREAM_BUFFER_THRESHOLD
from common.errors import (
    EXHAUSTED_KEY_MESSAGE, ErrorCode, ModelRefusal, RATE_LIMITED_KEY_MESSAGE, REJECTED_KEY_MESSAGE,
)
from llm import byok
from llm.provider import LLMProvider
from memory import get_memory, memory_session_key
from common.session_keys import (
    LAST_SUGGESTED_PROMPTS,
    TELEMETRY_EMITTED_EVENT_ID,
    UNIFIED_CLASSIFICATION,
)
from conversation.reply_language import language_instruction, localize_payload, translate_text
from conversation.telemetry import emit_prompt_event
from routing.unified_classifier import _pending_flow_context
from protocol.actions import Action, assistant_message

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Message extraction helpers
# ------------------------------------------------------------------

def get_user_message(session: Session) -> str:
    """Extract normalized message using protocol adapters."""
    request = parse_assistant_request(session)
    return request.message or ""


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
    # Pending-flow gate (ActiveFlow): when the assistant is awaiting a reply
    # to a question (replace/keep, GUI choice, smart-gen confirmation,
    # generator menu/config), the CLASSIFIER decides — with the pending
    # question in its context — whether this message ANSWERS it or is a NEW
    # REQUEST. Answers (and no-verdict, conservatively) stay in the current
    # state for the flow handler to consume; a new_request routes NORMALLY,
    # so it never lands in the wrong state (a modify trapped in the create
    # state, a decline trapped in a config flow).
    if _pending_flow_context(session) is not None:
        _uc_flow = session.get(UNIFIED_CLASSIFICATION)
        if getattr(_uc_flow, "pending_flow_action", None) != "new_request":
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
    # unusual state machines). BAF's answer is trusted as-is here.
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        matched_intent = session.event.predicted_intent.intent
        return matched_intent.name == target_intent_name

    return False


def json_no_intent_matched(session: Session) -> bool:
    """Check if no specific intent was matched (fallback).

    Also returns True when a pending confirmation suppressed intent matching,
    so the message stays in the current state for _common_preamble to handle.
    """
    if _pending_flow_context(session) is not None:
        _uc_flow = session.get(UNIFIED_CLASSIFICATION)
        if getattr(_uc_flow, "pending_flow_action", None) != "new_request":
            return True
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        matched_intent = session.event.predicted_intent.intent
        return matched_intent.name == 'fallback_intent'
    return True


# ------------------------------------------------------------------
# Reconnect recovery: buffer the last completed reply per session
# ------------------------------------------------------------------
# A long generation can outlive its WebSocket connection: the socket reconnects
# mid-flight and BAF routes the final reply to the now-dead connection, so the
# frontend never sees it and stays stuck on "still working…". We remember the
# last TERMINAL reply per STABLE session key (which survives reconnects — see
# ``memory_session_key``) and re-send it when the frontend asks, via a
# ``replay_last_response`` control message it fires on reconnect-while-waiting.
#
# Turn-scoped replay: a client that sends a ``turnId`` gets every frame of that
# turn stamped with it plus a per-turn ``replySeq``, and its replay request names
# the awaited turn and the seqs it already applied. Only that turn's
# unacknowledged terminal replies are re-sent — never the previous turn's reply,
# never one the reconnect's outbox flush already delivered.
_TERMINAL_REPLY_ACTIONS = frozenset({
    Action.INJECT_COMPLETE_SYSTEM, Action.MODIFY_MODEL, Action.AUTO_GENERATE_GUI,
    Action.TRIGGER_GENERATOR, Action.TRIGGER_GITHUB_IMPORT, Action.ASSISTANT_MESSAGE,
})
# Replayed only within a turn, where the client applies each (turnId, replySeq)
# at most once: a missed trigger is recovered, an applied one is never re-sent.
# Kept out of the legacy replay above, which has no dedupe (a paid run could
# start twice). create_diagram_tab precedes an injection with replaceExisting,
# so it must replay with it; stream_done carries the assembled streamed text.
_TURN_REPLY_ACTIONS = _TERMINAL_REPLY_ACTIONS | frozenset({
    Action.INJECT_ELEMENT, Action.TRIGGER_SMART_GENERATOR, Action.AGENT_ERROR,
    Action.TRIGGER_EXPORT, Action.TRIGGER_DEPLOY, Action.CREATE_DIAGRAM_TAB, Action.STREAM_DONE,
})
_REPLY_BUFFER_MAX = 200
_last_reply_buffer: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
# (session key, turn id) -> {"seq": last issued replySeq, "replies": [stamped terminal payloads]}
_turn_replies: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
_turn_lock = threading.Lock()


def _stamp_turn(session: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *payload* stamped with the request's turnId and the next
    replySeq, or *payload* unchanged when the request carries no turnId."""
    try:
        request = parse_assistant_request(session)
        turn_id = getattr(request, "turn_id", None)
        if not turn_id:
            return payload
        key = (memory_session_key(session, request), turn_id)
    except Exception as exc:
        logger.debug(f"[ReplayBuffer] turn stamp skipped (best-effort): {exc}")
        return payload
    with _turn_lock:
        state = _turn_replies.get(key)
        if state is None:
            state = _turn_replies[key] = {"seq": 0, "replies": []}
            while len(_turn_replies) > _REPLY_BUFFER_MAX:
                _turn_replies.popitem(last=False)
        else:
            # LRU on use, not creation: a long turn must not be evicted mid-way,
            # or its replySeq restarts and the client drops the final reply.
            _turn_replies.move_to_end(key)
        state["seq"] += 1
        return {**payload, "turnId": turn_id, "replySeq": state["seq"]}


def _buffer_terminal_reply(session: Session, payload: Dict[str, Any]) -> None:
    """Remember this session's last terminal reply so a reconnect that dropped
    it can replay it. Best-effort — never breaks the actual reply. Bounded LRU
    so it can't grow without limit."""
    try:
        action = payload.get("action")
        if action not in _TURN_REPLY_ACTIONS:
            return
        key = memory_session_key(session)
        if action in _TERMINAL_REPLY_ACTIONS:
            _last_reply_buffer[key] = payload
            _last_reply_buffer.move_to_end(key)
            while len(_last_reply_buffer) > _REPLY_BUFFER_MAX:
                _last_reply_buffer.popitem(last=False)
        turn_id = payload.get("turnId")
        if turn_id:
            with _turn_lock:
                state = _turn_replies.get((key, turn_id))
                if state is not None:
                    state["replies"].append(payload)
    except Exception as exc:
        logger.debug(f"[ReplayBuffer] store failed (best-effort): {exc}")


def replay_last_reply(session: Session, request: Any = None) -> bool:
    """Re-send buffered terminal replies after a reconnect.

    With a ``turnId`` in the request: that turn's terminal replies whose
    ``replySeq`` is not in ``appliedSeqs``, in order, with their original stamps.
    Without one (older client, voice turn): the session's last terminal reply.
    Sends to the session's CURRENT live connection (the heartbeat/slot-reclaim
    has already rebound it). Returns True when something was replayed; safe
    no-op otherwise. Never re-runs generation."""
    try:
        key = memory_session_key(session, request)
        turn_id = getattr(request, "turn_id", None)
        if turn_id:
            raw = getattr(request, "raw_payload", None) or {}
            applied = raw.get("appliedSeqs")
            applied = set(applied) if isinstance(applied, list) else set()
            with _turn_lock:
                state = _turn_replies.get((key, turn_id)) or {}
                pending = [p for p in state.get("replies", []) if p.get("replySeq") not in applied]
            logger.info(f"[Replay] turn {turn_id!r}: re-sending {len(pending)} unacknowledged reply(ies)")
            for payload in pending:
                session.reply(json.dumps(payload))  # keep the original stamps
            return bool(pending)
        payload = _last_reply_buffer.get(key)
        if payload is None:
            logger.info("[Replay] no buffered reply for this session — nothing to replay")
            return False
        logger.info(f"[Replay] re-sending last '{payload.get('action')}' reply after reconnect")
        session.reply(json.dumps(payload))  # keep the original stamps, if any
        return True
    except Exception as exc:
        logger.debug(f"[Replay] failed (best-effort): {exc}")
        return False


# ------------------------------------------------------------------
# Opt-in study telemetry
# ------------------------------------------------------------------

def _emit_prompt_telemetry(session: Session, action: str) -> None:
    """Record what the user asked and what the agent did with it.

    Called from the reply choke points (``reply_message`` / ``reply_payload``
    / ``reply_stream_done``) — the single funnel every terminal reply flows
    through — instead of instrumenting individual handlers. Emits at most ONE
    ``prompt`` event per incoming user message (first reply wins, keyed on the
    event identity like the request-parse cache), and only when the request
    carries a participant label — untagged sessions never produce
    telemetry. Best-effort by contract: any failure is swallowed and logged
    at debug level so telemetry can never delay or break a reply.
    """
    try:
        request = parse_assistant_request(session)
        participant = getattr(request, "pilot_participant", None)
        session_id = getattr(request, "session_id", None)
        if not participant or not session_id:
            return
        event_id = id(session.event) if session and session.event else None
        if event_id is not None and session.get(TELEMETRY_EMITTED_EVENT_ID) == event_id:
            return
        try:
            session.set(TELEMETRY_EMITTED_EVENT_ID, event_id)
        except Exception:
            pass
        emit_prompt_event(
            session_id=session_id,
            participant=participant,
            text=request.message or "",
            action_taken=action,
            diagram_type=getattr(request, "diagram_type", None),
        )
    except Exception as exc:
        logger.debug(f"[Telemetry] reply hook failed (best-effort): {exc}")


# ------------------------------------------------------------------
# Reply helpers
# ------------------------------------------------------------------

def reply_message(session: Session, message: str, *, telemetry_exempt: bool = False):
    """Send assistant message, wrapped for v2 protocol clients."""
    try:
        request = parse_assistant_request(session)
        shown = translate_text(session, message)
        if request.is_v2:
            sent = _send_to_session(session, assistant_message(shown))
        else:
            session.reply(shown)
    except Exception as exc:
        logger.error(f"❌ [Reply] Failed to send message: {exc}", exc_info=True)
        return

    # Record in conversation memory
    _record_assistant_response(session, message)

    # Buffer for reconnect recovery — assistant_message is the v2 terminal reply.
    try:
        if request.is_v2:
            _buffer_terminal_reply(session, sent)
    except Exception:
        pass

    # Telemetry: a plain message is an "assistant_message" reply.
    # ``telemetry_exempt`` marks the rare mid-turn notice (e.g. the long-
    # message truncation warning) that must not claim the turn's one event.
    if not telemetry_exempt:
        _emit_prompt_telemetry(session, Action.ASSISTANT_MESSAGE)


def reply_payload(session: Session, payload: Dict[str, Any]):
    """Send JSON payload response for both protocol versions."""
    logger.info(
        f"📤 [Reply] Sending payload: action={payload.get('action')}, "
        f"diagramType={payload.get('diagramType')}, "
        f"replaceExisting={payload.get('replaceExisting', 'NOT SET')}, "
        f"message={str(payload.get('message', ''))[:100]!r}"
    )
    logger.debug(f"[Reply] Full payload keys: {list(payload.keys())}")
    _remember_suggested_prompts(session, payload)
    try:
        sent = _send_to_session(session, localize_payload(session, payload))
    except Exception as exc:
        logger.error(f"❌ [Reply] Failed to send payload: {exc}", exc_info=True)
        return

    # Record in conversation memory so follow-up messages have context
    message = payload.get('message', '')
    if message:
        _record_assistant_response(session, message)

    # Buffer for reconnect recovery so a dropped terminal reply can be replayed.
    _buffer_terminal_reply(session, sent)

    # Telemetry: record the reply's action as what the agent did with
    # the user's message (progress keep-alives are not the reply).
    action = payload.get('action')
    if isinstance(action, str) and action != Action.PROGRESS:
        _emit_prompt_telemetry(session, action)


def _remember_suggested_prompts(session: Session, payload: Dict[str, Any]) -> None:
    """Remember this reply's quick-action prompts: a click sends one back as
    the user's message (see ``LAST_SUGGESTED_PROMPTS``)."""
    actions = payload.get("suggestedActions")
    if not isinstance(actions, list) or not actions:
        return
    try:
        session.set(LAST_SUGGESTED_PROMPTS, [
            a["prompt"].strip() for a in actions
            if isinstance(a, dict) and isinstance(a.get("prompt"), str) and a["prompt"].strip()
        ])
    except Exception as exc:
        logger.debug(f"Remembering quick-action prompts failed (best-effort): {exc}")


def emit_webapp_generate_prompt(session: Session) -> None:
    """The web-app pause prompt: after the app's spec + screens are built, ask the
    user to review or generate — instead of auto-running code generation. Single
    source of truth for the prompt (called from every path where a web-app GUI
    completes)."""
    from conversation.suggestions import get_post_spec_suggestions
    from conversation.reply_copy import continue_generating_prompt
    reply_payload(session, assistant_message(
        "Your screens are ready. " + continue_generating_prompt("web app"),
        suggestedActions=get_post_spec_suggestions("web_app"),
    ))


# Both error codes make the editor offer its API key button.
_USER_KEY_REPLIES: Dict[byok.KeyFault, Dict[str, Any]] = {
    byok.KeyFault.REJECTED: {
        "errorCode": ErrorCode.AUTH_ERROR.value,
        "message": REJECTED_KEY_MESSAGE,
        "retryable": False,
        "suggestedRecovery": "Check or replace your API key",
    },
    byok.KeyFault.EXHAUSTED: {
        "errorCode": ErrorCode.RATE_LIMIT.value,
        "message": EXHAUSTED_KEY_MESSAGE,
        "retryable": False,
        "suggestedRecovery": "Add credit or use a different API key",
    },
    # No suggestedRecovery: the editor's "Try again" chip sends it as the prompt.
    byok.KeyFault.RATE_LIMITED: {
        "errorCode": ErrorCode.RATE_LIMIT.value,
        "message": RATE_LIMITED_KEY_MESSAGE,
        "retryable": True,
    },
}


def user_key_reply(error: Optional[BaseException]) -> Optional[Dict[str, Any]]:
    """The reply for a provider refusing or limiting the user's own key, else
    ``None``. The provider's text never reaches the user."""
    fault = byok.user_key_fault(error)
    if fault is None:
        return None
    return {"action": Action.AGENT_ERROR, **_USER_KEY_REPLIES[fault]}


def _send_to_session(session: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Low-level helper: stamp *payload* with the turn (if any), serialize it as
    JSON and send it via the session. Every reply helper sends through here, so
    streaming and progress frames take the same WebSocket path as terminal
    replies. Returns the payload as sent.
    """
    payload = _stamp_turn(session, payload)
    session.reply(json.dumps(payload))
    return payload


def _record_assistant_response(session: Session, content: str) -> None:
    """Best-effort recording of assistant response in conversation memory."""
    try:
        if content and len(content) > 5:  # skip trivial messages
            # Keyed on the stable payload sessionId so memory survives
            # WebSocket reconnects (BAF session ids churn).
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
        "action": Action.STREAM_START,
        "streamId": stream_id,
    }
    _send_to_session(session, payload)
    return stream_id


def reply_stream_chunk(session: Session, chunk: str, stream_id: str):
    """Send a streaming text chunk to the frontend."""
    payload = {
        "action": Action.STREAM_CHUNK,
        "streamId": stream_id,
        "chunk": chunk,
        "done": False,
    }
    _send_to_session(session, payload)


def reply_stream_done(session: Session, stream_id: str, full_text: str = ""):
    """Signal the end of a streaming response."""
    payload = {
        "action": Action.STREAM_DONE,
        "streamId": stream_id,
        "fullText": full_text,
        "done": True,
    }
    _buffer_terminal_reply(session, _send_to_session(session, payload))

    # Telemetry: a completed stream is a conversational reply — the
    # frontend renders it as an assistant message, so record it as one.
    _emit_prompt_telemetry(session, Action.ASSISTANT_MESSAGE)


def reply_progress(session: Session, message: str, step: int = 0, total: int = 0):
    """Send a progress indicator to the frontend."""
    payload = {
        "action": Action.PROGRESS,
        "message": translate_text(session, message),
        "step": step,
        "total": total,
    }
    _send_to_session(session, payload)


def stream_llm_response(
    session: Session, llm_instance: Any, prompt: str, system_prompt: str = "",
    model: Optional[str] = None,
) -> str:
    """Stream an LLM response token-by-token to the frontend.

    Args:
        session: The WebSocket session.
        llm_instance: The BESSER LLMOpenAI instance the call runs on (see
            ``llm.provider.LLMProvider.stream``).
        prompt: The user prompt.
        system_prompt: Optional system prompt.
        model: Optional per-call model override (see ``model_config``);
            defaults to the instance's configured model.

    Returns:
        The full completed text.
    """
    instruction = language_instruction(session)
    if instruction:
        system_prompt = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction
    stream_id = reply_stream_start(session)
    full_text = ""

    try:
        chunks = LLMProvider(llm_instance).stream(prompt, system_prompt=system_prompt, model=model)
        full_text = _send_chunks(session, chunks, stream_id)
    except Exception as e:
        logger.error(f"❌ [Streaming] Error: {e}")
        key_reply = user_key_reply(e)
        if not full_text and key_reply is not None:
            # An error reply, not a chunk: the stream below closes empty.
            reply_payload(session, key_reply)
        elif not full_text:
            from common.errors import classify_error, get_recovery_hint
            if isinstance(e, ModelRefusal):
                full_text = e.user_message()
            else:
                error_code = classify_error(e)
                hint = get_recovery_hint(error_code)
                full_text = f"{hint['message']} Please {hint['recovery']}."
            reply_stream_chunk(session, full_text, stream_id)

    reply_stream_done(session, stream_id, full_text)
    _record_assistant_response(session, full_text)
    return full_text


def _send_chunks(session: Session, chunks: Iterable[str], stream_id: str) -> str:
    """Send text chunks to the frontend as they arrive, buffered to about a
    sentence, and return the whole text. A mid-stream failure keeps the text
    sent so far; a refusal propagates."""
    full_text = ""
    chunk_buffer = ""
    try:
        for content in chunks:
            full_text += content
            chunk_buffer += content
            # Flush on a full buffer or a natural break: neither floods the
            # WebSocket nor holds text back.
            if (
                len(chunk_buffer) >= STREAM_BUFFER_THRESHOLD
                or content.endswith(('\n', '.', '!', '?', ':'))
            ):
                reply_stream_chunk(session, chunk_buffer, stream_id)
                chunk_buffer = ""
    except ModelRefusal:
        raise
    except Exception as exc:
        logger.error(f"❌ [Streaming] Mid-stream error: {exc}")
    finally:
        # Always flush remaining buffer so no text is silently lost
        if chunk_buffer:
            reply_stream_chunk(session, chunk_buffer, stream_id)
    return full_text
