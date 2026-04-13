import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from baf.core.session import Session
from baf.library.transition.events.base_events import ReceiveJSONEvent

from .types import AssistantRequest, FileAttachment, WorkspaceContext, SUPPORTED_DIAGRAM_TYPES
from session_keys import PARSED_ASSISTANT_REQUEST, PARSED_REQUEST_EVENT_ID, VOICE_CONTEXT

logger = logging.getLogger(__name__)

DIAGRAM_PREFIX_PATTERN = re.compile(r"^\[DIAGRAM_TYPE:(\w+)\]\s*(.+)$", re.DOTALL)


def safe_json_loads(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_diagram_type(diagram_type: Any, default: str = "ClassDiagram") -> str:
    if isinstance(diagram_type, str) and diagram_type in SUPPORTED_DIAGRAM_TYPES:
        return diagram_type
    return default


def extract_event_payload(session: Session) -> Dict[str, Any]:
    if not session or not session.event:
        return {}

    event = session.event

    # Prefer structured payloads first for any event type.
    # Some runtimes expose `data/json` even when the event is logged as receive_message_text.
    json_payload = getattr(event, "json", None)
    if isinstance(json_payload, dict):
        return json_payload
    data_payload = getattr(event, "data", None)
    if isinstance(data_payload, dict):
        return data_payload
    payload_attr = getattr(event, "payload", None)
    if isinstance(payload_attr, dict):
        return payload_attr

    # Legacy path for explicit JSON event class.
    if isinstance(event, ReceiveJSONEvent):
        if isinstance(json_payload, dict):
            return json_payload
        if isinstance(data_payload, dict):
            return data_payload

    message = getattr(event, "message", None)
    if isinstance(message, dict):
        return message
    parsed = safe_json_loads(message)
    if parsed:
        return parsed

    # Fallback for runtimes that keep raw JSON on non-standard fields.
    for attr in ("text", "raw", "body"):
        candidate = getattr(event, attr, None)
        if isinstance(candidate, dict):
            return candidate
        parsed_candidate = safe_json_loads(candidate)
        if parsed_candidate:
            return parsed_candidate

    return {}


def _unwrap_v2_envelope(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap BESSER websocket payload envelope when v2 JSON is serialized in `message`.

    BESSER websocket keeps top-level fields such as `action`, `message`, `user_id`.
    Our v2 assistant payload is embedded as JSON string in `message`, so we need to
    recover it here.
    """
    if not isinstance(raw_payload, dict):
        return {}

    nested_message = raw_payload.get("message")
    nested_payload = safe_json_loads(nested_message)
    if not isinstance(nested_payload, dict):
        return raw_payload

    has_context = isinstance(nested_payload.get("context"), dict)
    has_v2_shape = (
        nested_payload.get("protocolVersion") == "2.0"
        and isinstance(nested_payload.get("action"), str)
        and isinstance(nested_payload.get("message"), str)
    )
    if not has_context and not has_v2_shape:
        return raw_payload

    merged_payload = dict(raw_payload)
    merged_payload.update(nested_payload)
    return merged_payload


def _derive_diagram_summaries_from_snapshot(project_snapshot: Any) -> list[Dict[str, Any]]:
    if not isinstance(project_snapshot, dict):
        return []
    diagrams = project_snapshot.get("diagrams")
    if not isinstance(diagrams, dict):
        return []

    summaries: list[Dict[str, Any]] = []
    for diagram_type, payload in diagrams.items():
        if not isinstance(diagram_type, str):
            continue
        # New format: payload is an array of diagram tabs
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    summaries.append({"diagramType": diagram_type})
                    continue
                summaries.append(
                    {
                        "diagramType": diagram_type,
                        "diagramId": item.get("id") if isinstance(item.get("id"), str) else None,
                        "title": item.get("title") if isinstance(item.get("title"), str) else None,
                    }
                )
            continue
        # Legacy format: payload is a single dict
        if not isinstance(payload, dict):
            summaries.append({"diagramType": diagram_type})
            continue
        summaries.append(
            {
                "diagramType": diagram_type,
                "diagramId": payload.get("id") if isinstance(payload.get("id"), str) else None,
                "title": payload.get("title") if isinstance(payload.get("title"), str) else None,
            }
        )
    return summaries


def strip_diagram_prefix(message: str) -> Tuple[str, Optional[str]]:
    if not isinstance(message, str):
        return "", None
    match = DIAGRAM_PREFIX_PATTERN.match(message.strip())
    if not match:
        return message.strip(), None
    return match.group(2).strip(), match.group(1)


def parse_v2_payload(raw_payload: Dict[str, Any], default_diagram_type: str = "ClassDiagram") -> AssistantRequest:
    raw_payload = _unwrap_v2_envelope(raw_payload)

    context_payload = raw_payload.get("context")
    context_payload = context_payload if isinstance(context_payload, dict) else {}

    raw_message = raw_payload.get("message")
    message_envelope = raw_message if isinstance(raw_message, dict) else {}

    message_text = ""
    if isinstance(raw_message, str):
        message_text = raw_message
    elif isinstance(message_envelope.get("message"), str):
        message_text = message_envelope["message"]

    cleaned_message, prefixed_diagram = strip_diagram_prefix(message_text)

    payload_diagram_type = (
        context_payload.get("activeDiagramType")
        or raw_payload.get("diagramType")
        or message_envelope.get("diagramType")
        or prefixed_diagram
        or default_diagram_type
    )

    active_diagram_type = normalize_diagram_type(
        payload_diagram_type,
        default=default_diagram_type,
    )
    # NOTE: ``activeModel`` used to be read from the payload here, but as of
    # PR 6.2 the frontend no longer sends it separately.  The model is now
    # always resolved from ``projectSnapshot``.  We intentionally ignore the
    # field if it still arrives from older frontends.

    project_snapshot = (
        context_payload.get("projectSnapshot")
        if isinstance(context_payload.get("projectSnapshot"), dict)
        else None
    )

    # Resolve the current model from the project snapshot using the active
    # diagram type and tab index, instead of the deprecated ``activeModel``.
    current_model: Optional[Dict[str, Any]] = None
    if isinstance(project_snapshot, dict):
        diagrams = project_snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            target = diagrams.get(active_diagram_type)
            if isinstance(target, list):
                # Multi-tab format: use currentDiagramIndices to pick the right tab.
                raw_indices = context_payload.get("currentDiagramIndices")
                idx = (
                    raw_indices.get(active_diagram_type, 0)
                    if isinstance(raw_indices, dict)
                    else 0
                )
                if 0 <= idx < len(target) and isinstance(target[idx], dict):
                    model_candidate = target[idx].get("model")
                    if isinstance(model_candidate, dict):
                        current_model = model_candidate
                # Fallback: first tab with a model
                if current_model is None:
                    for tab in target:
                        if isinstance(tab, dict) and isinstance(tab.get("model"), dict):
                            current_model = tab["model"]
                            break
            elif isinstance(target, dict) and isinstance(target.get("model"), dict):
                # Legacy single-dict format
                current_model = target["model"]
    diagram_summaries = (
        context_payload.get("diagramSummaries")
        if isinstance(context_payload.get("diagramSummaries"), list)
        else _derive_diagram_summaries_from_snapshot(project_snapshot)
    )

    raw_indices = context_payload.get("currentDiagramIndices")
    current_diagram_indices = raw_indices if isinstance(raw_indices, dict) else None

    context = WorkspaceContext(
        active_diagram_type=active_diagram_type,
        active_diagram_id=context_payload.get("activeDiagramId"),
        active_model=current_model,
        project_snapshot=project_snapshot,
        diagram_summaries=diagram_summaries,
        current_diagram_indices=current_diagram_indices,
    )

    # ── Parse file attachments ──
    raw_attachments = raw_payload.get("attachments")
    attachments = []
    if isinstance(raw_attachments, list):
        for att in raw_attachments:
            if isinstance(att, dict) and isinstance(att.get("content"), str):
                attachments.append(
                    FileAttachment(
                        filename=att.get("filename", "unknown") if isinstance(att.get("filename"), str) else "unknown",
                        content_b64=att["content"],
                        mime_type=att.get("mimeType", "") if isinstance(att.get("mimeType"), str) else "",
                    )
                )

    return AssistantRequest(
        action=raw_payload.get("action") if isinstance(raw_payload.get("action"), str) else "user_message",
        protocol_version="2.0",
        client_mode=raw_payload.get("clientMode") if isinstance(raw_payload.get("clientMode"), str) else "workspace",
        session_id=raw_payload.get("sessionId")
        if isinstance(raw_payload.get("sessionId"), str)
        else context_payload.get("sessionId")
        if isinstance(context_payload.get("sessionId"), str)
        else None,
        message=cleaned_message,
        diagram_type=active_diagram_type,
        diagram_id=context.active_diagram_id,
        current_model=current_model,
        context=context,
        raw_payload=raw_payload,
        attachments=attachments,
    )


def parse_assistant_request(session: Session, default_diagram_type: str = "ClassDiagram") -> AssistantRequest:
    # Cache the parsed request on the session to avoid redundant JSON
    # parsing — this function is called 3-5 times per message from
    # get_user_message(), get_diagram_type(), get_current_model(),
    # _common_preamble(), reply_message(), and route_to_generation().
    #
    # IMPORTANT: The cache is keyed on the *identity* of ``session.event``
    # so that a new incoming WebSocket message (which assigns a fresh event
    # object) automatically invalidates the stale cached request.
    cache_key = PARSED_ASSISTANT_REQUEST
    event_id_key = PARSED_REQUEST_EVENT_ID

    current_event_id = id(session.event) if session and session.event else None

    if current_event_id is not None:
        try:
            cached_event_id = session.get(event_id_key)
            if cached_event_id == current_event_id:
                cached = session.get(cache_key)
                if isinstance(cached, AssistantRequest):
                    return cached
        except Exception as exc:
            logger.debug(f"Session cache read failed (best-effort): {exc}")

    raw_payload = extract_event_payload(session)

    if not raw_payload:
        event_message = getattr(session.event, "message", "")
        cleaned_message, prefixed_diagram = strip_diagram_prefix(event_message if isinstance(event_message, str) else "")
        diagram_type = normalize_diagram_type(prefixed_diagram or default_diagram_type, default=default_diagram_type)

        # Voice messages arrive as plain text (after STT) with no JSON context.
        # The frontend sends the workspace context as a session variable
        # '_voice_context' right before the audio payload.
        voice_ctx = session.get(VOICE_CONTEXT)
        if isinstance(voice_ctx, dict):
            # Consume the context so it's not reused for the next message
            session.set(VOICE_CONTEXT, None)
            # Re-parse as if it were a v2 payload with the transcribed message
            synthetic_payload = {
                "action": "user_message",
                "protocolVersion": "2.0",
                "clientMode": "workspace",
                "message": cleaned_message,
                "context": voice_ctx,
            }
            request = parse_v2_payload(synthetic_payload, default_diagram_type=default_diagram_type)
            try:
                session.set(cache_key, request)
                session.set(event_id_key, current_event_id)
            except Exception as exc:
                logger.debug(f"Session cache write failed (best-effort): {exc}")
            return request

        context = WorkspaceContext(active_diagram_type=diagram_type)
        request = AssistantRequest(
            action="user_message",
            protocol_version="2.0",
            client_mode="workspace",
            message=cleaned_message,
            diagram_type=diagram_type,
            context=context,
            raw_payload={},
        )
        try:
            session.set(cache_key, request)
            session.set(event_id_key, current_event_id)
        except Exception as exc:
            logger.debug(f"Session cache write failed (best-effort): {exc}")
        return request

    request = parse_v2_payload(raw_payload, default_diagram_type=default_diagram_type)

    if not request.diagram_type:
        request.diagram_type = normalize_diagram_type(default_diagram_type, default=default_diagram_type)
    if not request.context.active_diagram_type:
        request.context.active_diagram_type = request.diagram_type

    try:
        session.set(cache_key, request)
        session.set(event_id_key, current_event_id)
    except Exception as exc:
        logger.debug(f"Session cache write failed (best-effort): {exc}")
    return request
