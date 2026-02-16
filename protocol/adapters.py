import json
import re
from typing import Any, Dict, Optional, Tuple

from besser.agent.core.session import Session
from besser.agent.library.transition.events.base_events import ReceiveJSONEvent

from .types import AssistantRequest, WorkspaceContext, SUPPORTED_DIAGRAM_TYPES

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
    current_model = context_payload.get("activeModel") or message_envelope.get("activeModel")
    if not isinstance(current_model, dict):
        fallback_model = raw_payload.get("currentModel")
        current_model = fallback_model if isinstance(fallback_model, dict) else None

    project_snapshot = (
        context_payload.get("projectSnapshot")
        if isinstance(context_payload.get("projectSnapshot"), dict)
        else None
    )
    diagram_summaries = (
        context_payload.get("diagramSummaries")
        if isinstance(context_payload.get("diagramSummaries"), list)
        else _derive_diagram_summaries_from_snapshot(project_snapshot)
    )

    context = WorkspaceContext(
        active_diagram_type=active_diagram_type,
        active_diagram_id=context_payload.get("activeDiagramId"),
        active_model=current_model,
        project_snapshot=project_snapshot,
        diagram_summaries=diagram_summaries,
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
    )


def parse_assistant_request(session: Session, default_diagram_type: str = "ClassDiagram") -> AssistantRequest:
    raw_payload = extract_event_payload(session)

    if not raw_payload:
        event_message = getattr(session.event, "message", "")
        cleaned_message, prefixed_diagram = strip_diagram_prefix(event_message if isinstance(event_message, str) else "")
        diagram_type = normalize_diagram_type(prefixed_diagram or default_diagram_type, default=default_diagram_type)
        context = WorkspaceContext(active_diagram_type=diagram_type)
        return AssistantRequest(
            action="user_message",
            protocol_version="2.0",
            client_mode="workspace",
            message=cleaned_message,
            diagram_type=diagram_type,
            context=context,
            raw_payload={},
        )

    request = parse_v2_payload(raw_payload, default_diagram_type=default_diagram_type)

    if not request.diagram_type:
        request.diagram_type = normalize_diagram_type(default_diagram_type, default=default_diagram_type)
    if not request.context.active_diagram_type:
        request.context.active_diagram_type = request.diagram_type

    return request
