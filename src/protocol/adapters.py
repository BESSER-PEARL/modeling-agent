import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from baf.core.session import Session
from baf.library.transition.events.base_events import ReceiveJSONEvent

from .types import AssistantRequest, FileAttachment, WorkspaceContext, SUPPORTED_DIAGRAM_TYPES
from common import agent_config
from common.session_keys import PARSED_ASSISTANT_REQUEST, PARSED_REQUEST_EVENT_ID, VOICE_CONTEXT
from .message_limits import validate_message_length
from .actions import Action


def _cap_user_message(msg: str) -> str:
    """Accept the full request or reject it; a partial spec is not safe input."""
    return validate_message_length(msg)

logger = logging.getLogger(__name__)

DIAGRAM_PREFIX_PATTERN = re.compile(r"^\[DIAGRAM_TYPE:(\w+)\]\s*(.+)$", re.DOTALL)

# Opt-in study participant labels (short tokens). Mirrors the telemetry
# collector's contract; anything else is dropped at the protocol boundary.
PILOT_PARTICIPANT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,16}$")


# Well above what the editor sends; the client is not trusted to keep it small.
MAX_REPO_RECAP_CHARS = 4_000
# ``recentHistory`` caps, the same as the editor's (MAX_RECENT_HISTORY_TURNS /
# _CHARS in AssistantClient.ts): enough turns to rebuild the conversation memory
# (its window is 20 messages), enforced again here.
MAX_RECENT_HISTORY_TURNS = 20
MAX_RECENT_HISTORY_CHARS = 4_000
_RECENT_HISTORY_ROLES = frozenset({"user", "assistant"})
# GitHub's own limits: owner up to 39 characters, repository up to 100.
REPO_FULL_NAME_PATTERN = re.compile(r"[A-Za-z0-9-]{1,39}/[A-Za-z0-9._-]{1,100}")


def _parse_pilot_participant(context_payload: Dict[str, Any]) -> Optional[str]:
    """Validate the optional ``context.pilotParticipant`` label (or drop it)."""
    candidate = context_payload.get("pilotParticipant")
    if isinstance(candidate, str) and PILOT_PARTICIPANT_PATTERN.match(candidate):
        return candidate
    return None


def _parse_github_repo(context_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate the optional ``context.githubRepo`` (or drop it)."""
    repo = context_payload.get("githubRepo")
    if (
        not isinstance(repo, dict)
        or not isinstance(repo.get("fullName"), str)
        or not REPO_FULL_NAME_PATTERN.fullmatch(repo["fullName"])
    ):
        return None
    recap = repo.get("recap")
    return {
        "full_name": repo["fullName"],
        "has_code": repo.get("hasCode") is True,
        "recap": recap[:MAX_REPO_RECAP_CHARS] if isinstance(recap, str) else "",
    }


# ``context.traceDigest`` (the editor's summary of the project trace). The
# enumerations the agent explains to the user; any other value is dropped.
TRACE_OUT_OF_DATE_REASONS = ("SOURCE_CHANGED", "TARGET_CHANGED", "UPSTREAM_CHANGED")
TRACE_CHANGE_KINDS = ("CREATED", "UPDATED", "DELETED")
TRACE_OPERATION_TYPES = ("ModelingSession", "CodeGeneration", "RequirementExtraction", "Conversation")


def _digest_str(value: Any, cap: int) -> Optional[str]:
    """*value* stripped and cut at *cap* characters, or None when it is not a non-empty string."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:cap] if value else None


def _digest_count(value: Any) -> Optional[int]:
    """A non-negative int (never a bool) up to ``TRACE_DIGEST_MAX_COUNT``, else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if not 0 <= value <= agent_config.TRACE_DIGEST_MAX_COUNT:
        return None
    return value


def _digest_names(value: Any, allowed: Optional[Tuple[str, ...]] = None) -> List[str]:
    """A list of short strings (only *allowed* ones when given), capped."""
    if not isinstance(value, list):
        return []
    names = [_digest_str(v, agent_config.TRACE_DIGEST_MAX_NAME_CHARS) for v in value]
    names = [n for n in names if n and (allowed is None or n in allowed)]
    return names[:agent_config.TRACE_DIGEST_MAX_LIST_ITEMS]


def _digest_fields(item: Dict[str, Any], names=(), texts=(), lists=(), counts=()) -> Dict[str, Any]:
    """The valid fields of *item*, by kind; invalid and unknown ones are dropped."""
    out: Dict[str, Any] = {}
    for key in names:
        value = _digest_str(item.get(key), agent_config.TRACE_DIGEST_MAX_NAME_CHARS)
        if value:
            out[key] = value
    for key in texts:
        value = _digest_str(item.get(key), agent_config.TRACE_DIGEST_MAX_TEXT_CHARS)
        if value:
            out[key] = value
    for key in lists:
        value = _digest_names(item.get(key))
        if value:
            out[key] = value
    for key in counts:
        value = _digest_count(item.get(key))
        if value is not None:
            out[key] = value
    return out


def _digest_list(value: Any, cap: int, parse_item) -> List[Dict[str, Any]]:
    """The first *cap* items of *value* that *parse_item* accepts (it returns None to drop one)."""
    if not isinstance(value, list):
        return []
    items = []
    for raw in value:
        if len(items) >= cap:
            break
        item = parse_item(raw) if isinstance(raw, dict) else None
        if item:
            items.append(item)
    return items


def _digest_operation(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = _digest_fields(raw, names=("id", "type", "status", "startedAt"), texts=("summary",),
                          lists=("by", "tools"))
    return item if item.get("type") or item.get("summary") else None


def _digest_requirement(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = _digest_fields(raw, names=("id", "kind"), texts=("text", "quote"), lists=("realizedBy",))
    if isinstance(raw.get("covered"), bool):
        item["covered"] = raw["covered"]
    return item if item.get("id") or item.get("text") else None


def _digest_out_of_date(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = _digest_fields(raw, names=("source", "target", "generation"))
    reasons = _digest_names(raw.get("reasons"), TRACE_OUT_OF_DATE_REASONS)
    if reasons:
        item["reasons"] = reasons
    return item if item.get("source") or item.get("target") else None


def _digest_generation(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = _digest_fields(raw, names=("generator", "at"), counts=("files", "outOfDateFiles"))
    return item if item.get("generator") else None


def _digest_change(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = raw.get("kind")
    if kind not in TRACE_CHANGE_KINDS:
        return None
    return {"kind": kind, **_digest_fields(raw, names=("feature", "at", "by", "via"), texts=("old", "new"))}


def _digest_source(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = _digest_fields(raw, names=("type", "name"), texts=("evidence",))
    return item if item.get("name") or item.get("type") else None


def _digest_derived(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = _digest_fields(raw, names=("path", "lines"))
    reasons = _digest_names(raw.get("outOfDate"), TRACE_OUT_OF_DATE_REASONS)
    if reasons:
        item["outOfDate"] = reasons
    return item if item.get("path") else None


def _digest_focus(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = _digest_fields(raw, names=("artifactId", "name", "elementType"))
    if not item.get("artifactId") or not item.get("name"):
        return None
    for key, cap, parse_item in (
        ("history", agent_config.TRACE_DIGEST_MAX_FOCUS_HISTORY, _digest_change),
        ("derivedFrom", agent_config.TRACE_DIGEST_MAX_FOCUS_DERIVED_FROM, _digest_source),
        ("derived", agent_config.TRACE_DIGEST_MAX_FOCUS_DERIVED, _digest_derived),
    ):
        entries = _digest_list(raw.get(key), cap, parse_item)
        if entries:
            item[key] = entries
    return item


def _digest_counts(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    counts: Dict[str, Any] = _digest_fields(raw, counts=("artifacts", "links", "outOfDateLinks"))
    operations = raw.get("operations")
    if isinstance(operations, dict):
        per_type = {k: _digest_count(operations.get(k)) for k in TRACE_OPERATION_TYPES}
        per_type = {k: v for k, v in per_type.items() if v is not None}
        if per_type:
            counts["operations"] = per_type
    return counts


_DIGEST_LISTS = (
    ("recentOperations", agent_config.TRACE_DIGEST_MAX_RECENT_OPERATIONS, _digest_operation),
    ("requirements", agent_config.TRACE_DIGEST_MAX_REQUIREMENTS, _digest_requirement),
    ("outOfDate", agent_config.TRACE_DIGEST_MAX_OUT_OF_DATE, _digest_out_of_date),
    ("generations", agent_config.TRACE_DIGEST_MAX_GENERATIONS, _digest_generation),
    ("focus", agent_config.TRACE_DIGEST_MAX_FOCUS, _digest_focus),
)


def _parse_trace_digest(context_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate the optional ``context.traceDigest`` (or drop it).

    Returns a copy holding only the known fields (the wire's names) that have
    the right shape, every list and string capped (``agent_config.TRACE_DIGEST_*``):
    ``schemaVersion``, ``recording``, ``counts``, ``recentOperations``,
    ``requirements``, ``outOfDate``, ``generations``, ``focus``. A digest
    that is not an object, or whose ``schemaVersion`` is not a positive int,
    is dropped (None). Never raises: the strings in it are user data.
    """
    raw = context_payload.get("traceDigest")
    if not isinstance(raw, dict):
        return None
    version = raw.get("schemaVersion", 1)
    if _digest_count(version) is None or version < 1:
        return None
    digest: Dict[str, Any] = {
        "schemaVersion": version,
        "recording": raw.get("recording") is not False,
    }
    counts = _digest_counts(raw.get("counts"))
    if counts:
        digest["counts"] = counts
    for key, cap, parse_item in _DIGEST_LISTS:
        items = _digest_list(raw.get(key), cap, parse_item)
        if items:
            digest[key] = items
    return digest


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
    cleaned_message = _cap_user_message(cleaned_message)

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

    pilot_participant = _parse_pilot_participant(context_payload)

    context = WorkspaceContext(
        active_diagram_type=active_diagram_type,
        active_diagram_id=context_payload.get("activeDiagramId"),
        active_model=current_model,
        project_snapshot=project_snapshot,
        diagram_summaries=diagram_summaries,
        current_diagram_indices=current_diagram_indices,
        pilot_participant=pilot_participant,
        github_repo=_parse_github_repo(context_payload),
        trace_digest=_parse_trace_digest(context_payload),
        traceability_enabled=context_payload.get("traceabilityEnabled") is True,
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
        action=raw_payload.get("action") if isinstance(raw_payload.get("action"), str) else Action.USER_MESSAGE,
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
        pilot_participant=pilot_participant,
        turn_id=_parse_turn_id(raw_payload),
        recent_history=_parse_recent_history(raw_payload),
    )


def _parse_recent_history(raw_payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """The payload's ``recentHistory``, clamped: entries that are not a
    ``{"role": "user" | "assistant", "content": <non-empty string>}`` are
    dropped, each content is cut at ``MAX_RECENT_HISTORY_CHARS`` and only the
    newest ``MAX_RECENT_HISTORY_TURNS`` are kept (oldest first)."""
    raw = raw_payload.get("recentHistory")
    if not isinstance(raw, list):
        return []
    turns: List[Dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        role, content = entry.get("role"), entry.get("content")
        if not isinstance(role, str) or role not in _RECENT_HISTORY_ROLES:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        turns.append({"role": role, "content": content[:MAX_RECENT_HISTORY_CHARS]})
    return turns[-MAX_RECENT_HISTORY_TURNS:]


def _parse_turn_id(raw_payload: Dict[str, Any]) -> Optional[str]:
    turn_id = raw_payload.get("turnId")
    if isinstance(turn_id, str) and 0 < len(turn_id) <= 128:
        return turn_id
    return None


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
        cleaned_message = _cap_user_message(cleaned_message)
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
                "action": Action.USER_MESSAGE,
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
            action=Action.USER_MESSAGE,
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
