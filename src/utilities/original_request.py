"""Keep verbatim requirements bound to their project, not just a chat session."""

from common.session_keys import (
    APP_TARGET, APP_TARGET_PROJECT_ID, ORIGINAL_APP_REQUEST, ORIGINAL_APP_REQUEST_PROJECT_ID,
    SCREENS_LISTED_PROJECT_ID,
)
from protocol.message_limits import validate_message_length

_TARGET_KINDS = ("web_app", "backend", "other")


def request_project_id(request):
    snapshot = getattr(getattr(request, "context", None), "project_snapshot", None)
    value = snapshot.get("id") if isinstance(snapshot, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def remember_original_request(session, message, project_id, *, screens_listed=None):
    """Store the app description of *project_id*. Whether it lists its screens
    is *screens_listed*, else the current classifier verdict; a follow-up
    appended to the stored description keeps what that one listed."""
    from common.session_keys import UNIFIED_CLASSIFICATION

    message = validate_message_length((message or "").strip(), label="The combined original specification and follow-up requests")
    previous = original_request_for_project(session, project_id)
    if screens_listed is None:
        screens_listed = bool(getattr(session.get(UNIFIED_CLASSIFICATION), "screens_listed", None))
    appended = bool(previous) and message.startswith(previous)
    kept = appended and session.get(SCREENS_LISTED_PROJECT_ID) == project_id
    session.set(SCREENS_LISTED_PROJECT_ID, project_id if project_id and (screens_listed or kept) else None)
    session.set(ORIGINAL_APP_REQUEST, message)
    session.set(ORIGINAL_APP_REQUEST_PROJECT_ID, project_id)


def original_request_for_project(session, project_id):
    # Unknown identity is never a wildcard, including legacy unscoped stashes.
    if not project_id or session.get(ORIGINAL_APP_REQUEST_PROJECT_ID) != project_id:
        return ""
    value = session.get(ORIGINAL_APP_REQUEST)
    return value if isinstance(value, str) else ""


def clear_original_request(session):
    remember_original_request(session, "", None)
    session.set(APP_TARGET, None)
    session.set(APP_TARGET_PROJECT_ID, None)
    session.set(SCREENS_LISTED_PROJECT_ID, None)


def remember_target(session, project_id, *, kind=None, description=None, architecture=None):
    """Record what the user wants built for this project. A newly stated kind
    replaces the remembered one (latest statement wins); an architecture is
    kept until a new one is named."""
    if not project_id:
        return
    previous = target_for_project(session, project_id) or {}
    if kind in _TARGET_KINDS:
        previous = {"architecture": previous.get("architecture"), "kind": kind,
                    "description": (description or "").strip()[:120]}
    if architecture and architecture.strip():
        previous = {**previous, "architecture": architecture.strip()[:60]}
    if not previous:
        return
    session.set(APP_TARGET, previous)
    session.set(APP_TARGET_PROJECT_ID, project_id)


def target_for_project(session, project_id):
    """``{"kind", "description", "architecture"}`` remembered for *project_id*,
    or None. ``kind`` is None when only an architecture was named."""
    if not project_id or session.get(APP_TARGET_PROJECT_ID) != project_id:
        return None
    value = session.get(APP_TARGET)
    return value if isinstance(value, dict) and value else None


def current_target(session, request):
    """The target for this request: what this message stated, else what was
    remembered for its project. ``{"kind", "description", "architecture"}`` or None."""
    from common.session_keys import UNIFIED_CLASSIFICATION

    remembered = target_for_project(session, request_project_id(request)) or {}
    verdict = session.get(UNIFIED_CLASSIFICATION)
    kind = getattr(verdict, "target_kind", None)
    stated = {
        "kind": kind,
        "description": getattr(verdict, "target_description", None) if kind else None,
        "architecture": getattr(verdict, "architecture", None),
    }
    merged = {key: stated.get(key) or remembered.get(key) for key in ("kind", "description", "architecture")}
    if kind:
        merged["description"] = stated["description"]
    return merged if any(merged.values()) else None
