"""Keep verbatim requirements bound to their project, not just a chat session."""

from session_keys import ORIGINAL_APP_REQUEST, ORIGINAL_APP_REQUEST_PROJECT_ID
from utilities.message_limits import validate_message_length


def request_project_id(request):
    snapshot = getattr(getattr(request, "context", None), "project_snapshot", None)
    value = snapshot.get("id") if isinstance(snapshot, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def remember_original_request(session, message, project_id):
    message = validate_message_length((message or "").strip(), label="The combined original specification and follow-up requests")
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
