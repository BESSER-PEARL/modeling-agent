"""Wire actions: the ``action`` of every message the agent receives or sends.

The editor's ``features/assistant/services/`` parses these strings, so they are
the wire contract (``docs/source/websocket_protocol.rst``). ``Action`` is a
``StrEnum``: a member equals its string and ``json.dumps`` writes the bare
string, so using a member instead of a literal leaves the JSON unchanged.

The builders return the envelopes that are built in many places, with the key
order of the literals they replace (``json.dumps`` keeps insertion order).
"""

from enum import StrEnum
from typing import Any, Dict, Optional


class Action(StrEnum):
    # Inbound (editor -> agent)
    USER_MESSAGE = "user_message"
    USER_VOICE = "user_voice"
    USER_SET_VARIABLE = "user_set_variable"
    FRONTEND_EVENT = "frontend_event"
    REPLAY_LAST_RESPONSE = "replay_last_response"

    # Outbound, terminal
    INJECT_ELEMENT = "inject_element"
    INJECT_COMPLETE_SYSTEM = "inject_complete_system"
    MODIFY_MODEL = "modify_model"
    ASSISTANT_MESSAGE = "assistant_message"
    AGENT_ERROR = "agent_error"
    CREATE_DIAGRAM_TAB = "create_diagram_tab"
    TRIGGER_GENERATOR = "trigger_generator"
    TRIGGER_SMART_GENERATOR = "trigger_smart_generator"
    TRIGGER_GITHUB_IMPORT = "trigger_github_import"
    TRIGGER_EXPORT = "trigger_export"
    TRIGGER_DEPLOY = "trigger_deploy"
    TRIGGER_GITHUB_SAVE = "trigger_github_save"
    AUTO_GENERATE_GUI = "auto_generate_gui"
    # After a streamed answer to a trace question: a button opening the
    # editor's Traceability page on a view (and an artifact of the digest).
    SHOW_TRACE = "show_trace"

    # Outbound, non-terminal
    PROGRESS = "progress"
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_DONE = "stream_done"

    # BAF's own frame around each reply string (set by BAF, never by the agent)
    AGENT_REPLY_STR = "agent_reply_str"


def assistant_message(message: Any, **fields: Any) -> Dict[str, Any]:
    """``{"action": "assistant_message", "message": ..., **fields}``."""
    return {"action": Action.ASSISTANT_MESSAGE, "message": message, **fields}


# The views of the editor's Traceability page a ``show_trace`` can open.
TRACE_VIEWS = ("lineage", "coverage", "timeline")


def show_trace(view: str, message: str, artifact_id: Optional[str] = None) -> Dict[str, Any]:
    """``{"action": "show_trace", "view": ..., ["artifactId": ...,] "message": ...}``.

    *artifact_id* must come from the request's ``traceDigest.focus``; it is
    left out when None.
    """
    if view not in TRACE_VIEWS:
        raise ValueError(f"unknown trace view {view!r}")
    payload: Dict[str, Any] = {"action": Action.SHOW_TRACE, "view": view}
    if artifact_id:
        payload["artifactId"] = artifact_id
    payload["message"] = message
    return payload


def inject_element(element: Dict[str, Any], diagram_type: str, message: str) -> Dict[str, Any]:
    """One element added to the canvas."""
    return {
        "action": Action.INJECT_ELEMENT,
        "element": element,
        "diagramType": diagram_type,
        "message": message,
    }


def inject_complete_system(
    system_spec: Dict[str, Any], diagram_type: str, message: str, **fields: Any,
) -> Dict[str, Any]:
    """A whole diagram injected from a system spec."""
    return {
        "action": Action.INJECT_COMPLETE_SYSTEM,
        "systemSpec": system_spec,
        "diagramType": diagram_type,
        "message": message,
        **fields,
    }


def model_reply(action: Action, diagram_type: str, model: Dict[str, Any], message: str) -> Dict[str, Any]:
    """A reply carrying the editor model itself (Quantum Circuit, GUI): *action*
    is ``inject_element``, ``inject_complete_system`` or ``modify_model``."""
    return {"action": action, "diagramType": diagram_type, "model": model, "message": message}
