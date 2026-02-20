import re
from typing import Any, Dict, List, Optional, Tuple

from besser.agent.core.session import Session

from protocol.types import AssistantRequest

GENERATOR_KEYWORDS: Dict[str, List[str]] = {
    "django": ["django"],
    "backend": ["full backend", "backend"],
    "web_app": [
        "web app",
        "web application",
        "frontend app",
        "frontend generator",
        "gui app",
        "gui generator",
        "graphical ui",
        "generate ui",
        "generate gui",
        "grapesjs",
    ],
    "sqlalchemy": ["sqlalchemy", "sql alchemy"],
    "sql": ["sql ddl", "sql schema", "generate sql", "sql"],
    "python": ["python classes", "generate python"],
    "java": ["java classes", "generate java"],
    "pydantic": ["pydantic"],
    "jsonschema": ["json schema", "jsonschema"],
    "smartdata": ["smart data", "smartdata"],
    "agent": ["besser agent", "agent generator", "generate agent"],
    "qiskit": ["qiskit", "quantum code", "quantum generator", "quantum circuit code", "ibm quantum"],
}

GENERATOR_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "django": ["project_name", "app_name", "containerization"],
    "backend": [],
    "sql": ["dialect"],
    "sqlalchemy": ["dbms"],
    "jsonschema": ["mode"],
    "smartdata": [],
    "qiskit": ["backend", "shots"],
}

DIALECT_VALUES = ["sqlite", "postgresql", "mysql", "mssql", "mariadb", "oracle"]
MODE_VALUES = ["regular", "smart_data"]
QISKIT_BACKENDS = ["aer_simulator", "fake_backend", "ibm_quantum"]


def _sanitize_identifier(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", (value or "").strip()).strip("_").lower()
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"p_{cleaned}"
    return cleaned


def detect_generator_type(message: str) -> Optional[str]:
    lower = (message or "").lower()
    for generator_type, keywords in GENERATOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower:
                return generator_type
    return None


def _extract_project_name_from_context(request: AssistantRequest) -> str:
    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict) and isinstance(snapshot.get("name"), str):
        return _sanitize_identifier(snapshot["name"], "besser_project")
    return "besser_project"


def _extract_app_name_from_context(request: AssistantRequest) -> str:
    snapshot = request.context.project_snapshot
    if not isinstance(snapshot, dict):
        return "core_app"

    diagrams = snapshot.get("diagrams")
    active_type = request.context.active_diagram_type
    if isinstance(diagrams, dict):
        active = diagrams.get(active_type)
        if isinstance(active, dict) and isinstance(active.get("title"), str):
            return _sanitize_identifier(active["title"], "core_app")
    return "core_app"


def parse_inline_generator_config(
    generator_type: str,
    message: str,
    request: AssistantRequest,
    existing_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = dict(existing_config or {})
    lower = (message or "").lower()

    if generator_type == "django":
        match_project = re.search(r"(?:project[_\s]?name|project)\s*[:=]\s*([a-zA-Z0-9_\-]+)", lower)
        if match_project:
            config["project_name"] = _sanitize_identifier(match_project.group(1), "besser_project")

        match_app = re.search(r"(?:app[_\s]?name|app)\s*[:=]\s*([a-zA-Z0-9_\-]+)", lower)
        if match_app:
            config["app_name"] = _sanitize_identifier(match_app.group(1), "core_app")

        if "containerization" not in config:
            if "docker" in lower or "container" in lower:
                config["containerization"] = True
            elif "no docker" in lower or "without docker" in lower:
                config["containerization"] = False

    elif generator_type == "sql":
        for dialect in DIALECT_VALUES:
            if dialect in lower:
                config["dialect"] = dialect
                break

    elif generator_type == "sqlalchemy":
        for dbms in DIALECT_VALUES:
            if dbms in lower:
                config["dbms"] = dbms
                break

    elif generator_type == "jsonschema":
        if "smart" in lower:
            config["mode"] = "smart_data"
        elif "regular" in lower:
            config["mode"] = "regular"

    elif generator_type == "backend":
        # Backend generator: optional framework preference
        for fw in ["fastapi", "flask", "django"]:
            if fw in lower:
                config["framework"] = fw
                break

    elif generator_type == "smartdata":
        # SmartData generator: optional output format
        if "json" in lower:
            config["output_format"] = "json"
        elif "rdf" in lower:
            config["output_format"] = "rdf"

    elif generator_type == "qiskit":
        for backend in QISKIT_BACKENDS:
            if backend in lower:
                config["backend"] = backend
                break
        shots_match = re.search(r"shots?\s*[:=]?\s*(\d+)", lower)
        if shots_match:
            config["shots"] = int(shots_match.group(1))

    return config


def _required_missing(generator_type: str, config: Dict[str, Any]) -> List[str]:
    required_fields = GENERATOR_REQUIRED_FIELDS.get(generator_type, [])
    return [field for field in required_fields if field not in config or config[field] in (None, "", [])]


def _build_config_prompt(generator_type: str, missing_fields: List[str]) -> str:
    if generator_type == "django":
        return (
            "To generate Django code I need: `project_name`, `app_name`, and `containerization`.\n"
            "Example: `project_name=my_project app_name=core_app containerization=true`."
        )
    if generator_type == "sql":
        return f"Choose SQL dialect: {', '.join(DIALECT_VALUES)}."
    if generator_type == "sqlalchemy":
        return f"Choose SQLAlchemy DBMS: {', '.join(DIALECT_VALUES)}."
    if generator_type == "jsonschema":
        return f"Choose JSON Schema mode: {', '.join(MODE_VALUES)}."
    if generator_type == "backend":
        return "Choose a backend framework: fastapi, flask, or django (default: django)."
    if generator_type == "smartdata":
        return "Choose SmartData output format: json or rdf (default: json)."
    if generator_type == "qiskit":
        return (
            "Provide qiskit backend and shots.\n"
            f"Backends: {', '.join(QISKIT_BACKENDS)}. Example: `backend=aer_simulator shots=1024`."
        )
    return f"I still need: {', '.join(missing_fields)}."


def _get_pending_state(session: Session) -> Tuple[Optional[str], Dict[str, Any]]:
    pending_generator = session.get("pending_generator_type")
    pending_config = session.get("pending_generator_config") or {}
    return pending_generator, pending_config if isinstance(pending_config, dict) else {}


def _set_pending_state(session: Session, generator_type: str, config: Dict[str, Any]) -> None:
    session.set("pending_generator_type", generator_type)
    session.set("pending_generator_config", config)


def _clear_pending_state(session: Session) -> None:
    """Clear pending generation state without triggering noisy missing-key errors."""
    try:
        session_data = session.get_dictionary()
    except Exception:
        session_data = {}

    for key in ("pending_generator_type", "pending_generator_config"):
        if isinstance(session_data, dict) and key in session_data:
            session.delete(key)


def _looks_like_mixed_modeling_and_generation(message: str) -> bool:
    lower = (message or "").lower()
    if not detect_generator_type(lower):
        return False

    modeling_keywords = [
        "class diagram",
        "object diagram",
        "state machine",
        "state diagram",
        "agent diagram",
        "gui diagram",
        "quantum circuit",
        "create class",
        "create an agent",
        "create state",
        "create model",
        "structural model",
        "model a",
        "design a",
    ]
    has_modeling_language = any(token in lower for token in modeling_keywords)
    has_multi_step_connector = any(token in lower for token in [" and ", " then ", " also ", " after ", " after that ", ";"])
    return has_modeling_language and has_multi_step_connector


def should_route_to_generation(session: Session, request: AssistantRequest) -> bool:
    if request.action == "frontend_event":
        return True
    pending_generator, _ = _get_pending_state(session)
    if pending_generator:
        return True
    if _looks_like_mixed_modeling_and_generation(request.message):
        return False
    return detect_generator_type(request.message) is not None


def _normalize_defaults(generator_type: str, request: AssistantRequest, config: Dict[str, Any]) -> Dict[str, Any]:
    if generator_type == "django":
        config.setdefault("project_name", _extract_project_name_from_context(request))
        app_name = _extract_app_name_from_context(request)
        if config.get("project_name") == app_name:
            app_name = f"{app_name}_app"
        config.setdefault("app_name", app_name)
        config.setdefault("containerization", False)
    elif generator_type == "sql":
        config.setdefault("dialect", "sqlite")
    elif generator_type == "sqlalchemy":
        config.setdefault("dbms", "sqlite")
    elif generator_type == "jsonschema":
        config.setdefault("mode", "regular")
    elif generator_type == "backend":
        config.setdefault("framework", "django")
    elif generator_type == "smartdata":
        config.setdefault("output_format", "json")
    elif generator_type == "qiskit":
        config.setdefault("backend", "aer_simulator")
        config.setdefault("shots", 1024)
    return config


def _handle_frontend_event(request: AssistantRequest) -> Dict[str, Any]:
    event_type = request.raw_payload.get("eventType")
    if event_type == "generator_result":
        ok = bool(request.raw_payload.get("ok"))
        message = request.raw_payload.get("message")
        metadata = request.raw_payload.get("metadata")
        result_message = message if isinstance(message, str) and message.strip() else (
            "Generation completed successfully." if ok else "Generation failed."
        )
        if isinstance(metadata, dict) and metadata.get("filename"):
            result_message = f"{result_message} File: {metadata['filename']}"
        return {"action": "assistant_message", "message": result_message}
    return {
        "action": "assistant_message",
        "message": "Received frontend event update.",
    }


def handle_generation_request(session: Session, request: AssistantRequest) -> Dict[str, Any]:
    if request.action == "frontend_event":
        return _handle_frontend_event(request)

    pending_generator, pending_config = _get_pending_state(session)
    detected_generator = detect_generator_type(request.message)
    generator_type = detected_generator or pending_generator

    if not generator_type:
        # Check if the user actually wants a diagram (GUI/frontend), not code
        _gui_hints = [
            "gui", "frontend", "no-code", "nocode", "grapesjs",
            "diagram", "ui diagram", "gui diagram", "create the gui",
        ]
        _lower_msg = (request.message or "").lower()
        if any(hint in _lower_msg for hint in _gui_hints):
            return {
                "action": "assistant_message",
                "message": (
                    "It sounds like you want to create a GUI diagram rather "
                    "than generate source code. Try saying something like: "
                    '**"create a GUI for the shoe store"** or '
                    '**"create the frontend diagram"**.'
                ),
            }
        return {
            "action": "assistant_message",
            "message": (
                "Tell me what to generate. Supported generators: django, backend, web_app, sql, "
                "sqlalchemy, python, java, pydantic, jsonschema, smartdata, agent, qiskit."
            ),
        }

    config = parse_inline_generator_config(
        generator_type=generator_type,
        message=request.message,
        request=request,
        existing_config=pending_config,
    )
    config = _normalize_defaults(generator_type, request, config)

    missing_fields = _required_missing(generator_type, config)
    if missing_fields:
        _set_pending_state(session, generator_type, config)
        return {
            "action": "assistant_message",
            "message": _build_config_prompt(generator_type, missing_fields),
        }

    _clear_pending_state(session)
    return {
        "action": "trigger_generator",
        "generatorType": generator_type,
        "config": config,
        "message": f"Triggering {generator_type} generation.",
    }
