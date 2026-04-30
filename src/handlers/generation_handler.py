import logging
import re
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

from baf.core.session import Session

from handlers.smart_generation_handler import (
    build_trigger_smart_generator_payload,
    classify_generation_request,
)
from protocol.types import AssistantRequest
from session_keys import (
    CONFIG_PROMPT_ATTEMPTS,
    PENDING_GENERATOR_CONFIG,
    PENDING_GENERATOR_TYPE,
    PENDING_SMART_GEN_INSTRUCTIONS,
    PENDING_SMART_GEN_PROVIDER,
    SKIP_MISMATCH_CHECK_ONCE,
    UNIFIED_CLASSIFICATION,
)
from unified_classifier import (
    UnifiedClassification,
    classify_message as _unified_classify_message,
)

try:
    from llm.provider import get_provider as _get_llm_provider
except ImportError:  # pragma: no cover — keeps the module importable in
    # test environments where the BAF stack isn't set up. Non-ImportError
    # failures (real bugs in llm.provider) are NOT swallowed.
    _get_llm_provider = None  # type: ignore[assignment]


def _classification_to_legacy(cls_obj: UnifiedClassification):
    """Adapt a :class:`UnifiedClassification` to the legacy
    :class:`GenerationClassification` API the dispatch code below
    expects. Returns an object with the same attributes
    ``route / generator_type / refined_instructions / provider / reason``.
    """
    from handlers.smart_generation_handler import GenerationClassification
    return GenerationClassification(
        route=cls_obj.generation_route or "deterministic",
        generator_type=cls_obj.generator_type,
        refined_instructions=cls_obj.refined_instructions,
        provider=cls_obj.provider,
        reason=cls_obj.reason,
    )


def _read_unified_mismatch_info(session: Session) -> Tuple[bool, Optional[str]]:
    """Read domain_mismatch / suggested_new_domain from the cached
    unified classification. Returns ``(False, None)`` when the cache is
    empty or the fields are not populated. Never raises.
    """
    cached: Optional[UnifiedClassification] = session.get(UNIFIED_CLASSIFICATION)
    if cached is None:
        return False, None
    is_mismatch = bool(getattr(cached, "domain_mismatch", False))
    suggested = getattr(cached, "suggested_new_domain", None)
    return is_mismatch, suggested


def _build_mismatch_confirmation(session: Session, classification, suggested: str) -> Dict[str, Any]:
    """Stash the smart-gen instructions and ask the user how to proceed.

    Triggered when the user's request describes a different domain than
    their existing class diagram. The user picks one of three quick
    actions and the agent reroutes accordingly:

    * "Update model + generate" → workflow_body picks up the stashed
      instructions and runs smart-gen after the new model is built.
    * "Generate anyway"         → ``SKIP_MISMATCH_CHECK_ONCE`` is set so
      the next pass through this handler skips this guard.
    * "Cancel"                  → clears all stashed state and stops.
    """
    refined = (classification.refined_instructions or "").strip()
    provider = classification.provider or "anthropic"
    session.set(PENDING_SMART_GEN_INSTRUCTIONS, refined)
    session.set(PENDING_SMART_GEN_PROVIDER, provider)

    return {
        "action": "assistant_message",
        "message": (
            f"Your existing class diagram doesn't match **{suggested}**. "
            f"Pick one:\n\n"
            f"• **Update model + generate** — I'll redesign the class "
            f"diagram for {suggested} first, then run the Vibe-Driven "
            f"Generator. Your current classes will be replaced.\n"
            f"• **Generate anyway** — keep your current model; the "
            f"generator will produce {suggested} code, but your diagram "
            f"won't match the generated code.\n"
            f"• **Cancel** — do nothing; you can edit the model yourself "
            f"first."
        ),
        "suggestedActions": [
            {
                "label": "Update model + generate",
                "prompt": f"create a complete {suggested} system and generate the code",
            },
            {
                "label": "Generate anyway",
                "prompt": "generate anyway with my current model",
            },
            {
                "label": "Cancel",
                "prompt": "cancel the generation",
            },
        ],
    }


def _get_classification_from_cache_or_classify(session, request):
    """Read the unified classifier's cached result, or classify on the fly.

    Prefers the per-message cache (populated by the priority-0 hook in
    ``state_bodies.add_unified_transitions``). Falls back to a direct
    LLM call only when the cache is empty — which happens in tests and
    in code paths that bypass the state-machine transitions.
    """
    cached: Optional[UnifiedClassification] = session.get(UNIFIED_CLASSIFICATION)
    if cached is not None and cached.intent == "generation_intent":
        return _classification_to_legacy(cached)
    # Fallback: call the legacy sub-router (one LLM call). Only reached
    # when the unified hook didn't run — typically tests.
    llm_provider = _get_llm_provider() if _get_llm_provider else None
    return classify_generation_request(request, llm_provider)

logger = logging.getLogger(__name__)

# Sentinel value for pending_generator when the user has been shown the
# generator selection menu and we're waiting for their choice.
_AWAITING_SELECTION = "_awaiting_selection"

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
    "export": [
        "export project", "export the project", "export my project",
        "export to json", "export into json", "export as json", "export json",
        "export to buml", "export into buml", "export as buml", "export buml",
        "export model", "export the model", "export my model",
        "download project", "download the project", "download my project",
        "save as json", "save as buml", "save project",
        "export diagram", "export the diagram",
    ],
    "deploy": [
        "deploy to render", "deploy on render", "deploy app", "deploy the app",
        "deploy application", "deploy the application", "deploy my app",
        "deploy to cloud", "deploy project", "deploy the project",
        "deploy my project", "render deploy", "publish app", "publish the app",
        "publish to render", "publish my app",
    ],
}

GENERATOR_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "django": ["project_name", "app_name", "containerization"],
    "backend": [],
    "sql": ["dialect"],
    "sqlalchemy": ["dbms"],
    "jsonschema": ["mode"],
    "smartdata": [],
    "qiskit": ["backend", "shots"],
    "export": ["format"],
    "deploy": [],
}

EXPORT_FORMATS = ["json", "buml"]

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


# Regex fallback patterns for natural phrasing that keyword lists may miss.
# These are tried only when no exact keyword matches.
_FUZZY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("export", re.compile(
        r"\b(?:export|download|save)\b.*\b(?:json|buml|project|model|diagram)\b", re.I)),
    ("deploy", re.compile(
        r"\b(?:deploy|publish)\b.*\b(?:render|cloud|app|application|project)\b", re.I)),
]


def detect_generator_type(message: str) -> Optional[str]:
    """Detect a code-generator keyword in *message*.

    This is a **pure detection** function — it returns the first matching
    generator type without judging whether the overall request is really a
    code-generation request.  Higher-level callers (``should_route_to_generation``,
    ``handle_generation_request``) apply contextual guards such as
    ``_is_modeling_request`` and ``_is_diagram_creation_request``.
    """
    lower = (message or "").lower()

    # 1. Exact keyword matching (fast path) — use word-boundary-aware check
    #    for short/ambiguous keywords to avoid substring false positives
    #    (e.g. "sql" matching inside "sqlalchemy").
    _BOUNDARY_KEYWORDS = {"sql", "backend"}
    for generator_type, keywords in GENERATOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword in _BOUNDARY_KEYWORDS:
                # Word-boundary match to avoid substring collisions
                if re.search(r'\b' + re.escape(keyword) + r'\b', lower):
                    return generator_type
            else:
                if keyword in lower:
                    return generator_type
    # 2. Regex fallback for flexible phrasing
    for generator_type, pattern in _FUZZY_PATTERNS:
        if pattern.search(lower):
            return generator_type
    return None


# Diagram-type tokens used to detect "generate a <diagram>" requests that
# should be treated as modeling (creation), not code generation.
_DIAGRAM_TYPE_TOKENS = [
    "class diagram", "object diagram", "state machine", "state diagram",
    "agent diagram", "gui diagram", "quantum circuit", "quantum diagram",
    "structural diagram", "domain model", "structural model",
]

# Pre-compiled patterns for _is_diagram_creation_request (avoid recompiling).
_CREATION_VERB_START = re.compile(
    r'^(?:please\s+|can you\s+|could you\s+|i want to\s+|i\'d like to\s+)?'
    r'(?:generate|create|build|design|make|model|draft|develop)\b'
)
_CREATION_VERB_ANYWHERE = re.compile(
    r'\b(?:generate|create|build|design|make|model|draft|develop)\b'
    r'.{0,30}'  # up to 30 chars between verb and diagram token
    r'\b(?:class diagram|object diagram|state (?:machine|diagram)|agent diagram'
    r'|gui diagram|quantum (?:circuit|diagram)|structural (?:diagram|model)|domain model)\b'
)
_NEED_PATTERN = re.compile(
    r'\b(?:need|want|give me|show me|produce|draw)\b.*\b(?:diagram|model|machine|circuit)\b'
)


def _is_diagram_creation_request(lower: str) -> bool:
    """Return True when the message asks to generate/create a *diagram* rather
    than generate source code from an existing model.

    Examples that should return True:
      - "generate a class diagram"
      - "generate the state machine for an order"
      - "i'd like you to generate a class diagram for a library system"
      - "can we build a state machine"

    Examples that should return False:
      - "generate django"
      - "generate python code"
      - "generate sql from my model"
    """
    # Must mention a diagram type
    if not any(token in lower for token in _DIAGRAM_TYPE_TOKENS):
        return False

    # Check 1: message starts with a creation verb (possibly with filler).
    if _CREATION_VERB_START.search(lower):
        return True

    # Check 2: creation verb appears anywhere NEAR a diagram type token.
    # Catches "I'd like you to generate a class diagram" and
    # "can we create a state machine for the order process".
    if _CREATION_VERB_ANYWHERE.search(lower):
        return True

    # Check 3: "I need a class diagram", "give me a state diagram", etc.
    if _NEED_PATTERN.search(lower):
        return True

    return False


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

    elif generator_type == "export":
        for fmt in EXPORT_FORMATS:
            if fmt in lower:
                config["format"] = fmt
                break

    # Deploy: no inline config needed — the frontend dialog handles everything.

    return config


def _required_missing(generator_type: str, config: Dict[str, Any]) -> List[str]:
    required_fields = GENERATOR_REQUIRED_FIELDS.get(generator_type, [])
    return [field for field in required_fields if field not in config or config[field] in (None, "", [])]


def _validate_config(generator_type: str, config: Dict[str, Any]) -> List[str]:
    """Return list of validation error messages for invalid config values."""
    errors = []
    if generator_type in ("sql",) and "dialect" in config:
        if config["dialect"] not in DIALECT_VALUES:
            errors.append(f"Invalid SQL dialect '{config['dialect']}'. Valid: {', '.join(DIALECT_VALUES)}")
    if generator_type in ("sqlalchemy",) and "dbms" in config:
        if config["dbms"] not in DIALECT_VALUES:
            errors.append(f"Invalid DBMS '{config['dbms']}'. Valid: {', '.join(DIALECT_VALUES)}")
    if generator_type == "jsonschema" and "mode" in config:
        if config["mode"] not in MODE_VALUES:
            errors.append(f"Invalid mode '{config['mode']}'. Valid: {', '.join(MODE_VALUES)}")
    if generator_type == "qiskit" and "backend" in config:
        if config["backend"] not in QISKIT_BACKENDS:
            errors.append(f"Invalid backend '{config['backend']}'. Valid: {', '.join(QISKIT_BACKENDS)}")
    return errors


def _build_config_prompt(
    generator_type: str,
    missing_fields: List[str],
    request: Optional[AssistantRequest] = None,
) -> str:
    # Build suggested defaults from the project context
    suggested_project = "my_project"
    suggested_app = "core_app"
    if request is not None:
        suggested_project = _extract_project_name_from_context(request)
        suggested_app = _extract_app_name_from_context(request)

    if generator_type == "django":
        return (
            "To generate your **Django** project, I need a few details:\n\n"
            f"- **Project name** — the top-level Django project (suggested: `{suggested_project}`)\n"
            f"- **App name** — the Django app inside it (suggested: `{suggested_app}`)\n"
            "- **Containerization** — include Docker setup? (`true` / `false`)\n\n"
            f"You can provide them like: `project_name={suggested_project} app_name={suggested_app} containerization=true`\n\n"
            "Or just say **use defaults** to accept the suggested values."
        )
    if generator_type == "sql":
        return (
            "Which **SQL dialect** should I target?\n\n"
            f"Options: {', '.join(f'`{d}`' for d in DIALECT_VALUES)}"
        )
    if generator_type == "sqlalchemy":
        return (
            "Which **database management system** should the SQLAlchemy code target?\n\n"
            f"Options: {', '.join(f'`{d}`' for d in DIALECT_VALUES)}"
        )
    if generator_type == "jsonschema":
        return (
            "Which **JSON Schema mode** would you like?\n\n"
            f"Options: {', '.join(f'`{m}`' for m in MODE_VALUES)}"
        )
    if generator_type == "backend":
        return (
            "Which **backend framework** should I use?\n\n"
            "Options: `fastapi`, `flask`, or `django`"
        )
    if generator_type == "smartdata":
        return (
            "Which **output format** for SmartData?\n\n"
            "Options: `json` or `rdf`"
        )
    if generator_type == "qiskit":
        return (
            "I need a couple of settings for the **Qiskit** generator:\n\n"
            f"- **Backend**: {', '.join(f'`{b}`' for b in QISKIT_BACKENDS)}\n"
            "- **Shots**: number of measurement repetitions (e.g. `1024`)\n\n"
            "Example: `backend=aer_simulator shots=1024`"
        )
    if generator_type == "export":
        return (
            "Which **format** would you like to export your project in?\n\n"
            "- `json` \u2014 full project snapshot as a JSON file\n"
            "- `buml` \u2014 B-UML textual notation\n\n"
            "Just type `json` or `buml`."
        )
    return f"I still need these settings: {', '.join(f'`{f}`' for f in missing_fields)}."


def _get_pending_state(session: Session) -> Tuple[Optional[str], Dict[str, Any]]:
    pending_generator = session.get(PENDING_GENERATOR_TYPE)
    pending_config = session.get(PENDING_GENERATOR_CONFIG) or {}
    return pending_generator, pending_config if isinstance(pending_config, dict) else {}


def _set_pending_state(session: Session, generator_type: str, config: Dict[str, Any]) -> None:
    session.set(PENDING_GENERATOR_TYPE, generator_type)
    session.set(PENDING_GENERATOR_CONFIG, config)


def _clear_pending_state(session: Session) -> None:
    """Clear pending generation state without triggering noisy missing-key errors."""
    try:
        session_data = session.get_dictionary()
    except Exception as exc:
        logger.debug(f"Session dictionary access failed (best-effort): {exc}")
        session_data = {}

    for key in (PENDING_GENERATOR_TYPE, PENDING_GENERATOR_CONFIG):
        if isinstance(session_data, dict) and key in session_data:
            session.delete(key)


def _clear_pending_smart_gen(session: Session) -> None:
    """Clear stashed smart-gen instructions and the skip-mismatch flag.

    Called after the user resolves a mismatch confirmation (via Generate
    Anyway, Cancel, or after the workflow_body completes a chained run)
    so a stale stash doesn't leak into a future unrelated request.
    """
    try:
        session_data = session.get_dictionary()
    except Exception as exc:
        logger.debug(f"Session dictionary access failed (best-effort): {exc}")
        session_data = {}

    for key in (
        PENDING_SMART_GEN_INSTRUCTIONS,
        PENDING_SMART_GEN_PROVIDER,
        SKIP_MISMATCH_CHECK_ONCE,
    ):
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


# Pattern: "create/build/design … for <anything>" — the "for" signals a domain
# context, making this a modeling request regardless of generator keywords.
_MODELING_VERB_FOR_PATTERN = re.compile(
    r'\b(?:create|build|design|model|make|develop|architect|plan|draft)\b'
    r'.{1,60}'       # up to 60 chars between verb and "for"
    r'\bfor\b'
    r'.{3,}',        # at least 3 chars after "for" (a real domain phrase)
    re.I,
)

# Pattern: "create/build/design a <noun-phrase>" where the noun phrase is NOT
# a bare generator keyword.  Catches "build me a web app", "create a booking
# platform", etc.  The negative lookahead excludes bare generator targets like
# "generate django" or "generate sql".
_MODELING_VERB_OBJECT_PATTERN = re.compile(
    r'\b(?:create|build|design|model|make|develop|architect|plan|draft)\b'
    r'\s+(?:me\s+|us\s+)?'               # optional "me"/"us"
    r'(?:a|an|the|my|our|this)?\s*'       # optional article
    r'(?!django\b|sql\b|python\b|java\b|pydantic\b|qiskit\b|backend\b)'  # NOT a bare generator
    r'(\w+(?:\s+\w+){1,4})',              # 2-5 word noun phrase
    re.I,
)

# Explicit generation phrases that override modeling detection.
_EXPLICIT_GENERATION_PHRASES = [
    "generate code", "generate the code", "generate source",
    "run generator", "trigger generator", "code generation",
    "source code", "export", "deploy",
]


def _is_modeling_request(message: str) -> bool:
    """Return True when the message is primarily asking to model/create/design
    a system, NOT to generate source code from an existing model.

    Uses pattern-based detection instead of a hardcoded domain list, so
    "create a web app for insurance claims" works just as well as
    "create a web app for hotel booking".
    """
    lower = (message or "").lower()

    # Fast path: diagram creation requests are always modeling.
    if _is_diagram_creation_request(lower):
        return True

    # Veto: explicit generation phrases always win.
    if any(g in lower for g in _EXPLICIT_GENERATION_PHRASES):
        return False

    # Pattern 1: "verb … for <domain>" — strongest signal.
    # "create a web app for hotel booking" ✓
    # "build a platform for managing inventory" ✓
    # "design a system for insurance claims" ✓
    if _MODELING_VERB_FOR_PATTERN.search(lower):
        return True

    # Pattern 2: "verb [article] <noun-phrase>" with 2+ words after article.
    # "create a booking platform" ✓  "build me a reservation system" ✓
    # "generate django" ✗ (bare generator keyword, excluded by lookahead)
    m = _MODELING_VERB_OBJECT_PATTERN.search(lower)
    if m:
        noun_phrase = m.group(1).strip()
        # The noun phrase must contain a domain/system word, not just a
        # generator keyword.  A noun phrase of 3+ words is strong enough
        # on its own ("hotel booking system").  For 2-word phrases, check
        # that at least one word isn't a pure generator keyword.
        words = noun_phrase.split()
        _GENERATOR_ONLY_WORDS = {
            "django", "sql", "python", "java", "pydantic", "qiskit",
            "sqlalchemy", "backend", "jsonschema", "smartdata", "agent",
        }
        if len(words) >= 3:
            return True
        if len(words) == 2 and not all(w in _GENERATOR_ONLY_WORDS for w in words):
            return True

    return False


def should_route_to_generation(session: Session, request: AssistantRequest) -> bool:
    """State-transition gatekeeper for ``generation_state``.

    Runs on every ``ReceiveJSONEvent``. We deliberately keep it cheap
    (NO LLM call, NO text heuristics) and defer all intent
    classification to the two places that already do that job:

      1. BAF's intent classifier (one ``gpt-4.1-mini`` call per message)
         decides whether this is a generation request. Its
         ``json_intent_matches`` transition routes to
         ``generation_state`` on its own.
      2. Inside ``generation_state``, ``handle_generation_request`` calls
         :func:`classify_generation_request` — ONE structured-output
         LLM call — for the smart-vs-deterministic sub-routing.

    The only jobs of this function are the two non-intent signals BAF's
    classifier can't see:

      * ``frontend_event`` callbacks (``generator_result``, etc.) —
        ``raw_payload`` events, not conversational messages.
      * ``pending_generator`` state — the user is mid-flow answering a
        multi-turn config prompt and the next message is config input,
        not a new intent.

    Older versions of this function ran text-content heuristics
    (``detect_generator_type``, ``_is_modeling_request``, phrase lists,
    etc.) as a safety net for BAF misclassifications. That net is
    removed — BAF's ``generation_intent`` description was strengthened
    to cover non-BESSER stacks (rails, rust, kotlin, …) so it routes
    correctly on its own.
    """
    if request.action == "frontend_event":
        return True
    pending_generator, _ = _get_pending_state(session)
    return bool(pending_generator)


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
    """Route a generation-state request to smart-gen, deterministic, or menu.

    BAF's intent classifier has already decided this is a generation
    request. Our job here is the SUB-routing: smart generator (for any
    custom stack / language BESSER doesn't have built-in) vs one of
    BESSER's deterministic generators (django / pydantic / sql / …) vs
    redirect-to-modeling (if BAF misclassified).

    We make ONE LLM call via :func:`classify_generation_request` that
    returns route + generator_type + refined_instructions all at once —
    replacing what used to be three calls + a pile of keyword
    heuristics.
    """
    if request.action == "frontend_event":
        return _handle_frontend_event(request)

    # Mismatch-confirmation shortcuts: when the user clicks one of the
    # quick-action buttons emitted by ``_build_mismatch_confirmation``,
    # short-circuit the classifier and act on the stashed smart-gen
    # payload directly. We match on phrase substrings (case-insensitive)
    # rather than exact equality so paraphrased messages still work.
    msg_lower = (request.message or "").strip().lower()
    has_pending_smart_gen = bool(session.get(PENDING_SMART_GEN_INSTRUCTIONS))

    if has_pending_smart_gen and "cancel the generation" in msg_lower:
        _clear_pending_smart_gen(session)
        _clear_pending_state(session)
        return {
            "action": "assistant_message",
            "message": "Cancelled. Your model is unchanged.",
        }

    if has_pending_smart_gen and "generate anyway" in msg_lower:
        # Skip the mismatch guard and re-run the smart-gen handoff with
        # the previously-stashed instructions / provider. The user has
        # explicitly accepted the model/code divergence.
        from handlers.smart_generation_handler import GenerationClassification
        stashed_instructions = session.get(PENDING_SMART_GEN_INSTRUCTIONS) or ""
        stashed_provider = session.get(PENDING_SMART_GEN_PROVIDER) or "anthropic"
        _clear_pending_smart_gen(session)
        if stashed_instructions.strip():
            return build_trigger_smart_generator_payload(
                GenerationClassification(
                    route="smart",
                    refined_instructions=stashed_instructions,
                    provider=stashed_provider,
                    reason="user chose to generate without updating the model",
                ),
                reason_prefix="generating with current model",
            )
        # Fall through to normal classification if the stash was empty.

    pending_generator, pending_config = _get_pending_state(session)

    if pending_generator and pending_generator != _AWAITING_SELECTION:
        # Multi-turn continuation: user is mid-flow answering a config
        # prompt (e.g. "what's your Django project name?"). Use the
        # already-picked generator and fall through to the config
        # parsing / dispatch path below without calling the LLM
        # classifier again.
        generator_type: Optional[str] = pending_generator
    else:
        # First-time-through: read the unified classifier's verdict
        # from the per-message cache (populated by state_bodies'
        # priority-0 hook). Falls back to a fresh call only when the
        # cache is empty — typically in tests that bypass the state
        # machine. Net result in production: ZERO extra LLM calls
        # here — the classification was already done before this
        # state body ran.
        classification = _get_classification_from_cache_or_classify(session, request)
        logger.info(
            "generation sub-route: route=%s generator_type=%s reason=%s",
            classification.route, classification.generator_type, classification.reason,
        )

        if classification.route == "smart":
            _clear_pending_state(session)
            session.set(CONFIG_PROMPT_ATTEMPTS, 0)

            # Domain-mismatch guard: when the user's request describes a
            # different domain than their existing class diagram, refuse
            # to silently rewrite generated code that won't match their
            # model. Surface the choice to the user via quick actions.
            # Honors a one-shot ``SKIP_MISMATCH_CHECK_ONCE`` flag so the
            # "Generate anyway" path doesn't loop on this question.
            skip_mismatch = bool(session.get(SKIP_MISMATCH_CHECK_ONCE))
            if skip_mismatch:
                session.set(SKIP_MISMATCH_CHECK_ONCE, False)
            else:
                is_mismatch, suggested = _read_unified_mismatch_info(session)
                if is_mismatch and suggested:
                    return _build_mismatch_confirmation(session, classification, suggested)

            return build_trigger_smart_generator_payload(classification)

        if classification.route == "modeling":
            _clear_pending_state(session)
            return {
                "action": "assistant_message",
                "message": (
                    "It looks like you want to **create a diagram or design a system**, "
                    "not generate code from an existing model. Try rephrasing as: "
                    '**"create a class diagram for …"** or **"design a system for …"** '
                    "so I can build the model first."
                ),
            }

        if classification.route == "other":
            _clear_pending_state(session)
            return {
                "action": "assistant_message",
                "message": (
                    "I didn't catch a clear code-generation request. If you want code, "
                    'try something like **"generate django"** for a built-in generator '
                    'or **"build me a rails api"** / **"generate code in rust"** for a '
                    "custom stack. If you want a diagram, try "
                    '**"create a class diagram for a library"**.'
                ),
            }

        # route == "deterministic" — fall through to the config-parse
        # / dispatch path with the classifier's picked generator type.
        # If ``_AWAITING_SELECTION`` was set, clear it; the classifier
        # has effectively answered which generator the user picked.
        if pending_generator == _AWAITING_SELECTION:
            _clear_pending_state(session)
        generator_type = classification.generator_type

    if not generator_type:
        _lower_msg = (request.message or "").lower()

        # If the message is really about creating a diagram (class diagram,
        # state diagram, etc.) rather than generating code, redirect the user
        # back to the modeling intent instead of showing the generator menu.
        _non_gui_diagram_tokens = [
            "class diagram", "object diagram", "state machine",
            "state diagram", "structural diagram", "domain model",
            "quantum circuit", "quantum diagram", "agent diagram",
        ]
        if any(token in _lower_msg for token in _non_gui_diagram_tokens):
            _clear_pending_state(session)
            return {
                "action": "assistant_message",
                "message": (
                    "It sounds like you want to **create a diagram**, not "
                    "generate source code. Try rephrasing as: "
                    '**"create a class diagram for a library system"** or '
                    '**"design a state machine for order processing"**.'
                ),
            }

        # Check if the user actually wants a GUI/frontend diagram, not code
        _gui_hints = [
            "gui", "frontend", "no-code", "nocode", "grapesjs",
            "ui diagram", "gui diagram", "create the gui",
        ]
        if any(hint in _lower_msg for hint in _gui_hints):
            _clear_pending_state(session)
            return {
                "action": "assistant_message",
                "message": (
                    "It sounds like you want to create a GUI diagram rather "
                    "than generate source code. Try saying something like: "
                    '**"create a GUI for the shoe store"** or '
                    '**"create the frontend diagram"**.'
                ),
            }
        _set_pending_state(session, _AWAITING_SELECTION, {})
        return {
            "action": "assistant_message",
            "message": (
                "What would you like me to generate? Here are the available options:\n\n"
                "**Web & Backend**: `django`, `backend`, `web_app`\n"
                "**Database**: `sql`, `sqlalchemy`\n"
                "**Code**: `python`, `java`, `pydantic`\n"
                "**Data formats**: `jsonschema`, `smartdata`\n"
                "**Other**: `agent`, `qiskit`\n\n"
                "**Export**: `export json` or `export buml`\n"
                "**Deploy**: `deploy to render`\n\n"
                "Just say something like *'generate sqlalchemy'* or *'export to json'*."
            ),
        }

    config = parse_inline_generator_config(
        generator_type=generator_type,
        message=request.message,
        request=request,
        existing_config=pending_config,
    )

    # Validate config enum values early
    config_errors = _validate_config(generator_type, config)
    if config_errors:
        _set_pending_state(session, generator_type, {})
        return {
            "action": "assistant_message",
            "message": "\n".join(config_errors) + "\n\nPlease provide a valid value.",
        }

    # "use defaults" shortcut — accept suggested values immediately
    _lower_msg = (request.message or "").lower()
    if "use default" in _lower_msg or "defaults" == _lower_msg.strip():
        config = _normalize_defaults(generator_type, request, config)

    missing_fields = _required_missing(generator_type, config)
    if missing_fields:
        # Track how many times we've prompted for config to avoid infinite loops
        config_attempts = (session.get(CONFIG_PROMPT_ATTEMPTS) or 0) + 1
        session.set(CONFIG_PROMPT_ATTEMPTS, config_attempts)

        if config_attempts >= 3:
            # After 2 failed attempts, auto-fill with defaults and proceed
            config = _normalize_defaults(generator_type, request, config)
            session.set(CONFIG_PROMPT_ATTEMPTS, 0)
        else:
            _set_pending_state(session, generator_type, config)
            prompt = _build_config_prompt(generator_type, missing_fields, request=request)
            if config_attempts >= 2:
                prompt += "\n\n*Or just say **use defaults** to proceed with suggested values.*"
            return {
                "action": "assistant_message",
                "message": prompt,
            }

    # Only apply defaults AFTER confirming none are required-but-missing,
    # so users are always asked for parameters the generator needs.
    config = _normalize_defaults(generator_type, request, config)

    _clear_pending_state(session)
    session.set(CONFIG_PROMPT_ATTEMPTS, 0)
    # ------------------------------------------------------------------
    # Special actions: export & deploy
    # ------------------------------------------------------------------
    if generator_type == "export":
        fmt = config.get("format", "json")
        return {
            "action": "trigger_export",
            "format": fmt,
            "message": f"Exporting your project as **{fmt.upper()}** \u2014 the download should start shortly.",
        }

    if generator_type == "deploy":
        return {
            "action": "trigger_deploy",
            "platform": "render",
            "config": {},
            "message": (
                "Opening the **Deploy to Render** dialog \u2014 "
                "please connect to GitHub if you haven\u2019t already, "
                "then fill in the repository details and hit **Publish**."
            ),
        }
    # Empty model guard: disabled because the WebSocket context may be stale
    # right after an injection (frontend has the model but sends pre-injection
    # snapshot with the next message). Frontend validates before calling backend.
    # context = getattr(request, 'context', None)
    # active_model = getattr(context, 'active_model', None) if context else None
    # if active_model is None:
    #     snapshot = getattr(context, 'project_snapshot', None) if context else None
    #     if isinstance(snapshot, dict):
    #         diagrams = snapshot.get('diagrams', {})
    #         active_type = getattr(context, 'active_diagram_type', None)
    #         if isinstance(diagrams, dict) and active_type:
    #             diagram_data = diagrams.get(active_type, {})
    #             if isinstance(diagram_data, dict):
    #                 active_model = diagram_data.get('model')
    #
    # if not active_model or (isinstance(active_model, dict) and not active_model.get('elements')):
    #     return {
    #         "action": "assistant_message",
    #         "message": (
    #             f"Your diagram is empty — please create a model first before "
    #             f"generating **{generator_type}** code. Try describing your system "
    #             f"(e.g. *\"create a library management system\"*)."
    #         ),
    #     }

    return {
        "action": "trigger_generator",
        "generatorType": generator_type,
        "config": config,
        "message": f"Starting **{generator_type}** code generation — this may take a moment.",
    }
