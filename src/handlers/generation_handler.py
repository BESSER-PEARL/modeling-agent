import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

from baf.core.session import Session

from handlers.smart_generation_handler import (
    build_trigger_smart_generator_payload,
)
from protocol.types import AssistantRequest
from utilities.model_context import is_diagram_nontrivial
from session_keys import (
    CONFIG_PROMPT_ATTEMPTS,
    LAST_SMART_GEN_AT,
    LAST_SMART_GEN_SUMMARY,
    MISMATCH_REGEN_PENDING,
    PENDING_GENERATOR_CONFIG,
    PENDING_GENERATOR_TYPE,
    PENDING_SMART_GEN_INSTRUCTIONS,
    PENDING_SMART_GEN_PROVIDER,
    PENDING_SMART_GEN_TIMESTAMP,
    SKIP_MISMATCH_CHECK_ONCE,
    UNIFIED_CLASSIFICATION,
)
from reply_copy import OUT_OF_SCOPE_REDIRECT, SPEC_DRIVEN_NAME
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
    """Adapt a :class:`UnifiedClassification` to the
    :class:`GenerationClassification` shape the dispatch code below
    expects (attributes ``route / generator_type / refined_instructions /
    provider / reason``).

    Non-generation verdicts can still reach the generation handler
    (keyword-routed messages, pending flows). Rather than second-guessing
    the unified classifier with another LLM call, honor its verdict:
    modeling intents run the modeling pipeline (create OR modify — see the
    ``route == "modeling"`` branch, which picks the mode from the cached
    intent), everything else gets the 'other' clarify reply.
    """
    from handlers.smart_generation_handler import GenerationClassification
    # generation_route is only MEANINGFUL on generation verdicts (the schema
    # says "REQUIRED when intent='generation_intent'") — but the LLM often
    # fills it as 'other' on non-generation intents too. Trusting that noise
    # sent a mismatch-rebuild ("create a class diagram for a hotel…",
    # intent=create, generation_route='other') to the clarify reply instead
    # of the modeling branch. Ignore the field unless the verdict is a
    # generation one; derive the route from the intent otherwise.
    route = (
        cls_obj.generation_route
        if cls_obj.intent == "generation_intent" else None
    )
    if not route:
        if cls_obj.intent in (
            "create_complete_system_intent", "modify_model_intent",
        ):
            route = "modeling"
        elif cls_obj.intent == "generation_intent":
            route = "deterministic"
        elif cls_obj.intent == "fallback_intent" and (
            cls_obj.reason or ""
        ).startswith("[classifier-error]"):
            # ERROR fallback (LLM down / parse failure — _safe_fallback tags
            # these): 'deterministic' with no generator_type shows the
            # generator MENU, the resilient pre-LLM behavior. A DELIBERATE
            # none-of-the-above verdict falls through to 'other' instead.
            route = "deterministic"
        else:
            route = "other"
    return GenerationClassification(
        route=route,
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


# Stashed smart-gen payloads expire after this long. The confirm handler
# rejects anything older so an abandoned dialog can never trigger a
# BYOK-spending run days later (B-2 stale-stash fix).
_SMART_GEN_STASH_TTL_SECONDS = 30 * 60


# Generator prerequisites are shared with the request planner. Keeping the
# contract here lets both direct generation and multi-step plans validate the
# same diagram requirements without a circular import.
GENERATOR_PREREQUISITES: Dict[str, List[str]] = {
    "web_app": ["ClassDiagram", "GUINoCodeDiagram"],
    "react": ["ClassDiagram", "GUINoCodeDiagram"],
    "flutter": ["ClassDiagram", "GUINoCodeDiagram"],
    "django": ["ClassDiagram"],
    "backend": ["ClassDiagram"],
    "sql": ["ClassDiagram"],
    "sqlalchemy": ["ClassDiagram"],
    "python": ["ClassDiagram"],
    "java": ["ClassDiagram"],
    "pydantic": ["ClassDiagram"],
    "jsonschema": ["ClassDiagram"],
    "smartdata": ["ClassDiagram"],
    "rest_api": ["ClassDiagram"],
    "rdf": ["ClassDiagram"],
    "agent": ["AgentDiagram"],
    "qiskit": ["QuantumCircuitDiagram"],
}


def _smart_gen_stash_is_fresh(timestamp: Any) -> bool:
    """True when the stash timestamp exists and is within the TTL.

    Stashes written before the timestamp key existed return False — they
    are by definition of unknown age and must not fire.
    """
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
        return False
    return (time.time() - timestamp) <= _SMART_GEN_STASH_TTL_SECONDS


def _stash_smart_gen(session: Session, instructions: str, provider: str) -> None:
    """Stash a smart-gen payload with a fresh timestamp (see TTL above)."""
    session.set(PENDING_SMART_GEN_INSTRUCTIONS, instructions)
    session.set(PENDING_SMART_GEN_PROVIDER, provider)
    session.set(PENDING_SMART_GEN_TIMESTAMP, time.time())


_SMART_GEN_CONFIRM_PHRASES = {
    "yes", "yes please", "confirm", "confirm it", "run it", "run it now",
    "please run it", "go ahead", "proceed", "start it", "start the generation",
    "generate anyway", "generate anyway with my current model",
}
# Whole-message phrases that EXIT a pending config-collection flow. Bare "no"
# is deliberately absent: mid-config it is usually a field answer ("Docker?"
# -> "no"); classifier decline verdicts cover the novel opt-out phrasings.
_CONFIG_CANCEL_PHRASES = {
    "cancel", "cancel it", "cancel that", "cancel generation",
    "cancel the generation", "stop", "stop it", "abort", "quit",
    "never mind", "nevermind", "forget it", "don't generate",
    "do not generate",
}
_SMART_GEN_CANCEL_PHRASES = {
    "no", "no thanks", "cancel", "cancel it", "cancel generation",
    "cancel the generation", "stop", "stop it", "abort", "never mind",
    "do not run it", "don't run it",
}


def _smart_gen_confirmation_decision(message: str) -> Optional[str]:
    """Return ``confirm``/``cancel`` only for an unambiguous whole reply."""
    normalized = re.sub(r"[,.!?]+", " ", (message or "").strip().lower())
    normalized = " ".join(normalized.split())
    if normalized in _SMART_GEN_CONFIRM_PHRASES:
        return "confirm"
    if normalized in _SMART_GEN_CANCEL_PHRASES:
        return "cancel"
    return None


def _norm_prompt(message: str) -> str:
    """Normalize a message for exact-prompt matching (mismatch-regen chain)."""
    return " ".join((message or "").strip().lower().split())


def _iter_models_of_type(context: Any, diagram_type: str):
    """Yield active and snapshot models for one diagram type."""
    seen: set[int] = set()
    if getattr(context, "active_diagram_type", None) == diagram_type:
        active_model = getattr(context, "active_model", None)
        if isinstance(active_model, dict):
            seen.add(id(active_model))
            yield active_model

    snapshot = getattr(context, "project_snapshot", None)
    diagrams = snapshot.get("diagrams") if isinstance(snapshot, dict) else None
    target = diagrams.get(diagram_type) if isinstance(diagrams, dict) else None
    entries = (
        target
        if isinstance(target, list)
        else ([target] if target is not None else [])
    )
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model = entry.get("model")
        if not isinstance(model, dict) and (
            "elements" in entry or "pages" in entry or "cols" in entry
        ):
            model = entry
        if isinstance(model, dict) and id(model) not in seen:
            seen.add(id(model))
            yield model


def _project_has_diagram_model(context: Any, diagram_type: str) -> bool:
    if context is None:
        return True
    return any(
        is_diagram_nontrivial(model, diagram_type)
        for model in _iter_models_of_type(context, diagram_type)
    )


def _project_has_any_model(context: Any) -> bool:
    """True if any canonical diagram model has meaningful content."""
    if context is None:
        return True
    diagram_types = {
        dtype
        for required in GENERATOR_PREREQUISITES.values()
        for dtype in required
    }
    snapshot = getattr(context, "project_snapshot", None)
    diagrams = snapshot.get("diagrams") if isinstance(snapshot, dict) else None
    if isinstance(diagrams, dict):
        diagram_types.update(key for key in diagrams if isinstance(key, str))
    active_type = getattr(context, "active_diagram_type", None)
    if isinstance(active_type, str):
        diagram_types.add(active_type)
    return any(_project_has_diagram_model(context, dtype) for dtype in diagram_types)


def _missing_generator_prerequisites(context: Any, generator_type: str) -> List[str]:
    """Return required diagram types that are absent or only seed content."""
    if context is None:
        return []
    return [
        dtype
        for dtype in GENERATOR_PREREQUISITES.get(generator_type, [])
        if not _project_has_diagram_model(context, dtype)
    ]


def _build_smart_gen_confirmation(
    session: Session,
    instructions: str,
    provider: str,
    *,
    reason_prefix: str = "",
) -> Dict[str, Any]:
    """Stash the smart-gen payload and ask for explicit confirmation.

    The Spec-Driven Agent runs on the USER'S OWN API key, so it must
    never start without an explicit confirmation — with a stored key the
    run would otherwise begin silently (B-2). The confirm/cancel phrases
    are handled at the top of :func:`handle_generation_request`.
    """
    refined = (instructions or "").strip()
    provider = provider or "anthropic"
    _stash_smart_gen(session, refined, provider)

    prefix = f"{reason_prefix}\n\n" if reason_prefix else ""

    # The instructions are NOT echoed back to the user: showing the LLM's
    # refined instructions read as fabricated requirements the user never
    # wrote. The run still uses the stashed ``refined`` instructions above.
    # Plain text (no clickable key link): the user can set up their own key
    # from the assistant's key settings when they want a different provider.
    return {
        "action": "assistant_message",
        "message": (
            f"{prefix}BESSER will generate your application from your "
            f"model using its built-in generators. If some of your "
            f"requirements are not supported by these generators, BESSER can "
            f"use an LLM to handle them.\n\n"
            f"BESSER uses Qwen as the default free model. You can also set up "
            f"your own API key to use a different provider or model.\n\n"
            f"Do you want to continue?"
        ),
        # Cancel action removed per product decision — the proposition offers
        # only Run; the user can simply not click it (or type another request)
        # to not proceed.
        "suggestedActions": [
            {
                "label": "Continue",
                "prompt": "generate anyway with my current model",
            },
        ],
    }


def _build_mismatch_confirmation(session: Session, classification, suggested: str) -> Dict[str, Any]:
    """Stash the smart-gen instructions and ask the user how to proceed.

    Triggered when the user's request describes a different domain than
    their existing class diagram. The user picks one of three quick
    actions and the agent reroutes accordingly:

    * "Update model + generate" → the create choke point in
      execution.model_operations picks up the stashed
      instructions and runs smart-gen after the new model is built.
    * "Generate anyway"         → ``SKIP_MISMATCH_CHECK_ONCE`` is set so
      the next pass through this handler skips this guard.
    * "Cancel"                  → clears all stashed state and stops.
    """
    refined = (classification.refined_instructions or "").strip()
    provider = classification.provider or "anthropic"
    rebuild_prompt = f"create a class diagram for {suggested}"
    _stash_smart_gen(session, refined, provider)
    # Arm the one-shot resume with the EXACT rebuild prompt the button sends.
    # The guard below and the create choke point
    # (execution.model_operations.execute_model_operation) only keep the stash
    # / fire the resume when the incoming message equals this prompt — so a
    # DIFFERENT create typed right after a mismatch abandons normally instead
    # of spuriously resuming the old domain's smart-gen.
    session.set(MISMATCH_REGEN_PENDING, rebuild_prompt)

    return {
        "action": "assistant_message",
        "message": (
            f"Your existing class diagram doesn't match **{suggested}**. "
            f"Pick one:\n\n"
            f"• **Update model + generate** — I'll redesign the class "
            f"diagram for {suggested} first, then run the Spec-Driven "
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
                # Route this to a pure CREATE (not the smart/generation route,
                # which would re-run this very mismatch check and loop). The
                # create rebuilds the new domain model; because this exact prompt
                # was stashed in MISMATCH_REGEN_PENDING above, the create choke
                # point recognizes it and resumes the stashed smart-gen right
                # after the rebuild, so "+ generate" actually runs.
                "prompt": rebuild_prompt,
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
    """Adapt the unified classifier's verdict for generation dispatch.

    The per-event cache is trusted only for GENERATION verdicts; anything
    else re-classifies this request's own text (see inline comment).

    The unified classifier is the SINGLE rulebook for generation
    sub-routing. The legacy generation-only classifier (a second prompt
    that had drifted out of sync — e.g. it contradicted the SQL-dialect
    and rest_api/backend deterministic rules) is retired; whatever the
    unified call decided is adapted via ``_classification_to_legacy``.
    """
    cached: Optional[UnifiedClassification] = session.get(UNIFIED_CLASSIFICATION)
    if cached is not None and cached.intent == "generation_intent":
        return _classification_to_legacy(cached)
    # Empty or NON-generation cache: the per-event cache belongs to the
    # ORIGINAL user message, and internally synthesized sub-requests (the
    # planner's Phase-2 "generate django" step of a compound "create X and
    # generate Y" plan) share that event — adapting the original create
    # verdict for them routed the generation back into modeling and RECURSED
    # instead of generating. Classify THIS request's own text directly
    # (uncached; same classifier-tier cost the retired legacy sub-router
    # paid in exactly these situations).
    llm_provider = _get_llm_provider() if _get_llm_provider else None
    return _classification_to_legacy(
        _unified_classify_message(request, llm_provider))

logger = logging.getLogger(__name__)

# Sentinel value for pending_generator when the user has been shown the
# generator selection menu and we're waiting for their choice.
_AWAITING_SELECTION = "_awaiting_selection"

GENERATOR_KEYWORDS: Dict[str, List[str]] = {
    "django": ["django"],
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
    "backend": ["full backend", "backend"],
    "sqlalchemy": ["sqlalchemy", "sql alchemy"],
    "sql": ["database", "relational database", "db schema", "sql ddl", "sql schema", "generate sql", "sql"],
    "python": ["python classes", "generate python"],
    "java": ["java classes", "generate java"],
    "pydantic": ["pydantic"],
    "jsonschema": ["json schema", "jsonschema"],
    "smartdata": ["smart data", "smartdata"],
    "agent": ["besser agent", "agent generator", "generate agent"],
    "qiskit": ["qiskit", "quantum code", "quantum generator", "quantum circuit code", "ibm quantum"],
    "rest_api": ["rest api", "rest_api", "generate rest api"],
    "rdf": ["rdf", "rdf generator", "rdf vocabulary", "generate rdf"],
    "export": [
        "export project", "export the project", "export my project",
        "export to json", "export into json", "export as json", "export json",
        "export to buml", "export into buml", "export as buml", "export buml",
        "export model", "export the model", "export my model",
        "download project", "download the project", "download my project",
        "save as json", "save as buml", "save project",
        "save the project", "save my project", "save project to json",
        "save the project to json", "save the project as json",
        "export diagram", "export the diagram",
    ],
    "deploy": [
        "deploy to render", "deploy on render", "deploy app", "deploy the app",
        "deploy application", "deploy the application", "deploy my app",
        "deploy to cloud", "deploy project", "deploy the project",
        "deploy my project", "render deploy", "publish app", "publish the app",
        "publish to render", "publish my app",
        "deploy this model", "deploy the model", "deploy my model",
        "deploy it", "deploy this", "go ahead and deploy",
        "push this to prod", "push to prod", "push it to prod",
        "ship it to production", "ship this to production",
    ],
}

GENERATOR_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "django": [],
    "backend": [],
    "sql": ["dialect"],
    "sqlalchemy": ["dbms"],
    "jsonschema": ["mode"],
    "smartdata": [],
    "qiskit": ["backend", "shots"],
    "rest_api": [],
    "rdf": [],
    "export": ["format"],
    "deploy": [],
}

EXPORT_FORMATS = ["json", "buml"]

DIALECT_VALUES = ["sqlite", "postgresql", "mysql", "mssql", "mariadb", "oracle"]
MODE_VALUES = ["regular", "smart_data"]
QISKIT_BACKENDS = ["aer_simulator", "fake_backend", "ibm_quantum"]

# Common ways users actually spell a dialect/DBMS that don't match a
# DIALECT_VALUES entry verbatim (e.g. "postgres" instead of "postgresql").
# Without this, a message that clearly names a dialect ("postgres SQL")
# still triggered the "which dialect?" config prompt (#QA bug).
_DIALECT_ALIASES: Dict[str, str] = {
    "postgres": "postgresql",
    "psql": "postgresql",
    "sql server": "mssql",
    "sqlserver": "mssql",
    "maria": "mariadb",
}


def _resolve_dialect(lower_message: str) -> Optional[str]:
    """Return the canonical DIALECT_VALUES name mentioned in *lower_message*.

    Checks exact ``DIALECT_VALUES`` first, then the common aliases above
    (word-boundary matched so e.g. "psql" doesn't match inside an
    unrelated word). Returns ``None`` when no dialect/DBMS is named.
    """
    for dialect in DIALECT_VALUES:
        if re.search(r"\b" + re.escape(dialect) + r"\b", lower_message):
            return dialect
    for alias, canonical in _DIALECT_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", lower_message):
            return canonical
    return None


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
        r"\b(?:deploy|publish|push|ship)\b.*\b(?:render|cloud|app|application|"
        r"project|model|prod|production|live)\b", re.I)),
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
    "bpmn", "business process", "process diagram",
]

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
        dialect = _resolve_dialect(lower)
        if dialect:
            config["dialect"] = dialect

    elif generator_type == "sqlalchemy":
        dbms = _resolve_dialect(lower)
        if dbms:
            config["dbms"] = dbms

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
    Anyway, Cancel, or after a chained mismatch-regen run completes)
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
        PENDING_SMART_GEN_TIMESTAMP,
        SKIP_MISMATCH_CHECK_ONCE,
        MISMATCH_REGEN_PENDING,
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


def should_route_to_generation(session: Session, request: AssistantRequest) -> bool:
    """State-transition gatekeeper for ``generation_state``.

    Runs on every ``ReceiveJSONEvent``. We deliberately keep it cheap
    (NO LLM call, NO text heuristics) and defer all intent
    classification to the two places that already do that job:

      1. The unified classifier (one classifier-tier call per message,
         with BAF's local Simple classifier as fallback) decides whether
         this is a generation request. Its ``json_intent_matches``
         transition routes to ``generation_state`` on its own.
      2. Inside ``generation_state``, ``handle_generation_request`` calls
         the unified classifier's cached verdict (no extra LLM
         call) for the smart-vs-deterministic sub-routing.

    The only jobs of this function are the two non-intent signals BAF's
    classifier can't see:

      * ``frontend_event`` callbacks (``generator_result``, etc.) —
        ``raw_payload`` events, not conversational messages.
      * pending generator/config or smart-generation confirmation state —
        the next message belongs to the in-progress flow, not a new intent.

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
    return bool(
        pending_generator or session.get(PENDING_SMART_GEN_INSTRUCTIONS)
    )


def _normalize_defaults(generator_type: str, request: AssistantRequest, config: Dict[str, Any]) -> Dict[str, Any]:
    if generator_type == "django":
        config.setdefault("project_name", _extract_project_name_from_context(request))
        app_name = _extract_app_name_from_context(request)
        if config.get("project_name") == app_name:
            app_name = f"{app_name}_app"
        config.setdefault("app_name", app_name)
        config.setdefault("containerization", True)
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


# Friendly, generator-specific completion lines for a deterministic run. The
# browser already shows a result card (generator, "0 tokens", Download), so the
# agent's reply is just one confirming sentence under it — worded for what was
# actually built rather than a generic "generation completed".
_GENERATOR_DONE_MESSAGES: Dict[str, str] = {
    "generate_sql": "Your database schema is generated and ready to download.",
    "generate_sqlalchemy": "Your SQLAlchemy models are generated and ready to download.",
    "generate_django": "Your Django project is generated and ready to download.",
    "generate_fastapi_backend": "Your FastAPI backend is generated and ready to download.",
    "generate_backend": "Your backend is generated and ready to download.",
    "generate_web_app": "Your web app is generated and ready to download.",
    "generate_python": "Your Python classes are generated and ready to download.",
    "generate_java": "Your Java classes are generated and ready to download.",
    "generate_pydantic": "Your Pydantic models are generated and ready to download.",
    "generate_json_object": "Your JSON objects are generated and ready to download.",
    "generate_json_schema": "Your JSON Schema is generated and ready to download.",
    "generate_pytorch": "Your PyTorch model is generated and ready to download.",
    "generate_tensorflow": "Your TensorFlow model is generated and ready to download.",
    "generate_bpmn": "Your BPMN process is generated and ready to download.",
    "generate_qiskit": "Your quantum circuit code is generated and ready to download.",
    "generate_supabase": "Your Supabase schema is generated and ready to download.",
}


def _handle_frontend_event(request: AssistantRequest, session=None) -> Dict[str, Any]:
    event_type = request.raw_payload.get("eventType")
    if event_type == "generator_result":
        ok = bool(request.raw_payload.get("ok"))
        message = request.raw_payload.get("message")
        metadata = request.raw_payload.get("metadata")
        if isinstance(metadata, dict) and metadata.get("smart"):
            return _handle_smart_generator_result(ok, message, metadata, session)
        if ok:
            # One generator-appropriate confirmation under the card. Do NOT
            # re-echo the "Generating…" trigger text (it reads as "starting"
            # AFTER completion) or append the filename (the card already has it).
            gen = metadata.get("generatorType") if isinstance(metadata, dict) else None
            result_message = _GENERATOR_DONE_MESSAGES.get(
                gen, "Your code is generated and ready to download."
            )
        else:
            result_message = message if isinstance(message, str) and message.strip() else "Generation failed."
        return {"action": "assistant_message", "message": result_message}
    return {
        "action": "assistant_message",
        "message": "Received frontend event update.",
    }


def _handle_smart_generator_result(
    ok: bool,
    message: Any,
    metadata: Dict[str, Any],
    session,
) -> Dict[str, Any]:
    """Outcome report for a smart-generation run the agent triggered.

    Previously the smart path was fire-and-forget: the agent that
    classified the request and refined the instructions never learned
    whether the run succeeded, what it cost, or why it failed — so it
    couldn't follow up and "why did it fail?" got a blank stare. This
    records the outcome in conversation memory and replies with an
    outcome-aware message + suggested next steps.
    """
    error_code = metadata.get("errorCode")
    cost = metadata.get("costUsd")
    generator_used = metadata.get("generator_used")
    # Cost is intentionally NOT surfaced to the user (kept out of chat).
    cost_text = ""

    if ok:
        incomplete = bool(metadata.get("incomplete"))
        incomplete_reason = metadata.get("incompleteReason")
        if incomplete:
            head = (
                f"The {SPEC_DRIVEN_NAME} produced output, but the run stopped early "
                "so it may be incomplete"
            )
            if incomplete_reason:
                head += f": {incomplete_reason}"
            parts = [head + cost_text + "."]
        else:
            parts = [f"{SPEC_DRIVEN_NAME} generation finished successfully" + cost_text + "."]
        if metadata.get("filename") or metadata.get("fileName"):
            parts.append(f"File: {metadata.get('filename') or metadata.get('fileName')}")
        if incomplete:
            parts.append(
                "You can run the generation again to finish the remaining changes."
            )
        result_message = " ".join(parts)
        suggestions = None
    elif error_code == "COST_CAP":
        result_message = (
            f"The {SPEC_DRIVEN_NAME} run hit its cost cap before finishing"
            + cost_text
            + ". You can retry with a larger budget, or narrow the "
            "instructions so less code needs to be generated."
        )
        suggestions = ["Retry with refined instructions"]
    elif error_code == "CANCELLED":
        result_message = f"The {SPEC_DRIVEN_NAME} run was stopped" + cost_text + "."
        suggestions = ["Retry the generation"]
    elif error_code == "INVALID_KEY":
        result_message = (
            f"{SPEC_DRIVEN_NAME} generation failed: the provider rejected the API key. "
            "Check the key in the AI settings and try again."
        )
        suggestions = None
    else:
        detail = message if isinstance(message, str) and message.strip() else None
        result_message = (
            f"{SPEC_DRIVEN_NAME} generation failed"
            + (f" ({error_code})" if error_code else "")
            + cost_text
            + ("." if not detail else f": {detail}")
        )
        suggestions = ["Retry with refined instructions"]

    # Record the outcome so follow-up turns ("why did it fail?",
    # "run it again") have context. Best-effort — memory failures must
    # never break the reply.
    if session is not None:
        if ok:
            # Structured recency signal for the classifier's SMART-GEN
            # FOLLOW-UP rule. Outside the memory try/except on purpose: it
            # must survive memory failures, and only SUCCESSFUL runs count —
            # "add auth to it" after a FAILED run is not a follow-up to
            # reuse-for-generation.
            session.set(LAST_SMART_GEN_AT, time.time())
        # Stash the outcome (success OR failure) so a follow-up QUESTION
        # about the finished run ("what we generated?") can be answered
        # from here instead of re-arming a new generation confirmation.
        session.set(LAST_SMART_GEN_SUMMARY, result_message)
        try:
            from memory import get_memory, memory_session_key

            # Stable payload sessionId so memory survives reconnects (B-5).
            session_id = memory_session_key(session)
            mem = get_memory(session_id)
            mem.add_assistant(f"[smart-generation outcome] {result_message}"[:500])
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not record smart-gen outcome in memory", exc_info=True)

    payload: Dict[str, Any] = {"action": "assistant_message", "message": result_message}
    if suggestions:
        payload["suggestedActions"] = suggestions
    return payload


# A question ABOUT a past generation: interrogative opener + a
# generated/built/created reference within the same short message, or the
# bare "what did/have/was ... generate(d)" forms. Precision-first — an
# imperative like "generate rust classes" never matches (no interrogative
# opener), and future-directed questions are excluded separately.
_PAST_GEN_QUESTION_RE = re.compile(
    r"^(what|which|show|tell|list|describe|explain)\b.{0,60}\b(generat|built|created|produced)",
    re.IGNORECASE,
)
# Future-directed words that turn "what ... generate" into a request for
# options/capabilities or a NEW run — those keep the normal routing.
_FUTURE_GEN_WORDS_RE = re.compile(
    r"\b(should|shall|can|could|would|will|next|now|again|regenerate|new)\b",
    re.IGNORECASE,
)

# --- Continue-from-GitHub guard ------------------------------------------
# Chat path for resuming a PAST generation that was pushed to GitHub:
# "continue from github.com/owner/repo" must hand the frontend a
# trigger_github_import action (the frontend calls the backend import
# endpoint, loads the returned project, and arms the modify machinery —
# the agent itself never touches GitHub or any HTTP endpoint).
# Deterministic and precision-first, like the past-generation guard above:
# an LLM verdict is never allowed to invent, miss, or swallow an import.
#
# URL form — unambiguous on its own: github.com/{owner}/{repo} with an
# optional scheme, optional ".git" suffix (stripped in the extractor) and
# optional /tree/{branch} segment. The repo pattern is dotted-segment
# shaped so a trailing sentence period is NOT swallowed
# ("...from github.com/x/y." → repo "y") while dotted repo names and
# ".git" still match. The lookbehind rejects lookalike hosts
# ("mygithub.com", "api.github.com").
_GITHUB_URL_RE = re.compile(
    r"(?<![A-Za-z0-9.-])(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})?)/"
    r"(?P<repo>[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)"
    r"(?:/tree/(?P<branch>[^\s?#]+))?",
    re.IGNORECASE,
)
# Continuation verbs (EN + FR "reprendre" forms) — the bare owner/repo
# form fires ONLY alongside one of these, and the classifier-side reroute
# (unified_classifier._post_validate) requires one even for URLs, so a
# "create a diagram like github.com/x/y" style request is never hijacked.
_GITHUB_CONTINUE_VERB_RE = re.compile(
    r"\b(?:continue|continuing|continues|resume|resuming|resumes|"
    r"load|loading|loads|import|importing|imports|"
    r"open|opening|opens|reprend\w*|reprise)\b",
    re.IGNORECASE,
)
# The word that makes a bare "owner/repo" mean a REPOSITORY. Without it
# (plus a continuation verb) a slash pair in ordinary prose
# ("src/handlers", "1/2") must never fire.
_GITHUB_REPO_WORD_RE = re.compile(
    r"\b(?:repos?|repository|repositories|github)\b", re.IGNORECASE,
)
# Bare {owner}/{repo}: same owner/repo shapes as the URL form; the
# lookarounds keep it from matching inside a longer path or URL.
_BARE_OWNER_REPO_RE = re.compile(
    r"(?<![\w./@-])"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})?)/"
    r"(?P<repo>[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)"
    r"(?![\w/-])",
)
# "branch <name>" / "branche <name>" (FR) named in the message text — used
# only when the URL carried no /tree/<branch> segment.
_GITHUB_BRANCH_WORD_RE = re.compile(
    r"\bbranche?\s+[\"'`]?"
    r"(?P<branch>[A-Za-z0-9_/-]+(?:\.[A-Za-z0-9_/-]+)*)",
    re.IGNORECASE,
)


def _extract_github_reference(
    message: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Extract ``(owner, repo, branch)`` from a GitHub reference in *message*.

    Two accepted shapes, precision-first (never fires on arbitrary "a/b"
    prose):

    * a github.com URL (scheme optional, ``.git`` and ``/tree/{branch}``
      suffixes optional) — unambiguous on its own;
    * a bare ``owner/repo`` — ONLY when the message ALSO contains a
      continuation verb (continue / resume / load / import / open /
      reprend…) AND a repo word (repo / repository / github).

    ``branch`` comes from the URL's ``/tree/<branch>`` segment, else a
    "branch <name>" phrase in the message, else ``None`` (the backend
    then uses the repo's default branch). Returns ``None`` when the
    message carries no GitHub reference.
    """
    msg = message or ""
    owner = repo = branch = None
    url_match = _GITHUB_URL_RE.search(msg)
    if url_match:
        owner = url_match.group("owner")
        repo = url_match.group("repo")
        branch = url_match.group("branch")
    elif _GITHUB_CONTINUE_VERB_RE.search(msg) and _GITHUB_REPO_WORD_RE.search(msg):
        bare_match = _BARE_OWNER_REPO_RE.search(msg)
        if bare_match:
            owner = bare_match.group("owner")
            repo = bare_match.group("repo")
    if not owner or not repo:
        return None
    if repo.lower().endswith(".git"):
        repo = repo[: -len(".git")]
    if not repo:
        return None
    if not re.search(r"[A-Za-z]", owner + repo):
        # Digits-only pairs ("1/2", "3/4") are prose — fractions or dates —
        # not a plausible GitHub reference.
        return None
    if branch:
        # A /tree/<branch> segment is greedy up to whitespace so slashed
        # branch names ("feature/login") survive; trim sentence punctuation.
        branch = branch.rstrip(".,;:!?)'\"/")
    if not branch:
        word_match = _GITHUB_BRANCH_WORD_RE.search(msg)
        if word_match:
            branch = word_match.group("branch")
    return owner, repo, branch or None


def _build_github_import_payload(
    owner: str, repo: str, branch: Optional[str],
) -> Dict[str, Any]:
    """Build the ``trigger_github_import`` action payload.

    The frontend handles the action: it calls the backend's GitHub import
    endpoint, loads the returned project into the editor, and arms the
    incremental-modify machinery. ``branch`` is ``None`` when the user
    named none (the backend then uses the repo's default branch).
    """
    branch_part = f" on branch {branch}" if branch else ""
    return {
        "action": "trigger_github_import",
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "message": (
            f"Importing **{owner}/{repo}**{branch_part} from GitHub — I'll "
            "load the project and you can continue modifying it from here. "
            "If the repo wasn't created by BESSER (no model inside), I'll "
            "ask you to open it in the editor first."
        ),
    }


# Normalized spellings of the mismatch quick-action label a user might TYPE
# instead of clicking. Matched only while a mismatch rebuild is stashed.
_MISMATCH_LABEL_ALIASES = {
    "update model + generate",
    "update model and generate",
    "update the model + generate",
    "update the model and generate",
    "update model plus generate",
    "update model generate",
}


def handle_generation_request(session: Session, request: AssistantRequest) -> Dict[str, Any]:
    """Route a generation-state request to smart-gen, deterministic, or menu.

    BAF's intent classifier has already decided this is a generation
    request. Our job here is the SUB-routing: smart generator (for any
    custom stack / language BESSER doesn't have built-in) vs one of
    BESSER's deterministic generators (django / pydantic / sql / …) vs
    redirect-to-modeling (if BAF misclassified).

    The sub-routing verdict (route + generator_type +
    refined_instructions) comes from the unified classifier's cached
    per-message classification — one rulebook, zero extra LLM calls.
    """
    if request.action == "frontend_event":
        return _handle_frontend_event(request, session)

    # Past-generation QUESTION guard (live bug 2026-09-01): after a smart
    # run finished, "What we generated" classified as generation_intent and
    # re-armed the whole smart-gen confirmation instead of being answered.
    # A past-tense/interrogative reference to the finished run — with no
    # future-directed word — is a question about the outcome, never a new
    # run. Deterministic, and only fires while a completed run is fresh.
    _msg = (request.message or "").strip()
    if _PAST_GEN_QUESTION_RE.search(_msg) and not _FUTURE_GEN_WORDS_RE.search(_msg):
        _summary = session.get(LAST_SMART_GEN_SUMMARY)
        _at = session.get(LAST_SMART_GEN_AT) or 0
        if _summary and (time.time() - float(_at)) < 1800:
            logger.info(
                "Past-generation question answered from stashed outcome "
                "(no new run armed): %r", _msg[:80],
            )
            return {
                "action": "assistant_message",
                "message": (
                    f"{_summary} Use the **Download** button on the run card "
                    "to save the files, or tell me what to change and I'll "
                    "modify the generated app."
                ),
            }

    # Continue-from-GitHub guard: "continue from github.com/x/y" (or
    # "continue from my repo x/y") means resuming a PAST generation that was
    # pushed to GitHub. Deterministic and terminal: extract owner/repo/branch
    # and hand the frontend a trigger_github_import action — it calls the
    # backend import endpoint, loads the project, and arms the modify
    # machinery. Runs BEFORE sub-route dispatch so no LLM verdict (smart /
    # deterministic / modeling) can swallow the import into a fresh run.
    _gh_ref = _extract_github_reference(_msg)
    if _gh_ref is not None:
        _gh_owner, _gh_repo, _gh_branch = _gh_ref
        # An import loads a DIFFERENT project: abandon any pending generator
        # config / smart-gen confirmation, exactly as the "different request
        # abandons this confirmation" path below would — so a later generic
        # "yes" can never spend against a stale pre-import run (B-2).
        _clear_pending_smart_gen(session)
        _clear_pending_state(session)
        logger.info(
            "Continue-from-GitHub: importing %s/%s (branch=%s)",
            _gh_owner, _gh_repo, _gh_branch,
        )
        return _build_github_import_payload(_gh_owner, _gh_repo, _gh_branch)

    # Pending smart-generation confirmation: short-circuit the classifier only
    # for an unambiguous whole yes/no/cancel reply. Qualified or mixed replies
    # are deliberately not treated as approval to spend the user's API key.
    msg_lower = (request.message or "").strip().lower()
    has_pending_smart_gen = bool(session.get(PENDING_SMART_GEN_INSTRUCTIONS))
    smart_gen_decision = (
        _smart_gen_confirmation_decision(msg_lower) if has_pending_smart_gen else None
    )
    if has_pending_smart_gen and smart_gen_decision is None:
        # ActiveFlow: the classifier may CANCEL on the user's behalf (novel
        # phrasings like "rather not" — cancelling is always safe) but must
        # NEVER CONFIRM: firing the generator SPENDS a run (the user's own
        # API key on BYOK), so confirmation stays exact-phrase/button-only
        # (B-2). Live lesson: "fast" — meant for the GUI choice — was read
        # by the LLM as an eager confirm and fired a run the user never
        # asked for.
        _uc_sc = session.get(UNIFIED_CLASSIFICATION)
        if (getattr(_uc_sc, "pending_flow_action", None) == "answer"
                and getattr(_uc_sc, "pending_flow_answer", None) == "cancel"):
            smart_gen_decision = "cancel"

    if smart_gen_decision == "cancel":
        _clear_pending_smart_gen(session)
        _clear_pending_state(session)
        return {
            "action": "assistant_message",
            "message": "Cancelled. Your model is unchanged.",
        }

    if smart_gen_decision == "confirm":
        # Explicit user confirmation: fire the smart-gen handoff with the
        # previously-stashed instructions / provider. This is the ONLY
        # place a trigger_smart_generator payload is emitted — every
        # other path stashes + asks first (B-2: the run spends the
        # user's own API key).
        from handlers.smart_generation_handler import GenerationClassification
        stashed_instructions = session.get(PENDING_SMART_GEN_INSTRUCTIONS) or ""
        stashed_provider = session.get(PENDING_SMART_GEN_PROVIDER) or "anthropic"
        stashed_ts = session.get(PENDING_SMART_GEN_TIMESTAMP)
        if not _smart_gen_stash_is_fresh(stashed_ts):
            # Reject stale confirmations: a stash older than the TTL may
            # belong to a long-abandoned dialog the user no longer means.
            _clear_pending_smart_gen(session)
            _clear_pending_state(session)
            return {
                "action": "assistant_message",
                "message": (
                    "That generation request expired — please ask again "
                    "and I'll prepare a fresh run."
                ),
            }
        _clear_pending_smart_gen(session)
        if stashed_instructions.strip():
            return build_trigger_smart_generator_payload(
                GenerationClassification(
                    route="smart",
                    refined_instructions=stashed_instructions,
                    provider=stashed_provider,
                    reason="user confirmed the run",
                ),
                reason_prefix="generating with current model",
            )
        # Fall through to normal classification if the stash was empty.

    # Typed "Update model + generate": users sometimes TYPE the mismatch
    # button's label instead of clicking it. The raw label re-classified as a
    # fresh smart-gen request and re-showed the mismatch question in a loop
    # (and, below, would have cleared the stashed run as a "different
    # request"). Treat any label alias as the button press: dispatch the
    # stashed rebuild prompt to the modeling path — the create choke point
    # then resumes the stashed smart-gen exactly like the real button.
    _regen_stash_alias = session.get(MISMATCH_REGEN_PENDING)
    if (
        isinstance(_regen_stash_alias, str) and _regen_stash_alias.strip()
        and _norm_prompt(request.message) in _MISMATCH_LABEL_ALIASES
    ):
        logger.info(
            "[Generation] Typed mismatch-button label — dispatching the "
            "stashed rebuild prompt"
        )
        request.message = _regen_stash_alias
        try:
            from execution import execute_planned_operations
            execute_planned_operations(
                session=session,
                request=request,
                default_mode="complete_system",
                matched_intent="create_complete_system_intent",
            )
            return None
        except Exception:
            logger.exception("[Generation] mismatch label dispatch failed")

    _regen_prompt = session.get(MISMATCH_REGEN_PENDING)
    _is_regen_rebuild = (
        isinstance(_regen_prompt, str)
        and _norm_prompt(request.message) == _norm_prompt(_regen_prompt)
    )
    if has_pending_smart_gen and not _is_regen_rebuild:
        _stash_intent = getattr(
            session.get(UNIFIED_CLASSIFICATION), "intent", None)
        if _stash_intent == "out_of_scope_intent":
            # Off-topic interjection ("draw me a cat") at the confirmation:
            # answer it and KEEP the prepared run so "Continue" still works.
            # (out_of_scope_state is unreachable here — the pending stash
            # suppresses intent routing.)
            return {"action": "assistant_message",
                    "message": OUT_OF_SCOPE_REDIRECT}
        if _stash_intent == "decline_intent":
            # A decline at the confirmation IS the answer: cancel cleanly.
            _clear_pending_smart_gen(session)
            _clear_pending_state(session)
            return {"action": "assistant_message",
                    "message": "Cancelled. Your model is unchanged."}
        # A different request abandons this confirmation. Clearing the stash
        # prevents a later generic "yes" from spending against an old run.
        # EXCEPTION: the domain-mismatch "Update model + generate" chain only
        # keeps the stash alive when THIS message is exactly the stashed rebuild
        # prompt — the intended continuation. A different create typed after a
        # mismatch falls through here and abandons normally (no spurious resume).
        _clear_pending_smart_gen(session)

    pending_generator, pending_config = _get_pending_state(session)
    _use_pending = bool(pending_generator and pending_generator != _AWAITING_SELECTION)

    # When a config-collection flow is pending (e.g. we asked for the Django
    # project name), the user may PIVOT instead of answering — "generate the
    # database" while mid-Django-config, or escalating to the smart generator.
    # Trust the fresh classification: if it's a clear request for a DIFFERENT
    # built-in generator, or the smart route, abandon the pending flow and
    # re-route — otherwise we'd re-prompt for the old generator's config
    # forever (the "asked for a database, got Django questions" bug).
    # Consult the CACHED classification (populated by state_bodies' priority-0
    # hook — read directly so there is NO extra LLM call here). If the cache is
    # empty we simply don't pivot and continue the pending flow.
    classification = None
    if _use_pending:
        # Opt-out FIRST: an explicit cancel or a decline verdict mid-config
        # must EXIT the flow. Without this, the reply looped the config
        # prompt and the CONFIG_PROMPT_ATTEMPTS >= 3 branch auto-filled
        # defaults and generated the very thing the user was refusing.
        _uc_cfg = session.get(UNIFIED_CLASSIFICATION)
        if (
            _norm_prompt(request.message) in _CONFIG_CANCEL_PHRASES
            or getattr(_uc_cfg, "intent", None) == "decline_intent"
            or (getattr(_uc_cfg, "pending_flow_action", None) == "answer"
                and getattr(_uc_cfg, "pending_flow_answer", None) == "cancel")
        ):
            logger.info(
                "[Generation] Pending '%s' config flow cancelled by the user",
                pending_generator,
            )
            _clear_pending_state(session)
            session.set(CONFIG_PROMPT_ATTEMPTS, 0)
            return {
                "action": "assistant_message",
                "message": "Cancelled — no code was generated. Your model is unchanged.",
            }
        _cached = session.get(UNIFIED_CLASSIFICATION)
        if _cached is not None and getattr(_cached, "intent", None) == "generation_intent":
            _fresh = _classification_to_legacy(_cached)
            pivoted = _fresh.route == "smart" or (
                _fresh.route == "deterministic"
                and _fresh.generator_type
                and _fresh.generator_type != pending_generator
            )
            if pivoted:
                logger.info(
                    "generation pivot: abandoning pending '%s' config flow for "
                    "route=%s generator_type=%s",
                    pending_generator, _fresh.route, _fresh.generator_type,
                )
                _clear_pending_state(session)
                session.set(CONFIG_PROMPT_ATTEMPTS, 0)
                _use_pending = False
                classification = _fresh  # reuse below; avoids a second lookup

    if _use_pending:
        # Multi-turn continuation: user is answering the pending generator's
        # config prompt (e.g. "what's your Django project name?"). Use the
        # already-picked generator and fall through to the config parsing /
        # dispatch path below without re-routing.
        generator_type: Optional[str] = pending_generator
    else:
        # First-time-through (or just-pivoted): read the unified classifier's
        # verdict from the per-message cache (populated by state_bodies'
        # priority-0 hook). Falls back to a fresh call only when the cache is
        # empty — typically in tests that bypass the state machine. Net result
        # in production: ZERO extra LLM calls — the classification was already
        # done before this state body ran (and reused above when pending).
        if classification is None:
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

            # Never fire directly: the smart generator spends the user's
            # own API key, so stash + ask for explicit confirmation (B-2).
            return _build_smart_gen_confirmation(
                session,
                classification.refined_instructions or "",
                classification.provider or "anthropic",
            )

        if classification.route == "modeling":
            _clear_pending_state(session)
            # The request is really a modeling request (create OR modify),
            # not code-gen. Build/edit the model directly instead of bouncing
            # the user with a "rephrase" message — classification is
            # non-deterministic, so this makes the outcome CONSISTENT
            # regardless of which path the message took. The MODE comes from
            # the unified verdict: a modify-shaped message ("add a Payment
            # class and regenerate") must MODIFY, not rebuild from scratch.
            # execute_planned_operations sends the reply itself; return None
            # so the caller adds nothing more (deferred import avoids a
            # module-load cycle).
            _uc = session.get(UNIFIED_CLASSIFICATION)
            if getattr(_uc, "intent", None) == "modify_model_intent":
                _mode, _intent = "modify_model", "modify_model_intent"
            else:
                _mode, _intent = "complete_system", "create_complete_system_intent"
            try:
                from execution import execute_planned_operations
                execute_planned_operations(
                    session=session,
                    request=request,
                    default_mode=_mode,
                    matched_intent=_intent,
                )
                return None
            except Exception as exc:
                logger.error(f"[GenRedirect] modeling build failed: {exc}", exc_info=True)
                return {
                    "action": "assistant_message",
                    "message": (
                        "It looks like you want to design a system. Try "
                        '**"create a class diagram for a library"**.'
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
            "bpmn", "business process", "process diagram",
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
                "**APIs & semantics**: `rest_api`, `rdf`\n"
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
    # Validate the diagrams this specific generator consumes. An unrelated
    # model must not satisfy a ClassDiagram prerequisite, and canonical GUI
    # models carry ``pages`` rather than an ``elements`` map.
    request_context = getattr(request, "context", None)
    missing_prerequisites = _missing_generator_prerequisites(
        request_context, generator_type,
    )
    if missing_prerequisites:
        labels = {
            "ClassDiagram": "Class Diagram",
            "GUINoCodeDiagram": "GUI Diagram",
            "AgentDiagram": "Agent Diagram",
            "QuantumCircuitDiagram": "Quantum Circuit Diagram",
        }
        missing_text = " and ".join(
            f"**{labels.get(dtype, dtype)}**" for dtype in missing_prerequisites
        )
        if not _project_has_any_model(request_context):
            lead = "Your workspace looks empty"
        else:
            lead = f"Your workspace is missing a usable {missing_text}"
        return {
            "action": "assistant_message",
            "message": (
                f"{lead} — **{generator_type}** generation requires {missing_text}. "
                f"Describe what you want first "
                f"(e.g. *\"create a library management system\"*), then ask me to "
                f"generate the code."
            ),
        }

    return {
        "action": "trigger_generator",
        "generatorType": generator_type,
        "config": config,
        "message": f"Starting **{generator_type}** code generation — this may take a moment.",
    }
