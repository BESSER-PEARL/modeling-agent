import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from prompts import planner as planner_prompts
from protocol.types import AssistantRequest, SUPPORTED_DIAGRAM_TYPES
from routing.generation_signals import (
    GENERATOR_KEYWORDS,
    GENERATOR_PREREQUISITES,
    detect_generator_type,
)

from .workspace_orchestrator import determine_target_diagram_types, keyword_target_hits

logger = logging.getLogger(__name__)

# Re-export canonical set for backward compat; prefer SUPPORTED_DIAGRAM_TYPES.
ALLOWED_DIAGRAM_TYPES: Set[str] = SUPPORTED_DIAGRAM_TYPES

ALLOWED_MODEL_MODES: Set[str] = {
    "complete_system",
    "modify_model",
}

ALLOWED_GENERATORS: Set[str] = set(GENERATOR_KEYWORDS.keys())

PLANNER_CONNECTORS = (
    " and ",
    " then ",
    ";",
    " also ",
    " after ",
    " finally ",
)

SEGMENT_SPLIT_PATTERN = re.compile(
    r"(?:"
    r"\s*;\s*"                         # semicolon separator
    r"|\s+and then\s+"                 # "and then"
    r"|\s+then\s+"                     # "then"
    r"|\s+also\s+"                     # "also"
    r"|\s+after that\s+"               # "after that"
    r"|\s+finally\s+"                  # "finally"
    r"|\s+next\s+"                     # "next"
    r"|(?:^|\n)\s*\d+[\.\)]\s+"        # numbered list items (1. / 2) / etc.)
    r"|(?:^|\n)\s*[-*]\s+"             # bulleted list items (- / * )
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_json_response(raw_response: str) -> str:
    cleaned = (raw_response or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _normalize_model_mode(mode: Any, default_mode: str) -> str:
    if isinstance(mode, str) and mode in ALLOWED_MODEL_MODES:
        return mode
    return default_mode


def _normalize_diagram_type(diagram_type: Any) -> Optional[str]:
    if isinstance(diagram_type, str) and diagram_type in ALLOWED_DIAGRAM_TYPES:
        return diagram_type
    return None


def _build_context_summary(request: AssistantRequest) -> str:
    lines: List[str] = []
    context = request.context

    lines.append(f"Active diagram type: {context.active_diagram_type or 'ClassDiagram'}")
    if context.active_diagram_id:
        lines.append(f"Active diagram id: {context.active_diagram_id}")

    # Surface the active tab index for each diagram type so the planner is
    # aware that multiple tabs may exist and which one is currently visible.
    if context.current_diagram_indices:
        index_parts = [
            f"{dt}=tab{idx}"
            for dt, idx in context.current_diagram_indices.items()
        ]
        if index_parts:
            lines.append(f"Active tab indices: {', '.join(index_parts)}")

    snapshot = context.project_snapshot
    if isinstance(snapshot, dict):
        project_name = snapshot.get("name")
        if isinstance(project_name, str) and project_name.strip():
            lines.append(f"Project name: {project_name.strip()}")

        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            summarized: List[str] = []
            for diagram_type, diagram_payload in diagrams.items():
                if isinstance(diagram_payload, list):
                    # Multi-tab: each entry in the list is a diagram tab
                    tabs = [d for d in diagram_payload if isinstance(d, dict)]
                    if not tabs:
                        continue
                    if len(tabs) == 1:
                        title = tabs[0].get("title")
                        label = diagram_type
                        if isinstance(title, str) and title.strip():
                            label = f"{diagram_type} ({title.strip()})"
                        summarized.append(label)
                    else:
                        tab_labels = []
                        for i, tab in enumerate(tabs):
                            title = tab.get("title")
                            if isinstance(title, str) and title.strip():
                                tab_labels.append(f'tab {i}: "{title.strip()}"')
                            else:
                                tab_labels.append(f"tab {i}")
                        summarized.append(f"{diagram_type} ({len(tabs)} tabs: {', '.join(tab_labels)})")
                elif isinstance(diagram_payload, dict):
                    title = diagram_payload.get("title")
                    label = diagram_type
                    if isinstance(title, str) and title.strip():
                        label = f"{diagram_type} ({title.strip()})"
                    summarized.append(label)
            if summarized:
                lines.append("Available diagrams: " + ", ".join(summarized[:8]))

    summaries = context.diagram_summaries or []
    if summaries:
        summary_labels: List[str] = []
        for item in summaries:
            if not isinstance(item, dict):
                continue
            diagram_type = item.get("diagramType")
            title = item.get("title")
            if isinstance(diagram_type, str):
                if isinstance(title, str) and title.strip():
                    summary_labels.append(f"{diagram_type} ({title.strip()})")
                else:
                    summary_labels.append(diagram_type)
        if summary_labels:
            lines.append("Diagram summaries: " + ", ".join(summary_labels[:8]))

    return "\n".join(lines)


def _extract_generation_request_fragment(message: str) -> str:
    lower = (message or "").lower()
    first_index: Optional[int] = None

    for keywords in GENERATOR_KEYWORDS.values():
        for keyword in keywords:
            index = lower.find(keyword)
            if index < 0:
                continue
            if first_index is None or index < first_index:
                first_index = index

    if first_index is None:
        return message

    fragment = message[first_index:].strip()
    return fragment or message


def _split_message_segments(message: str) -> List[str]:
    if not isinstance(message, str):
        return []
    normalized = message.strip()
    if not normalized:
        return []
    segments = [segment.strip(" .") for segment in SEGMENT_SPLIT_PATTERN.split(normalized) if segment.strip(" .")]
    return segments or [normalized]


def _match_segment_target(segment: str) -> Optional[str]:
    candidates = keyword_target_hits(segment.lower())
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _build_target_requests(message: str, targets: List[str]) -> Dict[str, str]:
    per_target_segments: Dict[str, List[str]] = {target: [] for target in targets}
    for segment in _split_message_segments(message):
        matched_target = _match_segment_target(segment)
        if matched_target in per_target_segments:
            per_target_segments[matched_target].append(segment)

    target_requests: Dict[str, str] = {}
    for target in targets:
        segments = per_target_segments.get(target) or []
        target_requests[target] = " and ".join(segments).strip() if segments else message.strip()
    return target_requests


def _should_use_llm_planner(
    message: str,
    inferred_target_count: int,
    has_generation_request: bool,
    matched_intent: Optional[str] = None,
) -> bool:
    lower = (message or "").lower()
    has_connector = any(connector in lower for connector in PLANNER_CONNECTORS)
    has_multi_clause = has_connector or lower.count(",") >= 2
    has_explicit_diagram_tokens = bool(keyword_target_hits(lower))

    # Fast path: when the intent classifier already resolved a single-
    # diagram intent and keyword inference agrees (exactly 1 target, no
    # generation request), the fallback operations are sufficient.
    _SINGLE_TARGET_INTENTS = {
        "create_complete_system_intent",
        "modify_model_intent",
    }
    if (
        matched_intent in _SINGLE_TARGET_INTENTS
        and inferred_target_count == 1
        and not has_generation_request
    ):
        return False

    if has_multi_clause and (inferred_target_count > 1 or has_generation_request):
        return True

    # Let the LLM planner split ambiguous multi-step prompts that mention both modeling and generation,
    # even if keyword inference only produced one target.
    if has_generation_request and has_multi_clause:
        return True

    # If the prompt is complex and diagram targeting is implicit, planner usually yields cleaner sub-requests.
    if has_multi_clause and not has_explicit_diagram_tokens and len(lower) > 120:
        return True

    # When neither explicit keywords nor discriminating patterns produced a
    # target (inferred_target_count == 0 from context fallback), and the
    # message is a creation intent with enough content to be meaningful,
    # let the LLM resolve the diagram type rather than defaulting blindly.
    if (
        matched_intent == "create_complete_system_intent"
        and inferred_target_count <= 1
        and not has_explicit_diagram_tokens
        and len(lower) > 40
    ):
        return True

    return False


# Modeling intents that should never trigger an automatic generation operation.
_MODELING_INTENTS = {
    "create_complete_system_intent",
    "modify_model_intent",
    "modeling_help_intent",
    "describe_model_intent",
}


def _fallback_operations(
    request: AssistantRequest,
    default_mode: str,
    matched_intent: Optional[str],
    llm_target_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    targets = determine_target_diagram_types(
        request, last_intent=matched_intent, max_targets=3, llm_target_type=llm_target_type
    )

    # ClassDiagram is a prerequisite for other diagram types (GUI, Object, etc.)
    # so it must always be processed first when present alongside others.
    if "ClassDiagram" in targets and targets[0] != "ClassDiagram":
        targets = ["ClassDiagram"] + [t for t in targets if t != "ClassDiagram"]

    target_requests = _build_target_requests(request.message, targets)
    operations: List[Dict[str, Any]] = [
        {
            "type": "model",
            "diagramType": target,
            "mode": default_mode,
            "request": target_requests.get(target, request.message),
        }
        for target in targets
    ]

    # Only append a generation operation when the user's intent is actually
    # about code generation, not when they happen to use a keyword like
    # "web app" in a modeling request.
    if matched_intent not in _MODELING_INTENTS:
        generator_type = detect_generator_type(request.message)
        if generator_type:
            generation_request = _extract_generation_request_fragment(request.message)
            operations.append(
                {
                    "type": "generation",
                    "generatorType": generator_type,
                    "config": {},
                    "request": generation_request,
                }
            )

    return operations


def _normalize_operations(
    raw_operations: Any,
    request: AssistantRequest,
    default_mode: str,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_operations, list):
        return []

    normalized: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for operation in raw_operations:
        if not isinstance(operation, dict):
            continue

        op_type = operation.get("type")
        if not isinstance(op_type, str):
            continue

        if op_type == "model":
            diagram_type = _normalize_diagram_type(operation.get("diagramType"))
            if not diagram_type:
                continue

            mode = _normalize_model_mode(operation.get("mode"), default_mode)
            op_request = operation.get("request")
            op_request = op_request.strip() if isinstance(op_request, str) else request.message
            if not isinstance(op_request, str) or not op_request.strip():
                continue

            key = (op_type, diagram_type, mode, op_request.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "type": "model",
                    "diagramType": diagram_type,
                    "mode": mode,
                    "request": op_request.strip(),
                }
            )
            continue

        if op_type == "generation":
            generator_type = operation.get("generatorType")
            if not isinstance(generator_type, str) or generator_type not in ALLOWED_GENERATORS:
                inferred = detect_generator_type(operation.get("request") if isinstance(operation.get("request"), str) else request.message)
                generator_type = inferred

            if not isinstance(generator_type, str) or generator_type not in ALLOWED_GENERATORS:
                continue

            config = operation.get("config")
            config = config if isinstance(config, dict) else {}

            key = (op_type, generator_type, "", json.dumps(config, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "type": "generation",
                    "generatorType": generator_type,
                    "config": config,
                }
            )

    # Enforce ordering: model ops before generation ops, ClassDiagram first among models
    model_ops = [op for op in normalized if op.get("type") == "model"]
    gen_ops = [op for op in normalized if op.get("type") == "generation"]
    if model_ops:
        # ClassDiagram must come first — other diagrams may depend on it
        class_ops = [op for op in model_ops if op.get("diagramType") == "ClassDiagram"]
        other_ops = [op for op in model_ops if op.get("diagramType") != "ClassDiagram"]
        model_ops = class_ops + other_ops
    normalized = model_ops + gen_ops

    return normalized


# ---------------------------------------------------------------------------
# Heuristic pre-decomposer: fast regex patterns for common request shapes.
# When a pattern matches the entire intent can be resolved without an LLM call.
# ---------------------------------------------------------------------------

_HEURISTIC_WEB_APP = re.compile(
    r"^(?:create|build|design|make)\s+(?:a\s+)?web\s*app(?:lication)?\s+(?:for\s+)?(?P<domain>.+)$",
    re.IGNORECASE,
)

_HEURISTIC_MODEL_AND_GENERATE = re.compile(
    r"^(?P<model_part>(?:create|build|design|make)\s+.+?)"
    r"\s+(?:and\s+)?(?:then\s+)?(?:generate|export|produce|output)\s+(?P<gen_part>.+)$",
    re.IGNORECASE,
)

_HEURISTIC_GENERATE_ONLY = re.compile(
    r"^(?:generate|export|produce|output)\s+(?P<gen_part>.+)$",
    re.IGNORECASE,
)

# Single-diagram creation patterns  ──────────────────────────────────
# These catch the most common request shapes so the LLM planner is
# never invoked for straightforward single-diagram creation.

_HEURISTIC_GUI = re.compile(
    r"^(?:create|build|design|make|generate|add)\s+(?:a\s+|the\s+)?"
    r"(?:gui|graphical\s*ui|frontend|web\s*ui|user\s*interface|ui)"
    r"(?:\s+(?:diagram|page|screen|layout))?"
    r"(?:\s+(?:for|of|from)\s+(?:this\s+(?:system|project|model)|my\s+(?:system|project|model|classes)|(?P<domain>.+)))?"
    r"\s*$",
    re.IGNORECASE,
)

_HEURISTIC_STATE_MACHINE = re.compile(
    r"^(?:create|build|design|make|add)\s+(?:a\s+|the\s+)?"
    r"(?:state\s*machine|state\s*diagram|workflow|lifecycle)"
    r"(?:\s+(?:diagram))?"
    r"(?:\s+(?:for|of)\s+(?P<domain>.+))?"
    r"\s*$",
    re.IGNORECASE,
)

_HEURISTIC_AGENT = re.compile(
    r"^(?:create|build|design|make|add)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:agent|chatbot|conversational\s*agent|agent\s*diagram|agent\s*model)"
    r"(?:\s+(?:diagram|model))?"
    r"(?:\s+(?:for|of|about|that)\s+(?P<domain>.+))?"
    r"\s*$",
    re.IGNORECASE,
)

_HEURISTIC_QUANTUM = re.compile(
    r"^(?:create|build|design|make|add|implement)\s+(?:a\s+|the\s+)?"
    r"(?:quantum\s*circuit|quantum\s*diagram|quantum\s*algorithm)"
    r"(?:\s+(?:diagram))?"
    r"(?:\s+(?:for|of|with)\s+(?P<domain>.+))?"
    r"\s*$",
    re.IGNORECASE,
)

_HEURISTIC_OBJECT = re.compile(
    r"^(?:create|build|design|make|add)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:object\s*diagram|object\s*model|object\s*instances?)"
    r"(?:\s+(?:for|of|from)\s+(?P<domain>.+))?"
    r"\s*$",
    re.IGNORECASE,
)

_HEURISTIC_CLASS = re.compile(
    r"^(?:create|build|design|make)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:class\s*diagram|class\s*model|domain\s*model)"
    r"(?:\s+(?:for|of|with)\s+(?P<domain>.+))?"
    r"\s*$",
    re.IGNORECASE,
)

_HEURISTIC_BPMN = re.compile(
    r"^(?:create|build|design|make)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:bpmn\s*(?:diagram|process|model)?|business\s*process(?:\s*diagram)?)"
    r"(?:\s+(?:for|of|with)\s+(?P<domain>.+))?"
    r"\s*$",
    re.IGNORECASE,
)

_HEURISTIC_COMPONENT = re.compile(
    r"^(?:create|build|design|make)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:uml\s+)?component\s*(?:diagram|model)"
    r"(?:\s+(?:for|of|with)\s+(?P<domain>.+))?"
    r"\s*$",
    re.IGNORECASE,
)

_HEURISTIC_DEPLOYMENT = re.compile(
    r"^(?:create|build|design|make)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:uml\s+)?deployment\s*(?:diagram|model)"
    r"(?:\s+(?:for|of|with)\s+(?P<domain>.+))?"
    r"\s*$",
    re.IGNORECASE,
)

# Map of (pattern, diagramType) for the single-diagram heuristics
_SINGLE_DIAGRAM_HEURISTICS = [
    (_HEURISTIC_GUI, "GUINoCodeDiagram"),
    (_HEURISTIC_STATE_MACHINE, "StateMachineDiagram"),
    (_HEURISTIC_AGENT, "AgentDiagram"),
    (_HEURISTIC_QUANTUM, "QuantumCircuitDiagram"),
    (_HEURISTIC_OBJECT, "ObjectDiagram"),
    (_HEURISTIC_CLASS, "ClassDiagram"),
    (_HEURISTIC_BPMN, "BPMN"),
    (_HEURISTIC_COMPONENT, "ComponentDiagram"),
    (_HEURISTIC_DEPLOYMENT, "DeploymentDiagram"),
]


def _try_heuristic_decomposition(
    message: str,
    request: AssistantRequest,
    default_mode: str,
) -> Optional[List[Dict[str, Any]]]:
    """Attempt a fast regex-based decomposition.  Returns *None* when no
    heuristic matches so the caller falls through to the LLM planner."""
    stripped = (message or "").strip()
    if not stripped:
        return None

    # Pattern 1: "create/build a web app for X"
    match = _HEURISTIC_WEB_APP.match(stripped)
    if match:
        domain = match.group("domain").strip().rstrip(".")
        return [
            {"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system",
             "request": f"create a class diagram for {domain}"},
            {"type": "model", "diagramType": "GUINoCodeDiagram", "mode": "complete_system",
             "request": f"create a GUI for {domain}"},
            {"type": "generation", "generatorType": "web_app", "config": {}},
        ]

    # Pattern 2: "create/design X and generate Y"
    match = _HEURISTIC_MODEL_AND_GENERATE.match(stripped)
    if match:
        model_part = match.group("model_part").strip()
        gen_part = match.group("gen_part").strip().rstrip(".")
        gen_type = detect_generator_type(gen_part)
        if gen_type:
            # Determine diagram type from the modeling part
            target = _match_segment_target(model_part) or "ClassDiagram"
            operations: List[Dict[str, Any]] = [
                {"type": "model", "diagramType": target, "mode": "complete_system",
                 "request": model_part},
            ]
            # If the generator needs additional prerequisite diagrams, add them
            prereqs = GENERATOR_PREREQUISITES.get(gen_type, [])
            for prereq in prereqs:
                if prereq != target:
                    domain_hint = re.sub(
                        r"^(?:create|build|design|make)\s+(?:a\s+)?", "", model_part, flags=re.IGNORECASE
                    ).strip()
                    operations.append(
                        {"type": "model", "diagramType": prereq, "mode": "complete_system",
                         "request": f"create {prereq} for {domain_hint}"}
                    )
            operations.append({"type": "generation", "generatorType": gen_type, "config": {}})
            return operations

    # Pattern 3: "generate X" (standalone generation, no modeling)
    match = _HEURISTIC_GENERATE_ONLY.match(stripped)
    if match:
        gen_type = detect_generator_type(stripped)
        if gen_type:
            return [{"type": "generation", "generatorType": gen_type, "config": {}}]

    # Pattern 4: Single-diagram creation ("create a gui for X", "add a
    # state machine for X", "create an agent for X", etc.)
    for pattern, diagram_type in _SINGLE_DIAGRAM_HEURISTICS:
        match = pattern.match(stripped)
        if match:
            domain = ""
            try:
                domain = (match.group("domain") or "").strip().rstrip(".")
            except (IndexError, AttributeError):
                pass
            req_text = stripped if domain else stripped
            return [
                {"type": "model", "diagramType": diagram_type, "mode": default_mode,
                 "request": req_text},
            ]

    return None


# ---------------------------------------------------------------------------
# Post-planning validation: ensure prerequisites and remove duplicates.
# ---------------------------------------------------------------------------

def _tab_has_content(tab: Any) -> bool:
    """True when a diagram tab holds something the user or the assistant made.

    A new editor project sends every tab with a model that is a non-empty dict
    without content, so a non-empty dict proves nothing. Any element counts:
    this is not ``is_diagram_nontrivial``, which skips small diagrams.
    """
    if not isinstance(tab, dict):
        return False
    # A legacy entry is the model itself; an entry with a "model" key is judged by it.
    model = tab["model"] if "model" in tab else tab
    if not isinstance(model, dict) or not model:
        return False
    if "pages" in model:
        # GUI: the editor's default tab is one page with no components, and
        # its screens are still to be built.
        return any(
            isinstance(frame, dict)
            and isinstance(frame.get("component"), dict)
            and bool(frame["component"].get("components"))
            for page in model.get("pages") or []
            if isinstance(page, dict)
            for frame in page.get("frames") or []
        )
    if "cols" in model:
        # Quantum circuit: any gate counts; 1 marks an empty cell.
        return any(
            not isinstance(column, list) or any(cell not in (1, None) for cell in column)
            for column in model.get("cols") or []
        )
    # UML-family models and the knowledge graph: any element counts.
    content = [model[key] for key in ("elements", "relationships", "nodes", "edges") if key in model]
    return any(content) if content else True


def _get_workspace_diagram_types(request: AssistantRequest) -> Set[str]:
    """Return the set of diagram types already present (with content) in the workspace.

    Judged from the project snapshot: a type is present when one of its tabs
    has content. Diagram summaries carry no content, so they stand in only
    when no snapshot was sent.
    """
    snapshot = request.context.project_snapshot
    diagrams = snapshot.get("diagrams") if isinstance(snapshot, dict) else None
    if isinstance(diagrams, dict):
        return {
            diagram_type
            for diagram_type, payload in diagrams.items()
            if any(
                _tab_has_content(tab)
                for tab in (payload if isinstance(payload, list) else [payload])
            )
        }
    existing: Set[str] = set()
    for summary in (request.context.diagram_summaries or []):
        if isinstance(summary, dict):
            dt = summary.get("diagramType")
            if isinstance(dt, str):
                existing.add(dt)
    return existing


# Words of a generate command that name no subject: the generators and their
# targets, the user's model, pronouns and filler.
_GENERATE_COMMAND_WORDS = frozenset(
    word
    for phrase in [
        *GENERATOR_PREREQUISITES, *GENERATOR_KEYWORDS,
        *(keyword for keywords in GENERATOR_KEYWORDS.values() for keyword in keywords),
    ]
    for word in re.findall(r"[a-z0-9]+", phrase.lower())
) | {
    "a", "an", "the", "my", "our", "this", "that", "these", "those", "it", "them", "its",
    "current", "existing", "same", "whole", "full", "complete", "new",
    "for", "of", "to", "from", "with", "in", "into", "as", "on", "and", "based", "using",
    "model", "models", "diagram", "diagrams", "class", "classes", "uml",
    "code", "source", "files", "file", "download", "now", "again", "please", "me", "us",
}


def _is_bare_generate_command(message: str) -> bool:
    """True for "generate django code for my model": a generate command whose
    words after the verb name only the generator, the user's model or filler."""
    match = _HEURISTIC_GENERATE_ONLY.match(message.strip())
    return bool(match) and set(
        re.findall(r"[a-z0-9]+", match.group("gen_part").lower())
    ) <= _GENERATE_COMMAND_WORDS


def _validate_and_fix_plan(
    operations: List[Dict[str, Any]],
    request: AssistantRequest,
) -> List[Dict[str, Any]]:
    """Post-planning validation pass.

    * Ensures prerequisite diagrams exist for every generation operation
      (either already in the workspace or earlier in the plan).
    * Removes exact duplicate operations.
    """
    workspace_diagrams = _get_workspace_diagram_types(request)

    # Collect diagram types that will be produced by model operations in the plan
    planned_diagrams: Set[str] = set()
    for op in operations:
        if op.get("type") == "model":
            dt = op.get("diagramType")
            if isinstance(dt, str):
                planned_diagrams.add(dt)

    # A bare "generate django code" with no model step names nothing to model:
    # no step is added for it, and the generation handler says what is missing.
    nothing_to_model = not planned_diagrams and _is_bare_generate_command(request.message or "")

    # Check each generation op (and each GUI build) for missing prerequisites
    # and inject them
    injected: List[Dict[str, Any]] = []
    for op in operations:
        if op.get("type") == "generation":
            gen_type = op.get("generatorType")
            if not isinstance(gen_type, str) or nothing_to_model:
                continue
            prereqs = GENERATOR_PREREQUISITES.get(gen_type, [])
        elif op.get("type") == "model":
            # Screens bind to the class diagram; without one the GUI handler
            # invents its own entities and no model exists.
            if op.get("diagramType") != "GUINoCodeDiagram" or op.get("mode") != "complete_system":
                continue
            prereqs = ["ClassDiagram"]
        else:
            continue
        for prereq in prereqs:
            if prereq not in planned_diagrams and prereq not in workspace_diagrams:
                # Build a helpful sub-request from the original user message
                domain_hint = request.message.strip()
                injected.append({
                    "type": "model",
                    "diagramType": prereq,
                    "mode": "complete_system",
                    "request": f"create {prereq} for: {domain_hint}",
                })
                planned_diagrams.add(prereq)

    if injected:
        # Insert injected model ops at the front (before existing ops),
        # but respect ClassDiagram-first ordering.
        all_model_ops = injected + [op for op in operations if op.get("type") == "model"]
        gen_ops = [op for op in operations if op.get("type") == "generation"]
        # Sort model ops: ClassDiagram first
        class_ops = [op for op in all_model_ops if op.get("diagramType") == "ClassDiagram"]
        other_ops = [op for op in all_model_ops if op.get("diagramType") != "ClassDiagram"]
        operations = class_ops + other_ops + gen_ops

    # Remove duplicates (preserve order)
    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for op in operations:
        key = json.dumps(op, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(op)

    return deduped


# ---------------------------------------------------------------------------
# Few-shot examples appended to the LLM planner prompt.
# ---------------------------------------------------------------------------


# Diagrams the from-scratch web-app route builds.
_WEB_APP_ROUTE_DIAGRAMS = {"ClassDiagram", "GUINoCodeDiagram"}

# "a UI", "with a simple GUI", "models and UI", "a frontend for ...": asked for
# after a word like "with" or "build", and not the modifier of another noun
# ("UI defects", "frontend developers", "a GUI toolkit").
_UI_ASKED_FOR_RE = re.compile(
    r"\b(?:with|and|plus|including|add|need|want|build|create|design|make|has|have)\s+"
    r"(?:[\w-]+\s+){0,2}?"
    r"(?:gui|ui|user\s+interface|front-?end|mockups?|wireframes?)\b"
    r"(?!\s+(?!(?:for|to|that|which|where|with|and|or|so|in|on|plus|please|too)\b)[\w-])",
    re.IGNORECASE,
)
# "pages" and "screens" are also domain nouns (a book's pages, a cinema's
# screens). They ask for an interface only when requested as one: "with
# screens", "pages for recipes", "a page to ...", "a detail page", "with an
# order page". Not followed by a comma: "we want screens, nails and hinges".
_SCREENS_AS_INTERFACE_RE = re.compile(
    r"\b(?:with|and|plus|including|design|build|create|make|add|need|want)\s+"
    r"(?:(?:a|an|the|some|few|several|simple|basic|\d+|two|three|four|five)\s+){0,2}screens\b(?!\s*,)"
    r"|\b(?:screens?|pages?)\s+(?:"
    r"for\s+\w+"
    r"|per\s+\w+"
    r"|where\b"
    r"|to\s+(?!(?:the|a|an|my|our|their|its|this|that|each|every)\b)\w+"
    r"|that\s+(?:list|show|display|let)s?\b)"
    r"|\b(?:home|landing|login|sign-?up|dashboard|detail|list|overview|admin|settings|main|start|search)"
    r"\s+(?:screens?|pages?)\b"
    r"|\b(?:with|and|plus|including)\s+(?:(?:a|an|the|one)\s+)?(?!\d)[\w-]+\s+(?:screens?|pages?)\b",
    re.IGNORECASE,
)
# A request for a model: "model a ...", "a domain/data/class model", "a class
# diagram", "a library model: ...".
_ASKS_FOR_A_MODEL_RE = re.compile(
    r"\bmodel\s+(?:a|an|the|my|our)\b|\b(?:domain|data|class)\s+model\b|\bclass\s+diagram\b|\bmodel\s*:",
    re.IGNORECASE,
)
# In a request for a model, screens are asked for explicitly: "and screens for
# it", "with pages to book ...", "with screens".
_SCREENS_ASKED_FOR_EXPLICITLY_RE = re.compile(
    r"\b(?:with|and|plus|including|also)\s+(?:(?:a|an|the|some|simple|basic)\s+)?(?:screens?|pages?)\b"
    r"(?=\s*(?:$|[.!?;]|for\s+(?:it|them|this|that|each|every|all)\b"
    r"|to\s+(?!(?:the|a|an|my|our|their|its|this|that|each|every)\b)\w+|where\b|that\b))",
    re.IGNORECASE,
)


def _asks_for_an_interface(message: str) -> bool:
    """True when the message asks for screens, not merely uses a screen word."""
    if _UI_ASKED_FOR_RE.search(message):
        return True
    if _ASKS_FOR_A_MODEL_RE.search(message):
        return bool(_SCREENS_ASKED_FOR_EXPLICITLY_RE.search(message))
    return bool(_SCREENS_AS_INTERFACE_RE.search(message))


def _without_web_gui_plan(
    operations: List[Dict[str, Any]], request: AssistantRequest, default_mode: str,
    keep_gui: bool,
) -> List[Dict[str, Any]]:
    """Drop the screens + ``web_app`` steps from a plan for a non-web target.

    The GUI editor designs web pages, so a native / desktop / CLI app is built
    from the class diagram by the smart generator instead, which the user
    starts after reviewing the model. ``keep_gui`` keeps the GUI diagram when
    it is what the user asked for.
    """
    kept = [
        op for op in operations
        if not (op.get("type") == "generation" and op.get("generatorType") == "web_app")
        and not (not keep_gui and op.get("type") == "model"
                 and op.get("diagramType") == "GUINoCodeDiagram"
                 and op.get("mode") == "complete_system")
    ]
    if len(kept) < len(operations):
        logger.info("Non-web target: removed the GUI diagram / web_app steps from the plan")
    if kept or not operations:
        return kept
    return [{"type": "model", "diagramType": "ClassDiagram", "mode": default_mode,
             "request": request.message}]


def plan_assistant_operations(
    request: AssistantRequest,
    default_mode: str,
    matched_intent: Optional[str],
    llm_predict: Callable[[str], str],
    llm_target_type: Optional[str] = None,
    app_target_kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build an ordered operation plan for the assistant.

    Returns operations shaped as:
    - {"type":"model","diagramType":"...","mode":"complete_system|modify_model","request":"..."}
    - {"type":"generation","generatorType":"...","config":{...}}

    ``app_target_kind`` is the classifier's target for this project
    (``web_app`` / ``backend`` / ``other``); ``other`` never adds the web GUI
    on its own, but keeps it when the classifier's target is the GUI diagram.
    """
    operations = _plan_operations(
        request, default_mode, matched_intent, llm_predict, llm_target_type, app_target_kind,
    )
    if app_target_kind == "other":
        return _without_web_gui_plan(
            operations, request, default_mode,
            keep_gui=llm_target_type == "GUINoCodeDiagram",
        )
    return operations


def _plan_operations(
    request: AssistantRequest,
    default_mode: str,
    matched_intent: Optional[str],
    llm_predict: Callable[[str], str],
    llm_target_type: Optional[str],
    app_target_kind: Optional[str],
) -> List[Dict[str, Any]]:

    # ----- Phase 0: fast heuristic pre-decomposition -----
    heuristic_result = _try_heuristic_decomposition(request.message, request, default_mode)
    if heuristic_result is not None:
        normalized = _normalize_operations(heuristic_result, request=request, default_mode=default_mode)
        if normalized:
            validated = _validate_and_fix_plan(normalized, request)
            logger.debug("Heuristic planner produced %d operations", len(validated))
            return validated

    # ----- Phase 0b: from-scratch app that asks for a UI -----
    # "Build a library platform with models and UI" takes the same route as
    # "create a web app for X". Left to the classifier, the target flips
    # between ClassDiagram (no screens) and GUINoCodeDiagram (screens on no
    # model). A request for any other diagram (the classifier's target, a
    # diagram the message names or, without a target, one it implies) keeps
    # its own plan.
    if (
        matched_intent == "create_complete_system_intent"
        and app_target_kind != "other"
        and _asks_for_an_interface(request.message or "")
        and set(determine_target_diagram_types(
            request, last_intent=matched_intent, max_targets=len(SUPPORTED_DIAGRAM_TYPES),
            llm_target_type=llm_target_type,
        )) <= _WEB_APP_ROUTE_DIAGRAMS
        and not _WEB_APP_ROUTE_DIAGRAMS & _get_workspace_diagram_types(request)
    ):
        message = request.message.strip()
        return _validate_and_fix_plan([
            {"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system",
             "request": (
                 f"create a class diagram for: {message}\n"
                 "Model the domain data only. The screens are built in the next "
                 "step, so add no UI, screen or view classes."
             )},
            {"type": "model", "diagramType": "GUINoCodeDiagram", "mode": "complete_system",
             "request": f"create a GUI for: {message}"},
            {"type": "generation", "generatorType": "web_app", "config": {}},
        ], request)

    # ----- Phase 1: keyword-based fallback -----
    fallback = _fallback_operations(
        request, default_mode=default_mode, matched_intent=matched_intent,
        llm_target_type=llm_target_type,
    )
    inferred_targets = determine_target_diagram_types(
        request, last_intent=matched_intent, max_targets=6, llm_target_type=llm_target_type,
    )
    # Cache detect_generator_type — called once and reused
    detected_gen = detect_generator_type(request.message)
    has_generation_request = detected_gen is not None

    if not _should_use_llm_planner(request.message, len(inferred_targets), has_generation_request, matched_intent=matched_intent):
        return _validate_and_fix_plan(fallback, request)

    # ----- Phase 2: LLM planner -----
    context_summary = _build_context_summary(request)

    # Give the LLM strong hints about which diagram types were detected
    detected_targets_hint = ", ".join(inferred_targets) if inferred_targets else "ClassDiagram"
    generation_hint = ""
    if detected_gen:
        generation_hint = f"\nDetected generation request: {detected_gen}"
        prereqs = GENERATOR_PREREQUISITES.get(detected_gen)
        if prereqs:
            generation_hint += f"\nPrerequisite diagrams for {detected_gen}: {', '.join(prereqs)}"

    # A1: Inject matched intent so the planner knows the classified intent
    intent_hint = ""
    if matched_intent:
        intent_hint = f"\nUser intent classified as: {matched_intent}"
    if app_target_kind == "other" and llm_target_type != "GUINoCodeDiagram":
        intent_hint += planner_prompts.NON_WEB_APP_HINT

    planner_prompt = planner_prompts.planner_prompt(request.message, context_summary, intent_hint, detected_targets_hint, generation_hint, json.dumps(GENERATOR_PREREQUISITES, indent=2))

    try:
        raw_response = llm_predict(planner_prompt)
        cleaned = _clean_json_response(raw_response)
        parsed = json.loads(cleaned)
        operations = parsed.get("operations") if isinstance(parsed, dict) else None
        normalized = _normalize_operations(operations, request=request, default_mode=default_mode)
        if normalized:
            validated = _validate_and_fix_plan(normalized, request)
            return validated
    except Exception as error:
        logger.warning("Planner JSON parsing failed, using fallback operations: %s", error)

    return _validate_and_fix_plan(fallback, request)
