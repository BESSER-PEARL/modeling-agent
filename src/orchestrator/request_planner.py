import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from handlers.generation_handler import GENERATOR_KEYWORDS, detect_generator_type
from protocol.types import AssistantRequest, SUPPORTED_DIAGRAM_TYPES

from .workspace_orchestrator import KEYWORD_TARGETS, determine_target_diagram_types

logger = logging.getLogger(__name__)

# Re-export canonical set for backward compat; prefer SUPPORTED_DIAGRAM_TYPES.
ALLOWED_DIAGRAM_TYPES: Set[str] = SUPPORTED_DIAGRAM_TYPES

ALLOWED_MODEL_MODES: Set[str] = {
    "single_element",
    "complete_system",
    "modify_model",
}

ALLOWED_GENERATORS: Set[str] = set(GENERATOR_KEYWORDS.keys())

# Maps generator type → diagram types that must exist before generation can run.
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
    "rest_api": ["ClassDiagram"],
    "agent": ["AgentDiagram"],
    "qiskit": ["QuantumCircuitDiagram"],
}

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

    snapshot = context.project_snapshot
    if isinstance(snapshot, dict):
        project_name = snapshot.get("name")
        if isinstance(project_name, str) and project_name.strip():
            lines.append(f"Project name: {project_name.strip()}")

        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            summarized: List[str] = []
            for diagram_type, diagram_payload in diagrams.items():
                if not isinstance(diagram_payload, dict):
                    continue
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
    segment_lower = segment.lower()
    candidates: List[Tuple[int, str]] = []
    for token, diagram_type in KEYWORD_TARGETS:
        index = segment_lower.find(token)
        if index >= 0:
            candidates.append((index, diagram_type))
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


def _should_use_llm_planner(message: str, inferred_target_count: int, has_generation_request: bool) -> bool:
    lower = (message or "").lower()
    has_connector = any(connector in lower for connector in PLANNER_CONNECTORS)
    has_multi_clause = has_connector or lower.count(",") >= 2
    has_explicit_diagram_tokens = any(token in lower for token, _ in KEYWORD_TARGETS)

    if has_multi_clause and (inferred_target_count > 1 or has_generation_request):
        return True

    # Let the LLM planner split ambiguous multi-step prompts that mention both modeling and generation,
    # even if keyword inference only produced one target.
    if has_generation_request and has_multi_clause:
        return True

    # If the prompt is complex and diagram targeting is implicit, planner usually yields cleaner sub-requests.
    if has_multi_clause and not has_explicit_diagram_tokens and len(lower) > 120:
        return True

    return False


# Modeling intents that should never trigger an automatic generation operation.
_MODELING_INTENTS = {
    "create_single_element_intent",
    "create_complete_system_intent",
    "modify_model_intent",
    "modeling_help_intent",
    "describe_model_intent",
}


def _fallback_operations(
    request: AssistantRequest,
    default_mode: str,
    matched_intent: Optional[str],
) -> List[Dict[str, Any]]:
    targets = determine_target_diagram_types(request, last_intent=matched_intent, max_targets=3)

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


def plan_assistant_operations(
    request: AssistantRequest,
    default_mode: str,
    matched_intent: Optional[str],
    llm_predict: Callable[[str], str],
) -> List[Dict[str, Any]]:
    """
    Build an ordered operation plan for the assistant.

    Returns operations shaped as:
    - {"type":"model","diagramType":"...","mode":"single_element|complete_system|modify_model","request":"..."}
    - {"type":"generation","generatorType":"...","config":{...}}
    """
    fallback = _fallback_operations(request, default_mode=default_mode, matched_intent=matched_intent)
    inferred_targets = determine_target_diagram_types(request, last_intent=matched_intent, max_targets=6)
    # Cache detect_generator_type — called once and reused
    detected_gen = detect_generator_type(request.message)
    has_generation_request = detected_gen is not None

    if not _should_use_llm_planner(request.message, len(inferred_targets), has_generation_request):
        return fallback

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

    planner_prompt = f"""You are an assistant operation planner for BESSER modeling.

User request:
{request.message}

Workspace context:
{context_summary}{intent_hint}

Detected diagram targets (from user message keywords): {detected_targets_hint}{generation_hint}

Create a JSON plan with an "operations" array.
Operation types:
1) model:
{{
  "type": "model",
  "diagramType": "ClassDiagram|ObjectDiagram|StateMachineDiagram|AgentDiagram|GUINoCodeDiagram|QuantumCircuitDiagram",
  "mode": "single_element|complete_system|modify_model",
  "request": "sub-request focused ONLY on this specific diagram type"
}}
2) generation:
{{
  "type": "generation",
  "generatorType": "django|backend|web_app|sql|sqlalchemy|python|java|pydantic|jsonschema|smartdata|agent|qiskit",
  "config": {{}}
}}

Generator prerequisites (the planner must ensure these diagrams are created BEFORE the generation step):
{json.dumps(GENERATOR_PREREQUISITES, indent=2)}

Rules:
- Emit one model operation for EACH detected diagram target, in dependency order.
- ClassDiagram always comes first — other diagrams depend on it.
- Each model operation's "request" should be a focused sub-request for that specific diagram only.
  IMPORTANT: Each sub-request MUST contain enough detail from the original request so the
  handler knows what to generate. Don't just say "create a class diagram" — include the
  domain details like "create a class diagram for a library system with books, authors, and members".
- If the user asks for generation, emit a generation operation AFTER all required model operations.
- If prerequisite diagrams are missing from the workspace, create them first.
- For complex multi-step requests like "create X system then generate Y", decompose into:
  1. model operation (create the diagram)
  2. generation operation (generate the code)
- If the user says "create a web app for X", this means:
  1. Create ClassDiagram for X (complete_system)
  2. Create GUINoCodeDiagram for X (complete_system)
  3. Generate web_app code
- Keep operations minimal and deterministic.
- Return ONLY valid JSON: {{"operations":[...]}}.
"""

    try:
        raw_response = llm_predict(planner_prompt)
        cleaned = _clean_json_response(raw_response)
        parsed = json.loads(cleaned)
        operations = parsed.get("operations") if isinstance(parsed, dict) else None
        normalized = _normalize_operations(operations, request=request, default_mode=default_mode)
        if normalized:
            return normalized
    except Exception as error:
        logger.warning("Planner JSON parsing failed, using fallback operations: %s", error)

    return fallback
