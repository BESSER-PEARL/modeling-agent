from typing import Any, Dict, List, Optional, Set, Tuple

from protocol.types import AssistantRequest, SUPPORTED_DIAGRAM_TYPES

KEYWORD_TARGETS = [
    # Class / Structural
    ("class diagram", "ClassDiagram"),
    ("class model", "ClassDiagram"),
    ("structural model", "ClassDiagram"),
    ("structural diagram", "ClassDiagram"),
    ("the structural", "ClassDiagram"),
    ("domain model", "ClassDiagram"),
    # Object
    ("object diagram", "ObjectDiagram"),
    ("object model", "ObjectDiagram"),
    # State Machine
    ("state machine", "StateMachineDiagram"),
    ("statemachine", "StateMachineDiagram"),
    ("state diagram", "StateMachineDiagram"),
    # Agent
    ("agent diagram", "AgentDiagram"),
    ("agent model", "AgentDiagram"),
    ("agent that", "AgentDiagram"),
    ("an agent", "AgentDiagram"),
    ("chatbot", "AgentDiagram"),
    # GUI
    ("gui diagram", "GUINoCodeDiagram"),
    ("graphical ui", "GUINoCodeDiagram"),
    ("web ui", "GUINoCodeDiagram"),
    ("the gui", "GUINoCodeDiagram"),
    ("a gui", "GUINoCodeDiagram"),
    ("gui generated", "GUINoCodeDiagram"),
    # Quantum
    ("quantum circuit", "QuantumCircuitDiagram"),
    ("quantum diagram", "QuantumCircuitDiagram"),
    ("quantum", "QuantumCircuitDiagram"),
    ("qiskit", "QuantumCircuitDiagram"),
]

IMPLICIT_TARGET_RULES: Dict[str, List[Tuple[str, int]]] = {
    "ClassDiagram": [
        ("structural", 5),
        ("domain model", 5),
        ("entity", 4),
        ("entities", 4),
        ("class", 4),
        ("attribute", 3),
        ("method", 3),
        ("relationship", 3),
        ("association", 3),
        ("inheritance", 3),
        ("business model", 3),
        ("system model", 3),
        ("model", 1),
        ("system", 1),
        ("application", 1),
        ("platform", 1),
    ],
    "ObjectDiagram": [
        ("object instance", 5),
        ("instances", 4),
        ("instance of", 4),
        ("runtime object", 4),
        ("snapshot", 3),
    ],
    "StateMachineDiagram": [
        ("lifecycle", 5),
        ("workflow state", 5),
        ("transition", 4),
        ("event", 3),
        ("state", 3),
        ("status", 3),
        ("flow", 3),
        ("process", 3),
        ("journey", 2),
    ],
    "AgentDiagram": [
        ("multi-agent", 5),
        ("conversational agent", 5),
        ("agent", 4),
        ("intent", 4),
        ("training phrase", 3),
        ("reply", 3),
        ("assistant", 2),
        ("chatbot", 2),
    ],
    "GUINoCodeDiagram": [
        ("gui", 4),
        ("ui", 4),
        ("user interface", 5),
        ("screen", 4),
        ("page", 4),
        ("dashboard", 4),
        ("form", 3),
        ("layout", 3),
        ("frontend", 3),
        ("wireframe", 3),
    ],
    "QuantumCircuitDiagram": [
        ("quantum", 6),
        ("qiskit", 6),
        ("qubit", 5),
        ("quantum circuit", 5),
        ("gate", 3),
        ("entanglement", 3),
    ],
}

FALLBACK_PRIORITY: Tuple[str, ...] = (
    "ClassDiagram",
    "ObjectDiagram",
    "StateMachineDiagram",
    "AgentDiagram",
    "GUINoCodeDiagram",
    "QuantumCircuitDiagram",
)


def _collect_explicit_targets(message_lower: str) -> List[str]:
    explicit: List[Tuple[int, str]] = []
    seen: Set[str] = set()
    for token, diagram_type in KEYWORD_TARGETS:
        index = message_lower.find(token)
        if index >= 0 and diagram_type not in seen:
            explicit.append((index, diagram_type))
            seen.add(diagram_type)
    explicit.sort(key=lambda item: item[0])
    return [diagram_type for _, diagram_type in explicit]


def _rank_implicit_targets(message_lower: str) -> List[str]:
    ranked: List[Tuple[int, int, str]] = []

    for diagram_type, rules in IMPLICIT_TARGET_RULES.items():
        score = 0
        first_index = 10**9
        for token, weight in rules:
            index = message_lower.find(token)
            if index >= 0:
                score += weight
                first_index = min(first_index, index)

        if score > 0:
            ranked.append((score, first_index, diagram_type))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [diagram_type for _, _, diagram_type in ranked]


def _normalize_context_type(diagram_type: Optional[str]) -> Optional[str]:
    if isinstance(diagram_type, str) and diagram_type in SUPPORTED_DIAGRAM_TYPES:
        return diagram_type
    return None


def _fallback_diagram_from_context(request: AssistantRequest, last_intent: Optional[str]) -> str:
    # For modify requests, staying on active diagram is generally the safest fallback.
    if last_intent == "modify_model_intent":
        active_type = _normalize_context_type(request.context.active_diagram_type)
        if active_type:
            return active_type

    active_type = _normalize_context_type(request.context.active_diagram_type)
    if active_type:
        return active_type

    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            for preferred in FALLBACK_PRIORITY:
                if preferred in diagrams:
                    return preferred

    request_type = _normalize_context_type(request.diagram_type)
    if request_type:
        return request_type

    summaries = request.context.diagram_summaries or []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        summary_type = _normalize_context_type(item.get("diagramType"))
        if summary_type:
            return summary_type

    # Last-resort fallback if client context is empty.
    return "ClassDiagram"


def determine_target_diagram_types(
    request: AssistantRequest,
    last_intent: Optional[str] = None,
    max_targets: int = 3,
) -> List[str]:
    """
    Resolve one or more diagram targets for a user message.

    Priority:
    1. Explicit diagram references in the prompt (ordered by first appearance)
    2. Implicit semantic hints (scored keyword rules)
    3. Active diagram fallback
    """
    message_lower = (request.message or "").lower()
    explicit_targets = _collect_explicit_targets(message_lower)
    if explicit_targets:
        return explicit_targets[:max_targets]

    implicit_targets = _rank_implicit_targets(message_lower)
    if implicit_targets:
        return implicit_targets[:max_targets]

    fallback = _fallback_diagram_from_context(request, last_intent=last_intent)
    return [fallback]


def determine_target_diagram_type(request: AssistantRequest, last_intent: Optional[str] = None) -> str:
    """
    Resolve a single primary diagram target for the current user message.
    """
    targets = determine_target_diagram_types(request, last_intent=last_intent, max_targets=1)
    return targets[0] if targets else _fallback_diagram_from_context(request, last_intent=last_intent)


def resolve_diagram_id(request: AssistantRequest, target_diagram_type: str) -> Optional[str]:
    if target_diagram_type == request.context.active_diagram_type and request.context.active_diagram_id:
        return request.context.active_diagram_id

    snapshot = request.context.project_snapshot
    if not isinstance(snapshot, dict):
        return None

    diagrams = snapshot.get("diagrams")
    if not isinstance(diagrams, dict):
        return None

    target_diagram = diagrams.get(target_diagram_type)
    if isinstance(target_diagram, dict):
        diagram_id = target_diagram.get("id")
        if isinstance(diagram_id, str):
            return diagram_id
    return None


def build_switch_diagram_action(target_diagram_type: str, reason: str = "") -> Dict[str, Any]:
    return {
        "action": "switch_diagram",
        "diagramType": target_diagram_type,
        "reason": reason or f"Switching to {target_diagram_type} based on your request.",
    }
