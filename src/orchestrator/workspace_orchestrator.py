import re
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
    ("grover", "QuantumCircuitDiagram"),
    ("shor's", "QuantumCircuitDiagram"),
    ("shor algorithm", "QuantumCircuitDiagram"),
    ("bell state", "QuantumCircuitDiagram"),
    ("hadamard", "QuantumCircuitDiagram"),
    ("cnot", "QuantumCircuitDiagram"),
    ("qubit", "QuantumCircuitDiagram"),
    ("qft", "QuantumCircuitDiagram"),
    ("entangle", "QuantumCircuitDiagram"),
    ("superposition", "QuantumCircuitDiagram"),
    ("quantum gate", "QuantumCircuitDiagram"),
    ("bernstein-vazirani", "QuantumCircuitDiagram"),
    ("deutsch-jozsa", "QuantumCircuitDiagram"),
    ("teleportation circuit", "QuantumCircuitDiagram"),
]


# ---------------------------------------------------------------------------
# Discriminating pattern rules — replacement for the old additive-weight
# IMPLICIT_TARGET_RULES.
#
# Each rule is a (diagram_type, compiled_regex) pair.  Patterns use
# AND-based logic: they require at least one *strong, discriminating*
# signal that unambiguously points to a diagram type.  Generic words
# like "system", "model", "application" are intentionally excluded —
# if none of the patterns match, we fall through to the context fallback
# (active diagram) rather than guessing wrong.
#
# Rules are checked in order; the FIRST match wins (most specific first).
# ---------------------------------------------------------------------------

_IMPLICIT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # ── Quantum (very specific vocabulary — match first) ──
    # Any single quantum-specific term is sufficient.
    ("QuantumCircuitDiagram", re.compile(
        r"\b(?:quantum|qubits?|qiskit|grover|shor|hadamard|cnot|superposition"
        r"|entangl\w*|qft|teleportation|bell\s*state"
        r"|bernstein|deutsch|gates?)\b", re.I)),

    # ── Object Diagram ──
    ("ObjectDiagram", re.compile(
        r"\b(?:object\s*instances?|instances?\s+of|runtime\s+objects?|instances)\b", re.I)),

    # ── State Machine (requires state-specific vocabulary) ──
    # Either a strong standalone signal (lifecycle, workflow state, transition)
    # or "state(s)" co-occurring with transition/flow/event/process.
    ("StateMachineDiagram", re.compile(
        r"\b(?:lifecycle|workflow\s*states?"
        r"|transitions?\b.{0,40}\b(?:states?|status)"
        r"|(?:states?|status)\b.{0,40}\b(?:transitions?|flows?|events?|process))\b", re.I)),

    # ── Agent Diagram ──
    ("AgentDiagram", re.compile(
        r"\b(?:multi[- ]?agents?|conversational\s+agents?|chatbots?"
        r"|agents?\b.{0,30}\b(?:intents?|training|reply|response)"
        r"|(?:intents?|training\s+phrases?)\b.{0,30}\b(?:agents?))\b", re.I)),

    # ── GUI / Frontend (must have explicit GUI vocabulary) ──
    ("GUINoCodeDiagram", re.compile(
        r"\b(?:gui|user\s*interface|wireframes?|no[- ]?code|grapesjs"
        r"|(?:frontend|screens?|pages?|layouts?|dashboards?)\b.{0,30}\b(?:design|create|build|diagram)"
        r"|(?:create|build|design)\b.{0,30}\b(?:frontend|screens?|pages?|layouts?))\b", re.I)),

    # ── Class Diagram (structural vocabulary — checked last among specifics) ──
    # Either a strong standalone signal (structural, domain model) or
    # class/entity co-occurring with attribute/method/relationship.
    ("ClassDiagram", re.compile(
        r"\b(?:structural|domain\s+model|business\s+model|system\s+model"
        r"|(?:class(?:es)?|entit(?:y|ies))\b.{0,40}\b(?:attributes?|methods?|relationships?|associations?|inheritance)"
        r"|(?:attributes?|methods?|relationships?|associations?|inheritance)\b.{0,40}\b(?:class(?:es)?|entit(?:y|ies)))\b",
        re.I)),
]

# Backward-compatible alias — some imports reference this name.
# Keep the old dict shape so nothing breaks at import time, but mark
# deprecated.  The actual scoring function `_rank_implicit_targets` now
# uses `_IMPLICIT_PATTERNS` instead.
IMPLICIT_TARGET_RULES: Dict[str, List[Tuple[str, int]]] = {
    "ClassDiagram": [("structural", 5), ("domain model", 5), ("class", 4)],
    "ObjectDiagram": [("object instance", 5), ("instances", 4)],
    "StateMachineDiagram": [("lifecycle", 5), ("transition", 4), ("state", 3)],
    "AgentDiagram": [("multi-agent", 5), ("agent", 4), ("intent", 4)],
    "GUINoCodeDiagram": [("gui", 4), ("user interface", 5), ("frontend", 3)],
    "QuantumCircuitDiagram": [("quantum", 6), ("qubit", 5), ("grover", 5)],
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
    """Match the message against discriminating patterns.

    Unlike the old additive-weight system, each pattern uses AND-based
    logic — it requires a *strong, unambiguous* signal (e.g. "lifecycle"
    alone, or "state" + "transition" together).  Generic words like
    "system" or "model" alone will NOT produce a match; the caller will
    fall through to the context-based fallback instead of guessing.

    Returns matched diagram types in pattern-priority order (most specific
    first).  Typically returns 0 or 1 results; multiple results are
    possible when the message mentions vocabulary from several diagram types.
    """
    matched: List[str] = []
    seen: Set[str] = set()
    for diagram_type, pattern in _IMPLICIT_PATTERNS:
        if diagram_type in seen:
            continue
        if pattern.search(message_lower):
            matched.append(diagram_type)
            seen.add(diagram_type)
    return matched


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
                payload = diagrams.get(preferred)
                if payload is None:
                    continue
                if isinstance(payload, list):
                    # Multi-tab: only count as present when at least one tab has content
                    if any(isinstance(d, dict) and isinstance(d.get("model"), dict) and d["model"] for d in payload):
                        return preferred
                elif isinstance(payload, dict):
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
    if isinstance(target_diagram, list):
        # Multi-tab: use the active index for this type, fall back to first entry.
        idx = (
            request.context.get_active_index(target_diagram_type)
            if hasattr(request.context, "get_active_index")
            else 0
        )
        # Try the active index first.
        if 0 <= idx < len(target_diagram) and isinstance(target_diagram[idx], dict):
            diagram_id = target_diagram[idx].get("id")
            if isinstance(diagram_id, str):
                return diagram_id
        # Fallback: return the id of the first dict entry found.
        for d in target_diagram:
            if isinstance(d, dict):
                diagram_id = d.get("id")
                if isinstance(diagram_id, str):
                    return diagram_id
    elif isinstance(target_diagram, dict):
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
