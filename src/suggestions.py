"""Contextual suggestion engine for the modeling assistant.

Generates relevant next-step actions based on what the user just did,
what diagrams exist, and what's available.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Per-diagram-type suggestion pools
# ------------------------------------------------------------------

# Each entry: (label, prompt)
# Pools are ordered by relevance; the engine picks from these based on
# context and available diagrams.

_CLASS_DIAGRAM_COMPLETE = [
    ("Generate Python code", "generate python"),
    ("Generate Django backend", "generate django"),
    ("Create a GUI for this system", "create a gui for this system"),
    ("Add a state machine", "create a state machine for this system"),
]

_CLASS_DIAGRAM_SINGLE = [
    ("Add another class", "create another class"),
    ("Add a relationship", "add a relationship between classes"),
    ("Generate Python code", "generate python"),
]

_CLASS_DIAGRAM_MODIFY = [
    ("Generate Python code", "generate python"),
    ("Describe my diagram", "describe my diagram"),
    ("Make another change", ""),
]

_STATE_MACHINE_SUGGESTIONS = [
    ("Generate Python code", "generate python"),
    ("Add more states", "add a new state"),
    ("Describe my state machine", "describe my state machine"),
]

_GUI_SUGGESTIONS_WITH_CLASS = [
    ("Generate web app", "generate web app"),
    ("Generate React frontend", "generate react"),
    ("Modify the GUI", ""),
]

_GUI_SUGGESTIONS_WITHOUT_CLASS = [
    ("Create the backend model", "create a class diagram for this GUI"),
    ("Modify the GUI", ""),
]

_AGENT_SUGGESTIONS = [
    ("Generate agent code", "generate agent"),
    ("Add more intents", "add a new intent"),
    ("Describe my agent", "describe my agent"),
]

_OBJECT_DIAGRAM_SUGGESTIONS = [
    ("Modify the object diagram", ""),
    ("Describe my object diagram", "describe my object diagram"),
    ("Generate Python code", "generate python"),
]

_QUANTUM_SUGGESTIONS = [
    ("Generate Qiskit code", "generate qiskit"),
    ("Add more gates", "add more quantum gates"),
    ("Describe my circuit", "describe my quantum circuit"),
]

_GENERATION_SUGGESTIONS = [
    ("Generate another format", "generate sql"),
    ("Modify the model", ""),
    ("Describe my diagram", "describe my diagram"),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _has_diagram(available_diagrams: Optional[List[str]], diagram_type: str) -> bool:
    """Check whether a specific diagram type exists in the workspace."""
    if not available_diagrams:
        return False
    return diagram_type in available_diagrams


def _build_actions(candidates: List[tuple], limit: int = 4) -> List[Dict[str, str]]:
    """Convert (label, prompt) tuples into action dicts, capped at *limit*."""
    actions: List[Dict[str, str]] = []
    for label, prompt in candidates[:limit]:
        actions.append({"label": label, "prompt": prompt})
    return actions


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_suggested_actions(
    diagram_type: str,
    operation_mode: str,
    available_diagrams: Optional[List[str]] = None,
    model_summary: Optional[Dict[str, Any]] = None,
    generator_type: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return a list of suggested next actions.

    Each action is a dict with:
      - ``label``: Human-readable display text
      - ``prompt``: Message the user can send (empty string for free-form)

    Returns 2-4 of the most relevant suggestions based on the diagram
    type, what operation just completed, and which diagrams already
    exist in the project.

    Context-aware: when ``model_summary`` includes element names, suggestions
    reference actual classes/states by name for a more personalized UX.

    Args:
        diagram_type: The diagram type that was just operated on.
        operation_mode: The mode of the operation that just completed.
        available_diagrams: List of diagram type strings in the project.
        model_summary: Optional dict with model metadata for context-aware suggestions.
        generator_type: If the last action was code generation, the generator used.
    """
    if not diagram_type and not generator_type:
        return []

    # --- Post-generation suggestions ---
    if operation_mode == "generation" or generator_type:
        return _suggestions_after_generation(
            generator_type, available_diagrams,
        )

    # --- Context-aware suggestions using model content ---
    if model_summary:
        context_suggestions = _context_aware_suggestions(
            diagram_type, operation_mode, model_summary, available_diagrams,
        )
        if context_suggestions:
            return context_suggestions

    # --- Diagram-specific suggestions ---
    handler = _DIAGRAM_SUGGESTION_HANDLERS.get(diagram_type)
    if handler is not None:
        return handler(operation_mode, available_diagrams)

    # Fallback: generic next-step suggestions
    return _build_actions([
        ("Generate Python code", "generate python"),
        ("Describe my diagram", "describe my diagram"),
    ])


# ------------------------------------------------------------------
# Diagram-specific suggestion builders
# ------------------------------------------------------------------

def _suggestions_for_class_diagram(
    operation_mode: str,
    available_diagrams: Optional[List[str]],
) -> List[Dict[str, str]]:
    if operation_mode == "single_element":
        return _build_actions(_CLASS_DIAGRAM_SINGLE)
    if operation_mode == "modify_model":
        return _build_actions(_CLASS_DIAGRAM_MODIFY)
    # complete_system or other
    return _build_actions(_CLASS_DIAGRAM_COMPLETE)


def _suggestions_for_state_machine(
    operation_mode: str,
    available_diagrams: Optional[List[str]],
) -> List[Dict[str, str]]:
    return _build_actions(_STATE_MACHINE_SUGGESTIONS)


def _suggestions_for_gui(
    operation_mode: str,
    available_diagrams: Optional[List[str]],
) -> List[Dict[str, str]]:
    if _has_diagram(available_diagrams, "ClassDiagram"):
        return _build_actions(_GUI_SUGGESTIONS_WITH_CLASS)
    return _build_actions(_GUI_SUGGESTIONS_WITHOUT_CLASS)


def _suggestions_for_agent(
    operation_mode: str,
    available_diagrams: Optional[List[str]],
) -> List[Dict[str, str]]:
    return _build_actions(_AGENT_SUGGESTIONS)


def _suggestions_for_object_diagram(
    operation_mode: str,
    available_diagrams: Optional[List[str]],
) -> List[Dict[str, str]]:
    return _build_actions(_OBJECT_DIAGRAM_SUGGESTIONS)


def _suggestions_for_quantum(
    operation_mode: str,
    available_diagrams: Optional[List[str]],
) -> List[Dict[str, str]]:
    return _build_actions(_QUANTUM_SUGGESTIONS)


_DIAGRAM_SUGGESTION_HANDLERS = {
    "ClassDiagram": _suggestions_for_class_diagram,
    "StateMachineDiagram": _suggestions_for_state_machine,
    "GUINoCodeDiagram": _suggestions_for_gui,
    "AgentDiagram": _suggestions_for_agent,
    "ObjectDiagram": _suggestions_for_object_diagram,
    "QuantumCircuitDiagram": _suggestions_for_quantum,
}


# ------------------------------------------------------------------
# Post-generation suggestions
# ------------------------------------------------------------------

def _suggestions_after_generation(
    generator_type: Optional[str],
    available_diagrams: Optional[List[str]],
) -> List[Dict[str, str]]:
    """Suggestions shown after a code-generation step."""
    candidates: List[tuple] = []

    # Suggest a different generator than the one just used
    alt_generators = {
        "python": ("Generate SQL", "generate sql"),
        "django": ("Generate SQLAlchemy", "generate sqlalchemy"),
        "sql": ("Generate Python code", "generate python"),
        "sqlalchemy": ("Generate Django", "generate django"),
        "react": ("Generate Python code", "generate python"),
        "qiskit": ("Add more gates", "add more quantum gates"),
        "agent": ("Add more intents", "add a new intent"),
    }
    gen_key = (generator_type or "").lower()
    alt = alt_generators.get(gen_key)
    if alt:
        candidates.append(("Generate another format", alt[1]))
    else:
        candidates.append(("Generate another format", "generate sql"))

    candidates.append(("Modify the model", ""))
    candidates.append(("Describe my diagram", "describe my diagram"))

    return _build_actions(candidates)


# ------------------------------------------------------------------
# Formatting helper for text messages
# ------------------------------------------------------------------

def _context_aware_suggestions(
    diagram_type: str,
    operation_mode: str,
    model_summary: Dict[str, Any],
    available_diagrams: Optional[List[str]],
) -> List[Dict[str, str]]:
    """Generate context-aware suggestions using actual model content.

    When element names are available from the model summary, suggestions
    reference them by name for a personalized experience.
    """
    element_names = model_summary.get("element_names", [])
    element_count = model_summary.get("element_count", 0)
    relationship_count = model_summary.get("relationship_count", 0)

    if not element_names:
        return []  # Fall through to static suggestions

    candidates: List[tuple] = []

    if diagram_type == "ClassDiagram":
        # Suggest adding attributes to a specific class
        if element_names and operation_mode == "complete_system":
            first_class = element_names[0]
            candidates.append(
                (f"Add attributes to {first_class}", f"add more attributes to {first_class}")
            )
        # If no relationships, suggest adding some
        if relationship_count == 0 and element_count > 1:
            candidates.append(
                ("Add relationships between classes", "add relationships between my classes")
            )
        # Cross-diagram suggestions
        if not _has_diagram(available_diagrams, "StateMachineDiagram"):
            candidates.append(
                ("Add a state machine", "create a state machine for this system")
            )
        if not _has_diagram(available_diagrams, "GUINoCodeDiagram"):
            candidates.append(
                ("Create a GUI", "create a gui for this system")
            )
        candidates.append(("Generate code", "generate python"))

    elif diagram_type == "StateMachineDiagram":
        if element_names:
            first_state = element_names[0]
            candidates.append(
                (f"Add transitions from {first_state}", f"add a transition from {first_state}")
            )
        candidates.append(("Generate code", "generate python"))

    if not candidates:
        return []

    return _build_actions(candidates, limit=4)


def format_suggestions_as_text(actions: List[Dict[str, str]]) -> str:
    """Format a list of suggested actions into a Markdown-friendly string.

    Used when the suggestions need to be embedded in a chat message
    alongside quality tips.
    """
    if not actions:
        return ""
    lines = ["**What's next?**"]
    for action in actions:
        label = action.get("label", "")
        prompt = action.get("prompt", "")
        if prompt:
            lines.append(f'- {label} \u2014 *"{prompt}"*')
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)
