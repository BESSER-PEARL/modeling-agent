"""
Quality Review
--------------
Post-generation quality analysis that produces proactive improvement
suggestions for the user.

After a model is generated, the review engine checks for:
- Missing attributes (e.g., User without email)
- Missing methods (e.g., Order without calculateTotal)
- Relationship completeness
- Cross-diagram suggestions (e.g., "You can now generate a GUI")
- Naming convention issues
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attribute completeness checks
# ---------------------------------------------------------------------------

# Classes that should typically have certain attributes
_EXPECTED_ATTRIBUTES: Dict[str, List[str]] = {
    "user": ["id", "email", "name", "password"],
    "customer": ["id", "name", "email", "phone"],
    "product": ["id", "name", "price", "description"],
    "order": ["id", "date", "status", "total"],
    "payment": ["id", "amount", "method", "date"],
    "account": ["id", "balance", "type"],
    "student": ["id", "name", "email"],
    "employee": ["id", "name", "email", "department"],
    "book": ["isbn", "title", "author"],
    "patient": ["id", "name", "dateOfBirth"],
    "appointment": ["id", "date", "status"],
    "task": ["id", "title", "status", "priority"],
    "message": ["id", "content", "timestamp"],
    "reservation": ["id", "date", "status"],
}


def _check_missing_attributes(classes: List[Dict[str, Any]]) -> List[str]:
    """Check if classes are missing commonly expected attributes."""
    suggestions: List[str] = []
    for cls in classes:
        class_name = cls.get("className", "")
        class_lower = class_name.lower()
        existing_attrs = {a.get("name", "").lower() for a in cls.get("attributes", [])}

        expected = _EXPECTED_ATTRIBUTES.get(class_lower, [])
        missing = [attr for attr in expected if attr not in existing_attrs]

        # Only suggest if significant attributes are missing
        if len(missing) >= 2:
            suggestions.append(
                f"**{class_name}** might benefit from: `{'`, `'.join(missing)}`"
            )
    return suggestions


# ---------------------------------------------------------------------------
# Relationship completeness checks
# ---------------------------------------------------------------------------

def _check_isolated_classes(
    classes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[str]:
    """Find classes with no relationships (isolated nodes)."""
    if len(classes) <= 1:
        return []

    connected_classes = set()
    for rel in relationships:
        connected_classes.add(rel.get("source", ""))
        connected_classes.add(rel.get("target", ""))

    isolated = [
        cls.get("className", "?")
        for cls in classes
        if cls.get("className", "") not in connected_classes
    ]

    if isolated:
        return [
            f"{'These classes have' if len(isolated) > 1 else 'This class has'} "
            f"no relationships: **{'**, **'.join(isolated)}**. "
            "Consider adding connections to other classes."
        ]
    return []


def _check_missing_id_attributes(classes: List[Dict[str, Any]]) -> List[str]:
    """Check if classes are missing an ID attribute."""
    missing_id = []
    for cls in classes:
        attrs = cls.get("attributes", [])
        has_id = any(
            a.get("name", "").lower() in ("id", "uuid", "key", "pk")
            for a in attrs
        )
        if not has_id and attrs:  # Only flag if class has other attributes
            missing_id.append(cls.get("className", "?"))

    if missing_id and len(missing_id) <= 3:
        return [
            f"Consider adding an `id` attribute to: **{'**, **'.join(missing_id)}**"
        ]
    return []


# ---------------------------------------------------------------------------
# Cross-diagram suggestions
# ---------------------------------------------------------------------------

def _cross_diagram_suggestions(
    diagram_type: str,
    available_diagrams: Optional[List[str]] = None,
) -> List[str]:
    """Suggest next steps based on what diagrams exist."""
    available = set(available_diagrams or [])
    suggestions: List[str] = []

    if diagram_type == "ClassDiagram":
        if "ObjectDiagram" not in available:
            suggestions.append(
                "Create an **Object Diagram** to test your classes with sample data"
            )
        if "StateMachineDiagram" not in available:
            suggestions.append(
                "Add a **State Machine** for entities with lifecycle (e.g., Order status flow)"
            )
        if "GUINoCodeDiagram" not in available:
            suggestions.append(
                "Generate a **GUI** from your class diagram for a visual interface"
            )

    elif diagram_type == "StateMachineDiagram":
        if "ClassDiagram" not in available:
            suggestions.append(
                "Create a **Class Diagram** to define the data model behind your states"
            )

    elif diagram_type == "GUINoCodeDiagram":
        suggestions.append(
            "You can now **generate a web app** from your GUI + class diagram"
        )

    return suggestions


# ---------------------------------------------------------------------------
# Main review entry point
# ---------------------------------------------------------------------------

def review_generated_model(
    result: Dict[str, Any],
    diagram_type: str,
    available_diagrams: Optional[List[str]] = None,
) -> Optional[str]:
    """Analyze a generated model result and produce quality suggestions.

    Returns a formatted suggestion string if issues are found, or ``None``
    if the model looks good.
    """
    if not isinstance(result, dict):
        return None

    suggestions: List[str] = []

    # Class diagram specific checks
    if diagram_type == "ClassDiagram":
        # Check both single element and complete system structures
        system_spec = result.get("systemSpec")
        element = result.get("element")

        if isinstance(system_spec, dict):
            classes = system_spec.get("classes", [])
            relationships = system_spec.get("relationships", [])

            # Only run checks for non-trivial diagrams
            if len(classes) >= 2:
                suggestions.extend(_check_missing_attributes(classes))
                suggestions.extend(_check_isolated_classes(classes, relationships))
                suggestions.extend(_check_missing_id_attributes(classes))

    # Cross-diagram suggestions
    cross = _cross_diagram_suggestions(diagram_type, available_diagrams)
    if cross:
        suggestions.extend(cross)

    if not suggestions:
        return None

    # Format as a compact suggestion block
    lines = ["**Suggestions to improve your model:**"]
    for s in suggestions[:5]:  # Cap at 5 suggestions
        lines.append(f"- {s}")

    return "\n".join(lines)
