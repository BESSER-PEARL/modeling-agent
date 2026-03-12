"""
Quality Review
--------------
Post-generation quality analysis that produces proactive improvement
suggestions for the user.

After a model is generated, the review engine checks for:
- God class detection (too many attributes/methods)
- Disconnected subgraphs (isolated classes)
- Missing ID attributes
- Naming convention consistency (camelCase vs snake_case)
- Missing common attributes for known entity types
- Bidirectional relationship suggestions
- Cross-diagram suggestions (e.g., "You can now generate a GUI")

Each suggestion carries a severity level and optional autoFix payload.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------

SEVERITY_WARNING = "warning"
SEVERITY_IMPROVEMENT = "improvement"
SEVERITY_INFO = "info"

# Severity priority for sorting (lower = higher priority)
_SEVERITY_ORDER = {SEVERITY_WARNING: 0, SEVERITY_IMPROVEMENT: 1, SEVERITY_INFO: 2}

MAX_SUGGESTIONS = 5

# ---------------------------------------------------------------------------
# Suggestion helper
# ---------------------------------------------------------------------------


def _suggestion(
    severity: str,
    target: str,
    message: str,
    auto_fix: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structured suggestion dict."""
    s: Dict[str, Any] = {
        "severity": severity,
        "target": target,
        "message": message,
    }
    if auto_fix is not None:
        s["autoFix"] = auto_fix
    return s


# ---------------------------------------------------------------------------
# Expected attributes for known entity types
# ---------------------------------------------------------------------------

_EXPECTED_ATTRIBUTES: Dict[str, List[Dict[str, Any]]] = {
    # User / Account entities
    "user": [
        {"name": "email", "type": "String", "visibility": "private"},
        {"name": "password", "type": "String", "visibility": "private"},
        {"name": "createdAt", "type": "Date", "visibility": "private"},
    ],
    "account": [
        {"name": "email", "type": "String", "visibility": "private"},
        {"name": "password", "type": "String", "visibility": "private"},
        {"name": "createdAt", "type": "Date", "visibility": "private"},
    ],
    # Order / Purchase entities
    "order": [
        {"name": "status", "type": "String", "visibility": "private"},
        {"name": "total", "type": "Float", "visibility": "private"},
        {"name": "date", "type": "Date", "visibility": "private"},
    ],
    "purchase": [
        {"name": "status", "type": "String", "visibility": "private"},
        {"name": "total", "type": "Float", "visibility": "private"},
        {"name": "date", "type": "Date", "visibility": "private"},
    ],
    # Product / Item entities
    "product": [
        {"name": "name", "type": "String", "visibility": "private"},
        {"name": "price", "type": "Float", "visibility": "private"},
        {"name": "description", "type": "String", "visibility": "private"},
    ],
    "item": [
        {"name": "name", "type": "String", "visibility": "private"},
        {"name": "price", "type": "Float", "visibility": "private"},
        {"name": "description", "type": "String", "visibility": "private"},
    ],
    # Address entity
    "address": [
        {"name": "street", "type": "String", "visibility": "private"},
        {"name": "city", "type": "String", "visibility": "private"},
        {"name": "zipCode", "type": "String", "visibility": "private"},
        {"name": "country", "type": "String", "visibility": "private"},
    ],
    # Other common entities
    "customer": [
        {"name": "email", "type": "String", "visibility": "private"},
        {"name": "phone", "type": "String", "visibility": "private"},
    ],
    "payment": [
        {"name": "amount", "type": "Float", "visibility": "private"},
        {"name": "method", "type": "String", "visibility": "private"},
        {"name": "date", "type": "Date", "visibility": "private"},
    ],
    "student": [
        {"name": "email", "type": "String", "visibility": "private"},
        {"name": "enrollmentDate", "type": "Date", "visibility": "private"},
    ],
    "employee": [
        {"name": "email", "type": "String", "visibility": "private"},
        {"name": "department", "type": "String", "visibility": "private"},
    ],
    "patient": [
        {"name": "dateOfBirth", "type": "Date", "visibility": "private"},
        {"name": "phone", "type": "String", "visibility": "private"},
    ],
    "appointment": [
        {"name": "date", "type": "Date", "visibility": "private"},
        {"name": "status", "type": "String", "visibility": "private"},
    ],
    "reservation": [
        {"name": "date", "type": "Date", "visibility": "private"},
        {"name": "status", "type": "String", "visibility": "private"},
    ],
    "message": [
        {"name": "content", "type": "String", "visibility": "private"},
        {"name": "timestamp", "type": "Date", "visibility": "private"},
    ],
    "task": [
        {"name": "title", "type": "String", "visibility": "private"},
        {"name": "status", "type": "String", "visibility": "private"},
        {"name": "priority", "type": "String", "visibility": "private"},
    ],
    "book": [
        {"name": "isbn", "type": "String", "visibility": "private"},
        {"name": "title", "type": "String", "visibility": "private"},
        {"name": "author", "type": "String", "visibility": "private"},
    ],
}


# ---------------------------------------------------------------------------
# Anti-pattern: God class detection
# ---------------------------------------------------------------------------

_GOD_CLASS_ATTR_THRESHOLD = 10
_GOD_CLASS_METHOD_THRESHOLD = 8


def _check_god_classes(classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag classes with too many attributes or methods."""
    suggestions: List[Dict[str, Any]] = []
    for cls in classes:
        class_name = cls.get("className", "?")
        num_attrs = len(cls.get("attributes", []))
        num_methods = len(cls.get("methods", []))

        reasons: List[str] = []
        if num_attrs > _GOD_CLASS_ATTR_THRESHOLD:
            reasons.append(f"{num_attrs} attributes")
        if num_methods > _GOD_CLASS_METHOD_THRESHOLD:
            reasons.append(f"{num_methods} methods")

        if reasons:
            suggestions.append(
                _suggestion(
                    severity=SEVERITY_WARNING,
                    target=class_name,
                    message=(
                        f"'{class_name}' has {' and '.join(reasons)}. "
                        f"Consider splitting this class into smaller, focused classes."
                    ),
                )
            )
    return suggestions


# ---------------------------------------------------------------------------
# Anti-pattern: Disconnected subgraphs (isolated classes)
# ---------------------------------------------------------------------------


def _check_isolated_classes(
    classes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
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

    if not isolated:
        return []

    suggestions: List[Dict[str, Any]] = []
    for name in isolated:
        suggestions.append(
            _suggestion(
                severity=SEVERITY_WARNING,
                target=name,
                message=(
                    f"'{name}' has no relationships with other classes. "
                    f"Consider connecting it to related entities."
                ),
            )
        )
    return suggestions


# ---------------------------------------------------------------------------
# Anti-pattern: Missing ID attribute — REMOVED
# ---------------------------------------------------------------------------
# This check was too noisy (flagged every class) and not useful because
# generators add IDs automatically.  Kept as a no-op for API compat.


def _check_missing_id_attributes(classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """No-op — ID attributes are added automatically by generators."""
    return []


# ---------------------------------------------------------------------------
# Anti-pattern: Naming convention inconsistency
# ---------------------------------------------------------------------------

_CAMEL_CASE_RE = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)+$")
_SNAKE_CASE_RE = re.compile(r"^[a-z]+(?:_[a-z0-9]+)+$")


def _check_naming_consistency(classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check if attribute names mix camelCase and snake_case."""
    camel_count = 0
    snake_count = 0
    camel_examples: List[str] = []
    snake_examples: List[str] = []

    for cls in classes:
        for attr in cls.get("attributes", []):
            name = attr.get("name", "")
            if _CAMEL_CASE_RE.match(name):
                camel_count += 1
                if len(camel_examples) < 3:
                    camel_examples.append(name)
            elif _SNAKE_CASE_RE.match(name):
                snake_count += 1
                if len(snake_examples) < 3:
                    snake_examples.append(name)

    # Only flag if both styles are present with at least 2 occurrences each
    if camel_count >= 2 and snake_count >= 2:
        return [
            _suggestion(
                severity=SEVERITY_WARNING,
                target="(all classes)",
                message=(
                    f"Attribute names mix camelCase (e.g., {', '.join(camel_examples)}) "
                    f"and snake_case (e.g., {', '.join(snake_examples)}). "
                    f"Pick one convention for consistency."
                ),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Missing common attributes for known entity types
# ---------------------------------------------------------------------------


def _check_missing_common_attributes(classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check if classes are missing commonly expected attributes for known entity types."""
    suggestions: List[Dict[str, Any]] = []
    for cls in classes:
        class_name = cls.get("className", "")
        class_lower = class_name.lower()
        existing_attrs = {a.get("name", "").lower() for a in cls.get("attributes", [])}

        expected_attrs = _EXPECTED_ATTRIBUTES.get(class_lower, [])
        if not expected_attrs:
            continue

        missing = [
            attr for attr in expected_attrs
            if attr["name"].lower() not in existing_attrs
        ]

        # Only suggest if at least 2 significant attributes are missing
        if len(missing) >= 2:
            missing_names = [m["name"] for m in missing]
            # Build autoFix for the first missing attribute only (simplest fix)
            first_missing = missing[0]
            suggestions.append(
                _suggestion(
                    severity=SEVERITY_IMPROVEMENT,
                    target=class_name,
                    message=(
                        f"'{class_name}' might benefit from: "
                        f"{', '.join(repr(n) for n in missing_names)}"
                    ),
                    auto_fix={
                        "action": "add_attribute",
                        "className": class_name,
                        "attribute": {
                            "name": first_missing["name"],
                            "type": first_missing["type"],
                            "visibility": first_missing["visibility"],
                        },
                    },
                )
            )
    return suggestions


# ---------------------------------------------------------------------------
# Bidirectional relationship check
# ---------------------------------------------------------------------------

# Pairs of class name patterns where bidirectional relationships are expected
_BIDIRECTIONAL_PAIRS = [
    ({"user", "customer", "person", "member"}, {"order", "purchase", "booking", "reservation"}),
    ({"student"}, {"course", "class"}),
    ({"doctor", "physician"}, {"patient"}),
    ({"teacher", "professor", "instructor"}, {"student"}),
    ({"employee"}, {"department"}),
    ({"author", "writer"}, {"book", "article", "post"}),
    ({"parent"}, {"child"}),
]


def _check_bidirectional_relationships(
    classes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Suggest bidirectional relationships where only one direction exists."""
    if len(classes) < 2 or not relationships:
        return []

    class_names_lower = {cls.get("className", "").lower(): cls.get("className", "") for cls in classes}

    # Build a set of existing relationship pairs (lowercase)
    rel_pairs = set()
    for rel in relationships:
        src = rel.get("source", "").lower()
        tgt = rel.get("target", "").lower()
        rel_pairs.add((src, tgt))
        rel_pairs.add((tgt, src))

    suggestions: List[Dict[str, Any]] = []
    checked_pairs = set()

    for group_a, group_b in _BIDIRECTIONAL_PAIRS:
        for name_a_lower, name_a_orig in class_names_lower.items():
            for name_b_lower, name_b_orig in class_names_lower.items():
                if name_a_lower == name_b_lower:
                    continue
                pair_key = tuple(sorted([name_a_lower, name_b_lower]))
                if pair_key in checked_pairs:
                    continue

                matches_a = any(pat in name_a_lower for pat in group_a)
                matches_b = any(pat in name_b_lower for pat in group_b)
                matches_reverse = any(pat in name_a_lower for pat in group_b) and any(
                    pat in name_b_lower for pat in group_a
                )

                if (matches_a and matches_b) or matches_reverse:
                    checked_pairs.add(pair_key)
                    # Check if relationship exists
                    if (name_a_lower, name_b_lower) not in rel_pairs:
                        suggestions.append(
                            _suggestion(
                                severity=SEVERITY_INFO,
                                target=f"{name_a_orig} <-> {name_b_orig}",
                                message=(
                                    f"'{name_a_orig}' and '{name_b_orig}' are commonly related. "
                                    f"Consider adding a relationship between them."
                                ),
                            )
                        )

    return suggestions


# ---------------------------------------------------------------------------
# Cross-diagram suggestions
# ---------------------------------------------------------------------------


def _cross_diagram_suggestions(
    diagram_type: str,
    available_diagrams: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Suggest next steps based on what diagrams exist."""
    available = set(available_diagrams or [])
    suggestions: List[Dict[str, Any]] = []

    if diagram_type == "ClassDiagram":
        if "ObjectDiagram" not in available:
            suggestions.append(
                _suggestion(
                    severity=SEVERITY_INFO,
                    target="(project)",
                    message="Create an Object Diagram to test your classes with sample data",
                )
            )
        if "StateMachineDiagram" not in available:
            suggestions.append(
                _suggestion(
                    severity=SEVERITY_INFO,
                    target="(project)",
                    message=(
                        "Add a State Machine for entities with lifecycle "
                        "(e.g., Order status flow)"
                    ),
                )
            )
        if "GUINoCodeDiagram" not in available:
            suggestions.append(
                _suggestion(
                    severity=SEVERITY_INFO,
                    target="(project)",
                    message="Generate a GUI from your class diagram for a visual interface",
                )
            )

    elif diagram_type == "StateMachineDiagram":
        if "ClassDiagram" not in available:
            suggestions.append(
                _suggestion(
                    severity=SEVERITY_INFO,
                    target="(project)",
                    message="Create a Class Diagram to define the data model behind your states",
                )
            )

    elif diagram_type == "GUINoCodeDiagram":
        suggestions.append(
            _suggestion(
                severity=SEVERITY_INFO,
                target="(project)",
                message="You can now generate a web app from your GUI + class diagram",
            )
        )

    return suggestions


# ---------------------------------------------------------------------------
# Prioritize and cap suggestions
# ---------------------------------------------------------------------------


def _prioritize_suggestions(
    suggestions: List[Dict[str, Any]],
    limit: int = MAX_SUGGESTIONS,
) -> List[Dict[str, Any]]:
    """Sort by severity (warning > improvement > info) and cap at limit."""
    suggestions.sort(key=lambda s: _SEVERITY_ORDER.get(s.get("severity", "info"), 99))
    return suggestions[:limit]


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

    all_suggestions: List[Dict[str, Any]] = []

    # Class diagram specific checks
    if diagram_type == "ClassDiagram":
        system_spec = result.get("systemSpec")

        if isinstance(system_spec, dict):
            classes = system_spec.get("classes", [])
            relationships = system_spec.get("relationships", [])

            # Only run checks for non-trivial diagrams
            if len(classes) >= 2:
                all_suggestions.extend(_check_god_classes(classes))
                all_suggestions.extend(_check_isolated_classes(classes, relationships))
                all_suggestions.extend(_check_missing_id_attributes(classes))
                all_suggestions.extend(_check_naming_consistency(classes))
                all_suggestions.extend(_check_missing_common_attributes(classes))
                all_suggestions.extend(
                    _check_bidirectional_relationships(classes, relationships)
                )

    # Cross-diagram suggestions
    all_suggestions.extend(
        _cross_diagram_suggestions(diagram_type, available_diagrams)
    )

    if not all_suggestions:
        return None

    # Prioritize and cap
    top_suggestions = _prioritize_suggestions(all_suggestions)

    # Format as a compact suggestion block
    lines = ["**Suggestions to improve your model:**"]
    for s in top_suggestions:
        severity_tag = s["severity"].upper()
        lines.append(f"- [{severity_tag}] {s['message']}")

    return "\n".join(lines)


def review_generated_model_structured(
    result: Dict[str, Any],
    diagram_type: str,
    available_diagrams: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Analyze a generated model and return structured suggestion dicts.

    Each suggestion has the shape::

        {
            "severity": "warning" | "improvement" | "info",
            "target": "<class or element name>",
            "message": "<human-readable suggestion>",
            "autoFix": { ... }  # optional
        }

    Returns an empty list if the model looks good.
    """
    if not isinstance(result, dict):
        return []

    all_suggestions: List[Dict[str, Any]] = []

    if diagram_type == "ClassDiagram":
        system_spec = result.get("systemSpec")

        if isinstance(system_spec, dict):
            classes = system_spec.get("classes", [])
            relationships = system_spec.get("relationships", [])

            if len(classes) >= 2:
                all_suggestions.extend(_check_god_classes(classes))
                all_suggestions.extend(_check_isolated_classes(classes, relationships))
                all_suggestions.extend(_check_missing_id_attributes(classes))
                all_suggestions.extend(_check_naming_consistency(classes))
                all_suggestions.extend(_check_missing_common_attributes(classes))
                all_suggestions.extend(
                    _check_bidirectional_relationships(classes, relationships)
                )

    all_suggestions.extend(
        _cross_diagram_suggestions(diagram_type, available_diagrams)
    )

    return _prioritize_suggestions(all_suggestions)
