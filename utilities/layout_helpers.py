"""
Layout Helpers
--------------
Position extraction, primary-element detection, and layout-anchor building
used by the workspace context builder and diagram handlers to provide
spatial awareness to the LLM.
"""

from typing import Any, Dict, List, Optional


def to_int(value: Any) -> Optional[int]:
    """Safely convert a value to an integer, returning None on failure."""
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def extract_element_position(element: Dict[str, Any]) -> Optional[Dict[str, Optional[int]]]:
    """Extract ``{x, y, width, height}`` from an element's bounds or position."""
    if not isinstance(element, dict):
        return None

    bounds = element.get("bounds")
    if isinstance(bounds, dict):
        x = to_int(bounds.get("x"))
        y = to_int(bounds.get("y"))
        if x is not None and y is not None:
            return {
                "x": x,
                "y": y,
                "width": to_int(bounds.get("width")),
                "height": to_int(bounds.get("height")),
            }

    position = element.get("position")
    if isinstance(position, dict):
        x = to_int(position.get("x"))
        y = to_int(position.get("y"))
        if x is not None and y is not None:
            return {"x": x, "y": y, "width": None, "height": None}

    return None


def is_primary_layout_element(diagram_type: str, element: Dict[str, Any]) -> bool:
    """Return True if *element* is a top-level visual element for *diagram_type*."""
    element_type = element.get("type")
    owner = element.get("owner")
    owner_is_root = not isinstance(owner, str) or not owner

    diagram_primary_types = {
        "ClassDiagram": {"Class"},
        "ObjectDiagram": {"Object"},
        "StateMachineDiagram": {"State", "StateInitialNode", "StateFinalNode"},
        "AgentDiagram": {"AgentState", "AgentIntent", "StateInitialNode"},
    }

    primary_types = diagram_primary_types.get(diagram_type)
    if isinstance(element_type, str) and primary_types:
        return element_type in primary_types

    noisy_types = {
        "ClassAttribute",
        "ClassMethod",
        "AgentStateBody",
        "AgentStateFallbackBody",
        "AgentIntentBody",
    }
    if isinstance(element_type, str) and element_type in noisy_types:
        return False

    return owner_is_root


def build_layout_anchor_lines(
    model_data: Any, diagram_type: str, limit: int = 18,
) -> List[str]:
    """Build compact layout-anchor lines from positioned elements."""
    if not isinstance(model_data, dict):
        return []

    elements = model_data.get("elements")
    if not isinstance(elements, dict):
        return []

    anchors: List[tuple[int, int, str]] = []
    for element_id, element in elements.items():
        if not isinstance(element, dict):
            continue
        if not is_primary_layout_element(diagram_type, element):
            continue

        position = extract_element_position(element)
        if not position:
            continue

        x = position["x"]
        y = position["y"]
        if not isinstance(x, int) or not isinstance(y, int):
            continue

        width = position.get("width")
        height = position.get("height")
        size_part = (
            f", w={width}, h={height}"
            if isinstance(width, int) and isinstance(height, int)
            else ""
        )
        element_type = element.get("type") if isinstance(element.get("type"), str) else "Element"
        name = element.get("name") if isinstance(element.get("name"), str) and element.get("name") else element_id
        line = f"- {element_type} '{name}': x={x}, y={y}{size_part}"
        anchors.append((y, x, line))

    anchors.sort(key=lambda item: (item[0], item[1]))
    return [line for _, _, line in anchors[:limit]]
