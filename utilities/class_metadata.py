"""
Class Metadata Extraction
-------------------------
Helpers that extract structured class metadata from a ClassDiagram model.
Used by the GUI chart/table generator to bind components to real class and
attribute IDs.
"""

from typing import Any, Dict, List, Optional

from .model_context import _clean_attr_name as _clean_attribute_name


_NUMERIC_TYPES = {"int", "float", "double", "decimal", "number", "long", "short"}
_STRING_TYPES = {"str", "string", "text", "char", "varchar"}


def _parse_attribute_type(element: Dict[str, Any]) -> str:
    """Extract the attribute type from an element, handling legacy and new formats."""
    attr_type = element.get("attributeType")
    if isinstance(attr_type, str) and attr_type.strip():
        return attr_type.strip().lower()
    # Legacy format: parse from name like "+age: int"
    name = element.get("name")
    if isinstance(name, str) and ":" in name:
        return name.rsplit(":", 1)[1].strip().lower()
    return "string"


def extract_class_metadata(model: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract structured class metadata from a ClassDiagram model.

    Returns a list of dicts, each with::

        {
            "id": "<class element id>",
            "name": "ClassName",
            "attributes": [
                {"id": "<attr id>", "name": "attrName", "type": "int", "isNumeric": True, "isString": False},
                ...
            ]
        }

    This mirrors the frontend ``getClassMetadata`` helper so the backend
    handler can produce chart components bound to real class/attribute IDs.
    """
    if not isinstance(model, dict):
        return []
    elements = model.get("elements")
    if not isinstance(elements, dict):
        return []

    # First pass: identify all Class elements
    classes: Dict[str, Dict[str, Any]] = {}
    for eid, element in elements.items():
        if not isinstance(element, dict):
            continue
        if element.get("type") == "Class":
            name = element.get("name")
            if isinstance(name, str) and name.strip():
                classes[eid] = {"id": eid, "name": name.strip(), "attributes": []}

    # Second pass: attach ClassAttribute elements to their owner class
    for eid, element in elements.items():
        if not isinstance(element, dict):
            continue
        if element.get("type") != "ClassAttribute":
            continue
        owner = element.get("owner")
        if not isinstance(owner, str) or owner not in classes:
            continue
        raw_name = element.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        attr_type = _parse_attribute_type(element)
        clean_name = _clean_attribute_name(raw_name)
        if not clean_name:
            continue
        classes[owner]["attributes"].append({
            "id": eid,
            "name": clean_name,
            "type": attr_type,
            "isNumeric": attr_type in _NUMERIC_TYPES,
            "isString": attr_type in _STRING_TYPES,
        })

    return list(classes.values())


def format_class_metadata_for_prompt(class_metadata: List[Dict[str, Any]]) -> str:
    """Format extracted class metadata into a compact string for LLM prompts.

    Produces something like::

        Available classes from the Class Diagram:
        - Class "Book" (id: abc123): name (str), pages (int), price (float)
        - Class "Author" (id: def456): firstName (str), lastName (str), age (int)
    """
    if not class_metadata:
        return ""
    lines = ["Available classes from the Class Diagram:"]
    for cls in class_metadata:
        attrs = cls.get("attributes", [])
        if attrs:
            attr_parts = [f"{a['name']} ({a['type']})" for a in attrs]
            attrs_str = ", ".join(attr_parts)
        else:
            attrs_str = "no attributes"
        lines.append(f"- Class \"{cls['name']}\" (id: {cls['id']}): {attrs_str}")
    return "\n".join(lines)
