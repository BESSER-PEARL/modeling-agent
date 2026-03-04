"""
Class Metadata Extraction
-------------------------
Helpers that extract structured class metadata from a ClassDiagram model.
Used by the GUI chart/table generator to bind components to real class and
attribute IDs.
"""

import re
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
    return "unknown"


def _clean_method_name(raw: str) -> str:
    """Extract a clean method name, stripping visibility prefix and parameters."""
    cleaned = raw.strip()
    # Strip UML visibility prefix (+/-/#/~)
    if cleaned and cleaned[0] in "+-#~":
        cleaned = cleaned[1:].strip()
    # Strip parameters and return type
    if "(" in cleaned:
        cleaned = cleaned[:cleaned.index("(")].strip()
    # Strip trailing return type after colon
    if ":" in cleaned:
        cleaned = cleaned[:cleaned.index(":")].strip()
    return cleaned


def _parse_method_params(raw: str) -> List[str]:
    """Extract parameter names from a method signature like ``+doStuff(x: int, y: str): void``."""
    m = re.search(r"\(([^)]*)\)", raw)
    if not m:
        return []
    params_str = m.group(1).strip()
    if not params_str:
        return []
    names: List[str] = []
    for part in params_str.split(","):
        part = part.strip()
        if ":" in part:
            part = part[:part.index(":")].strip()
        # Strip visibility prefix
        if part and part[0] in "+-#~":
            part = part[1:].strip()
        if part and part.lower() not in ("self", "session"):
            names.append(part)
    return names


def _is_instance_method(raw: str) -> bool:
    """Return True if the raw method signature contains ``self`` or ``session`` as first param."""
    m = re.search(r"\(([^)]*)\)", raw)
    if not m:
        return False
    params_str = m.group(1).strip()
    if not params_str:
        return False
    first_param = params_str.split(",")[0].strip()
    if ":" in first_param:
        first_param = first_param[:first_param.index(":")].strip()
    if first_param and first_param[0] in "+-#~":
        first_param = first_param[1:].strip()
    return first_param.lower() in ("self", "session")


def extract_class_metadata(model: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract structured class metadata from a ClassDiagram model.

    Returns a list of dicts, each with::

        {
            "id": "<class element id>",
            "name": "ClassName",
            "attributes": [
                {"id": "<attr id>", "name": "attrName", "type": "int", "isNumeric": True, "isString": False},
                ...
            ],
            "methods": [
                {"id": "<method id>", "name": "methodName", "isInstanceMethod": True, "params": ["x", "y"]},
                ...
            ],
            "associationEnds": [
                {"relationshipId": "rid", "targetClassId": "cid", "targetClassName": "Other",
                 "displayAttributeId": "aid", "displayAttributeName": "name"},
                ...
            ]
        }

    This mirrors the frontend ``getClassMetadata`` / ``getMethodsByClassId`` /
    ``getEndsByClassId`` helpers so the backend handler can produce chart,
    table, and button components bound to real class/attribute/method IDs.
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
                classes[eid] = {
                    "id": eid,
                    "name": name.strip(),
                    "attributes": [],
                    "methods": [],
                    "associationEnds": [],
                }

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

    # Third pass: attach ClassMethod elements to their owner class
    for eid, element in elements.items():
        if not isinstance(element, dict):
            continue
        if element.get("type") != "ClassMethod":
            continue
        owner = element.get("owner")
        if not isinstance(owner, str) or owner not in classes:
            continue
        raw_name = element.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        clean_name = _clean_method_name(raw_name)
        if not clean_name:
            continue
        classes[owner]["methods"].append({
            "id": eid,
            "name": clean_name,
            "isInstanceMethod": _is_instance_method(raw_name),
            "params": _parse_method_params(raw_name),
        })

    # Fourth pass: extract association ends from relationships
    relationships = model.get("relationships")
    if isinstance(relationships, dict):
        for rid, rel in relationships.items():
            if not isinstance(rel, dict):
                continue
            rel_type = rel.get("type", "")
            # Capture inheritance as metadata on the child class
            if rel_type == "ClassInheritance":
                source = rel.get("source")
                target = rel.get("target")
                if isinstance(source, dict) and isinstance(target, dict):
                    child_id = source.get("element")
                    parent_id = target.get("element")
                    if child_id in classes and parent_id in classes:
                        parent_name = classes[parent_id]["name"]
                        if "inheritsFrom" not in classes[child_id]:
                            classes[child_id]["inheritsFrom"] = []
                        classes[child_id]["inheritsFrom"].append(parent_name)
                continue
            source = rel.get("source")
            target = rel.get("target")
            if not isinstance(source, dict) or not isinstance(target, dict):
                continue
            source_id = source.get("element")
            target_id = target.get("element")
            if source_id in classes and target_id in classes:
                target_cls = classes[target_id]
                # Find first string-ish attribute in target class for display
                display_attr = None
                for attr in target_cls.get("attributes", []):
                    if attr.get("isString"):
                        display_attr = attr
                        break
                if not display_attr and target_cls.get("attributes"):
                    display_attr = target_cls["attributes"][0]

                classes[source_id]["associationEnds"].append({
                    "relationshipId": rid,
                    "targetClassId": target_id,
                    "targetClassName": target_cls["name"],
                    "displayAttributeId": display_attr["id"] if display_attr else None,
                    "displayAttributeName": display_attr["name"] if display_attr else None,
                })

    return list(classes.values())


def format_class_metadata_for_prompt(class_metadata: List[Dict[str, Any]]) -> str:
    """Format extracted class metadata into a compact string for LLM prompts.

    Produces something like::

        Available classes from the Class Diagram:
        - Class "Book" (id: abc123): name (str), pages (int), price (float)
          Methods: getTitle(), calculateDiscount(percent)
          Relationships → Author
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
        inherits = cls.get("inheritsFrom")
        inherit_str = f" extends {', '.join(inherits)}" if inherits else ""
        lines.append(f"- Class \"{cls['name']}\"{inherit_str} (id: {cls['id']}): {attrs_str}")

        methods = cls.get("methods", [])
        if methods:
            method_parts = [f"{m['name']}()" for m in methods]
            lines.append(f"  Methods: {', '.join(method_parts)}")

        ends = cls.get("associationEnds", [])
        if ends:
            target_names = list(dict.fromkeys(e["targetClassName"] for e in ends))
            lines.append(f"  Relationships → {', '.join(target_names)}")

    return "\n".join(lines)
