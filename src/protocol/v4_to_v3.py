"""Inbound v4 (React Flow) -> v3 (Apollon) diagram-model normalizer.

The WME frontend is moving from the Apollon/Redux editor (v3 models:
``elements`` / ``relationships`` dicts) to a React Flow editor (v4 models:
``nodes`` / ``edges`` lists). Everything downstream of the protocol boundary
in this agent (model summaries, class-diagram guards, layout, resolution…)
reads the v3 shape, so incoming v4 models are converted back to v3 once, in
``protocol.adapters.parse_v2_payload``.

This module is the Python inverse of the frontend's v3 -> v4 migrator
(``packages/library/lib/utils/versionConverter.ts``: ``convertV3ToV4`` and its
``convertV4ToV3Class`` / ``...StateMachine`` / ``...Agent`` / ``...User``
counterparts). The canonical mapping is documented in BESSER's
``docs/source/migrations/uml-v4-shape.md``.

Rules:

* Pure functions — the input is never mutated (every output container is new).
* Tolerant — missing or partial fields fall back to empty values; no KeyError.
* Only UML models are converted. ``GUINoCodeDiagram`` (GrapesJS) and
  ``QuantumCircuitDiagram`` (Quirk columns) are passed through untouched, as
  are models that are already v3.
* Only the agent's INPUT is converted. Its output payloads are format-agnostic
  semantic specs that the frontend converts itself.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "NON_UML_DIAGRAM_TYPES",
    "is_v4_model",
    "convert_v4_model_to_v3",
    "normalize_model",
    "normalize_project_snapshot",
]

# Diagram types whose models are not Apollon/React-Flow UML models.
NON_UML_DIAGRAM_TYPES = frozenset({"GUINoCodeDiagram", "QuantumCircuitDiagram"})

V3_VERSION = "3.0.0"

# ---------------------------------------------------------------------------
# Type / handle maps
# ---------------------------------------------------------------------------

# v4 node.type -> v3 element type, where they differ (inverse of
# ``convertV3NodeTypeToV4``). PascalCase StateMachine / Agent / User / NN node
# types are identical in both versions and fall through unchanged.
_NODE_TYPE_V4_TO_V3: Dict[str, str] = {
    "class": "Class",
    "package": "Package",
    "classAttribute": "ClassAttribute",
    "classMethod": "ClassMethod",
    "classoclconstraint": "ClassOCLConstraint",
    "objectName": "ObjectName",
    "objectAttribute": "ObjectAttribute",
    "objectMethod": "ObjectMethod",
    "comment": "Comments",
    "colorDescription": "ColorDescription",
    "titleAndDesctiption": "TitleAndDescription",
    # BPMN (lowerCamelCase in v4)
    "bpmnTask": "BPMNTask",
    "bpmnGateway": "BPMNGateway",
    "bpmnStartEvent": "BPMNStartEvent",
    "bpmnIntermediateEvent": "BPMNIntermediateEvent",
    "bpmnEndEvent": "BPMNEndEvent",
    "bpmnSubprocess": "BPMNSubprocess",
    "bpmnTransaction": "BPMNTransaction",
    "bpmnCallActivity": "BPMNCallActivity",
    "bpmnAnnotation": "BPMNAnnotation",
    "bpmnDataObject": "BPMNDataObject",
    "bpmnDataStore": "BPMNDataStore",
    "bpmnPool": "BPMNPool",
    "bpmnSwimlane": "BPMNSwimlane",
    "bpmnGroup": "BPMNGroup",
}

# v4 BPMN edge types -> v3 ``BPMNFlow`` + ``flowType`` discriminator.
_BPMN_EDGE_FLOW_TYPES: Dict[str, str] = {
    "BPMNSequenceFlow": "sequence",
    "BPMNMessageFlow": "message",
    "BPMNAssociationFlow": "association",
    "BPMNDataAssociationFlow": "dataAssociation",
}

# Other v4 edge types that were renamed by the migrator.
_EDGE_TYPE_V4_TO_V3: Dict[str, str] = {
    "CommentLink": "Link",
    "FlowChartFlowline": "FlowchartFlowline",
}

# v4 handle id -> v3 direction (inverse of ``convertV3HandleToV4``).
_HANDLE_V4_TO_V3: Dict[str, str] = {
    "top": "Up",
    "right": "Right",
    "bottom": "Down",
    "left": "Left",
    "right-top": "Upright",
    "left-top": "Upleft",
    "right-bottom": "Downright",
    "left-bottom": "Downleft",
    "top-right": "RightTop",
    "bottom-right": "RightBottom",
    "top-left": "LeftTop",
    "bottom-left": "LeftBottom",
}

_CLASS_STEREOTYPE_TO_V3: Dict[str, str] = {
    "abstract": "AbstractClass",
    "interface": "Interface",
    "enumeration": "Enumeration",
}
_V3_CLASS_TYPES = {"Class", "AbstractClass", "Interface", "Enumeration"}

# Node ``data`` keys that are *not* copied verbatim onto the v3 element because
# the per-type converters re-expand them into child elements (or they are
# v4-only render state).
_COLLAPSED_DATA_KEYS = {
    "attributes", "methods", "oclConstraints",          # Class / Object / User
    "bodies", "fallbackBodies",                         # State / AgentState
    "training_phrases", "entity_slots",                 # AgentIntent
}
_V4_ONLY_NODE_KEYS = {"initial", "measured", "selected", "dragging", "resizing"}

# Edge ``data`` keys consumed explicitly by the edge converter.
_EDGE_HANDLED_KEYS = {
    "points", "label", "name",
    "sourceRole", "targetRole", "sourceMultiplicity", "targetMultiplicity",
    "sourceNavigable", "targetNavigable",
    "isManuallyLayouted", "messages", "params",
    "transitionType", "predefined", "custom", "legacy", "legacyShape",
    "flowType",
}

_ZERO_BOUNDS = {"x": 0, "y": 0, "width": 0, "height": 0}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def is_v4_model(model: Any) -> bool:
    """True when *model* is a v4 (React Flow) diagram model.

    A model counts as v4 when its ``version`` starts with ``"4."`` or it
    carries a ``nodes`` list. A model that already has a v3 ``elements`` dict
    and no ``nodes`` list is never treated as v4.
    """
    if not isinstance(model, dict):
        return False
    has_nodes = isinstance(model.get("nodes"), list)
    if isinstance(model.get("elements"), dict) and not has_nodes:
        return False
    version = model.get("version")
    return has_nodes or (isinstance(version, str) and version.startswith("4."))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        # Some payloads key rows by id instead of listing them.
        return list(value.values())
    return []


def _num(value: Any, default: float = 0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _copy(value: Any) -> Any:
    """Deep copy so the output never aliases the (caller-owned) input."""
    return copy.deepcopy(value)


def _invert_handle(handle: Any) -> str:
    if not isinstance(handle, str) or not handle:
        return ""
    return _HANDLE_V4_TO_V3.get(handle, handle)


def _v3_node_type(v4_type: Any) -> str:
    if not isinstance(v4_type, str) or not v4_type:
        return ""
    mapped = _NODE_TYPE_V4_TO_V3.get(v4_type)
    if mapped:
        return mapped
    lowered = v4_type.lower()
    if lowered == "classoclconstraint":
        return "ClassOCLConstraint"
    if v4_type.startswith("bpmn") and len(v4_type) > 4:
        return "BPMN" + v4_type[4:]
    return v4_type


def _node_size(node: Dict[str, Any]) -> Tuple[float, float]:
    measured = _as_dict(node.get("measured"))
    style = _as_dict(node.get("style"))
    width = node.get("width")
    height = node.get("height")
    if not isinstance(width, (int, float)) or isinstance(width, bool) or not width:
        width = measured.get("width") or style.get("width") or 0
    if not isinstance(height, (int, float)) or isinstance(height, bool) or not height:
        height = measured.get("height") or style.get("height") or 0
    return _num(width), _num(height)


def _absolute_positions(nodes: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """v4 child positions are relative to ``parentId``; v3 bounds are absolute."""
    by_id = {n.get("id"): n for n in nodes if isinstance(n.get("id"), str)}
    cache: Dict[str, Tuple[float, float]] = {}

    def resolve(node_id: str, seen: set) -> Tuple[float, float]:
        if node_id in cache:
            return cache[node_id]
        node = by_id.get(node_id) or {}
        pos = _as_dict(node.get("position"))
        x, y = _num(pos.get("x")), _num(pos.get("y"))
        parent_id = node.get("parentId")
        if isinstance(parent_id, str) and parent_id in by_id and parent_id not in seen:
            px, py = resolve(parent_id, seen | {node_id})
            x, y = x + px, y + py
        cache[node_id] = (x, y)
        return x, y

    for nid in by_id:
        resolve(nid, {nid})
    return cache


def _row_id(row: Dict[str, Any], owner_id: str, kind: str, index: int) -> str:
    rid = row.get("id")
    if isinstance(rid, str) and rid:
        return rid
    return f"{owner_id}-{kind}-{index}"


def _child_element(
    row: Dict[str, Any],
    row_id: str,
    v3_type: str,
    owner_id: str,
    *,
    exclude: Iterable[str] = (),
) -> Dict[str, Any]:
    """Serialise one inline v4 row as a v3 child element owned by *owner_id*."""
    skip = {"id", "type", "owner", "bounds", *exclude}
    element: Dict[str, Any] = {k: _copy(v) for k, v in row.items() if k not in skip}
    element.update(
        {
            "id": row_id,
            "name": _str(row.get("name")),
            "type": v3_type,
            "owner": owner_id,
            "bounds": dict(_ZERO_BOUNDS),
        }
    )
    return element


# ---------------------------------------------------------------------------
# Per-node conversion
# ---------------------------------------------------------------------------


def _base_element(
    node: Dict[str, Any],
    v3_type: str,
    abs_pos: Tuple[float, float],
) -> Dict[str, Any]:
    data = _as_dict(node.get("data"))
    element: Dict[str, Any] = {
        k: _copy(v)
        for k, v in data.items()
        if k not in _COLLAPSED_DATA_KEYS and k not in _V4_ONLY_NODE_KEYS
    }
    width, height = _node_size(node)
    parent_id = node.get("parentId")
    element.update(
        {
            "id": node.get("id"),
            "name": _str(data.get("name")),
            "type": v3_type,
            "owner": parent_id if isinstance(parent_id, str) and parent_id else None,
            "bounds": {"x": abs_pos[0], "y": abs_pos[1], "width": width, "height": height},
        }
    )
    return element


def _convert_class_node(node, data, element, elements) -> None:
    node_id = element["id"]
    stereotype = data.get("stereotype")
    key = stereotype.strip().lower() if isinstance(stereotype, str) else ""
    if key in _CLASS_STEREOTYPE_TO_V3:
        element["type"] = _CLASS_STEREOTYPE_TO_V3[key]
        element.pop("stereotype", None)
    elif key == "oclconstraint":
        _convert_ocl_node(data, element)
        return
    else:
        element["type"] = "Class"
        if not key:
            element.pop("stereotype", None)

    attr_ids: List[str] = []
    for i, row in enumerate(_as_list(data.get("attributes"))):
        if not isinstance(row, dict):
            continue
        rid = _row_id(row, node_id, "attr", i)
        elements[rid] = _child_element(
            row, rid, "ClassAttribute", node_id, exclude=("parameters", "returnType")
        )
        attr_ids.append(rid)
    method_ids: List[str] = []
    for i, row in enumerate(_as_list(data.get("methods"))):
        if not isinstance(row, dict):
            continue
        rid = _row_id(row, node_id, "method", i)
        elements[rid] = _child_element(row, rid, "ClassMethod", node_id)
        method_ids.append(rid)
    element["attributes"] = attr_ids
    element["methods"] = method_ids

    # Owned OCL rows (the TS migrator may collapse constraints onto the class).
    for i, row in enumerate(_as_list(data.get("oclConstraints"))):
        if not isinstance(row, dict):
            continue
        rid = _row_id(row, node_id, "ocl", i)
        child = _child_element(row, rid, "ClassOCLConstraint", node_id)
        expression = _str(row.get("expression")) or _str(row.get("constraint"))
        child["constraint"] = expression
        child["expression"] = expression
        elements[rid] = child


def _convert_ocl_node(data, element) -> None:
    element["type"] = "ClassOCLConstraint"
    element.pop("stereotype", None)
    expression = _str(data.get("expression")) or _str(data.get("constraint"))
    # v3 stores the OCL body on ``constraint``; keep ``expression`` for tolerance.
    element["constraint"] = expression
    element["expression"] = expression


def _convert_object_node(node, data, element, elements) -> None:
    node_id = element["id"]
    attr_ids: List[str] = []
    for i, row in enumerate(_as_list(data.get("attributes"))):
        if not isinstance(row, dict):
            continue
        rid = _row_id(row, node_id, "attr", i)
        child = _child_element(row, rid, "ObjectAttribute", node_id, exclude=("value",))
        value = row.get("value")
        # v3 wire form stores the slot value inside the row name: "attr = value".
        if value is not None and value != "":
            child["name"] = f"{child['name']} = {value}"
        elements[rid] = child
        attr_ids.append(rid)
    method_ids: List[str] = []
    for i, row in enumerate(_as_list(data.get("methods"))):
        if not isinstance(row, dict):
            continue
        rid = _row_id(row, node_id, "method", i)
        elements[rid] = _child_element(row, rid, "ObjectMethod", node_id)
        method_ids.append(rid)
    element["attributes"] = attr_ids
    element["methods"] = method_ids
    icon = element.pop("icon", None)
    if icon:
        icon_id = f"{node_id}-icon"
        elements[icon_id] = {
            "id": icon_id, "name": "", "type": "ObjectIcon", "owner": node_id,
            "bounds": dict(_ZERO_BOUNDS), "icon": icon,
        }


def _convert_state_node(node, data, element, elements, body_type, fallback_type) -> None:
    """State / AgentState: re-expand inline ``bodies`` / ``fallbackBodies`` rows."""
    node_id = element["id"]
    body_ids: List[str] = []
    fallback_ids: List[str] = []
    for rows, v3_type, ids, kind in (
        (data.get("bodies"), body_type, body_ids, "body"),
        (data.get("fallbackBodies"), fallback_type, fallback_ids, "fallback"),
    ):
        for i, row in enumerate(_as_list(rows)):
            if not isinstance(row, dict):
                continue
            row_type = v3_type
            # Legacy AgentState rows discriminated fallback rows with ``kind``.
            if row.get("kind") == "fallback" and v3_type == body_type:
                row_type = fallback_type
            rid = _row_id(row, node_id, kind, i)
            elements[rid] = _child_element(row, rid, row_type, node_id, exclude=("kind",))
            (fallback_ids if row_type == fallback_type else ids).append(rid)
    element["bodies"] = body_ids
    element["fallbackBodies"] = fallback_ids
    element["hasBody"] = bool(body_ids)
    element["hasFallbackBody"] = bool(fallback_ids)


def _convert_agent_intent_node(node, data, element, elements) -> None:
    node_id = element["id"]
    phrase_ids: List[str] = []
    # TS editor stores training phrases on ``training_phrases``; the spec names
    # them ``bodies``. Accept both.
    phrase_rows = _as_list(data.get("training_phrases")) + _as_list(data.get("bodies"))
    for i, row in enumerate(phrase_rows):
        if not isinstance(row, dict):
            continue
        rid = _row_id(row, node_id, "phrase", i)
        if rid in elements:
            continue
        elements[rid] = _child_element(row, rid, "AgentIntentBody", node_id)
        phrase_ids.append(rid)
    for i, row in enumerate(_as_list(data.get("entity_slots"))):
        if not isinstance(row, dict):
            continue
        rid = _row_id(row, node_id, "slot", i)
        elements[rid] = _child_element(row, rid, "AgentIntentObjectComponent", node_id)
    element["bodies"] = phrase_ids
    element.setdefault("intent_description", "")
    if not isinstance(element.get("intent_description"), str):
        element["intent_description"] = ""


def _user_row_name(row: Dict[str, Any]) -> str:
    """Mirror ``UserModelName.tsx``: rows render as ``name op value``."""
    name = _str(row.get("name"))
    if name and any(ch in name for ch in "<>="):
        return name
    op = row.get("attributeOperator")
    value = row.get("value")
    if value is not None and value != "":
        return f"{name} {op if isinstance(op, str) and op else '='} {value}"
    if isinstance(op, str) and op:
        return f"{name} {op}"
    return name


def _convert_user_node(node, data, element, elements) -> None:
    node_id = element["id"]
    attr_ids: List[str] = []
    for i, row in enumerate(_as_list(data.get("attributes"))):
        if not isinstance(row, dict):
            continue
        rid = _row_id(row, node_id, "attr", i)
        child = _child_element(row, rid, "UserModelAttribute", node_id)
        child["name"] = _user_row_name(row)
        elements[rid] = child
        attr_ids.append(rid)
    element["attributes"] = attr_ids
    icon = element.pop("icon", None)
    if icon:
        icon_id = f"{node_id}-icon"
        elements[icon_id] = {
            "id": icon_id, "name": "", "type": "UserModelIcon", "owner": node_id,
            "bounds": dict(_ZERO_BOUNDS), "icon": icon,
        }


def _convert_node(node: Dict[str, Any], abs_pos, elements: Dict[str, Any]) -> Dict[str, Any]:
    data = _as_dict(node.get("data"))
    v3_type = _v3_node_type(node.get("type"))
    element = _base_element(node, v3_type, abs_pos)

    if v3_type in _V3_CLASS_TYPES:
        _convert_class_node(node, data, element, elements)
    elif v3_type == "ClassOCLConstraint":
        _convert_ocl_node(data, element)
    elif v3_type == "ObjectName":
        _convert_object_node(node, data, element, elements)
    elif v3_type == "State":
        _convert_state_node(node, data, element, elements, "StateBody", "StateFallbackBody")
    elif v3_type == "AgentState":
        _convert_state_node(
            node, data, element, elements, "AgentStateBody", "AgentStateFallbackBody"
        )
        if not isinstance(element.get("replyType"), str):
            element["replyType"] = "text"
    elif v3_type == "AgentIntent":
        _convert_agent_intent_node(node, data, element, elements)
    elif v3_type == "UserModelName":
        _convert_user_node(node, data, element, elements)
    elif v3_type == "NNReference":
        target = data.get("referenceTarget")
        if isinstance(target, str) and target and "referencedNN" not in element:
            element["referencedNN"] = target
    return element


# ---------------------------------------------------------------------------
# Per-edge conversion
# ---------------------------------------------------------------------------


def _endpoint(edge: Dict[str, Any], data: Dict[str, Any], side: str) -> Dict[str, Any]:
    end: Dict[str, Any] = {
        "element": _str(edge.get(side)),
        "direction": _invert_handle(edge.get(f"{side}Handle")),
    }
    role = data.get(f"{side}Role")
    if isinstance(role, str):
        end["role"] = role
    multiplicity = data.get(f"{side}Multiplicity")
    if isinstance(multiplicity, str):
        end["multiplicity"] = multiplicity
    navigable = data.get(f"{side}Navigable")
    if isinstance(navigable, bool):
        end["navigable"] = navigable
    return end


def _params_to_v3(params: Any) -> Optional[Dict[str, str]]:
    """v4 canonical params is an ordered list; v3 stored ``{"0": v0, ...}``."""
    if isinstance(params, str):
        return {"0": params} if params else None
    if isinstance(params, list):
        values = [p for p in params if isinstance(p, str)]
        return {str(i): v for i, v in enumerate(values)} or None
    if isinstance(params, dict):
        return {str(k): v for k, v in params.items() if isinstance(v, str)} or None
    return None


def _agent_transition_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical v3 AgentStateTransition fields (shape #1 / #2 of the spec)."""
    custom = _as_dict(data.get("custom"))
    predefined = _as_dict(data.get("predefined"))
    transition_type = data.get("transitionType")
    is_custom = (
        transition_type == "custom"
        or data.get("condition") == "custom_transition"
        or (transition_type is None and bool(custom.get("event") or custom.get("condition")))
    )
    if is_custom:
        event = custom.get("event") or data.get("customEvent") or "WildcardEvent"
        condition = custom.get("condition")
        if not isinstance(condition, list):
            condition = data.get("customConditions") if isinstance(data.get("customConditions"), list) else []
        return {
            "transitionType": "custom",
            "predefined": {"predefinedType": ""},
            "custom": {"event": event, "condition": _copy(condition)},
        }
    pre = _copy(predefined) if predefined else {}
    if not pre.get("predefinedType"):
        flat_type = data.get("predefinedType")
        pre["predefinedType"] = flat_type if isinstance(flat_type, str) and flat_type else "when_intent_matched"
    for key in ("intentName", "fileType", "conditionValue"):
        if key not in pre and key in data:
            pre[key] = _copy(data[key])
    return {
        "transitionType": "predefined",
        "predefined": pre,
        "custom": {"condition": []},
    }


def _convert_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
    data = _as_dict(edge.get("data"))
    edge_type = _str(edge.get("type"))

    points = [p for p in _as_list(data.get("points")) if isinstance(p, dict)]
    xs = [_num(p.get("x")) for p in points]
    ys = [_num(p.get("y")) for p in points]
    min_x, min_y = (min(xs), min(ys)) if points else (0, 0)
    max_x, max_y = (max(xs), max(ys)) if points else (0, 0)

    rel: Dict[str, Any] = {
        k: _copy(v) for k, v in data.items() if k not in _EDGE_HANDLED_KEYS
    }
    name = data.get("name")
    if not isinstance(name, str) or not name:
        name = _str(data.get("label"))
    rel.update(
        {
            "id": edge.get("id"),
            "name": name,
            "type": edge_type,
            "owner": None,
            "bounds": {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y},
            "path": [{"x": x - min_x, "y": y - min_y} for x, y in zip(xs, ys)],
            "source": _endpoint(edge, data, "source"),
            "target": _endpoint(edge, data, "target"),
        }
    )
    if data.get("isManuallyLayouted") is True:
        rel["isManuallyLayouted"] = True

    if edge_type in _BPMN_EDGE_FLOW_TYPES:
        rel["type"] = "BPMNFlow"
        rel["flowType"] = _str(data.get("flowType")) or _BPMN_EDGE_FLOW_TYPES[edge_type]
    elif edge_type == "BPMNFlow":
        rel["flowType"] = _str(data.get("flowType")) or "sequence"
    elif edge_type in _EDGE_TYPE_V4_TO_V3:
        rel["type"] = _EDGE_TYPE_V4_TO_V3[edge_type]
    elif isinstance(data.get("flowType"), str):
        rel["flowType"] = data["flowType"]

    if edge_type == "AgentStateTransition":
        rel.update(_agent_transition_fields(data))
        for legacy_key in ("predefinedType", "intentName", "fileType", "conditionValue",
                           "condition", "customEvent", "customConditions",
                           "variable", "operator", "targetValue"):
            rel.pop(legacy_key, None)

    params = _params_to_v3(data.get("params"))
    if params is not None:
        rel["params"] = params
    return rel


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------


def _fold_legacy_child_nodes(elements: Dict[str, Any]) -> None:
    """Link legacy v4 child *nodes* (bodies/phrases kept as nodes with a
    ``parentId``) into their parent's v3 id lists."""
    for eid, el in list(elements.items()):
        owner = el.get("owner")
        parent = elements.get(owner) if isinstance(owner, str) else None
        if not isinstance(parent, dict):
            continue
        etype, ptype = el.get("type"), parent.get("type")
        target_key = None
        if ptype == "State" and etype == "StateBody":
            target_key = "bodies"
        elif ptype == "State" and etype == "StateFallbackBody":
            target_key = "fallbackBodies"
        elif ptype == "AgentState" and etype == "AgentStateBody":
            target_key = "bodies"
        elif ptype == "AgentState" and etype == "AgentStateFallbackBody":
            target_key = "fallbackBodies"
        elif ptype == "AgentIntent" and etype == "AgentIntentBody":
            target_key = "bodies"
        elif ptype == "AgentIntent" and etype == "AgentIntentDescription":
            if not parent.get("intent_description"):
                parent["intent_description"] = el.get("name") or ""
        elif ptype in _V3_CLASS_TYPES and etype == "ClassAttribute":
            target_key = "attributes"
        elif ptype in _V3_CLASS_TYPES and etype == "ClassMethod":
            target_key = "methods"
        elif ptype in ("ObjectName", "UserModelName") and etype in ("ObjectAttribute", "UserModelAttribute"):
            target_key = "attributes"
        if target_key is None:
            continue
        ids = parent.get(target_key)
        if not isinstance(ids, list):
            ids = []
            parent[target_key] = ids
        if eid not in ids:
            ids.append(eid)
        if ptype in ("State", "AgentState"):
            parent["hasBody"] = bool(parent.get("bodies"))
            parent["hasFallbackBody"] = bool(parent.get("fallbackBodies"))


def _synthesize_agent_initial(
    nodes: List[Dict[str, Any]], elements: Dict[str, Any], relationships: Dict[str, Any]
) -> None:
    """v4 marks the agent entry state with ``data.initial``; v3 used a
    ``StateInitialNode`` wired by an ``AgentStateTransitionInit`` edge."""
    initial_node = next(
        (
            n for n in nodes
            if n.get("type") == "AgentState" and _as_dict(n.get("data")).get("initial") is True
        ),
        None,
    )
    if initial_node is None:
        return
    state_id = initial_node.get("id")
    already_wired = any(
        isinstance(rel, dict)
        and _as_dict(rel.get("target")).get("element") == state_id
        and (
            rel.get("type") == "AgentStateTransitionInit"
            or _as_dict(elements.get(_as_dict(rel.get("source")).get("element"))).get("type")
            == "StateInitialNode"
        )
        for rel in relationships.values()
    )
    if already_wired:
        return
    state_bounds = _as_dict(_as_dict(elements.get(state_id)).get("bounds"))
    marker_id = f"{state_id}__initial"
    edge_id = f"{state_id}__initial-edge"
    elements[marker_id] = {
        "id": marker_id, "name": "", "type": "StateInitialNode", "owner": None,
        "bounds": {
            "x": _num(state_bounds.get("x")) - 300,
            "y": _num(state_bounds.get("y")) + 20,
            "width": 45, "height": 45,
        },
    }
    relationships[edge_id] = {
        "id": edge_id, "name": "", "type": "AgentStateTransitionInit", "owner": None,
        "bounds": dict(_ZERO_BOUNDS), "path": [],
        "source": {"element": marker_id, "direction": "Right"},
        "target": {"element": state_id, "direction": "Left"},
    }


def _convert_components(components: Any) -> Dict[str, Any]:
    """Agent components kept off-canvas (``model.components``): entries shaped
    like v4 nodes are converted, anything else is passed through."""
    out: Dict[str, Any] = {}
    entries = (
        list(components.items())
        if isinstance(components, dict)
        else [(None, c) for c in components] if isinstance(components, list) else []
    )
    for key, comp in entries:
        if not isinstance(comp, dict):
            continue
        comp_id = comp.get("id") if isinstance(comp.get("id"), str) else key
        if not isinstance(comp_id, str):
            continue
        if isinstance(comp.get("data"), dict):
            extra: Dict[str, Any] = {}
            pos = _as_dict(comp.get("position"))
            element = _convert_node(
                {**comp, "id": comp_id}, (_num(pos.get("x")), _num(pos.get("y"))), extra
            )
            if "position" not in comp:
                # Off-canvas components carry no bounds in the v3 format either.
                element.pop("bounds", None)
            out.update(extra)
            out[comp_id] = element
        else:
            out[comp_id] = _copy(comp)
    return out


def _compute_size(elements: Dict[str, Any]) -> Dict[str, float]:
    width = height = 0.0
    for el in elements.values():
        b = _as_dict(el.get("bounds"))
        width = max(width, _num(b.get("x")) + _num(b.get("width")))
        height = max(height, _num(b.get("y")) + _num(b.get("height")))
    return {"width": width, "height": height}


def convert_v4_model_to_v3(model: Dict[str, Any], diagram_type: Optional[str] = None) -> Dict[str, Any]:
    """Convert one v4 diagram model to the v3 ``elements``/``relationships`` shape.

    *diagram_type* is the project-snapshot bucket key (``"BPMN"``,
    ``"ClassDiagram"``…); it only matters when the model has no ``type``.
    Non-v4 input and non-UML models (GUI / Quantum) are returned unchanged.
    """
    if not is_v4_model(model):
        return model
    model_type = model.get("type") if isinstance(model.get("type"), str) else diagram_type
    if model_type in NON_UML_DIAGRAM_TYPES or diagram_type in NON_UML_DIAGRAM_TYPES:
        return model

    nodes = [n for n in _as_list(model.get("nodes")) if isinstance(n, dict) and isinstance(n.get("id"), str)]
    edges = [e for e in _as_list(model.get("edges")) if isinstance(e, dict) and isinstance(e.get("id"), str)]

    abs_positions = _absolute_positions(nodes)
    elements: Dict[str, Any] = {}
    # Inline rows are emitted as the parent is converted; an explicit node that
    # shares a row's id is assigned afterwards and wins.
    for node in nodes:
        element = _convert_node(node, abs_positions.get(node["id"], (0, 0)), elements)
        elements[node["id"]] = element
    _fold_legacy_child_nodes(elements)

    relationships: Dict[str, Any] = {}
    for edge in edges:
        relationships[edge["id"]] = _convert_edge(edge)

    if model_type == "AgentDiagram" or any(n.get("type") == "AgentState" for n in nodes):
        _synthesize_agent_initial(nodes, elements, relationships)

    out: Dict[str, Any] = {
        k: _copy(v)
        for k, v in model.items()
        if k not in {"nodes", "edges", "version", "components", "referenceDiagramData"}
    }
    size = model.get("size")
    out.update(
        {
            "version": V3_VERSION,
            "type": model_type or "",
            "size": _copy(size) if isinstance(size, dict) else _compute_size(elements),
            "elements": elements,
            "relationships": relationships,
            "interactive": _copy(model["interactive"])
            if isinstance(model.get("interactive"), dict)
            else {"elements": {}, "relationships": {}},
            "assessments": _copy(model["assessments"]) if isinstance(model.get("assessments"), dict) else {},
        }
    )
    if "components" in model:
        out["components"] = _convert_components(model.get("components"))
    if "referenceDiagramData" in model:
        out["referenceDiagramData"] = normalize_model(model.get("referenceDiagramData"))
    return out


def normalize_model(model: Any, diagram_type: Optional[str] = None) -> Any:
    """Return a v3 view of *model*: v4 is converted, everything else is returned as-is.

    A v3 model that embeds a v4 ``referenceDiagramData`` gets that reference
    converted (in a shallow copy). Conversion errors are logged and the
    original model is returned, so a malformed diagram never breaks a request.
    """
    if not isinstance(model, dict):
        return model
    if diagram_type in NON_UML_DIAGRAM_TYPES or model.get("type") in NON_UML_DIAGRAM_TYPES:
        return model
    try:
        if is_v4_model(model):
            return convert_v4_model_to_v3(model, diagram_type)
        ref = model.get("referenceDiagramData")
        if is_v4_model(ref):
            return {**model, "referenceDiagramData": convert_v4_model_to_v3(ref)}
    except Exception:  # pragma: no cover — defensive: keep the request alive
        logger.warning(
            "[v4_to_v3] Could not convert %s model; passing it through unchanged",
            diagram_type or model.get("type") or "diagram",
            exc_info=True,
        )
    return model


def _normalize_entry(entry: Any, diagram_type: str) -> Any:
    """One project-snapshot tab: ``{id, title, model}`` (or a bare model)."""
    if not isinstance(entry, dict):
        return entry
    if isinstance(entry.get("model"), dict):
        converted = normalize_model(entry["model"], diagram_type)
        return entry if converted is entry["model"] else {**entry, "model": converted}
    if is_v4_model(entry):
        return normalize_model(entry, diagram_type)
    return entry


def normalize_project_snapshot(snapshot: Any) -> Any:
    """Convert every v4 diagram model in a project snapshot to v3.

    Handles both the multi-tab (``diagrams[T] = [tab, ...]``) and the legacy
    single-dict (``diagrams[T] = tab``) layouts. v3 models and the non-UML
    ``GUINoCodeDiagram`` / ``QuantumCircuitDiagram`` buckets are left untouched.
    Returns the input object itself when nothing needed converting, otherwise a
    new snapshot (the input is never mutated).
    """
    if not isinstance(snapshot, dict):
        return snapshot
    diagrams = snapshot.get("diagrams")
    if not isinstance(diagrams, dict):
        return snapshot

    new_diagrams: Dict[str, Any] = {}
    changed = False
    for diagram_type, payload in diagrams.items():
        if diagram_type in NON_UML_DIAGRAM_TYPES:
            new_diagrams[diagram_type] = payload
            continue
        if isinstance(payload, list):
            new_list = [_normalize_entry(item, diagram_type) for item in payload]
            if any(a is not b for a, b in zip(new_list, payload)):
                changed = True
                new_diagrams[diagram_type] = new_list
            else:
                new_diagrams[diagram_type] = payload
        else:
            new_entry = _normalize_entry(payload, diagram_type)
            changed = changed or new_entry is not payload
            new_diagrams[diagram_type] = new_entry

    if not changed:
        return snapshot
    return {**snapshot, "diagrams": new_diagrams}
