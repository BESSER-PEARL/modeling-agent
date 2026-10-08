"""Tests for the inbound v4 (React Flow) -> v3 (Apollon) model normalizer.

The React Flow frontend sends ``{nodes, edges}`` diagram models; everything
downstream of ``protocol/adapters.py`` reads the v3 ``{elements,
relationships}`` shape. ``protocol/v4_to_v3.py`` is the Python inverse of the
frontend's ``versionConverter.ts`` migrator. The v4 fixtures below are
hand-written to match what that migrator emits for the paired v3 fixtures.
"""

import copy

import pytest

from protocol.adapters import parse_v2_payload
from protocol.v4_to_v3 import (
    convert_v4_model_to_v3,
    is_v4_model,
    normalize_model,
    normalize_project_snapshot,
)
from utilities.model_context import (
    compact_model_summary,
    detailed_model_summary,
    is_diagram_nontrivial,
)
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

from tests.conftest import MINIMAL_CLASS_MODEL, make_v2_payload


# ---------------------------------------------------------------------------
# Representative class diagram: v3 (Apollon) and its v4 (React Flow) migration
# ---------------------------------------------------------------------------

V3_CLASS = {
    "version": "3.0.0",
    "type": "ClassDiagram",
    "size": {"width": 900, "height": 600},
    "interactive": {"elements": {}, "relationships": {}},
    "assessments": {},
    "elements": {
        "pkg": {"id": "pkg", "name": "shop", "type": "Package", "owner": None,
                "bounds": {"x": 10, "y": 20, "width": 400, "height": 300}},
        "person": {"id": "person", "name": "Person", "type": "AbstractClass", "owner": None,
                   "bounds": {"x": 500, "y": 20, "width": 200, "height": 110},
                   "attributes": ["person-name"], "methods": ["person-greet"]},
        "person-name": {"id": "person-name", "name": "name", "type": "ClassAttribute",
                        "owner": "person", "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                        "visibility": "public", "attributeType": "str"},
        "person-greet": {"id": "person-greet", "name": "greet", "type": "ClassMethod",
                         "owner": "person", "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                         "visibility": "public", "attributeType": "str", "returnType": "str",
                         "parameters": [{"id": "p1", "name": "greeting", "parameterType": "str"}]},
        "customer": {"id": "customer", "name": "Customer", "type": "Class", "owner": None,
                     "bounds": {"x": 500, "y": 300, "width": 200, "height": 80},
                     "attributes": ["customer-id"], "methods": []},
        "customer-id": {"id": "customer-id", "name": "customerId", "type": "ClassAttribute",
                        "owner": "customer", "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                        "visibility": "private", "attributeType": "int", "isId": True},
        "order": {"id": "order", "name": "Order", "type": "Class", "owner": "pkg",
                  "bounds": {"x": 60, "y": 80, "width": 200, "height": 80},
                  "attributes": ["order-total"], "methods": []},
        "order-total": {"id": "order-total", "name": "total", "type": "ClassAttribute",
                        "owner": "order", "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                        "visibility": "public", "attributeType": "float"},
        "status": {"id": "status", "name": "Status", "type": "Enumeration", "owner": None,
                   "bounds": {"x": 800, "y": 20, "width": 160, "height": 80},
                   "attributes": ["st-a", "st-i"], "methods": []},
        "st-a": {"id": "st-a", "name": "ACTIVE", "type": "ClassAttribute", "owner": "status",
                 "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                 "visibility": "public", "attributeType": "str"},
        "st-i": {"id": "st-i", "name": "INACTIVE", "type": "ClassAttribute", "owner": "status",
                 "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                 "visibility": "public", "attributeType": "str"},
        "ocl": {"id": "ocl", "name": "positiveTotal", "type": "ClassOCLConstraint", "owner": None,
                "bounds": {"x": 60, "y": 400, "width": 240, "height": 60},
                "constraint": "context Order inv positiveTotal: self.total > 0"},
    },
    "relationships": {
        "inh": {"id": "inh", "name": "", "type": "ClassInheritance", "owner": None,
                "bounds": {"x": 600, "y": 130, "width": 0, "height": 170},
                "path": [{"x": 0, "y": 170}, {"x": 0, "y": 0}],
                "source": {"element": "customer", "direction": "Up"},
                "target": {"element": "person", "direction": "Down"}},
        "assoc": {"id": "assoc", "name": "places", "type": "ClassBidirectional", "owner": None,
                  "bounds": {"x": 270, "y": 120, "width": 230, "height": 220},
                  "path": [{"x": 230, "y": 220}, {"x": 0, "y": 0}],
                  "source": {"element": "customer", "direction": "Left", "role": "customer",
                             "multiplicity": "1", "navigable": False},
                  "target": {"element": "order", "direction": "Right", "role": "orders",
                             "multiplicity": "0..*", "navigable": True}},
        "ocl-link": {"id": "ocl-link", "name": "", "type": "ClassOCLLink", "owner": None,
                     "bounds": {"x": 160, "y": 160, "width": 0, "height": 240},
                     "path": [{"x": 0, "y": 240}, {"x": 0, "y": 0}],
                     "source": {"element": "ocl", "direction": "Up"},
                     "target": {"element": "order", "direction": "Down"}},
    },
}


def _row(**kw):
    return kw


V4_CLASS = {
    "version": "4.0.0",
    "id": "class-tab-1",
    "title": "Shop",
    "type": "ClassDiagram",
    "size": {"width": 900, "height": 600},
    "assessments": {},
    "nodes": [
        {"id": "pkg", "type": "package", "position": {"x": 10, "y": 20},
         "width": 400, "height": 300, "measured": {"width": 400, "height": 300},
         "data": {"name": "shop"}},
        {"id": "person", "type": "class", "position": {"x": 500, "y": 20},
         "width": 200, "height": 110, "measured": {"width": 200, "height": 110},
         "data": {"name": "Person", "stereotype": "Abstract",
                  "attributes": [_row(id="person-name", name="name", visibility="public",
                                      attributeType="str")],
                  "methods": [_row(id="person-greet", name="greet", visibility="public",
                                   attributeType="str", returnType="str",
                                   parameters=[{"id": "p1", "name": "greeting",
                                                "parameterType": "str"}])]}},
        {"id": "customer", "type": "class", "position": {"x": 500, "y": 300},
         "width": 200, "height": 80, "measured": {"width": 200, "height": 80},
         "data": {"name": "Customer",
                  "attributes": [_row(id="customer-id", name="customerId", visibility="private",
                                      attributeType="int", isId=True)],
                  "methods": []}},
        # Child of the package: v4 position is RELATIVE to the parent.
        {"id": "order", "type": "class", "parentId": "pkg", "position": {"x": 50, "y": 60},
         "width": 200, "height": 80, "measured": {"width": 200, "height": 80},
         "data": {"name": "Order",
                  "attributes": [_row(id="order-total", name="total", visibility="public",
                                      attributeType="float")],
                  "methods": []}},
        {"id": "status", "type": "class", "position": {"x": 800, "y": 20},
         "width": 160, "height": 80, "measured": {"width": 160, "height": 80},
         "data": {"name": "Status", "stereotype": "Enumeration",
                  "attributes": [_row(id="st-a", name="ACTIVE", visibility="public", attributeType="str"),
                                 _row(id="st-i", name="INACTIVE", visibility="public", attributeType="str")],
                  "methods": []}},
        {"id": "ocl", "type": "ClassOCLConstraint", "position": {"x": 60, "y": 400},
         "width": 240, "height": 60, "measured": {"width": 240, "height": 60},
         "data": {"name": "positiveTotal",
                  "expression": "context Order inv positiveTotal: self.total > 0"}},
    ],
    "edges": [
        {"id": "inh", "source": "customer", "target": "person", "type": "ClassInheritance",
         "sourceHandle": "top", "targetHandle": "bottom",
         "data": {"label": "", "sourceRole": "", "targetRole": "", "sourceMultiplicity": "",
                  "targetMultiplicity": "", "isManuallyLayouted": False, "messages": [],
                  "points": [{"x": 600, "y": 300}, {"x": 600, "y": 130}]}},
        {"id": "assoc", "source": "customer", "target": "order", "type": "ClassBidirectional",
         "sourceHandle": "left", "targetHandle": "right",
         "data": {"label": "places", "name": "places",
                  "sourceRole": "customer", "sourceMultiplicity": "1",
                  "targetRole": "orders", "targetMultiplicity": "0..*",
                  "sourceNavigable": False, "targetNavigable": True,
                  "isManuallyLayouted": False, "messages": [],
                  "points": [{"x": 500, "y": 340}, {"x": 270, "y": 120}]}},
        {"id": "ocl-link", "source": "ocl", "target": "order", "type": "ClassOCLLink",
         "sourceHandle": "top", "targetHandle": "bottom",
         "data": {"label": "", "points": [{"x": 160, "y": 400}, {"x": 160, "y": 160}]}},
    ],
}


_MEMBER_FIELDS = ("name", "type", "owner", "visibility", "attributeType", "isId",
                  "returnType", "parameters")


def _class_semantics(model):
    """Structural projection of a v3 class model used for round-trip equality."""
    els = model["elements"]
    out = {}
    for eid, el in els.items():
        entry = {"type": el["type"], "name": el["name"], "owner": el.get("owner")}
        if el["type"] in ("Class", "AbstractClass", "Interface", "Enumeration", "Package",
                          "ClassOCLConstraint"):
            entry["bounds"] = el["bounds"]
        if el["type"] in ("Class", "AbstractClass", "Interface", "Enumeration"):
            entry["attributes"] = list(el.get("attributes") or [])
            entry["methods"] = list(el.get("methods") or [])
        if el["type"] in ("ClassAttribute", "ClassMethod"):
            entry.update({k: el.get(k) for k in _MEMBER_FIELDS})
        if el["type"] == "ClassOCLConstraint":
            entry["constraint"] = el.get("constraint")
        out[eid] = entry
    rels = {}
    for rid, rel in model["relationships"].items():
        ends = []
        for side in ("source", "target"):
            end = rel[side]
            ends.append({
                "element": end["element"],
                "direction": end.get("direction"),
                "role": end.get("role") or "",
                "multiplicity": end.get("multiplicity") or "",
                "navigable": end.get("navigable"),
            })
        rels[rid] = {"type": rel["type"], "name": rel.get("name") or "", "ends": ends,
                     "bounds": rel["bounds"], "path": rel["path"]}
    return out, rels


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetection:
    def test_v4_by_version_or_nodes(self):
        assert is_v4_model({"version": "4.0.0"})
        assert is_v4_model({"nodes": [], "edges": []})
        assert is_v4_model(V4_CLASS)

    def test_v3_and_non_models_are_not_v4(self):
        assert not is_v4_model(V3_CLASS)
        assert not is_v4_model(MINIMAL_CLASS_MODEL)
        assert not is_v4_model({"pages": []})
        assert not is_v4_model(None)
        assert not is_v4_model([])

    def test_v3_model_is_returned_unchanged(self):
        assert convert_v4_model_to_v3(V3_CLASS) is V3_CLASS
        assert normalize_model(V3_CLASS, "ClassDiagram") is V3_CLASS


# ---------------------------------------------------------------------------
# ClassDiagram
# ---------------------------------------------------------------------------

class TestClassDiagram:
    def test_round_trip_matches_v3(self):
        converted = convert_v4_model_to_v3(V4_CLASS)
        assert converted["version"].startswith("3.")
        assert _class_semantics(converted) == _class_semantics(V3_CLASS)

    def test_stereotypes_map_to_v3_types(self):
        v3 = convert_v4_model_to_v3(V4_CLASS)
        els = v3["elements"]
        assert els["person"]["type"] == "AbstractClass"
        assert els["status"]["type"] == "Enumeration"
        assert els["customer"]["type"] == "Class"
        assert "stereotype" not in els["person"]

    def test_stereotype_is_case_insensitive_and_interface_supported(self):
        model = {"version": "4.0.0", "type": "ClassDiagram", "edges": [], "nodes": [
            {"id": "i", "type": "class", "position": {"x": 0, "y": 0},
             "data": {"name": "Payable", "stereotype": "interface", "attributes": [], "methods": []}},
            {"id": "c", "type": "class", "position": {"x": 0, "y": 0},
             "data": {"name": "Svc", "stereotype": "service", "attributes": [], "methods": []}},
        ]}
        els = convert_v4_model_to_v3(model)["elements"]
        assert els["i"]["type"] == "Interface"
        # A freeform stereotype survives on a plain Class.
        assert els["c"]["type"] == "Class" and els["c"]["stereotype"] == "service"

    def test_child_position_becomes_absolute(self):
        order = convert_v4_model_to_v3(V4_CLASS)["elements"]["order"]
        assert order["owner"] == "pkg"
        assert order["bounds"] == {"x": 60, "y": 80, "width": 200, "height": 80}

    def test_association_ends_carry_roles_multiplicities_navigability(self):
        rel = convert_v4_model_to_v3(V4_CLASS)["relationships"]["assoc"]
        assert rel["name"] == "places"
        assert rel["source"] == {"element": "customer", "direction": "Left", "role": "customer",
                                 "multiplicity": "1", "navigable": False}
        assert rel["target"] == {"element": "order", "direction": "Right", "role": "orders",
                                 "multiplicity": "0..*", "navigable": True}
        # Absolute v4 points become bounds + relative v3 path.
        assert rel["bounds"] == {"x": 270, "y": 120, "width": 230, "height": 220}
        assert rel["path"] == [{"x": 230, "y": 220}, {"x": 0, "y": 0}]
        # Edge-only render state is not leaked onto the relationship.
        assert "messages" not in rel and "label" not in rel

    def test_navigability_absent_when_v4_has_none(self):
        rel = convert_v4_model_to_v3(V4_CLASS)["relationships"]["inh"]
        assert "navigable" not in rel["source"] and "navigable" not in rel["target"]

    def test_owned_ocl_rows_and_legacy_ocl_class(self):
        model = {"version": "4.0.0", "type": "ClassDiagram", "edges": [], "nodes": [
            {"id": "c", "type": "class", "position": {"x": 0, "y": 0},
             "data": {"name": "Order", "attributes": [], "methods": [],
                      "oclConstraints": [{"id": "o1", "name": "c1",
                                          "expression": "context Order inv: true"}]}},
            {"id": "legacy", "type": "class", "position": {"x": 0, "y": 0},
             "data": {"name": "c2", "stereotype": "oclConstraint",
                      "expression": "context Order inv: false"}},
            {"id": "lower", "type": "classoclconstraint", "position": {"x": 0, "y": 0},
             "data": {"name": "c3", "constraint": "context Order inv: 1 = 1"}},
        ]}
        els = convert_v4_model_to_v3(model)["elements"]
        assert els["o1"]["type"] == "ClassOCLConstraint" and els["o1"]["owner"] == "c"
        assert els["o1"]["constraint"] == "context Order inv: true"
        assert els["legacy"]["type"] == "ClassOCLConstraint"
        assert els["legacy"]["constraint"] == "context Order inv: false"
        assert els["lower"]["type"] == "ClassOCLConstraint"
        assert els["lower"]["constraint"] == "context Order inv: 1 = 1"

    def test_input_is_not_mutated(self):
        before = copy.deepcopy(V4_CLASS)
        out = convert_v4_model_to_v3(V4_CLASS)
        assert V4_CLASS == before
        # Output does not alias input containers.
        out["elements"]["person-greet"]["parameters"][0]["name"] = "changed"
        assert V4_CLASS == before

    def test_partial_nodes_are_tolerated(self):
        model = {"version": "4.0.0", "nodes": [
            {"id": "a", "type": "class"},                      # no data / position / size
            {"id": "b", "type": "class", "data": {"name": "B", "attributes": [{"name": "x"}]}},
            {"type": "class"},                                 # no id -> skipped
            "garbage",
        ], "edges": [{"id": "e", "type": "ClassBidirectional", "source": "a"}]}
        v3 = convert_v4_model_to_v3(model, "ClassDiagram")
        assert v3["type"] == "ClassDiagram"
        assert v3["elements"]["a"]["name"] == "" and v3["elements"]["a"]["attributes"] == []
        attr_id = v3["elements"]["b"]["attributes"][0]
        assert v3["elements"][attr_id]["name"] == "x"
        assert v3["relationships"]["e"]["target"]["element"] == ""


# ---------------------------------------------------------------------------
# ObjectDiagram
# ---------------------------------------------------------------------------

class TestObjectDiagram:
    MODEL = {"version": "4.0.0", "type": "ObjectDiagram", "nodes": [
        {"id": "o1", "type": "objectName", "position": {"x": 10, "y": 10}, "width": 180, "height": 70,
         "data": {"name": "alice", "classId": "customer", "className": "Customer", "icon": "<svg/>",
                  "attributes": [{"id": "a1", "name": "customerId", "value": "7",
                                  "attributeId": "customer-id", "attributeType": "int"},
                                 {"id": "a2", "name": "nickname"}],
                  "methods": []}},
        {"id": "o2", "type": "objectName", "position": {"x": 300, "y": 10},
         "data": {"name": "o42", "className": "Order", "attributes": []}},
    ], "edges": [
        {"id": "l1", "type": "ObjectLink", "source": "o1", "target": "o2",
         "sourceHandle": "right", "targetHandle": "left",
         "data": {"name": "places", "associationId": "assoc", "points": []}},
    ]}

    def test_objects_attributes_and_links(self):
        v3 = convert_v4_model_to_v3(self.MODEL)
        els = v3["elements"]
        assert els["o1"]["type"] == "ObjectName"
        assert els["o1"]["classId"] == "customer" and els["o1"]["className"] == "Customer"
        assert els["o1"]["attributes"] == ["a1", "a2"]
        # v3 wire form keeps the slot value in the row name.
        assert els["a1"]["name"] == "customerId = 7"
        assert els["a1"]["attributeId"] == "customer-id" and els["a1"]["owner"] == "o1"
        assert els["a2"]["name"] == "nickname"
        assert els["o1-icon"]["type"] == "ObjectIcon" and els["o1-icon"]["icon"] == "<svg/>"
        link = v3["relationships"]["l1"]
        assert link["type"] == "ObjectLink" and link["associationId"] == "assoc"
        assert link["source"]["element"] == "o1" and link["target"]["element"] == "o2"


# ---------------------------------------------------------------------------
# StateMachineDiagram
# ---------------------------------------------------------------------------

class TestStateMachineDiagram:
    MODEL = {"version": "4.0.0", "type": "StateMachineDiagram", "nodes": [
        {"id": "init", "type": "StateInitialNode", "position": {"x": 0, "y": 0}, "data": {"name": ""}},
        {"id": "idle", "type": "State", "position": {"x": 100, "y": 0}, "width": 160, "height": 100,
         "data": {"name": "Idle", "bodies": [{"id": "b1", "name": "wait()"}],
                  "fallbackBodies": [{"id": "f1", "name": "on_error()"}]}},
        {"id": "busy", "type": "State", "position": {"x": 400, "y": 0}, "data": {"name": "Busy"}},
        # Legacy shape: body kept as a child node with parentId.
        {"id": "b2", "type": "StateBody", "parentId": "busy", "position": {"x": 0, "y": 40},
         "data": {"name": "work()"}},
        {"id": "code", "type": "StateCodeBlock", "position": {"x": 0, "y": 300},
         "data": {"name": "helpers", "code": "def f(): pass", "language": "python"}},
        {"id": "end", "type": "StateFinalNode", "position": {"x": 700, "y": 0}, "data": {"name": ""}},
    ], "edges": [
        {"id": "t0", "type": "StateTransition", "source": "init", "target": "idle",
         "data": {"points": []}},
        {"id": "t1", "type": "StateTransition", "source": "idle", "target": "busy",
         "data": {"name": "start", "guard": "ready", "params": ["job", "prio"], "points": []}},
        {"id": "t2", "type": "StateTransition", "source": "busy", "target": "end",
         "data": {"name": "done", "points": []}},
    ]}

    def test_states_bodies_and_transitions(self):
        v3 = convert_v4_model_to_v3(self.MODEL)
        els, rels = v3["elements"], v3["relationships"]
        assert els["idle"]["bodies"] == ["b1"] and els["idle"]["fallbackBodies"] == ["f1"]
        assert els["idle"]["hasBody"] and els["idle"]["hasFallbackBody"]
        assert els["b1"] == {"id": "b1", "name": "wait()", "type": "StateBody", "owner": "idle",
                             "bounds": {"x": 0, "y": 0, "width": 0, "height": 0}}
        assert els["f1"]["type"] == "StateFallbackBody"
        # Legacy child node is linked into its parent's bodies list.
        assert els["b2"]["owner"] == "busy" and els["busy"]["bodies"] == ["b2"]
        assert els["b2"]["bounds"]["x"] == 400  # absolute
        assert els["code"]["code"] == "def f(): pass"
        assert rels["t1"]["guard"] == "ready"
        assert rels["t1"]["params"] == {"0": "job", "1": "prio"}
        assert "params" not in rels["t2"]

    def test_summaries(self):
        v3 = convert_v4_model_to_v3(self.MODEL)
        assert is_diagram_nontrivial(v3, "StateMachineDiagram")
        assert "2 state(s)" in compact_model_summary(v3, "StateMachineDiagram")
        detail = detailed_model_summary(v3, "StateMachineDiagram")
        assert "Idle | body: wait() | fallback: on_error()" in detail
        assert "Transition: Idle -> Busy start [ready]" in detail


# ---------------------------------------------------------------------------
# AgentDiagram
# ---------------------------------------------------------------------------

class TestAgentDiagram:
    MODEL = {"version": "4.0.0", "type": "AgentDiagram", "nodes": [
        {"id": "s1", "type": "AgentState", "position": {"x": 400, "y": 200},
         "data": {"name": "Greeting", "initial": True, "replyType": "text",
                  "bodies": [{"id": "r1", "name": "Hello!", "replyType": "text"}],
                  "fallbackBodies": [{"id": "r2", "name": "Sorry?", "replyType": "text"}]}},
        {"id": "s2", "type": "AgentState", "position": {"x": 800, "y": 200},
         "data": {"name": "Farewell", "bodies": [{"id": "r3", "name": "Bye", "replyType": "llm",
                                                   "system_message": "be nice"}]}},
        {"id": "i1", "type": "AgentIntent", "position": {"x": 0, "y": 0},
         "data": {"name": "bye_intent", "intent_description": "user leaves",
                  "training_phrases": [{"id": "p1", "name": "bye"}, {"id": "p2", "name": "see you"}],
                  "entity_slots": [{"id": "e1", "name": "slot", "entity": "city"}]}},
        {"id": "llm", "type": "AgentLLM", "position": {"x": 0, "y": 500},
         "data": {"name": "gpt", "provider": "openai", "parameters": {}}},
        {"id": "rag", "type": "AgentRagElement", "position": {"x": 0, "y": 600},
         "data": {"name": "docs", "ragDatabaseName": "docs_db"}},
    ], "edges": [
        {"id": "t1", "type": "AgentStateTransition", "source": "s1", "target": "s2",
         "data": {"transitionType": "predefined",
                  "predefined": {"predefinedType": "when_intent_matched", "intentName": "bye_intent"},
                  "points": []}},
        {"id": "t2", "type": "AgentStateTransition", "source": "s2", "target": "s1",
         "data": {"transitionType": "custom",
                  "custom": {"event": "ReceiveTextEvent", "condition": ["len(msg) > 0"]},
                  "points": []}},
    ]}

    def test_states_intents_components_and_transitions(self):
        v3 = convert_v4_model_to_v3(self.MODEL)
        els, rels = v3["elements"], v3["relationships"]
        assert els["s1"]["bodies"] == ["r1"] and els["s1"]["fallbackBodies"] == ["r2"]
        assert els["r1"]["type"] == "AgentStateBody" and els["r1"]["owner"] == "s1"
        assert els["r2"]["type"] == "AgentStateFallbackBody"
        assert els["r3"]["replyType"] == "llm" and els["r3"]["system_message"] == "be nice"
        assert els["s2"]["replyType"] == "text"  # default
        assert "initial" not in els["s1"]
        assert els["i1"]["bodies"] == ["p1", "p2"]
        assert els["p1"]["type"] == "AgentIntentBody" and els["p1"]["owner"] == "i1"
        assert els["e1"]["type"] == "AgentIntentObjectComponent" and els["e1"]["entity"] == "city"
        assert els["i1"]["intent_description"] == "user leaves"
        assert els["llm"]["provider"] == "openai"
        assert els["rag"]["ragDatabaseName"] == "docs_db"
        assert rels["t1"]["transitionType"] == "predefined"
        assert rels["t1"]["predefined"]["intentName"] == "bye_intent"
        assert rels["t1"]["custom"] == {"condition": []}
        assert rels["t2"]["custom"] == {"event": "ReceiveTextEvent", "condition": ["len(msg) > 0"]}
        assert rels["t2"]["predefined"] == {"predefinedType": ""}

    def test_initial_state_is_re_synthesized(self):
        v3 = convert_v4_model_to_v3(self.MODEL)
        marker = v3["elements"]["s1__initial"]
        assert marker["type"] == "StateInitialNode"
        init = v3["relationships"]["s1__initial-edge"]
        assert init["type"] == "AgentStateTransitionInit"
        assert init["source"]["element"] == "s1__initial" and init["target"]["element"] == "s1"

    def test_intent_bodies_spec_spelling_is_accepted(self):
        model = {"version": "4.0.0", "type": "AgentDiagram", "edges": [], "nodes": [
            {"id": "i", "type": "AgentIntent", "data": {"name": "hi", "bodies": [{"id": "x", "name": "hello"}]}},
        ]}
        els = convert_v4_model_to_v3(model)["elements"]
        assert els["i"]["bodies"] == ["x"] and els["x"]["type"] == "AgentIntentBody"
        assert els["i"]["intent_description"] == ""

    def test_off_canvas_components_are_converted(self):
        model = {"version": "4.0.0", "type": "AgentDiagram", "nodes": [], "edges": [],
                 "components": [{"id": "i", "type": "AgentIntent",
                                 "data": {"name": "hi", "training_phrases": [{"id": "x", "name": "hello"}]}}]}
        v3 = convert_v4_model_to_v3(model)
        comps = v3["components"]
        assert comps["i"]["type"] == "AgentIntent" and "bounds" not in comps["i"]
        assert comps["x"]["type"] == "AgentIntentBody" and comps["x"]["owner"] == "i"
        assert "hi" in detailed_model_summary(v3, "AgentDiagram")

    def test_summaries(self):
        v3 = convert_v4_model_to_v3(self.MODEL)
        assert is_diagram_nontrivial(v3, "AgentDiagram")
        detail = detailed_model_summary(v3, "AgentDiagram")
        assert "States (2): Greeting, Farewell" in detail
        assert "Greeting | replies: Hello!" in detail
        assert "bye_intent | training phrases: bye, see you" in detail
        assert "LLMs (1): gpt" in detail
        assert "Greeting -> Farewell (on intent: bye_intent)" in detail


# ---------------------------------------------------------------------------
# UserDiagram
# ---------------------------------------------------------------------------

class TestUserDiagram:
    def test_profile_boxes_and_criteria(self):
        model = {"version": "4.0.0", "type": "UserDiagram", "nodes": [
            {"id": "u", "type": "UserModelName", "position": {"x": 0, "y": 0},
             "data": {"name": "adult", "className": "User", "classId": "cls-user", "icon": "<svg/>",
                      "attributes": [
                          {"id": "a1", "name": "age", "attributeOperator": ">=", "value": "18"},
                          {"id": "a2", "name": "country == LU"},
                          {"id": "a3", "name": "email", "value": "x@y"},
                      ]}},
            {"id": "p", "type": "UserModelName", "data": {"name": "p", "className": "Personality",
                                                         "attributes": []}},
        ], "edges": [{"id": "l", "type": "UserModelLink", "source": "u", "target": "p", "data": {}}]}
        v3 = convert_v4_model_to_v3(model)
        els = v3["elements"]
        assert els["u"]["attributes"] == ["a1", "a2", "a3"]
        assert els["a1"]["name"] == "age >= 18" and els["a1"]["attributeOperator"] == ">="
        assert els["a2"]["name"] == "country == LU"
        assert els["a3"]["name"] == "email = x@y"
        assert els["u-icon"]["type"] == "UserModelIcon"
        assert v3["relationships"]["l"]["type"] == "UserModelLink"
        summary = compact_model_summary(v3, "UserDiagram")
        assert "2 profile box(es)" in summary and "1 link(s)" in summary
        assert "criteria: age >= 18" in detailed_model_summary(v3, "UserDiagram")


# ---------------------------------------------------------------------------
# BPMN
# ---------------------------------------------------------------------------

class TestBPMN:
    MODEL = {"version": "4.0.0", "type": "BPMNDiagram", "nodes": [
        {"id": "pool", "type": "bpmnPool", "position": {"x": 0, "y": 0}, "width": 800, "height": 300,
         "data": {"name": "Shop"}},
        {"id": "start", "type": "bpmnStartEvent", "parentId": "pool", "position": {"x": 40, "y": 100},
         "data": {"name": "", "eventType": "default"}},
        {"id": "task", "type": "bpmnTask", "parentId": "pool", "position": {"x": 150, "y": 90},
         "data": {"name": "Ship order", "taskType": "user", "marker": "none"}},
        {"id": "gw", "type": "bpmnGateway", "parentId": "pool", "position": {"x": 350, "y": 100},
         "data": {"name": "Paid?", "gatewayType": "exclusive"}},
        {"id": "store", "type": "bpmnDataStore", "position": {"x": 150, "y": 400}, "data": {"name": "DB"}},
    ], "edges": [
        {"id": "f1", "type": "BPMNSequenceFlow", "source": "start", "target": "task",
         "data": {"label": "", "points": []}},
        {"id": "f2", "type": "BPMNSequenceFlow", "source": "task", "target": "gw",
         "data": {"label": "next", "points": []}},
        {"id": "f3", "type": "BPMNDataAssociationFlow", "source": "task", "target": "store",
         "data": {"points": []}},
    ]}

    def test_nodes_and_flows(self):
        v3 = convert_v4_model_to_v3(self.MODEL, "BPMN")
        els, rels = v3["elements"], v3["relationships"]
        assert v3["type"] == "BPMNDiagram"
        assert els["task"]["type"] == "BPMNTask" and els["task"]["taskType"] == "user"
        assert els["task"]["owner"] == "pool" and els["task"]["bounds"]["x"] == 150
        assert els["gw"]["type"] == "BPMNGateway" and els["gw"]["gatewayType"] == "exclusive"
        assert els["store"]["type"] == "BPMNDataStore"
        assert rels["f1"]["type"] == "BPMNFlow" and rels["f1"]["flowType"] == "sequence"
        assert rels["f2"]["name"] == "next"
        assert rels["f3"]["flowType"] == "dataAssociation"

    def test_summary(self):
        v3 = convert_v4_model_to_v3(self.MODEL, "BPMN")
        detail = detailed_model_summary(v3, "BPMN")
        assert "[task] Ship order (Task user)" in detail
        assert "Flow: [task] Ship order -> [gw] Paid? [next]" in detail
        assert "3 relationship(s)" in compact_model_summary(v3, "BPMN")


# ---------------------------------------------------------------------------
# NNDiagram (generic structural conversion) and non-UML pass-through
# ---------------------------------------------------------------------------

class TestOtherTypes:
    def test_nn_layers_are_converted_structurally(self):
        model = {"version": "4.0.0", "type": "NNDiagram", "nodes": [
            {"id": "c", "type": "NNContainer", "position": {"x": 0, "y": 0}, "data": {"name": "net"}},
            {"id": "l1", "type": "Conv2DLayer", "parentId": "c", "position": {"x": 10, "y": 10},
             "data": {"name": "conv", "attributes": {"kernel_dim": "[3, 3]", "input_reused": False}}},
            {"id": "r", "type": "NNReference", "data": {"name": "ref", "referenceTarget": "c"}},
        ], "edges": [{"id": "n", "type": "NNNext", "source": "l1", "target": "r", "data": {}}]}
        v3 = convert_v4_model_to_v3(model)
        assert v3["elements"]["l1"]["type"] == "Conv2DLayer"
        assert v3["elements"]["l1"]["owner"] == "c"
        assert v3["elements"]["r"]["referencedNN"] == "c"
        assert v3["relationships"]["n"]["type"] == "NNNext"

    @pytest.mark.parametrize("diagram_type,model", [
        ("GUINoCodeDiagram", {"version": "4.0.0", "pages": [{"name": "Home"}], "nodes": []}),
        ("QuantumCircuitDiagram", {"version": "4.0.0", "cols": [["H"]], "nodes": []}),
    ])
    def test_gui_and_quantum_pass_through(self, diagram_type, model):
        assert normalize_model(model, diagram_type) is model
        assert convert_v4_model_to_v3({**model, "type": diagram_type}) == {**model, "type": diagram_type}


# ---------------------------------------------------------------------------
# Project snapshot
# ---------------------------------------------------------------------------

def _mixed_snapshot():
    gui_model = {"version": "4.0.0", "pages": [{"name": "Home"}], "nodes": []}
    return {
        "id": "proj-1",
        "name": "Shop",
        "diagrams": {
            "ClassDiagram": [
                {"id": "c1", "title": "Shop", "model": copy.deepcopy(V4_CLASS)},
                {"id": "c2", "title": "Legacy", "model": copy.deepcopy(MINIMAL_CLASS_MODEL)},
            ],
            "ObjectDiagram": [{"id": "o1", "title": "Objects", "model": copy.deepcopy(V3_CLASS)}],
            "GUINoCodeDiagram": [{"id": "g1", "title": "GUI", "model": gui_model}],
            "QuantumCircuitDiagram": [{"id": "q1", "title": "Q", "model": {"cols": []}}],
            # Legacy single-dict layout with a v4 model.
            "StateMachineDiagram": {"id": "s1", "title": "SM",
                                    "model": copy.deepcopy(TestStateMachineDiagram.MODEL)},
        },
    }


class TestProjectSnapshot:
    def test_mixed_snapshot(self):
        snapshot = _mixed_snapshot()
        before = copy.deepcopy(snapshot)
        out = normalize_project_snapshot(snapshot)

        assert snapshot == before, "input snapshot must not be mutated"
        assert out is not snapshot
        assert out["id"] == "proj-1" and out["name"] == "Shop"

        cls_tabs = out["diagrams"]["ClassDiagram"]
        assert cls_tabs[0]["id"] == "c1" and cls_tabs[0]["title"] == "Shop"
        assert "elements" in cls_tabs[0]["model"] and "nodes" not in cls_tabs[0]["model"]
        # v3 tabs, the v3 object diagram, GUI and Quantum are passed through as-is.
        assert cls_tabs[1] is snapshot["diagrams"]["ClassDiagram"][1]
        assert out["diagrams"]["ObjectDiagram"] is snapshot["diagrams"]["ObjectDiagram"]
        assert out["diagrams"]["GUINoCodeDiagram"] is snapshot["diagrams"]["GUINoCodeDiagram"]
        assert out["diagrams"]["QuantumCircuitDiagram"] is snapshot["diagrams"]["QuantumCircuitDiagram"]
        sm = out["diagrams"]["StateMachineDiagram"]
        assert sm["id"] == "s1" and "elements" in sm["model"]

    def test_all_v3_snapshot_is_returned_as_is(self):
        snapshot = {"diagrams": {"ClassDiagram": [{"id": "c", "model": MINIMAL_CLASS_MODEL}]}}
        assert normalize_project_snapshot(snapshot) is snapshot

    def test_non_dict_input(self):
        assert normalize_project_snapshot(None) is None
        assert normalize_project_snapshot({"diagrams": None}) == {"diagrams": None}

    def test_v4_reference_diagram_inside_v3_object_model(self):
        obj_model = {"version": "3.0.0", "type": "ObjectDiagram", "elements": {}, "relationships": {},
                     "referenceDiagramData": copy.deepcopy(V4_CLASS)}
        out = normalize_project_snapshot({"diagrams": {"ObjectDiagram": [{"model": obj_model}]}})
        ref = out["diagrams"]["ObjectDiagram"][0]["model"]["referenceDiagramData"]
        assert ref["elements"]["customer"]["type"] == "Class"
        assert "nodes" in obj_model["referenceDiagramData"]  # input untouched


# ---------------------------------------------------------------------------
# Adapter boundary (protocol/adapters.py)
# ---------------------------------------------------------------------------

def _v4_payload(message="add an email attribute to Customer", **kw):
    return make_v2_payload(message, "ClassDiagram", project_snapshot=_mixed_snapshot(), **kw)


class TestAdapterBoundary:
    def test_current_model_is_v3(self):
        request = parse_v2_payload(_v4_payload())
        model = request.current_model
        assert isinstance(model.get("elements"), dict) and "nodes" not in model
        assert request.context.active_model is model
        tab = request.context.project_snapshot["diagrams"]["ClassDiagram"][0]
        assert tab["model"] is model
        assert request.context.diagram_summaries[0]["diagramId"] == "c1"
        # raw_payload's context carries the same normalized snapshot.
        assert request.raw_payload["context"]["projectSnapshot"] is request.context.project_snapshot

    def test_summaries_are_non_empty_for_v4_input(self):
        model = parse_v2_payload(_v4_payload()).current_model
        assert is_diagram_nontrivial(model, "ClassDiagram")
        compact = compact_model_summary(model, "ClassDiagram")
        assert "Customer" in compact and "Order" in compact
        assert "3 relationship(s)" in compact
        detail = detailed_model_summary(model, "ClassDiagram")
        assert "Customer | attributes: customerId: int" in detail
        # The summary lists only plain ``Class`` elements (same for v3 input), so
        # the abstract parent is shown by id.
        assert "Generalizations (1): Customer extends" in detail
        assert 'Customer -> Order (association) [Customer 1, Order 0..*] "places"' in detail

    def test_class_modify_path_finds_v4_classes(self):
        model = parse_v2_payload(_v4_payload()).current_model
        handler = ClassDiagramHandler(None)
        classes, attrs, methods = handler._build_model_index(model)
        assert {"person", "customer", "order", "status"} <= classes
        assert {"name", "customerid", "total"} <= attrs
        assert "greet" in methods
        assert handler._enum_names_in_model(model) == {"Status"}
        impact = handler._build_impact_context(model)
        # Person is abstract: its inheritance edge is part of the impact map too.
        assert "Customer: ClassInheritance -> Person, ClassBidirectional -> Order" in impact
        # A removal naming an existing member survives the phantom-target check.
        spec = {"action": "modify_model", "modification": {
            "action": "remove_attribute",
            "target": {"className": "Order", "attributeName": "total"}}}
        assert handler._drop_phantom_target_ops(spec, model) == []

    def test_active_model_is_normalized_too(self):
        payload = make_v2_payload("hi", "ClassDiagram", active_model=copy.deepcopy(V4_CLASS))
        request = parse_v2_payload(payload)
        active = request.raw_payload["context"]["activeModel"]
        assert "elements" in active and "nodes" not in active

    def test_v3_payload_is_unchanged(self):
        snapshot = {"diagrams": {"ClassDiagram": [{"id": "c", "model": copy.deepcopy(MINIMAL_CLASS_MODEL)}]}}
        request = parse_v2_payload(make_v2_payload("hi", "ClassDiagram", project_snapshot=snapshot))
        assert request.current_model == MINIMAL_CLASS_MODEL
        assert request.context.project_snapshot == snapshot
