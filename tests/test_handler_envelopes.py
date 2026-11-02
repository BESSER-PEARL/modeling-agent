"""Every handler's reply envelope, byte for byte, under a fake LLM provider.

The frozen outputs in ``fixtures/handler_envelopes.json`` were captured from
the handlers before the generation skeleton moved into
``BaseDiagramHandler`` (template methods) and the wire actions into
``protocol.actions``. Each case drives a public entry point
(``generate_complete_system`` / ``generate_single_element`` /
``generate_modification``) through the success path, an LLM failure, a model
refusal and a crash inside the handler, and compares ``json.dumps`` of the
envelope: same keys, same key order, same values.

Regenerate (only for a deliberate wire change):
``python tests/test_handler_envelopes.py``
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from common.errors import LLMPredictionError, ModelRefusal
from diagram_handlers.core import base_handler as base_handler_module
from diagram_handlers.core.base_handler import BaseDiagramHandler
from diagram_handlers.registry.factory import DiagramHandlerFactory

FIXTURE = Path(__file__).parent / "fixtures" / "handler_envelopes.json"

_LONG = (
    " The system must also track opening hours, staff shifts, membership tiers,"
    " overdue fines, reservations that expire after three days, branch transfers,"
    " and a notification history so members are reminded before a due date."
)

CANNED: Dict[str, Dict[str, Any]] = {
    "CompactSystemClassSpec": {
        "name": "LibrarySystem",
        "classes": [
            {"n": "Book", "a": ["title: str", "price: float"], "m": ["decreasePrice(percent: float)"], "k": ""},
            {"n": "Author", "a": ["name: str"], "m": [], "k": ""},
            {"n": "BookStatus", "a": ["AVAILABLE", "LOANED"], "m": [], "k": "enum"},
        ],
        "rels": [
            {"f": "Author", "t": "Book", "k": "assoc", "how_many_TARGET_for_one_SOURCE": "*",
             "how_many_SOURCE_for_one_TARGET": "1..*", "l": "writes"},
        ],
        "ocl": ["context Book inv positivePrice: self.price > 0"],
    },
    "SingleClassSpec": {
        "className": "Book",
        "attributes": [{"name": "title", "type": "str"}, {"name": "pages", "type": "int"}],
        "methods": [{"name": "borrow", "returnType": "bool"}],
    },
    "SystemObjectSpec": {
        "systemName": "LibraryObjects",
        "objects": [
            {"objectName": "book1", "className": "Book", "attributes": [{"name": "title", "value": "Dune"}]},
            {"objectName": "author1", "className": "Author", "attributes": [{"name": "name", "value": "Herbert"}]},
        ],
        "links": [{"source": "author1", "target": "book1"}],
    },
    "SingleObjectSpec": {
        "objectName": "book1: Book", "className": "Book",
        "attributes": [{"name": "title", "value": "Dune"}],
    },
    "ObjectModificationResponse": {
        "modifications": [
            {"action": "add_object", "target": {"objectName": "book2"},
             "changes": {"className": "Book", "attributes": [{"name": "title", "value": "Emma"}]}},
        ],
    },
    "SystemStateMachineSpec": {
        "systemName": "OrderFlow",
        "states": [
            {"stateName": "Pending", "stateType": "regular", "entryAction": "notify"},
            {"stateName": "Shipped", "stateType": "regular"},
            {"stateName": "Lost", "stateType": "regular"},
        ],
        "transitions": [
            {"source": "Pending", "target": "Shipped", "trigger": "ship"},
            {"source": "Shipped", "target": "Ghost", "trigger": "vanish"},
        ],
    },
    "SingleStateSpec": {"stateName": "Paid", "stateType": "regular", "entryAction": "charge"},
    "StateMachineModificationResponse": {
        "modifications": [
            {"action": "add_state", "target": {"stateName": "Cancelled"}, "changes": {"name": "Cancelled"}},
        ],
    },
    "SystemAgentSpec": {
        "systemName": "PizzaBot",
        "intents": [{"intentName": "Order", "trainingPhrases": ["I want a pizza"]}],
        "states": [
            {"stateName": "greeting", "replies": [{"text": "Hi!"}]},
            {"stateName": "ordering", "replies": [{"text": "Which pizza?"}]},
        ],
        "transitions": [
            {"source": "greeting", "target": "ordering", "condition": "when_intent_matched", "conditionValue": "Order"},
        ],
        "llms": [{"name": "gpt", "provider": "openai"}],
    },
    "AgentSingleElementSpec": {"type": "state", "stateName": "farewell", "replies": [{"text": "Bye!"}]},
    "SystemBPMNSpec": {
        "systemName": "OrderProcess",
        "nodes": [
            {"id": "start", "name": "Start", "type": "startEvent"},
            {"id": "t1", "name": "Check Stock", "type": "task"},
            {"id": "t2", "name": "Ship", "type": "task"},
            {"id": "end", "name": "End", "type": "endEvent"},
        ],
        "flows": [
            {"source": "start", "target": "t1"},
            {"source": "t1", "target": "end"},
        ],
    },
    "SystemUserProfileSpec": {
        "systemName": "Learner",
        "profiles": [
            {"className": "Personal_Information", "attributes": [{"name": "age", "operator": ">=", "value": "16"}]},
        ],
    },
    "SystemQuantumCircuitSpec": {
        "qubitCount": 2, "algorithmName": "Bell",
        "operations": [
            {"gate": "H", "row": 0, "column": 0},
            {"gate": "CNOT", "column": 1, "controlRow": 0, "targetRow": 1},
        ],
    },
    "SingleQuantumGateSpec": {"operation": {"gate": "CNOT", "column": 0, "controlRow": 0, "targetRow": 1}},
    "QuantumModificationSpec": {"mode": "append", "operations": [{"gate": "X", "row": 1, "column": 2}]},
    "AuthoredSystemGUISpec": {
        "projectName": "Shop", "domain": "startup",
        "pages": [
            {"name": "Home", "sections": [
                {"html": "<section class=\"ds-hero\"><h1>Shop</h1></section>"},
                {"bind": {"kind": "bar_chart", "title": "Sales",
                          "sampleData": [{"name": "Mon", "value": 3}, {"name": "Tue", "value": 5}]}},
            ]},
            {"name": "About", "sections": [{"html": "<section><p>About us</p></section>"}, {}]},
        ],
    },
    "SingleGUIElementSpec": {"pageName": "Home", "section": {"type": "content", "title": "News", "body": "Latest"}},
    "SystemComponentSpec": {
        "systemName": "Shop",
        "subsystems": [{"id": "be", "name": "Backend", "stereotype": "subsystem"}],
        "components": [
            {"id": "api", "name": "API", "owner": "be", "stereotype": "solution"},
            {"id": "db", "name": "DB", "owner": "be", "stereotype": "db"},
        ],
        "dependencies": [{"source": "api", "target": "db", "stereotype": "uses"}],
    },
    "ComponentModificationResponse": {
        "message": "ok",
        "modifications": [
            {"action": "add_component", "target": {"elementName": "Cache"}, "changes": {"name": "Cache", "owner": "Backend"}},
            {"action": "modify_element", "target": {"elementName": "Ghost"}, "changes": {"name": "Spirit"}},
        ],
    },
    "ClassModificationResponse": {
        "modifications": [
            {"action": "add_attribute", "target": {"className": "Book"},
             "changes": {"name": "isbn", "type": "str"}},
            {"action": "modify_class", "target": {"className": "Author"}, "changes": {"name": "Writer"}},
        ],
    },
    "AgentModificationResponse": {
        "modifications": [
            {"action": "add_state", "target": {"stateName": "farewell"},
             "changes": {"replies": [{"text": "Bye!"}]}},
            {"action": "add_transition", "target": {"sourceStateName": "greeting", "targetStateName": "farewell"},
             "changes": {"condition": "auto"}},
        ],
    },
    "UserProfileModificationResponse": {
        "modifications": [
            {"action": "modify_attribute_value", "target": {"profileName": "personal_information", "attributeName": "age"},
             "changes": {"value": "18", "operator": ">="}},
        ],
    },
    "BPMNModificationResponse": {
        "message": "ok",
        "modifications": [
            {"action": "add_task", "target": {"nodeName": "Pack"}, "changes": {"name": "Pack"}},
            {"action": "add_flow", "target": {}, "changes": {"source": "Check Stock", "target": "Pack"}},
        ],
    },
    "DeploymentModificationResponse": {
        "message": "ok",
        "modifications": [
            {"action": "add_node", "target": {"elementName": "Cache Host"}, "changes": {"name": "Cache Host"}},
        ],
    },
    "GUIModificationBatchSpec": {
        "operations": [{"operation": "add_page", "pageName": "Contact",
                        "section": {"type": "content", "title": "Contact us", "body": "Write to us"}}],
    },
    "SystemDeploymentSpec": {
        "systemName": "Prod",
        "nodes": [{"id": "web", "name": "Web Server", "stereotype": "server"}],
        "artifacts": [{"id": "app", "name": "app.war", "owner": "web"}],
        "deployComponents": [{"id": "c1", "name": "App", "stereotype": "solution", "manifestedBy": "app"}],
    },
}

_CLASS_MODEL = {
    "elements": {
        "c1": {"id": "c1", "type": "Class", "name": "Book", "attributes": ["a1"], "methods": []},
        "a1": {"id": "a1", "type": "ClassAttribute", "name": "+ title: str", "owner": "c1"},
        "c2": {"id": "c2", "type": "Class", "name": "Author", "attributes": [], "methods": []},
    },
    "relationships": {},
}
_STATE_MODEL = {
    "elements": {"s1": {"id": "s1", "type": "State", "name": "Pending"}},
    "relationships": {},
}
_AGENT_MODEL = {
    "elements": {
        "s1": {"id": "s1", "type": "AgentState", "name": "greeting"},
        "i1": {"id": "i1", "type": "AgentIntent", "name": "Hello"},
    },
    "relationships": {},
}
_BPMN_MODEL = {
    "elements": {
        "n1": {"id": "n1", "type": "BPMNStartEvent", "name": "Start"},
        "n2": {"id": "n2", "type": "BPMNTask", "name": "Check Stock"},
    },
    "relationships": {},
}
_DEPLOYMENT_MODEL = {
    "elements": {"web": {"id": "web", "type": "DeploymentNode", "name": "Web Server"}},
    "relationships": {},
}
_COMPONENT_MODEL = {
    "elements": {
        "be": {"id": "be", "type": "Subsystem", "name": "Backend"},
        "api": {"id": "api", "type": "Component", "name": "API", "owner": "be"},
    },
    "relationships": {},
}


class _Exploding:
    """A parse result whose dump crashes: drives a handler's generic-error path."""

    def model_dump_json(self, *args, **kwargs):
        return "{}"

    def model_dump(self, *args, **kwargs):
        raise RuntimeError("boom")


class FakeProvider:
    """Stands in for ``llm.provider.LLMProvider``: canned structured outputs."""

    model_name = "fake-model"

    def __init__(self, mode: str):
        self.mode = mode

    def supports_structured_outputs(self) -> bool:
        return True

    def structured_model(self, model=None) -> str:
        return model or self.model_name

    def parse(self, messages, schema, **kwargs):
        if self.mode == "llm_error":
            raise LLMPredictionError("fake failure")
        if self.mode == "refusal":
            raise ModelRefusal("openai", "violence")
        if self.mode == "crash":
            return _Exploding()
        return schema.model_validate(CANNED[schema.__name__])

    def complete(self, prompt, **kwargs) -> str:
        if "extract ONLY the class/entity names" in prompt:
            return '["Book", "Author"]'
        if "User Request: Create a" in prompt:
            return '{"className": "Book", "attributes": [{"name": "title", "type": "str"}], "methods": []}'
        return "Design analysis: two classes, one association."


# (case id, diagram type, entry point, user request, kwargs, modes)
_ALL = ("ok", "llm_error", "refusal", "crash")
_SPECS: List[tuple] = [
    ("class.system", "ClassDiagram", "generate_complete_system", "Create a library with books and authors", {}, _ALL),
    ("class.system_long", "ClassDiagram", "generate_complete_system",
     "Create a library with books and authors." + _LONG, {"raw_request": "Create a library." + _LONG}, ("ok",)),
    ("class.single", "ClassDiagram", "generate_single_element", "Create a Book class", {}, _ALL),
    ("object.system", "ObjectDiagram", "generate_complete_system", "Create two library objects", {}, _ALL),
    ("object.system_ref", "ObjectDiagram", "generate_complete_system", "Create two library objects",
     {"reference_diagram": _CLASS_MODEL}, ("ok",)),
    ("object.single", "ObjectDiagram", "generate_single_element", "Create a book object",
     {"reference_diagram": _CLASS_MODEL}, _ALL),
    ("object.modify", "ObjectDiagram", "generate_modification", "add another book", {}, _ALL),
    ("state.system", "StateMachineDiagram", "generate_complete_system", "Create an order state machine", {}, _ALL),
    ("state.system_long", "StateMachineDiagram", "generate_complete_system",
     "Create an order state machine." + _LONG, {}, ("ok",)),
    ("state.single", "StateMachineDiagram", "generate_single_element", "Add a Paid state", {}, _ALL),
    ("state.modify", "StateMachineDiagram", "generate_modification", "add a Cancelled state",
     {"current_model": _STATE_MODEL}, _ALL),
    ("agent.system", "AgentDiagram", "generate_complete_system", "Create a pizza ordering bot", {}, _ALL),
    ("agent.system_multi", "AgentDiagram", "generate_complete_system", "Create a multi-agent system", {}, ("ok",)),
    ("agent.single", "AgentDiagram", "generate_single_element", "Add a farewell state", {}, _ALL),
    ("bpmn.system", "BPMN", "generate_complete_system", "Model an order process", {}, _ALL),
    ("bpmn.system_agentic", "BPMN", "generate_complete_system", "Model an agentic order process with AI agent lanes",
     {}, ("ok", "llm_error", "crash")),
    ("bpmn.single", "BPMN", "generate_single_element", "Add a Review task", {}, ("ok",)),
    ("user.system", "UserDiagram", "generate_complete_system", "A teenager learner profile", {}, _ALL),
    ("quantum.system", "QuantumCircuitDiagram", "generate_complete_system", "Create a Bell state", {}, _ALL),
    ("quantum.single", "QuantumCircuitDiagram", "generate_single_element", "Add a CNOT", {}, _ALL),
    ("quantum.modify", "QuantumCircuitDiagram", "generate_modification", "add an X on qubit 1", {}, _ALL),
    ("gui.system", "GUINoCodeDiagram", "generate_complete_system", "Build a shop dashboard", {}, _ALL),
    ("gui.single", "GUINoCodeDiagram", "generate_single_element", "Add a news section", {}, _ALL),
    ("component.system", "ComponentDiagram", "generate_complete_system", "A shop architecture", {}, _ALL),
    ("component.single", "ComponentDiagram", "generate_single_element", "Add a Billing component", {}, ("ok",)),
    ("component.modify", "ComponentDiagram", "generate_modification", "add a cache, rename Ghost",
     {"current_model": _COMPONENT_MODEL}, _ALL),
    ("class.modify", "ClassDiagram", "generate_modification", "add isbn to Book, rename Author",
     {"current_model": _CLASS_MODEL}, _ALL),
    ("class.modify_empty", "ClassDiagram", "generate_modification", "add isbn to Book", {}, ("ok",)),
    ("agent.modify", "AgentDiagram", "generate_modification", "add a farewell state after greeting",
     {"current_model": _AGENT_MODEL}, _ALL),
    ("bpmn.modify", "BPMN", "generate_modification", "add a Pack task after Check Stock",
     {"current_model": _BPMN_MODEL}, _ALL),
    ("user.modify", "UserDiagram", "generate_modification", "make the age at least 18", {}, _ALL),
    ("deployment.modify", "DeploymentDiagram", "generate_modification", "add a cache host",
     {"current_model": _DEPLOYMENT_MODEL}, _ALL),
    ("gui.modify", "GUINoCodeDiagram", "generate_modification", "add a contact page", {}, _ALL),
    ("deployment.system", "DeploymentDiagram", "generate_complete_system", "A production deployment", {}, _ALL),
]
CASES = [(f"{cid}.{mode}", dtype, entry, req, kw, mode) for cid, dtype, entry, req, kw, modes in _SPECS for mode in modes]


def run_case(dtype: str, entry: str, request: str, kwargs: Dict[str, Any], mode: str) -> str:
    """Call one handler entry point under the fake provider; return the envelope JSON."""
    handler = DiagramHandlerFactory(llm=None).get_handler(dtype)
    provider = FakeProvider(mode)
    patches = pytest.MonkeyPatch()
    try:
        patches.setattr(BaseDiagramHandler, "llm_provider", property(lambda self: provider))
        patches.setattr(base_handler_module.time, "sleep", lambda *_: None)
        envelope = getattr(handler, entry)(request, **kwargs)
    except Exception as exc:  # an entry point may re-raise provider errors
        envelope = {"raised": type(exc).__name__, "text": str(exc)}
    finally:
        patches.undo()
    return json.dumps(envelope)


def _frozen() -> Dict[str, str]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_id,dtype,entry,request_text,kwargs,mode", CASES, ids=[c[0] for c in CASES])
def test_envelope_is_byte_identical(case_id, dtype, entry, request_text, kwargs, mode):
    assert run_case(dtype, entry, request_text, kwargs, mode) == _frozen()[case_id]


def test_every_case_is_frozen():
    assert sorted(_frozen()) == sorted(c[0] for c in CASES)


def _capture(path: Optional[Path] = None) -> None:
    out = {c[0]: run_case(*c[1:]) for c in CASES}
    (path or FIXTURE).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    _capture()
