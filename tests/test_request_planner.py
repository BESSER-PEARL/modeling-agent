"""Tests for the request planner (orchestrator/request_planner.py)."""

import copy
import json
from pathlib import Path

import pytest
from orchestrator.request_planner import (
    GENERATOR_PREREQUISITES,
    plan_assistant_operations,
    _try_heuristic_decomposition,
    _split_message_segments,
    _match_segment_target,
    _should_use_llm_planner,
    _normalize_operations,
    _fallback_operations,
    _validate_and_fix_plan,
    _get_workspace_diagram_types,
)
from protocol.types import AssistantRequest, WorkspaceContext


def _make_request(message: str, active_diagram_type: str = "ClassDiagram") -> AssistantRequest:
    return AssistantRequest(
        message=message,
        diagram_type=active_diagram_type,
        context=WorkspaceContext(active_diagram_type=active_diagram_type),
    )


# What the editor sends as ``projectSnapshot`` for a new project
# (``createDefaultProject``): every diagram type has one tab whose model is a
# non-empty dict with no content.
_EDITOR_DEFAULT_PROJECT = json.loads(
    (Path(__file__).parent / "fixtures" / "editor_default_project.json").read_text(encoding="utf-8")
)

_BOOK_CLASS = {"id": "c1", "name": "Book", "type": "Class", "owner": None}


def _editor_request(
    message: str,
    active_diagram_type: str = "ClassDiagram",
    *,
    classes: bool = False,
    screens: bool = False,
) -> AssistantRequest:
    """A request as the editor sends it: the whole project snapshot, optionally
    with a class and a designed screen in it."""
    snapshot = copy.deepcopy(_EDITOR_DEFAULT_PROJECT)
    if classes:
        snapshot["diagrams"]["ClassDiagram"][0]["model"]["elements"] = {"c1": dict(_BOOK_CLASS)}
    if screens:
        wrapper = snapshot["diagrams"]["GUINoCodeDiagram"][0]["model"]["pages"][0]["frames"][0]["component"]
        wrapper["components"] = [{"type": "text", "content": "Books"}]
    return AssistantRequest(
        message=message,
        diagram_type=active_diagram_type,
        context=WorkspaceContext(
            active_diagram_type=active_diagram_type,
            project_snapshot=snapshot,
            current_diagram_indices=snapshot["currentDiagramIndices"],
        ),
    )


def _noop_predict(prompt: str) -> str:
    """Fake LLM predict that returns an empty plan."""
    return json.dumps({"operations": []})


def test_api_and_semantic_generators_share_class_diagram_prerequisite():
    assert GENERATOR_PREREQUISITES["rest_api"] == ["ClassDiagram"]
    assert GENERATOR_PREREQUISITES["rdf"] == ["ClassDiagram"]


# ---------------------------------------------------------------------------
# _split_message_segments
# ---------------------------------------------------------------------------

class TestSplitMessageSegments:
    def test_simple(self):
        segments = _split_message_segments("create a class diagram")
        assert segments == ["create a class diagram"]

    def test_with_connectors(self):
        segments = _split_message_segments("create a class diagram and then generate django code")
        assert len(segments) == 2

    def test_with_semicolons(self):
        segments = _split_message_segments("create a class; generate sql")
        assert len(segments) == 2

    def test_empty(self):
        assert _split_message_segments("") == []

    def test_non_string(self):
        assert _split_message_segments(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _match_segment_target
# ---------------------------------------------------------------------------

class TestMatchSegmentTarget:
    def test_class_diagram(self):
        assert _match_segment_target("create a class diagram for hotel") == "ClassDiagram"

    def test_state_machine(self):
        assert _match_segment_target("model the state machine for orders") == "StateMachineDiagram"

    def test_no_match(self):
        assert _match_segment_target("say hello") is None

    def test_keyword_inside_another_word_is_not_a_match(self):
        assert _match_segment_target("store the personal details of each customer") is None
        assert _match_segment_target("add a guide page for new members") is None

    def test_plural_keyword_matches(self):
        assert _match_segment_target("describe the personas of the shop") == "UserDiagram"


# ---------------------------------------------------------------------------
# _should_use_llm_planner
# ---------------------------------------------------------------------------

class TestShouldUseLlmPlanner:
    def test_simple_request_no_planner(self):
        assert not _should_use_llm_planner("create a User class", 1, False)

    def test_multi_target_with_connector(self):
        assert _should_use_llm_planner(
            "create a class diagram and a state machine diagram", 2, False
        )

    def test_generation_with_connector(self):
        assert _should_use_llm_planner(
            "create a class diagram and then generate django code", 1, True
        )

    def test_keyword_inside_another_word_is_not_an_explicit_diagram(self):
        # No diagram is named, so the long multi-clause request goes to the planner.
        msg = (
            "I need to keep the personal details of every customer, their orders, "
            "the invoices and the payments, with a history of each change"
        )
        assert len(msg) > 120
        assert _should_use_llm_planner(msg, 1, False)

    def test_long_complex_no_diagram_keywords(self):
        msg = "I want to build something complex with multiple entities and relationships and attributes " * 2
        assert _should_use_llm_planner(msg, 1, False)


# ---------------------------------------------------------------------------
# _normalize_operations
# ---------------------------------------------------------------------------

class TestNormalizeOperations:
    def test_valid_model_operation(self):
        ops = [{"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system", "request": "create a hotel"}]
        request = _make_request("create a hotel")
        result = _normalize_operations(ops, request, "complete_system")
        assert len(result) == 1
        assert result[0]["type"] == "model"

    def test_invalid_diagram_type_skipped(self):
        ops = [{"type": "model", "diagramType": "FooDiagram", "mode": "complete_system", "request": "create"}]
        request = _make_request("")
        result = _normalize_operations(ops, request, "complete_system")
        assert len(result) == 0

    def test_deduplication(self):
        ops = [
            {"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system", "request": "create a hotel"},
            {"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system", "request": "create a hotel"},
        ]
        request = _make_request("create a hotel")
        result = _normalize_operations(ops, request, "complete_system")
        assert len(result) == 1

    def test_generation_operation(self):
        ops = [{"type": "generation", "generatorType": "django", "config": {}}]
        request = _make_request("generate django code")
        result = _normalize_operations(ops, request, "complete_system")
        assert len(result) == 1
        assert result[0]["generatorType"] == "django"

    def test_invalid_generator_type_inferred(self):
        ops = [{"type": "generation", "generatorType": "invalid_gen", "request": "generate django"}]
        request = _make_request("generate django code")
        result = _normalize_operations(ops, request, "complete_system")
        # Should infer django from the request message
        assert len(result) == 1
        assert result[0]["generatorType"] == "django"

    def test_non_list_returns_empty(self):
        request = _make_request("")
        assert _normalize_operations("not a list", request, "complete_system") == []


# ---------------------------------------------------------------------------
# _fallback_operations
# ---------------------------------------------------------------------------

class TestFallbackOperations:
    def test_single_target(self):
        request = _make_request("create a class diagram for hotel")
        ops = _fallback_operations(request, "complete_system", None)
        assert len(ops) >= 1
        assert ops[0]["type"] == "model"
        assert ops[0]["diagramType"] == "ClassDiagram"

    def test_with_generation(self):
        request = _make_request("create a class diagram and generate django code")
        ops = _fallback_operations(request, "complete_system", None)
        gen_ops = [o for o in ops if o["type"] == "generation"]
        assert len(gen_ops) == 1
        assert gen_ops[0]["generatorType"] == "django"


# ---------------------------------------------------------------------------
# plan_assistant_operations (integration with noop LLM)
# ---------------------------------------------------------------------------

class TestPlanAssistantOperations:
    def test_simple_request(self):
        request = _make_request("create a User class")
        ops = plan_assistant_operations(request, "single_element", "create_single_element_intent", _noop_predict)
        assert len(ops) >= 1
        assert ops[0]["type"] == "model"

    def test_empty_message(self):
        request = _make_request("")
        ops = plan_assistant_operations(request, "complete_system", None, _noop_predict)
        # Even with empty message, should produce at least a fallback
        assert len(ops) >= 1


# ---------------------------------------------------------------------------
# "create a web app ..." → class diagram, screens, paused web_app generation
# ---------------------------------------------------------------------------

_WEB_APP_PLAN = [
    ("model", "ClassDiagram"),
    ("model", "GUINoCodeDiagram"),
    ("generation", "web_app"),
]


def _plan_shape(ops):
    return [(o["type"], o.get("diagramType") or o.get("generatorType")) for o in ops or []]


class TestWebAppPlan:
    @pytest.mark.parametrize("message", [
        "create a web app for a bookshop",
        "Build a web application for tracking orders and customers",
        "Build me a web app to manage orders and customers",
        "Create me a web application that lets customers place orders",
        "Make us a web app for a bookshop",
        "I want a web app to manage tasks and their deadlines",
        "I need a web application for a bookshop",
        "I'd like a web app where authors publish books",
        "I would like a web app for a bookshop",
        "I want to build a web app for a bookshop",
        "I need you to create a web app for a bookshop",
        "Can you build a web app for a bookshop?",
        "Could you please make me a webapp to manage tasks",
        "Please create a web app for a bookshop.",
    ])
    def test_common_openings_get_the_web_app_plan(self, message):
        ops = _try_heuristic_decomposition(message, _make_request(message), "complete_system")
        assert _plan_shape(ops) == _WEB_APP_PLAN

    @pytest.mark.parametrize("message, subject", [
        ("create a web app for a bookshop", "a bookshop"),
        ("Can you build a web app for a bookshop?", "a bookshop"),
        ("Build me a web app to manage orders", "a web app to manage orders"),
        ("I need a web app where authors publish books", "a web app where authors publish books"),
        ("Create me a web application that lets customers place orders",
         "a web app that lets customers place orders"),
    ])
    def test_step_requests_read_as_sentences(self, message, subject):
        ops = _try_heuristic_decomposition(message, _make_request(message), "complete_system")
        assert ops[0]["request"] == f"create a class diagram for {subject}"
        assert ops[1]["request"] == f"create a GUI for {subject}"

    @pytest.mark.parametrize("message", [
        "build a web app",
        "what is a web app for a bookshop?",
        "we built a web app for a bookshop last year",
        "I want to rename the web app class to Portal",
        "can you explain what a web app is",
        "add a page to the web app for orders",
        "I need the web app to show a list of orders",
        "a web app for a bookshop",
    ])
    def test_other_messages_do_not_get_the_web_app_plan(self, message):
        ops = _try_heuristic_decomposition(message, _make_request(message), "complete_system")
        assert _plan_shape(ops) != _WEB_APP_PLAN

    @pytest.mark.parametrize("message", [
        # The words after "web app" say where it comes from, not what it is for.
        "I need a web app generated from my class diagram",
        "could you create a web app from this model",
        "create a web app using my class diagram",
        "build a web app based on this model",
        "I want a web app like the one we discussed",
        "I'd like a web application instead of a mobile app",
        # Nothing but filler.
        "Can you build a web app for me?",
        "make us a web app for us",
        "I want a web app for it",
        "build me a web app please",
        "Can you build a web app for me, please?",
        # "web application" is the start of a longer name.
        "I need a web application firewall rule model",
        "I want a web app store listing model",
    ])
    def test_web_app_without_a_domain_does_not_get_the_plan(self, message):
        ops = _try_heuristic_decomposition(message, _make_request(message), "complete_system")
        assert _plan_shape(ops) != _WEB_APP_PLAN

    def test_trailing_please_is_not_part_of_the_domain(self):
        message = "Create a web app for a bookshop, please."
        ops = _try_heuristic_decomposition(message, _make_request(message), "complete_system")
        assert ops[0]["request"] == "create a class diagram for a bookshop"


# ---------------------------------------------------------------------------
# GUI builds need a class diagram first
# ---------------------------------------------------------------------------

class TestGuiNeedsClassDiagram:
    """Live report: "Build a complete library management platform with models
    and UI" on an empty workspace. The classifier targeted GUINoCodeDiagram,
    the plan was a lone GUI op, and the user got 9 invented screens with no
    class diagram on the canvas (and no Basic/AI screen choice, which is only
    asked when a class diagram exists)."""

    MSG = "Build a complete library management platform with models and UI"

    @staticmethod
    def _shape(ops):
        return [(o["type"], o.get("diagramType") or o.get("generatorType")) for o in ops]

    @pytest.mark.parametrize("make_request", [_make_request, _editor_request])
    @pytest.mark.parametrize("target", ["GUINoCodeDiagram", "ClassDiagram", None])
    def test_ui_request_on_empty_workspace_takes_the_web_app_route(self, target, make_request):
        # The classifier's target flips between runs; live, ClassDiagram gave
        # a model with no screens and GUINoCodeDiagram gave screens with no model.
        request = make_request(self.MSG)
        ops = plan_assistant_operations(
            request, "complete_system", "create_complete_system_intent",
            _noop_predict, llm_target_type=target,
        )
        assert self._shape(ops) == [
            ("model", "ClassDiagram"),
            ("model", "GUINoCodeDiagram"),
            ("generation", "web_app"),
        ]
        assert ops[0]["mode"] == "complete_system"
        assert self.MSG in ops[0]["request"]

    @pytest.mark.parametrize("make_request", [_make_request, _editor_request])
    def test_lone_gui_op_on_empty_workspace_gets_class_diagram_first(self, make_request):
        # Prerequisite pass, independent of the web-app route.
        request = make_request(self.MSG)
        ops = _validate_and_fix_plan(
            [{"type": "model", "diagramType": "GUINoCodeDiagram",
              "mode": "complete_system", "request": self.MSG}],
            request,
        )
        assert self._shape(ops) == [
            ("model", "ClassDiagram"),
            ("model", "GUINoCodeDiagram"),
        ]

    @pytest.mark.parametrize("make_request", [_make_request, _editor_request])
    def test_create_without_ui_words_stays_a_class_diagram(self, make_request):
        request = make_request("Build a complete library management platform")
        ops = plan_assistant_operations(
            request, "complete_system", "create_complete_system_intent",
            _noop_predict, llm_target_type="ClassDiagram",
        )
        assert self._shape(ops) == [("model", "ClassDiagram")]

    def test_existing_class_diagram_is_not_rebuilt(self):
        request = _make_request("create a GUI for my model")
        request.context.project_snapshot = {
            "diagrams": {"ClassDiagram": [{"model": {"elements": {"c1": dict(_BOOK_CLASS)}}}]},
        }
        ops = plan_assistant_operations(
            request, "complete_system", "create_complete_system_intent",
            _noop_predict, llm_target_type="GUINoCodeDiagram",
        )
        assert self._shape(ops) == [("model", "GUINoCodeDiagram")]

    def test_existing_class_diagram_in_the_editor_is_not_rebuilt(self):
        request = _editor_request("create a GUI for my model", classes=True)
        ops = plan_assistant_operations(
            request, "complete_system", "create_complete_system_intent",
            _noop_predict, llm_target_type="GUINoCodeDiagram",
        )
        assert self._shape(ops) == [("model", "GUINoCodeDiagram")]

    @pytest.mark.parametrize("classes, screens", [(True, False), (False, True), (True, True)])
    def test_ui_request_on_a_started_project_is_not_the_from_scratch_route(self, classes, screens):
        request = _editor_request(self.MSG, classes=classes, screens=screens)
        ops = plan_assistant_operations(
            request, "complete_system", "create_complete_system_intent",
            _noop_predict, llm_target_type="ClassDiagram",
        )
        assert self._shape(ops) == [("model", "ClassDiagram")]

    @pytest.mark.parametrize("make_request", [_make_request, _editor_request])
    def test_gui_modify_does_not_add_class_diagram(self, make_request):
        request = make_request("add a footer to the home page", "GUINoCodeDiagram")
        ops = plan_assistant_operations(
            request, "modify_model", "modify_model_intent",
            _noop_predict, llm_target_type="GUINoCodeDiagram",
        )
        assert ("model", "ClassDiagram") not in self._shape(ops)


class TestBareGenerateCommandAddsNoModelStep:
    """"generate django code" names nothing to model: the planner adds no
    diagram step for it, and the generation handler says what is missing."""

    @pytest.mark.parametrize("message, intent, generator", [
        ("generate django code", "create_complete_system_intent", "django"),
        ("generate sql", "modify_model_intent", "sql"),
        ("generate the web app", "create_complete_system_intent", "web_app"),
        ("Generate the web app.", "create_complete_system_intent", "web_app"),
        ("generate qiskit code", "create_complete_system_intent", "qiskit"),
        ("generate the django code from my model", "create_complete_system_intent", "django"),
    ])
    @pytest.mark.parametrize("classes", [False, True])
    def test_only_the_generation_is_planned(self, message, intent, generator, classes):
        ops = plan_assistant_operations(
            _editor_request(message, classes=classes), "complete_system", intent, _noop_predict,
        )
        assert _plan_shape(ops) == [("generation", generator)]

    def test_generate_request_with_a_domain_still_gets_its_diagrams(self):
        ops = plan_assistant_operations(
            _editor_request("generate a web app for a bookshop"), "complete_system",
            "create_complete_system_intent", _noop_predict,
        )
        assert _plan_shape(ops) == _WEB_APP_PLAN
        assert "bookshop" in ops[0]["request"]

    def test_a_planned_model_step_still_gets_its_prerequisite(self):
        # The plan has something to model, so the missing class diagram is added.
        request = _editor_request("generate the web app")
        ops = _validate_and_fix_plan(
            [{"type": "model", "diagramType": "GUINoCodeDiagram", "mode": "complete_system",
              "request": "create a GUI for a bookshop"},
             {"type": "generation", "generatorType": "web_app", "config": {}}],
            request,
        )
        assert _plan_shape(ops) == _WEB_APP_PLAN

    def test_generation_with_an_existing_small_circuit_plans_no_rebuild(self):
        request = _editor_request("generate qiskit code for my circuit")
        circuit = request.context.project_snapshot["diagrams"]["QuantumCircuitDiagram"][0]["model"]
        circuit["cols"] = [["H", 1], [1, "X"], ["Z", 1]]
        ops = _validate_and_fix_plan(
            [{"type": "generation", "generatorType": "qiskit", "config": {}}], request,
        )
        assert _plan_shape(ops) == [("generation", "qiskit")]


class TestFromScratchRouteIsForAppRequests:
    """On a new editor project, the class diagram + screens + generate plan is
    for a request that asks for an app with an interface. A request for another
    diagram, or one that only uses a screen word, keeps its own plan."""

    @staticmethod
    def _plan(message, target, request=None):
        return plan_assistant_operations(
            request or _editor_request(message), "complete_system",
            "create_complete_system_intent", _noop_predict, llm_target_type=target,
        )

    @pytest.mark.parametrize("message, target", [
        ("Model the invoice lifecycle as a state machine, from the payment page to archiving",
         "StateMachineDiagram"),
        ("I need a state machine that follows a parcel from the tracking screen to delivery",
         "StateMachineDiagram"),
        ("an agent that answers questions about the pages of the staff manual", "AgentDiagram"),
        ("Draw the BPMN of how a refund moves from the returns page to the bank", "BPMN"),
        ("draw the deployment of the frontend, the API and the database", "DeploymentDiagram"),
        ("a user profile for people who prefer high contrast in the UI", "UserDiagram"),
        ("a quantum circuit that prepares a bell state and fits on the start page",
         "QuantumCircuitDiagram"),
    ])
    def test_request_for_another_diagram_builds_that_diagram(self, message, target):
        ops = self._plan(message, target)
        assert _plan_shape(ops) == [("model", target)]

    @pytest.mark.parametrize("message, expected", [
        ("I need a state machine that follows a parcel from the tracking screen to delivery",
         "StateMachineDiagram"),
        ("Draw the BPMN of how a refund moves from the returns page to the bank", "BPMN"),
        ("a user profile for people who prefer high contrast in the UI", "UserDiagram"),
    ])
    @pytest.mark.parametrize("target", [None, "ClassDiagram"])
    def test_a_named_diagram_is_not_replaced_by_the_web_app_plan(self, message, expected, target):
        # No usable classifier target: the diagram the message names still wins.
        shape = _plan_shape(self._plan(message, target))
        assert ("model", expected) in shape
        assert ("generation", "web_app") not in shape
        assert ("model", "GUINoCodeDiagram") not in shape

    @pytest.mark.parametrize("extra", ["AgentDiagram", "StateMachineDiagram"])
    def test_project_with_another_diagram_and_no_classes(self, extra):
        request = _editor_request("Build a second agent that explains the UI of the product")
        elements = {"AgentDiagram": {"a1": {"id": "a1", "name": "greet", "type": "AgentState"}},
                    "StateMachineDiagram": {"s1": {"id": "s1", "name": "Open", "type": "State"}}}
        request.context.project_snapshot["diagrams"][extra][0]["model"]["elements"] = elements[extra]
        assert _plan_shape(self._plan(request.message, "AgentDiagram", request)) == [("model", "AgentDiagram")]

    @pytest.mark.parametrize("message", [
        "a library catalogue: each book has a title, an author and a number of pages",
        "Design a class diagram that covers books, their pages and their authors",
        "Design a CMS data model: Site, Page, Block, Author. A page has many blocks.",
        "a cinema booking model: each cinema has several screens and each screen has seats",
        "a shop that sells screens and keyboards, with stock per warehouse",
        "a notebook model where every page belongs to one section",
    ])
    def test_screen_word_as_a_domain_noun_stays_a_class_diagram(self, message):
        assert _plan_shape(self._plan(message, "ClassDiagram")) == [("model", "ClassDiagram")]

    @pytest.mark.parametrize("message", [
        "Build an order tracker with models and a UI",
        "Build an inventory system with a frontend",
        "a booking system with a GUI",
        "Build a task tracker with screens for adding and listing tasks",
        "Build a task tracker and some screens",
        "an order tracker with a list page and a detail page",
        "a ticketing system with pages to list and edit tickets",
        "a course catalogue with a page for each course",
        "Build a bookshop with a landing page and a catalogue",
    ])
    @pytest.mark.parametrize("target", ["ClassDiagram", "GUINoCodeDiagram", None])
    def test_app_request_that_asks_for_an_interface_gets_the_web_app_plan(self, message, target):
        assert _plan_shape(self._plan(message, target)) == _WEB_APP_PLAN


# ---------------------------------------------------------------------------
# Which diagram types the workspace already has
# ---------------------------------------------------------------------------

class TestWorkspaceDiagramTypes:
    def test_a_new_editor_project_has_no_diagrams(self):
        assert _get_workspace_diagram_types(_editor_request("hello")) == set()

    def test_tabs_with_content_are_present(self):
        request = _editor_request("hello", classes=True, screens=True)
        assert _get_workspace_diagram_types(request) == {"ClassDiagram", "GUINoCodeDiagram"}

    def test_a_gui_page_without_components_is_not_screens(self):
        request = _editor_request("hello", classes=True)
        assert _get_workspace_diagram_types(request) == {"ClassDiagram"}

    @pytest.mark.parametrize("diagram_type, content", [
        ("ObjectDiagram", {"elements": {"o1": {"id": "o1", "name": "book1", "type": "Object"}}}),
        ("StateMachineDiagram", {"elements": {"s1": {"id": "s1", "name": "Open", "type": "State"}}}),
        ("AgentDiagram", {"elements": {"a1": {"id": "a1", "name": "greet", "type": "AgentState"}}}),
        ("BPMN", {"elements": {"t1": {"id": "t1", "name": "Ship order", "type": "BPMNTask"}}}),
        ("ComponentDiagram", {"elements": {"k1": {"id": "k1", "name": "Billing", "type": "Component"}}}),
        ("KnowledgeGraphDiagram", {"nodes": [{"id": "n1", "label": "Book"}]}),
    ])
    def test_each_type_is_present_once_it_has_content(self, diagram_type, content):
        request = _editor_request("hello")
        request.context.project_snapshot["diagrams"][diagram_type][0]["model"].update(content)
        assert _get_workspace_diagram_types(request) == {diagram_type}

    def test_any_tab_with_content_counts(self):
        request = _editor_request("hello")
        tabs = request.context.project_snapshot["diagrams"]["ClassDiagram"]
        tabs.append({"id": "second", "title": "Second", "model": {"elements": {"c1": dict(_BOOK_CLASS)}}})
        assert _get_workspace_diagram_types(request) == {"ClassDiagram"}

    @pytest.mark.parametrize("diagram_type, content", [
        # Small real diagrams: nothing in them is a "typical" element.
        ("ClassDiagram", {"elements": {
            "c1": {"id": "c1", "name": "Shape", "type": "AbstractClass"},
            "e1": {"id": "e1", "name": "Colour", "type": "Enumeration"},
        }}),
        ("QuantumCircuitDiagram", {"cols": [["H", 1], [1, "X"], ["Z", 1]]}),
        ("QuantumCircuitDiagram", {"cols": [[1, "H"]]}),
        ("StateMachineDiagram", {"elements": {"i1": {"id": "i1", "name": "", "type": "StateInitialNode"}}}),
    ])
    def test_a_small_diagram_is_present(self, diagram_type, content):
        request = _editor_request("hello")
        request.context.project_snapshot["diagrams"][diagram_type][0]["model"].update(content)
        assert _get_workspace_diagram_types(request) == {diagram_type}

    def test_a_quantum_circuit_of_empty_cells_is_absent(self):
        request = _editor_request("hello")
        request.context.project_snapshot["diagrams"]["QuantumCircuitDiagram"][0]["model"]["cols"] = [[1, 1], [1]]
        assert _get_workspace_diagram_types(request) == set()

    @pytest.mark.parametrize("diagram_type", sorted(_EDITOR_DEFAULT_PROJECT["diagrams"]))
    def test_a_tab_without_a_model_is_absent(self, diagram_type):
        request = _editor_request("hello")
        request.context.project_snapshot["diagrams"][diagram_type][0]["model"] = None
        assert _get_workspace_diagram_types(request) == set()

    def test_summaries_stand_in_when_no_snapshot_was_sent(self):
        request = _make_request("hello")
        request.context.diagram_summaries = [{"diagramType": "ClassDiagram", "title": "Shop"}]
        assert _get_workspace_diagram_types(request) == {"ClassDiagram"}

    def test_summaries_of_empty_tabs_do_not_make_them_present(self):
        # The agent derives one summary per tab when the editor sends none.
        request = _editor_request("hello")
        request.context.diagram_summaries = [
            {"diagramType": diagram_type, "diagramId": tabs[0]["id"], "title": tabs[0]["title"]}
            for diagram_type, tabs in request.context.project_snapshot["diagrams"].items()
        ]
        assert _get_workspace_diagram_types(request) == set()
