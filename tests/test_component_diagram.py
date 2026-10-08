"""Tests for ComponentDiagramHandler.

Covers: routing, schema defaults, fallback, model summary,
generate_modification behaviour, and the server-side ref guardrail.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_component_routing_explicit():
    import importlib.util, os
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    spec = importlib.util.spec_from_file_location(
        "orchestrator.workspace_orchestrator",
        os.path.join(src, "orchestrator", "workspace_orchestrator.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("orchestrator.workspace_orchestrator", mod)
    spec.loader.exec_module(mod)
    determine = mod.determine_target_diagram_type

    from protocol.types import AssistantRequest, WorkspaceContext
    assert determine(AssistantRequest(
        message="create a component diagram for an order processing system",
        context=WorkspaceContext(),
    )) == "ComponentDiagram"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_component_schema_defaults():
    from schemas import SystemComponentSpec
    d = SystemComponentSpec(
        components=[{"id": "svc", "name": "OrderService", "owner": None}]
    ).model_dump()
    assert d["components"][0]["stereotype"] == "solution"
    assert d["subsystems"] == []
    assert d["dependencies"] == []


def test_component_modification_target_fields():
    from schemas.component_diagram import ComponentModificationTarget
    t = ComponentModificationTarget(elementId="abc-123", elementName=None)
    assert t.elementId == "abc-123"
    assert t.elementName is None


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def test_component_fallback_envelope():
    from diagram_handlers.types.component_diagram_handler import ComponentDiagramHandler
    r = ComponentDiagramHandler(None).generate_fallback_system()
    assert r["action"] == "inject_complete_system"
    assert r["diagramType"] == "ComponentDiagram"
    spec = r["systemSpec"]
    assert len(spec["components"]) >= 1
    assert len(spec["subsystems"]) >= 1


def test_component_single_element_envelope():
    from diagram_handlers.types.component_diagram_handler import ComponentDiagramHandler
    r = ComponentDiagramHandler(None).generate_single_element("OrderService")
    assert r["action"] == "inject_complete_system"
    assert r["diagramType"] == "ComponentDiagram"
    assert r["systemSpec"]["components"][0]["stereotype"] == "solution"


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def test_component_model_summary_shows_elements():
    from utilities.model_context import detailed_model_summary
    model = {
        "elements": {
            "sub-01": {"type": "Subsystem", "name": "Backend", "stereotype": "subsystem", "owner": None},
            "comp-01": {"type": "Component", "name": "UserService", "stereotype": "solution", "owner": "sub-01"},
        },
        "relationships": {
            "dep-01": {
                "type": "ComponentDependency",
                "source": {"element": "comp-01"},
                "target": {"element": "sub-01"},
                "stereotype": "uses",
            }
        },
    }
    summary = detailed_model_summary(model, "ComponentDiagram")
    assert "Backend" in summary
    assert "UserService" in summary
    assert "[comp-01]" in summary
    assert "--uses-->" in summary


# ---------------------------------------------------------------------------
# generate_modification — behaviour
# ---------------------------------------------------------------------------

def test_component_generate_modification_element_not_found_returns_assistant_message(monkeypatch):
    """When the LLM signals elementFound=False, generate_modification must return
    an assistant_message — never forward an empty modify_model to the WME."""
    from diagram_handlers.types.component_diagram_handler import ComponentDiagramHandler
    from schemas.component_diagram import ComponentModificationResponse

    not_found = ComponentModificationResponse(
        elementFound=False,
        modifications=[],
        message="I couldn't find 'GhostService' in this diagram.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return not_found

    monkeypatch.setattr(ComponentDiagramHandler, "predict_structured", fake_predict)
    h = ComponentDiagramHandler(None)
    result = h.generate_modification("remove GhostService", current_model=None)

    assert result["action"] == "assistant_message"
    assert "GhostService" in result["message"]


def test_component_generate_modification_add_component_returns_modify_model(monkeypatch):
    """A successful add_component modification must produce a modify_model action."""
    from diagram_handlers.types.component_diagram_handler import ComponentDiagramHandler
    from schemas.component_diagram import (
        ComponentModificationResponse, ComponentModification,
        ComponentModificationTarget, ComponentModificationChanges,
    )

    ok_response = ComponentModificationResponse(
        elementFound=True,
        modifications=[
            ComponentModification(
                action="add_component",
                target=ComponentModificationTarget(elementName="PaymentService"),
                changes=ComponentModificationChanges(stereotype="solution", owner="backend"),
            )
        ],
        message="Added PaymentService component.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(ComponentDiagramHandler, "predict_structured", fake_predict)
    h = ComponentDiagramHandler(None)
    result = h.generate_modification("add a PaymentService component", current_model=None)

    assert result["action"] == "modify_model"
    assert "PaymentService" in result.get("message", "")


def test_component_generate_modification_add_dependency_returns_modify_model(monkeypatch):
    """add_dependency with valid source/target produces modify_model."""
    from diagram_handlers.types.component_diagram_handler import ComponentDiagramHandler
    from schemas.component_diagram import (
        ComponentModificationResponse, ComponentModification,
        ComponentModificationTarget, ComponentModificationChanges,
    )

    model = {
        "elements": {
            "comp-01": {"type": "Component", "name": "OrderService"},
            "comp-02": {"type": "Component", "name": "PaymentService"},
        }
    }

    ok_response = ComponentModificationResponse(
        elementFound=True,
        modifications=[
            ComponentModification(
                action="add_dependency",
                target=ComponentModificationTarget(),
                changes=ComponentModificationChanges(
                    source="comp-01", target="comp-02", dependencyStereotype="uses"
                ),
            )
        ],
        message="Added dependency.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(ComponentDiagramHandler, "predict_structured", fake_predict)
    h = ComponentDiagramHandler(None)
    result = h.generate_modification(
        "connect OrderService to PaymentService", current_model=model
    )

    assert result["action"] == "modify_model"


# ---------------------------------------------------------------------------
# Server-side ref guardrail
# ---------------------------------------------------------------------------

def test_component_guardrail_drops_modification_with_hallucinated_ref(monkeypatch):
    """When the LLM says elementFound=True but the target ID doesn't exist in
    current_model, the server-side guardrail must catch it."""
    from diagram_handlers.types.component_diagram_handler import ComponentDiagramHandler
    from schemas.component_diagram import (
        ComponentModificationResponse, ComponentModification,
        ComponentModificationTarget,
    )

    model = {
        "elements": {
            "comp-real": {"type": "Component", "name": "RealService"},
        }
    }

    hallucinated = ComponentModificationResponse(
        elementFound=True,  # LLM lies — element doesn't exist
        modifications=[
            ComponentModification(
                action="remove_element",
                target=ComponentModificationTarget(elementId="ghost-uuid-999"),
                changes=None,
            )
        ],
        message="Removed ghost element.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return hallucinated

    monkeypatch.setattr(ComponentDiagramHandler, "predict_structured", fake_predict)
    h = ComponentDiagramHandler(None)
    result = h.generate_modification("remove the non-existent component", current_model=model)

    assert result["action"] == "assistant_message"


def test_component_guardrail_drops_dependency_with_missing_source(monkeypatch):
    """add_dependency whose source doesn't exist in the model is dropped by the guardrail."""
    from diagram_handlers.types.component_diagram_handler import ComponentDiagramHandler
    from schemas.component_diagram import (
        ComponentModificationResponse, ComponentModification,
        ComponentModificationTarget, ComponentModificationChanges,
    )

    model = {
        "elements": {
            "comp-real": {"type": "Component", "name": "RealService"},
        }
    }

    bad_dep = ComponentModificationResponse(
        elementFound=True,
        modifications=[
            ComponentModification(
                action="add_dependency",
                target=ComponentModificationTarget(),
                changes=ComponentModificationChanges(source="ghost-src", target="comp-real"),
            )
        ],
        message="Added dependency.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return bad_dep

    monkeypatch.setattr(ComponentDiagramHandler, "predict_structured", fake_predict)
    h = ComponentDiagramHandler(None)
    result = h.generate_modification("connect ghost to RealService", current_model=model)

    assert result["action"] == "assistant_message"


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

def test_component_suggestions_have_nonempty_prompts():
    from suggestions import get_suggested_actions
    actions = get_suggested_actions("ComponentDiagram", "complete_system", [])
    for action in actions:
        assert action.get("prompt"), (
            f"Chip '{action.get('label')}' has empty prompt"
        )
