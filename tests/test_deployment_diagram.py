"""Tests for DeploymentDiagramHandler.

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

def test_deployment_routing_explicit():
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
        message="create a deployment diagram for a containerized microservices setup",
        context=WorkspaceContext(),
    )) == "DeploymentDiagram"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_deployment_schema_defaults():
    from schemas import SystemDeploymentSpec
    d = SystemDeploymentSpec(
        nodes=[{"id": "n1", "name": "ProductionServer"}]
    ).model_dump()
    assert d["nodes"][0]["stereotype"] == "node"
    assert d["artifacts"] == []
    assert d["deployComponents"] == []
    assert d["dependencies"] == []


def test_deployment_modification_target_fields():
    from schemas.deployment_diagram import DeploymentModificationTarget
    t = DeploymentModificationTarget(elementId="node-abc", elementName=None)
    assert t.elementId == "node-abc"
    assert t.elementName is None


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def test_deployment_fallback_envelope():
    from diagram_handlers.types.deployment_diagram_handler import DeploymentDiagramHandler
    r = DeploymentDiagramHandler(None).generate_fallback_system()
    assert r["action"] == "inject_complete_system"
    assert r["diagramType"] == "DeploymentDiagram"
    spec = r["systemSpec"]
    assert len(spec["nodes"]) >= 1


def test_deployment_single_element_envelope():
    from diagram_handlers.types.deployment_diagram_handler import DeploymentDiagramHandler
    r = DeploymentDiagramHandler(None).generate_single_element("ProductionServer")
    assert r["action"] == "inject_complete_system"
    assert r["diagramType"] == "DeploymentDiagram"
    assert r["systemSpec"]["nodes"][0]["stereotype"] == "node"


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def test_deployment_model_summary_shows_elements():
    from utilities.model_context import detailed_model_summary
    model = {
        "elements": {
            "node-01": {"type": "DeploymentNode", "name": "Production Server", "stereotype": "node"},
            "art-01": {"type": "DeploymentArtifact", "name": "WebApp", "owner": "node-01"},
            "dcomp-01": {"type": "DeploymentComponent", "name": "WebApp", "stereotype": "solution"},
        },
        "relationships": {
            "dep-01": {
                "type": "DeploymentDependency",
                "source": {"element": "art-01"},
                "target": {"element": "dcomp-01"},
            }
        },
    }
    summary = detailed_model_summary(model, "DeploymentDiagram")
    assert "Production Server" in summary
    assert "WebApp" in summary
    assert "[node-01]" in summary
    assert "--->" in summary


# ---------------------------------------------------------------------------
# generate_modification — behaviour
# ---------------------------------------------------------------------------

def test_deployment_generate_modification_element_not_found_returns_assistant_message(monkeypatch):
    """When the LLM signals elementFound=False, generate_modification must return
    an assistant_message — never forward an empty modify_model to the WME."""
    from diagram_handlers.types.deployment_diagram_handler import DeploymentDiagramHandler
    from schemas.deployment_diagram import DeploymentModificationResponse

    not_found = DeploymentModificationResponse(
        elementFound=False,
        modifications=[],
        message="I couldn't find 'GhostNode' in this diagram.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return not_found

    monkeypatch.setattr(DeploymentDiagramHandler, "predict_structured", fake_predict)
    h = DeploymentDiagramHandler(None)
    result = h.generate_modification("remove GhostNode", current_model=None)

    assert result["action"] == "assistant_message"
    assert "GhostNode" in result["message"]


def test_deployment_generate_modification_add_node_returns_modify_model(monkeypatch):
    """A successful add_node modification must produce a modify_model action."""
    from diagram_handlers.types.deployment_diagram_handler import DeploymentDiagramHandler
    from schemas.deployment_diagram import (
        DeploymentModificationResponse, DeploymentModification,
        DeploymentModificationTarget, DeploymentModificationChanges,
    )

    ok_response = DeploymentModificationResponse(
        elementFound=True,
        modifications=[
            DeploymentModification(
                action="add_node",
                target=DeploymentModificationTarget(elementName="StagingServer"),
                changes=DeploymentModificationChanges(stereotype="server"),
            )
        ],
        message="Added StagingServer node.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(DeploymentDiagramHandler, "predict_structured", fake_predict)
    h = DeploymentDiagramHandler(None)
    result = h.generate_modification("add a StagingServer node", current_model=None)

    assert result["action"] == "modify_model"
    assert "StagingServer" in result.get("message", "")


def test_deployment_generate_modification_add_artifact_returns_modify_model(monkeypatch):
    """add_artifact with a known owner node produces modify_model."""
    from diagram_handlers.types.deployment_diagram_handler import DeploymentDiagramHandler
    from schemas.deployment_diagram import (
        DeploymentModificationResponse, DeploymentModification,
        DeploymentModificationTarget, DeploymentModificationChanges,
    )

    model = {
        "elements": {
            "node-01": {"type": "DeploymentNode", "name": "ProductionServer"},
        }
    }

    ok_response = DeploymentModificationResponse(
        elementFound=True,
        modifications=[
            DeploymentModification(
                action="add_artifact",
                target=DeploymentModificationTarget(elementName="APIServer"),
                changes=DeploymentModificationChanges(owner="node-01"),
            )
        ],
        message="Added APIServer artifact.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(DeploymentDiagramHandler, "predict_structured", fake_predict)
    h = DeploymentDiagramHandler(None)
    result = h.generate_modification(
        "add an APIServer artifact to ProductionServer", current_model=model
    )

    assert result["action"] == "modify_model"
    assert "APIServer" in result.get("message", "")


# ---------------------------------------------------------------------------
# Server-side ref guardrail
# ---------------------------------------------------------------------------

def test_deployment_guardrail_drops_modification_with_hallucinated_ref(monkeypatch):
    """When the LLM says elementFound=True but the target ID doesn't exist,
    the server-side guardrail must catch it."""
    from diagram_handlers.types.deployment_diagram_handler import DeploymentDiagramHandler
    from schemas.deployment_diagram import (
        DeploymentModificationResponse, DeploymentModification,
        DeploymentModificationTarget,
    )

    model = {
        "elements": {
            "node-real": {"type": "DeploymentNode", "name": "RealNode"},
        }
    }

    hallucinated = DeploymentModificationResponse(
        elementFound=True,  # LLM lies — element doesn't exist
        modifications=[
            DeploymentModification(
                action="remove_element",
                target=DeploymentModificationTarget(elementId="ghost-uuid-999"),
                changes=None,
            )
        ],
        message="Removed ghost node.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return hallucinated

    monkeypatch.setattr(DeploymentDiagramHandler, "predict_structured", fake_predict)
    h = DeploymentDiagramHandler(None)
    result = h.generate_modification("remove the non-existent node", current_model=model)

    assert result["action"] == "assistant_message"


def test_deployment_guardrail_drops_dependency_with_missing_target(monkeypatch):
    """add_dependency whose target doesn't exist in the model is dropped by the guardrail."""
    from diagram_handlers.types.deployment_diagram_handler import DeploymentDiagramHandler
    from schemas.deployment_diagram import (
        DeploymentModificationResponse, DeploymentModification,
        DeploymentModificationTarget, DeploymentModificationChanges,
    )

    model = {
        "elements": {
            "art-real": {"type": "DeploymentArtifact", "name": "RealArtifact"},
        }
    }

    bad_dep = DeploymentModificationResponse(
        elementFound=True,
        modifications=[
            DeploymentModification(
                action="add_dependency",
                target=DeploymentModificationTarget(),
                changes=DeploymentModificationChanges(source="art-real", target="ghost-target"),
            )
        ],
        message="Added dependency.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return bad_dep

    monkeypatch.setattr(DeploymentDiagramHandler, "predict_structured", fake_predict)
    h = DeploymentDiagramHandler(None)
    result = h.generate_modification("connect RealArtifact to ghost", current_model=model)

    assert result["action"] == "assistant_message"


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

def test_deployment_suggestions_have_nonempty_prompts():
    from suggestions import get_suggested_actions
    actions = get_suggested_actions("DeploymentDiagram", "complete_system", [])
    for action in actions:
        assert action.get("prompt"), (
            f"Chip '{action.get('label')}' has empty prompt"
        )
