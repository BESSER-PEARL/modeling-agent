def test_bpmn_routing_explicit():
    import importlib, sys
    # Import workspace_orchestrator directly to avoid orchestrator/__init__.py
    # pulling in request_planner -> handlers.generation_handler -> baf (not installed).
    import importlib.util, os
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    spec = importlib.util.spec_from_file_location(
        "orchestrator.workspace_orchestrator",
        os.path.join(src, "orchestrator", "workspace_orchestrator.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("orchestrator.workspace_orchestrator", mod)
    spec.loader.exec_module(mod)
    determine_target_diagram_type = mod.determine_target_diagram_type

    from protocol.types import AssistantRequest, WorkspaceContext
    assert determine_target_diagram_type(
        AssistantRequest(
            message="create a BPMN process for handling an order",
            context=WorkspaceContext(),
        )
    ) == "BPMN"


def test_bpmn_schema_defaults():
    from schemas import SystemBPMNSpec
    d = SystemBPMNSpec(nodes=[{"id": "t", "name": "Do", "type": "task"}]).model_dump()
    assert d["nodes"][0]["taskType"] == "default"


def test_bpmn_modification_target_has_node_id():
    from schemas.bpmn import BPMNModificationTarget
    t = BPMNModificationTarget(nodeId="abc-uuid-123", nodeName=None)
    assert t.nodeId == "abc-uuid-123"
    assert t.nodeName is None


def test_bpmn_validation_adds_start_end():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine(
        {"nodes": [{"id": "t", "name": "Do", "type": "task"}], "flows": []}
    )
    assert {"startEvent", "endEvent"} <= {n["type"] for n in spec["nodes"]}


def test_bpmn_fallback_envelope():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    r = BPMNDiagramHandler(None).generate_fallback_system()
    assert r["action"] == "inject_complete_system" and r["diagramType"] == "BPMN"
    assert "nodes" in r["systemSpec"] and "flows" in r["systemSpec"]


def test_bpmn_model_summary_named_node():
    from utilities.model_context import detailed_model_summary
    model = {
        "elements": {"elem-abc-123": {"type": "BPMNTask", "name": "Check"}},
        "relationships": {},
    }
    summary = detailed_model_summary(model, "BPMN")
    assert "Check" in summary
    assert "[elem-abc-123]" in summary


def test_bpmn_model_summary_unnamed_node_uses_id():
    from utilities.model_context import detailed_model_summary
    model = {
        "elements": {
            "uuid-gate-01": {"type": "BPMNGateway", "name": "", "gatewayType": "parallel"},
            "uuid-gate-02": {"type": "BPMNGateway", "name": "", "gatewayType": "parallel"},
        },
        "relationships": {},
    }
    summary = detailed_model_summary(model, "BPMN")
    # Both unnamed gateways must appear with their distinct element ids — not as "(unnamed)"
    assert "[uuid-gate-01]" in summary
    assert "[uuid-gate-02]" in summary
    assert "(unnamed)" not in summary


def test_bpmn_model_summary_flow_uses_ids():
    from utilities.model_context import detailed_model_summary
    model = {
        "elements": {
            "task-01": {"type": "BPMNTask", "name": "Prepare Draft"},
            "gw-01": {"type": "BPMNGateway", "name": "", "gatewayType": "exclusive"},
        },
        "relationships": {
            "flow-01": {
                "type": "BPMNFlow",
                "source": {"element": "task-01"},
                "target": {"element": "gw-01"},
                "name": "",
            }
        },
    }
    summary = detailed_model_summary(model, "BPMN")
    assert "[task-01]" in summary
    assert "[gw-01]" in summary
    assert "Flow:" in summary


def test_bpmn_suggestions_have_nonempty_prompts():
    from suggestions import get_suggested_actions
    actions = get_suggested_actions("BPMN", "complete_system", [])
    for action in actions:
        assert action.get("prompt"), (
            f"Chip '{action.get('label')}' has empty prompt — WME will no-op when user clicks it"
        )
