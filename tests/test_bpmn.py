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


def test_bpmn_generate_modification_element_not_found_returns_assistant_message(monkeypatch):
    """When the LLM signals elementFound=False, generate_modification must return
    an assistant_message action — never forward an empty modify_model to the WME."""
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas.bpmn import BPMNModificationResponse

    not_found_response = BPMNModificationResponse(
        elementFound=False,
        modifications=[],
        message="I couldn't find 'Buy Groceries' in this diagram.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return not_found_response

    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", fake_predict)
    h = BPMNDiagramHandler(None)
    result = h.generate_modification("remove Buy Groceries", current_model=None)

    assert result["action"] == "assistant_message"
    assert "Buy Groceries" in result["message"]


def test_bpmn_generate_modification_add_task_returns_modify_model(monkeypatch):
    """A successful add_task modification must produce a modify_model action
    with the task name present in the message."""
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas.bpmn import BPMNModificationResponse, BPMNModification, BPMNModificationTarget

    ok_response = BPMNModificationResponse(
        elementFound=True,
        modifications=[
            BPMNModification(
                action="add_task",
                target=BPMNModificationTarget(nodeName="Send Invoice"),
                changes={"taskType": "send"},
            )
        ],
        message="Added Send Invoice task.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", fake_predict)
    h = BPMNDiagramHandler(None)
    result = h.generate_modification("add a Send Invoice task", current_model=None)

    assert result["action"] == "modify_model"
    assert "Send Invoice" in result.get("message", "")


def test_bpmn_generate_modification_add_flow_message_shows_arrow(monkeypatch):
    """add_flow modification message must show source → target names resolved
    from the current model, not raw element IDs."""
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas.bpmn import BPMNModificationResponse, BPMNModification, BPMNModificationTarget, BPMNModificationChanges

    model = {
        "elements": {
            "task-01": {"type": "BPMNTask", "name": "Ship Order", "taskType": "default"},
            "task-02": {"type": "BPMNTask", "name": "Send Invoice", "taskType": "send"},
        }
    }

    ok_response = BPMNModificationResponse(
        elementFound=True,
        modifications=[
            BPMNModification(
                action="add_flow",
                target=BPMNModificationTarget(nodeName=None),
                changes=BPMNModificationChanges(source="task-01", target="task-02"),
            )
        ],
        message="Added flow.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", fake_predict)
    h = BPMNDiagramHandler(None)
    result = h.generate_modification("connect Ship Order to Send Invoice", current_model=model)

    assert result["action"] == "modify_model"
    msg = result.get("message", "")
    assert "Ship Order" in msg
    assert "Send Invoice" in msg
    assert "→" in msg


def test_bpmn_generate_modification_remove_flow(monkeypatch):
    """remove_flow must produce a modify_model result."""
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas.bpmn import BPMNModificationResponse, BPMNModification, BPMNModificationTarget, BPMNModificationChanges

    model = {
        "elements": {
            "task-01": {"type": "BPMNTask", "name": "Check Stock"},
            "task-02": {"type": "BPMNTask", "name": "Ship Order"},
        }
    }

    ok_response = BPMNModificationResponse(
        elementFound=True,
        modifications=[
            BPMNModification(
                action="remove_flow",
                target=BPMNModificationTarget(nodeName=None),
                changes=BPMNModificationChanges(source="task-01", target="task-02"),
            )
        ],
        message="Removed flow.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", fake_predict)
    h = BPMNDiagramHandler(None)
    result = h.generate_modification("remove the flow between Check Stock and Ship Order", current_model=model)

    assert result["action"] == "modify_model"
    msg = result.get("message", "")
    assert "Check Stock" in msg
    assert "Ship Order" in msg


def test_bpmn_guardrail_drops_modification_with_hallucinated_ref(monkeypatch):
    """When the LLM says elementFound=True but the target ID doesn't exist,
    the server-side guardrail must catch it and return an assistant_message."""
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas.bpmn import BPMNModificationResponse, BPMNModification, BPMNModificationTarget

    model = {
        "elements": {
            "task-real": {"type": "BPMNTask", "name": "Real Task"},
        }
    }

    hallucinated_response = BPMNModificationResponse(
        elementFound=True,  # LLM lies — element doesn't exist
        modifications=[
            BPMNModification(
                action="remove_element",
                target=BPMNModificationTarget(nodeId="ghost-uuid-999", nodeName=None),
                changes=None,
            )
        ],
        message="Removed ghost element.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return hallucinated_response

    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", fake_predict)
    h = BPMNDiagramHandler(None)
    result = h.generate_modification("remove the non-existent element", current_model=model)

    assert result["action"] == "assistant_message"


def test_bpmn_guardrail_unnamed_element_resolved_by_type_label(monkeypatch):
    """remove_element targeting an unnamed node by ID shows its type label in
    the message, not the raw Apollon UUID."""
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas.bpmn import BPMNModificationResponse, BPMNModification, BPMNModificationTarget

    model = {
        "elements": {
            "uuid-gw-01": {"type": "BPMNGateway", "name": "", "gatewayType": "parallel"},
        }
    }

    ok_response = BPMNModificationResponse(
        elementFound=True,
        modifications=[
            BPMNModification(
                action="remove_element",
                target=BPMNModificationTarget(nodeId="uuid-gw-01", nodeName=None),
                changes=None,
            )
        ],
        message="Removed gateway.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", fake_predict)
    h = BPMNDiagramHandler(None)
    result = h.generate_modification("remove the parallel gateway", current_model=model)

    assert result["action"] == "modify_model"
    msg = result.get("message", "")
    assert "Parallel Gateway" in msg
    assert "uuid-gw-01" not in msg


def test_bpmn_suggestions_have_nonempty_prompts():
    from suggestions import get_suggested_actions
    actions = get_suggested_actions("BPMN", "complete_system", [])
    for action in actions:
        assert action.get("prompt"), (
            f"Chip '{action.get('label')}' has empty prompt — WME will no-op when user clicks it"
        )
