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


def test_bpmn_schema_pools_default_empty():
    from schemas import SystemBPMNSpec
    d = SystemBPMNSpec(nodes=[{"id": "t", "name": "Do", "type": "task"}]).model_dump()
    assert d["pools"] == []
    assert d["nodes"][0]["poolId"] is None
    assert d["nodes"][0]["laneId"] is None
    assert d["nodes"][0]["owner"] is None


def test_bpmn_schema_pool_with_lanes_round_trips():
    from schemas import SystemBPMNSpec
    d = SystemBPMNSpec(
        nodes=[{"id": "bake", "name": "Bake Pizza", "type": "task", "poolId": "vendor", "laneId": "chef"}],
        pools=[
            {"id": "vendor", "name": "Pizza Vendor", "lanes": [
                {"id": "chef", "name": "Pizza Chef"},
                {"id": "clerk", "name": "Clerk"},
            ]},
        ],
    ).model_dump()
    assert d["pools"][0]["id"] == "vendor"
    assert {l["id"] for l in d["pools"][0]["lanes"]} == {"chef", "clerk"}
    assert d["nodes"][0]["poolId"] == "vendor"
    assert d["nodes"][0]["laneId"] == "chef"
    assert d["nodes"][0]["owner"] is None


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


def test_bpmn_validation_connects_orphaned_end_event_via_underconnected_gateway():
    """Reproduces a real generation bug: an exclusive gateway with only one
    outgoing flow ('yes') and a matching end event ('Order Cancelled') that the
    model clearly intended as the 'no' branch but never emitted a flow for."""
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [
            {"id": "start", "name": "Order Placed", "type": "startEvent"},
            {"id": "check_stock", "name": "Check Stock", "type": "task"},
            {"id": "gw", "name": "Items Available?", "type": "gateway"},
            {"id": "prepare", "name": "Prepare Package", "type": "task"},
            {"id": "completed", "name": "Order Completed", "type": "endEvent"},
            {"id": "cancelled", "name": "Order Cancelled", "type": "endEvent"},
        ],
        "flows": [
            {"source": "start", "target": "check_stock", "name": ""},
            {"source": "check_stock", "target": "gw", "name": ""},
            {"source": "gw", "target": "prepare", "name": "yes"},
            {"source": "prepare", "target": "completed", "name": ""},
        ],
    })
    targets = {f["target"] for f in spec["flows"]}
    assert "cancelled" in targets, "orphaned end event must get an incoming flow"
    new_flow = next(f for f in spec["flows"] if f["target"] == "cancelled")
    assert new_flow["source"] == "gw", "should reconnect from the under-connected gateway"
    assert new_flow["name"] == "no", "should infer the opposite label of the existing 'yes' branch"


def test_bpmn_validation_connects_orphaned_task_falls_back_to_previous_node():
    """No gateway is available to reconnect from -- falls back to the
    previous node in generation order rather than leaving it disconnected."""
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [
            {"id": "start", "name": "Start", "type": "startEvent"},
            {"id": "a", "name": "Do A", "type": "task"},
            {"id": "b", "name": "Do B", "type": "task"},
            {"id": "end", "name": "End", "type": "endEvent"},
        ],
        "flows": [
            {"source": "start", "target": "a", "name": ""},
            # "b" is never connected -- the bug this test guards against.
            {"source": "a", "target": "end", "name": ""},
        ],
    })
    targets = {f["target"] for f in spec["flows"]}
    assert "b" in targets
    new_flow = next(f for f in spec["flows"] if f["target"] == "b")
    assert new_flow["source"] == "a"


def test_bpmn_validation_drops_dangling_pool_ref():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [{"id": "t", "name": "Do", "type": "task", "poolId": "ghost", "laneId": "ghost_lane"}],
        "flows": [],
        "pools": [{"id": "real", "name": "Real Pool", "lanes": []}],
    })
    node = next(n for n in spec["nodes"] if n["id"] == "t")
    assert node["poolId"] is None
    assert node["laneId"] is None
    assert node["owner"] is None


def test_bpmn_validation_drops_dangling_lane_ref_keeps_pool():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [{"id": "t", "name": "Do", "type": "task", "poolId": "vendor", "laneId": "ghost_lane"}],
        "flows": [],
        "pools": [{"id": "vendor", "name": "Vendor", "lanes": [{"id": "chef", "name": "Chef"}]}],
    })
    node = next(n for n in spec["nodes"] if n["id"] == "t")
    assert node["poolId"] == "vendor"
    assert node["laneId"] is None
    assert node["owner"] is None


def test_bpmn_validation_keeps_valid_pool_and_lane_refs():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [{"id": "t", "name": "Do", "type": "task", "poolId": "vendor", "laneId": "chef"}],
        "flows": [],
        "pools": [{"id": "vendor", "name": "Vendor", "lanes": [{"id": "chef", "name": "Chef"}]}],
    })
    node = next(n for n in spec["nodes"] if n["id"] == "t")
    assert node["poolId"] == "vendor"
    assert node["laneId"] == "chef"
    assert node["owner"] == "chef"
    assert spec["pools"][0]["id"] == "vendor"


def test_bpmn_validation_no_pools_clears_all_refs():
    """A node that hallucinates a poolId when the spec declares no pools at
    all must be normalized back to a flat node."""
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [{"id": "t", "name": "Do", "type": "task", "poolId": "vendor", "laneId": "chef"}],
        "flows": [],
    })
    node = next(n for n in spec["nodes"] if n["id"] == "t")
    assert node["poolId"] is None
    assert node["laneId"] is None
    assert node["owner"] is None
    assert spec["pools"] == []


def test_bpmn_validation_infers_lane_owner_from_neighbors():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [
            {"id": "start", "name": "Patient Arrives", "type": "startEvent", "poolId": "hospital"},
            {"id": "register", "name": "Register Patient", "type": "task", "taskType": "user", "poolId": "hospital", "laneId": "receptionist"},
            {"id": "vitals", "name": "Take Vitals", "type": "task", "taskType": "manual", "poolId": "hospital", "laneId": "nurse"},
            {"id": "examine", "name": "Examine Patient", "type": "task", "taskType": "user", "poolId": "hospital", "laneId": "doctor"},
            {"id": "end", "name": "Patient Examined", "type": "endEvent", "poolId": "hospital"},
        ],
        "flows": [
            {"source": "start", "target": "register", "name": ""},
            {"source": "register", "target": "vitals", "name": ""},
            {"source": "vitals", "target": "examine", "name": ""},
            {"source": "examine", "target": "end", "name": ""},
        ],
        "pools": [{
            "id": "hospital",
            "name": "Hospital",
            "lanes": [
                {"id": "receptionist", "name": "Receptionist"},
                {"id": "nurse", "name": "Nurse"},
                {"id": "doctor", "name": "Doctor"},
            ],
        }],
    })
    nodes = {n["id"]: n for n in spec["nodes"]}
    assert nodes["start"]["laneId"] == "receptionist"
    assert nodes["start"]["owner"] == "receptionist"
    assert nodes["end"]["laneId"] == "doctor"
    assert nodes["end"]["owner"] == "doctor"



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

