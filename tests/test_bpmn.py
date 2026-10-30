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


def test_bpmn_validation_replaces_dangling_pool_ref_with_a_declared_pool():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [{"id": "t", "name": "Do", "type": "task", "poolId": "ghost", "laneId": "ghost_lane"}],
        "flows": [],
        "pools": [{"id": "real", "name": "Real Pool", "lanes": []}],
    })
    node = next(n for n in spec["nodes"] if n["id"] == "t")
    assert node["poolId"] == "real"
    assert node["laneId"] is None
    assert node["owner"] is None


def test_bpmn_validation_replaces_dangling_lane_ref_with_a_lane_of_the_pool():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [{"id": "t", "name": "Do", "type": "task", "poolId": "vendor", "laneId": "ghost_lane"}],
        "flows": [],
        "pools": [{"id": "vendor", "name": "Vendor", "lanes": [{"id": "chef", "name": "Chef"}]}],
    })
    node = next(n for n in spec["nodes"] if n["id"] == "t")
    assert node["poolId"] == "vendor"
    assert node["laneId"] == "chef"
    assert node["owner"] == "chef"


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

def test_agentic_bpmn_schema_preserves_lanes_and_governance():
    from schemas import SystemBPMNSpec

    spec = SystemBPMNSpec(
        systemName="Review Swarm",
        pools=[
            {
                "id": "swarm",
                "name": "Review Swarm",
                "lanes": [
                    {
                        "id": "supervisor",
                        "name": "Supervisor",
                        "isAgentic": True,
                        "role": "supervision",
                        "trustScore": 90,
                        "multiplicity": 1,
                    },
                    {
                        "id": "reviewers",
                        "name": "Reviewer",
                        "isAgentic": True,
                        "role": "solution",
                        "trustScore": 75,
                        "multiplicity": 3,
                    },
                ],
            }
        ],
        nodes=[
            {
                "id": "start",
                "name": "Start",
                "type": "startEvent",
                "poolId": "swarm",
                "laneId": "supervisor",
                "owner": "supervisor",
            },
            {
                "id": "merge",
                "name": "Select Result",
                "type": "gateway",
                "gatewayType": "parallel",
                "poolId": "swarm",
                "laneId": "supervisor",
                "owner": "supervisor",
                "isAgentic": True,
                "gatewayRole": "merging",
                "trustScore": 90,
                "governancePolicy": "MajorityPolicy",
            },
        ],
    ).model_dump()

    assert spec["pools"][0]["lanes"][0]["role"] == "supervision"
    assert spec["pools"][0]["lanes"][1]["multiplicity"] == 3
    assert "swimlanes" not in spec["pools"][0]
    assert spec["nodes"][1]["governancePolicy"] == "MajorityPolicy"
    assert spec["nodes"][1]["laneId"] == "supervisor"


def test_bpmn_model_summary_includes_agentic_lane_and_governance():
    from utilities.model_context import detailed_model_summary

    model = {
        "elements": {
            "pool-1": {"type": "BPMNPool", "name": "Review Swarm"},
            "lane-1": {
                "type": "BPMNSwimlane",
                "name": "Supervisor",
                "owner": "pool-1",
                "isAgentic": True,
                "role": "supervision",
                "trustScore": 90,
                "multiplicity": 1,
            },
            "gateway-1": {
                "type": "BPMNGateway",
                "name": "Select Result",
                "owner": "lane-1",
                "gatewayType": "parallel",
                "isAgentic": True,
                "gatewayRole": "merging",
                "trustScore": 90,
                "governanceDsl": "Policy: MajorityPolicy",
            },
        },
        "relationships": {},
    }

    summary = detailed_model_summary(model, "BPMN")

    assert "Pool: [pool-1] Review Swarm" in summary
    assert "role=supervision" in summary
    assert "multiplicity=1" in summary
    assert "lane=Supervisor" in summary
    assert "role=merging" in summary
    assert "governanceDsl=set" in summary


# -- Every flow node of a pooled process sits in a pool (and lane) ----------
# Live repro (agentic PR review): the start event, both gateways and the end
# events came back with poolId/laneId null, landed outside every pool, and the
# validator reported "SequenceFlow crosses a process boundary" for each flow
# between them and the pooled review tasks.

def _pr_review_spec():
    def node(nid, ntype, pool=None, lane=None, **extra):
        return {"id": nid, "name": nid, "type": ntype, "poolId": pool, "laneId": lane, "owner": lane, **extra}

    return {
        "systemName": "Pull Request Review Process",
        "pools": [
            {"id": "reviewerAgents", "name": "Reviewer Agents",
             "lanes": [{"id": "reviewer", "name": "Reviewer", "isAgentic": True, "role": "solution"}]},
            {"id": "mergeAgent", "name": "Merge Agent",
             "lanes": [{"id": "merger", "name": "Merge Agent", "isAgentic": True, "role": "supervision"}]},
        ],
        "nodes": [
            node("startEvent", "startEvent"),
            node("split", "gateway", gatewayType="parallel", isAgentic=True, gatewayRole="diverging"),
            node("security", "task", "reviewerAgents", "reviewer"),
            node("style", "task", "reviewerAgents", "reviewer"),
            node("join", "gateway", gatewayType="parallel", isAgentic=True, gatewayRole="merging"),
            node("merge", "task", "mergeAgent", "merger"),
            node("merged", "endEvent"),
            node("failed", "endEvent"),
        ],
        "flows": [
            {"source": "startEvent", "target": "split"},
            {"source": "split", "target": "security"},
            {"source": "split", "target": "style"},
            {"source": "security", "target": "join"},
            {"source": "style", "target": "join"},
            {"source": "join", "target": "merge"},
            {"source": "merge", "target": "merged"},
            {"source": "merge", "target": "failed"},
        ],
    }


# ---------------------------------------------------------------------------
# Data objects (BESSER-PEARL/BESSER#627): the assistant used to have no way to
# emit them, so "show each input as a data object" was silently dropped, and
# a follow-up asking for the missing data objects could not be applied.
# ---------------------------------------------------------------------------

def _battery_spec():
    """Two steps of the battery-cell process from the issue, with their inputs."""
    return {
        "systemName": "Cell Production",
        "nodes": [
            {"id": "start", "name": "Cell production started", "type": "startEvent"},
            {"id": "mixing", "name": "Mixing", "type": "task"},
            {"id": "coating", "name": "Coating", "type": "task"},
            {"id": "end", "name": "Cell ready", "type": "endEvent"},
        ],
        "flows": [
            {"source": "start", "target": "mixing", "name": ""},
            {"source": "mixing", "target": "coating", "name": ""},
            {"source": "coating", "target": "end", "name": ""},
        ],
        "dataObjects": [
            {"id": "binder", "name": "Binder"},
            {"id": "solvent", "name": "Solvent"},
            {"id": "slurry", "name": "Slurry"},
        ],
        "dataAssociations": [
            {"dataObject": "binder", "node": "mixing", "direction": "input"},
            {"dataObject": "solvent", "node": "mixing", "direction": "input"},
            {"dataObject": "slurry", "node": "mixing", "direction": "output"},
            {"dataObject": "slurry", "node": "coating", "direction": "input"},
        ],
    }


def _assert_every_node_placed(spec):
    lanes_of = {p["id"]: {lane["id"] for lane in p["lanes"]} for p in spec["pools"]}
    for node in spec["nodes"]:
        assert node["poolId"] in lanes_of, node
        assert node["laneId"] in lanes_of[node["poolId"]], node
        assert node["owner"] == node["laneId"], node


def test_bpmn_pooled_process_places_events_and_gateways_by_their_neighbours():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine(_pr_review_spec())
    _assert_every_node_placed(spec)
    nodes = {n["id"]: n for n in spec["nodes"]}
    # Chained through the unplaced split gateway to the reviewer tasks.
    assert (nodes["startEvent"]["poolId"], nodes["startEvent"]["laneId"]) == ("reviewerAgents", "reviewer")
    assert nodes["split"]["laneId"] == "reviewer"
    # A predecessor's lane wins over a successor's.
    assert nodes["join"]["laneId"] == "reviewer"
    assert nodes["merged"]["laneId"] == nodes["failed"]["laneId"] == "merger"


def test_bpmn_pooled_process_without_placed_neighbours_uses_the_first_lane():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [
            {"id": "start", "name": "Start", "type": "startEvent"},
            {"id": "do", "name": "Do", "type": "task"},
            {"id": "end", "name": "End", "type": "endEvent", "poolId": "shop"},
        ],
        "flows": [{"source": "start", "target": "do"}, {"source": "do", "target": "end"}],
        "pools": [{"id": "shop", "name": "Shop", "lanes": [{"id": "clerk", "name": "Clerk"},
                                                          {"id": "chef", "name": "Chef"}]}],
    })
    _assert_every_node_placed(spec)
    assert {n["laneId"] for n in spec["nodes"]} == {"clerk"}


def test_bpmn_node_in_a_laneless_pool_keeps_no_lane():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine({
        "nodes": [
            {"id": "order", "name": "Order", "type": "task", "poolId": "customer"},
            {"id": "start", "name": "Hungry", "type": "startEvent"},
        ],
        "flows": [{"source": "start", "target": "order"}],
        "pools": [{"id": "customer", "name": "Customer", "lanes": []}],
    })
    assert {(n["poolId"], n["laneId"], n["owner"]) for n in spec["nodes"]} == {("customer", None, None)}


def test_bpmn_prompts_require_pool_membership_for_events_and_gateways():
    from diagram_handlers.types.bpmn_diagram_handler import (
        AGENTIC_DESIGN_RULES, MODIFY_SYSTEM_PROMPT_BPMN, BPMNDiagramHandler,
    )
    from schemas.bpmn import BPMNNodeSpec
    for text in (BPMNDiagramHandler(None).get_system_prompt(), AGENTIC_DESIGN_RULES,
                 BPMNNodeSpec.model_fields["poolId"].description):
        assert "start/end events and gateways" in text
    assert "changes.owner" in MODIFY_SYSTEM_PROMPT_BPMN


_LANED_MODEL = {"elements": {
    "pool-1": {"type": "BPMNPool", "name": "Review Swarm", "owner": None},
    "lane-w": {"type": "BPMNSwimlane", "name": "Writer", "owner": "pool-1"},
    "lane-s": {"type": "BPMNSwimlane", "name": "Supervisor", "owner": "pool-1"},
    "t-draft": {"type": "BPMNTask", "name": "Draft Answer", "owner": "lane-w"},
    "t-check": {"type": "BPMNTask", "name": "Check Answer", "owner": "lane-s"},
}, "relationships": {}}


def _bpmn_modify(monkeypatch, model, mods):
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas import BPMNModificationResponse

    monkeypatch.setattr(
        BPMNDiagramHandler, "predict_structured",
        lambda self, prompt, schema, **kw: BPMNModificationResponse.model_validate(
            {"modifications": mods, "message": "ok"}),
    )
    result = BPMNDiagramHandler(None).generate_modification("add it", current_model=model)
    return result.get("modifications") or [result.get("modification")]


def _added_owners(mods):
    return {m["target"]["nodeName"]: (m.get("changes") or {}).get("owner")
            for m in mods if m["action"] in ("add_task", "add_gateway", "add_event")}


def test_bpmn_added_nodes_go_into_the_lane_of_the_node_they_connect_to(monkeypatch):
    mods = _bpmn_modify(monkeypatch, _LANED_MODEL, [
        {"action": "add_gateway", "target": {"nodeName": "Approved?"}},
        {"action": "add_event", "target": {"nodeName": "Published"}, "changes": {"eventKind": "end"}},
        {"action": "add_flow", "target": {}, "changes": {"source": "t-check", "target": "Approved?"}},
        {"action": "add_flow", "target": {}, "changes": {"source": "Approved?", "target": "Published"}},
    ])
    assert _added_owners(mods) == {"Approved?": "lane-s", "Published": "lane-s"}


def test_bpmn_added_node_without_a_neighbour_goes_into_the_first_lane(monkeypatch):
    mods = _bpmn_modify(monkeypatch, _LANED_MODEL, [
        {"action": "add_task", "target": {"nodeName": "Log Result"}},
        {"action": "add_task", "target": {"nodeName": "Escalate"}, "changes": {"owner": "Supervisor"}},
    ])
    assert _added_owners(mods) == {"Log Result": "lane-w", "Escalate": "Supervisor"}


def test_bpmn_added_node_is_not_forced_into_a_lane_when_a_pool_has_none(monkeypatch):
    model = {"elements": {
        "c-pool": {"type": "BPMNPool", "name": "Customer", "owner": None},
        "v-pool": {"type": "BPMNPool", "name": "Vendor", "owner": None},
        "v-lane": {"type": "BPMNSwimlane", "name": "Clerk", "owner": "v-pool"},
        "c-order": {"type": "BPMNTask", "name": "Order Pizza", "owner": "c-pool"},
    }, "relationships": {}}
    mods = _bpmn_modify(monkeypatch, model, [
        {"action": "add_task", "target": {"nodeName": "Pay"}},
        {"action": "add_task", "target": {"nodeName": "Eat"}},
        {"action": "add_flow", "target": {}, "changes": {"source": "Order Pizza", "target": "Eat"}},
    ])
    assert _added_owners(mods) == {"Pay": None, "Eat": None}


def test_bpmn_modifications_always_carry_a_changes_object(monkeypatch):
    """Both editors' BPMN modifiers read m.changes.<field> unguarded; an add with
    no changes went out as changes: null and threw, e.g. reading 'gatewayType'."""
    flat = {"elements": {"t-check": {"type": "BPMNTask", "name": "Check", "owner": None}}, "relationships": {}}
    mods = _bpmn_modify(monkeypatch, flat, [
        {"action": "add_task", "target": {"nodeName": "Ship"}},
        {"action": "add_gateway", "target": {"nodeName": "Paid?"}},
        {"action": "add_event", "target": {"nodeName": "Done"}},
        {"action": "remove_element", "target": {"nodeName": "Check"}},
    ])
    assert [m["action"] for m in mods] == ["add_task", "add_gateway", "add_event", "remove_element"]
    assert all(isinstance(m["changes"], dict) for m in mods), [m["changes"] for m in mods]


# -- A flow that cannot be a message flow stays inside one pool -------------
# Both editors turn a flow between pools into a message flow, which a gateway
# cannot carry (nor a start event send or an end event receive); the backend
# drops it. Live two-pool shape: join gateway -> Merge task was lost (7 of 8).

def _pools_of(spec):
    return {n["id"]: n["poolId"] for n in spec["nodes"]}


def test_bpmn_pools_split_by_a_gateway_flow_merge_and_keep_their_lanes():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine(_pr_review_spec())
    assert [p["id"] for p in spec["pools"]] == ["reviewerAgents"]
    assert [lane["id"] for lane in spec["pools"][0]["lanes"]] == ["reviewer", "merger"]
    assert set(_pools_of(spec).values()) == {"reviewerAgents"}
    _assert_every_node_placed(spec)
    nodes = {n["id"]: n for n in spec["nodes"]}
    assert (nodes["merge"]["laneId"], nodes["join"]["laneId"]) == ("merger", "reviewer")


def _collaboration(extra_nodes=(), extra_flows=()):
    def node(nid, ntype, pool):
        return {"id": nid, "name": nid, "type": ntype, "poolId": pool}

    return {
        "pools": [{"id": "customer", "name": "Customer", "lanes": []},
                  {"id": "vendor", "name": "Vendor", "lanes": [{"id": "clerk", "name": "Clerk"}]}],
        "nodes": [node("hungry", "startEvent", "customer"), node("order", "task", "customer"),
                  node("sent", "endEvent", "customer"),
                  node("received", "startEvent", "vendor"), node("bake", "task", "vendor"),
                  node("done", "endEvent", "vendor"), *extra_nodes],
        "flows": [{"source": "hungry", "target": "order"}, {"source": "order", "target": "sent"},
                  {"source": "received", "target": "bake"}, {"source": "bake", "target": "done"},
                  # real collaboration: task -> task and end event -> start event
                  {"source": "order", "target": "bake"}, {"source": "sent", "target": "received"},
                  *extra_flows],
    }


def test_bpmn_message_flows_between_tasks_and_events_keep_both_pools():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine(_collaboration())
    assert [p["id"] for p in spec["pools"]] == ["customer", "vendor"]
    assert _pools_of(spec) == {"hungry": "customer", "order": "customer", "sent": "customer",
                               "received": "vendor", "bake": "vendor", "done": "vendor"}


def test_bpmn_stray_gateway_moves_to_the_pool_of_all_its_neighbours():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine(_collaboration(
        extra_nodes=[{"id": "ok", "name": "Dough ok?", "type": "gateway", "poolId": "customer"}],
        extra_flows=[{"source": "bake", "target": "ok"}, {"source": "ok", "target": "done", "name": "yes"},
                     {"source": "ok", "target": "bake", "name": "no"}],
    ))
    assert [p["id"] for p in spec["pools"]] == ["customer", "vendor"]
    ok = next(n for n in spec["nodes"] if n["id"] == "ok")
    assert (ok["poolId"], ok["laneId"], ok["owner"]) == ("vendor", "clerk", "clerk")


def test_bpmn_schema_data_objects_default_empty():
    from schemas import SystemBPMNSpec
    d = SystemBPMNSpec(nodes=[{"id": "t", "name": "Do", "type": "task"}]).model_dump()
    assert d["dataObjects"] == []
    assert d["dataAssociations"] == []


def test_bpmn_schema_data_objects_round_trip():
    from schemas import SystemBPMNSpec
    d = SystemBPMNSpec.model_validate(_battery_spec()).model_dump()
    assert [o["name"] for o in d["dataObjects"]] == ["Binder", "Solvent", "Slurry"]
    assert d["dataAssociations"][2] == {"dataObject": "slurry", "node": "mixing", "direction": "output"}


def test_bpmn_validation_keeps_data_objects_out_of_the_sequence_flow():
    """The repair pass must not wire data objects into the process."""
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    spec = BPMNDiagramHandler(None)._validate_and_refine(_battery_spec())

    assert len(spec["dataObjects"]) == 3
    assert len(spec["dataAssociations"]) == 4
    data_ids = {o["id"] for o in spec["dataObjects"]}
    assert not any(f["source"] in data_ids or f["target"] in data_ids for f in spec["flows"])
    assert len(spec["flows"]) == 3
    assert not any(n["id"] in data_ids for n in spec["nodes"])


def test_bpmn_validation_drops_dangling_and_duplicate_data_associations():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    raw = _battery_spec()
    raw["dataAssociations"] += [
        {"dataObject": "ghost", "node": "mixing", "direction": "input"},
        {"dataObject": "binder", "node": "no_such_task", "direction": "input"},
        {"dataObject": "binder", "node": "mixing", "direction": "input"},
    ]
    spec = BPMNDiagramHandler(None)._validate_and_refine(raw)
    assert len(spec["dataAssociations"]) == 4


def test_bpmn_validation_renames_data_object_colliding_with_node_id():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    raw = _battery_spec()
    raw["dataObjects"].append({"id": "coating", "name": "Coating Recipe"})
    raw["dataAssociations"].append({"dataObject": "coating", "node": "coating", "direction": "input"})
    spec = BPMNDiagramHandler(None)._validate_and_refine(raw)

    recipe = next(o for o in spec["dataObjects"] if o["name"] == "Coating Recipe")
    assert recipe["id"] != "coating"
    assert {"dataObject": recipe["id"], "node": "coating", "direction": "input"} in spec["dataAssociations"]


def test_bpmn_system_message_mentions_data_objects():
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    h = BPMNDiagramHandler(None)
    assert "3 data object(s)" in h._build_system_message(h._validate_and_refine(_battery_spec()))


def test_bpmn_modification_adds_data_object_and_connects_it(monkeypatch):
    """The follow-up from the issue ("add the missing data objects") must reach
    the editor as add_data_object + add_flow, not be rejected by the guardrail."""
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas.bpmn import BPMNModification, BPMNModificationResponse, BPMNModificationTarget

    model = {"elements": {"task-mixing": {"type": "BPMNTask", "name": "Mixing"}}}
    response = BPMNModificationResponse(
        elementFound=True,
        modifications=[
            BPMNModification(action="add_data_object", target=BPMNModificationTarget(nodeName="Binder")),
            BPMNModification(
                action="add_flow", target=BPMNModificationTarget(),
                changes={"source": "Binder", "target": "task-mixing"},
            ),
        ],
        message="Added the Binder input to Mixing.",
    )

    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", lambda self, *a, **k: response)
    result = BPMNDiagramHandler(None).generate_modification(
        "add binder as an input data object of Mixing", current_model=model,
    )

    assert result["action"] == "modify_model"
    actions = [m["action"] for m in result["modifications"]]
    assert actions == ["add_data_object", "add_flow"]
