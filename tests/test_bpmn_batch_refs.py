def test_bpmn_guardrail_keeps_flows_to_task_added_earlier_in_same_batch(monkeypatch):
    import sys
    from pathlib import Path

    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas.bpmn import (
        BPMNModification,
        BPMNModificationChanges,
        BPMNModificationResponse,
        BPMNModificationTarget,
    )

    model = {
        "elements": {
            "start-01": {"type": "BPMNStartEvent", "name": ""},
            "task-01": {"type": "BPMNTask", "name": "Showcase BPMN editor", "taskType": "user"},
            "end-01": {"type": "BPMNEndEvent", "name": ""},
        }
    }

    ok_response = BPMNModificationResponse(
        elementFound=True,
        modifications=[
            BPMNModification(
                action="remove_flow",
                target=BPMNModificationTarget(nodeName=None),
                changes=BPMNModificationChanges(source="Showcase BPMN editor", target="End Event"),
            ),
            BPMNModification(
                action="add_task",
                target=BPMNModificationTarget(nodeName="Record video demo"),
                changes=BPMNModificationChanges(name="Record video demo", taskType="user"),
            ),
            BPMNModification(
                action="add_flow",
                target=BPMNModificationTarget(nodeName=None),
                changes=BPMNModificationChanges(source="Showcase BPMN editor", target="Record video demo"),
            ),
            BPMNModification(
                action="add_flow",
                target=BPMNModificationTarget(nodeName=None),
                changes=BPMNModificationChanges(source="Record video demo", target="End Event"),
            ),
        ],
        message="Added task and rewired flow.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", fake_predict)
    h = BPMNDiagramHandler(None)
    result = h.generate_modification("add a task for Record video demo", current_model=model)

    assert result["action"] == "modify_model"
    assert len(result["modifications"]) == 4
    assert [m["action"] for m in result["modifications"]] == [
        "remove_flow",
        "add_task",
        "add_flow",
        "add_flow",
    ]


def test_bpmn_normalizes_embedded_task_connections_and_synthetic_ids(monkeypatch):
    import sys
    from pathlib import Path

    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas.bpmn import (
        BPMNModification,
        BPMNModificationChanges,
        BPMNModificationResponse,
        BPMNModificationTarget,
    )

    model = {
        "elements": {
            "start-01": {"type": "BPMNStartEvent", "name": ""},
            "task-01": {"type": "BPMNTask", "name": "Showcase BPMN editor", "taskType": "user"},
            "end-01": {"type": "BPMNEndEvent", "name": ""},
        }
    }

    ok_response = BPMNModificationResponse(
        elementFound=True,
        modifications=[
            BPMNModification(
                action="add_task",
                target=BPMNModificationTarget(
                    nodeId="new-task-record-video-demo-1",
                    nodeName="Record video demo 1 (Task)",
                ),
                changes=BPMNModificationChanges(
                    name="Record video demo 1",
                    taskType="user",
                    source="task-01",
                    target="new-task-record-video-demo-1",
                ),
            ),
            BPMNModification(
                action="add_task",
                target=BPMNModificationTarget(
                    nodeId="new-task-record-video-demo-2",
                    nodeName="Record video demo 2 (Task)",
                ),
                changes=BPMNModificationChanges(
                    name="Record video demo 2",
                    taskType="user",
                    source="new-task-record-video-demo-1",
                    target="new-task-record-video-demo-2",
                ),
            ),
            BPMNModification(
                action="remove_flow",
                target=BPMNModificationTarget(nodeName=None),
                changes=BPMNModificationChanges(source="task-01", target="end-01"),
            ),
            BPMNModification(
                action="add_flow",
                target=BPMNModificationTarget(nodeName=None),
                changes=BPMNModificationChanges(
                    source="new-task-record-video-demo-2",
                    target="end-01",
                ),
            ),
        ],
        message="Added two sequential tasks after Showcase BPMN editor.",
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return ok_response

    monkeypatch.setattr(BPMNDiagramHandler, "predict_structured", fake_predict)
    h = BPMNDiagramHandler(None)
    result = h.generate_modification("Add a second task for Record video demo", current_model=model)

    assert result["action"] == "modify_model"
    actions = [m["action"] for m in result["modifications"]]
    assert actions == [
        "add_task",
        "add_flow",
        "add_task",
        "add_flow",
        "remove_flow",
        "add_flow",
    ]

    first_task = result["modifications"][0]
    second_task = result["modifications"][2]
    assert first_task["target"]["nodeName"] == "Record video demo 1"
    assert second_task["target"]["nodeName"] == "Record video demo 2"
    assert result["modifications"][1]["changes"]["target"] == "Record video demo 1"
    assert result["modifications"][3]["changes"]["source"] == "Record video demo 1"
    assert result["modifications"][3]["changes"]["target"] == "Record video demo 2"
    assert result["modifications"][5]["changes"]["source"] == "Record video demo 2"
