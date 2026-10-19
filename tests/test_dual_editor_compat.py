"""Outbound specs must work in both the production (v3) and the React Flow (v4) editor.

The agent ships before the v4 editor, so every spec it emits is applied by the
v3 modifiers (``features/assistant/services/modifiers`` on WME ``main``) first.
"""

from unittest.mock import patch


def _sm_model():
    """The traffic light the v3 repro used: an unnamed initial node points at Red."""
    def state(sid, name):
        return {"id": sid, "type": "State", "name": name, "owner": None}

    def transition(rid, src, dst, name=""):
        return {"id": rid, "type": "StateTransition", "name": name,
                "source": {"element": src}, "target": {"element": dst}}

    return {
        "type": "StateMachineDiagram",
        "elements": {
            "init": {"id": "init", "type": "StateInitialNode", "name": "", "owner": None},
            "red": state("red", "Red"),
            "green": state("green", "Green"),
            "amber": state("amber", "Amber"),
        },
        "relationships": {
            "t-init": transition("t-init", "init", "red"),
            "t-rg": transition("t-rg", "red", "green", "chronometer_finished"),
            "t-ga": transition("t-ga", "green", "amber", "chronometer_finished"),
            "t-ar": transition("t-ar", "amber", "red", "chronometer_finished"),
        },
    }


def _sm_modify(mods, model=None):
    from diagram_handlers.types.state_machine_handler import StateMachineHandler
    from schemas.state_machine import StateMachineModificationResponse

    parsed = StateMachineModificationResponse.model_validate({"modifications": mods})
    handler = StateMachineHandler(None)
    with patch.object(StateMachineHandler, "predict_structured", return_value=parsed):
        spec = handler.generate_modification("remove it", current_model=model or _sm_model())
    return spec, spec.get("modifications") or [spec.get("modification")]


class TestStateMachineTransitionRemoval:
    """v3 ``remove_element`` reads only ``stateName``: an endpoint-only target
    matched the unnamed initial pseudostate and deleted it (r9/prod repro)."""

    def test_endpoint_removal_becomes_remove_transition_by_id(self):
        spec, mods = _sm_modify([
            {"action": "remove_element", "target": {"sourceState": "Red", "targetState": "Green"}},
        ])
        assert mods == [{
            "action": "remove_transition",
            "target": {"sourceState": "Red", "targetState": "Green", "transitionId": "t-rg"},
            "changes": None,
        }]
        assert spec["message"] == "Removed transition **Red → Green**."

    def test_no_state_removal_without_a_state_name_is_emitted(self):
        _, mods = _sm_modify([
            {"action": "add_state", "target": {"stateName": "Blink"}, "changes": {"stateType": "regular"}},
            {"action": "remove_element", "target": {"sourceState": "(initial)", "targetState": "Red"}},
            {"action": "remove_element", "target": {"sourceState": "Amber"}},
        ])
        assert [m["action"] for m in mods] == ["add_state", "remove_transition"]
        assert mods[1]["target"]["transitionId"] == "t-init"
        assert not any(m["action"] == "remove_element" and not m["target"].get("stateName") for m in mods)

    def test_state_removal_is_untouched(self):
        _, mods = _sm_modify([{"action": "remove_element", "target": {"stateName": "Amber"}}])
        assert mods == [{"action": "remove_element", "target": {
            "stateName": "Amber", "sourceState": None, "targetState": None}, "changes": None}]

    def test_unresolved_transition_never_falls_back_to_remove_element(self):
        _, mods = _sm_modify([
            {"action": "remove_element", "target": {"sourceState": "Red", "targetState": "Amber"}},
        ])
        assert mods == [{"action": "remove_transition",
                         "target": {"sourceState": "Red", "targetState": "Amber"}, "changes": None}]
