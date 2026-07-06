"""Tests for the same-batch orphaned-state bug in AgentDiagram modifications.

The bug: on an existing state-machine agent, a modify like "add a
BMWIntegrationState and relate it to the flow" added the new state but silently
dropped the connecting transition (it was validated against the OLD model only,
so its endpoint looked non-existent), leaving the state orphaned while the reply
still claimed "Applied N changes".

Covered here:
- FIX 1: ``_validate_modifications`` KEEPS a transition whose endpoint is a state
  added in the SAME batch, and still DROPS a transition to a genuinely absent
  state; kept ops are reordered so add_state/add_intent precede transitions.
- FIX 2: the orphan backstop appends an honest "isn't connected yet" note when a
  new state has no connecting transition, and stays silent when one exists.
- FIX 3: modification messages render intent names / transition endpoints instead
  of the literal "element".
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagram_handlers.types.agent_diagram_handler import AgentDiagramHandler


# ---------------------------------------------------------------------------
# Fixtures (mirror test_agent_modification_safety.py)
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, response="{}"):
        self.responses = [response]
        self._i = 0
        self.name = "fake-model"

    def predict(self, prompt):
        r = self.responses[self._i % len(self.responses)]
        self._i += 1
        return r


def _handler(response="{}"):
    return AgentDiagramHandler(_FakeLLM(response))


def _agent_model_with(states=None, intents=None, rag=None):
    elements = {}
    i = 0
    for name in states or []:
        i += 1
        elements[f"s{i}"] = {"id": f"s{i}", "name": name, "type": "AgentState"}
    for name in intents or []:
        i += 1
        elements[f"i{i}"] = {"id": f"i{i}", "name": name, "type": "AgentIntent"}
    for name in rag or []:
        i += 1
        elements[f"r{i}"] = {"id": f"r{i}", "name": name, "type": "AgentRagElement"}
    return {"type": "AgentDiagram", "elements": elements, "relationships": {}}


def _mods_json(modifications):
    return json.dumps({"modifications": modifications})


# ---------------------------------------------------------------------------
# FIX 1 — same-batch transition endpoints survive validation
# ---------------------------------------------------------------------------

class TestSameBatchTransition:
    def setup_method(self):
        self.h = _handler()
        self.model = _agent_model_with(states=["welcome"])

    def test_transition_into_same_batch_added_state_is_kept(self):
        """The exact bug: add a state + a transition targeting it in one batch."""
        mods = [
            {"action": "add_state",
             "target": {"stateName": "BMWIntegrationState"},
             "changes": {"replies": [{"text": "hi", "replyType": "text"}]}},
            {"action": "add_transition",
             "target": {"sourceStateName": "welcome",
                        "targetStateName": "BMWIntegrationState"},
             "changes": {"condition": "auto"}},
        ]
        res = self.h._validate_modifications(mods, self.model)
        actions = [m["action"] for m in res["kept"]]
        assert actions.count("add_state") == 1
        assert actions.count("add_transition") == 1
        assert res["skipped"] == []

    def test_transition_out_of_same_batch_added_state_is_kept(self):
        """Source (not just target) may be the just-added state."""
        mods = [
            {"action": "add_state",
             "target": {"stateName": "BMWIntegrationState"}, "changes": {}},
            {"action": "add_transition",
             "target": {"sourceStateName": "BMWIntegrationState",
                        "targetStateName": "welcome"},
             "changes": {"condition": "auto"}},
        ]
        res = self.h._validate_modifications(mods, self.model)
        assert len([m for m in res["kept"] if m["action"] == "add_transition"]) == 1

    def test_transition_into_same_batch_added_intent_is_kept(self):
        """A transition triggered by a just-added intent must also survive."""
        mods = [
            {"action": "add_intent",
             "target": {"intentName": "OrderPizza"},
             "changes": {"trainingPhrases": ["pizza please"]}},
            {"action": "add_state",
             "target": {"stateName": "orderState"}, "changes": {}},
            {"action": "add_transition",
             "target": {"sourceStateName": "welcome",
                        "targetStateName": "orderState"},
             "changes": {"condition": "when_intent_matched",
                         "intentName": "OrderPizza"}},
        ]
        res = self.h._validate_modifications(mods, self.model)
        assert len([m for m in res["kept"] if m["action"] == "add_transition"]) == 1

    def test_transition_to_nonexistent_state_still_dropped(self):
        """pending must not over-rescue: a truly unknown endpoint is still dropped."""
        mods = [
            {"action": "add_state",
             "target": {"stateName": "BMWIntegrationState"}, "changes": {}},
            {"action": "add_transition",
             "target": {"sourceStateName": "welcome", "targetStateName": "ghost"},
             "changes": {"condition": "auto"}},
        ]
        res = self.h._validate_modifications(mods, self.model)
        assert len([m for m in res["kept"] if m["action"] == "add_transition"]) == 0
        assert any(s.get("reason") == "missing_target" for s in res["skipped"])

    def test_duplicate_add_state_does_not_join_pending(self):
        """An add_state duplicating an existing state is 'exists', but a
        transition to that (existing) state is still valid via ``states``."""
        model = _agent_model_with(states=["welcome"])
        mods = [
            {"action": "add_state",
             "target": {"stateName": "welcome"}, "changes": {}},   # dup → skipped
            {"action": "add_transition",
             "target": {"sourceStateName": "welcome", "targetStateName": "welcome"},
             "changes": {"condition": "auto"}},
        ]
        res = self.h._validate_modifications(mods, model)
        # add_state dropped as 'exists'; transition kept (welcome exists).
        assert any(s.get("reason") == "exists" for s in res["skipped"])
        assert len([m for m in res["kept"] if m["action"] == "add_transition"]) == 1


# ---------------------------------------------------------------------------
# FIX 1 — stable reorder (adds before transitions)
# ---------------------------------------------------------------------------

class TestKeptOrdering:
    def setup_method(self):
        self.h = _handler()

    def test_add_state_precedes_transition_even_when_llm_reverses_order(self):
        model = _agent_model_with(states=["welcome"])
        mods = [
            {"action": "add_transition",
             "target": {"sourceStateName": "welcome", "targetStateName": "newState"},
             "changes": {"condition": "auto"}},
            {"action": "add_state",
             "target": {"stateName": "newState"}, "changes": {}},
        ]
        res = self.h._validate_modifications(mods, model)
        actions = [m["action"] for m in res["kept"]]
        assert "add_state" in actions and "add_transition" in actions
        assert actions.index("add_state") < actions.index("add_transition")

    def test_add_intent_precedes_transition(self):
        model = _agent_model_with(states=["welcome", "support"])
        mods = [
            {"action": "add_transition",
             "target": {"sourceStateName": "welcome", "targetStateName": "support"},
             "changes": {"condition": "when_intent_matched", "intentName": "Help"}},
            {"action": "add_intent",
             "target": {"intentName": "Help"},
             "changes": {"trainingPhrases": ["help me"]}},
        ]
        res = self.h._validate_modifications(mods, model)
        actions = [m["action"] for m in res["kept"]]
        assert actions.index("add_intent") < actions.index("add_transition")


# ---------------------------------------------------------------------------
# FIX 2 — orphan backstop helpers (pure)
# ---------------------------------------------------------------------------

class TestOrphanDetection:
    def setup_method(self):
        self.h = _handler()

    def test_added_state_without_transition_is_orphan(self):
        kept = [
            {"action": "add_state", "target": {"stateName": "BMWIntegrationState"},
             "changes": {}},
        ]
        assert self.h._find_orphan_added_states(kept) == ["BMWIntegrationState"]

    def test_added_state_with_transition_is_not_orphan(self):
        kept = [
            {"action": "add_state", "target": {"stateName": "BMWIntegrationState"},
             "changes": {}},
            {"action": "add_transition",
             "target": {"sourceStateName": "welcome",
                        "targetStateName": "BMWIntegrationState"},
             "changes": {}},
        ]
        assert self.h._find_orphan_added_states(kept) == []

    def test_note_names_the_state(self):
        note = self.h._orphan_states_note(["BMWIntegrationState"])
        assert "BMWIntegrationState" in note
        assert "isn't connected" in note

    def test_note_empty_when_no_orphans(self):
        assert self.h._orphan_states_note([]) == ""


# ---------------------------------------------------------------------------
# FIX 2 — orphan backstop end-to-end (generate_modification)
# ---------------------------------------------------------------------------

class TestOrphanBackstopEndToEnd:
    def test_add_state_without_transition_appends_note(self):
        h = _handler(_mods_json([
            {"action": "add_state",
             "target": {"stateName": "BMWIntegrationState"},
             "changes": {"replies": [{"text": "hi", "replyType": "text"}]}},
        ]))
        model = _agent_model_with(states=["welcome"])
        result = h.generate_modification(
            "add a BMWIntegrationState", model,
            raw_request="add a BMWIntegrationState",
        )
        assert result["action"] == "modify_model"
        assert "BMWIntegrationState" in result["message"]
        assert "isn't connected" in result["message"]

    def test_add_state_with_transition_has_no_note(self):
        h = _handler(_mods_json([
            {"action": "add_state",
             "target": {"stateName": "BMWIntegrationState"},
             "changes": {"replies": [{"text": "hi", "replyType": "text"}]}},
            {"action": "add_transition",
             "target": {"sourceStateName": "welcome",
                        "targetStateName": "BMWIntegrationState"},
             "changes": {"condition": "auto"}},
        ]))
        model = _agent_model_with(states=["welcome"])
        result = h.generate_modification(
            "add a BMWIntegrationState and connect welcome to it", model,
            raw_request="add a BMWIntegrationState and connect welcome to it",
        )
        assert result["action"] == "modify_model"
        assert "isn't connected" not in result["message"]


# ---------------------------------------------------------------------------
# FIX 3 — friendly modification messages (no more literal "element")
# ---------------------------------------------------------------------------

class TestModificationMessages:
    def setup_method(self):
        self.h = _handler()

    def test_add_intent_message_uses_intent_name(self):
        name = self.h._build_mod_target_name(
            "add_intent", {"intentName": "OrderPizza"},
            {"action": "add_intent", "target": {"intentName": "OrderPizza"}},
        )
        assert name == "OrderPizza"
        msg = self.h._friendly_mod_message("add_intent", name)
        assert "OrderPizza" in msg
        assert "element" not in msg.lower()

    def test_add_transition_message_renders_endpoints(self):
        target = {"sourceStateName": "greeting", "targetStateName": "support"}
        name = self.h._build_mod_target_name(
            "add_transition", target,
            {"action": "add_transition", "target": target},
        )
        assert "greeting" in name and "support" in name
        msg = self.h._friendly_mod_message("add_transition", name)
        assert "greeting" in msg and "support" in msg
        assert "element" not in msg.lower()

    def test_add_state_body_falls_back_to_changes_name(self):
        # No stateName in target — resolve the display name from changes.name.
        name = self.h._build_mod_target_name(
            "add_state_body", {},
            {"action": "add_state_body", "target": {},
             "changes": {"name": "welcomeState"}},
        )
        assert name == "welcomeState"

    def test_batch_message_has_no_element_placeholder(self):
        mods = [
            {"action": "add_intent", "target": {"intentName": "OrderPizza"},
             "changes": {"trainingPhrases": ["pizza"]}},
            {"action": "add_transition",
             "target": {"sourceStateName": "welcome", "targetStateName": "orderState"},
             "changes": {}},
        ]
        msg = self.h._friendly_batch_message(mods)
        assert "element" not in msg.lower()
        assert "OrderPizza" in msg
        assert "welcome" in msg and "orderState" in msg
