"""Safety tests for AgentDiagram modifications.

Covers:
- DATA LOSS: a modify request must never delete/empty the diagram; a
  failed or empty modification must leave the model unchanged.
- no hallucinated removals/adds; stray-word input must ask for
  clarification rather than mutate the model.
- add_transition referencing a missing element must not be emitted.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagram_handlers.types.agent_diagram_handler import (
    AgentDiagramHandler,
    _EmptyModificationError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeLLM:
    """LLM stub.  ``predict`` returns a canned JSON string.

    The handler's ``predict_structured`` falls back to ``predict_with_retry``
    (which calls ``predict``) + schema validation when no OpenAI ``.parse()``
    client is present, so returning schema-shaped JSON drives the path.
    """

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
    """Build a minimal agent diagram model dict (Apollon-ish shape)."""
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
# _looks_like_actionable_request  (ambiguity guard)
# ---------------------------------------------------------------------------

class TestActionableHeuristic:
    @pytest.mark.parametrize("text", ["ggg", "gggIntent", "dfdf", "", "   ", "xyz abc"])
    def test_stray_words_not_actionable(self, text):
        assert AgentDiagramHandler._looks_like_actionable_request(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "add an intent called Greeting",
            "remove the welcome state",
            "rename greeting to start",
            "add a transition from greeting to support",
            "create a new answerState between questionState and endSessionState",
        ],
    )
    def test_real_instructions_actionable(self, text):
        assert AgentDiagramHandler._looks_like_actionable_request(text) is True

    def test_none_not_actionable(self):
        assert AgentDiagramHandler._looks_like_actionable_request(None) is False


# ---------------------------------------------------------------------------
# _index_existing_agent_model
# ---------------------------------------------------------------------------

class TestIndexModel:
    def test_indexes_by_type(self):
        model = _agent_model_with(states=["welcome"], intents=["Greeting"], rag=["KB"])
        idx = AgentDiagramHandler._index_existing_agent_model(model)
        assert idx["states"] == {"welcome"}
        assert idx["intents"] == {"greeting"}
        assert idx["rag"] == {"kb"}

    def test_empty_on_missing_model(self):
        idx = AgentDiagramHandler._index_existing_agent_model({})
        assert idx == {"states": set(), "intents": set(), "rag": set(), "components": {}}


# ---------------------------------------------------------------------------
# _validate_modifications  (no hallucinated removals/edits/dupes)
# ---------------------------------------------------------------------------

class TestValidateModifications:
    def setup_method(self):
        self.h = _handler()
        self.model = _agent_model_with(states=["welcome"], intents=["Greeting"])

    def test_remove_missing_element_dropped(self):
        # "Removed dfdf" when dfdf never existed -> must be dropped
        mods = [{"action": "remove_element", "target": {"intentName": "dfdf"}, "changes": {}}]
        res = self.h._validate_modifications(mods, self.model)
        assert res["kept"] == []
        assert res["skipped"][0]["reason"] == "missing_target"

    def test_remove_existing_element_kept(self):
        mods = [{"action": "remove_element", "target": {"stateName": "welcome"}, "changes": {}}]
        res = self.h._validate_modifications(mods, self.model)
        assert len(res["kept"]) == 1

    def test_add_duplicate_intent_dropped(self):
        # gggIntent / Greeting already exists -> don't re-add
        mods = [{"action": "add_intent", "target": {"intentName": "Greeting"},
                 "changes": {"trainingPhrases": ["hi"]}}]
        res = self.h._validate_modifications(mods, self.model)
        assert res["kept"] == []
        assert res["skipped"][0]["reason"] == "exists"

    def test_add_new_intent_kept(self):
        mods = [{"action": "add_intent", "target": {"intentName": "OrderPizza"},
                 "changes": {"trainingPhrases": ["pizza please"]}}]
        res = self.h._validate_modifications(mods, self.model)
        assert len(res["kept"]) == 1

    def test_modify_missing_state_dropped(self):
        mods = [{"action": "modify_state", "target": {"stateName": "nope"},
                 "changes": {"name": "renamed"}}]
        res = self.h._validate_modifications(mods, self.model)
        assert res["kept"] == []

    def test_add_transition_missing_endpoint_dropped(self):
        # transition to an element that doesn't exist
        mods = [{"action": "add_transition",
                 "target": {"sourceStateName": "welcome", "targetStateName": "ghost"},
                 "changes": {}}]
        res = self.h._validate_modifications(mods, self.model)
        assert res["kept"] == []

    def test_add_transition_valid_kept(self):
        m = _agent_model_with(states=["welcome", "support"])
        mods = [{"action": "add_transition",
                 "target": {"sourceStateName": "welcome", "targetStateName": "support"},
                 "changes": {}}]
        res = self.h._validate_modifications(mods, m)
        assert len(res["kept"]) == 1

    def test_mixed_batch_keeps_only_valid(self):
        mods = [
            {"action": "add_intent", "target": {"intentName": "NewOne"},
             "changes": {"trainingPhrases": ["a"]}},          # keep
            {"action": "remove_element", "target": {"intentName": "ghost"},
             "changes": {}},                                   # drop
        ]
        res = self.h._validate_modifications(mods, self.model)
        assert len(res["kept"]) == 1
        assert res["kept"][0]["action"] == "add_intent"


# ---------------------------------------------------------------------------
# generate_modification  (end-to-end data-loss guards)
# ---------------------------------------------------------------------------

class TestGenerateModificationSafety:
    def test_ambiguous_input_returns_clarification_no_mutation(self):
        # stray word must NOT mutate; returns assistant_message
        h = _handler(_mods_json([
            {"action": "add_intent", "target": {"intentName": "gggIntent"},
             "changes": {"trainingPhrases": ["x"]}},
        ]))
        model = _agent_model_with(states=["welcome"])
        result = h.generate_modification("ggg", model, raw_request="ggg")
        assert result["action"] == "assistant_message"
        assert "modification" not in result and "modifications" not in result

    def test_all_mods_invalid_returns_clarification_not_empty_modify(self):
        # nothing applicable -> must NOT emit a modify_model payload.
        h = _handler(_mods_json([
            {"action": "remove_element", "target": {"intentName": "dfdf"}, "changes": {}},
        ]))
        model = _agent_model_with(states=["welcome"], intents=["Greeting"])
        result = h.generate_modification(
            "remove the dfdf intent", model, raw_request="remove the dfdf intent",
        )
        assert result["action"] == "assistant_message"
        # Existing diagram untouched: no modify payload at all.
        assert "modifications" not in result
        assert "modification" not in result

    def test_valid_modification_passes_through(self):
        h = _handler(_mods_json([
            {"action": "add_intent", "target": {"intentName": "OrderPizza"},
             "changes": {"trainingPhrases": ["pizza please", "i want pizza"]}},
        ]))
        model = _agent_model_with(states=["welcome"])
        result = h.generate_modification(
            "add an intent called OrderPizza", model,
            raw_request="add an intent called OrderPizza",
        )
        assert result["action"] == "modify_model"
        assert "modification" in result or "modifications" in result

    def test_fallback_modification_is_non_mutating(self):
        # The fallback path must never return a model-mutating payload.
        h = _handler()
        result = h.generate_fallback_modification("whatever")
        assert result["action"] == "assistant_message"
        assert "modification" not in result and "modifications" not in result


# ---------------------------------------------------------------------------
# _clarify_response shape
# ---------------------------------------------------------------------------

def test_clarify_response_shape():
    h = _handler()
    r = h._clarify_response("hello")
    assert r["action"] == "assistant_message"
    assert r["diagramType"] == "AgentDiagram"
    assert r["message"] == "hello"


# ---------------------------------------------------------------------------
# Merge of develop's agent-component format (components section, new add_*)
# ---------------------------------------------------------------------------

def _new_format_model():
    """Editor's new format: intents/components live in ``components``."""
    return {
        "type": "AgentDiagram",
        "elements": {"s1": {"id": "s1", "name": "welcome", "type": "AgentState"}},
        "relationships": {},
        "components": {
            "i1": {"id": "i1", "name": "Greeting", "type": "AgentIntent"},
            "l1": {"id": "l1", "name": "gpt4", "type": "AgentLLM"},
            "g1": {"id": "g1", "gui_id": "orderForm", "type": "AgentGUI"},
        },
    }


class TestComponentFormatValidation:
    def setup_method(self):
        self.h = _handler()
        self.model = _new_format_model()

    def test_intent_in_components_is_an_existing_target(self):
        mods = [{"action": "add_intent_training_phrase", "target": {"intentName": "Greeting"},
                 "changes": {"trainingPhrase": "hey"}}]
        res = self.h._validate_modifications(mods, self.model)
        assert len(res["kept"]) == 1

    def test_duplicate_intent_in_components_dropped(self):
        mods = [{"action": "add_intent", "target": {"intentName": "Greeting"},
                 "changes": {"trainingPhrases": ["hi"]}}]
        res = self.h._validate_modifications(mods, self.model)
        assert res["kept"] == []
        assert res["skipped"][0]["reason"] == "exists"

    @pytest.mark.parametrize("action", ["add_llm", "add_tool", "add_skill", "add_workspace"])
    def test_new_component_add_kept(self, action):
        mods = [{"action": action, "target": {"name": "fresh"}, "changes": {}}]
        res = self.h._validate_modifications(mods, self.model)
        assert len(res["kept"]) == 1

    def test_duplicate_llm_dropped(self):
        mods = [{"action": "add_llm", "target": {"name": "GPT4"}, "changes": {}}]
        res = self.h._validate_modifications(mods, self.model)
        assert res["kept"] == []
        assert res["skipped"][0]["reason"] == "exists"

    def test_duplicate_gui_by_gui_id_dropped(self):
        mods = [{"action": "add_gui", "target": {"name": "Order"}, "changes": {"gui_id": "orderForm"}}]
        res = self.h._validate_modifications(mods, self.model)
        assert res["kept"] == []


def test_modification_uses_shared_prompt_with_safety_rules():
    """generate_modification must send MODIFY_SYSTEM_PROMPT_AGENT (develop's
    component actions + reply types) and it must carry the safety rules."""
    from diagram_handlers.types.agent_diagram_handler import MODIFY_SYSTEM_PROMPT_AGENT

    seen = []

    class _RecordingLLM(_FakeLLM):
        def predict(self, prompt):
            seen.append(prompt)
            return super().predict(prompt)

    h = AgentDiagramHandler(_RecordingLLM(_mods_json([
        {"action": "add_llm", "target": {"name": "claude"}, "changes": {"provider": "anthropic"}},
    ])))
    result = h.generate_modification(
        "add an LLM called claude", _new_format_model(), raw_request="add an LLM called claude",
    )
    assert result["action"] == "modify_model"
    assert seen and "- add_llm:" in seen[0]
    for rule in ("NEVER remove or rename", "Only ADD elements", "ALSO emit an add_transition"):
        assert rule in MODIFY_SYSTEM_PROMPT_AGENT
