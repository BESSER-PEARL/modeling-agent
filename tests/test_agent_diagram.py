r"""
Tests for modeling-agent#8: replyType="code" agent replies must always be a
complete `def name(session):` function, matching what BESSER's
agent_model_builder.py extracts via re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', ...)
and writes verbatim into the generated agent's source file. Code without a
"def" produces a NameError in the generated agent (the callable name the
builder falls back to was never defined).
"""


def test_ensure_code_reply_already_a_function_is_unchanged():
    from diagram_handlers.types.agent_diagram_handler import _ensure_code_reply_is_function
    text = "def log_message(session):\n    print(session.event.message)"
    assert _ensure_code_reply_is_function(text, "logState_reply_0") == text


def test_ensure_code_reply_wraps_bare_statements():
    from diagram_handlers.types.agent_diagram_handler import _ensure_code_reply_is_function
    wrapped = _ensure_code_reply_is_function("print(session.event.message)", "logState_reply_0")
    assert wrapped.startswith("def logstate_reply_0(session):\n")
    assert "    print(session.event.message)" in wrapped
    # The wrapped result must itself satisfy the downstream extraction regex.
    import re
    assert re.search(r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", wrapped)


def test_ensure_code_reply_wraps_multiline_bare_statements_preserving_blank_lines():
    from diagram_handlers.types.agent_diagram_handler import _ensure_code_reply_is_function
    wrapped = _ensure_code_reply_is_function(
        "x = 1\n\nprint(x)", "myState_reply_1",
    )
    lines = wrapped.splitlines()
    assert lines[0] == "def mystate_reply_1(session):"
    assert lines[1] == "    x = 1"
    assert lines[2] == ""  # blank lines are not indented
    assert lines[3] == "    print(x)"


def test_ensure_code_reply_empty_text_returns_unchanged():
    from diagram_handlers.types.agent_diagram_handler import _ensure_code_reply_is_function
    assert _ensure_code_reply_is_function("", "hint") == ""


def test_ensure_code_reply_sanitizes_name_hint():
    from diagram_handlers.types.agent_diagram_handler import _ensure_code_reply_is_function
    wrapped = _ensure_code_reply_is_function("do_thing()", "My State! #1")
    assert wrapped.startswith("def my_state_1(session):\n")


def test_normalize_state_spec_wraps_code_replies():
    from diagram_handlers.types.agent_diagram_handler import AgentDiagramHandler
    handler = AgentDiagramHandler(None)
    spec = handler._normalize_state_spec(
        {
            "stateName": "logState",
            "replies": [{"text": "print(session.event.message)", "replyType": "code"}],
        },
        "add a function that logs the message",
    )
    reply = spec["replies"][0]
    assert reply["replyType"] == "code"
    assert reply["text"].startswith("def ")
    assert "print(session.event.message)" in reply["text"]


def test_normalize_state_spec_leaves_text_replies_unwrapped():
    from diagram_handlers.types.agent_diagram_handler import AgentDiagramHandler
    handler = AgentDiagramHandler(None)
    spec = handler._normalize_state_spec(
        {"stateName": "greet", "replies": [{"text": "Hello!", "replyType": "text"}]},
        "add a greeting",
    )
    assert spec["replies"][0]["text"] == "Hello!"


def test_fix_code_replies_in_modifications_wraps_add_state_body():
    from diagram_handlers.types.agent_diagram_handler import AgentDiagramHandler
    mods = [
        {
            "action": "add_state_body",
            "target": {"stateName": "logState"},
            "changes": {"text": "print(session.event.message)", "replyType": "code"},
        }
    ]
    fixed = AgentDiagramHandler._fix_code_replies_in_modifications(mods)
    assert fixed[0]["changes"]["text"].startswith("def ")


def test_fix_code_replies_in_modifications_wraps_add_state_replies_list():
    from diagram_handlers.types.agent_diagram_handler import AgentDiagramHandler
    mods = [
        {
            "action": "add_state",
            "target": {"stateName": "welcomeState"},
            "changes": {
                "replies": [
                    {"text": "Welcome!", "replyType": "text"},
                    {"text": "x = compute()", "replyType": "code"},
                ]
            },
        }
    ]
    fixed = AgentDiagramHandler._fix_code_replies_in_modifications(mods)
    replies = fixed[0]["changes"]["replies"]
    assert replies[0]["text"] == "Welcome!"  # text reply untouched
    assert replies[1]["text"].startswith("def ")  # code reply wrapped


def test_fix_code_replies_in_modifications_leaves_non_code_untouched():
    from diagram_handlers.types.agent_diagram_handler import AgentDiagramHandler
    mods = [
        {
            "action": "modify_state",
            "target": {"stateName": "greet"},
            "changes": {"name": "welcomeState"},
        }
    ]
    fixed = AgentDiagramHandler._fix_code_replies_in_modifications(mods)
    assert fixed == mods


# ---------------------------------------------------------------------------
# Agent components, extended reply fields and layout (PR #19)
# ---------------------------------------------------------------------------

def _handler():
    from diagram_handlers.types.agent_diagram_handler import AgentDiagramHandler
    return AgentDiagramHandler(None)


def test_normalize_reply_list_passes_through_type_specific_fields():
    replies = _handler()._normalize_reply_list(
        [
            {"text": "Answer from docs", "replyType": "rag", "ragDatabaseName": "faq", "llm_name": "gpt"},
            {"text": "Show form", "replyType": "gui_reply", "guiId": "orderForm", "system_message": None},
            {"text": "Pick one", "replyType": "ws_options", "ws_options": "A\nB"},
        ],
        default_text="",
    )
    assert replies[0] == {"text": "Answer from docs", "replyType": "rag", "ragDatabaseName": "faq", "llm_name": "gpt"}
    # None-valued fields are dropped, set ones kept
    assert replies[1] == {"text": "Show form", "replyType": "gui_reply", "guiId": "orderForm"}
    assert replies[2]["ws_options"] == "A\nB"


def test_normalize_intent_keeps_intent_description():
    h = _handler()
    intent = h._normalize_intent_spec(
        {"intentName": "Greet", "intentDescription": "User says hi", "trainingPhrases": ["hi"]}, "")
    assert intent["intentDescription"] == "User says hi"
    no_desc = h._normalize_intent_spec({"intentName": "Greet", "trainingPhrases": ["hi"]}, "")
    assert "intentDescription" not in no_desc


def test_normalize_system_spec_keeps_named_components():
    spec = _handler()._normalize_system_spec(
        {
            "systemName": "Shop",
            "states": [{"stateName": "welcome", "replies": [{"text": "Hi"}]}],
            "llms": [{"name": "gpt", "provider": "openai"}, {"name": ""}],
            "ragElements": [{"name": "faq"}],
            "tools": [{"name": "search", "code": "def search(q): ..."}],
            "skills": [{"name": "tone", "content": "polite"}],
            "workspaces": [{"name": "ws", "path": "/data"}],
            "guis": [{"gui_id": "orderForm"}, {"gui_id": ""}],
        },
        "build a shop agent",
    )
    assert [c["name"] for c in spec["llms"]] == ["gpt"]  # unnamed dropped
    assert spec["ragElements"][0]["name"] == "faq"
    assert spec["tools"][0]["name"] == "search"
    assert spec["skills"][0]["name"] == "tone"
    assert spec["workspaces"][0]["path"] == "/data"
    assert [g["gui_id"] for g in spec["guis"]] == ["orderForm"]


def test_modify_prompt_documents_new_component_actions():
    from diagram_handlers.types.agent_diagram_handler import MODIFY_SYSTEM_PROMPT_AGENT
    for action in ("add_llm", "add_tool", "add_skill", "add_workspace", "add_gui", "add_rag_element"):
        assert f"- {action}:" in MODIFY_SYSTEM_PROMPT_AGENT


def test_prompts_list_every_reply_type_from_schema():
    from typing import get_args
    from schemas.agent_diagram import ReplyType
    from diagram_handlers.types.agent_diagram_handler import MODIFY_SYSTEM_PROMPT_AGENT
    single_prompt = _handler().get_system_prompt()
    for v in get_args(ReplyType):
        assert f'"{v}"' in MODIFY_SYSTEM_PROMPT_AGENT
        assert f'"{v}"' in single_prompt


def _system_spec():
    return {
        "systemName": "Bot",
        "hasInitialNode": True,
        "intents": [
            {"type": "intent", "intentName": "Greet", "trainingPhrases": ["hi"]},
            {"type": "intent", "intentName": "Bye", "trainingPhrases": ["bye"]},
        ],
        "states": [
            {"type": "state", "stateName": "welcome", "replies": [{"text": "Hi", "replyType": "text"}]},
            {"type": "state", "stateName": "farewell", "replies": [{"text": "Bye", "replyType": "text"}]},
        ],
        "transitions": [
            {"source": "initial", "target": "welcome", "condition": "auto", "conditionValue": ""},
            {"source": "welcome", "target": "farewell", "condition": "when_intent_matched", "conditionValue": "Bye"},
        ],
    }


def test_layout_new_format_gives_intents_no_position():
    from diagram_handlers.core.layout_engine import layout_agent_system
    spec = _system_spec()
    spec["intents"][0]["position"] = {"x": 5, "y": 5}  # hallucinated — must be stripped
    new_format_model = {"elements": {}, "relationships": {}, "components": {}}
    layout_agent_system(spec, new_format_model)
    assert all("position" not in i for i in spec["intents"])
    initial_y = spec["initialNode"]["position"]["y"]
    state_ys = [s["position"]["y"] for s in spec["states"]]
    # States start right below the initial node — no reserved intent band.
    assert min(state_ys) - initial_y < 250


def test_layout_without_existing_model_is_new_format():
    from diagram_handlers.core.layout_engine import layout_agent_system
    spec = _system_spec()
    layout_agent_system(spec, None)
    assert all("position" not in i for i in spec["intents"])
    assert all("position" in s for s in spec["states"])


def test_layout_old_format_still_places_intents_above_states():
    from diagram_handlers.core.layout_engine import layout_agent_system
    old_model = {
        "elements": {
            "i0": {"type": "AgentIntent", "name": "Existing", "owner": None,
                   "bounds": {"x": -2000, "y": -2000, "width": 160, "height": 100}},
        },
        "relationships": {},
    }
    spec = _system_spec()
    layout_agent_system(spec, old_model)
    assert all("position" in i for i in spec["intents"])
    max_intent_y = max(i["position"]["y"] for i in spec["intents"])
    assert min(s["position"]["y"] for s in spec["states"]) > max_intent_y


def test_layout_single_intent_new_vs_old_format():
    from diagram_handlers.core.layout_engine import layout_agent_single
    new_spec = layout_agent_single({"type": "intent", "intentName": "Greet", "position": {"x": 1, "y": 1}},
                                   {"elements": {}, "components": {}})
    assert "position" not in new_spec
    old_spec = layout_agent_single(
        {"type": "intent", "intentName": "Greet"},
        {"elements": {"i0": {"type": "AgentIntent", "name": "X", "bounds": {"x": 0, "y": 0, "width": 10, "height": 10}}}},
    )
    assert "position" in old_spec
    state_spec = layout_agent_single({"type": "state", "stateName": "s"}, None)
    assert "position" in state_spec
