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
