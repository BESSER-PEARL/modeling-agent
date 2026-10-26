"""A screenshot or mockup attached in the chat becomes a GUI (#643).

Before: every attached image went to the UML image converter ("Analyze this
image of a UML diagram"), the typed message was ignored, and the GUI
generator could not take images, so a page mockup came back as a class
diagram or an error. Now a UI image reaches the GUI generator's own
authoring calls through the normal GUI creation path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

import execution.file_handling as fh
import execution.model_operations as modeling
from byok import BYOKClient
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from diagram_handlers.types.gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from handlers.file_conversion_handler import (
    GUI_MOCKUP_ACTION,
    _build_image_prompt,
    _parse_llm_response,
)
from protocol.types import AssistantRequest, FileAttachment, WorkspaceContext
from schemas import AuthoredSystemGUISpec
from session_keys import PENDING_GUI_CHOICE
from tests.conftest import MINIMAL_CLASS_MODEL, FakeSession
from utilities.image_inputs import anthropic_content, content_text, openai_content

PNG = "data:image/png;base64,iVBORw0KGgo="


# -- image content per provider -------------------------------------------

def test_openai_content_adds_image_parts_after_the_text():
    assert openai_content("hi") == "hi"
    assert openai_content("hi", [PNG]) == [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": PNG}},
    ]


def test_anthropic_content_sends_base64_blocks_before_the_text():
    assert anthropic_content("hi", [PNG]) == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
        {"type": "text", "text": "hi"},
    ]


def test_content_text_logs_no_image_bytes():
    assert content_text(openai_content("hi", [PNG])) == "hi"


# -- the LLM call layer carries images -----------------------------------

class _ParseLLM:
    name = "gpt-4o-mini"

    def __init__(self, outcomes):
        self.requests = []
        outcomes = list(outcomes)

        def parse(**kwargs):
            self.requests.append(kwargs)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.client = SimpleNamespace(
            beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))),
        )


def _parsed(spec):
    message = SimpleNamespace(parsed=spec, content="{}", refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


@pytest.fixture
def _no_sleep(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_structured_call_sends_the_images_and_retries_on_the_text_part(_no_sleep):
    import openai
    from openai.types import CompletionUsage

    class Spec(BaseModel):
        name: str

    truncated = openai.LengthFinishReasonError(
        completion=SimpleNamespace(usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
    )
    llm = _ParseLLM([truncated, _parsed(Spec(name="Home"))])
    handler = ClassDiagramHandler(llm)

    handler.predict_structured("Build it", Spec, images=[PNG])

    first, retry = (r["messages"][-1]["content"] for r in llm.requests)
    assert first[1] == {"type": "image_url", "image_url": {"url": PNG}}
    assert "Be CONCISE" not in first[0]["text"]
    assert "Be CONCISE" in retry[0]["text"] and retry[1]["image_url"]["url"] == PNG


def test_free_text_call_sends_the_images(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="plan", refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)

    llm = SimpleNamespace(name="gpt-4o-mini", client=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    assert ClassDiagramHandler(llm)._predict_raw("Plan it", model="gpt-5-mini", images=[PNG]) == "plan"
    assert calls[0]["messages"][0]["content"][1]["image_url"]["url"] == PNG


def test_own_key_calls_send_the_images():
    openai_calls, anthropic_calls = [], []
    openai_client = BYOKClient.__new__(BYOKClient)
    openai_client.provider, openai_client._user_model, openai_client._base_url = "openai", None, None
    openai_client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: openai_calls.append(kw) or SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", refusal=None), finish_reason="stop")],
            usage=None,
        ),
    )))
    anthropic_client = BYOKClient.__new__(BYOKClient)
    anthropic_client.provider, anthropic_client._user_model, anthropic_client._base_url = "anthropic", None, None
    anthropic_client._client = SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: anthropic_calls.append(kw) or SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")], usage=None, stop_reason="end_turn",
        ),
    ))

    openai_client.predict_raw("Build it", images=[PNG])
    anthropic_client.predict_raw("Build it", images=[PNG])

    assert openai_calls[0]["messages"][0]["content"][1]["image_url"]["url"] == PNG
    assert anthropic_calls[0]["messages"][0]["content"][0]["source"]["data"] == "iVBORw0KGgo="


# -- the GUI generator reproduces the mockup -------------------------------

def _gui_with_capture(monkeypatch):
    handler = GUINoCodeDiagramHandler(None)
    captured = {}

    def two_pass(**kwargs):
        captured.update(kwargs)
        return AuthoredSystemGUISpec.model_validate({
            "projectName": "Museum",
            "pages": [{"name": "Home", "sections": [
                {"html": "<section class='ds-section'><h2>Welcome to the museum</h2></section>"},
            ]}],
        })

    monkeypatch.setattr(handler, "predict_two_pass_structured", two_pass)
    return handler, captured


def test_gui_generation_sends_the_mockup_and_the_reproduce_rules(monkeypatch):
    handler, captured = _gui_with_capture(monkeypatch)

    result = handler.generate_complete_system("Build this page", images=[PNG])

    assert result["action"] == "inject_complete_system"
    assert captured["images"] == [PNG]
    assert "REFERENCE MOCKUP" in captured["system_prompt"]
    assert "Copy every visible text exactly" in captured["system_prompt"]
    assert "mockups of the screens to build" in captured["reasoning_prompt"]


def test_gui_generation_without_images_is_unchanged(monkeypatch):
    handler, captured = _gui_with_capture(monkeypatch)

    handler.generate_complete_system("Build a museum landing page")

    assert not captured["images"]
    assert "REFERENCE MOCKUP" not in captured["system_prompt"]


# -- the converter recognises a UI image -----------------------------------

def test_image_prompt_offers_the_ui_answer():
    assert "user interface" in _build_image_prompt(detect_mockup=True)
    assert "user interface" not in _build_image_prompt()  # PDF pages stay diagram-only


def test_ui_verdict_becomes_the_mockup_action_for_images_only():
    verdict = '{"diagramType": "GUINoCodeDiagram"}'
    assert _parse_llm_response(verdict, "home.png", "image", allow_mockup=True) == {
        "action": GUI_MOCKUP_ACTION,
    }
    assert _parse_llm_response(verdict, "spec.pdf", "PDF")["action"] != GUI_MOCKUP_ACTION


# -- attachment routing ---------------------------------------------------

@pytest.fixture
def attach(monkeypatch):
    """Run handle_file_attachments with the conversion and the GUI op faked."""
    import llm.provider
    import utilities.model_resolution

    replies, operations, conversions = [], [], []
    monkeypatch.setattr(llm.provider, "get_provider", lambda: None)
    monkeypatch.setattr(fh, "reply_payload", lambda session, payload: replies.append(payload))
    monkeypatch.setattr(fh, "user_openai_key", lambda: "sk-test")
    monkeypatch.setattr(fh.ctx, "gpt_predict_json", lambda *a, **k: "{}", raising=False)
    monkeypatch.setattr(
        fh, "execute_model_operation", lambda **kwargs: operations.append(kwargs),
    )
    monkeypatch.setattr(utilities.model_resolution, "resolve_target_model", lambda *a: None)

    def run(message, verdict_type, conversion_result, filename="home.png"):
        monkeypatch.setattr(
            fh, "get_or_classify",
            lambda *a: SimpleNamespace(target_diagram_type=verdict_type),
        )
        monkeypatch.setattr(
            fh, "convert_file_to_diagram_spec",
            lambda **kwargs: conversions.append(kwargs) or conversion_result,
        )
        request = AssistantRequest(message=message, attachments=[
            FileAttachment(filename=filename, content_b64="aGk=", mime_type="image/png"),
        ])
        assert fh.handle_file_attachments(FakeSession(), request)
        return replies, operations, conversions

    return run


def test_a_mockup_asked_for_as_a_gui_goes_to_the_gui_generator(attach):
    replies, operations, conversions = attach("Make the GUI from this mockup", "GUINoCodeDiagram", None)

    assert conversions == []  # no UML conversion call
    (op,) = operations
    assert op["operation"] == {
        "diagramType": "GUINoCodeDiagram",
        "mode": "complete_system",
        "request": "Make the GUI from this mockup",
        "images": ["data:image/png;base64,aGk="],
    }
    assert replies == []


def test_a_ui_image_without_text_is_detected_and_built(attach):
    replies, operations, conversions = attach("", None, {"action": GUI_MOCKUP_ACTION})

    assert conversions[0]["target_diagram_type"] is None
    assert operations[0]["operation"]["request"] == fh._MOCKUP_REQUEST
    assert replies == []  # the internal action never reaches the frontend


def test_a_diagram_image_still_converts_with_the_requested_type(attach):
    converted = {"action": "inject_complete_system", "diagramType": "StateMachineDiagram"}
    replies, operations, conversions = attach("This is a state machine", "StateMachineDiagram", converted)

    assert conversions[0]["target_diagram_type"] == "StateMachineDiagram"
    assert replies == [converted]
    assert operations == []


def test_a_pdf_is_never_treated_as_a_mockup(attach):
    converted = {"action": "inject_complete_system", "diagramType": "ClassDiagram"}
    replies, operations, _ = attach("Make the GUI from this", "GUINoCodeDiagram", converted, "spec.pdf")

    assert replies == [converted]
    assert operations == []


# -- the GUI operation ----------------------------------------------------

@pytest.fixture
def gui_op(monkeypatch):
    handler = MagicMock()
    handler.generate_complete_system.return_value = {"action": "assistant_message", "message": "ok"}
    factory = MagicMock()
    factory.get_handler.return_value = handler
    monkeypatch.setattr(modeling.ctx, "diagram_factory", factory)
    for name in ("reply_progress", "reply_message", "reply_payload"):
        monkeypatch.setattr(modeling, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(modeling, "resolve_target_model", lambda *args: None)
    monkeypatch.setattr(modeling, "resolve_class_diagram", lambda request: MINIMAL_CLASS_MODEL)
    monkeypatch.setattr(modeling, "build_workspace_context_block", lambda *args: "")

    def run(operation):
        session = FakeSession()
        request = AssistantRequest(message=operation["request"], context=WorkspaceContext())
        modeling.execute_model_operation(session, request, operation, "complete_system")
        return session, handler

    return run


def test_a_mockup_skips_the_crud_or_ai_question_and_reaches_the_handler(gui_op):
    session, handler = gui_op({
        "diagramType": "GUINoCodeDiagram", "mode": "complete_system",
        "request": "Build the GUI shown in the attached mockup.", "images": [PNG],
    })

    assert session.get(PENDING_GUI_CHOICE) is None
    assert handler.generate_complete_system.call_args.kwargs["images"] == [PNG]
    assert handler.generate_complete_system.call_args.kwargs["class_metadata"]


def test_a_typed_gui_request_still_asks_crud_or_ai(gui_op):
    session, handler = gui_op({
        "diagramType": "GUINoCodeDiagram", "mode": "complete_system", "request": "Build screens",
    })

    assert session.get(PENDING_GUI_CHOICE) is not None
    handler.generate_complete_system.assert_not_called()
