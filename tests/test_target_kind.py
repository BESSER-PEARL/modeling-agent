"""Any target, not only web apps.

A prod review (in Spanish): asking for a Windows 11 local app or a mobile app
got a web app, MVC was ignored, and "no, una app nativa, no una web app" was
classified ``decline_intent``, which dropped the request. The replies were in
English. These tests pin the fix: the classifier names the target, it is
remembered per project, the planner and the confirmation honour it, a "no, X
instead" corrects instead of cancelling, and replies come in the user's
language.
"""

import itertools
import json
import time

import pytest

from protocol.types import AssistantRequest, WorkspaceContext
from session_keys import (
    APP_TARGET,
    PENDING_SMART_GEN_INSTRUCTIONS,
    PENDING_SMART_GEN_ORIGINAL_REQUEST,
    PENDING_SMART_GEN_PROVIDER,
    PENDING_SMART_GEN_TIMESTAMP,
    REPLY_LANGUAGE,
    UNIFIED_CLASSIFICATION,
)
from unified_classifier import (
    _SYSTEM_PROMPT,
    UnifiedClassification,
    _post_validate,
    classify_message,
    get_or_classify,
)

from tests.conftest import FakeSession, make_session

_MODEL = {"elements": {"c1": {"type": "Class", "name": "Product", "owner": None}},
          "relationships": {}}
_SNAPSHOT = {"id": "shop-1", "name": "Shop", "diagrams": {"ClassDiagram": [{"model": _MODEL}]}}

_WEB_CONFIRMATION_START = (
    "BESSER will generate your application from the designed specs using its "
    "built-in generators. If some of your requirements are not supported by "
    "these generators, BESSER can use an LLM to handle them.\n\n"
    "BESSER includes a free model by default"
)


def _uc(**fields):
    return UnifiedClassification(**{"intent": "generation_intent", "reason": "stub", **fields})


def _request(message, snapshot=_SNAPSHOT):
    return AssistantRequest(
        message=message,
        context=WorkspaceContext(
            active_diagram_type="ClassDiagram", active_model=_MODEL, project_snapshot=snapshot,
        ),
    )


class _Provider:
    def __init__(self, decision):
        self.decision = decision
        self.messages = None

    def parse(self, *, messages, schema, temperature, max_tokens):
        self.messages = messages
        return self.decision


_event_ids = itertools.count()


class _EventSession(FakeSession):
    """A FakeSession whose event id keys the classification cache."""

    def __init__(self, message, snapshot=_SNAPSHOT):
        super().__init__()
        self.event.id = f"evt-{next(_event_ids)}"
        self.request = _request(message, snapshot)

    def next_message(self, message):
        self.event.id = f"evt-{next(_event_ids)}"
        self.request = _request(message, self.request.context.project_snapshot)


def _classify(session, decision, monkeypatch):
    import handlers.generation_handler as generation
    monkeypatch.setattr(generation, "recent_smart_gen_for_project", lambda s, w: False)
    provider = _Provider(decision)
    result = get_or_classify(session, session.request, provider)
    return result, provider


# ---------------------------------------------------------------------
# Classifier: fields and rules
# ---------------------------------------------------------------------

class TestClassifierCarriesTheTarget:
    def test_schema_has_target_language_and_architecture_fields(self):
        fields = UnifiedClassification.model_fields
        for name in ("target_kind", "target_description", "reply_language", "architecture"):
            assert name in fields
        verdict = _uc(target_kind="other", target_description="native mobile app")
        assert verdict.target_kind == "other"
        with pytest.raises(Exception):
            _uc(target_kind="mobile")

    def test_rulebook_names_the_traps(self):
        # The reviewer's cases and the look-alikes that are not targets.
        for phrase in ("mobile-friendly", "REST API for my mobile app", "Windows server",
                       "cancellation windows", "native speakers", "una app de Windows 11",
                       "no, una app nativa, no una web app",
                       # Live findings on gpt-4o-mini: a domain was returned as the
                       # target, a correction kept the remembered target, and a
                       # screen added to a generated native app became a web GUI edit.
                       "never a target", "never the REMEMBERED one",
                       "'añade una pantalla de ajustes'"):
            assert phrase in _SYSTEM_PROMPT, phrase
        assert "FIRST state the target explicitly" in _SYSTEM_PROMPT

    def test_classifier_sees_the_remembered_target_and_language(self):
        provider = _Provider(_uc(generation_route="smart", refined_instructions="x"))
        classify_message(
            _request("generate the application"), provider,
            remembered_target={"kind": "other", "description": "native mobile app",
                               "architecture": "MVC"},
            conversation_language="es",
        )
        block = provider.messages[1]["content"]
        assert "REMEMBERED TARGET" in block and "native mobile app" in block and "MVC" in block
        assert "CONVERSATION LANGUAGE: es" in block

    def test_guards_keep_the_target_when_they_rebuild_the_verdict(self):
        verdict = UnifiedClassification(
            intent="create_complete_system_intent", target_diagram_type="GUINoCodeDiagram",
            model_disposition="new_from_scratch", target_kind="other",
            target_description="Windows 11 desktop app", reply_language="ES", reason="stub",
        )
        result = _post_validate(verdict, "quiero una app de windows 11 local para el inventario")
        assert result.target_diagram_type == "ClassDiagram"  # data-model-first guard ran
        assert (result.target_kind, result.target_description) == ("other", "Windows 11 desktop app")
        assert result.reply_language == "es"

    @pytest.mark.parametrize("raw,expected", [("es", "es"), ("pt-BR", "pt"), ("spanish", None), ("", None)])
    def test_reply_language_shape(self, raw, expected):
        result = _post_validate(_uc(intent="hello_intent", reply_language=raw or None), "hola")
        assert result.reply_language == expected


class TestNoInsteadIsACorrection:
    def test_decline_that_names_a_new_target_becomes_the_corrected_request(self):
        verdict = UnifiedClassification(
            intent="decline_intent", target_kind="other", target_description="native mobile app",
            reply_language="es", reason="stub",
        )
        result = _post_validate(verdict, "no, una app nativa, no una web app")
        assert result.intent == "generation_intent"
        assert result.generation_route == "smart"
        assert result.pending_flow_action == "new_request"
        assert result.target_description == "native mobile app"
        assert "una app nativa" in result.refined_instructions

    def test_cancel_answer_that_names_a_new_target_is_not_a_cancel(self):
        verdict = _uc(generation_route="smart", refined_instructions="Build a desktop app",
                      pending_flow_action="answer", pending_flow_answer="cancel",
                      target_kind="other", target_description="desktop app")
        result = _post_validate(verdict, "no, make it a desktop app")
        assert result.pending_flow_action == "new_request"
        assert result.pending_flow_answer is None
        assert result.refined_instructions == "Build a desktop app"

    @pytest.mark.parametrize("message", ["no", "no thanks", "nada"])
    def test_bare_no_still_declines(self, message):
        result = _post_validate(UnifiedClassification(intent="decline_intent", reason="stub"), message)
        assert result.intent == "decline_intent"

    def test_correction_at_the_confirmation_reissues_it_with_the_new_target(self, monkeypatch):
        import handlers.generation_handler as generation
        session = make_session("no, una app nativa, no una web app",
                               active_model=_MODEL, project_snapshot=_SNAPSHOT)
        monkeypatch.setattr(generation, "_active_project_id", lambda s: "shop-1")
        generation.remember_original_request(session, "quiero una app para mi tienda", "shop-1")
        session.set(PENDING_SMART_GEN_INSTRUCTIONS, "Build a web app for the shop")
        session.set(PENDING_SMART_GEN_PROVIDER, "anthropic")
        session.set(PENDING_SMART_GEN_TIMESTAMP, time.time())
        verdict = _post_validate(UnifiedClassification(
            intent="decline_intent", target_kind="other", target_description="native mobile app",
            reason="stub"), "no, una app nativa, no una web app")
        session.set(UNIFIED_CLASSIFICATION, verdict)
        from utilities.original_request import remember_target
        remember_target(session, "shop-1", kind="other", description="native mobile app")

        result = generation.handle_generation_request(
            session, _request("no, una app nativa, no una web app"))

        assert "cancelled" not in result["message"].lower()
        assert "**native mobile app**" in result["message"]
        assert "Target: native mobile app" in session.get(PENDING_SMART_GEN_INSTRUCTIONS)
        original = session.get(PENDING_SMART_GEN_ORIGINAL_REQUEST)
        assert "quiero una app para mi tienda" in original
        assert "no, una app nativa, no una web app" in original


# ---------------------------------------------------------------------
# Remembering the target per project (plan table C)
# ---------------------------------------------------------------------

class TestTargetMemory:
    def test_stated_in_first_prompt_survives_a_generic_generate(self, monkeypatch):
        session = _EventSession("quiero una app móvil para controlar mis gastos")
        _classify(session, UnifiedClassification(
            intent="create_complete_system_intent", target_kind="other",
            target_description="native mobile app", reply_language="es", reason="stub"), monkeypatch)
        session.next_message("generate the application")
        _, provider = _classify(session, _uc(generation_route="smart", refined_instructions="x"), monkeypatch)
        assert session.get(APP_TARGET)["description"] == "native mobile app"
        assert "native mobile app" in provider.messages[1]["content"]
        assert session.get(REPLY_LANGUAGE) == "es"

    def test_latest_statement_wins_and_architecture_is_kept(self, monkeypatch):
        session = _EventSession("crea una aplicación con arquitectura MVC para una biblioteca")
        _classify(session, UnifiedClassification(
            intent="create_complete_system_intent", architecture="MVC", reason="stub"), monkeypatch)
        session.next_message("actually make it a desktop app")
        _classify(session, _uc(generation_route="smart", refined_instructions="x",
                               target_kind="other", target_description="desktop app"), monkeypatch)
        assert session.get(APP_TARGET) == {"kind": "other", "description": "desktop app",
                                           "architecture": "MVC"}

    def test_another_project_does_not_inherit_the_target(self, monkeypatch):
        session = _EventSession("an android app for gym tracking")
        _classify(session, UnifiedClassification(
            intent="create_complete_system_intent", target_kind="other",
            target_description="native Android app", reason="stub"), monkeypatch)
        session.next_message("generate the app")
        _, same = _classify(session, _uc(generation_route="smart", refined_instructions="x"), monkeypatch)
        assert "REMEMBERED TARGET" in same.messages[1]["content"]
        session.next_message("generate the app")
        session.request.context.project_snapshot = {"id": "other-project", "diagrams": {}}
        _, other = _classify(session, _uc(generation_route="smart", refined_instructions="x"), monkeypatch)
        assert "REMEMBERED TARGET" not in other.messages[1]["content"]

    def test_quick_action_click_keeps_the_conversation_language(self, monkeypatch):
        from session_keys import LAST_SUGGESTED_PROMPTS
        session = _EventSession("quiero una app de windows 11")
        session.set(REPLY_LANGUAGE, "es")
        session.set(LAST_SUGGESTED_PROMPTS, ["generate the application"])
        session.next_message("generate the application")
        _classify(session, _uc(generation_route="smart", refined_instructions="x",
                               reply_language="en"), monkeypatch)
        assert session.get(REPLY_LANGUAGE) == "es"
        session.next_message("je veux une appli de bureau")
        _classify(session, _uc(generation_route="smart", refined_instructions="x",
                               reply_language="fr"), monkeypatch)
        assert session.get(REPLY_LANGUAGE) == "fr"  # a typed message does switch it

    def test_non_web_follow_up_screen_is_not_a_web_gui_edit(self):
        # Row 6: "add a settings screen" to a generated native app edits its code.
        verdict = _uc(generation_route="smart", refined_instructions="Add a settings screen")
        kept = _post_validate(verdict.model_copy(), "add a settings screen",
                              follow_up_target_kind="other")
        assert kept.intent == "generation_intent"
        web = _post_validate(verdict.model_copy(), "add a settings screen")
        assert web.target_diagram_type == "GUINoCodeDiagram"  # web behaviour unchanged


# ---------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------

def _empty_request(message):
    return AssistantRequest(message=message, context=WorkspaceContext(active_diagram_type="ClassDiagram"))


class TestPlannerForNonWebTargets:
    def test_ui_request_for_a_native_app_plans_no_web_gui(self):
        from orchestrator import plan_assistant_operations
        message = "build a mobile app with screens for tracking my expenses"
        web = plan_assistant_operations(_empty_request(message), "complete_system",
                                        "create_complete_system_intent", lambda p: "{}")
        assert any(op.get("diagramType") == "GUINoCodeDiagram" for op in web)
        native = plan_assistant_operations(_empty_request(message), "complete_system",
                                           "create_complete_system_intent", lambda p: "{}",
                                           app_target_kind="other")
        assert [op.get("diagramType") for op in native] == ["ClassDiagram"]
        assert not any(op.get("generatorType") == "web_app" for op in native)

    def test_llm_planner_web_plan_is_filtered_and_told(self):
        from orchestrator import plan_assistant_operations
        prompts = []

        def predict(prompt):
            prompts.append(prompt)
            return json.dumps({"operations": [
                {"type": "model", "diagramType": "ClassDiagram", "mode": "complete_system",
                 "request": "inventory"},
                {"type": "model", "diagramType": "GUINoCodeDiagram", "mode": "complete_system",
                 "request": "gui"},
                {"type": "generation", "generatorType": "web_app"},
            ]})
        # Multi-clause and long enough that the LLM planner runs.
        message = ("I need an expense tracker for my phone, with budgets per category, recurring "
                   "expenses, monthly reports, and reminders when I overspend my budget.")
        ops = plan_assistant_operations(
            _empty_request(message), "complete_system", None, predict, app_target_kind="other",
        )
        assert prompts and "NOT a web app" in prompts[0]
        assert [op.get("diagramType") for op in ops] == ["ClassDiagram"]

    def test_flutter_generator_is_never_offered(self):
        from handlers.generation_handler import GENERATOR_PREREQUISITES
        from suggestions import GENERATOR_ARTIFACT_LABELS
        assert "flutter" not in GENERATOR_PREREQUISITES
        assert "flutter" not in GENERATOR_ARTIFACT_LABELS


# ---------------------------------------------------------------------
# Confirmation copy
# ---------------------------------------------------------------------

class TestConfirmationNamesTheTarget:
    def _confirm(self, monkeypatch, **target):
        import handlers.generation_handler as generation
        session = make_session("generate the application", active_model=_MODEL, project_snapshot=_SNAPSHOT)
        monkeypatch.setattr(generation, "_active_project_id", lambda s: "shop-1")
        session.set(UNIFIED_CLASSIFICATION, _uc(generation_route="smart", **target))
        payload = generation._build_smart_gen_confirmation(
            session, "Build the inventory application.", "anthropic",
            user_message="generate the application")
        return session, payload

    def test_non_web_target_is_named_and_honestly_unverified(self, monkeypatch):
        session, payload = self._confirm(
            monkeypatch, target_kind="other", target_description="Windows 11 desktop app")
        assert "**Windows 11 desktop app**" in payload["message"]
        assert "no built-in generator" in payload["message"]
        assert "can't compile" in payload["message"]
        assert session.get(PENDING_SMART_GEN_INSTRUCTIONS).startswith(
            "Target: Windows 11 desktop app (not a web app).")
        assert payload["suggestedActions"][0]["prompt"] == "generate anyway with my current model"

    def test_named_architecture_is_acknowledged(self, monkeypatch):
        session, payload = self._confirm(monkeypatch, architecture="MVC")
        assert "**MVC** architecture" in payload["message"]
        assert "Architecture: MVC." in session.get(PENDING_SMART_GEN_INSTRUCTIONS)

    @pytest.mark.parametrize("target", [{}, {"target_kind": "web_app", "target_description": "web app"}])
    def test_web_wording_is_unchanged(self, monkeypatch, target):
        session, payload = self._confirm(monkeypatch, **target)
        assert payload["message"].startswith(_WEB_CONFIRMATION_START)
        assert session.get(PENDING_SMART_GEN_INSTRUCTIONS) == "Build the inventory application."

    def test_post_spec_button_names_the_target(self):
        from suggestions import get_post_spec_suggestions
        first = get_post_spec_suggestions(None, label="native mobile app")[0]
        assert first == {"label": "Generate native mobile app", "prompt": "generate the application"}


# ---------------------------------------------------------------------
# Replies in the user's language
# ---------------------------------------------------------------------

class _Translator:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, prompt):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("LLM down")
        texts = json.loads(prompt[prompt.rindex("{\"texts\""):])["texts"]
        return json.dumps({"texts": [f"ES<{t}>" for t in texts]})


@pytest.fixture
def translator(monkeypatch):
    import reply_language
    fake = _Translator()
    monkeypatch.setattr(reply_language, "_predict_json", fake)
    reply_language._cache.clear()
    return fake


def _spanish_session():
    session = make_session("quiero una app")
    session.set(REPLY_LANGUAGE, "es")
    return session


class TestReplyTranslation:
    def test_payload_message_and_labels_translated_in_one_call(self, translator):
        from session_helpers import reply_payload
        session = _spanish_session()
        model = {"elements": {"c1": {"name": "Book"}}}
        reply_payload(session, {
            "action": "inject_complete_system", "message": "Created **Book**.", "model": model,
            "suggestedActions": [{"label": "Continue", "prompt": "generate anyway with my current model"}],
        })
        sent = session.last_reply_json()
        assert sent["message"] == "ES<Created **Book**.>"
        assert sent["suggestedActions"] == [
            {"label": "ES<Continue>", "prompt": "generate anyway with my current model"}]
        assert sent["model"] == model
        assert len(translator.calls) == 1

    def test_message_and_progress_translated_and_cached(self, translator):
        from session_helpers import reply_message, reply_progress
        session = _spanish_session()
        reply_message(session, "Cancelled. Your model is unchanged.")
        reply_message(session, "Cancelled. Your model is unchanged.")
        reply_progress(session, "Building your class diagram", 1, 2)
        sent = [json.loads(r) for r in session.replies]
        assert sent[0]["message"] == "ES<Cancelled. Your model is unchanged.>"
        assert sent[2]["message"] == "ES<Building your class diagram>"
        assert len(translator.calls) == 2  # the repeated reply came from the cache

    def test_english_is_never_translated(self, translator):
        from session_helpers import reply_message
        session = make_session("hello")
        reply_message(session, "Hello!")
        assert session.last_reply_json()["message"] == "Hello!"
        assert translator.calls == []

    def test_failure_sends_english(self, monkeypatch, caplog):
        import reply_language
        from session_helpers import reply_message
        monkeypatch.setattr(reply_language, "_predict_json", _Translator(fail=True))
        reply_language._cache.clear()
        session = _spanish_session()
        reply_message(session, "Do you want to continue?")
        assert session.last_reply_json()["message"] == "Do you want to continue?"
        assert "Reply translation to es failed" in caplog.text

    def test_translation_that_breaks_a_link_keeps_english(self, monkeypatch):
        import reply_language
        reply_language._cache.clear()
        monkeypatch.setattr(reply_language, "_predict_json",
                            lambda p: json.dumps({"texts": ["[configura tu clave](wme:clave)"]}))
        source = "[set up your own API key](wme:add-key)"
        assert reply_language.translate_texts([source], "es") == [source]

    def test_llm_replies_are_asked_for_the_users_language(self):
        from session_helpers import stream_llm_response
        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return iter(())

        class _Client:
            chat = type("Chat", (), {"completions": _Completions()})()

        llm = type("LLM", (), {"client": _Client(), "name": "gpt-4o-mini"})()
        session = _spanish_session()
        stream_llm_response(session, llm, "what can you do?")
        system = captured["messages"][0]
        assert system["role"] == "system" and "Reply in Spanish" in system["content"]
