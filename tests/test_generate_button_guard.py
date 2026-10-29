"""The post-build "Generate the code" button must reach the Spec-Driven flow.

After the agent builds a class diagram, ``suggestions.py`` offers "Generate the
code", which sends "generate the application". Live on 2026-10-29 with the
classifier on gpt-6-luna, 4 of 12 such clicks came back as

    intent=generation_intent route=deterministic disp=reuse_for_generation

with ``generator_type=None``, which shows the generator menu instead of the
smart-generation confirmation (gpt-4o-mini: smart 12 of 12). A rulebook rule
scoped to "right after a build" fixed that but also moved the hand-drawn and
empty-workspace readings of the same words (gpt-4o-mini, empty workspace:
menu 12/12 -> smart 12/12), so those must keep the classifier's verdict.
"""

from protocol.types import AssistantRequest, WorkspaceContext
from unified_classifier import UnifiedClassification, get_or_classify
from utilities.original_request import remember_original_request

from tests.conftest import FakeSession

_PROJECT = "proj-library"


def _request(message="generate the application", project_id=_PROJECT):
    return AssistantRequest(
        message=message,
        context=WorkspaceContext(
            active_diagram_type="ClassDiagram",
            project_snapshot={"id": project_id, "name": "Library", "diagrams": {}},
        ),
    )


class _Provider:
    def __init__(self, decision):
        self.decision = decision

    def parse(self, *, messages, schema, temperature, max_tokens):
        return self.decision


_MENU = UnifiedClassification(
    intent="generation_intent", generation_route="deterministic",
    model_disposition="reuse_for_generation", reason="live gpt-6-luna verdict",
)
_WEB_APP = UnifiedClassification(
    intent="generation_intent", generation_route="deterministic", generator_type="web_app",
    model_disposition="reuse_for_generation", reason="live gpt-4o-mini hand-drawn verdict",
)


def _agent_built_session():
    session = FakeSession()
    remember_original_request(session, "create a class diagram for a library management system", _PROJECT)
    return session


def test_post_build_menu_verdict_goes_to_smart():
    result = get_or_classify(_agent_built_session(), _request(), _Provider(_MENU))
    assert result.generation_route == "smart"
    assert result.generator_type is None
    assert result.model_disposition == "reuse_for_generation"
    assert result.refined_instructions


def test_post_build_web_app_verdict_goes_to_smart():
    result = get_or_classify(_agent_built_session(), _request("Generate the application."),
                             _Provider(_WEB_APP))
    assert result.generation_route == "smart"
    assert result.generator_type is None


def test_hand_drawn_model_keeps_the_classifier_verdict():
    # No agent build recorded for this project: the user drew the model.
    result = get_or_classify(FakeSession(), _request(), _Provider(_WEB_APP))
    assert (result.generation_route, result.generator_type) == ("deterministic", "web_app")


def test_build_in_another_project_does_not_count():
    session = _agent_built_session()
    result = get_or_classify(session, _request(project_id="proj-other"), _Provider(_MENU))
    assert result.generation_route == "deterministic"


def test_unknown_project_fails_closed():
    session = _agent_built_session()
    result = get_or_classify(session, _request(project_id=None), _Provider(_MENU))
    assert result.generation_route == "deterministic"


def test_other_phrasings_keep_the_classifier_verdict():
    django = UnifiedClassification(
        intent="generation_intent", generation_route="deterministic",
        generator_type="django", reason="named generator",
    )
    result = get_or_classify(_agent_built_session(), _request("generate the application in django"),
                             _Provider(django))
    assert result.generator_type == "django"
    menu = get_or_classify(_agent_built_session(), _request("how do I see the app"), _Provider(_MENU))
    assert menu.generation_route == "deterministic"


def test_non_generation_intent_is_left_alone():
    modify = UnifiedClassification(intent="modify_model_intent", reason="answer")
    result = get_or_classify(_agent_built_session(), _request(), _Provider(modify))
    assert result.intent == "modify_model_intent"
