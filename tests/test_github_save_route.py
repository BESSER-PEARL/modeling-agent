"""'Save to GitHub' in the chat opens the editor's GitHub save, not Render.

Live evidence (experimental, 2026-11-20 13:03 UTC): right after a Spec-Driven
run, "save to github" was classified generator_type='deploy' and the agent
answered "Opening the **Deploy to Render** dialog", because the agent had no
GitHub save action. These tests pin the new ``github`` route end to end.
"""

import pytest

from codegen.generation_handler import handle_generation_request
from codegen.smart_generation_handler import GenerationClassification
from protocol.actions import Action
from routing.generation_signals import detect_generator_type, is_github_save_request
from routing.unified_classifier import UnifiedClassification, _post_validate
from tests.conftest import FakeSession
from tests.test_generation_handler import _make_request, _patch_classifier


SAVE_TO_GITHUB = [
    "save to github",
    "Save it to GitHub please",
    "push the web app to github",
    "push the generated code to my github repo",
    "commit this to the repository",
    "save the project to github",
    "upload the code to my repo",
]

NOT_SAVE_TO_GITHUB = [
    "deploy to render",
    "push this to prod",
    "push it to github and deploy it on render",
    "export as json",
    "save the project",
    "continue from github.com/acme/shop",
    "start from my github repo",
    "create a class diagram for a github clone",
]


@pytest.mark.parametrize("message", SAVE_TO_GITHUB)
def test_save_to_github_is_detected(message):
    assert is_github_save_request(message)
    assert detect_generator_type(message) == "github"


@pytest.mark.parametrize("message", NOT_SAVE_TO_GITHUB)
def test_other_requests_are_not_github_saves(message):
    assert not is_github_save_request(message)
    assert detect_generator_type(message) != "github"


def _deterministic(generator_type):
    return UnifiedClassification(
        intent="generation_intent",
        generation_route="deterministic",
        generator_type=generator_type,
        reason="test",
    )


BUILD_THEN_PUSH = [
    ("create a class diagram for a library and generate a django app and push it to github", "django"),
    ("generate a django backend and commit it to my repo", "django"),
    ("push it to github once the fastapi backend is generated", "backend"),
]


@pytest.mark.parametrize("message, generator", BUILD_THEN_PUSH)
def test_a_build_request_that_ends_with_a_push_keeps_its_generator(message, generator):
    """Review of 2026-11-23: the planner read "generate a django app and push it to github"
    as a GitHub save, so a "yes" opened the save dialog and Django was never generated."""
    assert detect_generator_type(message) == generator


@pytest.mark.parametrize("llm_type", ["deploy", "export", None])
def test_post_validate_corrects_a_github_save_typed_as_something_else(llm_type):
    # The live misroute: the LLM picked 'deploy' for "save to github".
    result = _post_validate(_deterministic(llm_type), "save to github")
    assert result.generator_type == "github"


def test_post_validate_promotes_a_refused_github_save():
    result = UnifiedClassification(intent="generation_intent", generation_route="other", reason="test")
    result = _post_validate(result, "push it to github")
    assert (result.generation_route, result.generator_type) == ("deterministic", "github")


@pytest.mark.parametrize("message", ["deploy to render", "push this to prod"])
def test_post_validate_leaves_deploys_alone(message):
    assert _post_validate(_deterministic("deploy"), message).generator_type == "deploy"


def test_github_save_emits_the_editor_action(monkeypatch):
    _patch_classifier(
        monkeypatch,
        GenerationClassification(route="deterministic", generator_type="github", reason="test"),
    )
    reply = handle_generation_request(FakeSession(), _make_request("save to github"))
    assert reply["action"] == Action.TRIGGER_GITHUB_SAVE == "trigger_github_save"
    assert "Render" not in reply["message"]
