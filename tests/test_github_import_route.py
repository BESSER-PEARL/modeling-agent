"""Continue-from-GitHub chat path: "continue from github.com/x/y" must emit
a ``trigger_github_import`` action (the frontend calls the backend import
endpoint, loads the project, and arms the modify machinery — the agent only
detects the intent and extracts owner/repo/branch).

Two deterministic guards under test:
  * handlers.generation_handler — extraction + the terminal payload, placed
    BEFORE sub-route dispatch in handle_generation_request;
  * unified_classifier._post_validate — reroutes a github-continuation
    message that the LLM parked in fallback/modify back to
    generation_intent so it reaches the handler guard at all.
"""

import pytest

from handlers.generation_handler import (
    _extract_github_reference,
    handle_generation_request,
)
from protocol.types import AssistantRequest, WorkspaceContext
from unified_classifier import UnifiedClassification, _post_validate

from tests.conftest import FakeSession


_CLASS_MODEL = {
    "elements": {
        "class-1": {"type": "Class", "name": "Book"},
    },
    "relationships": {},
}


def _make_request(message: str, action: str = "user_message") -> AssistantRequest:
    return AssistantRequest(
        action=action,
        message=message,
        context=WorkspaceContext(
            active_diagram_type="ClassDiagram",
            active_model=_CLASS_MODEL,
            project_snapshot={
                "name": "TestProject",
                "diagrams": {
                    "ClassDiagram": [{"model": _CLASS_MODEL}],
                },
            },
        ),
    )


# ---------------------------------------------------------------------------
# _extract_github_reference — pure extraction
# ---------------------------------------------------------------------------

class TestExtractGithubReference:
    def test_full_https_url(self):
        assert _extract_github_reference(
            "continue from https://github.com/armen/hotel-app"
        ) == ("armen", "hotel-app", None)

    def test_bare_github_com_url(self):
        assert _extract_github_reference(
            "continue from github.com/armen/hotel-app"
        ) == ("armen", "hotel-app", None)

    def test_url_with_tree_branch(self):
        assert _extract_github_reference(
            "continue from https://github.com/armen/hotel-app/tree/dev"
        ) == ("armen", "hotel-app", "dev")

    def test_url_with_git_suffix(self):
        assert _extract_github_reference(
            "load my previous generation from GitHub "
            "https://github.com/armen/hotel-app.git"
        ) == ("armen", "hotel-app", None)

    def test_url_trailing_sentence_period_not_swallowed(self):
        assert _extract_github_reference(
            "please continue from github.com/armen/hotel-app."
        ) == ("armen", "hotel-app", None)

    def test_slashed_branch_survives(self):
        assert _extract_github_reference(
            "resume https://github.com/armen/hotel-app/tree/feature/login"
        ) == ("armen", "hotel-app", "feature/login")

    def test_branch_word_in_message(self):
        assert _extract_github_reference(
            "continue from github.com/armen/hotel-app on branch dev"
        ) == ("armen", "hotel-app", "dev")

    def test_bare_owner_repo_with_verb_and_repo_word(self):
        assert _extract_github_reference(
            "continue from my repo armen/hotel-app"
        ) == ("armen", "hotel-app", None)

    def test_french_continuation(self):
        assert _extract_github_reference(
            "reprends depuis mon repo github.com/armen/hotel-app"
        ) == ("armen", "hotel-app", None)

    def test_bare_pair_alone_does_not_fire(self):
        # No continuation verb, no repo word — arbitrary "a/b" prose.
        assert _extract_github_reference("a/b") is None
        assert _extract_github_reference("look at armen/hotel-app") is None

    def test_bare_pair_with_verb_but_no_repo_word_does_not_fire(self):
        # Precision rule: the bare form needs BOTH signals.
        assert _extract_github_reference("continue from armen/hotel-app") is None

    def test_path_like_prose_does_not_fire_without_repo_shape(self):
        # Digits-only pairs are prose (fractions/dates), never a repo.
        assert _extract_github_reference("continue the repo work on 1/2") is None

    def test_lookalike_host_does_not_fire(self):
        assert _extract_github_reference(
            "continue from mygithub.com/armen/hotel-app"
        ) is None

    def test_ordinary_generation_request_does_not_fire(self):
        assert _extract_github_reference("generate django code") is None


# ---------------------------------------------------------------------------
# handle_generation_request — the terminal trigger_github_import payload
# ---------------------------------------------------------------------------

class TestGithubImportGuard:
    def test_full_url_emits_import_payload(self):
        session = FakeSession()
        result = handle_generation_request(
            session,
            _make_request("continue from https://github.com/armen/hotel-app"),
        )
        assert result["action"] == "trigger_github_import"
        assert result["owner"] == "armen"
        assert result["repo"] == "hotel-app"
        assert result["branch"] is None
        assert "Importing **armen/hotel-app**" in result["message"]
        assert "open it in the editor first" in result["message"]

    def test_tree_branch_lands_in_payload_and_message(self):
        session = FakeSession()
        result = handle_generation_request(
            session,
            _make_request(
                "continue from github.com/armen/hotel-app/tree/dev"
            ),
        )
        assert result["action"] == "trigger_github_import"
        assert result["branch"] == "dev"
        assert "on branch dev" in result["message"]

    def test_bare_owner_repo_with_verb_and_repo_word_fires(self):
        session = FakeSession()
        result = handle_generation_request(
            session, _make_request("resume from my repository armen/hotel-app"),
        )
        assert result["action"] == "trigger_github_import"
        assert (result["owner"], result["repo"]) == ("armen", "hotel-app")

    def test_import_abandons_pending_smart_gen_stash(self):
        """An import loads a DIFFERENT project — a stale pre-import smart-gen
        confirmation must not survive for a later generic "yes" to spend."""
        import time as _t
        from session_keys import (
            PENDING_GENERATOR_TYPE,
            PENDING_SMART_GEN_INSTRUCTIONS,
            PENDING_SMART_GEN_PROVIDER,
            PENDING_SMART_GEN_TIMESTAMP,
        )
        session = FakeSession()
        session.set(PENDING_SMART_GEN_INSTRUCTIONS, "build a shop app")
        session.set(PENDING_SMART_GEN_PROVIDER, "anthropic")
        session.set(PENDING_SMART_GEN_TIMESTAMP, _t.time())
        session.set(PENDING_GENERATOR_TYPE, "django")
        result = handle_generation_request(
            session, _make_request("continue from github.com/armen/hotel-app"),
        )
        assert result["action"] == "trigger_github_import"
        assert not session.get(PENDING_SMART_GEN_INSTRUCTIONS)
        assert not session.get(PENDING_GENERATOR_TYPE)

    def test_plain_generation_request_unaffected(self):
        """No GitHub reference → the guard stays out of the way (this message
        falls through to normal sub-routing; without a cached verdict or a
        provider it lands on the resilient generator menu)."""
        session = FakeSession()
        result = handle_generation_request(
            session, _make_request("generate django code"),
        )
        assert result["action"] != "trigger_github_import"


# ---------------------------------------------------------------------------
# unified_classifier._post_validate — classifier-side reroute
# ---------------------------------------------------------------------------

def _uc(intent, **kw):
    return UnifiedClassification(intent=intent, reason="test", **kw)


class TestGithubContinuationReroute:
    @pytest.mark.parametrize("wrong_intent", [
        "fallback_intent",
        "modify_model_intent",
        "modeling_help_intent",
    ])
    def test_misrouted_continuation_reroutes_to_generation(self, wrong_intent):
        out = _post_validate(
            _uc(wrong_intent), "continue from github.com/armen/hotel-app",
        )
        assert out.intent == "generation_intent"
        assert out.pending_flow_action == "new_request"

    def test_french_continuation_reroutes(self):
        out = _post_validate(
            _uc("fallback_intent"),
            "reprends depuis mon repo github.com/armen/hotel-app",
        )
        assert out.intent == "generation_intent"
        assert out.pending_flow_action == "new_request"

    def test_bare_repo_continuation_reroutes(self):
        out = _post_validate(
            _uc("modify_model_intent"),
            "load my previous generation from my repo armen/hotel-app",
        )
        assert out.intent == "generation_intent"

    def test_url_without_continuation_verb_keeps_classified_intent(self):
        """A URL used as a design reference is NOT a continuation — the
        reroute must not hijack a legitimate create."""
        out = _post_validate(
            _uc("create_complete_system_intent"),
            "create a class diagram like the one in github.com/armen/hotel-app",
        )
        assert out.intent == "create_complete_system_intent"

    def test_plain_modify_untouched(self):
        out = _post_validate(
            _uc("modify_model_intent"), "add a Payment class",
        )
        assert out.intent == "modify_model_intent"

    def test_generation_verdict_passes_through_unchanged(self):
        """Already generation_intent → the handler guard takes it from
        there; the reroute must not rewrite the verdict."""
        out = _post_validate(
            _uc(
                "generation_intent",
                generation_route="deterministic",
                generator_type="django",
            ),
            "continue from github.com/armen/hotel-app",
        )
        assert out.intent == "generation_intent"
        assert out.generator_type == "django"
