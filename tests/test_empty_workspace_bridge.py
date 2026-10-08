"""Tests for same-turn create→generate prerequisite bridging.

Repro: one prompt ("create a web app with a map and biking routes") is planned
into [create ClassDiagram, create GUINoCodeDiagram, generate web_app]. The
create ops push their models straight to the frontend but never write them back
into the backend request context, so the Phase-2 generate step read the original
EMPTY snapshot and wrongly replied "Your workspace looks empty — there's no
model to turn into web_app code yet."

The bridge records canonical in-turn creations in the working project snapshot
so generator-specific prerequisites see them. A GUI must be represented by
``pages`` (not a generic ``elements`` marker), and a genuinely empty workspace
still gets guidance.
"""

import pytest

from execution.model_operations import (
    _record_created_model_in_snapshot,
    _elements_from_result,
)
from handlers.generation_handler import (
    _missing_generator_prerequisites,
    _project_has_any_model,
    handle_generation_request,
)
from handlers.smart_generation_handler import GenerationClassification
from protocol.types import AssistantRequest, WorkspaceContext
from utilities.request_builders import build_generation_request

from tests.conftest import FakeSession


def _ctx(snapshot):
    return WorkspaceContext(project_snapshot=snapshot)


_CLASS_CREATE_PAYLOAD = {
    "action": "inject_complete_system",
    "systemSpec": {"classes": [{"className": "Book"}, {"className": "Member"}]},
    "diagramType": "ClassDiagram",
}

_GUI_CREATE_PAYLOAD = {
    "action": "inject_complete_system",
    "diagramType": "GUINoCodeDiagram",
    "model": {
        "pages": [{"id": "home", "name": "Home", "frames": []}],
        "styles": [],
    },
}

_AGENT_CREATE_PAYLOAD = {
    "action": "inject_complete_system",
    "diagramType": "AgentDiagram",
    "systemSpec": {
        "states": [{"stateName": "welcome"}],
        "intents": [{"intentName": "Greeting"}],
    },
}


# ---------------------------------------------------------------------------
# Snapshot-bridge helper + empty-workspace guard
# ---------------------------------------------------------------------------

class TestSnapshotBridgeHelper:
    def test_empty_snapshot_reads_as_empty(self):
        ctx = _ctx({"name": "P", "diagrams": {}})
        assert _project_has_any_model(ctx) is False

    def test_bridge_makes_created_class_diagram_visible(self):
        """After a create bridges its model in, the guard must return True."""
        ctx = _ctx({"name": "P", "diagrams": {}})
        assert _project_has_any_model(ctx) is False
        assert _record_created_model_in_snapshot(ctx, "ClassDiagram", _CLASS_CREATE_PAYLOAD) is True
        assert _project_has_any_model(ctx) is True

    def test_zero_class_create_does_not_bridge(self):
        """A create that produced nothing must NOT fake a non-empty workspace."""
        ctx = _ctx({"name": "P", "diagrams": {}})
        payload = {"action": "inject_complete_system", "systemSpec": {"classes": []}}
        assert _record_created_model_in_snapshot(ctx, "ClassDiagram", payload) is False
        assert _project_has_any_model(ctx) is False

    def test_existing_nonempty_model_left_untouched(self):
        snap = {
            "name": "P",
            "diagrams": {
                "ClassDiagram": [
                    {"model": {"elements": {"x": {"type": "Class", "name": "X"}}}}
                ]
            },
        }
        ctx = _ctx(snap)
        # Already visible → no-op, and the real entry is preserved.
        assert _record_created_model_in_snapshot(ctx, "ClassDiagram", _CLASS_CREATE_PAYLOAD) is False
        assert snap["diagrams"]["ClassDiagram"][0]["model"]["elements"]["x"]["name"] == "X"

    def test_unrecognized_gui_payload_does_not_fake_a_usable_gui(self):
        ctx = _ctx({"name": "P", "diagrams": {}})
        assert _record_created_model_in_snapshot(
            ctx, "GUINoCodeDiagram", {"action": "inject_complete_system"}
        ) is False
        assert _project_has_any_model(ctx) is False
        assert _missing_generator_prerequisites(ctx, "web_app") == [
            "ClassDiagram", "GUINoCodeDiagram",
        ]

    def test_canonical_gui_pages_are_visible_without_elements_map(self):
        ctx = _ctx({"name": "P", "diagrams": {}})

        assert _record_created_model_in_snapshot(
            ctx, "GUINoCodeDiagram", _GUI_CREATE_PAYLOAD,
        ) is True
        assert _project_has_any_model(ctx) is True

    def test_agent_system_spec_satisfies_agent_generator_prerequisite(self):
        ctx = _ctx({"name": "P", "diagrams": {}})

        assert _record_created_model_in_snapshot(
            ctx, "AgentDiagram", _AGENT_CREATE_PAYLOAD,
        ) is True
        assert _missing_generator_prerequisites(ctx, "agent") == []

    def test_elements_from_result_shapes(self):
        # systemSpec (class diagram)
        assert _elements_from_result(_CLASS_CREATE_PAYLOAD)
        # empty systemSpec → empty
        assert _elements_from_result({"systemSpec": {"classes": []}}) == {}
        # editor-model style
        assert _elements_from_result({"model": {"elements": {"a": {}}}}) == {"a": {}}

    def test_bridged_snapshot_flows_into_generation_request(self):
        """The bridge mutates the shared snapshot, so build_generation_request
        (which copies the snapshot by reference) carries it to the generate step."""
        ctx = _ctx({"name": "P", "diagrams": {}})
        _record_created_model_in_snapshot(ctx, "ClassDiagram", _CLASS_CREATE_PAYLOAD)
        assert _missing_generator_prerequisites(ctx, "web_app") == [
            "GUINoCodeDiagram",
        ]
        _record_created_model_in_snapshot(
            ctx, "GUINoCodeDiagram", _GUI_CREATE_PAYLOAD,
        )
        base = AssistantRequest(
            message="generate web_app", diagram_type="ClassDiagram",
            current_model=None, context=ctx,
        )
        gen_req = build_generation_request(base, generator_type="web_app")
        assert _missing_generator_prerequisites(gen_req.context, "web_app") == []


# ---------------------------------------------------------------------------
# handle_generation_request end-to-end (classifier stubbed like the existing
# generation-handler tests)
# ---------------------------------------------------------------------------

def _patch_classifier(monkeypatch, decision: GenerationClassification):
    import handlers.generation_handler as gen_mod
    from unified_classifier import UnifiedClassification

    unified = UnifiedClassification(
        intent="generation_intent",
        generation_route=decision.route,
        generator_type=decision.generator_type,
        refined_instructions=decision.refined_instructions,
        provider=decision.provider,
        reason=decision.reason,
    )

    class _FakeProvider:
        def parse(self, *, messages, schema, temperature, max_tokens):
            return unified

    monkeypatch.setattr(
        gen_mod, "_get_llm_provider", lambda: _FakeProvider(), raising=False,
    )


class TestGenerateAfterInTurnCreate:
    def test_generate_after_create_is_not_refused_as_empty(self, monkeypatch):
        """Create ran earlier in the same plan → generate proceeds (no "empty")."""
        _patch_classifier(monkeypatch, GenerationClassification(
            route="deterministic", generator_type="python", reason="python",
        ))
        ctx = _ctx({"name": "P", "diagrams": {}})
        _record_created_model_in_snapshot(ctx, "ClassDiagram", _CLASS_CREATE_PAYLOAD)
        request = AssistantRequest(message="generate python", context=ctx)
        result = handle_generation_request(FakeSession(), request)
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "python"
        assert "looks empty" not in (result.get("message") or "").lower()

    def test_generate_only_on_empty_workspace_still_refused(self, monkeypatch):
        """Generate-only on a genuinely empty workspace still gets guidance."""
        _patch_classifier(monkeypatch, GenerationClassification(
            route="deterministic", generator_type="python", reason="python",
        ))
        ctx = _ctx({"name": "P", "diagrams": {}})
        request = AssistantRequest(message="generate python", context=ctx)
        result = handle_generation_request(FakeSession(), request)
        assert result["action"] == "assistant_message"
        assert "looks empty" in result["message"].lower()
