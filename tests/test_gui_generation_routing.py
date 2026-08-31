"""'Generate the GUI' must route to GUI-diagram creation, never the web-app
code generator. Live bug: after a GUI build, 'generate the gui' stashed a
smart-gen confirmation, and every 'no, generate the gui model' reply was
itself re-misrouted — re-creating the identical confirmation in a loop.
The _post_validate guard breaks that class of loop deterministically.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unified_classifier import UnifiedClassification, _post_validate  # noqa: E402


def _gen(route="smart", **kw):
    return UnifiedClassification(
        intent="generation_intent", generation_route=route,
        refined_instructions="build it", reason="test", **kw,
    )


class TestGuiDiagramGuard:
    def test_generate_the_gui_reroutes_to_modeling(self):
        for msg in (
            "hello generate the gui",
            "no generate the gui model for my library",
            "no but generate the gui model according to my model",
            "create the screens for my model",
            "generate the user interface",
        ):
            out = _post_validate(_gen(), msg)
            assert out.intent == "create_complete_system_intent", msg
            assert out.target_diagram_type == "GUINoCodeDiagram", msg
            assert out.pending_flow_action == "new_request", msg

    def test_web_app_requests_stay_generation(self):
        for msg in (
            "generate the web app",
            "generate the application code with a nice gui",
            "generate the frontend code for the ui",
            "generate a django app",
        ):
            out = _post_validate(_gen(), msg)
            assert out.intent == "generation_intent", msg

    def test_non_gui_generation_untouched(self):
        out = _post_validate(_gen(route="deterministic", generator_type="sql"),
                             "generate sql")
        assert out.intent == "generation_intent"
        assert out.generator_type == "sql"


class TestGuiPageEditGuard:
    """Marathon bug: "add a Reports page" with an existing GUI stashed a
    spec-driven web-app confirmation. Page/screen edits are GUI-diagram
    modifications, deterministically."""

    def test_page_edits_reroute_to_gui_modify(self):
        for msg in (
            "add a Reports page",
            "remove the Settings page",
            "rename the Home screen",
            "add a page for bookings",
        ):
            out = _post_validate(_gen(), msg)
            assert out.intent == "modify_model_intent", msg
            assert out.target_diagram_type == "GUINoCodeDiagram", msg

    def test_app_page_requests_stay_generation(self):
        for msg in (
            "generate the web app with a login page",
            "add a payment page to the backend code",
        ):
            out = _post_validate(_gen(), msg)
            assert out.intent == "generation_intent", msg
