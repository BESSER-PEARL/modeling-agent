"""GUI agent overhaul: structured generation, custom themes, non-destructive
failure, and the Studio-grade modification path (outline + batch +
edit_section). Pins the contracts introduced when the free-text create path
and the single-op modify path were retired.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagram_handlers.types.gui_nocode_diagram_handler import (  # noqa: E402
    GUI_COMPLETE_SYSTEM_MAX_TOKENS,
    GUINoCodeDiagramHandler,
    _nav_header_component,
    _parse_llm_css,
)
from diagram_handlers.types.gui_design_system import (  # noqa: E402
    merge_theme_overrides,
    stylesheet_rules_from_tokens,
)
from schemas import (  # noqa: E402
    AuthoredGUIPageSpec,
    AuthoredGUISectionSpec,
    AuthoredSystemGUISpec,
    GUIModificationBatchSpec,
    GUIModificationSpec,
    GUIThemeSpec,
)


HTML_SECTION = (
    "<section class='ds-section'><h2 class='ds-heading'>Overview</h2>"
    "<p>Concrete copy.</p></section>"
)


def _authored_spec(**overrides):
    base = dict(
        projectName="Clinic",
        domain="health",
        pages=[AuthoredGUIPageSpec(
            name="Home",
            sections=[
                AuthoredGUISectionSpec(html=HTML_SECTION),
                AuthoredGUISectionSpec(),  # empty — must be dropped
            ],
        )],
    )
    base.update(overrides)
    return AuthoredSystemGUISpec(**base)


class TestStructuredGeneration:
    def _run(self, monkeypatch, spec=None, raise_exc=None):
        handler = GUINoCodeDiagramHandler(None)

        def fake_two_pass_structured(**kwargs):
            assert kwargs["response_schema"] is AuthoredSystemGUISpec
            if raise_exc:
                raise raise_exc
            return spec

        monkeypatch.setattr(
            handler, "predict_two_pass_structured", fake_two_pass_structured,
        )
        return handler.generate_complete_system("create a patient portal")

    def test_builds_pages_from_structured_spec(self, monkeypatch):
        result = self._run(monkeypatch, spec=_authored_spec())
        assert result["action"] == "inject_complete_system"
        model = result["model"]
        assert len(model["pages"]) == 1
        assert model["styles"], "domain stylesheet must be attached"

    def test_custom_theme_overrides_stylesheet(self, monkeypatch):
        spec = _authored_spec(theme=GUIThemeSpec(primary="#14532d", radius="0px"))
        result = self._run(monkeypatch, spec=spec)
        css = str(result["model"]["styles"])
        assert "#14532d" in css
        # The preset health primary must have been replaced.
        assert "#0d8b8b" not in css.split("--ds-primary")[1][:30]

    def test_failure_never_replaces_screens_with_stub(self, monkeypatch):
        result = self._run(monkeypatch, raise_exc=RuntimeError("boom"))
        assert result["action"] == "assistant_message"
        assert result.get("error") is True
        assert "model" not in result  # nothing injected — screens untouched

    def test_token_budget_extended_for_gui_schemas(self):
        handler = GUINoCodeDiagramHandler(None)
        assert handler._structured_max_tokens(AuthoredSystemGUISpec) == GUI_COMPLETE_SYSTEM_MAX_TOKENS
        assert handler._structured_max_tokens(GUIModificationBatchSpec) == GUI_COMPLETE_SYSTEM_MAX_TOKENS


def _model_with_sections():
    handler = GUINoCodeDiagramHandler(None)
    spec = _authored_spec()
    fake = spec.model_dump()
    pages = []
    for page in fake["pages"]:
        page["sections"] = [s for s in page["sections"] if s.get("html")]
        pages.append(handler._parse_page_spec(
            {"name": page["name"], "sections": page["sections"]},
            None, all_page_names=[page["name"]], project_name="Clinic",
        ))
    return handler, {"pages": pages, "styles": [], "assets": [], "symbols": []}


class TestModification:
    def test_page_outline_lists_headings(self):
        handler, model = _model_with_sections()
        outline = handler._page_outline(model)
        assert "Home" in outline
        assert "Overview" in outline

    def test_edit_section_replaces_in_place(self):
        handler, model = _model_with_sections()
        spec = {
            "operation": "edit_section",
            "pageName": "Home",
            "sectionTitle": "Overview",
            "section": {
                "html": (
                    "<section class='ds-section'><h2 class='ds-heading'>"
                    "Today's schedule</h2><p>Updated copy.</p></section>"
                ),
            },
        }
        model, message = handler._apply_modification_spec(model, spec, ["Home"], None)
        assert "Updated" in message
        outline = handler._page_outline(model)
        assert "Today's schedule" in outline
        assert "Overview" not in outline

    def test_edit_section_missing_target_is_safe(self):
        handler, model = _model_with_sections()
        spec = {
            "operation": "edit_section",
            "pageName": "Home",
            "sectionTitle": "Nonexistent",
            "section": {"html": HTML_SECTION},
        }
        before = handler._page_outline(model)
        model, message = handler._apply_modification_spec(model, spec, ["Home"], None)
        assert "couldn't find" in message
        assert handler._page_outline(model) == before

    def test_batch_applies_multiple_operations(self, monkeypatch):
        handler, model = _model_with_sections()

        batch = GUIModificationBatchSpec(operations=[
            GUIModificationSpec(operation="rename_page", pageName="Home", newPageName="Overview"),
            GUIModificationSpec(operation="rename_section", pageName="Overview",
                                sectionTitle="Overview", newSectionTitle="Daily summary"),
        ])

        def fake_structured(prompt, schema, **kwargs):
            assert schema is GUIModificationBatchSpec
            assert "CURRENT APP" in kwargs.get("system_prompt", "")
            return batch

        monkeypatch.setattr(handler, "predict_structured", fake_structured)
        result = handler.generate_modification(
            "please adjust several things", current_model=model,
            raw_request="please adjust several things",
        )
        assert result["action"] == "modify_model"
        names = [p["name"] for p in result["model"]["pages"]]
        assert names == ["Overview"]
        assert "Daily summary" in handler._page_outline(result["model"])
        assert result["message"].count("- ") == 2  # combined two-line message

    def test_llm_failure_keeps_model(self, monkeypatch):
        handler, model = _model_with_sections()

        def fake_structured(*args, **kwargs):
            raise ValueError("no parse")

        monkeypatch.setattr(handler, "predict_structured", fake_structured)
        result = handler.generate_modification(
            "please adjust several things", current_model=model,
            raw_request="please adjust several things",
        )
        assert result["action"] == "modify_model"
        assert result["model"]["pages"], "original model must be preserved"


class TestThemeMerging:
    def test_overrides_merge_onto_preset(self):
        tokens = merge_theme_overrides("default", {"primary": "#111111"})
        assert tokens["palette"]["primary"] == "#111111"
        assert tokens["palette"]["background"]  # preset value survives

    def test_unsafe_values_rejected(self):
        tokens = merge_theme_overrides("default", {
            "primary": "url(javascript:alert(1))",
            "radius": "12px; } body { display:none",
        })
        assert tokens["palette"]["primary"] == "#2563eb"  # preset kept
        assert tokens["radius"] == "8px"

    def test_rules_built_from_custom_tokens(self):
        tokens = merge_theme_overrides("default", {"heroBackground": "#000000"})
        rules = stylesheet_rules_from_tokens(tokens)
        hero = [r for r in rules if r.get("selectors") == [{"name": "ds-hero"}]][0]
        assert hero["style"]["background"] == "#000000"


class TestFreeAuthorship:
    def test_llm_css_lands_in_model_styles(self, monkeypatch):
        handler = GUINoCodeDiagramHandler(None)
        spec = _authored_spec()
        spec = spec.model_copy(update={
            "css": ".app-hero { background: linear-gradient(90deg, #111, #333); padding: 4rem 2rem; }",
        })
        monkeypatch.setattr(
            handler, "predict_two_pass_structured", lambda **kw: spec,
        )
        result = handler.generate_complete_system("create a patient portal")
        blob = str(result["model"]["styles"])
        assert ".app-hero" in blob
        assert "linear-gradient" in blob

    def test_malicious_css_is_dropped_not_fatal(self, monkeypatch):
        handler = GUINoCodeDiagramHandler(None)
        spec = _authored_spec()
        spec = spec.model_copy(update={
            "css": ".x { background: url(http://evil.example/x.png) }",
        })
        monkeypatch.setattr(
            handler, "predict_two_pass_structured", lambda **kw: spec,
        )
        result = handler.generate_complete_system("create a patient portal")
        assert result["action"] == "inject_complete_system"
        assert "evil.example" not in str(result["model"]["styles"])

    def test_nav_header_wears_theme_tokens(self):
        from diagram_handlers.types.gui_design_system import theme_tokens
        tokens = theme_tokens("finance")
        nav = _nav_header_component(
            page_names=["Home", "Trades"], active_page="Home",
            project_name="Meridian", tokens=tokens,
        )
        blob = str(nav)
        assert "#0e5c43" in blob      # finance primary on the active link
        assert "#2563eb" not in blob  # the old hardcoded blue is gone
        # The font must be the THEME's stack (which now leads with the shipped
        # Inter variable font), not a hardcoded literal.
        assert tokens["font"] in blob

    def test_wrapper_wears_theme_background(self, monkeypatch):
        handler = GUINoCodeDiagramHandler(None)
        monkeypatch.setattr(
            handler, "predict_two_pass_structured",
            lambda **kw: _authored_spec(domain="finance"),
        )
        result = handler.generate_complete_system("a trading terminal")
        wrapper = result["model"]["pages"][0]["frames"][0]["component"]
        assert wrapper["style"]["background-color"] == "#eef1f0"  # finance bg

    def test_parse_llm_css_media_queries(self):
        rules = _parse_llm_css(
            "@media (max-width: 600px) { .app-x { display: none; } } .app-x { color: red; }"
        )
        kinds = sorted(r.get("atRuleType", "plain") for r in rules)
        assert kinds == ["media", "plain"]
