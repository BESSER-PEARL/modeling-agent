"""Tests for GUI (GUINoCodeDiagram) modification operations.

Covers the bug where modifying an existing GUI always failed with
"I couldn't process that GUI modification". The root cause was a typeless
``Any`` field in ``GUISampleDataPoint`` that OpenAI strict structured outputs
rejected, breaking every ``generate_modification`` call.

These tests exercise:
- the deterministic fast-paths (rename page/section, recolor, reorder) which
  apply without any LLM call,
- LLM-spec application via ``_apply_modification_spec``,
- the safety guarantee that a failed modification never empties the model.
"""

import copy
import pytest

from diagram_handlers.types.gui_nocode_diagram_handler import (
    GUINoCodeDiagramHandler,
    _ensure_page_wrapper,
    _iter_section_components,
    _section_label,
)
from diagram_handlers.core.base_handler import LLMPredictionError
from schemas.gui_diagram import GUISampleDataPoint, GUIModificationSpec


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler() -> GUINoCodeDiagramHandler:
    # Construct without __init__ so we never touch the LLM client.
    return GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)


@pytest.fixture
def model(handler) -> dict:
    """A realistic two-page GUI model built via the handler's page parser."""
    page_mgmt = handler._parse_page_spec(
        {
            "name": "Page Management",
            "sections": [
                {"type": "hero", "title": "Welcome", "body": "Hi", "ctaLabel": "Go"},
                {"type": "content", "title": "Other card", "body": "x"},
                {"type": "content", "title": "Recent edits", "body": "stuff"},
                {"type": "footer", "title": "Proj"},
            ],
        },
        None,
        all_page_names=["Page Management", "Home"],
        project_name="App",
    )
    home = handler._parse_page_spec(
        {
            "name": "Home",
            "sections": [
                {"type": "hero", "title": "Home Hero", "body": "h", "ctaLabel": "S"},
                {"type": "content", "title": "Cards", "body": "a"},
            ],
        },
        None,
        all_page_names=["Page Management", "Home"],
        project_name="App",
    )
    return {
        "pages": [page_mgmt, home],
        "styles": [],
        "assets": [],
        "symbols": [],
        "version": "0.21.13",
    }


def _page_names(m):
    return [p["name"] for p in m["pages"]]


def _section_labels(m, page_name):
    for page in m["pages"]:
        if page["name"] == page_name:
            wrapper = _ensure_page_wrapper(page)
            return [_section_label(c) for _pl, _i, c in _iter_section_components(wrapper)]
    return None


def _page_bg(m, idx=0):
    return m["pages"][idx]["frames"][0]["component"]["style"].get("background-color")


# ---------------------------------------------------------------------------
# Schema regression: the field that broke everything must stay concrete
# ---------------------------------------------------------------------------


class TestSchemaStrictSafe:
    def test_sample_data_accepts_int_and_string(self):
        assert GUISampleDataPoint(name="Q1", value=100).value == 100
        assert GUISampleDataPoint(name="Q1", value="100").value == "100"

    def test_modification_schema_has_no_typeless_property(self):
        """Every leaf property must carry a concrete type / anyOf / enum.

        A typeless property (the old ``value: Any``) is what OpenAI strict
        structured outputs reject with a 400, breaking every modify call.
        """
        schema = GUIModificationSpec.model_json_schema()

        def walk(node):
            offenders = []
            if isinstance(node, dict):
                props = node.get("properties")
                if isinstance(props, dict):
                    for key, val in props.items():
                        if isinstance(val, dict) and not any(
                            k in val for k in ("type", "anyOf", "$ref", "enum", "allOf")
                        ):
                            offenders.append(key)
                for val in node.values():
                    offenders += walk(val)
            elif isinstance(node, list):
                for val in node:
                    offenders += walk(val)
            return offenders

        assert walk(schema) == []


# ---------------------------------------------------------------------------
# Deterministic fast-paths (no LLM)
# ---------------------------------------------------------------------------


class TestDeterministicModify:
    def test_rename_page_multiword(self, handler, model):
        # Bug #29 — the exact failing request.
        res = handler._try_deterministic_modify(
            model, "Rename Page Management to Page Management Test", _page_names(model)
        )
        assert res is not None
        applied, message = res
        assert "Page Management Test" in _page_names(applied)
        assert "Page Management" not in _page_names(applied)
        assert "renamed" in message.lower()

    def test_recolor_whole_gui(self, handler, model):
        # Bug #5 — "change the GUI color to red".
        res = handler._try_deterministic_modify(
            model, "could you change the GUI color to red?", _page_names(model)
        )
        assert res is not None
        applied, _ = res
        assert _page_bg(applied) == "#e74c3c"

    def test_recolor_specific_section(self, handler, model):
        res = handler._try_deterministic_modify(
            model, "change the hero color to blue", _page_names(model)
        )
        assert res is not None
        applied, message = res
        assert "hero" in message.lower()
        # Hero section now has a solid background-color, gradient removed.
        wrapper = _ensure_page_wrapper(applied["pages"][0])
        hero = next(
            c
            for _pl, _i, c in _iter_section_components(wrapper)
            if "assistant-hero" in c.get("attributes", {}).get("class", "")
        )
        assert hero["style"].get("background-color") == "#2563eb"
        assert "background" not in hero["style"]

    def test_reorder_section_to_top(self, handler, model):
        # Bug #24 — align the "Recent edits" card.
        before = _section_labels(model, "Page Management")
        assert before.index("Other card") < before.index("Recent edits")
        res = handler._try_deterministic_modify(
            model, "move the Recent edits card to the top", _page_names(model)
        )
        assert res is not None
        applied, _ = res
        after = _section_labels(applied, "Page Management")
        assert after.index("Recent edits") < after.index("Other card")

    def test_rename_section(self, handler, model):
        res = handler._try_deterministic_modify(
            model, "rename Recent edits to Latest changes", _page_names(model)
        )
        assert res is not None
        applied, _ = res
        labels = _section_labels(applied, "Page Management")
        assert "Latest changes" in labels
        assert "Recent edits" not in labels

    @pytest.mark.parametrize(
        "request_text",
        [
            "please make the dashboard fancier somehow",
            "add a table of users",
            "make the page more modern",
        ],
    )
    def test_open_ended_requests_fall_through_to_llm(self, handler, model, request_text):
        # These must NOT be misread as a deterministic recolor/rename.
        res = handler._try_deterministic_modify(model, request_text, _page_names(model))
        assert res is None


# ---------------------------------------------------------------------------
# LLM-spec application
# ---------------------------------------------------------------------------


class TestApplySpec:
    def test_remove_section(self, handler, model):
        spec = {"operation": "remove_section", "pageName": "Page Management",
                "sectionTitle": "Recent edits"}
        applied, message = handler._apply_modification_spec(
            model, spec, _page_names(model), None
        )
        assert "Recent edits" not in _section_labels(applied, "Page Management")
        assert "removed" in message.lower()

    def test_add_page(self, handler, model):
        spec = {"operation": "add_page", "pageName": "Page Management",
                "newPageName": "Settings"}
        applied, _ = handler._apply_modification_spec(
            model, spec, _page_names(model), None
        )
        assert "Settings" in _page_names(applied)

    def test_remove_last_page_is_refused(self, handler):
        single = handler._parse_page_spec(
            {"name": "Home", "sections": [{"type": "hero", "title": "T"}]},
            None, all_page_names=["Home"], project_name="App",
        )
        m = {"pages": [single], "styles": [], "assets": [], "symbols": [],
             "version": "0.21.13"}
        spec = {"operation": "remove_page", "pageName": "Home"}
        applied, message = handler._apply_modification_spec(m, spec, ["Home"], None)
        # SAFETY: the last page is never removed.
        assert _page_names(applied) == ["Home"]
        assert "unchanged" in message.lower()


# ---------------------------------------------------------------------------
# Safety: a failed modification never empties / destroys the model
# ---------------------------------------------------------------------------


class TestModificationSafety:
    def test_llm_failure_preserves_model(self, handler, model):
        def boom(*args, **kwargs):
            raise LLMPredictionError("simulated 400 BadRequest")

        handler.predict_structured = boom
        original_labels = _section_labels(model, "Page Management")

        # An open-ended request that the deterministic path won't handle, so
        # it reaches the (now failing) LLM path.
        result = handler.generate_modification(
            "please make the dashboard fancier somehow",
            copy.deepcopy(model),
            raw_request="please make the dashboard fancier somehow",
            class_metadata=None,
        )

        assert result["action"] == "modify_model"
        # Model is intact — same pages and sections, nothing emptied.
        assert len(result["model"]["pages"]) == len(model["pages"])
        assert _section_labels(result["model"], "Page Management") == original_labels
        # Message is informative, not the old generic "rephrase".
        assert "unchanged" in result["message"].lower()

    def test_llm_spec_path_applies_change(self, handler, model):
        class _Parsed:
            def model_dump(self):
                # The modify path now consumes a BATCH of operations.
                return {
                    "operations": [{
                        "operation": "rename_section",
                        "pageName": "Page Management",
                        "sectionTitle": "Recent edits",
                        "newSectionTitle": "Latest",
                    }],
                }

        handler.predict_structured = lambda *a, **k: _Parsed()
        result = handler.generate_modification(
            "do that tricky thing",
            copy.deepcopy(model),
            raw_request="do that tricky thing",
            class_metadata=None,
        )
        assert result["action"] == "modify_model"
        labels = _section_labels(result["model"], "Page Management")
        assert "Latest" in labels
        assert "Recent edits" not in labels
