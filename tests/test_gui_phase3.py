"""Phase 3 tests for the GUINoCodeDiagram handler — LLM-authored HTML sections
and structured widget binding.

Phase 3 is the core rewrite: the LLM now AUTHORS rich themed HTML per section
(using the ``.ds-*`` design system) instead of picking from a closed type-menu
that Python renders in one hardcoded skin. These tests pin down the invariants
that keep that risky change safe:

* an ``html`` section becomes an editable component def-tree that keeps its
  ``.ds-*`` classes, a stable section class, and a leading heading (so the
  modification edit-ops can still locate it),
* a ``bind`` section produces the exact recognizer-compatible typed widget
  (``type:'table'`` + ``data-source`` + ``columns``) spliced at the
  ``<!--WIDGET:-->`` slot inside the LLM-authored chrome,
* the picked design ``domain`` populates ``model['styles']`` after assembly,
* unparseable / empty input never crashes — it degrades to a typed builder or
  a themed content box,
* the schema additions stay OpenAI-strict-output compatible, and
* legacy ``type:``-based sections still render via the old builders.

Pure-function style: handlers are built with ``__new__`` so no LLM client is
touched (mirrors tests/test_gui_phase0.py / test_gui_modification.py).
"""

import json

import pytest

from diagram_handlers.types.gui_nocode_diagram_handler import (
    GUINoCodeDiagramHandler,
    _build_section_component,
    _normalize_html_section,
    _collect_data_uri_assets,
    _section_has_heading,
)
from schemas.gui_diagram import (
    SingleGUIElementSpec,
    GUIModificationSpec,
    GUISectionSpec,
    GUIBindSpec,
    SystemGUISpec,
)


@pytest.fixture
def handler() -> GUINoCodeDiagramHandler:
    # Construct without __init__ so we never touch the LLM client.
    return GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)


# Class metadata in the shape ``_resolve_class_binding`` / the typed builders
# expect. Includes ``type`` so it also survives ``format_class_metadata_for_prompt``
# on the complete-system path.
BOOK_METADATA = [
    {
        "id": "cls-book",
        "name": "Book",
        "attributes": [
            {"id": "a-title", "name": "title", "type": "str", "isNumeric": False, "isString": True},
            {"id": "a-pages", "name": "pages", "type": "int", "isNumeric": True, "isString": False},
            {"id": "a-price", "name": "price", "type": "float", "isNumeric": True, "isString": False},
        ],
    }
]


def _blob(node) -> str:
    return json.dumps(node)


def _class_of(node) -> str:
    attrs = node.get("attributes") if isinstance(node, dict) else None
    return attrs.get("class", "") if isinstance(attrs, dict) else ""


# ---------------------------------------------------------------------------
# 1. HTML section → editable def-tree with ds classes + stable class + heading
# ---------------------------------------------------------------------------

def test_html_section_preserves_ds_classes_and_tags():
    section = {
        "html": (
            "<section class='ds-hero'>"
            "<h1 class='ds-heading'>Residency Certificate Services</h1>"
            "<p>Apply for and track official residency certificates online.</p>"
            "<a class='ds-btn ds-btn-primary' href='#'>Start an application</a>"
            "</section>"
        )
    }
    comp = _build_section_component(section, None)
    # Tag identity preserved (never collapsed to a div) and NOT type-stamped.
    assert comp["tagName"] == "section"
    assert "type" not in comp  # the frontend assigns text type, not us
    assert _class_of(comp) == "ds-hero"
    blob = _blob(comp)
    assert "ds-heading" in blob and "ds-btn-primary" in blob
    # Heading survives as editable content, not a widget.
    assert "Residency Certificate Services" in blob
    assert _section_has_heading(comp)


def test_html_section_injects_missing_heading_and_class():
    # LLM omitted BOTH a root class and a leading heading — we inject stable ones
    # so the modification edit-ops (_match_section / _set_section_heading) locate it.
    section = {"title": "Overview", "html": "<div><p>Just a paragraph, no heading.</p></div>"}
    comp = _build_section_component(section, None)
    assert _class_of(comp)  # a stable class was injected
    assert _section_has_heading(comp)
    # The injected heading uses the section title.
    assert "Overview" in _blob(comp)


def test_html_section_wraps_multiple_top_level_nodes_into_one_section():
    # Two sibling nodes must collapse to exactly ONE section node so downstream
    # edit-ops (one section == one node) keep matching.
    nodes = _normalize_html_section(
        [
            {"tagName": "p", "content": "a"},
            {"tagName": "p", "content": "b"},
        ],
        title="Grouped",
    )
    assert nodes["tagName"] == "section"
    assert _class_of(nodes) == "ds-section"
    assert _section_has_heading(nodes)


def test_html_section_parses_through_page_spec(handler):
    """An html section survives _parse_page_spec into a real frame def-tree."""
    page = handler._parse_page_spec(
        {
            "name": "Home",
            "sections": [
                {"html": "<section class='ds-hero'><h1 class='ds-heading'>Hi</h1></section>"},
                {"html": "<section class='ds-section'><h2 class='ds-heading'>Body</h2><p>x</p></section>"},
            ],
        },
        None,
    )
    wrapper = page["frames"][0]["component"]
    blob = _blob(wrapper)
    assert "ds-hero" in blob and "ds-heading" in blob
    # ds-hero is full-width -> hoisted to a top-level component (not inside main).
    top_classes = [_class_of(c) for c in wrapper["components"]]
    assert "ds-hero" in top_classes
    # The non-hero section is centered inside the assistant-main container.
    assert "assistant-main" in top_classes


# ---------------------------------------------------------------------------
# 2. bind section → typed widget spliced at the <!--WIDGET:--> slot
# ---------------------------------------------------------------------------

def test_bind_table_splices_typed_widget_at_slot():
    section = {
        "title": "Recent applications",
        "bind": {"kind": "table", "className": "Book"},
        "html": (
            "<section class='ds-section'>"
            "<h3 class='ds-heading'>Recent applications</h3>"
            "<div class='ds-table-wrap'><!--WIDGET:table--></div>"
            "</section>"
        ),
    }
    comp = _build_section_component(section, BOOK_METADATA)
    blob = _blob(comp)
    # The LLM chrome (ds-section wrapper + ds-table-wrap) is preserved...
    assert _class_of(comp) == "ds-section"
    assert "ds-table-wrap" in blob
    # ...and the recognizer-compatible typed table widget is spliced in.
    assert '"type": "table"' in blob
    assert "data-source" in blob  # bound to the Book class id
    assert "cls-book" in blob
    assert "columns" in blob  # auto-generated column defs from attributes


def test_bind_chart_produces_recognizer_widget():
    section = {
        "title": "By district",
        "bind": {
            "kind": "bar_chart",
            "className": "Book",
            "sampleData": [{"name": "North", "value": 40}, {"name": "South", "value": 65}],
        },
        "html": "<section class='ds-section'><h3 class='ds-heading'>By district</h3><div class='ds-card'><!--WIDGET:chart--></div></section>",
    }
    comp = _build_section_component(section, BOOK_METADATA)
    blob = _blob(comp)
    assert '"type": "bar-chart"' in blob
    assert "series" in blob


def test_bind_without_chrome_card_wraps_widget():
    section = {"title": "KPIs", "bind": {"kind": "table", "className": "Book"}}
    comp = _build_section_component(section, BOOK_METADATA)
    blob = _blob(comp)
    # No chrome -> themed ds-card section wrapping the widget, with a heading.
    assert _class_of(comp) == "ds-section"
    assert "ds-card" in blob
    assert '"type": "table"' in blob
    assert _section_has_heading(comp)


def test_bind_chrome_without_slot_still_renders_widget():
    # LLM forgot the <!--WIDGET:--> marker: the widget must still render.
    section = {
        "title": "Apps",
        "bind": {"kind": "table", "className": "Book"},
        "html": "<section class='ds-section'><h3 class='ds-heading'>Apps</h3></section>",
    }
    comp = _build_section_component(section, BOOK_METADATA)
    blob = _blob(comp)
    assert '"type": "table"' in blob  # not silently dropped
    assert _section_has_heading(comp)


# ---------------------------------------------------------------------------
# 3. domain → model['styles'] populated after generate/assembly
# ---------------------------------------------------------------------------

def test_complete_system_populates_domain_styles(handler):
    handler.predict_two_pass = lambda **kw: json.dumps(
        {
            "projectName": "Residency Portal",
            "domain": "government",
            "pages": [
                {
                    "name": "Home",
                    "sections": [
                        {"html": "<section class='ds-hero'><h1 class='ds-heading'>Residency</h1><p>Apply</p></section>"},
                        {
                            "title": "Applications",
                            "bind": {"kind": "table", "className": "Book"},
                            "html": "<section class='ds-section'><h3 class='ds-heading'>Applications</h3><div class='ds-table-wrap'><!--WIDGET:table--></div></section>",
                        },
                    ],
                }
            ],
        }
    )
    res = handler.generate_complete_system(
        "a government residency certificate portal", class_metadata=BOOK_METADATA
    )
    assert res["action"] == "inject_complete_system"
    styles = res["model"]["styles"]
    assert isinstance(styles, list) and len(styles) > 0
    # Every rule is a real GrapesJS style object (has a "style" mapping).
    assert all(isinstance(r, dict) and "style" in r for r in styles)
    # Government theme injects its ds-* component classes.
    names = {
        s["name"]
        for r in styles
        for s in (r.get("selectors") or [])
        if isinstance(s, dict) and s.get("name")
    }
    assert "ds-card" in names and "ds-hero" in names


def test_domain_falls_back_to_pick_domain_when_spec_omits_it(handler):
    # Spec has no top-level "domain" -> assembly derives it from the request text.
    handler.predict_two_pass = lambda **kw: json.dumps(
        {"projectName": "Bank", "pages": [{"name": "H", "sections": [
            {"html": "<section class='ds-section'><h2 class='ds-heading'>Accounts</h2></section>"}]}]}
    )
    res = handler.generate_complete_system("a fintech banking investment portfolio dashboard")
    styles = res["model"]["styles"]
    assert len(styles) > 0
    # Finance theme carries its signature primary green token somewhere.
    assert any("#0e5c43" in json.dumps(r) for r in styles)


# ---------------------------------------------------------------------------
# 4. unparseable / empty html → graceful fallback (never crash)
# ---------------------------------------------------------------------------

def test_empty_html_falls_back_to_content_box():
    comp = _build_section_component({"html": "   ", "title": "T", "body": "B"}, None)
    # Renders SOMETHING (a themed content box), never crashes/returns empty.
    assert isinstance(comp, dict) and comp.get("tagName") == "section"
    assert "T" in _blob(comp)


def test_html_with_only_script_falls_back():
    # Sanitiser drops <script>; nothing renderable remains -> content-box fallback.
    comp = _build_section_component(
        {"html": "<script>alert(1)</script>", "title": "Safe", "body": "ok"}, None
    )
    assert isinstance(comp, dict) and comp.get("tagName")
    assert "alert" not in _blob(comp)
    assert "Safe" in _blob(comp)


def test_bind_unknown_kind_does_not_crash():
    # An out-of-vocabulary bind kind still yields a renderable widget.
    section = {"title": "X", "bind": {"kind": "table"}, "html": "<section class='ds-section'><h3>X</h3><!--WIDGET:x--></section>"}
    comp = _build_section_component(section, None)
    assert isinstance(comp, dict) and comp.get("tagName")


# ---------------------------------------------------------------------------
# 5. strict-outputs: schema still builds with the new fields
# ---------------------------------------------------------------------------

def test_new_fields_stay_strict_output_compatible():
    """html + bind + domain must not break OpenAI strict structured output —
    a bad shape 400s EVERY GUI generate/modify call."""
    strict = pytest.importorskip("openai.lib._pydantic")
    to_strict = strict.to_strict_json_schema
    to_strict(SingleGUIElementSpec)
    to_strict(GUIModificationSpec)


def test_section_spec_accepts_html_and_bind():
    s = GUISectionSpec.model_validate(
        {
            "html": "<section class='ds-section'><h2>Hi</h2></section>",
            "bind": {"kind": "table", "className": "Book", "columns": ["title", "pages"]},
        }
    )
    assert s.type is None  # type is now optional
    assert s.html.startswith("<section")
    assert isinstance(s.bind, GUIBindSpec)
    assert s.bind.kind == "table" and s.bind.columns == ["title", "pages"]


def test_system_spec_accepts_domain():
    spec = SystemGUISpec.model_validate(
        {"domain": "health", "pages": [{"pageName": "Home", "sections": []}]}
    )
    assert spec.domain == "health"


def test_single_element_with_html_section(handler):
    """generate_single_element must work with an html-only strict section."""
    section = GUISectionSpec.model_validate(
        {"html": "<section class='ds-section'><h2 class='ds-heading'>Note</h2><p>hello</p></section>"}
    )
    comp = _build_section_component(section.model_dump(), None)
    assert comp["tagName"] == "section"
    assert "Note" in _blob(comp)


# ---------------------------------------------------------------------------
# 6. back-compat: legacy type-based sections still render via old builders
# ---------------------------------------------------------------------------

def test_legacy_hero_still_renders():
    comp = _build_section_component(
        {"type": "hero", "title": "Welcome", "body": "Hi", "ctaLabel": "Go"}, None
    )
    assert _class_of(comp) == "assistant-hero"
    assert "Welcome" in _blob(comp)


def test_legacy_content_and_table_still_render():
    content = _build_section_component({"type": "content", "title": "C", "body": "b"}, None)
    assert _class_of(content) == "assistant-content"
    table = _build_section_component({"type": "table", "title": "T", "className": "Book"}, BOOK_METADATA)
    assert '"type": "table"' in _blob(table)


def test_legacy_no_type_defaults_to_content():
    comp = _build_section_component({"title": "Plain", "body": "text"}, None)
    assert _class_of(comp) == "assistant-content"


# ---------------------------------------------------------------------------
# 7. end-to-end: mocked two-pass html+bind spec → full inject payload
# ---------------------------------------------------------------------------

def test_end_to_end_html_and_bind_system(handler):
    handler.predict_two_pass = lambda **kw: json.dumps(
        {
            "projectName": "Care Portal",
            "domain": "health",
            "pages": [
                {
                    "name": "Portal",
                    "sections": [
                        {"html": "<section class='ds-hero'><h1 class='ds-heading'>Your Patient Portal</h1><p>Book care</p></section>"},
                        {
                            "title": "Appointments",
                            "bind": {"kind": "table", "className": "Book"},
                            "html": "<section class='ds-section'><h3 class='ds-heading'>Appointments</h3><div class='ds-table-wrap'><!--WIDGET:table--></div></section>",
                        },
                        {"html": "<footer class='ds-footer'><div class='ds-container'><p>&copy; Lakeside Health</p></div></footer>"},
                    ],
                }
            ],
        }
    )
    res = handler.generate_complete_system("a health patient care portal", class_metadata=BOOK_METADATA)

    assert res["action"] == "inject_complete_system"
    assert res["diagramType"] == "GUINoCodeDiagram"
    model = res["model"]
    # pages present, styles non-empty, assets + symbols present, version stamped.
    assert len(model["pages"]) == 1
    assert isinstance(model["styles"], list) and len(model["styles"]) > 0
    assert isinstance(model["assets"], list)
    assert "symbols" in model and "version" in model

    blob = json.dumps(model["pages"])
    # Authored html survives, the typed table widget is spliced, footer full-width.
    assert "ds-hero" in blob and "ds-footer" in blob
    assert '"type": "table"' in blob and "cls-book" in blob


def test_end_to_end_collects_data_uri_assets():
    pages = [
        {
            "frames": [
                {
                    "component": {
                        "components": [
                            {"tagName": "img", "attributes": {"src": "data:image/png;base64,AAAA"}},
                            {"tagName": "img", "attributes": {"src": "#skip"}},
                        ]
                    }
                }
            ]
        }
    ]
    assets = _collect_data_uri_assets(pages)
    assert assets == [{"type": "image", "src": "data:image/png;base64,AAAA"}]
