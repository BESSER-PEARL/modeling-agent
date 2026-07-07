"""Tests for the LLM-HTML → GrapesJS component-tree converter (Phase 1).

Verifies the invariants the GUI editor depends on: tag identity is preserved
(never collapsed to div), inline ``style`` becomes a css dict, text content is
preserved and NOT ``type``-stamped, markup is sanitised (script/style/on*/
external href+src stripped while ``data:``/``#`` survive), data-bound-widget
spoofing is rejected, unknown tags drop but keep their children, nesting is
correct, and ``<!--WIDGET:x-->`` / ``data-widget-slot`` slots are locatable and
replaceable.

Pure-function style — no LLM client is touched.
"""

import pytest

from diagram_handlers.types.gui_html_converter import (
    html_to_components,
    find_widget_slots,
    replace_widget_slot,
    WIDGET_SLOT_TYPE,
)


def _first(nodes):
    """Return the single top-level node (fails loudly if there isn't one)."""
    assert len(nodes) == 1, f"expected 1 top-level node, got {len(nodes)}: {nodes}"
    return nodes[0]


# ---------------------------------------------------------------------------
# Tag identity + text content
# ---------------------------------------------------------------------------

def test_heading_keeps_tag_identity_not_div():
    node = _first(html_to_components("<h2>Analytics</h2>"))
    assert node["tagName"] == "h2"          # NOT coerced to div
    assert node["content"] == "Analytics"   # text preserved
    assert "type" not in node               # frontend stamps type, not us


@pytest.mark.parametrize("tag", ["p", "span", "a", "li", "label", "blockquote"])
def test_editable_text_tags_are_bare_and_not_type_stamped(tag):
    node = _first(html_to_components(f"<{tag}>Hi</{tag}>"))
    assert node["tagName"] == tag
    assert node["content"] == "Hi"
    assert "type" not in node


def test_text_preserved_as_textnode_for_mixed_content():
    node = _first(html_to_components("<p>Hello <strong>world</strong></p>"))
    assert node["tagName"] == "p"
    assert "type" not in node
    kids = node["components"]
    # A textnode carries the leading text; the <strong> keeps its own identity.
    assert kids[0] == {"type": "textnode", "content": "Hello "}
    assert kids[1]["tagName"] == "strong"
    assert kids[1]["content"] == "world"


def test_whitespace_between_tags_is_dropped():
    node = _first(html_to_components("<ul>\n  <li>A</li>\n  <li>B</li>\n</ul>"))
    assert node["tagName"] == "ul"
    assert [c["tagName"] for c in node["components"]] == ["li", "li"]
    assert [c["content"] for c in node["components"]] == ["A", "B"]


# ---------------------------------------------------------------------------
# Inline style parsing
# ---------------------------------------------------------------------------

def test_inline_style_parsed_into_dict():
    node = _first(html_to_components(
        '<div style="background: #fff; padding: 12px 24px; color:#1e293b"></div>'
    ))
    assert node["style"] == {
        "background": "#fff",
        "padding": "12px 24px",
        "color": "#1e293b",
    }


def test_style_value_with_url_and_colon_survives():
    node = _first(html_to_components(
        '<div style="background-image: url(data:image/png;base64,AAAA); margin:0"></div>'
    ))
    # First-colon split keeps the url() value; ;base64 inside url() is not a
    # declaration separator.
    assert node["style"]["background-image"] == "url(data:image/png;base64,AAAA)"
    assert node["style"]["margin"] == "0"


# ---------------------------------------------------------------------------
# class / id / data-* retention
# ---------------------------------------------------------------------------

def test_class_id_and_data_attrs_kept():
    node = _first(html_to_components(
        '<section class="hero" id="top" data-role="banner">x</section>'
    ))
    assert node["attributes"]["class"] == "hero"
    assert node["attributes"]["id"] == "top"
    assert node["attributes"]["data-role"] == "banner"


# ---------------------------------------------------------------------------
# Sanitisation
# ---------------------------------------------------------------------------

def test_script_and_style_elements_are_stripped():
    nodes = html_to_components(
        "<div><script>alert(1)</script><style>.x{color:red}</style><p>ok</p></div>"
    )
    node = _first(nodes)
    # Only the <p> survives; script/style content is gone entirely.
    assert [c["tagName"] for c in node["components"]] == ["p"]
    assert node["components"][0]["content"] == "ok"
    assert "alert" not in str(nodes)
    assert "color:red" not in str(nodes)


def test_event_handler_attributes_stripped():
    node = _first(html_to_components(
        '<button onclick="steal()" onmouseover="x()" class="cta">Go</button>'
    ))
    attrs = node.get("attributes", {})
    assert "onclick" not in attrs
    assert "onmouseover" not in attrs
    assert attrs["class"] == "cta"


def test_external_href_and_src_dropped_but_element_kept():
    a = _first(html_to_components('<a href="https://evil.example/x">link</a>'))
    assert a["tagName"] == "a"
    assert "href" not in a.get("attributes", {})   # external dropped
    assert a["content"] == "link"                  # element + text kept

    img = _first(html_to_components('<img src="https://cdn.example/a.png">'))
    assert img["tagName"] == "img"
    assert "src" not in img.get("attributes", {})


def test_data_uri_and_anchor_href_src_kept():
    a = _first(html_to_components('<a href="#section">jump</a>'))
    assert a["attributes"]["href"] == "#section"

    data_uri = "data:image/png;base64,iVBORw0KGgo="
    img = _first(html_to_components(f'<img src="{data_uri}">'))
    assert img["attributes"]["src"] == data_uri


def test_javascript_uri_dropped():
    a = _first(html_to_components('<a href="javascript:alert(1)">x</a>'))
    assert "href" not in a.get("attributes", {})


# ---------------------------------------------------------------------------
# Widget-spoof guard
# ---------------------------------------------------------------------------

def test_widget_type_and_data_source_spoof_rejected():
    node = _first(html_to_components(
        '<div data-gjs-type="table" data-source="cls-1" '
        'series="[]" columns="[]" class="fake">rows</div>'
    ))
    attrs = node.get("attributes", {})
    # A plain div, NOT a data-bound widget: no spoofing attrs, no top-level type.
    assert node["tagName"] == "div"
    assert node.get("type") is None
    assert "data-gjs-type" not in attrs
    assert "data-source" not in attrs
    assert "series" not in attrs
    assert "columns" not in attrs
    assert attrs["class"] == "fake"


def test_plain_table_is_allowed_as_markup():
    # A real (non data-bound) HTML table keeps tag identity and never becomes a
    # widget (which would carry type:"table", not tagName:"table").
    node = _first(html_to_components(
        "<table><thead><tr><th>H</th></tr></thead>"
        "<tbody><tr><td>1</td></tr></tbody></table>"
    ))
    assert node["tagName"] == "table"
    assert node.get("type") is None
    assert node["components"][0]["tagName"] == "thead"


def test_form_controls_survive_with_attributes():
    # Regression: an LLM-authored form must keep its input/select/textarea
    # controls (they used to be flattened away, leaving labels with no fields).
    form = _first(html_to_components(
        '<form class="ds-form">'
        '<div class="ds-field"><label for="n">Name</label>'
        '<input class="ds-input" type="text" name="n" placeholder="Jane" /></div>'
        '<div class="ds-field"><label for="r">Reason</label>'
        '<textarea class="ds-input" name="r" rows="3"></textarea></div>'
        '<div class="ds-field"><label for="o">Office</label>'
        '<select class="ds-input" name="o"><option>North</option><option>South</option></select></div>'
        '<button class="ds-btn ds-btn-primary" type="submit">Submit</button>'
        '</form>'
    ))
    assert form["tagName"] == "form"
    tags = []

    def _walk(n):
        if isinstance(n, dict):
            if n.get("tagName"):
                tags.append(n["tagName"])
            for c in n.get("components") or []:
                _walk(c)

    _walk(form)
    assert tags.count("input") == 1
    assert tags.count("textarea") == 1
    assert tags.count("select") == 1
    assert tags.count("option") == 2
    assert tags.count("label") == 3
    # the input is a void element and preserves its benign attributes
    inp = next(c for c in _iter(form) if c.get("tagName") == "input")
    assert inp["attributes"]["type"] == "text"
    assert inp["attributes"]["name"] == "n"
    assert inp["attributes"]["placeholder"] == "Jane"
    assert inp["attributes"]["class"] == "ds-input"
    assert "components" not in inp  # void: no children frame left open


def _iter(node):
    yield node
    for c in node.get("components") or []:
        yield from _iter(c)


# ---------------------------------------------------------------------------
# Unknown / unsafe tag handling
# ---------------------------------------------------------------------------

def test_unknown_tag_dropped_but_children_kept():
    nodes = html_to_components(
        "<marquee><p>keep me</p><span>and me</span></marquee>"
    )
    # <marquee> is dropped; its safe children are flattened up to the top level.
    assert [n["tagName"] for n in nodes] == ["p", "span"]
    assert nodes[0]["content"] == "keep me"
    assert nodes[1]["content"] == "and me"


def test_unknown_tag_flattens_into_surrounding_parent():
    node = _first(html_to_components(
        "<section><custom-el><h3>Title</h3></custom-el></section>"
    ))
    assert node["tagName"] == "section"
    # The <h3> survives as a direct child of <section>, custom-el gone.
    assert [c["tagName"] for c in node["components"]] == ["h3"]


# ---------------------------------------------------------------------------
# Nesting + svg
# ---------------------------------------------------------------------------

def test_nested_structure_is_correct():
    node = _first(html_to_components(
        '<section class="hero">'
        "<h1>Welcome</h1>"
        '<div class="cta"><button>Sign up</button></div>'
        "</section>"
    ))
    assert node["tagName"] == "section"
    assert node["attributes"]["class"] == "hero"
    h1, div = node["components"]
    assert h1["tagName"] == "h1" and h1["content"] == "Welcome"
    assert div["tagName"] == "div" and div["attributes"]["class"] == "cta"
    assert div["components"][0]["tagName"] == "button"
    assert div["components"][0]["content"] == "Sign up"


def test_inline_svg_icon_preserved_with_viewbox_case_restored():
    node = _first(html_to_components(
        '<svg viewBox="0 0 24 24"><path d="M4 4h16"></path></svg>'
    ))
    assert node["tagName"] == "svg"
    # html.parser lowercases attribute names; viewBox case is restored for svg.
    assert node["attributes"]["viewBox"] == "0 0 24 24"
    assert node["components"][0]["tagName"] == "path"
    assert node["components"][0]["attributes"]["d"] == "M4 4h16"


# ---------------------------------------------------------------------------
# Widget slots
# ---------------------------------------------------------------------------

def test_widget_comment_emits_locatable_marker():
    nodes = html_to_components(
        "<section><h2>Sales</h2><!--WIDGET:sales_chart--></section>"
    )
    section = _first(nodes)
    marker = section["components"][1]
    assert marker["type"] == WIDGET_SLOT_TYPE
    assert marker["attributes"]["data-widget-slot"] == "sales_chart"
    assert find_widget_slots(nodes) == ["sales_chart"]


def test_widget_slot_div_emits_marker():
    nodes = html_to_components('<div data-widget-slot="kpis"></div>')
    marker = _first(nodes)
    assert marker["type"] == WIDGET_SLOT_TYPE
    assert marker["attributes"]["data-widget-slot"] == "kpis"
    assert find_widget_slots(nodes) == ["kpis"]


def test_replace_widget_slot_swaps_in_widget_and_is_non_mutating():
    nodes = html_to_components(
        "<main><h2>Dashboard</h2><!--WIDGET:table1--></main>"
    )
    widget = {"type": "table", "attributes": {"data-source": "cls-book"}}

    swapped = replace_widget_slot(nodes, "table1", widget)

    new_main = swapped[0]
    assert new_main["components"][0]["tagName"] == "h2"
    # The marker is gone, the real typed widget is spliced in its place.
    assert new_main["components"][1] == widget
    assert find_widget_slots(swapped) == []

    # Original tree untouched (deep, non-mutating).
    assert find_widget_slots(nodes) == ["table1"]


def test_replace_widget_slot_deep_copies_widget_per_slot():
    nodes = html_to_components(
        "<div><!--WIDGET:w--></div><div><!--WIDGET:w--></div>"
    )
    widget = {"type": "metric-card", "attributes": {}}
    swapped = replace_widget_slot(nodes, "w", widget)

    first = swapped[0]["components"][0]
    second = swapped[1]["components"][0]
    assert first == widget and second == widget
    assert first is not second           # distinct deepcopies
    assert first is not widget


def test_find_widget_slots_recurses_and_returns_ids_in_order():
    nodes = html_to_components(
        "<main>"
        '<section><!--WIDGET:a--></section>'
        '<div data-widget-slot="b"></div>'
        "</main>"
    )
    assert find_widget_slots(nodes) == ["a", "b"]


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_empty_or_non_string_input_returns_empty_list():
    assert html_to_components("") == []
    assert html_to_components("   ") == []
    assert html_to_components(None) == []  # type: ignore[arg-type]


def test_void_elements_do_not_swallow_following_siblings():
    node = _first(html_to_components("<div><br><hr><p>after</p></div>"))
    tags = [c["tagName"] for c in node["components"]]
    assert tags == ["br", "hr", "p"]
    assert node["components"][2]["content"] == "after"
