"""Tests for the per-domain GUI design system (Phase 2).

The design system turns the always-empty GrapesJS ``styles[]`` array into a
real, per-domain visual identity. These tests pin down the three public
contracts the rest of the rebuild depends on:

1. Every domain resolves to a theme carrying the full set of design tokens.
2. ``stylesheet_rules(domain)`` emits GrapesJS rule objects in exactly the
   project-data ``styles[]`` shape ``loadProjectData`` consumes.
3. ``block_exemplars(domain)`` yields ``.ds-*``-based HTML with widget
   placeholders in the data blocks, and ``pick_domain`` routes sample requests
   to the right domain.

Pure functions — no LLM client is touched.
"""

import pytest

from diagram_handlers.types.gui_design_system import (
    DOMAINS,
    THEMES,
    stylesheet_rules,
    block_exemplars,
    pick_domain,
    theme_tokens,
)


# ---------------------------------------------------------------------------
# Themes / tokens
# ---------------------------------------------------------------------------

_PALETTE_KEYS = {
    "primary", "secondary", "accent", "surface",
    "background", "text", "muted", "border",
}
_TYPE_KEYS = {"h1", "h2", "h3", "body", "small"}


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_domain_has_a_theme_with_token_fields(domain):
    tokens = theme_tokens(domain)

    # Palette — all eight colour roles present and non-empty strings.
    palette = tokens["palette"]
    assert _PALETTE_KEYS.issubset(palette.keys())
    assert all(isinstance(v, str) and v for v in palette.values())

    # Type scale — h1..h3 + body + small, each with a size and a weight.
    type_scale = tokens["type_scale"]
    assert _TYPE_KEYS.issubset(type_scale.keys())
    for step in _TYPE_KEYS:
        assert "size" in type_scale[step]
        assert "weight" in type_scale[step]

    # Spacing scale, radius, shadow and a font stack all present.
    assert {"xs", "sm", "md", "lg", "xl"}.issubset(tokens["spacing"].keys())
    assert isinstance(tokens["radius"], str) and tokens["radius"]
    assert isinstance(tokens["shadow"], str) and tokens["shadow"]
    assert isinstance(tokens["font"], str) and tokens["font"]


def test_theme_tokens_returns_a_copy():
    """Callers must not be able to mutate the shared THEMES table."""
    tokens = theme_tokens("finance")
    tokens["palette"]["primary"] = "#000000"
    assert THEMES["finance"]["palette"]["primary"] != "#000000"


def test_theme_tokens_unknown_domain_falls_back_to_default():
    assert theme_tokens("nonsense")["name"] == "default"


def test_fonts_are_system_only_no_external_webfonts():
    """CSP blocks webfonts — no Google Fonts / @import / url() references."""
    for domain in DOMAINS:
        tokens = theme_tokens(domain)
        for key in ("font", "font_heading", "font_mono"):
            stack = tokens[key].lower()
            assert "http" not in stack
            assert "url(" not in stack
            assert "@import" not in stack


# ---------------------------------------------------------------------------
# stylesheet_rules — GrapesJS project-data shape
# ---------------------------------------------------------------------------

def _is_class_rule(rule):
    sel = rule.get("selectors")
    return (
        isinstance(sel, list)
        and len(sel) == 1
        and isinstance(sel[0], dict)
        and isinstance(sel[0].get("name"), str)
        and sel[0]["name"]
    )


def _is_selectors_add_rule(rule):
    return isinstance(rule.get("selectorsAdd"), str) and rule["selectorsAdd"]


def _style_is_flat_str_dict(style):
    return (
        isinstance(style, dict)
        and bool(style)
        and all(isinstance(k, str) for k in style)
        and all(isinstance(v, str) for v in style.values())
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_stylesheet_rules_returns_nonempty_valid_rule_objects(domain):
    rules = stylesheet_rules(domain)
    assert isinstance(rules, list) and rules

    for rule in rules:
        # Each rule is either a class rule or a selectorsAdd (base/compound) rule.
        assert _is_class_rule(rule) or _is_selectors_add_rule(rule), rule
        # ``style`` is always a flat dict[str, str].
        assert _style_is_flat_str_dict(rule["style"]), rule


@pytest.mark.parametrize("domain", DOMAINS)
def test_stylesheet_rules_include_ds_card_and_ds_hero(domain):
    class_names = {
        rule["selectors"][0]["name"]
        for rule in stylesheet_rules(domain)
        if _is_class_rule(rule)
    }
    assert "ds-card" in class_names
    assert "ds-hero" in class_names


@pytest.mark.parametrize("domain", DOMAINS)
def test_stylesheet_rules_cover_the_core_component_classes(domain):
    class_names = {
        rule["selectors"][0]["name"]
        for rule in stylesheet_rules(domain)
        if _is_class_rule(rule)
    }
    expected = {
        "ds-page", "ds-container", "ds-section", "ds-hero", "ds-heading",
        "ds-card", "ds-grid-2", "ds-grid-3", "ds-kpi", "ds-kpi-value",
        "ds-kpi-label", "ds-table-wrap", "ds-footer", "ds-btn",
        "ds-btn-primary", "ds-nav", "ds-notice", "ds-badge", "ds-field",
        "ds-label", "ds-input",
    }
    assert expected.issubset(class_names)


def test_stylesheet_rules_include_a_base_reset_rule():
    rules = stylesheet_rules("default")
    base = [r for r in rules if _is_selectors_add_rule(r)]
    assert base, "expected at least one selectorsAdd base/token rule"
    assert any(r["selectorsAdd"] == "*" for r in base)


def test_stylesheet_rules_differ_between_domains():
    """The whole point: two domains must not share one skin."""
    gov = {r["style"].get("background-color") for r in stylesheet_rules("government")}
    startup = {r["style"].get("background-color") for r in stylesheet_rules("startup")}
    assert gov != startup


# ---------------------------------------------------------------------------
# block_exemplars
# ---------------------------------------------------------------------------

_BLOCK_KEYS = {"hero", "kpi_row", "table_card", "chart_card", "footer", "form", "notice"}


@pytest.mark.parametrize("domain", DOMAINS)
def test_block_exemplars_are_nonempty_html_referencing_ds_classes(domain):
    blocks = block_exemplars(domain)
    assert _BLOCK_KEYS.issubset(blocks.keys())
    for key, html in blocks.items():
        assert isinstance(html, str) and html.strip(), key
        assert "ds-" in html, key
        # No external images or webfonts (CSP-blocked).
        assert "<img" not in html.lower(), key
        assert "@import" not in html.lower(), key


@pytest.mark.parametrize("domain", DOMAINS)
def test_data_blocks_contain_widget_placeholder(domain):
    blocks = block_exemplars(domain)
    assert "<!--WIDGET:" in blocks["table_card"]
    assert "<!--WIDGET:" in blocks["chart_card"]


@pytest.mark.parametrize("domain", DOMAINS)
def test_form_exemplar_uses_form_control_classes(domain):
    form = block_exemplars(domain)["form"]
    assert "ds-field" in form
    assert "ds-label" in form
    assert "ds-input" in form


# ---------------------------------------------------------------------------
# pick_domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("government residency certificate", "government"),
        ("trading dashboard", "finance"),
        ("patient portal", "health"),
        ("SaaS landing", "startup"),
        ("a simple app to track my books", "default"),
    ],
)
def test_pick_domain_maps_sample_requests(text, expected):
    assert pick_domain(text) == expected


def test_pick_domain_handles_empty_and_invalid_input():
    assert pick_domain("") == "default"
    assert pick_domain("   ") == "default"
    assert pick_domain(None) == "default"  # type: ignore[arg-type]


def test_pick_domain_result_is_always_a_known_domain():
    for text in ("hospital appointment booking", "invoice ledger", "random words", ""):
        assert pick_domain(text) in DOMAINS
