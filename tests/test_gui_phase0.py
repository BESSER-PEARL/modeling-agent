"""Phase 0 quick-win tests for the GUINoCodeDiagram handler.

Covers three low-risk, high-visibility fixes to complete-system generation:

1. Truncation salvage — a truncated multi-page JSON keeps its complete pages
   instead of collapsing to the one-line "Welcome" stub (the #1 quality killer
   for ambitious requests).
2. two_column — the recursive ``left``/``right`` sub-sections survive schema
   validation and render both halves (previously dropped by the strict schema,
   leaving empty "Left"/"Content" boxes, issue #10).
3. stats_grid — LLM-provided ``value`` figures are preserved (not discarded,
   issue #7) and class-bound cards spread across DISTINCT numeric fields.

Plus a strict-outputs smoke test: the schema additions must not regress the
OpenAI strict structured-output shape used by ``generate_single_element`` /
``GUIModificationSpec`` (a typeless field there 400s EVERY GUI call).

Pure-function style: handlers are built with ``__new__`` so no LLM client is
touched (see tests/test_gui_modification.py for the established pattern).
"""

import json
import pytest

from diagram_handlers.types.gui_nocode_diagram_handler import (
    GUINoCodeDiagramHandler,
    _salvage_truncated_system,
    _extract_balanced_objects,
    _two_column_component,
    _stats_grid_component,
    _build_section_component,
)
from schemas.gui_diagram import (
    SingleGUIElementSpec,
    GUIModificationSpec,
    GUISectionSpec,
    GUIStatItem,
)


@pytest.fixture
def handler() -> GUINoCodeDiagramHandler:
    # Construct without __init__ so we never touch the LLM client.
    return GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)


# Class metadata in the shape ``_resolve_class_binding`` / ``_pick_data_field``
# expect (mirrors utilities.class_metadata.extract_class_metadata output).
BOOK_METADATA = [
    {
        "id": "cls-book",
        "name": "Book",
        "attributes": [
            {"id": "a-pages", "name": "pages", "isNumeric": True, "isString": False},
            {"id": "a-price", "name": "price", "isNumeric": True, "isString": False},
            {"id": "a-title", "name": "title", "isNumeric": False, "isString": True},
        ],
    }
]


# ---------------------------------------------------------------------------
# Fix 1 — truncation salvage
# ---------------------------------------------------------------------------

# Two fully-emitted pages, then a third page cut off mid-stream. Note the
# braces/brackets embedded inside a string value — the scanner must not let
# them corrupt its depth count.
TRUNCATED_SYSTEM_JSON = """{"projectName": "City Library", "pages": [
  {"name": "Catalog", "sections": [
    {"type": "content", "title": "Browse", "body": "Books {with} [brackets] and a \\" quote"},
    {"type": "footer", "items": ["Hours"]}
  ]},
  {"name": "Members", "sections": [
    {"type": "table", "title": "Members", "className": "Member"}
  ]},
  {"name": "Loans", "sections": [
    {"type": "stats_grid", "stats": [{"label": "Ov"""


def test_extract_balanced_objects_respects_strings_and_drops_partial():
    start = TRUNCATED_SYSTEM_JSON.find("[", TRUNCATED_SYSTEM_JSON.find('"pages"')) + 1
    objs = _extract_balanced_objects(TRUNCATED_SYSTEM_JSON, start)
    # Two complete page objects extracted; the truncated third is dropped.
    assert len(objs) == 2
    assert all(json.loads(o).get("name") for o in objs)


def test_salvage_keeps_complete_pages():
    salvaged = _salvage_truncated_system(TRUNCATED_SYSTEM_JSON)
    assert salvaged is not None
    assert salvaged["projectName"] == "City Library"
    assert [p["name"] for p in salvaged["pages"]] == ["Catalog", "Members"]
    # Result must be clean, complete JSON.
    assert json.loads(json.dumps(salvaged))


def test_salvage_returns_none_when_nothing_recoverable():
    # First (and only) page truncated before any brace closes → unrecoverable.
    txt = '{"projectName": "X", "pages": [ {"name": "Home", "sections": [ {"type'
    assert _salvage_truncated_system(txt) is None
    assert _salvage_truncated_system("not json at all") is None


def test_complete_system_failure_never_injects_welcome_stub(handler, monkeypatch):
    """The structured create path errors NON-DESTRUCTIVELY on failure.

    The old free-text path truncated mid-JSON and (in the worst case) replaced
    the user's screens with a Welcome stub presented as success. Structured
    output can't emit malformed JSON, and a hard failure now returns a plain
    error message with NO model payload — the existing screens are untouched.
    """
    def _boom(**kwargs):
        raise RuntimeError("provider exploded")

    handler.predict_two_pass_structured = _boom

    result = handler.generate_complete_system("Build a big library app")

    assert result["action"] == "assistant_message"
    assert result.get("error") is True
    assert "model" not in result


def test_complete_system_minimal_structured_spec_still_injects(handler, monkeypatch):
    from schemas import AuthoredSystemGUISpec

    handler.predict_two_pass_structured = lambda **kwargs: AuthoredSystemGUISpec(
        projectName="X",
        pages=[{"name": "Home", "sections": [
            {"html": "<section class='ds-section'><h2 class='ds-heading'>Hi</h2></section>"},
        ]}],
    )

    result = handler.generate_complete_system("Build something")

    assert result["action"] == "inject_complete_system"
    assert [p["name"] for p in result["model"]["pages"]] == ["Home"]


# ---------------------------------------------------------------------------
# Fix 2.1 — two_column left/right
# ---------------------------------------------------------------------------

def test_two_column_left_right_survive_schema_validation():
    section = GUISectionSpec.model_validate(
        {
            "type": "two_column",
            "title": "Books & Genres",
            "left": {"type": "table", "title": "Books", "className": "Book"},
            "right": {"type": "pie_chart", "title": "By genre", "className": "Book"},
        }
    )
    assert section.left is not None and section.left.type == "table"
    assert section.right is not None and section.right.type == "pie_chart"
    # model_dump (what generate_single_element feeds the builder) keeps them.
    dumped = section.model_dump()
    assert dumped["left"]["title"] == "Books"
    assert dumped["right"]["title"] == "By genre"


def test_two_column_renders_both_halves():
    section = {
        "type": "two_column",
        "title": "Split",
        "left": {"type": "table", "title": "Books", "className": "Book"},
        "right": {"type": "pie_chart", "title": "By genre", "className": "Book"},
    }
    comp = _two_column_component(section, None)
    blob = json.dumps(comp)
    assert "Books" in blob and "By genre" in blob
    # No placeholder "Left"/"Right"/"Content" boxes when real halves exist.
    assert "\"Left\"" not in blob and "\"Right\"" not in blob


def test_two_column_dispatches_through_build_section():
    section = {
        "type": "two_column",
        "left": {"type": "content", "title": "Alpha", "body": "a"},
        "right": {"type": "content", "title": "Beta", "body": "b"},
    }
    comp = _build_section_component(section, None)
    blob = json.dumps(comp)
    assert "Alpha" in blob and "Beta" in blob


# ---------------------------------------------------------------------------
# Fix 2.2 — stats_grid values + distinct fields
# ---------------------------------------------------------------------------

def test_stats_grid_stat_field_validates():
    section = GUISectionSpec.model_validate(
        {
            "type": "stats_grid",
            "stats": [
                {"label": "Total Users", "value": "1,234"},
                {"label": "Revenue", "value": "$9,999"},
            ],
        }
    )
    assert [(s.label, s.value) for s in section.stats] == [
        ("Total Users", "1,234"),
        ("Revenue", "$9,999"),
    ]


def test_stats_grid_preserves_provided_values():
    section = {
        "type": "stats_grid",
        "title": "KPIs",
        "stats": [
            {"label": "Users", "value": "1,234"},
            {"label": "Revenue", "value": "$9,999"},
        ],
    }
    comp = _stats_grid_component(section, None)
    cards = [c for c in comp["components"] if c.get("type") == "metric-card"]
    values = [c["attributes"].get("metric-value") for c in cards]
    assert "1,234" in values
    assert "$9,999" in values


def test_stats_grid_reads_value_from_items_fallback():
    # The free-text complete-system path historically emits stats under
    # ``items`` as dicts — those values must survive too.
    section = {
        "type": "stats_grid",
        "items": [{"label": "On loan", "value": "1,045"}],
    }
    comp = _stats_grid_component(section, None)
    cards = [c for c in comp["components"] if c.get("type") == "metric-card"]
    assert cards[0]["attributes"]["metric-value"] == "1,045"


def test_stats_grid_spreads_across_distinct_numeric_fields():
    section = {
        "type": "stats_grid",
        "className": "Book",
        "stats": [
            {"label": "Pages", "value": "320"},
            {"label": "Price", "value": "12"},
        ],
    }
    comp = _stats_grid_component(section, BOOK_METADATA)
    cards = [c for c in comp["components"] if c.get("type") == "metric-card"]
    fields = [c["attributes"].get("data-field") for c in cards]
    # Each card binds to its OWN numeric attribute, not all to the first.
    assert fields[0] == "a-pages"
    assert fields[1] == "a-price"
    assert fields[0] != fields[1]


# ---------------------------------------------------------------------------
# Strict-outputs smoke test (the CRITICAL constraint)
# ---------------------------------------------------------------------------

def test_schema_additions_stay_strict_output_compatible():
    """The new recursive left/right + typed stats must not break OpenAI
    strict structured output — a bad shape 400s EVERY GUI generate/modify."""
    strict = pytest.importorskip("openai.lib._pydantic")
    to_strict = strict.to_strict_json_schema
    # Must not raise for either strict schema.
    to_strict(SingleGUIElementSpec)
    to_strict(GUIModificationSpec)


def test_single_element_spec_builds_with_two_column_and_stats():
    spec = SingleGUIElementSpec.model_validate(
        {
            "pageName": "Home",
            "section": {
                "type": "two_column",
                "left": {"type": "stats_grid", "stats": [{"label": "A", "value": "1"}]},
                "right": {"type": "content", "title": "Hi"},
            },
        }
    )
    assert spec.section.left.stats[0].value == "1"
    assert spec.section.right.type == "content"


def test_modification_spec_builds_with_new_fields():
    spec = GUIModificationSpec.model_validate(
        {
            "operation": "append_section",
            "pageName": "Home",
            "section": {
                "type": "two_column",
                "left": {"type": "table", "title": "T"},
                "right": {"type": "stats_grid", "stats": [{"label": "X", "value": "9"}]},
            },
        }
    )
    assert spec.section.left.title == "T"
    assert spec.section.right.stats[0].value == "9"
