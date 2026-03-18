"""Tests for GUI chart/table generation in the GUINoCodeDiagram handler.

Covers:
- extract_class_metadata() helper
- format_class_metadata_for_prompt() helper
- _chart_component(), _table_component(), _dashboard_component() builders
- _build_section_component() dispatch for chart/table/dashboard types
- _resolve_class_binding(), _pick_label_field(), _pick_data_field() helpers
"""

import json
import pytest
from typing import Any, Dict, List

from utilities.class_metadata import extract_class_metadata, format_class_metadata_for_prompt
from diagram_handlers.types.gui_nocode_diagram_handler import (
    _build_section_component,
    _chart_component,
    _table_component,
    _dashboard_component,
    _resolve_class_binding,
    _pick_label_field,
    _pick_data_field,
    _build_series,
)


# ---------------------------------------------------------------------------
# Fixtures — realistic ClassDiagram model
# ---------------------------------------------------------------------------

LIBRARY_CLASS_MODEL: Dict[str, Any] = {
    "version": "3.0.0",
    "type": "ClassDiagram",
    "elements": {
        "cls-book-1": {
            "id": "cls-book-1",
            "name": "Book",
            "type": "Class",
            "owner": None,
            "bounds": {"x": 100, "y": 100, "width": 200, "height": 150},
        },
        "attr-book-title": {
            "id": "attr-book-title",
            "name": "+title: str",
            "type": "ClassAttribute",
            "owner": "cls-book-1",
            "attributeType": "str",
        },
        "attr-book-pages": {
            "id": "attr-book-pages",
            "name": "+pages: int",
            "type": "ClassAttribute",
            "owner": "cls-book-1",
            "attributeType": "int",
        },
        "attr-book-price": {
            "id": "attr-book-price",
            "name": "+price: float",
            "type": "ClassAttribute",
            "owner": "cls-book-1",
            "attributeType": "float",
        },
        "cls-author-1": {
            "id": "cls-author-1",
            "name": "Author",
            "type": "Class",
            "owner": None,
            "bounds": {"x": 400, "y": 100, "width": 200, "height": 120},
        },
        "attr-author-name": {
            "id": "attr-author-name",
            "name": "+name: string",
            "type": "ClassAttribute",
            "owner": "cls-author-1",
            "attributeType": "string",
        },
        "attr-author-age": {
            "id": "attr-author-age",
            "name": "+age: int",
            "type": "ClassAttribute",
            "owner": "cls-author-1",
            "attributeType": "int",
        },
    },
    "relationships": {},
}

EMPTY_CLASS_MODEL: Dict[str, Any] = {
    "version": "3.0.0",
    "type": "ClassDiagram",
    "elements": {},
    "relationships": {},
}

NO_ATTRIBUTES_MODEL: Dict[str, Any] = {
    "version": "3.0.0",
    "type": "ClassDiagram",
    "elements": {
        "cls-empty": {
            "id": "cls-empty",
            "name": "EmptyClass",
            "type": "Class",
            "owner": None,
        },
    },
    "relationships": {},
}


# ---------------------------------------------------------------------------
# extract_class_metadata
# ---------------------------------------------------------------------------

class TestExtractClassMetadata:
    def test_extracts_classes_and_attributes(self):
        metadata = extract_class_metadata(LIBRARY_CLASS_MODEL)
        assert len(metadata) == 2
        names = {cls["name"] for cls in metadata}
        assert names == {"Book", "Author"}

    def test_book_attributes(self):
        metadata = extract_class_metadata(LIBRARY_CLASS_MODEL)
        book = next(cls for cls in metadata if cls["name"] == "Book")
        assert len(book["attributes"]) == 3

        title_attr = next(a for a in book["attributes"] if a["name"] == "title")
        assert title_attr["type"] == "str"
        assert title_attr["isString"] is True
        assert title_attr["isNumeric"] is False
        assert title_attr["id"] == "attr-book-title"

        pages_attr = next(a for a in book["attributes"] if a["name"] == "pages")
        assert pages_attr["type"] == "int"
        assert pages_attr["isNumeric"] is True
        assert pages_attr["isString"] is False

    def test_empty_model_returns_empty(self):
        assert extract_class_metadata(EMPTY_CLASS_MODEL) == []

    def test_none_returns_empty(self):
        assert extract_class_metadata(None) == []

    def test_class_without_attributes(self):
        metadata = extract_class_metadata(NO_ATTRIBUTES_MODEL)
        assert len(metadata) == 1
        assert metadata[0]["name"] == "EmptyClass"
        assert metadata[0]["attributes"] == []

    def test_legacy_type_parsing(self):
        """When attributeType is missing, parse from name."""
        model = {
            "elements": {
                "cls-1": {"id": "cls-1", "name": "Foo", "type": "Class", "owner": None},
                "attr-1": {
                    "id": "attr-1",
                    "name": "+score: double",
                    "type": "ClassAttribute",
                    "owner": "cls-1",
                },
            },
            "relationships": {},
        }
        metadata = extract_class_metadata(model)
        attr = metadata[0]["attributes"][0]
        assert attr["name"] == "score"
        assert attr["type"] == "double"
        assert attr["isNumeric"] is True


# ---------------------------------------------------------------------------
# format_class_metadata_for_prompt
# ---------------------------------------------------------------------------

class TestFormatClassMetadataForPrompt:
    def test_format_output(self):
        metadata = extract_class_metadata(LIBRARY_CLASS_MODEL)
        result = format_class_metadata_for_prompt(metadata)
        assert "Available classes from the Class Diagram:" in result
        assert 'Class "Book"' in result
        assert 'Class "Author"' in result
        assert "pages (int)" in result
        assert "title (str)" in result

    def test_empty_metadata(self):
        assert format_class_metadata_for_prompt([]) == ""


# ---------------------------------------------------------------------------
# _resolve_class_binding
# ---------------------------------------------------------------------------

class TestResolveClassBinding:
    @pytest.fixture()
    def metadata(self):
        return extract_class_metadata(LIBRARY_CLASS_MODEL)

    def test_by_class_name(self, metadata):
        spec = {"className": "Author"}
        result = _resolve_class_binding(spec, metadata)
        assert result is not None
        assert result["name"] == "Author"

    def test_by_class_name_case_insensitive(self, metadata):
        spec = {"className": "book"}
        result = _resolve_class_binding(spec, metadata)
        assert result is not None
        assert result["name"] == "Book"

    def test_by_class_id(self, metadata):
        spec = {"classId": "cls-author-1"}
        result = _resolve_class_binding(spec, metadata)
        assert result is not None
        assert result["name"] == "Author"

    def test_fallback_to_first(self, metadata):
        spec = {"className": "NonExistent"}
        result = _resolve_class_binding(spec, metadata)
        assert result is not None  # Falls back to first class with attributes

    def test_no_metadata_returns_none(self):
        assert _resolve_class_binding({}, None) is None
        assert _resolve_class_binding({}, []) is None


# ---------------------------------------------------------------------------
# _pick_label_field / _pick_data_field
# ---------------------------------------------------------------------------

class TestPickFields:
    @pytest.fixture()
    def book_class(self):
        metadata = extract_class_metadata(LIBRARY_CLASS_MODEL)
        return next(cls for cls in metadata if cls["name"] == "Book")

    def test_pick_label_field_prefers_string(self, book_class):
        result = _pick_label_field(book_class)
        assert result is not None
        assert result["isString"] is True
        assert result["name"] == "title"

    def test_pick_data_field_prefers_numeric(self, book_class):
        result = _pick_data_field(book_class)
        assert result is not None
        assert result["isNumeric"] is True
        assert result["name"] in ("pages", "price")

    def test_pick_from_empty_class(self):
        cls = {"id": "cls-1", "name": "Empty", "attributes": []}
        assert _pick_label_field(cls) is None
        assert _pick_data_field(cls) is None


# ---------------------------------------------------------------------------
# _chart_component
# ---------------------------------------------------------------------------

class TestChartComponent:
    @pytest.fixture()
    def metadata(self):
        return extract_class_metadata(LIBRARY_CLASS_MODEL)

    def test_bar_chart_structure(self, metadata):
        spec = {"title": "Book Stats", "className": "Book"}
        result = _chart_component("bar-chart", spec, metadata)
        assert result["type"] == "bar-chart"
        attrs = result["attributes"]
        assert attrs["chart-title"] == "Book Stats"
        assert attrs["class"] == "bar-chart-component"
        # Series should be present
        series = json.loads(attrs["series"])
        assert isinstance(series, list)
        assert len(series) > 0
        # Each series should reference Book class
        for s in series:
            assert s["data-source"] == "cls-book-1"

    def test_pie_chart_has_direct_binding(self, metadata):
        spec = {"title": "Pie", "className": "Book"}
        result = _chart_component("pie-chart", spec, metadata)
        assert result["type"] == "pie-chart"
        attrs = result["attributes"]
        # Pie charts have direct data-source, label-field, data-field
        assert attrs["data-source"] == "cls-book-1"
        assert "label-field" in attrs
        assert "data-field" in attrs

    def test_line_chart_defaults(self, metadata):
        spec = {"title": "Trend", "className": "Author"}
        result = _chart_component("line-chart", spec, metadata)
        assert result["type"] == "line-chart"
        attrs = result["attributes"]
        assert attrs["curve-type"] == "monotone"
        assert attrs["show-tooltip"] == "true"

    def test_chart_without_metadata(self):
        spec = {"title": "No Data"}
        result = _chart_component("bar-chart", spec, None)
        assert result["type"] == "bar-chart"
        attrs = result["attributes"]
        # Without metadata, charts still get generic dummy data for preview
        series = json.loads(attrs["series"])
        assert isinstance(series, list)
        assert len(series) == 1
        assert len(series[0]["data"]) > 0

    def test_chart_with_empty_metadata(self):
        spec = {"title": "Empty"}
        result = _chart_component("bar-chart", spec, [])
        assert result["type"] == "bar-chart"

    def test_chart_style(self, metadata):
        spec = {"title": "Test"}
        result = _chart_component("bar-chart", spec, metadata)
        assert "min-height" in result["style"]
        assert result["style"]["width"] == "100%"


# ---------------------------------------------------------------------------
# _build_series
# ---------------------------------------------------------------------------

class TestBuildSeries:
    @pytest.fixture()
    def book_class(self):
        metadata = extract_class_metadata(LIBRARY_CLASS_MODEL)
        return next(cls for cls in metadata if cls["name"] == "Book")

    def test_auto_series_from_numeric_attrs(self, book_class):
        series_json = _build_series("bar-chart", book_class, {})
        series = json.loads(series_json)
        # Book has 2 numeric attrs: pages, price → 2 auto-generated series
        assert len(series) == 2
        for s in series:
            assert s["data-source"] == "cls-book-1"
            assert "label-field" in s
            assert "data-field" in s
            assert "color" in s

    def test_explicit_series(self, book_class):
        spec = {
            "series": [
                {"name": "Custom", "labelField": "attr-book-title", "dataField": "attr-book-pages"}
            ]
        }
        series_json = _build_series("bar-chart", book_class, spec)
        series = json.loads(series_json)
        assert len(series) == 1
        assert series[0]["name"] == "Custom"
        assert series[0]["label-field"] == "attr-book-title"
        assert series[0]["data-field"] == "attr-book-pages"

    def test_max_three_auto_series(self):
        """Even with many numeric attrs, cap at 3 series."""
        cls = {
            "id": "cls-x",
            "name": "ManyNums",
            "attributes": [
                {"id": f"attr-{i}", "name": f"num{i}", "type": "int", "isNumeric": True, "isString": False}
                for i in range(10)
            ],
        }
        series_json = _build_series("bar-chart", cls, {})
        series = json.loads(series_json)
        assert len(series) == 3


# ---------------------------------------------------------------------------
# _table_component
# ---------------------------------------------------------------------------

class TestTableComponent:
    @pytest.fixture()
    def metadata(self):
        return extract_class_metadata(LIBRARY_CLASS_MODEL)

    def test_table_structure(self, metadata):
        spec = {"title": "Books Table", "className": "Book"}
        result = _table_component(spec, metadata)
        assert result["type"] == "table"
        attrs = result["attributes"]
        assert attrs["data-source"] == "cls-book-1"
        columns = json.loads(attrs["columns"])
        # Should have field columns for title, pages, price
        field_cols = [c for c in columns if c.get("columnType") == "field"]
        assert len(field_cols) == 3
        labels = {c["label"] for c in field_cols}
        assert "Title" in labels
        assert "Pages" in labels

    def test_table_without_metadata(self):
        spec = {"title": "Empty Table"}
        result = _table_component(spec, None)
        assert result["type"] == "table"
        assert "data-source" not in result["attributes"]


# ---------------------------------------------------------------------------
# _dashboard_component
# ---------------------------------------------------------------------------

class TestDashboardComponent:
    @pytest.fixture()
    def metadata(self):
        return extract_class_metadata(LIBRARY_CLASS_MODEL)

    def test_dashboard_has_table_and_charts(self, metadata):
        spec = {"title": "Book Dashboard", "className": "Book"}
        result = _dashboard_component(spec, metadata)
        assert result["tagName"] == "section"
        assert "assistant-dashboard" in result["attributes"]["class"]

        # Flatten all nested components
        def find_types(comp, types=None):
            if types is None:
                types = []
            if isinstance(comp, dict):
                if comp.get("type"):
                    types.append(comp["type"])
                for child in comp.get("components", []):
                    find_types(child, types)
            return types

        all_types = find_types(result)
        assert "table" in all_types
        # Book has numeric attrs so should have at least one chart
        chart_types = [t for t in all_types if t.endswith("-chart")]
        assert len(chart_types) >= 1


# ---------------------------------------------------------------------------
# _build_section_component dispatch
# ---------------------------------------------------------------------------

class TestBuildSectionComponentDispatch:
    @pytest.fixture()
    def metadata(self):
        return extract_class_metadata(LIBRARY_CLASS_MODEL)

    @pytest.mark.parametrize("section_type", [
        "table", "data_table",
    ])
    def test_table_dispatch(self, section_type, metadata):
        spec = {"type": section_type, "title": "Test Table", "className": "Book"}
        result = _build_section_component(spec, metadata)
        # _build_section_component wraps data components in a card (tagName=section)
        assert result["tagName"] == "section"
        assert "assistant-card" in result["attributes"]["class"]
        # The actual table component is inside components (after the h2 title)
        inner = result["components"][-1]
        assert inner["type"] == "table"
        assert inner["attributes"]["data-source"] == "cls-book-1"

    @pytest.mark.parametrize("section_type,expected_chart_type", [
        ("bar_chart", "bar-chart"),
        ("bar-chart", "bar-chart"),
        ("pie_chart", "pie-chart"),
        ("pie-chart", "pie-chart"),
        ("line_chart", "line-chart"),
        ("line-chart", "line-chart"),
        ("radar_chart", "radar-chart"),
        ("chart", "bar-chart"),
    ])
    def test_chart_dispatch(self, section_type, expected_chart_type, metadata):
        spec = {"type": section_type, "title": "Test", "className": "Book"}
        result = _build_section_component(spec, metadata)
        # _build_section_component wraps chart components in a card (tagName=section)
        assert result["tagName"] == "section"
        assert "assistant-card" in result["attributes"]["class"]
        # The actual chart component is inside components (after the h2 title)
        inner = result["components"][-1]
        assert inner["type"] == expected_chart_type

    def test_dashboard_dispatch(self, metadata):
        spec = {"type": "dashboard", "title": "Overview", "className": "Book"}
        result = _build_section_component(spec, metadata)
        assert result["tagName"] == "section"
        assert "dashboard" in result["attributes"]["class"]

    def test_legacy_types_still_work(self, metadata):
        """Existing section types should continue to work."""
        hero = _build_section_component({"type": "hero", "title": "Welcome"}, metadata)
        assert hero["tagName"] == "section"

        features = _build_section_component(
            {"type": "feature_list", "title": "Features", "items": ["A", "B"]}, metadata
        )
        assert features["tagName"] == "section"

        form = _build_section_component(
            {"type": "form", "title": "Contact", "fields": ["Name"]}, metadata
        )
        assert form["tagName"] == "section"

        content = _build_section_component({"type": "content", "title": "About"}, metadata)
        assert content["tagName"] == "section"

    def test_without_metadata_falls_back(self):
        """Chart sections should work even without class metadata."""
        result = _build_section_component({"type": "bar_chart", "title": "Test"})
        # _build_section_component wraps chart components in a card (tagName=section)
        assert result["tagName"] == "section"
        assert "assistant-card" in result["attributes"]["class"]
        # The actual chart component is inside components (after the h2 title)
        inner = result["components"][-1]
        assert inner["type"] == "bar-chart"
        attrs = inner["attributes"]
        # Without metadata, charts still get generic dummy data for preview
        series = json.loads(attrs["series"])
        assert isinstance(series, list)
        assert len(series) == 1
        assert len(series[0]["data"]) > 0
