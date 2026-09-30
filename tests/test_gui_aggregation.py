"""AI-designed charts and metric cards use BESSER's aggregation.

BESSER's web-app generator now reads a metric card's / chart's
``aggregation`` (count, sum, avg, min, max; a chart groups by its label
field): a metric card with no field counts records, with a field and no
aggregation it sums it. The design prompt said "a chart plots valueField per
record, it does not aggregate", and on the saved library/hotel/tasks AI
designs 0 of 4 charts aggregated. Checked end to end through that generator.
"""

import json

from diagram_handlers.types.gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from schemas import AuthoredSystemGUISpec
from schemas.gui_diagram import GUIBindSpec

from tests.test_gui_actions import _nodes


def _attr(aid, name, typ, **flags):
    return {
        "id": aid, "name": name, "type": typ,
        "isNumeric": typ in ("int", "float"), "isString": typ == "str",
        "isId": False, "isOptional": False, "isDerived": False, "hasDefault": False,
        **flags,
    }


META = [
    {
        "id": "cls-book", "name": "Book",
        "attributes": [
            _attr("b-title", "title", "str"),
            _attr("b-genre", "genre", "str"),
            _attr("b-copies", "copies", "int"),
            _attr("b-notes", "notes", "str", isOptional=True),
            _attr("b-score", "score", "float", isDerived=True),
        ],
        "methods": [{"id": "m-reserve", "name": "reserve", "isInstanceMethod": True, "params": []}],
    },
    {
        "id": "cls-member", "name": "Member",
        "attributes": [_attr("m-name", "name", "str"), _attr("m-email", "email", "str")],
        "methods": [],
    },
    {
        "id": "cls-loan", "name": "Loan",
        "attributes": [_attr("l-due", "due_date", "date"), _attr("l-fine", "fine", "float")],
        "methods": [],
    },
]


def _generate(pages, meta=META):
    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)
    handler.predict_two_pass_structured = (
        lambda **kw: AuthoredSystemGUISpec(projectName="Library", pages=pages)
    )
    res = handler.generate_complete_system("library", class_metadata=meta)
    assert res["action"] == "inject_complete_system"
    return res["model"]


def _widget(nodes, kind):
    return next(n for n in nodes if n.get("type") == kind)


# -- aggregation --------------------------------------------------------------

def _dashboard(*binds):
    return [{"name": "Dashboard", "sections": [{"bind": b} for b in binds]}]


def test_metric_without_a_field_counts_records():
    # It used to sum the first numeric attribute ('Books' showed a sum of copies).
    card = _widget(_nodes(_generate(_dashboard(
        {"kind": "metric_card", "className": "Book", "title": "Books"},
    )), "Dashboard"), "metric-card")
    assert card["attributes"]["aggregation"] == "count"
    assert "data-field" not in card["attributes"]


def test_metric_aggregates_its_field():
    nodes = _nodes(_generate(_dashboard(
        {"kind": "metric_card", "className": "Loan", "title": "Average fine", "valueField": "fine", "aggregation": "avg"},
        {"kind": "metric_card", "className": "Book", "title": "Copies", "valueField": "copies"},
    )), "Dashboard")
    avg, total = [n for n in nodes if n.get("type") == "metric-card"]
    assert (avg["attributes"]["data-field"], avg["attributes"]["aggregation"]) == ("l-fine", "avg")
    assert (total["attributes"]["data-field"], total["attributes"]["aggregation"]) == ("b-copies", "sum")


def test_count_chart_groups_records_by_label():
    chart = _widget(_nodes(_generate(_dashboard(
        {"kind": "bar_chart", "className": "Book", "title": "Books by genre", "labelField": "genre", "aggregation": "count"},
    )), "Dashboard"), "bar-chart")
    assert chart["attributes"]["aggregation"] == "count"
    series = json.loads(chart["attributes"]["series"])
    assert len(series) == 1
    assert series[0]["label-field"] == "b-genre"
    assert "data-field" not in series[0]


def test_pie_series_carries_its_fields_and_aggregation():
    # BESSER binds a pie through its series; the fields on the chart alone
    # left the generated pie grouping by a missing 'name'.
    pie = _widget(_nodes(_generate(_dashboard(
        {"kind": "pie_chart", "className": "Book", "title": "Copies by genre",
         "labelField": "genre", "valueField": "copies", "aggregation": "sum"},
    )), "Dashboard"), "pie-chart")
    assert pie["attributes"]["aggregation"] == "sum"
    series = json.loads(pie["attributes"]["series"])[0]
    assert (series["label-field"], series["data-field"]) == ("b-genre", "b-copies")


def test_chart_without_aggregation_is_unchanged():
    chart = _widget(_nodes(_generate(_dashboard(
        {"kind": "line_chart", "className": "Loan", "labelField": "due_date", "valueField": "fine"},
    )), "Dashboard"), "line-chart")
    assert "aggregation" not in chart["attributes"]


def test_bind_schema_and_prompt_offer_aggregation():
    assert GUIBindSpec(kind="metric_card", aggregation="avg").aggregation == "avg"
    captured = {}

    def _predict(**kw):
        captured.update(kw)
        return AuthoredSystemGUISpec(projectName="L", pages=_dashboard({"kind": "table", "className": "Book"}))

    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)
    handler.predict_two_pass_structured = _predict
    handler.generate_complete_system("library", class_metadata=META)
    assert "does not aggregate" not in captured["system_prompt"]
    assert "count|sum|avg|min|max" in captured["system_prompt"]
    assert "count of records" in captured["reasoning_prompt"]
