"""AI-designed charts and metric cards use BESSER's aggregation.

BESSER's web-app generator now reads a metric card's / chart's
``aggregation`` (count, sum, avg, min, max; a chart groups by its label
field): a metric card with no field counts records, with a field and no
aggregation it sums it. The design prompt said "a chart plots valueField per
record, it does not aggregate", and on the saved library/hotel/tasks AI
designs 0 of 4 charts aggregated. Checked end to end through that generator.
"""

import json

import pytest

from diagram_handlers.types.gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from schemas import AuthoredSystemGUISpec
from schemas.gui_diagram import GUIBindSpec, GUIModificationBatchSpec
from utilities.class_metadata import extract_class_metadata, format_class_metadata_for_prompt

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


def test_metric_cards_carry_no_trend_caption():
    # BESSER renders show-trend as a fixed "Latest value" caption, wrong under
    # a count or any other aggregate; it is on unless set to false.
    from diagram_handlers.types.gui_nocode_diagram_handler import (
        _stat_placeholder, _stats_grid_component,
    )
    nodes = _nodes(_generate(_dashboard(
        {"kind": "metric_card", "className": "Book", "title": "Books"},
        {"kind": "metric_card", "className": "Loan", "title": "Average fine", "valueField": "fine", "aggregation": "avg"},
    )), "Dashboard")
    cards = [n for n in nodes if n.get("type") == "metric-card"]
    grid = _stats_grid_component({"stats": [{"label": "Books", "value": "12"}]}, META)
    cards += [c for c in grid["components"] if c.get("type") == "metric-card"]
    cards.append(_stat_placeholder("Total", META[0]))
    assert len(cards) == 4
    assert all(c["attributes"]["show-trend"] == "false" for c in cards)

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


# -- conditional metrics ------------------------------------------------------

LOAN_STATES = ["active", "returned", "overdue"]

FILTER_META = [
    {
        "id": "cls-loan", "name": "Loan",
        "attributes": [
            _attr("l-status", "status", "loanstatus", enumLiterals=LOAN_STATES),
            _attr("l-renewed", "renewed", "bool"),
            _attr("l-fine", "fine", "float"),
            _attr("l-due", "dueDate", "date"),
            _attr("l-desk", "desk", "str"),
        ],
        "methods": [],
    },
]


def _metric_cards(*binds):
    model = _generate(_dashboard(*binds), FILTER_META)
    return [n for n in _nodes(model, "Dashboard") if n.get("type") == "metric-card"]


def _filtered(field, value, operator="=="):
    return {"kind": "metric_card", "className": "Loan", "title": "Some loans",
            "filter": {"field": field, "operator": operator, "value": value}}


def test_conditional_count_carries_its_filter():
    # Without the filter a card titled 'Active loans' showed every loan.
    card, = _metric_cards(_filtered("status", "Active"))
    assert card["attributes"]["aggregation"] == "count"
    assert card["attributes"]["filter"] == "status == active"


def test_filter_on_a_boolean_a_number_a_date_and_a_text():
    cards = _metric_cards(
        _filtered("renewed", "True"),
        _filtered("fine", "0", ">"),
        _filtered("due_date", "2030-01-31", "<="),
        _filtered("desk", "Main hall", "!="),
    )
    assert [c["attributes"]["filter"] for c in cards] == [
        "renewed == true", "fine > 0", "dueDate <= 2030-01-31", "desk != 'Main hall'",
    ]


def test_metric_whose_condition_the_model_cannot_evaluate_is_left_out():
    # An unfiltered card under a conditional title shows a wrong number.
    cards = _metric_cards(
        _filtered("state", "active"),              # no such attribute
        _filtered("status", "lost"),               # not a literal of the enumeration
        _filtered("status", "active", ">"),        # an enumeration has no order
        _filtered("fine", "a lot", ">"),           # not a number
        {"kind": "metric_card", "className": "Loan", "title": "Loans"},
    )
    assert [c["attributes"]["metric-title"] for c in cards] == ["Loans"]
    assert "filter" not in cards[0]["attributes"]


def test_blank_filter_is_no_filter():
    card, = _metric_cards({"kind": "metric_card", "className": "Loan", "title": "Loans",
                           "filter": {"field": "", "operator": "==", "value": ""}})
    assert "filter" not in card["attributes"]


def test_bind_filter_is_strict_output_compatible():
    strict = pytest.importorskip("openai.lib._pydantic")
    schema = strict.to_strict_json_schema(AuthoredSystemGUISpec)
    assert "filter" in schema["$defs"]["GUIBindSpec"]["properties"]


def test_class_metadata_lists_enumeration_literals():
    model = {
        "elements": {
            "c1": {"id": "c1", "type": "Class", "name": "Loan"},
            "a1": {"id": "a1", "type": "ClassAttribute", "owner": "c1", "name": "status",
                   "attributeType": "LoanStatus"},
            "a2": {"id": "a2", "type": "ClassAttribute", "owner": "c1", "name": "fine",
                   "attributeType": "float"},
            "e1": {"id": "e1", "type": "Enumeration", "name": "LoanStatus"},
            "l1": {"id": "l1", "type": "ClassAttribute", "owner": "e1", "name": "active"},
            "l2": {"id": "l2", "type": "ClassAttribute", "owner": "e1", "name": "returned"},
        },
        "relationships": {},
    }
    metadata = extract_class_metadata(model)
    status, fine = metadata[0]["attributes"]
    assert status["enumLiterals"] == ["active", "returned"]
    assert "enumLiterals" not in fine
    assert "status (loanstatus: active | returned)" in format_class_metadata_for_prompt(metadata)


def test_modification_reports_a_metric_it_cannot_compute():
    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)
    model = _generate(_dashboard({"kind": "table", "className": "Loan"}), FILTER_META)
    handler.predict_structured = lambda *a, **k: GUIModificationBatchSpec(operations=[{
        "operation": "append_section", "pageName": "Dashboard",
        "section": {"bind": _filtered("status", "lost")},
    }])
    res = handler.generate_modification(
        "add a lost loans card", model, raw_request="add a lost loans card",
        class_metadata=FILTER_META,
    )
    assert not [n for n in _nodes(res["model"], "Dashboard") if n.get("type") == "metric-card"]
    assert "condition" in res["message"]


def test_prompt_offers_the_metric_filter():
    captured = {}

    def _predict(**kw):
        captured.update(kw)
        return AuthoredSystemGUISpec(projectName="L", pages=_dashboard({"kind": "table", "className": "Loan"}))

    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)
    handler.predict_two_pass_structured = _predict
    handler.generate_complete_system("library", class_metadata=FILTER_META)
    assert '"filter"' in captured["system_prompt"]
    assert "status (loanstatus: active | returned | overdue)" in captured["system_prompt"]
    assert "filter: status == active" in captured["reasoning_prompt"]


# -- a filter never costs more than its own card -------------------------------

STAGES = ["DRAFT", "IN_REVIEW", "In", "on hold", "In progress"]
STAGE_META = [{
    "id": "cls-paper", "name": "Paper",
    "attributes": [
        _attr("p-stage", "stage", "stage", enumLiterals=STAGES),
        _attr("p-open", "open", "bool"),
        _attr("p-pages", "pages", "int"),
        _attr("p-sent", "sentAt", "datetime"),
        _attr("p-author", "author", "str"),
    ],
    "methods": [],
}]


def _paper(field, value, operator="==", title="Some papers"):
    return {"kind": "metric_card", "className": "Paper", "title": title,
            "filter": {"field": field, "operator": operator, "value": value}}


def _paper_filters(*binds):
    model = _generate(_dashboard(*binds), STAGE_META)
    return [n["attributes"].get("filter") for n in _nodes(model, "Dashboard")
            if n.get("type") == "metric-card"]


def test_filter_values_and_operators_as_models_write_them():
    # A true/0 value or '=' used to fail schema validation, and with it the
    # whole design.
    assert _paper_filters(
        _paper("open", True, "="),
        _paper("open", 0),
        _paper("pages", 0, "gt"),
        _paper("pages", 12.5, "<>"),
        _paper("stage", "draft", "eq"),
    ) == ["open == true", "open == false", "pages > 0", "pages != 12.5", "stage == DRAFT"]


def test_filter_given_as_text_or_without_its_parts_does_not_void_the_design():
    assert _paper_filters(
        {"kind": "metric_card", "className": "Paper", "title": "Open", "filter": "open == true"},
        {"kind": "metric_card", "className": "Paper", "title": "All", "filter": {}},
        {"kind": "metric_card", "className": "Paper", "title": "Odd", "filter": {"field": "pages"}},
        _paper("author", "Ann", "contains"),
    ) == ["open == true", None]


def test_filter_values_the_generated_app_reads_are_accepted():
    assert _paper_filters(
        _paper("stage", "In"),
        _paper("stage", "in review"),
        _paper("stage", "'DRAFT'"),
        _paper("stage", "on hold"),
        _paper("pages", "1e3", ">="),
        _paper("sentAt", "2030-01-31T08:00:00Z", ">="),
        _paper("author", "O'Neil"),
        _paper("author", "R&D"),
        _paper("open", "yes"),
    ) == [
        "stage == In", "stage == IN_REVIEW", "stage == DRAFT", "stage == on hold",
        "pages >= 1000", "sentAt >= 2030-01-31T08:00:00Z",
        "author == 'O'Neil'", "author == 'R&D'", "open == true",
    ]


def test_filter_values_the_generated_app_would_misread_are_not():
    assert _paper_filters(
        _paper("stage", "in review 2"),
        _paper("pages", "many", ">"),
        _paper("sentAt", "last week", ">"),
        # A number in a locale format: '1,5' was read as 15, '1.000,50' as 1.0005.
        _paper("pages", "1,5", ">"),
        _paper("pages", "1.000,50", ">"),
        _paper("pages", "1,000", "<"),
        _paper("pages", "1 000", "<"),
    ) == []


def test_filter_numbers_are_written_without_rounding():
    # 1e-7 was written as 0 and 0.1234567 as 0.123457: the card counted other records.
    assert _paper_filters(
        _paper("pages", "1e-7", ">"),
        _paper("pages", 1e-7, ">"),
        _paper("pages", "0.1234567", ">"),
        _paper("pages", 1e21, "<"),
        _paper("pages", "-5", ">="),
    ) == [
        "pages > 0.0000001", "pages > 0.0000001", "pages > 0.1234567",
        "pages < 1000000000000000000000", "pages >= -5",
    ]


def test_filter_value_the_text_form_would_split_is_written_as_json():
    # The app reads 'and', ',', ';' and a word 'in' / 'contains' in a text
    # filter as syntax; it reads a filter written as JSON as one condition.
    filters = _paper_filters(
        _paper("stage", "In progress"),
        _paper("author", "Rock and roll", "!="),
        _paper("author", "Acme, Inc."),
        _paper("author", "checked in today"),
    )
    assert [json.loads(f) for f in filters] == [
        {"field": "stage", "op": "==", "value": "In progress"},
        {"field": "author", "op": "!=", "value": "Rock and roll"},
        {"field": "author", "op": "==", "value": "Acme, Inc."},
        {"field": "author", "op": "==", "value": "checked in today"},
    ]


def test_reply_names_a_metric_card_that_was_left_out():
    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)
    handler.predict_two_pass_structured = lambda **kw: AuthoredSystemGUISpec(
        projectName="L", pages=_dashboard(
            {"kind": "table", "className": "Paper"},
            _paper("stage", "lost", title="Lost papers"),
        ))
    res = handler.generate_complete_system("papers", class_metadata=STAGE_META)
    assert res["action"] == "inject_complete_system"
    assert "**Lost papers**" in res["message"] and "left out" in res["message"]
    # Nothing to say when every card was built.
    handler.predict_two_pass_structured = lambda **kw: AuthoredSystemGUISpec(
        projectName="L", pages=_dashboard(_paper("stage", "draft")))
    assert "left out" not in handler.generate_complete_system("papers", class_metadata=STAGE_META)["message"]
