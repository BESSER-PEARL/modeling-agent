"""AI-designed data widgets must be specific and bound to the right class.

Measured on three AI designs before this fix: 15 of 17 tables were titled
"Data Table", charts "Bar Chart"/"Pie Chart", the metric "Metric" showing the
last row's id, chart fields were picked by heuristic, and an unknown class
name silently bound to the first class. The bind spec now carries a title and
the chart/metric fields; unmatched names render static content.
"""

import json

import pytest

from diagram_handlers.types.gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from schemas import AuthoredSystemGUISpec

TASK_METADATA = [
    {
        "id": "cls-task",
        "name": "Task",
        "attributes": [
            {"id": "a-id", "name": "id", "type": "int", "isNumeric": True, "isString": False},
            {"id": "a-title", "name": "title", "type": "str", "isNumeric": False, "isString": True},
            {"id": "a-hours", "name": "estimate_hours", "type": "float", "isNumeric": True, "isString": False},
        ],
        "methods": [{"id": "m-complete", "name": "complete", "isInstanceMethod": True, "params": []}],
    },
    {
        "id": "cls-project",
        "name": "Project",
        "attributes": [
            {"id": "p-name", "name": "name", "type": "str", "isNumeric": False, "isString": True},
        ],
        "methods": [],
    },
]


def _generate(pages, class_metadata=TASK_METADATA):
    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)
    handler.predict_two_pass_structured = (
        lambda **kw: AuthoredSystemGUISpec(projectName="Tracker", pages=pages)
    )
    res = handler.generate_complete_system("team task tracker", class_metadata=class_metadata)
    assert res["action"] == "inject_complete_system"
    return res["model"]


def _nodes(model, page_name):
    page = next(p for p in model["pages"] if p["name"] == page_name)
    out = []

    def _walk(n):
        if isinstance(n, dict):
            out.append(n)
            for c in n.get("components") or []:
                _walk(c)

    _walk(page["frames"][0]["component"])
    return out


def _table(nodes, class_id):
    return next(n for n in nodes if n.get("type") == "table"
                and n["attributes"].get("data-source") == class_id)


def test_bind_spec_title_and_fields_stay_strict_output_compatible():
    strict = pytest.importorskip("openai.lib._pydantic")
    schema = strict.to_strict_json_schema(AuthoredSystemGUISpec)
    bind = schema["$defs"]["GUIBindSpec"]["properties"]
    assert {"title", "labelField", "valueField"} <= set(bind)


def test_bind_title_propagates_to_widgets():
    model = _generate([{"name": "Tasks", "sections": [
        {"bind": {"kind": "table", "className": "Task", "title": "Open tasks"}},
        {"bind": {"kind": "bar_chart", "className": "Task", "title": "Hours by task",
                  "labelField": "title", "valueField": "estimate_hours"}},
        {"bind": {"kind": "metric_card", "className": "Task", "title": "Hours planned",
                  "valueField": "estimate_hours"}},
    ]}])
    nodes = _nodes(model, "Tasks")
    table = _table(nodes, "cls-task")
    chart = next(n for n in nodes if n.get("type") == "bar-chart")
    metric = next(n for n in nodes if n.get("type") == "metric-card")
    assert table["attributes"]["chart-title"] == "Open tasks"
    assert chart["attributes"]["chart-title"] == "Hours by task"
    series = json.loads(chart["attributes"]["series"])
    assert [(s["label-field"], s["data-field"]) for s in series] == [("a-title", "a-hours")]
    assert metric["attributes"]["metric-title"] == "Hours planned"
    assert metric["attributes"]["data-field"] == "a-hours"


def test_untitled_widget_title_comes_from_chrome_heading():
    model = _generate([{"name": "Tasks", "sections": [
        {"bind": {"kind": "table", "className": "Task"},
         "html": "<section class='ds-section'><h3>Overdue work</h3><!--WIDGET:table--></section>"},
    ]}])
    assert _table(_nodes(model, "Tasks"), "cls-task")["attributes"]["chart-title"] == "Overdue work"


def test_metric_never_shows_an_id():
    model = _generate([{"name": "Tasks", "sections": [
        {"bind": {"kind": "metric_card", "className": "Task", "valueField": "title"}},
    ]}])
    metric = next(n for n in _nodes(model, "Tasks") if n.get("type") == "metric-card")
    assert metric["attributes"]["data-field"] == "a-hours"


def test_unmatched_class_name_renders_static_not_first_class():
    model = _generate([{"name": "Tasks", "sections": [
        {"bind": {"kind": "table", "className": "Widget", "columns": ["A"],
                  "rows": [{"cells": ["1"]}]}},
    ]}])
    assert "data-source" not in json.dumps(model["pages"])


def test_plural_class_name_binds_to_that_class():
    model = _generate([{"name": "Work", "sections": [{"bind": {"kind": "table", "className": "projects"}}]}])
    assert _table(_nodes(model, "Work"), "cls-project")


# -- widget markers -----------------------------------------------------------

def _slot_nodes(model):
    return [
        n for p in model["pages"] for n in _nodes(model, p["name"])
        if n.get("type") == "widget-slot"
    ]


def test_surplus_widget_markers_leave_no_unloadable_node():
    # The editor has no 'widget-slot' component: a second marker used to stay
    # in the section as a node it cannot load.
    model = _generate([
        {"name": "Tasks", "sections": [
            {"bind": {"kind": "table", "className": "Task", "title": "Open tasks"},
             "html": "<section class='ds-section'><h2>Open tasks</h2>"
                     "<div class='ds-card'><!--WIDGET:table--></div>"
                     "<div class='ds-card'><!--WIDGET:chart--></div>"
                     "<div class='ds-card'><!--WIDGET:table--></div></section>"},
            {"html": "<section class='ds-section'><h2>Notes</h2><!--WIDGET:table--></section>"},
        ]},
    ])
    assert _slot_nodes(model) == []
    # One bind, one widget: a repeated marker does not duplicate the table.
    tables = [n for n in _nodes(model, "Tasks") if n.get("type") == "table"]
    assert len(tables) == 1


# -- class-bound table columns ------------------------------------------------

def _attr(attr_id, name, type_="str", **flags):
    return {"id": attr_id, "name": name, "type": type_,
            "isNumeric": type_ in ("int", "float"), "isString": type_ == "str", **flags}


ORDER_METADATA = [
    {
        "id": "cls-order",
        "name": "Order",
        "attributes": [
            _attr("o-id", "id"),
            _attr("o-number", "orderNumber"),
            _attr("o-due", "dueDate", "date"),
            _attr("o-total", "total_amount", "float"),
        ],
        "methods": [],
        "associationEnds": [{
            "relationshipId": "r-1", "targetClassId": "cls-customer",
            "targetClassName": "Customer",
            "displayAttributeId": "c-id", "displayAttributeName": "id",
        }],
    },
    {
        "id": "cls-customer",
        "name": "Customer",
        "attributes": [_attr("c-id", "id"), _attr("c-email", "email"), _attr("c-name", "name")],
        "methods": [],
    },
    {
        "id": "cls-room",
        "name": "Room",
        "attributes": [_attr("r-id", "id"), _attr("r-number", "number", "int", isExternalId=True),
                       _attr("r-floor", "floor", "int")],
        "methods": [],
    },
    {
        "id": "cls-booking",
        "name": "Booking",
        "attributes": [_attr("b-id", "id"), _attr("b-nights", "nights", "int")],
        "methods": [],
        "associationEnds": [{
            "relationshipId": "r-2", "targetClassId": "cls-room", "targetClassName": "Room",
            "displayAttributeId": "r-id", "displayAttributeName": "id",
        }],
    },
]


def _columns(bind, page="Orders", class_id="cls-order"):
    model = _generate([{"name": page, "sections": [{"bind": bind}]}], ORDER_METADATA)
    return json.loads(_table(_nodes(model, page), class_id)["attributes"]["columns"])


def test_table_keeps_the_designed_columns_in_order():
    columns = _columns({"kind": "table", "className": "Order",
                        "columns": ["Due date", "orderNumber", "Customer"]})
    assert [(c["columnType"], c["field"]) for c in columns] == [
        ("field", "dueDate"), ("field", "orderNumber"), ("lookup", "Customer"),
    ]


def test_table_without_designed_columns_hides_the_surrogate_id():
    columns = _columns({"kind": "table", "className": "Order"})
    assert [c["field"] for c in columns] == ["orderNumber", "dueDate", "total_amount", "Customer"]


def test_table_columns_that_name_no_attribute_fall_back_to_all():
    columns = _columns({"kind": "table", "className": "Order", "columns": ["Reference", "When"]})
    assert [c["field"] for c in columns] == ["orderNumber", "dueDate", "total_amount", "Customer"]


def test_table_shows_the_id_when_the_design_asks_for_it():
    columns = _columns({"kind": "table", "className": "Order", "columns": ["ID", "orderNumber"]})
    assert [c["field"] for c in columns] == ["id", "orderNumber"]


def test_table_labels_read_as_words():
    labels = {c["field"]: c["label"] for c in _columns({"kind": "table", "className": "Order"})}
    assert labels == {
        "orderNumber": "Order number", "dueDate": "Due date",
        "total_amount": "Total amount", "Customer": "Customer",
    }


def test_related_record_is_shown_by_a_readable_attribute():
    customer = _columns({"kind": "table", "className": "Order"})[-1]
    assert (customer["lookupEntity"], customer["lookupField"]) == ("cls-customer", "name")
    # No text attribute: the natural key, never the surrogate id.
    room = _columns({"kind": "table", "className": "Booking"}, "Bookings", "cls-booking")[-1]
    assert room["lookupField"] == "number"


def test_form_and_chart_labels_split_camel_case():
    model = _generate([{"name": "Orders", "sections": [
        {"bind": {"kind": "form", "className": "Order"}},
        {"bind": {"kind": "bar_chart", "className": "Order", "labelField": "orderNumber"}},
        {"bind": {"kind": "metric_card", "className": "Order", "valueField": "total_amount",
                  "aggregation": "sum"}},
    ]}], ORDER_METADATA)
    nodes = _nodes(model, "Orders")
    assert "Order number" in [n.get("content") for n in nodes if n.get("tagName") == "label"]
    chart = next(n for n in nodes if n.get("type") == "bar-chart")
    assert [s["name"] for s in json.loads(chart["attributes"]["series"])] == ["Total amount"]
    metric = next(n for n in nodes if n.get("type") == "metric-card")
    assert metric["attributes"]["metric-title"] == "Order total amount"


@pytest.mark.parametrize("name, label", [
    ("targetDate", "Target date"), ("unit_price", "Unit price"), ("xCoordinate", "X coordinate"),
    ("mp3File", "Mp3 file"), ("sha256Hash", "Sha256 hash"),
])
def test_label_of_a_model_name(name, label):
    from diagram_handlers.types.gui_nocode_diagram_handler import _human_label

    assert _human_label(name) == label
