"""'Experimental AI design' GUIs must have working links and buttons.

Measured on three AI designs (library, hotel, tasks) before this fix: 0 of 55
buttons had a click action, 0 of 9 class methods were wired (Basic CRUD: 6/6)
and 23 links went to '/'. Causes: the converter dropped '/route' hrefs, a
plain <button> carried no action, and only the Basic CRUD page builder
emitted method buttons.

The encodings asserted here are the ones the BESSER GUI processor + web-app
generator already turn into behaviour (verified end to end): a link href equal
to the page's processor route, an action-button with data-action-type
navigate + data-target-screen=<page id>, and the Basic CRUD run-method
action-button whose data-instance-source is the page's table id.
"""

from diagram_handlers.types.gui_html_converter import html_to_components
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
        "methods": [
            {"id": "m-complete", "name": "complete", "isInstanceMethod": True, "params": []},
            {"id": "m-archive", "name": "archive", "isInstanceMethod": False, "params": []},
        ],
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


def _text(node):
    parts = [node.get("content") or ""]
    for c in node.get("components") or []:
        if isinstance(c, dict):
            parts.append(_text(c))
    return " ".join(p for p in parts if p).strip()


def _by_text(nodes, text):
    return next(n for n in nodes if _text(n) == text and n.get("tagName") in ("a", "button"))


# -- converter ---------------------------------------------------------------

def test_converter_keeps_in_app_route_href():
    a = html_to_components('<a href="/tasks">Tasks</a>')[0]
    assert a["attributes"]["href"] == "/tasks"


def test_converter_still_drops_protocol_relative_href():
    a = html_to_components('<a href="//evil.example/x">x</a>')[0]
    assert "href" not in a.get("attributes", {})


def test_converter_drops_raw_action_attributes():
    # A guessed id in raw wiring renders a half-wired button; only the handler
    # writes these, from resolved pages/methods.
    b = html_to_components(
        '<button data-action-type="run-method" data-method-class="x" '
        'data-target-screen="y" data-method="complete">Go</button>'
    )[0]
    attrs = b.get("attributes", {})
    assert "data-action-type" not in attrs
    assert "data-method-class" not in attrs
    assert "data-target-screen" not in attrs
    assert attrs["data-method"] == "complete"  # friendly vocabulary survives


# -- links and navigation -----------------------------------------------------

PAGES = [
    {"name": "Overview", "sections": [
        {"html": "<section class='app-head'><h1>Overview</h1>"
                 "<a href='/tasks'>Open tasks</a>"
                 "<a href='#'>View all projects</a>"
                 "<button class='app-btn' data-page='Projects'>Projects</button></section>"},
    ]},
    {"name": "Tasks", "sections": [
        {"bind": {"kind": "table", "className": "Task", "title": "Open tasks"},
         "html": "<section class='ds-section'><h2>Board</h2><div class='ds-card'><!--WIDGET:table--></div>"
                 "<button class='app-btn' data-method='complete'>Mark done</button></section>"},
    ]},
    {"name": "Projects", "sections": [
        {"bind": {"kind": "table", "className": "Project"}},
    ]},
]


def test_page_links_resolve_to_real_page_routes():
    model = _generate(PAGES)
    nodes = _nodes(model, "Overview")
    assert _by_text(nodes, "Open tasks")["attributes"]["href"] == "/tasks"
    # '#' used to reach the generator as an empty href and render as '/'.
    assert _by_text(nodes, "View all projects")["attributes"]["href"] == "/projects"


def test_navigation_button_targets_page_id():
    model = _generate(PAGES)
    ids = {p["name"]: p["id"] for p in model["pages"]}
    btn = _by_text(_nodes(model, "Overview"), "Projects")
    assert btn["type"] == "action-button"
    assert btn["attributes"]["data-action-type"] == "navigate"
    assert btn["attributes"]["data-target-screen"] == ids["Projects"]
    assert btn["target-screen"] == f"page:{ids['Projects']}"
    assert btn["attributes"]["class"] == "app-btn"  # authored look kept


def test_nav_header_and_route_use_the_processor_route():
    # BESSER derives '/' + name.lower().replace(' ', '-'); the old nav slug
    # ('/rooms-rates') was redirected to the start page by the app router.
    model = _generate([
        {"name": "Rooms & Rates", "sections": [{"html": "<section class='s'><h2>R</h2></section>"}]},
        {"name": "Home", "sections": [{"html": "<section class='s'><h2>H</h2></section>"}]},
    ], class_metadata=None)
    page = next(p for p in model["pages"] if p["name"] == "Rooms & Rates")
    assert page["route_path"] == "/rooms-&-rates"
    hrefs = {n["attributes"]["href"] for n in _nodes(model, "Home") if n.get("type") == "link"}
    assert "/rooms-&-rates" in hrefs


# -- method buttons -----------------------------------------------------------

def _table(nodes, class_id):
    return next(n for n in nodes if n.get("type") == "table"
                and n["attributes"].get("data-source") == class_id)


def test_data_method_button_becomes_run_method_button():
    model = _generate(PAGES)
    nodes = _nodes(model, "Tasks")
    table_id = _table(nodes, "cls-task")["attributes"]["id"]
    btn = _by_text(nodes, "Mark done")
    assert btn["type"] == "action-button"
    assert btn["attributes"]["data-action-type"] == "run-method"
    assert btn["attributes"]["data-method-class"] == "cls-task"
    assert btn["attributes"]["data-method"] == "m-complete"
    assert btn["attributes"]["data-instance-source"] == table_id
    # Already placed by the design: no duplicate in an auto-added row.
    methods = [n for n in nodes if n.get("type") == "action-button" and n.get("method") == "m-complete"]
    assert len(methods) == 1


def test_bound_table_gets_its_method_button_row():
    model = _generate([
        {"name": "Tasks", "sections": [{"bind": {"kind": "table", "className": "Task"}}]},
    ])
    nodes = _nodes(model, "Tasks")
    table_id = _table(nodes, "cls-task")["attributes"]["id"]
    buttons = [n for n in nodes if n.get("type") == "action-button"]
    assert [b["attributes"]["data-method"] for b in buttons] == ["m-complete", "m-archive"]
    assert {b["attributes"]["data-instance-source"] for b in buttons} == {table_id}


def test_table_ids_are_unique_across_pages():
    model = _generate([
        {"name": "A", "sections": [{"bind": {"kind": "table", "className": "Task"}}]},
        {"name": "B", "sections": [{"bind": {"kind": "table", "className": "Task"}}]},
    ])
    ids = [_table(_nodes(model, n), "cls-task")["attributes"]["id"] for n in ("A", "B")]
    assert len(set(ids)) == 2


def test_method_without_table_navigates_to_the_class_page():
    # Live hotel design: "Check in" on a folio page with no Booking table
    # became a MethodButton with no instance source, so it called
    # '/booking/{booking_id}/methods/...' literally and failed. The generated
    # app addresses every method through a selected row, static or not.
    model = _generate([
        {"name": "Folio", "sections": [
            {"html": "<section class='s'><h2>Folio</h2>"
                     "<button data-method='complete' data-class='Task'>Finish</button>"
                     "<button data-method='archive'>Archive</button></section>"},
        ]},
        {"name": "Board", "sections": [{"bind": {"kind": "table", "className": "Task"}}]},
    ])
    board_id = next(p["id"] for p in model["pages"] if p["name"] == "Board")
    for label in ("Finish", "Archive"):
        btn = _by_text(_nodes(model, "Folio"), label)
        assert btn["attributes"]["data-action-type"] == "navigate"
        assert btn["attributes"]["data-target-screen"] == board_id
