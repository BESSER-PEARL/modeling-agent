"""Modifying an AI-designed GUI must wire new content like generation does.

Links, navigation buttons, method buttons, method rows and stable ids were
only produced by the complete-system path. A page or section added through
``generate_modification`` kept '#' links, action-less buttons, no method row
under its tables and no page id, so a button pointing at it could not target
it. The same page pass now runs over the whole modified model; it has to be a
no-op on content that is already wired.
"""

import copy

from diagram_handlers.types.gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from schemas import AuthoredSystemGUISpec
from schemas.gui_diagram import GUIModificationBatchSpec

from tests.test_gui_actions import PAGES, TASK_METADATA, _by_text, _nodes, _table, _text


def _handler():
    return GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)


def _generate():
    handler = _handler()
    handler.predict_two_pass_structured = (
        lambda **kw: AuthoredSystemGUISpec(projectName="Tracker", pages=PAGES)
    )
    return handler.generate_complete_system("tracker", class_metadata=TASK_METADATA)["model"]


def _modify(model, operations, request="add a reports page"):
    handler = _handler()
    handler.predict_structured = (
        lambda *a, **k: GUIModificationBatchSpec(operations=operations)
    )
    res = handler.generate_modification(
        request, copy.deepcopy(model), raw_request=request, class_metadata=TASK_METADATA,
    )
    assert res["action"] == "modify_model"
    return res["model"]


ADD_REPORTS = [
    {
        "operation": "add_page",
        "pageName": "Reports",
        "newPageName": "Reports",
        "section": {
            "bind": {"kind": "table", "className": "Task", "title": "Task report"},
            "html": "<section class='ds-section'><h2>Task report</h2>"
                    "<a href='#'>Back to overview</a>"
                    "<button class='app-btn' data-page='Tasks'>Open the board</button>"
                    "<div class='ds-card'><!--WIDGET:table--></div></section>",
        },
    },
    {
        "operation": "append_section",
        "pageName": "Overview",
        "section": {
            "html": "<section class='ds-section'><h2>Reporting</h2>"
                    "<button data-page='Reports'>See reports</button></section>",
        },
    },
]


def _page(model, name):
    return next(p for p in model["pages"] if p["name"] == name)


def _method_buttons(nodes, method_id):
    return [n for n in nodes if n.get("type") == "action-button" and n.get("method") == method_id]


def test_added_page_gets_an_id_and_existing_ids_are_kept():
    before = _generate()
    after = _modify(before, ADD_REPORTS)
    assert {p["name"]: p["id"] for p in before["pages"]}.items() <= {
        p["name"]: p["id"] for p in after["pages"]
    }.items()
    assert _page(after, "Reports").get("id")
    # Pages the edit did not touch are left as they were, apart from the new
    # page's entry in their nav header (components[0]).
    for name in ("Tasks", "Projects"):
        body = lambda m: _page(m, name)["frames"][0]["component"]["components"][1:]
        assert body(after) == body(before)


def test_added_page_links_and_buttons_are_wired():
    after = _modify(_generate(), ADD_REPORTS)
    ids = {p["name"]: p["id"] for p in after["pages"]}
    nodes = _nodes(after, "Reports")
    assert _by_text(nodes, "Back to overview")["attributes"]["href"] == "/overview"
    board = _by_text(nodes, "Open the board")
    assert board["type"] == "action-button"
    assert board["attributes"]["data-action-type"] == "navigate"
    assert board["attributes"]["data-target-screen"] == ids["Tasks"]


def test_existing_page_button_resolves_to_the_new_page():
    after = _modify(_generate(), ADD_REPORTS)
    btn = _by_text(_nodes(after, "Overview"), "See reports")
    assert btn["attributes"]["data-action-type"] == "navigate"
    assert btn["attributes"]["data-target-screen"] == _page(after, "Reports")["id"]


def test_added_table_gets_one_method_row():
    after = _modify(_generate(), ADD_REPORTS)
    nodes = _nodes(after, "Reports")
    table_id = _table(nodes, "cls-task")["attributes"]["id"]
    for method_id in ("m-complete", "m-archive"):
        buttons = _method_buttons(nodes, method_id)
        assert len(buttons) == 1
        assert buttons[0]["attributes"]["data-instance-source"] == table_id


def test_restyle_after_add_page_changes_no_wiring():
    once = _modify(_generate(), ADD_REPORTS)
    # Deterministic recolor path: must not add rows or move ids.
    handler = _handler()
    res = handler.generate_modification(
        "change the color to red", copy.deepcopy(once),
        raw_request="change the color to red", class_metadata=TASK_METADATA,
    )
    again = res["model"]
    for name in ("Overview", "Tasks", "Reports"):
        for method_id in ("m-complete", "m-archive"):
            assert len(_method_buttons(_nodes(again, name), method_id)) == len(
                _method_buttons(_nodes(once, name), method_id)
            )
    assert [p["id"] for p in again["pages"]] == [p["id"] for p in once["pages"]]


def test_page_pass_is_a_no_op_on_a_wired_model():
    from diagram_handlers.types.gui_nocode_diagram_handler import _wire_model_actions

    generated = _generate()
    rewired = copy.deepcopy(generated)
    _wire_model_actions(rewired, TASK_METADATA)
    assert rewired == generated

    modified = _modify(generated, ADD_REPORTS)
    twice = copy.deepcopy(modified)
    _wire_model_actions(twice, TASK_METADATA)
    assert twice == modified


def test_renamed_page_keeps_its_incoming_links():
    before = _generate()
    handler = _handler()
    res = handler.generate_modification(
        "rename Tasks to Task Board", copy.deepcopy(before),
        raw_request="rename Tasks to Task Board", class_metadata=TASK_METADATA,
    )
    after = res["model"]
    assert _page(after, "Task Board")["id"] == _page(before, "Tasks")["id"]
    link = _by_text(_nodes(after, "Overview"), "Open tasks")
    assert link["attributes"]["href"] == "/task-board"


def test_single_element_is_wired():
    from schemas.gui_diagram import SingleGUIElementSpec

    handler = _handler()
    handler.predict_structured = lambda *a, **k: SingleGUIElementSpec(
        pageName="Tasks",
        section={
            "bind": {"kind": "table", "className": "Task", "title": "Open tasks"},
            "html": "<section class='ds-section'><h2>Open tasks</h2><a href='#'>Home</a>"
                    "<div class='ds-card'><!--WIDGET:table--></div></section>",
        },
    )
    res = handler.generate_single_element("add a task table", class_metadata=TASK_METADATA)
    nodes = _nodes(res["model"], "Tasks")
    assert _by_text(nodes, "Home")["attributes"]["href"] == "/home"
    for method_id in ("m-complete", "m-archive"):
        assert len(_method_buttons(nodes, method_id)) == 1


def test_modification_drops_a_stray_widget_marker():
    # A live hotel design saved before surplus markers were handled kept a
    # 'widget-slot' node, and it survives the editor round-trip.
    before = _generate()
    overview = before["pages"][0]["frames"][0]["component"]["components"]
    overview.append({"type": "widget-slot", "attributes": {"data-widget-slot": "table"}})
    after = _modify(before, ADD_REPORTS)
    slots = [n for p in after["pages"] for n in _nodes(after, p["name"]) if n.get("type") == "widget-slot"]
    assert slots == []


def test_modification_prompt_states_the_interaction_vocabulary():
    # The page pass reads intent from href='/route', data-page and
    # data-method; the modify prompt never asked for them.
    seen = {}

    def _capture(*a, **k):
        seen["prompt"] = k["system_prompt"]
        return GUIModificationBatchSpec(operations=ADD_REPORTS)

    handler = _handler()
    handler.predict_structured = _capture
    handler.generate_modification(
        "add a reports page", _generate(), raw_request="add a reports page",
        class_metadata=TASK_METADATA,
    )
    for marker in ("<a href='/route'>", "data-page='Page Name'", "data-method='methodName'"):
        assert marker in seen["prompt"]


# -- navigation after page edits ---------------------------------------------

def _nav_routes(model, page_name):
    """Hrefs of the auto-injected nav header on *page_name*, in order."""
    page = _page(model, page_name)
    comps = page["frames"][0]["component"]["components"]
    nav = next(c for c in comps if "assistant-nav-header" in c.get("attributes", {}).get("class", ""))
    out = []

    def _walk(n):
        if isinstance(n, dict):
            if n.get("type") == "link" or n.get("tagName") == "a":
                out.append(n["attributes"]["href"])
            for c in n.get("components") or []:
                _walk(c)

    _walk(nav)
    return out


def test_added_page_is_in_every_nav_header():
    after = _modify(_generate(), ADD_REPORTS)
    for name in ("Overview", "Tasks", "Projects"):
        assert _nav_routes(after, name) == ["/overview", "/tasks", "/projects", "/reports"]


def test_added_page_gets_the_sibling_header_with_itself_active():
    after = _modify(_generate(), ADD_REPORTS)
    assert _nav_routes(after, "Reports") == ["/overview", "/tasks", "/projects", "/reports"]
    tasks_nav = _page(after, "Tasks")["frames"][0]["component"]["components"][0]
    reports_nav = _page(after, "Reports")["frames"][0]["component"]["components"][0]
    links = lambda nav: {l["attributes"]["href"]: l for l in _links_in(nav)}
    # Active look moves from the sibling's own entry to the new page's entry.
    assert links(reports_nav)["/reports"]["style"] == links(tasks_nav)["/tasks"]["style"]
    assert links(reports_nav)["/tasks"]["style"] == links(tasks_nav)["/overview"]["style"]


def _links_in(node):
    out = []
    if isinstance(node, dict):
        if node.get("type") == "link" or node.get("tagName") == "a":
            out.append(node)
        for c in node.get("components") or []:
            out += _links_in(c)
    return out


def test_nav_entries_are_not_duplicated_on_rerun():
    from diagram_handlers.types.gui_nocode_diagram_handler import _wire_model_actions

    after = _modify(_generate(), ADD_REPORTS)
    again = copy.deepcopy(after)
    _wire_model_actions(again, TASK_METADATA)
    assert again == after
    restyled = _handler().generate_modification(
        "change the color to red", copy.deepcopy(after),
        raw_request="change the color to red", class_metadata=TASK_METADATA,
    )["model"]
    assert _nav_routes(restyled, "Tasks") == ["/overview", "/tasks", "/projects", "/reports"]


def test_authored_nav_listing_pages_gets_the_new_page():
    handler = _handler()
    pages = [dict(p) for p in PAGES]
    pages[0] = {"name": "Overview", "sections": [
        {"html": "<section class='s'><h2>Overview</h2><nav class='app-tabs'><ul>"
                 "<li class='tab'><a href='/overview'>Overview</a></li>"
                 "<li class='tab'><a href='/tasks'>Tasks</a></li></ul></nav></section>"},
    ]}
    handler.predict_two_pass_structured = (
        lambda **kw: AuthoredSystemGUISpec(projectName="Tracker", pages=pages)
    )
    before = handler.generate_complete_system("tracker", class_metadata=TASK_METADATA)["model"]
    after = _modify(before, ADD_REPORTS)
    tabs = next(n for n in _nodes(after, "Overview") if n.get("tagName") == "ul")
    assert [_links_in(li)[0]["attributes"]["href"] for li in tabs["components"]] == [
        "/overview", "/tasks", "/reports"
    ]
    assert tabs["components"][-1]["attributes"]["class"] == "tab"
    assert _text(tabs["components"][-1]) == "Reports"


def test_removed_page_leaves_every_nav():
    before = _generate()
    after = _modify(before, [{"operation": "remove_page", "pageName": "Projects"}], "remove projects")
    for name in ("Overview", "Tasks"):
        assert _nav_routes(after, name) == ["/overview", "/tasks"]


def test_renamed_page_is_renamed_in_every_nav():
    before = _generate()
    after = _modify(
        before,
        [{"operation": "rename_page", "pageName": "Projects", "newPageName": "Portfolio"}],
        "rename projects",
    )
    for name in ("Overview", "Tasks", "Portfolio"):
        links = _links_in(_page(after, name)["frames"][0]["component"]["components"][0])
        assert [(l["attributes"]["href"], _text(l)) for l in links][-1] == ("/portfolio", "Portfolio")


def test_modification_lifts_a_saved_link_out_of_a_paragraph():
    # Exact node from the saved live hotel design (also after the editor
    # round-trip, where the <p> may carry type 'text').
    from tests.test_gui_actions import _flattened_actions

    before = _generate()
    crumb = {"tagName": "p", "type": "text", "components": [
        {"tagName": "a", "attributes": {"href": "/tasks"}, "content": "Tasks"},
        {"type": "textnode", "content": " / Task #58241"},
    ]}
    before["pages"][0]["frames"][0]["component"]["components"].append(crumb)
    after = _modify(before, ADD_REPORTS)
    for page in after["pages"]:
        assert _flattened_actions(page["frames"][0]["component"]) == []
    lifted = next(n for n in _nodes(after, "Overview") if _text(n) == "Tasks / Task #58241")
    assert lifted["tagName"] == "div" and "type" not in lifted
    assert _text(lifted) == "Tasks / Task #58241"


def test_nav_of_task_links_is_not_treated_as_a_page_list():
    # Live hotel design: a guest nav with "Manage booking" and "Staff sign in"
    # (links to two pages, not named after them) got a staff "Billing" entry.
    handler = _handler()
    pages = [dict(p) for p in PAGES]
    pages[0] = {"name": "Overview", "sections": [
        {"html": "<section class='s'><h2>Overview</h2><nav class='guest-nav'>"
                 "<a href='/tasks'>Manage my work</a><a href='/projects'>Staff sign in</a>"
                 "</nav></section>"},
    ]}
    handler.predict_two_pass_structured = (
        lambda **kw: AuthoredSystemGUISpec(projectName="Tracker", pages=pages)
    )
    before = handler.generate_complete_system("tracker", class_metadata=TASK_METADATA)["model"]
    after = _modify(before, ADD_REPORTS)
    guest = next(n for n in _nodes(after, "Overview") if n.get("tagName") == "nav"
                 and n["attributes"].get("class") == "guest-nav")
    assert [l["attributes"]["href"] for l in _links_in(guest)] == ["/tasks", "/projects"]
