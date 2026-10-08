"""AI-designed record buttons and forms act on the class's data.

BESSER's web-app generator now turns
  * an action-button with data-action-type create/update/delete +
    data-entity-class (+ data-instance-source = a table on the page) into a
    button that runs through that table (add dialog, edit/delete the selected
    row; a create with the table on another page navigates there first);
  * a <form data-source=<class id>> whose inputs are named after attributes
    into a form that POSTs a record.
Measured on the saved library/hotel/tasks AI designs before this change: 0
record buttons (15 buttons, e.g. 'New task', navigated to the page they sat
on) and 0 of 1 forms submitted; its inputs also sat inside <label>s, which
BESSER flattens to text. Encodings below were checked end to end through
that generator.
"""

import copy

from diagram_handlers.types.gui_html_converter import html_to_components
from diagram_handlers.types.gui_nocode_diagram_handler import (
    GUINoCodeDiagramHandler,
    _wire_model_actions,
)
from schemas import AuthoredSystemGUISpec
from schemas.gui_diagram import GUIModificationBatchSpec
from utilities.class_metadata import extract_class_metadata

from tests.test_gui_actions import _by_text, _nodes
from tests.test_gui_aggregation import META, _generate


def _forms(nodes):
    return [n for n in nodes if n.get("tagName") == "form"]


def _inputs(form):
    out = []

    def _walk(n):
        if isinstance(n, dict):
            if n.get("tagName") == "input":
                out.append(n)
            for c in n.get("components") or []:
                _walk(c)

    _walk(form)
    return out


def _table_id(nodes, class_id):
    return next(
        n["attributes"]["id"] for n in nodes
        if n.get("type") == "table" and n["attributes"].get("data-source") == class_id
    )


# -- record buttons -------------------------------------------------------------

CATALOG = {"name": "Catalog", "sections": [
    {"bind": {"kind": "table", "className": "Book", "title": "Books"},
     "html": "<section class='ds-section'><h2>Books</h2><div class='ds-card'><!--WIDGET:table--></div>"
             "<button class='app-btn' data-page='Catalog'>New book</button>"
             "<button>Edit book</button><button>Delete selected</button>"
             "<button>Add to reading list</button></section>"},
]}


def test_add_button_next_to_the_table_creates_through_it():
    # Saved tasks design: "New task" (data-page=Tasks) on the Tasks page
    # navigated to the page it was on - a button that did nothing.
    nodes = _nodes(_generate([CATALOG]), "Catalog")
    btn = _by_text(nodes, "New book")
    assert btn["type"] == "action-button"
    assert btn["attributes"]["data-action-type"] == "create"
    assert btn["attributes"]["data-entity-class"] == "cls-book"
    assert btn["attributes"]["data-instance-source"] == _table_id(nodes, "cls-book")
    assert btn["action-type"] == "create" and btn["entity-class"] == "cls-book"
    assert btn["attributes"]["class"] == "app-btn"  # authored look kept


def test_edit_and_delete_act_on_the_selected_row():
    nodes = _nodes(_generate([CATALOG]), "Catalog")
    table_id = _table_id(nodes, "cls-book")
    edit = _by_text(nodes, "Edit book")
    assert edit["attributes"]["data-action-type"] == "update"
    assert edit["attributes"]["data-instance-source"] == table_id
    # A bare "Delete selected" means the page's only table.
    delete = _by_text(nodes, "Delete selected")
    assert delete["attributes"]["data-action-type"] == "delete"
    assert delete["attributes"]["data-entity-class"] == "cls-book"
    assert delete["instance-source"] == table_id
    assert delete["confirmation-required"] == "true"


def test_add_to_a_list_is_not_a_record_button():
    btn = _by_text(_nodes(_generate([CATALOG]), "Catalog"), "Add to reading list")
    assert btn.get("attributes", {}).get("data-action-type") not in ("create", "update", "delete")


def test_add_button_elsewhere_creates_on_the_table_page():
    model = _generate([
        {"name": "Home", "sections": [
            {"html": "<section class='s'><h1>Home</h1><button>Add book</button>"
                     "<button>Edit book</button></section>"},
        ]},
        CATALOG,
    ])
    nodes = _nodes(model, "Home")
    add = _by_text(nodes, "Add book")
    assert add["attributes"]["data-action-type"] == "create"
    assert add["attributes"]["data-entity-class"] == "cls-book"
    # No table here: the generated app finds the Book table's page itself.
    assert "data-instance-source" not in add["attributes"]
    # Edit needs a selected row, so without a table here it stays navigation.
    catalog_id = next(p["id"] for p in model["pages"] if p["name"] == "Catalog")
    edit = _by_text(nodes, "Edit book")
    assert edit["attributes"]["data-action-type"] == "navigate"
    assert edit["attributes"]["data-target-screen"] == catalog_id


def test_a_method_named_by_the_label_still_runs_the_method():
    meta = copy.deepcopy(META)
    meta[0]["methods"].append({"id": "m-remove", "name": "remove_copy", "isInstanceMethod": True, "params": []})
    nodes = _nodes(_generate([
        {"name": "Catalog", "sections": [
            {"bind": {"kind": "table", "className": "Book"},
             "html": "<section class='s'><h2>B</h2><!--WIDGET:table--><button>Remove copy</button></section>"},
        ]},
    ], meta), "Catalog")
    assert _by_text(nodes, "Remove copy")["attributes"]["data-action-type"] == "run-method"


def test_prompt_describes_record_buttons_and_create_forms():
    captured = {}

    def _predict(**kw):
        captured.update(kw)
        return AuthoredSystemGUISpec(projectName="L", pages=[CATALOG])

    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)
    handler.predict_two_pass_structured = _predict
    handler.generate_complete_system("library", class_metadata=META)
    prompt = captured["system_prompt"]
    assert "navigates (data-page) to the page holding X's table" not in prompt
    assert "'Edit X' / 'Delete X'" in prompt
    assert "<form data-class='ClassName'>" in prompt


# -- create forms -------------------------------------------------------------

def test_bound_form_widget_posts_a_record_of_its_class():
    nodes = _nodes(_generate([{"name": "Join", "sections": [
        {"bind": {"kind": "form", "className": "Book", "title": "Add a book", "columns": ["Title", "Notes"]}},
    ]}]), "Join")
    form = _forms(nodes)[0]
    assert form["attributes"]["data-source"] == "cls-book"
    inputs = _inputs(form)
    # Named columns first, then every attribute a record needs; never the
    # derived score the server computes.
    assert [i["attributes"]["name"] for i in inputs] == ["title", "notes", "genre", "copies"]
    assert inputs[3]["attributes"]["type"] == "number"
    labels = [n for n in nodes if n.get("tagName") == "label"]
    assert [lbl["attributes"]["for"] for lbl in labels] == [i["attributes"]["id"] for i in inputs]
    submit = next(n for n in nodes if n.get("tagName") == "button")
    assert submit["attributes"]["type"] == "submit" and submit.get("type") != "action-button"
    assert submit["content"] == "Add book"


def test_form_widget_without_a_class_stays_unbound():
    # A contact form in a one-class app must not post to that class.
    form = _forms(_nodes(_generate([{"name": "Contact", "sections": [
        {"bind": {"kind": "form", "title": "Contact us", "columns": ["Name", "Message"]}},
    ]}], META[:1]), "Contact"))[0]
    assert "data-source" not in form.get("attributes", {})


def _html_form_page(markup):
    return [{"name": "Page", "sections": [{"html": f"<section class='s'><h2>F</h2>{markup}</section>"}]}]


def test_authored_form_with_data_class_is_bound():
    nodes = _nodes(_generate(_html_form_page(
        "<form data-class='Member'><label for='n'>Name</label><input id='n'>"
        "<label for='e'>E-mail</label><input id='e' name='email'>"
        "<button type='submit' data-page='Page'>Join</button></form>"
    )), "Page")
    form = _forms(nodes)[0]
    assert form["attributes"]["data-source"] == "cls-member"
    assert "data-class" not in form["attributes"]
    assert [i["attributes"]["name"] for i in _inputs(form)] == ["name", "email"]
    # The bound form's own submit is left for the form, not turned into navigation.
    assert _by_text(nodes, "Join").get("type") != "action-button"


def test_authored_form_is_inferred_from_its_labels():
    # Saved hotel design: a guest form whose inputs had only <label>s and
    # submitted nothing.
    form = _forms(_nodes(_generate(_html_form_page(
        "<form><label>Due date</label><input type='date'>"
        "<input type='number' placeholder='Fine'><button>Save</button></form>"
    )), "Page"))[0]
    assert form["attributes"]["data-source"] == "cls-loan"
    assert [i["attributes"]["name"] for i in _inputs(form)] == ["due_date", "fine"]


def test_inputs_wrapped_in_labels_are_kept():
    # Live tasks design: <label class='ds-field'><span>Title</span><input></label>.
    # BESSER turns a label into plain text, so the bound form lost every input.
    nodes = html_to_components(
        "<form><label class='ds-field'><span class='ds-label'>Title</span>"
        "<input class='ds-input' name='title'></label>"
        "<label><input type='checkbox' name='done'> Done</label></form>"
    )
    field, check = nodes[0]["components"]
    assert field["tagName"] == "div" and field["attributes"]["class"] == "ds-field"
    label, control = field["components"]
    assert (label["tagName"], label["content"], label["attributes"]) == (
        "label", "Title", {"class": "ds-label", "for": "title"},
    )
    assert control["attributes"]["name"] == "title"
    # Text after a checkbox still labels that checkbox, not the next input.
    assert check["components"][0]["attributes"]["for"] == "done"
    # A label holding only text is untouched.
    plain = html_to_components("<label for='x'>Plain</label>")[0]
    assert plain == {"tagName": "label", "attributes": {"for": "x"}, "content": "Plain"}


def test_search_contact_and_lookup_forms_stay_unbound():
    for markup in (
        "<form><input type='search' name='q'><button>Search</button></form>",
        "<form><input name='name'><input name='email'><textarea name='message'></textarea><button>Send</button></form>",
        # names Book attributes but cannot create one: genre and copies missing
        "<form><input name='title'><button>Find</button></form>",
    ):
        form = _forms(_nodes(_generate(_html_form_page(markup)), "Page"))[0]
        assert "data-source" not in (form.get("attributes") or {}), markup


def test_converter_still_drops_a_raw_form_data_source():
    form = html_to_components("<form data-source='cls-x' data-class='Book'></form>")[0]
    assert "data-source" not in form["attributes"]
    assert form["attributes"]["data-class"] == "Book"


def test_class_metadata_carries_what_a_create_form_needs():
    model = {"elements": {
        "c": {"type": "Class", "name": "Room"},
        "a": {"type": "ClassAttribute", "owner": "c", "name": "number", "attributeType": "int", "isId": True},
        "b": {"type": "ClassAttribute", "owner": "c", "name": "note", "attributeType": "str", "isOptional": True},
        "d": {"type": "ClassAttribute", "owner": "c", "name": "total", "attributeType": "float", "isDerived": True},
        "e": {"type": "ClassAttribute", "owner": "c", "name": "floor", "attributeType": "int", "defaultValue": "1"},
    }, "relationships": {}}
    attrs = {a["name"]: a for a in extract_class_metadata(model)[0]["attributes"]}
    assert attrs["number"]["isId"] and not attrs["number"]["isOptional"]
    assert attrs["note"]["isOptional"]
    assert attrs["total"]["isDerived"]
    assert attrs["floor"]["hasDefault"] and not attrs["note"]["hasDefault"]


# -- modification path ---------------------------------------------------------

def _modify(model, operations):
    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)
    handler.predict_structured = lambda *a, **k: GUIModificationBatchSpec(operations=operations)
    res = handler.generate_modification(
        "add a members page", copy.deepcopy(model), raw_request="add a members page", class_metadata=META,
    )
    assert res["action"] == "modify_model"
    return res["model"]


def test_modification_wires_record_buttons_and_forms_once():
    model = _modify(_generate([CATALOG]), [{
        "operation": "add_page", "pageName": "Members", "newPageName": "Members",
        "section": {
            "bind": {"kind": "table", "className": "Member"},
            "html": "<section class='s'><h2>Members</h2><!--WIDGET:table-->"
                    "<button>New member</button>"
                    "<form><input name='name'><input name='email'><button>Join</button></form></section>",
        },
    }])
    nodes = _nodes(model, "Members")
    btn = _by_text(nodes, "New member")
    assert btn["attributes"]["data-action-type"] == "create"
    assert btn["attributes"]["data-instance-source"] == _table_id(nodes, "cls-member")
    assert _forms(nodes)[0]["attributes"]["data-source"] == "cls-member"
    again = copy.deepcopy(model)
    _wire_model_actions(again, META)
    assert again == model
