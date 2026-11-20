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

import pytest

from diagram_handlers.types.gui_html_converter import html_to_components
from diagram_handlers.types.gui_nocode_diagram_handler import (
    GUINoCodeDiagramHandler,
    _node_text,
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



def _tables(nodes):
    return {n["attributes"]["chart-title"]: n for n in nodes if n.get("type") == "table"}


def test_read_only_table_has_no_add_edit_delete():
    # A table that only shows records (results, a list to browse) must not
    # turn its screen into a record-management page.
    nodes = _nodes(_generate([{"name": "Overview", "sections": [
        {"bind": {"kind": "table", "className": "Book", "title": "Manage books", "editable": True}},
        {"bind": {"kind": "table", "className": "Loan", "title": "Past loans", "editable": False}},
        {"bind": {"kind": "table", "className": "Member", "title": "Members"}},
        {"bind": {"kind": "dashboard", "className": "Loan", "title": "Loan overview", "editable": False}},
    ]}]), "Overview")
    tables = _tables(nodes)
    assert tables["Manage books"]["attributes"]["action-buttons"] == "true"
    assert tables["Past loans"]["attributes"]["action-buttons"] == "false"
    assert tables["Loan Data"]["attributes"]["action-buttons"] == "false"
    # Not stated: the table keeps its add/edit/delete actions.
    assert tables["Members"]["attributes"]["action-buttons"] == "true"


def test_bind_schema_and_prompt_offer_read_only_tables():
    from prompts import gui as gui_prompts
    from schemas.gui_diagram import GUIBindSpec

    assert GUIBindSpec(kind="table", editable=False).editable is False
    assert GUIBindSpec(kind="table").editable is None
    authoring = gui_prompts.complete_system_prompt("default", "", "", "", "")
    modifying = gui_prompts.modification_prompt("- Home: hero", META, "")
    for prompt in (authoring, modifying):
        assert '"editable": false' in prompt


def test_prompt_rules_out_dead_filter_inputs_next_to_a_bound_table():
    # The generated table sorts and filters in its own header; an input drawn
    # outside a create form is wired to nothing.
    from prompts import gui as gui_prompts

    rules = gui_prompts.interactions_block(True)
    assert "already sorts and filters" in rules
    assert "no search box, filter input or sort control" in rules

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


# -- records that must be linked to another record -----------------------------

def _end(element, multiplicity, role):
    return {"element": element, "multiplicity": multiplicity, "role": role}


# An order line is part of exactly one order: its create call needs the order.
ORDER_META = extract_class_metadata({
    "elements": {
        "c-order": {"id": "c-order", "type": "Class", "name": "Order"},
        "a-number": {"id": "a-number", "type": "ClassAttribute", "owner": "c-order",
                     "name": "number", "attributeType": "str"},
        "c-line": {"id": "c-line", "type": "Class", "name": "OrderLine"},
        "a-qty": {"id": "a-qty", "type": "ClassAttribute", "owner": "c-line",
                  "name": "quantity", "attributeType": "int"},
        "a-note": {"id": "a-note", "type": "ClassAttribute", "owner": "c-line",
                   "name": "note", "attributeType": "str", "isOptional": True},
    },
    "relationships": {"r1": {"type": "ClassComposition", "source": _end("c-line", "0..*", "lines"),
                             "target": _end("c-order", "1", "order")}},
})

LINE_FORM = (
    "<form data-class='OrderLine' class='app-form'><label for='o'>Order</label>"
    "<select id='o' name='order'></select><label for='q'>Quantity</label>"
    "<input id='q' type='number'><label for='n'>Note</label><input id='n'>"
    "<button type='submit' class='app-save'>Save line</button></form>"
)


def test_designed_form_of_a_record_that_needs_a_parent_stays_a_form():
    # The generated app adds the select of the parent record to a bound form,
    # so the design keeps its form, controls and submit button as authored.
    model = _generate([{"name": "Lines", "sections": [
        {"bind": {"kind": "table", "className": "OrderLine", "title": "Lines"}},
        {"html": f"<section class='s'><h2>New line</h2>{LINE_FORM}</section>"},
    ]}], ORDER_META)
    form, = _forms(_nodes(model, "Lines"))
    authored, = html_to_components(LINE_FORM)
    # Bound to its class; only the two inputs gain their attribute's name, and
    # the quantity a line needs is required.
    assert form["attributes"] == {"class": "app-form", "data-source": "c-line"}
    authored["attributes"] = form["attributes"]
    authored["components"][3]["attributes"].update(name="quantity", required="required")
    authored["components"][5]["attributes"]["name"] = "note"
    assert form == authored
    again = copy.deepcopy(model)
    _wire_model_actions(again, ORDER_META)
    assert again == model


def test_form_widget_of_such_a_record_stays_a_form():
    nodes = _nodes(_generate([{"name": "Entry", "sections": [
        {"bind": {"kind": "form", "className": "OrderLine"}},
    ]}], ORDER_META), "Entry")
    form, = _forms(nodes)
    assert form["attributes"]["data-source"] == "c-line"
    assert [i["attributes"]["name"] for i in _inputs(form)] == ["quantity"]


# -- enumeration and required fields on create forms ---------------------------

# Literals deliberately out of alphabetical order: the form keeps the model's order.
SHOP_META = extract_class_metadata({
    "elements": {
        "c-sale": {"id": "c-sale", "type": "Class", "name": "Sale"},
        "a-ref": {"id": "a-ref", "type": "ClassAttribute", "owner": "c-sale",
                  "name": "reference", "attributeType": "str"},
        "a-state": {"id": "a-state", "type": "ClassAttribute", "owner": "c-sale",
                    "name": "state", "attributeType": "SaleState"},
        "a-memo": {"id": "a-memo", "type": "ClassAttribute", "owner": "c-sale",
                   "name": "memo", "attributeType": "str", "isOptional": True},
        "a-paid": {"id": "a-paid", "type": "ClassAttribute", "owner": "c-sale",
                   "name": "paid", "attributeType": "bool"},
        "a-channel": {"id": "a-channel", "type": "ClassAttribute", "owner": "c-sale",
                      "name": "channel", "attributeType": "Channel", "isOptional": True},
        "e-channel": {"id": "e-channel", "type": "Enumeration", "name": "Channel"},
        "l-online": {"id": "l-online", "type": "ClassAttribute", "owner": "e-channel", "name": "Online"},
        "l-store": {"id": "l-store", "type": "ClassAttribute", "owner": "e-channel", "name": "Store"},
        "e-state": {"id": "e-state", "type": "Enumeration", "name": "SaleState"},
        "l-1": {"id": "l-1", "type": "ClassAttribute", "owner": "e-state", "name": "Placed"},
        "l-2": {"id": "l-2", "type": "ClassAttribute", "owner": "e-state", "name": "Shipped"},
        "l-3": {"id": "l-3", "type": "ClassAttribute", "owner": "e-state", "name": "Delivered"},
        "l-4": {"id": "l-4", "type": "ClassAttribute", "owner": "e-state", "name": "Cancelled"},
    },
    "relationships": {},
})
SALE_STATES = ["Placed", "Shipped", "Delivered", "Cancelled"]


def _controls(form):
    out = []

    def _walk(n):
        if isinstance(n, dict):
            if n.get("tagName") in ("input", "select", "textarea"):
                out.append(n)
            for c in n.get("components") or []:
                _walk(c)

    _walk(form)
    return {c["attributes"]["name"]: c for c in out}


def _assert_enum_select(control):
    assert control["tagName"] == "select"
    assert "type" not in control["attributes"]
    # BESSER's form converter reads the choices from data-options.
    assert control["attributes"]["data-options"] == ",".join(SALE_STATES)
    assert [(o["tagName"], o["attributes"]["value"], o["content"]) for o in control["components"]] == [
        ("option", s, s) for s in SALE_STATES
    ]


def test_form_widget_offers_enumeration_literals_as_a_select():
    form, = _forms(_nodes(_generate([{"name": "New sale", "sections": [
        {"bind": {"kind": "form", "className": "Sale", "columns": ["reference", "state", "memo"]}},
    ]}], SHOP_META), "New sale"))
    controls = _controls(form)
    _assert_enum_select(controls["state"])
    assert controls["reference"]["attributes"]["type"] == "text"


def _options(select):
    return [(o["attributes"]["value"], o.get("content")) for o in select["components"]]


def test_an_optional_enumeration_select_starts_with_an_empty_choice():
    # Without it the first literal is preselected and saved for a field the user left alone.
    form, = _forms(_nodes(_generate([{"name": "New sale", "sections": [
        {"bind": {"kind": "form", "className": "Sale", "columns": ["reference", "state", "channel"]}},
    ]}], SHOP_META), "New sale"))
    controls = _controls(form)
    assert _options(controls["channel"]) == [("", ""), ("Online", "Online"), ("Store", "Store")]
    assert "required" not in controls["channel"]["attributes"]
    # BESSER's converter reads the choices from data-options: literals only.
    assert controls["channel"]["attributes"]["data-options"] == "Online,Store"
    # A required one has no empty choice and keeps "required".
    assert _options(controls["state"]) == [(s, s) for s in SALE_STATES]
    assert controls["state"]["attributes"]["required"] == "required"


def test_an_authored_control_of_an_optional_enumeration_starts_with_an_empty_choice():
    form, = _forms(_nodes(_generate([{"name": "Page", "sections": [{"html": (
        "<section class='s'><h2>F</h2><form data-class='Sale'>"
        "<label for='r'>Reference</label><input id='r'>"
        "<label for='s'>State</label><input id='s'>"
        "<label for='c'>Channel</label><select id='c'><option value=''>Choose one</option>"
        "<option value='Online'>Online</option><option value='Store'>In store</option></select>"
        "<label for='p'>Paid</label><input id='p' type='checkbox'>"
        "<button type='submit'>Save</button></form></section>"
    )}]}], SHOP_META), "Page"))
    channel = _controls(form)["channel"]
    assert _options(channel)[0] == ("", "")
    assert [value for value, _ in _options(channel)] == ["", "Online", "Store"]


def test_authored_form_input_of_an_enumeration_becomes_a_select():
    form, = _forms(_nodes(_generate([{"name": "Page", "sections": [{"html": (
        "<section class='s'><h2>F</h2><form data-class='Sale'>"
        "<label for='r'>Reference</label><input id='r'>"
        "<label for='s'>State</label><input id='s' list='states' placeholder='Placed or Shipped'>"
        "<label for='p'>Paid</label><input id='p' type='checkbox'>"
        "<button type='submit'>Save</button></form></section>"
    )}]}], SHOP_META), "Page"))
    assert form["attributes"]["data-source"] == "c-sale"
    state = _controls(form)["state"]
    _assert_enum_select(state)
    assert "list" not in state["attributes"] and "placeholder" not in state["attributes"]
    assert state["attributes"]["id"] == "s"



def test_form_widget_marks_the_fields_a_record_needs_as_required():
    form, = _forms(_nodes(_generate([{"name": "New sale", "sections": [
        {"bind": {"kind": "form", "className": "Sale", "columns": ["reference", "state", "memo", "paid"]}},
    ]}], SHOP_META), "New sale"))
    controls = _controls(form)
    assert controls["reference"]["attributes"].get("required") == "required"
    assert controls["state"]["attributes"].get("required") == "required"
    assert "required" not in controls["memo"]["attributes"]
    # A required checkbox would have to be ticked; unticked already sends false.
    assert "required" not in controls["paid"]["attributes"]


def test_a_request_for_a_simple_design_overrides_the_signature_moment_rule():
    from prompts import gui as gui_prompts

    prompt = gui_prompts.complete_system_prompt("default", "", "", "", "")
    rule = prompt[prompt.index("6. VISUAL HIERARCHY"):prompt.index("7. SAMPLE DATA")]
    assert "simple, minimal or plain design" in rule
    assert "no signature moment" in rule
    # It drops decoration, not the varied sample data or realistic labels.
    assert "realism" not in rule[rule.index("EXCEPTION"):]


def test_editable_rule_is_given_only_with_a_class_diagram():
    from prompts import gui as gui_prompts

    assert '"editable"' not in gui_prompts.modification_prompt("- Home: hero", [], "")


def _radio_form(choices):
    form, = _forms(_nodes(_generate([{"name": "Page", "sections": [{"html": (
        "<section class='s'><h2>F</h2><form data-class='Sale'>"
        "<label for='r'>Reference</label><input id='r'>" + choices +
        "<label for='p'>Paid</label><input id='p' type='checkbox'>"
        "<button type='submit'>Save</button></form></section>"
    )}]}], SHOP_META), "Page"))
    assert form["attributes"]["data-source"] == "c-sale"
    return form


@pytest.mark.parametrize("choices", [
    "<input type='radio' name='state' value='Placed'> Placed"
    "<input type='radio' name='state' value='Shipped'> Shipped",
    "<fieldset><legend>State</legend>"
    "<label><input type='radio' name='state' value='Placed'> Placed</label>"
    "<label><input type='radio' name='state' value='Shipped'> Shipped</label></fieldset>",
    "<div class='ds-field'><span>State</span>"
    "<input type='radio' id='s1' name='state' value='Placed'><label for='s1'>Placed</label>"
    "<input type='radio' id='s2' name='state' value='Shipped'><label for='s2'>Shipped</label></div>",
    "<div>Placed <input type='radio' name='state' value='Placed'>"
    " Shipped <input type='radio' name='state' value='Shipped'></div>",
    "<div><input type='radio' name='state' value='Placed'><span>Placed</span>"
    "<input type='radio' name='state' value='Shipped'><span>Shipped</span></div>",
    "<div><div><input type='radio' id='s1' name='state' value='Placed'><label for='s1'>Placed</label></div>"
    "<div><input type='radio' id='s2' name='state' value='Shipped'><label for='s2'>Shipped</label></div></div>",
])
def test_authored_radio_group_of_an_enumeration_keeps_its_radios_and_captions(choices):
    form = _radio_form(choices)
    nodes = _nodes_of(form)
    radios = [n for n in nodes if (n.get("attributes") or {}).get("type") == "radio"]
    assert [(r["tagName"], r["attributes"]["name"]) for r in radios] == [("input", "state")] * 2
    assert not [n for n in nodes if n.get("tagName") == "select"]
    assert "Placed" in _node_text(form) and "Shipped" in _node_text(form)
    # HTML requires a choice from the group when one radio of the name is required
    assert [r["attributes"].get("required") for r in radios] == ["required", None]


def test_authored_bound_form_marks_the_fields_a_record_needs_as_required():
    form, = _forms(_nodes(_generate([{"name": "Page", "sections": [{"html": (
        "<section class='s'><h2>F</h2><form data-class='Sale'>"
        "<label for='r'>Reference</label><input id='r'>"
        "<label for='s'>State</label><input id='s'>"
        "<label for='m'>Memo</label><input id='m'>"
        "<label for='p'>Paid</label><input id='p' type='checkbox'>"
        "<button type='submit'>Save</button></form></section>"
    )}]}], SHOP_META), "Page"))
    controls = _controls(form)
    assert controls["reference"]["attributes"].get("required") == "required"
    assert controls["state"]["attributes"].get("required") == "required"
    assert "required" not in controls["memo"]["attributes"]
    assert "required" not in controls["paid"]["attributes"]


def _nodes_of(node):
    out = [node]
    for child in node.get("components") or []:
        out.extend(_nodes_of(child))
    return out


def test_a_class_is_managed_on_its_editable_table_page_not_a_read_only_one():
    tables = [
        {"name": "Archive", "sections": [
            {"bind": {"kind": "table", "className": "Book", "title": "Past books", "editable": False}}]},
        {"name": "Books", "sections": [
            {"bind": {"kind": "table", "className": "Book", "title": "All books"}}]},
    ]
    home = {"html": "<section class='s'><h2>Home</h2>"
                    "<button data-method='reserve' data-class='Book'>Reserve a book</button></section>"}
    # The method runs on a selected row, so the button sends the user to the
    # page where Book records are managed: when generated, and when the page
    # is added by an edit.
    generated = _generate(tables + [{"name": "Home", "sections": [home]}])
    modified = _modify(_generate(tables), [
        {"operation": "add_page", "pageName": "Home", "newPageName": "Home", "section": home},
    ])
    for model in (generated, modified):
        books_id = next(p["id"] for p in model["pages"] if p["name"] == "Books")
        btn = _by_text(_nodes(model, "Home"), "Reserve a book")
        assert btn["attributes"]["data-target-screen"] == books_id


def test_rewriting_a_read_only_table_section_keeps_it_read_only():
    model = _generate([{"name": "Archive", "sections": [{
        "bind": {"kind": "table", "className": "Loan", "title": "Past loans", "editable": False},
        "html": "<section class='ds-section'><h2>Loan history</h2><div class='ds-card'><!--WIDGET:table--></div></section>",
    }]}])

    def rewrite(bind):
        return _modify(model, [{"operation": "edit_section", "pageName": "Archive", "sectionTitle": "Loan history",
                                "section": {"bind": {"kind": "table", "className": "Loan", **bind}}}])

    unstated = next(n for n in _nodes(rewrite({"title": "Old loans"}), "Archive") if n.get("type") == "table")
    assert unstated["attributes"]["action-buttons"] == "false"
    stated = next(n for n in _nodes(rewrite({"title": "Old loans", "editable": True}), "Archive")
                  if n.get("type") == "table")
    assert stated["attributes"]["action-buttons"] == "true"
