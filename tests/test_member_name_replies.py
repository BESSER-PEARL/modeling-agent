"""Modification replies name the member the change is about, never junk.

Live on editor.besser-pearl.org (2026-11-09), 4 of 15 class-diagram edits
replied with text the model had written into ``target.attributeName``:

- "add a stock attribute (int) to Product" -> "Added attribute **stock
  quantity (int) in Product.** to **Product**." (the editor added ``stock``,
  from ``changes.name``)
- "add an attribute country: str to Brand" -> ``target.attributeName`` held
  ~9.3k characters of the prompt's own workspace context, quoted in the reply
- "rename aliases in Brand to altNames" -> "I couldn't find a **aliases שב?**
  field to update."
- the same rename with ``target.attributeName: null`` -> "Updated attribute in
  **Brand**." although the editor changes nothing without a member name

The model responses below are the ones those runs returned.
"""
import copy
import json

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from prompts import class_diagram as class_prompts
from pydantic import ValidationError

from schemas.class_diagram import ClassModification, ClassModificationChanges, ClassModificationResponse

from tests.conftest import FakeLLM


BRAND_MODEL = {
    "version": "3.0.0", "type": "ClassDiagram",
    "elements": {
        "c1": {"id": "c1", "name": "Brand", "type": "Class", "owner": None,
               "attributes": ["a1", "a2"], "methods": []},
        "a1": {"id": "a1", "name": "name: str", "type": "ClassAttribute", "owner": "c1"},
        "a2": {"id": "a2", "name": "aliases: str", "type": "ClassAttribute", "owner": "c1"},
        "c2": {"id": "c2", "name": "Product", "type": "Class", "owner": None,
               "attributes": ["a3"], "methods": []},
        "a3": {"id": "a3", "name": "price: float", "type": "ClassAttribute", "owner": "c2"},
    },
    "relationships": {},
}

ECHOED_CONTEXT = (
    "country.name: str, aliases: str\n-   - Product | attributes: price: float\n"
    "Other diagrams in project:\n- ClassDiagram: ClassDiagram: 2 class(es): Brand, "
    "Product and 0 relationship(s).\n\nCurrent class diagram:\n- Classes (2): Brand, Product\n"
) * 40


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


def _modify(mod: dict, request: str = "edit the model") -> dict:
    llm = FakeLLM(json.dumps({"modifications": [mod]}))
    return ClassDiagramHandler(llm).generate_modification(request, copy.deepcopy(BRAND_MODEL))


def _add_attribute(class_name: str, target_text, name: str, type_: str) -> dict:
    return {"action": "add_attribute",
            "target": {"className": class_name, "attributeName": target_text},
            "changes": {"name": name, "type": type_}}


@pytest.mark.parametrize("target_text", [
    "stock quantity (int) in Product.",
    ECHOED_CONTEXT,
], ids=["inline-text", "echoed-context"])
def test_an_add_reply_names_the_added_attribute(target_text):
    result = _modify(_add_attribute("Product", target_text, "stock", "int"),
                     "add a stock attribute (int) to Product")
    assert result["action"] == "modify_model"
    assert result["message"] == "Added attribute **stock** to **Product**."


def test_an_add_method_reply_names_the_added_method():
    result = _modify({"action": "add_method",
                      "target": {"className": "Brand", "methodName": "isActive(): bool on Brand"},
                      "changes": {"name": "isActive", "returnType": "bool"}})
    assert result["message"] == "Added method **isActive** to **Brand**."


def test_a_rename_without_the_member_name_is_not_reported_as_done():
    result = _modify({"action": "modify_attribute",
                      "target": {"className": "Brand", "attributeName": None},
                      "changes": {"name": "altNames"}},
                     "rename aliases in Brand to altNames")
    assert result["action"] == "assistant_message"
    assert "Updated" not in result["message"]
    assert result["message"] == (
        "I couldn't tell which attribute of **Brand** to update. "
        "Which one did you mean?"
    )


@pytest.mark.parametrize("garbled", ["aliases שב?", "aliases another?", "name wall?"])
def test_a_garbled_member_name_is_never_quoted(garbled):
    result = _modify({"action": "modify_attribute",
                      "target": {"className": "Brand", "attributeName": garbled},
                      "changes": {"name": "altNames"}},
                     "rename aliases in Brand to altNames")
    assert result["action"] == "assistant_message"
    assert garbled not in result["message"]
    assert result["message"] == (
        "I couldn't tell which attribute of **Brand** to update. "
        "Which one did you mean?"
    )


def test_an_absent_but_well_formed_member_is_still_named():
    result = _modify({"action": "remove_element",
                      "target": {"className": "Brand", "attributeName": "priority"}})
    assert result["message"] == "I couldn't find a **priority** field to remove."


def test_a_well_formed_rename_still_reads_as_a_rename():
    result = _modify({"action": "modify_attribute",
                      "target": {"className": "Brand", "attributeName": "aliases"},
                      "changes": {"name": "altNames"}})
    assert result["message"] == "Renamed attribute **aliases** to **altNames** in **Brand**."


def test_the_schema_tells_the_model_which_field_holds_which_name():
    schema = json.dumps(ClassModificationResponse.model_json_schema())
    assert "null for add_attribute" in schema
    assert "null for add_method" in schema
    assert "PascalCase, ONE word only)" not in schema


def test_an_added_member_may_use_the_member_name_length():
    """A member name is capped at 50 characters (AttributeSpec / MethodSpec);
    the class-name cap of 30 failed the whole modification."""
    name = "preferredCommunicationChannel"  # 29
    assert ClassModificationChanges(name=name + "Code").name == name + "Code"  # 33


@pytest.mark.parametrize("action", ["modify_class", "merge_classes"])
def test_a_class_name_keeps_the_class_name_length(action):
    with pytest.raises(ValidationError, match="longer than 30"):
        ClassModification(action=action, target={"className": "Brand"},
                          changes={"name": "PreferredCommunicationChannelCode"})
    assert ClassModification(action=action, target={"className": "Brand"},
                             changes={"name": "PreferredChannel"}).changes.name == "PreferredChannel"


PERSON_MODEL = {
    "version": "3.0.0", "type": "ClassDiagram",
    "elements": {
        "c1": {"id": "c1", "name": "Person", "type": "Class", "owner": None,
               "attributes": ["a1", "a2", "a3"], "methods": ["m1"]},
        "a1": {"id": "a1", "name": "+ first-name: str", "type": "ClassAttribute", "owner": "c1"},
        "a2": {"id": "a2", "name": "+ last name: str", "type": "ClassAttribute", "owner": "c1"},
        "a3": {"id": "a3", "name": "+ price: int", "type": "ClassAttribute", "owner": "c1"},
        "m1": {"id": "m1", "name": "+ get-age(): int", "type": "ClassMethod", "owner": "c1"},
    },
    "relationships": {},
}


def _modify_person(mod: dict) -> dict:
    llm = FakeLLM(json.dumps({"modifications": [mod]}))
    return ClassDiagramHandler(llm).generate_modification("edit the model", copy.deepcopy(PERSON_MODEL))


# BESSER rejects "first-name" ("Hyphens are not allowed") and "last name"; the
# editor's [auto-fix] repair renames exactly these, so they must reach it.
@pytest.mark.parametrize("mod, reply", [
    ({"action": "modify_attribute", "target": {"className": "Person", "attributeName": "first-name"},
      "changes": {"name": "first_name"}},
     "Renamed attribute **first-name** to **first_name** in **Person**."),
    ({"action": "modify_attribute", "target": {"className": "Person", "attributeName": "last name"},
      "changes": {"name": "lastName"}},
     "Renamed attribute **last name** to **lastName** in **Person**."),
    ({"action": "modify_method", "target": {"className": "Person", "methodName": "get-age"},
      "changes": {"name": "getAge"}},
     "Renamed method **get-age** to **getAge** in **Person**."),
    ({"action": "remove_element", "target": {"className": "Person", "attributeName": "first-name"}},
     "Removed attribute **first-name** from **Person**."),
], ids=["hyphen-rename", "space-rename", "hyphen-method-rename", "hyphen-remove"])
def test_a_real_member_with_an_invalid_name_is_kept(mod, reply):
    result = _modify_person(mod)
    assert result["action"] == "modify_model"
    assert result["message"] == reply


@pytest.mark.parametrize("mod, reply", [
    ({"action": "remove_element", "target": {"className": "Person", "methodName": "get-age()"}},
     "Removed method **get-age** from **Person**."),
    ({"action": "modify_attribute", "target": {"className": "Person", "attributeName": "price: int"},
      "changes": {"type": "float"}},
     "Updated attribute **price** in **Person**."),
], ids=["method-with-parens", "attribute-with-type"])
def test_a_member_written_with_its_signature_is_named_bare(mod, reply):
    assert _modify_person(mod)["message"] == reply


def test_a_rename_to_a_name_that_is_not_an_identifier_is_still_reported():
    """The editor sets changes.name as written; hiding it reported "Updated"."""
    result = _modify_person({"action": "modify_attribute",
                             "target": {"className": "Person", "attributeName": "price"},
                             "changes": {"name": "unit-price"}})
    assert result["message"] == "Renamed attribute **price** to **unit-price** in **Person**."


def test_a_type_change_naming_the_attribute_only_in_changes_name_is_kept():
    """The editor falls back to changes.name to find the attribute."""
    result = _modify_person({"action": "modify_attribute", "target": {"className": "Person"},
                             "changes": {"name": "price", "type": "float"}})
    assert result["action"] == "modify_model"
    assert result["message"] == "Updated attribute **price** in **Person**."


def test_the_prompt_shows_an_attribute_rename():
    prompt = class_prompts.MODIFY_SYSTEM_PROMPT_CLASS
    assert 'target.attributeName="aliases", changes.name="altNames"' in prompt
