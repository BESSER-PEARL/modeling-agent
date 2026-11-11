"""The reply of a modification names the member and its owner.

Issue #744: a modify_attribute on ``Brand.aliases`` replied "Updated attribute in
attribute aliases from Brand": the target name was already a phrase, and the
action label wrapped it again.
"""
import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from diagram_handlers.types.user_profile_handler import UserProfileDiagramHandler

H = ClassDiagramHandler(None)


def _say(action, target, changes=None):
    mod = {"action": action, "target": target, **({"changes": changes} if changes else {})}
    return H._describe_modification(action, target, mod)


@pytest.mark.parametrize("action, target, changes, message", [
    ("modify_attribute", {"className": "Brand", "attributeName": "aliases"}, {"type": "str"},
     "Updated attribute **aliases** in **Brand**."),
    ("modify_attribute", {"className": "Brand", "attributeName": "aliases"}, {"name": "nicknames"},
     "Renamed attribute **aliases** to **nicknames** in **Brand**."),
    ("add_attribute", {"className": "Brand"}, {"name": "country", "type": "str"},
     "Added attribute **country** to **Brand**."),
    ("remove_element", {"className": "Brand", "attributeName": "aliases"}, None,
     "Removed attribute **aliases** from **Brand**."),
    ("modify_method", {"className": "Car", "methodName": "drive"}, {"returnType": "bool"},
     "Updated method **drive** in **Car**."),
    ("remove_element", {"className": "Car", "methodName": "drive"}, None,
     "Removed method **drive** from **Car**."),
    ("remove_element", {"className": "Car"}, None, "Removed **Car**."),
    ("add_attribute", {"className": "Brand"}, None, "Added an attribute to **Brand**."),
    # A member name the reply may not quote never falls back to the owner:
    # "Removed **Brand**." would read as the class being removed.
    ("remove_element", {"className": "Brand", "attributeName": "name wall?"}, None,
     "Removed an attribute from **Brand**."),
])
def test_the_message_reads_naturally(action, target, changes, message):
    assert _say(action, target, changes) == message


def test_names_are_still_sanitized():
    assert _say("modify_attribute", {"className": "Brand},", "attributeName": '"aliases"'}) == (
        "Updated attribute **aliases** in **Brand**."
    )


def test_a_batch_names_every_member():
    mods = [
        {"action": "modify_attribute", "target": {"className": "Brand", "attributeName": "aliases"},
         "changes": {"type": "str"}},
        {"action": "add_method", "target": {"className": "Car"}, "changes": {"name": "drive"}},
    ]
    assert H._friendly_batch_message(mods) == (
        "Applied 2 changes:\n- Updated attribute **aliases** in **Brand**.\n- Added method **drive** to **Car**."
    )


def test_a_user_profile_attribute_value_names_attribute_and_profile():
    h = UserProfileDiagramHandler(None)
    target = {"profileName": "personal_information", "attributeName": "age"}
    assert h._describe_modification("modify_attribute_value", target, {"target": target}) == (
        "Updated attribute **age** on **personal_information**."
    )


def test_a_type_without_a_member_index_quotes_member_names_as_written():
    """Only the class diagram checks names against its model; an object or
    user-profile attribute such as "first-name" is quoted as before."""
    h = UserProfileDiagramHandler(None)
    target = {"profileName": "alice", "attributeName": "first-name"}
    assert h._describe_modification("modify_attribute_value", target, {"target": target}) == (
        "Updated attribute **first-name** on **alice**."
    )


def test_a_user_profile_target_with_a_class_still_names_the_profile():
    h = UserProfileDiagramHandler(None)
    target = {"profileName": "alice_profile", "className": "Personal_Information", "attributeName": "age"}
    assert h._describe_modification("modify_attribute_value", target, {"target": target}) == (
        "Updated attribute **age** on **alice_profile**."
    )
