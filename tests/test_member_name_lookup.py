"""A method must be findable by its bare name.

A user asked for a social-media platform; the generated model
gave User both a ``follow`` method and a ``follow`` association end, which the
BESSER validator correctly reports:

    Class 'User' defines method 'follow' with the name of one of its attributes
    or association ends. A generated object can only carry one of them under
    that name.

The assistant announced "Validation found 1 issue(s) - fixing it now..." and
then answered "I couldn't find a follow method to update." The method was
right there.

``_clean_member_name`` was written for ATTRIBUTES ("email: str" -> "email") and
reused for METHODS, where the first ``:`` belongs to a parameter rather than a
return type:

    "follow(user: User)"  ->  "follow(user"     never matches "follow"
    "follow()"            ->  "follow()"        never matches "follow"

So every method lookup failed, not just the auto-fix: any "rename/remove that
method" request hit the same phantom-target reply.
"""

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


clean = ClassDiagramHandler._clean_member_name


# ------------------------------------------------------------ methods


@pytest.mark.parametrize("stored", [
    "follow",
    "follow()",
    "follow(user: User)",
    "+follow(user: User)",
    "follow(user: User): bool",
    "-follow(a: int, b: str): None",
    "  follow( user : User )  ",
])
def test_a_method_reduces_to_its_bare_name(stored):
    assert clean(stored) == "follow"


def test_the_exact_live_failure(stored="follow(user: User)"):
    """The value that produced "I couldn't find a follow method to update"."""
    assert clean(stored) == "follow", (
        "a method with a typed parameter must still be findable by name"
    )


# --------------------------------------------------------- attributes


@pytest.mark.parametrize("stored,expected", [
    ("email", "email"),
    ("email: str", "email"),
    ("+email: str", "email"),
    ("#createdAt: datetime", "createdAt"),
    ("~count: int", "count"),
])
def test_an_attribute_still_drops_its_type(stored, expected):
    """The attribute behaviour this helper was written for must not regress."""
    assert clean(stored) == expected


# ------------------------------------------------------------- edges


@pytest.mark.parametrize("stored", [None, "", "   "])
def test_empty_input_is_empty(stored):
    assert clean(stored) == ""


def test_a_method_and_an_end_of_the_same_name_are_both_indexed():
    """The collision the validator reports: the model legitimately holds a
    `follow` method and a `follow` association end. Both must reduce to the
    same bare name, otherwise the fix cannot target either."""
    assert clean("follow(user: User)") == clean("follow")


def test_the_index_finds_a_method_with_parameters():
    """End-to-end through _build_model_index, the way the phantom-target check
    actually runs."""
    handler = ClassDiagramHandler.__new__(ClassDiagramHandler)
    model = {
        "elements": {
            "c1": {"type": "Class", "name": "User",
                   "attributes": ["a1"], "methods": ["m1"]},
            "a1": {"type": "ClassAttribute", "name": "email: str", "owner": "c1"},
            "m1": {"type": "ClassMethod", "name": "+follow(user: User): bool",
                   "owner": "c1"},
        }
    }
    class_names, attr_names, method_names = handler._build_model_index(model)
    assert "user" in class_names
    assert "email" in attr_names
    assert "follow" in method_names, (
        "the auto-fix looks the method up by bare name; if it is not indexed "
        "that way the fix reports a phantom target and gives up"
    )
