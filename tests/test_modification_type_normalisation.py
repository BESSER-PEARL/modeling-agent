"""The modify path had no type guard at all.

Every type guard - ``_sanitize_member_types``,
``_rewrite_class_typed_attributes``, ``_declare_referenced_enumerations`` -
is invoked from ``generate_complete_system``. A type arriving through a
modification therefore reached BUML with nothing looking at it, so the same
``LocalDate`` that a full generation canonicalises to ``date`` went through
a modify op untouched, and the two entry points into one metamodel
disagreed.

This covers the deterministic half. Coercing a genuinely unknown type and
rewriting a class-typed attribute into an association both need the
declared-type set, and are deliberately not attempted from a single op.
"""

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler(None)


def _batch(*mods) -> dict:
    return {"modifications": list(mods)}


def test_an_added_attribute_type_is_canonicalised(handler):
    spec = _batch({
        "action": "add_attribute",
        "target": {"className": "Booking"},
        "changes": {"name": "arrivalDate", "type": "LocalDate"},
    })

    handler._normalise_modification_types(spec)

    assert spec["modifications"][0]["changes"]["type"] == "date"


def test_a_method_return_type_is_canonicalised(handler):
    spec = _batch({
        "action": "add_method",
        "target": {"className": "Booking"},
        "changes": {"name": "total", "returnType": "BigDecimal"},
    })

    handler._normalise_modification_types(spec)

    assert spec["modifications"][0]["changes"]["returnType"] == "float"


def test_attributes_on_an_added_class_are_canonicalised(handler):
    spec = _batch({
        "action": "add_class",
        "target": {},
        "changes": {
            "className": "Invoice",
            "attributes": [
                {"name": "issued", "type": "LocalDateTime"},
                {"name": "amount", "type": "BigDecimal"},
                {"name": "note", "type": "str"},
            ],
        },
    })

    handler._normalise_modification_types(spec)

    types = [a["type"] for a in spec["modifications"][0]["changes"]["attributes"]]
    assert types == ["datetime", "float", "str"]


def test_method_parameters_are_canonicalised(handler):
    spec = _batch({
        "action": "add_method",
        "target": {"className": "Loan"},
        "changes": {
            "name": "renew",
            "parameters": [{"name": "until", "type": "LocalDate"}],
        },
    })

    handler._normalise_modification_types(spec)

    params = spec["modifications"][0]["changes"]["parameters"]
    assert params[0]["type"] == "date"


def test_a_canonical_type_is_left_alone(handler):
    spec = _batch({
        "action": "add_attribute",
        "target": {"className": "Booking"},
        "changes": {"name": "nights", "type": "int"},
    })

    handler._normalise_modification_types(spec)

    assert spec["modifications"][0]["changes"]["type"] == "int"


def test_a_class_typed_attribute_is_not_touched(handler):
    """Deciding this needs the declared-type set; one op cannot know."""
    spec = _batch({
        "action": "add_attribute",
        "target": {"className": "Booking"},
        "changes": {"name": "guest", "type": "Guest"},
    })

    handler._normalise_modification_types(spec)

    assert spec["modifications"][0]["changes"]["type"] == "Guest"


def test_a_bare_op_outside_a_batch_is_handled(handler):
    spec = {
        "action": "add_attribute",
        "target": {"className": "Booking"},
        "changes": {"name": "arrivalDate", "type": "Timestamp"},
    }

    handler._normalise_modification_types(spec)

    assert spec["changes"]["type"] == "datetime"


def test_malformed_ops_do_not_raise(handler):
    spec = _batch(
        {"action": "add_attribute"},                      # no changes
        {"action": "add_attribute", "changes": None},     # changes is None
        "not a dict",                                      # not an op at all
        {"action": "add_class", "changes": {"attributes": [None, "x"]}},
    )

    handler._normalise_modification_types(spec)  # must not raise
