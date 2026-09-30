"""A Java-flavoured type name must not silently become a string.

The frontend canonicalises these before anything reaches BUML
(``typeNormalization.ts`` TYPE_ALIASES: LocalDate -> date, BigDecimal ->
float, and so on). The agent did not, so ``LocalDate`` matched no entry in
``_PRIMITIVE_ATTR_TYPES``, was not a declared class name, and fell through
to the unresolvable-type coercion - which replaces it with ``String``.

A date becoming a string is not cosmetic: the generated column stops being
a date, comparisons stop ordering, and nothing downstream reports it,
because from the coercion's point of view it did its job.

Found while fixing the sibling defect in the same guard, where a genuine
enumeration reference was being coerced the same way.
"""

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    # The LLM argument is only needed by methods that call it.
    return ClassDiagramHandler(None)


def _spec(attr_type: str) -> dict:
    return {
        "classes": [
            {
                "className": "Booking",
                "attributes": [{"name": "arrivalDate", "type": attr_type}],
                "methods": [],
            }
        ]
    }


def _attr_type(handler, attr_type: str) -> str:
    spec = _spec(attr_type)
    handler._rewrite_class_typed_attributes(spec)
    return spec["classes"][0]["attributes"][0]["type"]


@pytest.mark.parametrize("raw,canonical", [
    ("LocalDate", "date"),
    ("localDate", "date"),
    ("LocalDateTime", "datetime"),
    ("Timestamp", "datetime"),
    ("Instant", "datetime"),
    ("LocalTime", "time"),
    ("BigDecimal", "float"),
    ("BigInteger", "int"),
    ("UUID", "str"),
    ("GUID", "str"),
    ("Character", "str"),
    ("Object", "any"),
])
def test_an_aliased_type_is_normalised_not_coerced(handler, raw, canonical):
    assert _attr_type(handler, raw) == canonical


@pytest.mark.parametrize("raw", ["str", "int", "bool", "float", "date", "datetime"])
def test_a_canonical_primitive_is_untouched(handler, raw):
    assert _attr_type(handler, raw) == raw


@pytest.mark.parametrize("raw", ["String", "Integer", "Boolean"])
def test_the_existing_primitive_spellings_still_pass_through(handler, raw):
    """These were already in _PRIMITIVE_ATTR_TYPES and are returned as-is."""
    assert _attr_type(handler, raw) == raw


def test_a_genuinely_unknown_type_is_still_coerced(handler):
    """The coercion must keep doing its job for a real hallucination."""
    assert _attr_type(handler, "Sparkliness") == "String"


def test_a_class_typed_attribute_still_becomes_an_association(handler):
    """The guard's primary job, which the alias check must not intercept.

    A class-typed attribute is not kept as an attribute: it is rewritten
    into an Association and dropped from the class, with the attribute name
    carried over as the role so startLocation/endLocation stay distinct.
    """
    spec = {
        "classes": [
            {"className": "Booking",
             "attributes": [{"name": "guest", "type": "Guest"}], "methods": []},
            {"className": "Guest", "attributes": [], "methods": []},
        ]
    }
    handler._rewrite_class_typed_attributes(spec)

    assert spec["classes"][0]["attributes"] == []
    assert len(spec["relationships"]) == 1
    rel = spec["relationships"][0]
    assert (rel["type"], rel["source"], rel["target"]) == ("Association", "Booking", "Guest")
    assert rel["name"] == "guest"
