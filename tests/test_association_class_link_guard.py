"""Guards found by the first full local end-to-end run (prompt -> app).

Three defects, all of the same shape: ONE bad token in the LLM's output
destroys the whole model rather than the one member it belongs to. Each test
below was verified to FAIL against the pre-fix code.

1. ``expand_compact_spec`` copied the compact ``ac`` field through verbatim,
   so an association class naming a class that does not exist raised out of
   ``SystemClassSpec`` validation, was swallowed by ``generate_complete_system``'s
   blanket except, and dropped the request onto the incremental fallback.
   Observed live: Qwen/Qwen3-30B-A3B on the grant-applications prompt.

2. An association class joined to its endpoints by ORDINARY links as well as
   by its attachment makes ``SQLAlchemyGenerator`` name the same foreign key
   two ways (``bookings_id`` on the link table, ``BookingRoom.booking_id`` in
   the relationship), and the generated ``sql_alchemy.py`` raises
   AttributeError on import - the delivered app is dead before the agent
   writes a line. Observed live on two consecutive Qwen hotel runs, once with
   Association duplicates and once with Composition.

3. A method parameter or return type naming nothing in the spec
   ('List[Bill]', 'dict', 'Money') raised ValueError/ConversionError out of
   the class-diagram converter and took every other class with it. Attributes
   already had this guard; methods did not.
"""

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from schemas.compact_class_diagram import CompactSystemClassSpec, expand_compact_spec


@pytest.fixture
def handler():
    # Established pattern: the LLM arg is only needed by methods that call it.
    return ClassDiagramHandler(None)


def _compact(ac, extra_classes=()):
    return CompactSystemClassSpec(
        name="S",
        classes=[{"n": "Application", "a": ["reference: str!"], "m": [], "k": ""},
                 {"n": "Reviewer", "a": ["name: str"], "m": [], "k": ""},
                 *extra_classes],
        rels=[{"f": "Application", "t": "Reviewer", "k": "assoc",
               "how_many_TARGET_for_one_SOURCE": "1..*",
               "how_many_SOURCE_for_one_TARGET": "0..*",
               "l": "reviewers", "ls": "applications", "ac": ac}],
        ocl=[])


# -- 1. compact expansion must never raise on a bad `ac` ---------------------

@pytest.mark.parametrize("ac,extra,why", [
    ("Assessment", (), "names no class at all (the live Qwen failure)"),
    ("Reviewer", (), "names one of the association's own endpoints"),
    ("Status", ({"n": "Status", "a": ["A", "B"], "m": [], "k": "enum"},),
     "names an enumeration"),
    ("Base", ({"n": "Base", "a": [], "m": [], "k": "abstract"},),
     "names an abstract class"),
])
def test_unusable_association_class_is_dropped_not_fatal(ac, extra, why):
    spec = expand_compact_spec(_compact(ac, extra))
    assert spec.relationships[0].associationClass is None, why
    # The point of the guard: the rest of the model survives.
    assert {c.className for c in spec.classes} >= {"Application", "Reviewer"}


def test_valid_association_class_is_kept():
    spec = expand_compact_spec(
        _compact("Assessment",
                 ({"n": "Assessment", "a": ["mark: float"], "m": [], "k": ""},)))
    assert spec.relationships[0].associationClass == "Assessment"


# -- 2. redundant association-class links --------------------------------

def _spec_with_link_kind(kind):
    """Booking-Room carrying BookingRoom, PLUS the duplicate endpoint links."""
    return {
        "systemName": "HotelBookingSystem",
        "classes": [
            {"className": n, "isAbstract": False, "isEnumeration": False,
             "attributes": [{"name": "ref", "type": "str", "visibility": "public",
                             "isExternalId": False, "isDerived": False,
                             "isOptional": False, "defaultValue": None}],
             "methods": []}
            for n in ("Booking", "Room", "BookingRoom")],
        "relationships": [
            {"type": "Association", "source": "Booking", "target": "Room",
             "sourceMultiplicity": "0..*", "targetMultiplicity": "1..*",
             "name": "rooms", "sourceRole": "bookings",
             "associationClass": "BookingRoom"},
            {"type": kind, "source": "BookingRoom", "target": "Room",
             "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
             "name": "room", "sourceRole": "bookingRooms"},
            {"type": kind, "source": "BookingRoom", "target": "Booking",
             "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
             "name": "booking", "sourceRole": "bookingRooms"},
        ],
        "constraints": [],
    }


@pytest.mark.parametrize("kind", ["Association", "Composition", "Aggregation"])
def test_redundant_association_class_links_are_dropped(handler, kind):
    spec = _spec_with_link_kind(kind)
    handler._drop_redundant_association_class_links(spec)
    assert len(spec["relationships"]) == 1
    assert spec["relationships"][0]["associationClass"] == "BookingRoom"


def test_inheritance_to_an_association_class_is_left_alone(handler):
    spec = _spec_with_link_kind("Inheritance")
    handler._drop_redundant_association_class_links(spec)
    # Wrong in other ways, but a different claim and not this guard's business.
    assert len(spec["relationships"]) == 3


def test_model_without_an_association_class_is_untouched(handler):
    spec = _spec_with_link_kind("Association")
    spec["relationships"][0].pop("associationClass")
    before = [dict(r) for r in spec["relationships"]]
    handler._drop_redundant_association_class_links(spec)
    assert spec["relationships"] == before


def test_link_to_a_class_outside_the_association_is_kept(handler):
    spec = _spec_with_link_kind("Association")
    spec["classes"].append({"className": "ExtraCharge", "isAbstract": False,
                            "isEnumeration": False, "attributes": [], "methods": []})
    spec["relationships"].append(
        {"type": "Composition", "source": "BookingRoom", "target": "ExtraCharge",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*",
         "name": "extraCharges", "sourceRole": "chargedTo"})
    handler._drop_redundant_association_class_links(spec)
    remaining = {(r["source"], r["target"]) for r in spec["relationships"]}
    assert ("BookingRoom", "ExtraCharge") in remaining


# -- 3. unresolvable method member types ------------------------------------

def _method_spec(return_type, param_type):
    return {
        "systemName": "T", "relationships": [], "constraints": [],
        "classes": [
            {"className": "Order", "isAbstract": False, "isEnumeration": False,
             "attributes": [],
             "methods": [{"name": "act", "returnType": return_type,
                          "visibility": "public",
                          "parameters": ([{"name": "p", "type": param_type}]
                                         if param_type else []),
                          "isAbstract": False, "implementationType": "none",
                          "code": None}]},
            {"className": "Bill", "isAbstract": False, "isEnumeration": False,
             "attributes": [], "methods": []},
        ],
    }


@pytest.mark.parametrize("bad", ["List[Bill]", "dict", "Map<String,Bill>"])
def test_unresolvable_parameter_type_is_coerced(handler, bad):
    spec = _method_spec("bool", bad)
    handler._sanitize_member_types(spec)
    assert spec["classes"][0]["methods"][0]["parameters"][0]["type"] == "str"


@pytest.mark.parametrize("bad", ["Money", "List[Order]"])
def test_unresolvable_return_type_is_coerced(handler, bad):
    spec = _method_spec(bad, None)
    handler._sanitize_member_types(spec)
    assert spec["classes"][0]["methods"][0]["returnType"] == "any"


@pytest.mark.parametrize("return_type,param_type", [
    ("bool", "Bill"),      # a class in the spec
    ("float", "int"),      # primitives
    ("Bill", "str"),
])
def test_resolvable_member_types_are_untouched(handler, return_type, param_type):
    spec = _method_spec(return_type, param_type)
    handler._sanitize_member_types(spec)
    method = spec["classes"][0]["methods"][0]
    assert method["returnType"] == return_type
    assert method["parameters"][0]["type"] == param_type
