"""The create reply counts only the rules the user stated.

Live on editor.besser-pearl.org (2026-11-09), "Create a class diagram for a
hotel with rooms, guests and bookings" replied "I also noted 1 rule(s) you
mentioned, though they aren't shown on the canvas yet." The user stated no
rule: ``_break_mandatory_cycles`` had relaxed ``Hotel.rooms`` (1..*) and kept
the minimum as its own invariant ``rooms_at_least_1``, and the reply counted
every constraint as the user's. The second half was stale too: the editor
draws each constraint as an OCL box linked to its class
(``ClassDiagramConverter.createConstraints``).
"""
import pytest

from diagram_handlers.core import base_handler as base_handler_module
from diagram_handlers.core.base_handler import BaseDiagramHandler
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

# The live hotel model's mandatory cycle: a Hotel has at least one Room and a
# Room belongs to exactly one Hotel.
HOTEL = {
    "name": "HotelBookingSystem",
    "classes": [
        {"n": "Hotel", "a": ["name: str"], "m": [], "k": ""},
        {"n": "Room", "a": ["number: str"], "m": [], "k": ""},
    ],
    "rels": [
        {"f": "Room", "t": "Hotel", "k": "comp", "how_many_TARGET_for_one_SOURCE": "1",
         "how_many_SOURCE_for_one_TARGET": "1..*", "l": "hotel", "ls": "rooms"},
    ],
    "ocl": [],
}


class _Provider:
    def __init__(self, canned):
        self.canned = canned

    def supports_structured_outputs(self):
        return True

    def structured_model(self, model=None):
        return model or "fake-model"

    def parse(self, messages, schema, **kwargs):
        return schema.model_validate(self.canned)


def _create(monkeypatch, canned) -> dict:
    provider = _Provider(canned)
    monkeypatch.setattr(BaseDiagramHandler, "llm_provider", property(lambda self: provider))
    monkeypatch.setattr(base_handler_module.time, "sleep", lambda *_: None)
    request = "Create a class diagram for a hotel with rooms"
    return ClassDiagramHandler(None).generate_complete_system(request, raw_request=request)


def test_a_relaxed_cycle_is_not_counted_as_a_user_rule(monkeypatch):
    reply = _create(monkeypatch, HOTEL)
    constraints = reply["systemSpec"]["constraints"]
    assert [c["name"] for c in constraints] == ["rooms_at_least_1"]
    assert "you mentioned" not in reply["message"]
    assert "you stated" not in reply["message"]
    assert "**Hotel.rooms** is optional in the diagram and its minimum of 1 is kept as an OCL rule" in reply["message"]


def test_user_rules_are_counted_alone_and_not_called_hidden(monkeypatch):
    canned = dict(HOTEL, ocl=["context Room inv validNumber: self.number <> ''"])
    reply = _create(monkeypatch, canned)
    assert len(reply["systemSpec"]["constraints"]) == 2
    assert "I also added the rule you stated as an OCL constraint." in reply["message"]
    assert "**Hotel.rooms** is optional in the diagram" in reply["message"]
    assert "aren't shown on the canvas" not in reply["message"]


def test_two_relaxed_cycles_are_named_together(monkeypatch):
    canned = dict(HOTEL, classes=HOTEL["classes"] + [{"n": "Guest", "a": ["name: str"], "m": [], "k": ""}],
                  rels=HOTEL["rels"] + [
                      {"f": "Guest", "t": "Hotel", "k": "comp", "how_many_TARGET_for_one_SOURCE": "1",
                       "how_many_SOURCE_for_one_TARGET": "1..*", "l": "hotel", "ls": "guests"},
                  ])
    message = _create(monkeypatch, canned)["message"]
    assert "**Hotel.rooms** and **Hotel.guests** are optional in the diagram and their minimums are kept as OCL rules" in message


@pytest.mark.parametrize("ocl,expected", [
    ([], None),
    (["context Room inv a: self.number <> ''", "context Hotel inv b: self.name <> ''"],
     "I also added the 2 rules you stated as OCL constraints."),
])
def test_without_a_cycle_only_user_rules_are_mentioned(monkeypatch, ocl, expected):
    canned = dict(HOTEL, rels=[dict(HOTEL["rels"][0], how_many_SOURCE_for_one_TARGET="0..*")], ocl=ocl)
    message = _create(monkeypatch, canned)["message"]
    assert "OCL rule" not in message
    if expected is None:
        assert "rule" not in message
    else:
        assert expected in message
