"""The reply to a generated class diagram says what the guards changed.

A guard that renames an association end changes what the user sees on the
canvas. The reply names those changes, so a ``_1`` end does not appear
unexplained.
"""

import pytest

import diagram_handlers.types.class_diagram_handler as handler_mod
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from schemas.compact_class_diagram import (
    CompactClassSpec,
    CompactRelationshipSpec,
    CompactSystemClassSpec,
)


@pytest.fixture
def handler():
    return ClassDiagramHandler(None)


def _message(handler, **changes):
    spec = {"systemName": "Books", "relationships": [],
            "classes": [{"className": n, "attributes": [], "methods": []}
                        for n in ("Book", "Shelf")]}
    return handler._build_system_message(spec, **changes)


SUMMARY = "Your **Books** spec is ready — it captures **Book**, **Shelf**."


def test_no_sentence_when_nothing_changed(handler):
    assert _message(handler) == SUMMARY


def _rename(owner, other, new, old, holder=None, attribute=True):
    return {"owner": owner, "other": other, "new": new, "old": old,
            "holder": holder or owner, "attribute": attribute}


def test_a_renamed_end_names_the_class_the_old_name_and_why(handler):
    assert _message(handler, renamed=[_rename("Book", "Shelf", "shelf_1", "shelf")]) == SUMMARY + (
        " Book's link to Shelf is now called **shelf_1**, because Book already has an "
        "attribute named **shelf**.")


def test_an_end_renamed_for_another_link(handler):
    message = _message(handler, renamed=[
        _rename("Doctor", "Department", "department_1", "department", attribute=False)])
    assert message == SUMMARY + (
        " Doctor's link to Department is now called **department_1**, because Doctor already "
        "has another link named **department**.")


def test_an_end_renamed_for_a_subclass_attribute(handler):
    message = _message(handler, renamed=[_rename("Media", "Tag", "tags_1", "tags", "Video")])
    assert message == SUMMARY + (
        " Media's link to Tag is now called **tags_1**, because Video already has an "
        "attribute named **tags**.")


def test_more_than_three_renamed_ends_are_counted(handler):
    message = _message(handler, renamed=[
        _rename("Book", other, f"{other.lower()}_1", other.lower())
        for other in ("Shelf", "Author", "Genre", "Publisher", "Series")])
    assert message == SUMMARY + (
        " Book's link to Shelf is now called **shelf_1**, because Book already has an "
        "attribute named **shelf**. Book's link to Author is now called **author_1**, because "
        "Book already has an attribute named **author**. Book's link to Genre is now called "
        "**genre_1**, because Book already has an attribute named **genre**. 2 more "
        "association end(s) were renamed the same way.")


def test_generated_reply_says_what_the_guards_changed(monkeypatch):
    compact = CompactSystemClassSpec(
        name="Books",
        classes=[
            CompactClassSpec(n="Book", a=["title: str", "shelf: str"], m=[], k=""),
            CompactClassSpec(n="Shelf", a=["room: str"], m=[], k=""),
        ],
        rels=[CompactRelationshipSpec(
            f="Book", t="Shelf", k="assoc", how_many_TARGET_for_one_SOURCE="1",
            how_many_SOURCE_for_one_TARGET="0..*", l="shelf", ls="books")],
        ocl=[],
    )
    monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
    handler = ClassDiagramHandler(None)
    monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **_: compact)

    result = handler.generate_complete_system("books on shelves, each with a shelf label")

    assert result["message"].endswith(
        " Book's link to Shelf is now called **shelf_1**, because Book already has an "
        "attribute named **shelf**.")


def test_generated_reply_is_unchanged_when_the_guards_changed_nothing(monkeypatch):
    compact = CompactSystemClassSpec(
        name="Books",
        classes=[CompactClassSpec(n="Book", a=["title: str"], m=[], k=""),
                 CompactClassSpec(n="Shelf", a=["room: str"], m=[], k="")],
        rels=[CompactRelationshipSpec(
            f="Book", t="Shelf", k="assoc", how_many_TARGET_for_one_SOURCE="1",
            how_many_SOURCE_for_one_TARGET="0..*", l="shelf", ls="books")],
        ocl=[],
    )
    monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
    handler = ClassDiagramHandler(None)
    monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **_: compact)

    result = handler.generate_complete_system("books on shelves")

    assert result["message"] == (
        "Your **Books** spec is ready — it captures **Book**, **Shelf**, "
        "with 1 relationship(s) linking them.")
