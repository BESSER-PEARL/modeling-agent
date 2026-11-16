"""The reply to a generated class diagram says what the guards changed.

A guard that turns an id attribute into an association, or renames an
association end, changes what the user sees on the canvas. The reply names
those changes in one sentence, so a link or a ``_1`` end does not appear
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


def test_one_converted_id(handler):
    message = _message(
        handler, converted=[{"owner": "Book", "attribute": "shelfId", "target": "Shelf"}])
    assert message == SUMMARY + " **Book.shelfId** became an association to **Shelf**."


def test_several_converted_ids(handler):
    message = _message(handler, converted=[
        {"owner": "Book", "attribute": "shelfId", "target": "Shelf"},
        {"owner": "Book", "attribute": "authorId", "target": "Author"}])
    assert message == SUMMARY + (
        " **Book.shelfId** and **Book.authorId** became associations to the classes they name.")


def test_more_than_three_are_counted(handler):
    message = _message(handler, converted=[
        {"owner": "Book", "attribute": f"{name}Id", "target": name.title()}
        for name in ("shelf", "author", "genre", "publisher", "series")])
    assert message == SUMMARY + (
        " **Book.shelfId**, **Book.authorId**, **Book.genreId** and 2 more became "
        "associations to the classes they name.")


def test_renamed_ends(handler):
    assert _message(handler, renamed=["shelf_1"]) == SUMMARY + (
        " The association end **shelf_1** was renamed so that no class has two members of one name.")
    assert _message(handler, renamed=["shelf_1", "book_1"]) == SUMMARY + (
        " The association ends **shelf_1** and **book_1** were renamed so that no class has "
        "two members of one name.")


def test_a_conversion_and_a_rename_share_one_sentence(handler):
    message = _message(
        handler, renamed=["shelf_1"],
        converted=[{"owner": "Book", "attribute": "shelfId", "target": "Shelf"}])
    assert message == SUMMARY + (
        " **Book.shelfId** became an association to **Shelf** and the association end "
        "**shelf_1** was renamed so that no class has two members of one name.")


def test_generated_reply_says_what_the_guards_changed(monkeypatch):
    compact = CompactSystemClassSpec(
        name="Books",
        classes=[
            CompactClassSpec(n="Book", a=["title: str", "shelfId: str", "genres: str"], m=[], k=""),
            CompactClassSpec(n="Shelf", a=["room: str"], m=[], k=""),
            CompactClassSpec(n="Genre", a=["name: str"], m=[], k=""),
        ],
        rels=[CompactRelationshipSpec(
            f="Book", t="Genre", k="assoc", how_many_TARGET_for_one_SOURCE="0..*",
            how_many_SOURCE_for_one_TARGET="0..*", l="genres", ls="books")],
        ocl=[],
    )
    monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
    handler = ClassDiagramHandler(None)
    monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **_: compact)

    result = handler.generate_complete_system("books on shelves, with genres")

    assert result["message"].endswith(
        " **Book.shelfId** became an association to **Shelf** and the association end "
        "**genres_1** was renamed so that no class has two members of one name.")


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
