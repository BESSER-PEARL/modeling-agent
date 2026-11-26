"""Removing an OCL constraint through the assistant.

BESSER-PEARL/modeling-agent#98: "remove book_has_title constraint" deleted the
Book class. BESSER-PEARL/BESSER#837: "remove the OCL constraints" left them all
on the diagram. Root cause: the modify prompt never listed the constraints and
the target schema had no field to name one, so the only removal the LLM could
express was the constraint's class.

A constraint box (``ClassOCLConstraint``) can hold several ``context`` blocks:
the agent's own generator writes every invariant of a class into one box. The
name is the handle of one block, not of the box.
"""

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from execution.model_operations import _count_removed_top_level_elements
from schemas import ClassModificationResponse
from schemas.class_diagram import ClassModificationTarget
from utilities.model_context import detailed_model_summary, ocl_constraint_blocks


def _library_model():
    """Book with two invariants in ONE box, Account with an unnamed precondition."""
    return {
        "elements": {
            "c-book": {"id": "c-book", "name": "Book", "type": "Class", "owner": None,
                       "attributes": ["a-title"], "methods": []},
            "a-title": {"id": "a-title", "name": "title", "type": "ClassAttribute",
                        "owner": "c-book", "attributeType": "str"},
            "c-account": {"id": "c-account", "name": "Account", "type": "Class", "owner": None,
                          "attributes": [], "methods": ["m-deposit"]},
            "m-deposit": {"id": "m-deposit", "name": "+deposit(amount: int)", "type": "ClassMethod",
                          "owner": "c-account"},
            "ocl-book": {"id": "ocl-book", "name": "", "type": "ClassOCLConstraint", "owner": None,
                         "constraint": "context Book inv book_has_title: self.title <> ''\n\n"
                                       "context Book inv title_is_short: self.title.size() < 200"},
            "ocl-acc": {"id": "ocl-acc", "name": "OCL", "type": "ClassOCLConstraint", "owner": None,
                        "constraint": "context Account::deposit(amount: Integer) pre: amount > 0"},
        },
        "relationships": {
            "l-book": {"id": "l-book", "type": "ClassOCLLink",
                       "source": {"element": "ocl-book"}, "target": {"element": "c-book"}},
        },
    }


def _run(llm_output, request, model=None):
    parsed = ClassModificationResponse.model_validate(llm_output)
    prompts = []

    def _predict(*args, **kwargs):
        prompts.append(" ".join(str(a) for a in args))
        return parsed

    handler = ClassDiagramHandler(llm=None)
    handler.predict_structured = _predict
    handler._modification_crash_reply = lambda exc, *a, **k: pytest.fail(f"crashed: {exc!r}")
    return handler.generate_modification(request, model or _library_model()), prompts


def _ops(result):
    return result.get("modifications") or ([result["modification"]] if result.get("modification") else [])


# --------------------------------------------------------------------------
# What the LLM is shown
# --------------------------------------------------------------------------

def test_every_constraint_block_gets_a_label():
    blocks = ocl_constraint_blocks(_library_model())
    assert [(b["label"], b["context"], b["element_id"]) for b in blocks] == [
        ("book_has_title", "Book", "ocl-book"),
        ("title_is_short", "Book", "ocl-book"),
        ("Account::deposit pre", "Account", "ocl-acc"),
    ]


def test_the_model_summary_lists_the_constraints():
    summary = detailed_model_summary(_library_model(), "ClassDiagram")
    assert "OCL constraints (3)" in summary
    assert "book_has_title" in summary
    assert "title_is_short" in summary
    assert "Account::deposit pre" in summary


def test_the_modify_prompt_shows_the_constraint_names():
    _result, prompts = _run({"modifications": [
        {"action": "remove_element", "target": {"constraintName": "book_has_title"}},
    ]}, "remove book_has_title constraint")
    # Names the request does not mention reach the LLM too ("remove all of them").
    assert prompts and "title_is_short" in prompts[0]
    assert "Account::deposit pre" in prompts[0]


# --------------------------------------------------------------------------
# What the LLM can say
# --------------------------------------------------------------------------

def test_the_target_schema_names_a_constraint():
    assert ClassModificationTarget(constraintName="book_has_title").constraintName == "book_has_title"
    assert "constraintName" in ClassModificationTarget.model_json_schema()["properties"]


def test_a_constraint_removal_reaches_the_editor_unchanged():
    result, _ = _run({"modifications": [
        {"action": "remove_element", "target": {"constraintName": "book_has_title"}},
    ]}, "remove book_has_title constraint")

    assert result["action"] == "modify_model"
    [op] = _ops(result)
    assert op["target"].get("constraintName") == "book_has_title"
    # Never promoted into a class removal.
    assert not op["target"].get("className")


def test_removing_all_constraints_keeps_one_op_per_constraint():
    result, _ = _run({"modifications": [
        {"action": "remove_element", "target": {"className": "Book", "constraintName": "book_has_title"}},
        {"action": "remove_element", "target": {"className": "Book", "constraintName": "title_is_short"}},
        {"action": "remove_element", "target": {"className": "Account",
                                                "constraintName": "Account::deposit pre"}},
    ]}, "remove the OCL constraints")

    assert result["action"] == "modify_model"
    assert [op["target"]["constraintName"] for op in _ops(result)] == [
        "book_has_title", "title_is_short", "Account::deposit pre",
    ]


def test_an_unknown_constraint_is_reported_not_sent():
    result, _ = _run({"modifications": [
        {"action": "remove_element", "target": {"constraintName": "no_such_rule"}},
    ]}, "remove the no_such_rule constraint")

    assert result["action"] == "assistant_message"
    assert "no_such_rule" in result["message"]


def test_constraint_removals_do_not_count_as_deleting_the_model():
    plan = {"modifications": [
        {"action": "remove_element", "target": {"className": "Book", "constraintName": n}}
        for n in ("a", "b", "c")
    ]}
    assert _count_removed_top_level_elements(plan) == 0
