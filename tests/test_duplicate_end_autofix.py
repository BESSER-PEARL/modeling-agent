"""Deterministic repair for "two association ends with the same name".

Live bug: the editor's validate-and-repair loop reported
"The class 'Doctor' cannot have two association ends with the same name:
'department'", the agent replied "Modified: Updated relationship in
Doctor - Department" — but it had only renamed the relationship LABEL
(changes.name), which the validator ignores, so the same error persisted.

The repair is now deterministic (no LLM): the handler locates the duplicated
ends in the diagram JSON and renames all but one via modify_relationship +
changes.roleName (which the frontend applies to rel.target.role — the actual
association-end name). When no rename can be applied it says so honestly
instead of claiming success.

Handlers are instantiated with a None LLM — the deterministic path never
calls the model (see tests/test_diagram_handlers.py for the pattern).
"""

import copy

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


def _handler() -> ClassDiagramHandler:
    return ClassDiagramHandler(None)


def _model(rel_overrides=None, extra_relationships=None, extra_elements=None):
    """Doctor/Department with two associations whose Department-side ends both
    fall back to the effective name 'department' (empty roles)."""
    model = {
        "elements": {
            "doc-1": {"id": "doc-1", "type": "Class", "name": "Doctor"},
            "dep-1": {"id": "dep-1", "type": "Class", "name": "Department"},
        },
        "relationships": {
            "rel-1": {
                "id": "rel-1", "type": "ClassBidirectional", "name": "works_in",
                "source": {"element": "doc-1", "role": "", "multiplicity": "*"},
                "target": {"element": "dep-1", "role": "", "multiplicity": "1"},
            },
            "rel-2": {
                "id": "rel-2", "type": "ClassBidirectional", "name": "heads",
                "source": {"element": "doc-1", "role": "", "multiplicity": "0..1"},
                "target": {"element": "dep-1", "role": "", "multiplicity": "1"},
            },
        },
    }
    if rel_overrides:
        for rel_id, override in rel_overrides.items():
            model["relationships"][rel_id].update(copy.deepcopy(override))
    if extra_relationships:
        model["relationships"].update(copy.deepcopy(extra_relationships))
    if extra_elements:
        model["elements"].update(copy.deepcopy(extra_elements))
    return model


_ERROR_LINE = (
    "The class 'Doctor' cannot have two association ends "
    "with the same name: 'department'"
)
_AUTO_FIX_MSG = (
    "[auto-fix] The last change left the diagram with validation errors. "
    "Fix exactly these, changing only what is necessary:\n"
    f"- {_ERROR_LINE}"
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetection:
    def test_pure_auto_fix_message(self):
        errors, pure = ClassDiagramHandler._detect_duplicate_end_errors(_AUTO_FIX_MSG)
        assert errors == [("Doctor", "department")]
        assert pure is True

    def test_mixed_message_is_not_pure(self):
        mixed = _AUTO_FIX_MSG + "\n- Invalid type 'str?' for the parameter 'description'"
        errors, pure = ClassDiagramHandler._detect_duplicate_end_errors(mixed)
        assert errors == [("Doctor", "department")]
        assert pure is False

    def test_plain_modification_request_does_not_trigger(self):
        errors, pure = ClassDiagramHandler._detect_duplicate_end_errors(
            "add an email attribute to User")
        assert errors == []
        assert pure is False

    def test_error_pasted_without_bullets_is_pure(self):
        errors, pure = ClassDiagramHandler._detect_duplicate_end_errors(
            f"please fix this: {_ERROR_LINE}")
        assert errors == [("Doctor", "department")]
        assert pure is True


# ---------------------------------------------------------------------------
# Deterministic repair (no LLM — the handler holds a None model)
# ---------------------------------------------------------------------------

class TestDeterministicRepair:
    def test_repairs_duplicate_end_by_renaming_the_role(self):
        result = _handler().generate_modification(_AUTO_FIX_MSG, _model())
        assert result["action"] == "modify_model"
        mod = result["modification"]
        assert mod["action"] == "modify_relationship"
        # The SECOND duplicate is renamed; the first keeps its name.
        assert mod["target"]["relationshipId"] == "rel-2"
        assert mod["changes"]["roleName"] == "department_1"
        # The fix renames the END (roleName), never the relationship label.
        assert "name" not in mod["changes"]
        assert "department_1" in result["message"]

    def test_three_duplicates_get_distinct_unique_names(self):
        model = _model(extra_relationships={
            "rel-3": {
                "id": "rel-3", "type": "ClassComposition", "name": "",
                "source": {"element": "doc-1", "role": "", "multiplicity": "1"},
                "target": {"element": "dep-1", "role": "", "multiplicity": "*"},
            },
        })
        result = _handler().generate_modification(_AUTO_FIX_MSG, model)
        assert result["action"] == "modify_model"
        mods = result["modifications"]
        assert [m["target"]["relationshipId"] for m in mods] == ["rel-2", "rel-3"]
        new_names = [m["changes"]["roleName"] for m in mods]
        assert new_names == ["department_1", "department_2"]

    def test_unique_name_skips_names_already_taken(self):
        # rel-2's Department end is fine ('boss_department'); a third
        # association already uses 'department_1', so the rename of the
        # colliding rel-3 end must skip to 'department_2'.
        model = _model(
            rel_overrides={
                "rel-2": {"target": {"element": "dep-1", "role": "boss_department",
                                      "multiplicity": "1"}},
            },
            extra_relationships={
                "rel-3": {
                    "id": "rel-3", "type": "ClassBidirectional", "name": "",
                    "source": {"element": "doc-1", "role": "", "multiplicity": "1"},
                    "target": {"element": "dep-1", "role": "department_1",
                               "multiplicity": "*"},
                },
                "rel-4": {
                    "id": "rel-4", "type": "ClassBidirectional", "name": "",
                    "source": {"element": "doc-1", "role": "", "multiplicity": "1"},
                    "target": {"element": "dep-1", "role": "department",
                               "multiplicity": "*"},
                },
            },
        )
        result = _handler().generate_modification(_AUTO_FIX_MSG, model)
        assert result["action"] == "modify_model"
        mod = result["modification"]
        assert mod["target"]["relationshipId"] == "rel-4"
        assert mod["changes"]["roleName"] == "department_2"

    def test_keeps_source_side_end_and_renames_target_side(self):
        # rel-2 is reversed (Department → Doctor): Doctor's duplicate end
        # there is the JSON SOURCE endpoint, which the frontend modifier
        # cannot rename — so it is KEPT and the rel-1 target end is renamed.
        model = _model(rel_overrides={
            "rel-2": {
                "source": {"element": "dep-1", "role": "", "multiplicity": "1"},
                "target": {"element": "doc-1", "role": "", "multiplicity": "0..1"},
            },
        })
        result = _handler().generate_modification(_AUTO_FIX_MSG, model)
        assert result["action"] == "modify_model"
        mod = result["modification"]
        assert mod["target"]["relationshipId"] == "rel-1"
        assert mod["changes"]["roleName"] == "department_1"

    def test_inherited_duplicate_across_generalization_is_repaired(self):
        # Person → Department gives every Doctor (child of Person) an
        # inherited 'department' end; Doctor's own association duplicates it.
        model = _model(
            rel_overrides={
                # Re-point rel-2 at the parent class so the duplicate spans
                # the inheritance chain instead of sitting on Doctor twice.
                "rel-2": {
                    "source": {"element": "per-1", "role": "", "multiplicity": "*"},
                    "target": {"element": "dep-1", "role": "", "multiplicity": "1"},
                },
            },
            extra_elements={
                "per-1": {"id": "per-1", "type": "Class", "name": "Person"},
            },
            extra_relationships={
                "rel-inh": {
                    "id": "rel-inh", "type": "ClassInheritance", "name": "",
                    "source": {"element": "doc-1", "role": ""},
                    "target": {"element": "per-1", "role": ""},
                },
            },
        )
        result = _handler().generate_modification(_AUTO_FIX_MSG, model)
        assert result["action"] == "modify_model"
        mod = result["modification"]
        assert mod["target"]["relationshipId"] == "rel-2"
        assert mod["changes"]["roleName"] == "department_1"


# ---------------------------------------------------------------------------
# Honesty: never claim a fix that was not applied
# ---------------------------------------------------------------------------

class TestHonestReplies:
    def test_unfixable_source_side_duplicates_reply_honestly(self):
        # BOTH duplicated ends sit on JSON source endpoints (both
        # associations point Department → Doctor); the frontend modifier can
        # only rename target-side roles, so no automatic fix exists.
        model = _model(rel_overrides={
            "rel-1": {
                "source": {"element": "dep-1", "role": "", "multiplicity": "1"},
                "target": {"element": "doc-1", "role": "", "multiplicity": "*"},
            },
            "rel-2": {
                "source": {"element": "dep-1", "role": "", "multiplicity": "1"},
                "target": {"element": "doc-1", "role": "", "multiplicity": "0..1"},
            },
        })
        result = _handler().generate_modification(_AUTO_FIX_MSG, model)
        assert result["action"] == "assistant_message"
        assert "couldn't" in result["message"].lower()
        assert "rename" in result["message"].lower()

    def test_missing_model_replies_honestly(self):
        result = _handler().generate_modification(_AUTO_FIX_MSG, None)
        assert result["action"] == "assistant_message"
        assert "couldn't" in result["message"].lower()

    def test_no_duplicate_found_replies_honestly(self):
        # The roles are already unique — nothing to fix; the agent must not
        # fabricate a modification or claim success.
        model = _model(rel_overrides={
            "rel-2": {"target": {"element": "dep-1", "role": "headed_department",
                                  "multiplicity": "1"}},
        })
        result = _handler().generate_modification(_AUTO_FIX_MSG, model)
        assert result["action"] == "assistant_message"
        assert "couldn't" in result["message"].lower()


# ---------------------------------------------------------------------------
# Mixed-error post-guard: uniqueness is enforced on top of LLM output
# ---------------------------------------------------------------------------

class TestMixedPostGuard:
    def test_strips_label_rename_and_appends_role_fix(self):
        handler = _handler()
        llm_spec = {
            "action": "modify_model",
            "modification": {
                "action": "modify_relationship",
                "target": {"sourceClass": "Doctor", "targetClass": "Department"},
                # The historical failure shape: a relationship-LABEL rename,
                # which the validator ignores.
                "changes": {"name": "DoctorDepartment2"},
            },
            "message": "Updated relationship in Doctor - Department.",
        }
        result = handler._apply_duplicate_end_post_guard(
            llm_spec, [("Doctor", "department")], _model(),
        )
        mods = result.get("modifications") or [result["modification"]]
        # The label rename is gone; the deterministic role rename is present.
        assert all(m["changes"].get("name") != "DoctorDepartment2" for m in mods)
        role_mods = [m for m in mods if m["changes"].get("roleName")]
        assert len(role_mods) == 1
        assert role_mods[0]["target"]["relationshipId"] == "rel-2"
        assert role_mods[0]["changes"]["roleName"] == "department_1"
        assert "department_1" in result["message"]

    def test_keeps_unrelated_llm_mods(self):
        handler = _handler()
        llm_spec = {
            "action": "modify_model",
            "modifications": [
                {
                    "action": "modify_attribute",
                    "target": {"className": "Doctor", "attributeName": "name"},
                    "changes": {"type": "String"},
                },
            ],
            "message": "Updated attribute in Doctor.",
        }
        result = handler._apply_duplicate_end_post_guard(
            llm_spec, [("Doctor", "department")], _model(),
        )
        mods = result["modifications"]
        assert any(m["action"] == "modify_attribute" for m in mods)
        assert any(m["changes"].get("roleName") == "department_1" for m in mods)


# ---------------------------------------------------------------------------
# Schema: the LLM can now express an end rename at all
# ---------------------------------------------------------------------------

class TestSchemaRoleName:
    def test_role_name_is_accepted_by_the_modification_schema(self):
        from schemas import ClassModificationResponse
        parsed = ClassModificationResponse.model_validate({
            "modifications": [{
                "action": "modify_relationship",
                "target": {"sourceClass": "Doctor", "targetClass": "Department"},
                "changes": {"roleName": "headedDepartment"},
            }],
        })
        assert parsed.modifications[0].changes.roleName == "headedDepartment"
