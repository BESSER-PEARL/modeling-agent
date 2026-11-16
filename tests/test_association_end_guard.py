"""Association-end uniqueness at CREATION time.

Two associations between the same pair of classes with no distinct role
names collide once injected: the frontend converter emits
``source.role=rel.sourceRole||''`` and ``target.role=rel.name||''``, and the
validator derives every empty role from the lowercased endpoint-class name —
so a Doctor who "works in" AND "heads" a Department gets two 'department'
ends and a guaranteed "cannot have two association ends with the same name"
error on rich models.

``ClassDiagramHandler._ensure_unique_association_ends`` post-processes the
generated spec before injection: it NAMES the colliding ends, writing
``name`` for a target end and ``sourceRole`` for a source end, using the
validator's own derivation and counting ends inherited across generalization
chains. It never swaps endpoints — ``name`` and ``sourceRole`` label ENDS, so
a swap moves each name to the opposite end.

The modification path gets the same protection for ``add_relationship`` mods
(``_ensure_unique_ends_for_added_relationships``), reusing the repair
helpers against the current model. Handlers run with a None LLM — the guards
are pure post-processing.
"""

import copy

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


def _handler() -> ClassDiagramHandler:
    return ClassDiagramHandler(None)


def _assoc(src, tgt, name=None, type_="Association", sm="1", tm="0..*"):
    return {
        "type": type_, "source": src, "target": tgt,
        "sourceMultiplicity": sm, "targetMultiplicity": tm, "name": name,
    }


def _spec(rels, classes=("Doctor", "Department")):
    return {
        "systemName": "Hospital",
        "classes": [
            {"className": c, "attributes": [], "methods": []} for c in classes
        ],
        "relationships": [copy.deepcopy(r) for r in rels],
    }


def _effective_ends(spec):
    """Mirror the injected-model derivation: for rel S→T the source class owns
    an end named (rel.name or lower(T)); the target class owns an end named
    (rel.sourceRole or lower(S))."""
    ends = {}
    inheritance_edges = []
    for rel in spec["relationships"]:
        rel_type = str(rel.get("type") or "").strip().lower()
        if rel_type in ("inheritance", "generalization"):
            inheritance_edges.append((rel["source"], rel["target"]))
            continue
        label = (rel.get("name") or "").strip()
        source_role = (rel.get("sourceRole") or "").strip()
        ends.setdefault(rel["source"], []).append(label or rel["target"].lower())
        ends.setdefault(rel["target"], []).append(source_role or rel["source"].lower())
    return ends, inheritance_edges


def _assert_no_duplicate_ends(spec):
    ends, inheritance_edges = _effective_ends(spec)
    for cls in ends:
        chain = ClassDiagramHandler._reach_over_edges(inheritance_edges, {cls})
        names = [n for c in chain for n in ends.get(c, [])]
        assert len(names) == len(set(names)), (
            f"class '{cls}' would see duplicate association ends: {sorted(names)}"
        )


# ---------------------------------------------------------------------------
# Complete-system spec guard
# ---------------------------------------------------------------------------

class TestSpecEndUniqueness:
    def test_two_unlabeled_parallel_associations_become_unique(self):
        """The recurring shape: two unnamed Doctor→Department links."""
        spec = _spec([
            _assoc("Doctor", "Department", sm="*", tm="1"),
            _assoc("Doctor", "Department", sm="0..1", tm="1"),
        ])
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_duplicate_ends(spec)
        # Both colliding ends are named in place; nothing is reoriented and no
        # multiplicity moves.
        assert [(r["source"], r["target"]) for r in spec["relationships"]] == [
            ("Doctor", "Department"), ("Doctor", "Department")]
        assert [(r["sourceMultiplicity"], r["targetMultiplicity"])
                for r in spec["relationships"]] == [("*", "1"), ("0..1", "1")]
        second = spec["relationships"][1]
        assert second["name"] == "department_1"
        assert second["sourceRole"] == "doctor_1"

    def test_opposite_orientation_distinct_names_untouched(self):
        """An already-clean pair — distinct roles, opposite orientations —
        must pass through byte-identical."""
        rels = [
            _assoc("Doctor", "Department", name="worksIn", sm="*", tm="1"),
            _assoc("Department", "Doctor", name="headedBy", sm="1", tm="0..1"),
        ]
        spec = _spec(rels)
        before = copy.deepcopy(spec)
        _handler()._ensure_unique_association_ends(spec)
        assert spec == before
        _assert_no_duplicate_ends(spec)

    def test_same_orientation_named_pair_keeps_names_and_orientation(self):
        """Distinct names fix Doctor's side; the shared target class still gets
        two source-derived 'doctor' ends, which a source role now fixes without
        touching the user's names or the stated direction."""
        spec = _spec([
            _assoc("Doctor", "Department", name="worksIn", sm="*", tm="1"),
            _assoc("Doctor", "Department", name="heads", sm="0..1", tm="1"),
        ])
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_duplicate_ends(spec)
        assert sorted((r.get("name") or "") for r in spec["relationships"]) == [
            "heads", "worksIn"]
        assert {(r["source"], r["target"]) for r in spec["relationships"]} == {
            ("Doctor", "Department")}
        assert spec["relationships"][1]["sourceRole"] == "doctor_1"

    def test_inherited_ends_are_counted(self):
        """A parent's end collides with a child's own end across the
        generalization chain (matching the repair's closure semantics)."""
        person_rel = _assoc("Person", "Department", sm="*", tm="1")
        doctor_rel = _assoc("Doctor", "Department", sm="0..1", tm="1")
        spec = _spec(
            [
                person_rel,
                {"type": "Inheritance", "source": "Doctor", "target": "Person",
                 "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": None},
                doctor_rel,
            ],
            classes=("Person", "Doctor", "Department"),
        )
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_duplicate_ends(spec)
        # The parent's link is the kept occurrence; the child's is renamed.
        assert not (spec["relationships"][0].get("name") or "")
        assert spec["relationships"][2]["name"] == "department_1"

    def test_single_unnamed_self_association_gets_a_name(self):
        """A self-association contributes BOTH ends to the same class — even
        one unnamed self-link collides with itself. The source end takes the
        suffix, so the association's own label stays free."""
        spec = _spec(
            [_assoc("Employee", "Employee", sm="0..1", tm="*")],
            classes=("Employee",),
        )
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_duplicate_ends(spec)
        assert spec["relationships"][0]["sourceRole"] == "employee_1"

    def test_new_names_avoid_existing_relationship_labels(self):
        """Assigned end names double as relationship labels, which must stay
        unique among labels too."""
        spec = _spec(
            [
                _assoc("Doctor", "Department"),
                _assoc("Doctor", "Department"),
                _assoc("Hospital", "Clinic", name="doctor_1"),
            ],
            classes=("Doctor", "Department", "Hospital", "Clinic"),
        )
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_duplicate_ends(spec)
        labels = [(r.get("name") or "").lower() for r in spec["relationships"]]
        assert len([l for l in labels if l]) == len(set(l for l in labels if l))

    def test_composition_pairs_are_not_reoriented(self):
        """Composition orientation is semantic (whole→part) — never flipped;
        the nameable side is still made unique."""
        spec = _spec([
            _assoc("Doctor", "Department", type_="Composition", sm="1", tm="*"),
            _assoc("Doctor", "Department", type_="Composition", sm="1", tm="*"),
        ])
        _handler()._ensure_unique_association_ends(spec)
        for rel in spec["relationships"]:
            assert (rel["source"], rel["target"]) == ("Doctor", "Department")
        ends, _ = _effective_ends(spec)
        doctor_ends = ends["Doctor"]
        assert len(doctor_ends) == len(set(doctor_ends))

    def test_inheritance_only_spec_untouched(self):
        spec = _spec(
            [{"type": "Inheritance", "source": "Doctor", "target": "Person",
              "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": None}],
            classes=("Person", "Doctor"),
        )
        before = copy.deepcopy(spec)
        _handler()._ensure_unique_association_ends(spec)
        assert spec == before


# ---------------------------------------------------------------------------
# An attribute and an association end of one name on a class
# ---------------------------------------------------------------------------

def _attr_spec(attributes, rels, classes=("Article", "Keyword")):
    """*attributes*: {class name: [attribute dicts]}."""
    spec = _spec(rels, classes=classes)
    for cls in spec["classes"]:
        cls["attributes"] = copy.deepcopy(attributes.get(cls["className"], []))
    return spec


def _attr_names(spec, class_name):
    cls = next(c for c in spec["classes"] if c["className"] == class_name)
    return [a["name"] for a in cls["attributes"]]


def _assert_no_attribute_named_like_an_end(spec):
    ends, inheritance_edges = _effective_ends(spec)
    for cls in spec["classes"]:
        line = ClassDiagramHandler._reach_over_edges(inheritance_edges, {cls["className"]})
        end_names = {n for c in line for n in ends.get(c, [])}
        clash = end_names & set(_attr_names(spec, cls["className"]))
        assert not clash, f"'{cls['className']}' has attribute and end named {sorted(clash)}"


class TestAttributeEndNameClash:
    def test_end_named_like_an_attribute_is_renamed(self):
        spec = _attr_spec(
            {"Article": [{"name": "title", "type": "str"}, {"name": "keywords", "type": "str"}]},
            [dict(_assoc("Article", "Keyword", name="keywords", sm="0..*", tm="0..*"),
                  sourceRole="articles")],
        )
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_attribute_named_like_an_end(spec)
        assert _attr_names(spec, "Article") == ["title", "keywords"]
        assert spec["relationships"][0]["name"] == "keywords_1"
        assert spec["relationships"][0]["sourceRole"] == "articles"

    def test_source_end_named_like_an_attribute_is_renamed(self):
        # Keyword is the target: it reaches Article through sourceRole.
        spec = _attr_spec(
            {"Keyword": [{"name": "articles", "type": "str"}]},
            [dict(_assoc("Article", "Keyword", name="keywords", sm="0..*", tm="0..*"),
                  sourceRole="articles")],
        )
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_attribute_named_like_an_end(spec)
        assert spec["relationships"][0]["sourceRole"] == "articles_1"
        assert spec["relationships"][0]["name"] == "keywords"

    def test_unnamed_end_takes_the_class_name_an_attribute_already_uses(self):
        spec = _attr_spec(
            {"Article": [{"name": "keyword", "type": "str"}]},
            [_assoc("Article", "Keyword", sm="0..*", tm="0..1")],
        )
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_attribute_named_like_an_end(spec)
        assert spec["relationships"][0]["name"] == "keyword_1"

    def test_inherited_attribute_counts(self):
        spec = _attr_spec(
            {"Post": [{"name": "keywords", "type": "str"}]},
            [{"type": "Inheritance", "source": "Article", "target": "Post",
              "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": None},
             dict(_assoc("Article", "Keyword", name="keywords", sm="0..*", tm="0..*"),
                  sourceRole="articles")],
            classes=("Post", "Article", "Keyword"),
        )
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_attribute_named_like_an_end(spec)
        assert spec["relationships"][1]["name"] == "keywords_1"

    def test_sibling_attribute_does_not_count(self):
        spec = _attr_spec(
            {"Memo": [{"name": "keywords", "type": "str"}]},
            [{"type": "Inheritance", "source": "Article", "target": "Post",
              "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": None},
             {"type": "Inheritance", "source": "Memo", "target": "Post",
              "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": None},
             dict(_assoc("Article", "Keyword", name="keywords", sm="0..*", tm="0..*"),
                  sourceRole="articles")],
            classes=("Post", "Article", "Memo", "Keyword"),
        )
        before = copy.deepcopy(spec)
        _handler()._ensure_unique_association_ends(spec)
        assert spec == before

    def test_integer_attribute_named_like_a_to_one_end_is_the_link(self):
        """``room: int`` beside a to-one end ``room`` holds the room's id."""
        spec = _attr_spec(
            {"Booking": [{"name": "nights", "type": "int"}, {"name": "room", "type": "int"}]},
            [dict(_assoc("Booking", "Room", name="room", sm="0..*", tm="1"),
                  sourceRole="bookings")],
            classes=("Booking", "Room"),
        )
        _handler()._ensure_unique_association_ends(spec)
        assert _attr_names(spec, "Booking") == ["nights"]
        assert spec["relationships"][0]["name"] == "room"

    def test_integer_attribute_named_like_a_to_many_end_is_kept(self):
        """``items: int`` beside a to-many end is a count, not a reference."""
        spec = _attr_spec(
            {"Cart": [{"name": "items", "type": "int"}]},
            [dict(_assoc("Cart", "Item", name="items", sm="1", tm="0..*"),
                  sourceRole="cart")],
            classes=("Cart", "Item"),
        )
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_attribute_named_like_an_end(spec)
        assert _attr_names(spec, "Cart") == ["items"]
        assert spec["relationships"][0]["name"] == "items_1"

    def test_a_name_a_rule_reads_is_left_on_both(self):
        """Renaming the end would silently point ``self.keywords->size()`` at
        the text attribute."""
        spec = _attr_spec(
            {"Article": [{"name": "keywords", "type": "str"}]},
            [dict(_assoc("Article", "Keyword", name="keywords", sm="0..*", tm="0..*"),
                  sourceRole="articles")],
        )
        spec["constraints"] = [{
            "context": "Article", "name": "tagged",
            "expression": "context Article inv tagged: self.keywords->size() >= 1"}]
        before = copy.deepcopy(spec)
        _handler()._ensure_unique_association_ends(spec)
        assert spec == before

    def test_an_integer_a_method_body_reads_is_kept(self):
        spec = _attr_spec(
            {"Booking": [{"name": "room", "type": "int"}]},
            [dict(_assoc("Booking", "Room", name="room", sm="0..*", tm="1"),
                  sourceRole="bookings")],
            classes=("Booking", "Room"),
        )
        spec["classes"][0]["methods"] = [
            {"name": "door", "code": "def door(self): return self.room + 100"}]
        before = copy.deepcopy(spec)
        _handler()._ensure_unique_association_ends(spec)
        assert spec == before

    def test_no_clash_is_untouched(self):
        spec = _attr_spec(
            {"Article": [{"name": "title", "type": "str"}]},
            [dict(_assoc("Article", "Keyword", name="keywords", sm="0..*", tm="0..*"),
                  sourceRole="articles")],
        )
        before = copy.deepcopy(spec)
        _handler()._ensure_unique_association_ends(spec)
        assert spec == before

    def test_generated_system_has_one_member_per_name(self, monkeypatch):
        """Through generate_complete_system: a collection attribute beside the
        composition it restates, and a text attribute named like an end."""
        import diagram_handlers.types.class_diagram_handler as handler_mod
        from schemas.compact_class_diagram import (
            CompactClassSpec, CompactRelationshipSpec, CompactSystemClassSpec,
        )

        def rel(f, t, k, per_source, per_target, name, source_role):
            return CompactRelationshipSpec(
                f=f, t=t, k=k, how_many_TARGET_for_one_SOURCE=per_source,
                how_many_SOURCE_for_one_TARGET=per_target, l=name, ls=source_role)

        compact = CompactSystemClassSpec(
            name="Blog",
            classes=[
                CompactClassSpec(n="Article", m=[], k="",
                                 a=["title: str", "sections: List[Section]", "keywords: str"]),
                CompactClassSpec(n="Section", a=["heading: str"], m=[], k=""),
                CompactClassSpec(n="Keyword", a=["word: str"], m=[], k=""),
            ],
            rels=[rel("Section", "Article", "comp", "1", "0..*", "article", "sections"),
                  rel("Article", "Keyword", "assoc", "0..*", "0..*", "keywords", "articles")],
            ocl=[],
        )
        monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
        handler = _handler()
        monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **kwargs: compact)

        spec = handler.generate_complete_system("a blog")["systemSpec"]

        _assert_no_attribute_named_like_an_end(spec)
        assert _attr_names(spec, "Article") == ["title", "keywords"]
        assert [(r["type"], r["source"], r["target"], r["name"]) for r in spec["relationships"]] == [
            ("Composition", "Section", "Article", "article"),
            ("Association", "Article", "Keyword", "keywords_1"),
        ]


# ---------------------------------------------------------------------------
# Modification-path guard (add_relationship)
# ---------------------------------------------------------------------------

_LINKED_MODEL = {
    "elements": {
        "doc-1": {"id": "doc-1", "type": "Class", "name": "Doctor"},
        "dep-1": {"id": "dep-1", "type": "Class", "name": "Department"},
        "hos-1": {"id": "hos-1", "type": "Class", "name": "Hospital"},
    },
    "relationships": {
        "rel-1": {
            "id": "rel-1", "type": "ClassBidirectional", "name": "",
            "source": {"element": "doc-1", "role": "", "multiplicity": "*"},
            "target": {"element": "dep-1", "role": "", "multiplicity": "1"},
        },
    },
}


def _add_mod(src, tgt, rel_type="Association", name=None):
    changes = {"relationshipType": rel_type}
    if name:
        changes["name"] = name
    return {
        "action": "add_relationship",
        "target": {"sourceClass": src, "targetClass": tgt},
        "changes": changes,
    }


class TestAddRelationshipEndGuard:
    def test_second_link_between_linked_pair_is_reoriented_and_repaired(self):
        """Adding a second unnamed Doctor–Department link: both orientations
        collide against the existing unnamed link, so the guard flips the new
        one toward the side whose existing end IS renameable, names it, and
        appends a roleName repair for the existing relationship."""
        spec = {
            "action": "modify_model",
            "modification": _add_mod("Doctor", "Department"),
            "message": "Added relationship to Doctor → Department.",
        }
        _handler()._ensure_unique_ends_for_added_relationships(
            spec, copy.deepcopy(_LINKED_MODEL),
        )
        mods = spec["modifications"]
        add = next(m for m in mods if m["action"] == "add_relationship")
        repair = next(m for m in mods if m["action"] == "modify_relationship")
        # The new link was flipped so its fixed (source-derived) end lands on
        # the side where the existing colliding end can be renamed.
        assert add["target"]["sourceClass"] == "Department"
        assert add["target"]["targetClass"] == "Doctor"
        assert add["changes"]["name"] == "doctor_1"
        # The existing relationship's colliding end got a companion rename.
        assert repair["target"]["relationshipId"] == "rel-1"
        assert repair["changes"]["roleName"] == "department_1"

    def test_add_between_unlinked_pair_untouched(self):
        mod = _add_mod("Doctor", "Hospital")
        spec = {
            "action": "modify_model",
            "modification": copy.deepcopy(mod),
            "message": "Added relationship to Doctor → Hospital.",
        }
        _handler()._ensure_unique_ends_for_added_relationships(
            spec, copy.deepcopy(_LINKED_MODEL),
        )
        assert "modifications" not in spec
        assert spec["modification"] == mod

    def test_add_with_distinct_name_to_named_pair_untouched(self):
        """When the existing pair link carries distinct roles already, a new
        named link in the opposite orientation needs no adjustment."""
        model = copy.deepcopy(_LINKED_MODEL)
        model["relationships"]["rel-1"]["target"]["role"] = "worksIn"
        mod = _add_mod("Department", "Doctor", name="headedBy")
        spec = {
            "action": "modify_model",
            "modification": copy.deepcopy(mod),
            "message": "Added relationship to Department → Doctor.",
        }
        _handler()._ensure_unique_ends_for_added_relationships(spec, model)
        assert "modifications" not in spec
        assert spec["modification"] == mod
