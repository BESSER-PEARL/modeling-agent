"""Association-end uniqueness at CREATION time.

Two associations between the same pair of classes with no distinct role
names collide once injected: the frontend converter emits ``source.role=''``
and ``target.role=rel.name||''``, and the validator derives every empty role
from the lowercased endpoint-class name — so a Doctor who "works in" AND
"heads" a Department gets two 'department' ends and a guaranteed
"cannot have two association ends with the same name" error on rich models.

``ClassDiagramHandler._ensure_unique_association_ends`` post-processes the
generated spec before injection: it reorients a later same-direction parallel
Association (direction is not semantic for the bidirectional type; the flip
makes the otherwise-unnameable source-derived end name-addressable) and then
assigns unique relationship names using the validator's own derivation,
counting ends inherited across generalization chains — matching the
duplicate-end repair's semantics.

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
    """Mirror the injected-model derivation: for rel S→T the source class
    owns an end named (rel.name or lower(T)); the target class owns an end
    named lower(S) (the injected source role is always empty)."""
    ends = {}
    inheritance_edges = []
    for rel in spec["relationships"]:
        rel_type = str(rel.get("type") or "").strip().lower()
        if rel_type in ("inheritance", "generalization"):
            inheritance_edges.append((rel["source"], rel["target"]))
            continue
        label = (rel.get("name") or "").strip()
        ends.setdefault(rel["source"], []).append(label or rel["target"].lower())
        ends.setdefault(rel["target"], []).append(rel["source"].lower())
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
        """The recurring live shape: two unnamed Doctor→Department links."""
        spec = _spec([
            _assoc("Doctor", "Department", sm="*", tm="1"),
            _assoc("Doctor", "Department", sm="0..1", tm="1"),
        ])
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_duplicate_ends(spec)
        orientations = {
            (r["source"], r["target"]) for r in spec["relationships"]
        }
        # One link was reoriented so BOTH classes' colliding ends became
        # name-addressable (the injected source role is never nameable).
        assert orientations == {("Doctor", "Department"), ("Department", "Doctor")}
        # The flip preserved the multiplicities' meaning by swapping them.
        flipped = next(
            r for r in spec["relationships"] if r["source"] == "Department"
        )
        assert (flipped["sourceMultiplicity"], flipped["targetMultiplicity"]) == ("1", "0..1")

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

    def test_same_orientation_named_pair_keeps_names_and_flips_one(self):
        """Distinct names alone cannot fix a same-direction pair (the shared
        target class still gets two source-derived 'doctor' ends) — one link
        is flipped, but the user's names are preserved."""
        spec = _spec([
            _assoc("Doctor", "Department", name="worksIn", sm="*", tm="1"),
            _assoc("Doctor", "Department", name="heads", sm="0..1", tm="1"),
        ])
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_duplicate_ends(spec)
        names = sorted(
            (r.get("name") or "") for r in spec["relationships"]
        )
        assert names == ["heads", "worksIn"]
        orientations = {
            (r["source"], r["target"]) for r in spec["relationships"]
        }
        assert orientations == {("Doctor", "Department"), ("Department", "Doctor")}

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
        one unnamed self-link collides with itself."""
        spec = _spec(
            [_assoc("Employee", "Employee", sm="0..1", tm="*")],
            classes=("Employee",),
        )
        _handler()._ensure_unique_association_ends(spec)
        _assert_no_duplicate_ends(spec)
        assert spec["relationships"][0]["name"] == "employee_1"

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
