"""Run a class-diagram spec through BESSER and report its model-contract verdict.

Used by ``test_hotel_model_contract.py``. Runs in its own interpreter so the
checks come from the BESSER checkout named in ``argv[1]`` (the workspace
sibling) even when the test process has already bound ``besser`` to another
install — ``test_confirmation.py`` does, and a failed ``import besser.agent``
still leaves the package in ``sys.modules``.

stdin:  ``{"kind": "spec", "data": <systemSpec>}`` — the agent's
        ``inject_complete_system`` payload, converted to editor JSON the way
        the frontend's ``ClassDiagramConverter.convertCompleteSystem`` does it
        (source role empty, target role = relationship name, one OCL box per
        context class; attributes left out, the checks navigate ends only) —
        or ``{"kind": "diagram", "data": <editor diagram>}`` as exported.
stdout: ``{"skip": reason}`` when this BESSER cannot answer, else
        ``{"validate": ..., "ocl_warnings": [...], "constraints": [...],
        "classes": {name: {"kind": ..., "ends": {end: [type, min, max]}}}}``.
"""
import json
import sys

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

try:
    from besser.BUML.metamodel.structural import DomainModel
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (  # noqa: E501
        process_class_diagram,
    )
except ImportError as exc:  # pragma: no cover - environment dependent
    print(json.dumps({"skip": f"BESSER is not importable: {exc}"}))
    sys.exit(0)

if not hasattr(DomainModel, "_validate_mandatory_cycles"):
    print(json.dumps({"skip": "this BESSER checkout predates the model-contract checks"}))
    sys.exit(0)


def editor_json(system_spec):
    elements, relationships, ids = {}, {}, {}
    for i, cls in enumerate(system_spec["classes"]):
        cid = f"class-{i}"
        ids[cls["className"]] = cid
        elements[cid] = {
            "id": cid, "name": cls["className"],
            "type": "Enumeration" if cls.get("isEnumeration") else "Class",
            "owner": None, "attributes": [], "methods": [],
            "bounds": {"x": 0, "y": 0, "width": 200, "height": 100},
        }
    type_map = {
        "inheritance": "ClassInheritance", "generalization": "ClassInheritance",
        "composition": "ClassComposition", "aggregation": "ClassAggregation",
    }
    for i, rel in enumerate(system_spec["relationships"]):
        rid = f"rel-{i}"
        relationships[rid] = {
            "id": rid, "name": rel.get("name") or "",
            "type": type_map.get(str(rel.get("type") or "").lower(), "ClassBidirectional"),
            "source": {"element": ids[rel["source"]], "direction": "Left",
                       "multiplicity": rel.get("sourceMultiplicity") or "1", "role": ""},
            "target": {"element": ids[rel["target"]], "direction": "Right",
                       "multiplicity": rel.get("targetMultiplicity") or "1",
                       "role": rel.get("name") or ""},
        }
    by_context = {}
    for con in system_spec.get("constraints") or []:
        expression = str(con.get("expression") or "").strip()
        if expression and con.get("context") in ids:
            by_context.setdefault(con["context"], []).append(expression)
    for i, (context, expressions) in enumerate(by_context.items()):
        oid, lid = f"ocl-{i}", f"ocllink-{i}"
        elements[oid] = {
            "id": oid, "name": "", "type": "ClassOCLConstraint", "owner": None,
            "bounds": {"x": -700, "y": i * 170, "width": 640, "height": 130},
            "description": "", "constraint": "\n\n".join(expressions),
        }
        relationships[lid] = {
            "id": lid, "name": "", "type": "ClassOCLLink", "owner": None,
            "source": {"element": oid, "direction": "Right", "multiplicity": "", "role": ""},
            "target": {"element": ids[context], "direction": "Left", "multiplicity": "", "role": ""},
        }
    return {
        "title": system_spec.get("systemName") or "Model",
        "model": {"version": "3.0.0", "type": "ClassDiagram",
                  "elements": elements, "relationships": relationships},
    }


def main():
    payload = json.load(sys.stdin)
    diagram = payload["data"] if payload["kind"] == "diagram" else editor_json(payload["data"])
    out, sys.stdout = sys.stdout, sys.stderr  # BESSER chatters on stdout while converting
    model = process_class_diagram(diagram)
    report = {
        "validate": model.validate(raise_exception=False),
        "ocl_warnings": list(getattr(model, "ocl_warnings", [])),
        "constraints": sorted(c.name for c in model.constraints),
        "classes": {
            cls.name: {
                "kind": type(cls).__name__,
                "ends": {
                    end.name: [end.type.name, end.multiplicity.min, end.multiplicity.max]
                    for end in cls.association_ends()
                },
            }
            for cls in model.get_classes()
        },
    }
    out.write(json.dumps(report))


if __name__ == "__main__":
    main()
