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
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

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


def real_editor_json(system_spec):
    """Use the production TS converter, not the simplified contract-only shim."""
    frontend = Path(__file__).resolve().parents[2] / "BESSER/besser/utilities/web_modeling_editor/frontend"
    if not shutil.which("node") or not (frontend / "node_modules/esbuild").is_dir():
        return None
    converter = frontend / "packages/webapp/src/main/features/assistant/services/converters/ClassDiagramConverter.ts"
    script = """
const { buildSync } = require('esbuild');
const fs = require('fs');
const bundle = buildSync({entryPoints: [process.argv[1]], bundle: true,
  platform: 'node', format: 'cjs', write: false}).outputFiles[0].text;
const loaded = {exports: {}};
new Function('module', 'exports', 'require', bundle)(loaded, loaded.exports, require);
const spec = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(new loaded.exports.ClassDiagramConverter().convertCompleteSystem(spec)));
"""
    result = subprocess.run(
        ["node", "-e", script, str(converter)], cwd=frontend,
        input=json.dumps(system_spec), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return {"title": system_spec["systemName"], "model": json.loads(result.stdout)}


def native_http_report(model):
    """Generate trusted fixture code and exercise its attributed-link API in SQLite."""
    import asyncio
    import importlib
    import httpx
    from besser.generators.backend.backend_generator import BackendGenerator

    saved_cwd = os.getcwd()
    saved_url = os.environ.get("DATABASE_URL")
    with tempfile.TemporaryDirectory(prefix="besser_native_contract_") as temporary:
        database = None
        try:
            BackendGenerator(model=model, output_dir=temporary).generate()
            os.environ["DATABASE_URL"] = f"sqlite:///{Path(temporary).as_posix()}/contract.db"
            sys.path.insert(0, temporary)
            os.chdir(temporary)
            app = importlib.import_module("main_api").app
            database = importlib.import_module("database")

            async def exercise():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://contract"
                ) as client:
                    async def create(kind, data):
                        response = await client.post(f"/{kind}/", json=data)
                        assert response.status_code == 200, response.text
                        body = response.json()
                        return body.get(kind, body)["id"]

                    room_id = await create("room", {"roomNumber": "101"})
                    guest_id = await create("guest", {"name": "Ada"})
                    for number, price, extras in ((1, 100.0, 12.5), (2, 200.0, 0.0)):
                        await create("booking", {
                            "number": number, "guests": [guest_id],
                            "rooms": [{"target": room_id, "agreedPrice": price, "extraCharges": extras}],
                        })
                    empty = await client.post("/booking/", json={"number": 3, "guests": [guest_id], "rooms": []})
                    duplicate = await client.post("/room/", json={"roomNumber": "101"})
                    return {"empty_rooms_status": empty.status_code, "duplicate_room_status": duplicate.status_code}

            report = asyncio.run(exercise())
            orm = importlib.import_module("sql_alchemy")
            with database.SessionLocal() as session:
                links = session.query(orm.ReservedRoom).all()
                report["links"] = sorted([link.agreedPrice, link.extraCharges] for link in links)
                assert session.query(orm.Booking).count() == 2, "failed create must not persist a booking"
            return report
        finally:
            if database is not None:
                database.SessionLocal.kw["bind"].dispose()
            os.chdir(saved_cwd)
            if temporary in sys.path:
                sys.path.remove(temporary)
            if saved_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = saved_url


def main():
    payload = json.load(sys.stdin)
    if payload["kind"] == "native":
        diagram = real_editor_json(payload["data"])
        if diagram is None:
            print(json.dumps({"skip": "native round-trip needs Node and the sibling frontend's installed esbuild"}))
            return
    else:
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
    if payload["kind"] == "native":
        report["http"] = native_http_report(model)
    out.write(json.dumps(report))


if __name__ == "__main__":
    main()
