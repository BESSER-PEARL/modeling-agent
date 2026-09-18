"""Live probe: does the hotel request still invert association ends?

Sends the hotel spec N times and reports, per run, the multiplicity the agent
put on each end plus whether any class ends up needing another class before it
can be created (a mandatory single end, which becomes a NOT NULL FK and a
required create field).

    python tests/live/probe_hotel_multiplicity.py [runs]

Env: AGENT_WS_URL (default wss://experimental.besser-pearl.org/agent)
"""
import asyncio
import json
import os
import pathlib
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AGENT_WS_URL", "wss://experimental.besser-pearl.org/agent")

from _agent_ws import connect as agent_ws_connect  # noqa: E402
from test_nl_generation_scenarios import _unwrap, AGENT_WS_URL  # noqa: E402

TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "240"))

SPEC_PATH = pathlib.Path(
    os.environ.get("HOTEL_SPEC", r"C:\Users\sulejmani\Desktop\BESSER-Experimental\hotel_spec.txt"))
PROMPT = SPEC_PATH.read_text(encoding="utf-8")

# What a correct reading of the request gives, as (source_class, target_class)
# -> the end that must NOT be a mandatory single. Keyed by the pair so the
# check survives whichever orientation the agent emits.
SHARED_ENTITIES = {"Person", "Guest", "Employee", "Room"}


async def _send(ws, sid, text):
    inner = {"action": "user_message", "protocolVersion": "2.0", "clientMode": "widget",
             "sessionId": sid, "message": text,
             "context": {"activeDiagramType": "ClassDiagram"}}
    await ws.send(json.dumps({"action": "user_message", "user_id": sid,
                              "message": json.dumps(inner)}))


async def _await_system(ws):
    t0 = time.monotonic()
    while time.monotonic() - t0 < TIMEOUT:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=20)
        except asyncio.TimeoutError:
            continue
        m = _unwrap(raw)
        if not m:
            continue
        if m.get("action") == "inject_complete_system":
            return m.get("systemSpec") or {}
        if m.get("action") in ("agent_error", "error"):
            return {"__error__": m.get("message")}
    return {}


def _mandatory_single(value):
    text = str(value or "").strip().replace(" ", "")
    return text in ("1", "1..1")


def analyse(spec, index):
    if spec.get("__error__"):
        print(f"  run {index}: ERROR {spec['__error__'][:160]}")
        return None
    rels = [r for r in spec.get("relationships", [])
            if str(r.get("type", "")).lower() not in ("inheritance", "generalization")]
    classes = [c.get("className") for c in spec.get("classes", [])]
    constraints = spec.get("constraints", []) or []

    print(f"  run {index}: {len(classes)} classes, {len(rels)} assoc, "
          f"{len(constraints)} OCL")
    blocked = []
    for r in rels:
        src, tgt = r.get("source"), r.get("target")
        sm = r.get("sourceMultiplicity")
        tm = r.get("targetMultiplicity")
        name = r.get("name") or ""
        print(f"     {src:<14}[{str(sm):>5}] --{name:<15}-- [{str(tm):>5}]{tgt}")
        # A mandatory single on the TARGET end means every SOURCE needs a target.
        if _mandatory_single(tm) and src in SHARED_ENTITIES:
            blocked.append(f"{src} needs a {tgt}")
        if _mandatory_single(sm) and tgt in SHARED_ENTITIES:
            blocked.append(f"{tgt} needs a {src}")

    pairs = {}
    for r in rels:
        key = tuple(sorted((str(r.get("source")), str(r.get("target")))))
        pairs[key] = pairs.get(key, 0) + 1
    dupes = {k: v for k, v in pairs.items() if v > 1}

    print(f"     -> uncreatable-alone: {blocked or 'none'}")
    print(f"     -> duplicated pairs : {dupes or 'none'}")
    return {"blocked": blocked, "dupes": dupes, "ocl": len(constraints),
            "classes": len(classes), "rels": len(rels)}


async def main(runs):
    print(f"agent: {AGENT_WS_URL}")
    print(f"prompt: {SPEC_PATH} ({len(PROMPT)} chars)\n")
    results = []
    for i in range(1, runs + 1):
        try:
            async with agent_ws_connect(AGENT_WS_URL) as ws:
                await _send(ws, "s_" + uuid.uuid4().hex[:6], PROMPT)
                spec = await _await_system(ws)
        except Exception as exc:  # noqa: BLE001
            print(f"  run {i}: connection failed: {exc}")
            continue
        if not spec:
            print(f"  run {i}: no inject_complete_system within {TIMEOUT}s")
            continue
        r = analyse(spec, i)
        if r:
            results.append(r)
        print()

    if not results:
        print("no usable runs")
        return 1
    clean = [r for r in results if not r["blocked"] and not r["dupes"]]
    print("=" * 64)
    print(f"runs analysed        : {len(results)}")
    print(f"no inverted end      : {sum(1 for r in results if not r['blocked'])}/{len(results)}")
    print(f"no duplicate pair    : {sum(1 for r in results if not r['dupes'])}/{len(results)}")
    print(f"OCL captured         : {sum(1 for r in results if r['ocl'])}/{len(results)}")
    print(f"fully clean          : {len(clean)}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)))
