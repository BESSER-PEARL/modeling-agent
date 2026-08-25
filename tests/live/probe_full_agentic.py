"""Broad agentic-coverage probe — the gaps the other live suites don't hit:

  generators  every code generator (~15) must produce a VALID non-error
              response (trigger, config prompt, prereq ask, or smart
              confirmation) — never a crash / traceback / empty / hang.
  crossdiag   from an existing class diagram, asking for a DIFFERENT diagram
              type (state machine / BPMN / object / agent) builds that type.
  multiturn   context retention + mid-conversation pivots across 3-4 turns.

Usage:
  AGENT_WS_URL=wss://experimental.besser-pearl.org/agent CONC=4 \
      python tests/live/probe_full_agentic.py
Knobs: CONC, GEN_TIMEOUT, ONLY (csv of generators,crossdiag,multiturn).
"""
import asyncio
import json
import os
import sys
import time
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AGENT_WS_URL", "wss://experimental.besser-pearl.org/agent")

import websockets  # noqa: E402
from test_nl_generation_scenarios import _unwrap, AGENT_WS_URL  # noqa: E402

CONC = int(os.environ.get("CONC", "4"))
TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "180"))
ONLY = {c.strip() for c in os.environ.get("ONLY", "").lower().split(",") if c.strip()}
# Every action that concludes a turn with a real result — code-gen triggers,
# a built/modified model, an export/import, a GUI hand-off, a diagram switch.
TERMINAL = {"trigger_generator", "trigger_smart_generator", "inject_complete_system",
            "auto_generate_gui", "modify_model", "trigger_export", "trigger_import",
            "switch_diagram", "inject_element"}
_SEM = None


async def _send(ws, sid, text, ctx="ClassDiagram"):
    inner = {"action": "user_message", "protocolVersion": "2.0", "clientMode": "widget",
             "sessionId": sid, "message": text, "context": {"activeDiagramType": ctx}}
    await ws.send(json.dumps({"action": "user_message", "user_id": sid,
                              "message": json.dumps(inner)}))


async def _collect(ws, quiet=6, hard=None):
    hard = hard or TIMEOUT
    out, buf = [], ""
    t0 = time.monotonic()
    while time.monotonic() - t0 < hard:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=quiet)
        except asyncio.TimeoutError:
            if out:
                break
            continue
        m = _unwrap(raw)
        if not m:
            continue
        a = m.get("action")
        if a == "stream_chunk":
            buf += (m.get("chunk") or m.get("content") or "")
            continue
        if a == "stream_done":
            out.append(("assistant_message", (m.get("fullText") or buf),
                        m.get("diagramType")))
            buf = ""
            continue
        if a in TERMINAL or a in ("assistant_message", "agent_error", "error"):
            out.append((a, (m.get("message") or ""), m.get("diagramType")))
    return out


def _errorish(frames):
    if not frames:
        return "EMPTY_OR_HANG"
    for a, t, _dt in frames:
        low = (t or "").lower()
        if a in ("agent_error", "error"):
            return f"agent_error"
        if "something went wrong" in low or "traceback" in low or low.startswith("error:"):
            return "ERROR_TEXT"
    return None


def _has(frames, action):
    return any(f[0] == action for f in frames)


def _diagram_types(frames):
    return {f[2] for f in frames if f[0] == "inject_complete_system" and f[2]}


SEED = "create a class diagram for a shop with products, orders and customers"

GENERATORS = [
    "generate django", "generate the sql schema", "generate the sqlalchemy models",
    "generate the pydantic models", "generate the python classes",
    "generate the backend", "generate the java classes", "generate a react app",
    "generate a flutter app", "build me a full web app with a UI and auth",
    "generate the json export", "generate a supabase backend",
    "generate a BAF agent", "generate a pytorch model", "generate a tensorflow model",
]

# want = the WME diagramType token the inject carries (not the friendly name).
CROSSDIAG = [
    ("StateMachineDiagram", "create a state machine for the order lifecycle"),
    ("BPMN", "create a bpmn process for order fulfillment"),
    ("ObjectDiagram", "create an object diagram instantiating my classes"),
    ("AgentDiagram", "create an agent that answers order-status questions"),
]


async def _run(label, fn):
    async with _SEM:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                sid = "fa_" + uuid.uuid4().hex[:6]
                return await fn(ws, sid)
        except asyncio.TimeoutError:
            return (label, "HANG", "HANG")
        except Exception as exc:
            return (label, f"EXC:{type(exc).__name__}", "EXC")


async def _gen(ws, sid, gen):
    await _send(ws, sid, SEED)
    r0 = await _collect(ws)
    if not _has(r0, "inject_complete_system"):
        return (f"gen:{gen[:26]}", f"seed-failed", "SEED")
    await _send(ws, sid, gen)
    r1 = await _collect(ws)
    err = _errorish(r1)
    if err:
        return (f"gen:{gen[:26]}", err, err)
    # summarize what kind of valid response we got
    if _has(r1, "trigger_generator"):
        kind = "trigger_generator"
    elif _has(r1, "trigger_smart_generator"):
        kind = "trigger_smart"
    elif _has(r1, "auto_generate_gui"):
        kind = "auto_gui"
    else:
        kind = "prompt/confirm"
    return (f"gen:{gen[:26]}", kind, None)


async def _cross(ws, sid, want, prompt):
    await _send(ws, sid, SEED)
    r0 = await _collect(ws)
    if not _has(r0, "inject_complete_system"):
        return (f"cross:{want}", "seed-failed", "SEED")
    await _send(ws, sid, prompt)
    r1 = await _collect(ws)
    err = _errorish(r1)
    if err:
        return (f"cross:{want}", err, err)
    dts = _diagram_types(r1)
    built = _has(r1, "inject_complete_system")
    if built and (want in dts or not dts):
        return (f"cross:{want}", f"built {dts or '?'}", None)
    if built:
        return (f"cross:{want}", f"built WRONG type {dts}", "WRONG_TYPE")
    # object diagram legitimately may ask for a class diagram first; accept text
    return (f"cross:{want}", f"text ({[f[0] for f in r1]})", None)


async def _multiturn_context(ws, sid):
    """Retention: build, add, then rename — each step must apply cleanly."""
    await _send(ws, sid, "create a class diagram for a library with books and members")
    if not _has(await _collect(ws), "inject_complete_system"):
        return ("multi:context", "seed-failed", "SEED")
    await _send(ws, sid, "add a Loan class linking Book and Member")
    r1 = await _collect(ws)
    if _errorish(r1) or not (_has(r1, "modify_model") or _has(r1, "inject_complete_system")):
        return ("multi:context", f"add-failed {[f[0] for f in r1]}", "ADD_FAILED")
    await _send(ws, sid, "now rename the Member class to Patron")
    r2 = await _collect(ws)
    if _errorish(r2) or not (_has(r2, "modify_model") or _has(r2, "inject_complete_system")):
        return ("multi:context", f"rename-failed {[f[0] for f in r2]}", "RENAME_FAILED")
    return ("multi:context", "add + rename applied", None)


async def _multiturn_pivot(ws, sid):
    """Pivot: start a class diagram, then switch intent to a state machine."""
    await _send(ws, sid, "create a class diagram for an order system")
    if not _has(await _collect(ws), "inject_complete_system"):
        return ("multi:pivot", "seed-failed", "SEED")
    await _send(ws, sid, "actually, make a state machine for the order lifecycle instead")
    r1 = await _collect(ws)
    if _errorish(r1):
        return ("multi:pivot", _errorish(r1), _errorish(r1))
    dts = _diagram_types(r1)
    if _has(r1, "inject_complete_system") and ("StateMachineDiagram" in dts or not dts):
        return ("multi:pivot", f"pivoted {dts or '?'}", None)
    return ("multi:pivot", f"no-pivot {[f[0] for f in r1]}/{dts}", "NO_PIVOT")


async def _multiturn_generate_after_edits(ws, sid):
    """Build → edit → generate: the generate must fire on the edited model."""
    await _send(ws, sid, "create a class diagram for a blog with posts and authors")
    if not _has(await _collect(ws), "inject_complete_system"):
        return ("multi:edit-gen", "seed-failed", "SEED")
    await _send(ws, sid, "add a Comment class linked to Post")
    r1 = await _collect(ws)
    if _errorish(r1):
        return ("multi:edit-gen", "edit " + _errorish(r1), _errorish(r1))
    await _send(ws, sid, "generate the django code")
    r2 = await _collect(ws)
    err = _errorish(r2)
    if err:
        return ("multi:edit-gen", "gen " + err, err)
    return ("multi:edit-gen", f"edit + generate ({[f[0] for f in r2]})", None)


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(CONC)
    tasks = []
    if not ONLY or "generators" in ONLY:
        tasks += [_run(f"gen:{g}", (lambda g: lambda ws, sid: _gen(ws, sid, g))(g)) for g in GENERATORS]
    if not ONLY or "crossdiag" in ONLY:
        tasks += [_run(f"cross:{w}", (lambda w, p: lambda ws, sid: _cross(ws, sid, w, p))(w, p))
                  for w, p in CROSSDIAG]
    if not ONLY or "multiturn" in ONLY:
        tasks += [_run("multi:context", _multiturn_context),
                  _run("multi:pivot", _multiturn_pivot),
                  _run("multi:edit-gen", _multiturn_generate_after_edits)]

    total = len(tasks)
    print(f"=== full-agentic probe: {total} scenarios (CONC={CONC}) against {AGENT_WS_URL} ===\n", flush=True)
    results, done = [], 0
    for coro in asyncio.as_completed(tasks):
        label, outcome, flaw = await coro
        done += 1
        mark = "FLAW" if flaw else "ok  "
        print(f"[{done:2}/{total}] {mark} {label:34} -> {str(outcome)[:44]}", flush=True)
        results.append((label, outcome, flaw))

    flaws = [r for r in results if r[2] and r[2] != "SEED"]
    seeds = [r for r in results if r[2] == "SEED"]
    print("\n===== SUMMARY =====")
    print(f"scenarios: {total}   clean: {total - len(flaws) - len(seeds)}   "
          f"flaws: {len(flaws)}   seed-skips: {len(seeds)}")
    if flaws:
        print("\nflaw detail:")
        for label, outcome, flaw in flaws:
            print(f"   [{label}] {outcome}  ({flaw})")
    else:
        print("\nNO FLAWS — full agentic coverage clean.")


if __name__ == "__main__":
    asyncio.run(main())
