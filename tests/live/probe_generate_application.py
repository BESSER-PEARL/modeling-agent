"""Live probe: where does the "Generate the code" button prompt land?

After the agent builds a class diagram, ``suggestions.py`` offers "Generate
the code", which sends "generate the application". With a usable model that
must reach the Spec-Driven (smart) confirmation. On gpt-6-luna it was routed
4 times in 12 to ``generation_route='deterministic'`` with
``generator_type=null``, which shows the generator menu instead (2026-10-29).
hand_drawn and no_model are reported for comparison; they have no single
expected answer.

Scenarios, each run REPS times:
  post_build  build a library model, then click "Generate the code"
  hand_drawn  a hand-drawn library model in the workspace, no history
  no_model    empty workspace

Run::

  AGENT_WS_URL=ws://localhost:8765 REPS=12 CONC=4 \
      python tests/live/probe_generate_application.py
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _agent_ws import connect as agent_ws_connect  # noqa: E402
from test_nl_generation_scenarios import _unwrap, AGENT_WS_URL  # noqa: E402

REPS = int(os.environ.get("REPS", "12"))
CONC = int(os.environ.get("CONC", "4"))
SCENARIOS = [s for s in os.environ.get("SCENARIOS", "post_build,hand_drawn,no_model").split(",") if s]
TIMEOUT = 240
PROMPT = "generate the application"
TERMINAL = {"inject_complete_system", "modify_model", "trigger_generator",
            "trigger_smart_generator", "assistant_message", "agent_error", "error"}


def _class_model(names):
    return {
        "version": "3.0.0", "type": "ClassDiagram",
        "elements": {f"c{i}": {"id": f"c{i}", "name": n, "type": "Class",
                               "attributes": {}, "methods": {}}
                     for i, n in enumerate(names)},
        "relationships": {},
    }


_LIBRARY = _class_model(["Book", "Author", "Member", "Loan"])


async def _send(ws, sid, text, model=None):
    # The editor sends the project id on every turn; the agent scopes its
    # build record to it.
    context = {"activeDiagramType": "ClassDiagram"}
    snapshot = {"id": "p_" + sid, "name": "Probe", "diagrams": {}}
    if model is not None:
        context["activeModel"] = model
        snapshot["diagrams"] = {"ClassDiagram": [{"model": model}]}
    context["projectSnapshot"] = snapshot
    inner = {"action": "user_message", "protocolVersion": "2.0", "clientMode": "widget",
             "sessionId": sid, "message": text, "context": context}
    await ws.send(json.dumps({"action": "user_message", "user_id": sid, "message": json.dumps(inner)}))


async def _first_terminal(ws):
    buf = ""
    t0 = time.monotonic()
    while time.monotonic() - t0 < TIMEOUT:
        m = _unwrap(await asyncio.wait_for(ws.recv(), timeout=TIMEOUT))
        if not m:
            continue
        a = m.get("action")
        if a == "stream_chunk":
            buf += m.get("chunk") or m.get("content") or ""
        elif a == "stream_done":
            return {"action": "assistant_message", "message": m.get("fullText") or buf,
                    "suggestedActions": m.get("suggestedActions") or []}
        elif a in TERMINAL:
            return m
    return {"action": "timeout"}


def _verdict(frame) -> str:
    a = frame.get("action")
    text = (frame.get("message") or "")
    text = text if isinstance(text, str) else json.dumps(text)
    low = text.lower()
    if a == "trigger_smart_generator" or "besser will generate your application" in low \
            or "update your existing app" in low:
        return "smart"
    if "what would you like me to generate" in low:
        return "menu"
    if a == "trigger_generator":
        return "deterministic"
    if a in ("inject_complete_system", "modify_model"):
        return "modeling"
    return f"other:{a}"


def _spec_class_names(frame):
    spec = frame.get("systemSpec") or frame.get("spec") or {}
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            spec = {}
    return [c.get("className") or c.get("name") for c in spec.get("classes", []) if isinstance(c, dict)]


async def _run(scenario):
    sid = "g_" + uuid.uuid4().hex[:8]
    async with agent_ws_connect(AGENT_WS_URL) as ws:
        model = None
        if scenario == "post_build":
            await _send(ws, sid, "create a class diagram for a library management system")
            built = await _first_terminal(ws)
            if built.get("action") != "inject_complete_system":
                return f"build-failed:{built.get('action')}", 0.0
            model = _class_model([n for n in _spec_class_names(built) if n] or ["Book", "Member"])
            # Drain trailing frames of the build turn (suggestions, follow-ups).
            try:
                while True:
                    await asyncio.wait_for(ws.recv(), timeout=6)
            except asyncio.TimeoutError:
                pass
        elif scenario == "hand_drawn":
            model = _LIBRARY
        t0 = time.monotonic()
        await _send(ws, sid, PROMPT, model=model)
        frame = await _first_terminal(ws)
        return _verdict(frame), time.monotonic() - t0


async def main():
    sem = asyncio.Semaphore(CONC)

    async def one(s):
        async with sem:
            try:
                return s, *(await _run(s))
            except Exception as exc:  # report, keep going
                return s, f"EXC:{type(exc).__name__}", 0.0

    print(f"=== '{PROMPT}' routing: {SCENARIOS} x {REPS} against {AGENT_WS_URL} ===", flush=True)
    results = await asyncio.gather(*(one(s) for s in SCENARIOS for _ in range(REPS)))
    by = collections.defaultdict(collections.Counter)
    lat = collections.defaultdict(list)
    for s, v, dt in results:
        by[s][v] += 1
        if dt:
            lat[s].append(dt)
    for s in SCENARIOS:
        mean = sum(lat[s]) / len(lat[s]) if lat[s] else 0.0
        print(f"{s:11s} {dict(by[s])}  mean reply {mean:.1f}s", flush=True)
    bad = by["post_build"].total() - by["post_build"]["smart"]
    return 1 if "post_build" in SCENARIOS and bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
