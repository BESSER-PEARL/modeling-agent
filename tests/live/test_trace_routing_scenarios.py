"""Live probe: the describe_model / traceability_question tie-break, with and without a trace digest.

WHY THIS EXISTS
---------------
Most requests carry no ``context.traceDigest`` (recording off, an older
editor, or the trace not loaded yet). Without one, a design-rationale question
about the model ("why does Book have an Author association?") must still get a
real answer from the describe-model body, and only an EXPLICIT trace question
("which code was generated from Book?") gets the fixed "no trace record" reply.
With a digest, rationale and provenance questions are answered from the trace,
and a question about what the model contains stays describe_model. The rule
lives in the classifier prompt (``prompts/classifier.py``) with a deterministic
net in ``routing/guards.trace_question_guard``; unit tests pin both, but only a
live run shows what the classifier actually does.

Each scenario is probed ``REPEATS`` times; a single forbidden outcome fails it.

HOW TO RUN
----------
Standalone::

    AGENT_WS_URL=ws://localhost:8765 REPEATS=3 python tests/live/test_trace_routing_scenarios.py

As pytest (skipped unless explicitly enabled, since it needs a live agent)::

    RUN_LIVE_AGENT_TESTS=1 AGENT_WS_URL=ws://localhost:8765 \\
        python -m pytest tests/live/test_trace_routing_scenarios.py

Env:
  AGENT_WS_URL         required, e.g. ws://localhost:8765
  REPEATS              probes per scenario (default 2)
  RUN_LIVE_AGENT_TESTS gate for the pytest wrapper (unset => skipped)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

try:
    import websockets
    from _agent_ws import connect as agent_ws_connect
    from test_nl_generation_scenarios import _unwrap
except Exception:  # pragma: no cover - only needed for the live run
    websockets = None
    agent_ws_connect = None
    _unwrap = None

AGENT_WS_URL = os.environ.get("AGENT_WS_URL", "")
REPEATS = int(os.environ.get("REPEATS", "2"))
REPLY_TIMEOUT = 90
# How long to wait after stream_done for the show_trace button that follows a trace answer.
BUTTON_GRACE = 3.0


def _cls(cid, name, attributes=()):
    return {"id": cid, "type": "Class", "name": name, "attributes": [
        {"id": f"{cid}-{a}", "name": a, "attributeType": "str"} for a in attributes]}


MODEL = {
    "elements": {
        "c-book": _cls("c-book", "Book", ("isbn", "title")),
        "c-author": _cls("c-author", "Author", ("name",)),
        "c-loan": _cls("c-loan", "Loan", ("due",)),
        "c-member": _cls("c-member", "Member", ("name",)),
    },
    "relationships": {
        "r1": {"id": "r1", "type": "ClassBidirectional", "source": {"element": "c-book"},
               "target": {"element": "c-author"}, "name": "writtenBy"},
        "r2": {"id": "r2", "type": "ClassBidirectional", "source": {"element": "c-loan"},
               "target": {"element": "c-book"}, "name": "lends"},
    },
}


def _focus(artifact_id, name):
    return {
        "artifactId": artifact_id, "name": name, "elementType": "Class",
        "history": [{"kind": "CREATED", "at": "2026-10-30T09:00:00Z", "by": "Agent:BESSER assistant",
                     "via": "modeling-assistant"}],
        "derivedFrom": [{"type": "Requirement", "name": f"R1: members borrow books ({name})",
                         "evidence": "AIEvidence: requirement extraction"}],
        "derived": [{"path": "sql_alchemy.py", "lines": "40-72"}],
    }


def _digest(*focus):
    digest = {
        "schemaVersion": 1, "recording": True,
        "counts": {"artifacts": 12, "links": 9, "outOfDateLinks": 0,
                   "operations": {"ModelingSession": 2, "CodeGeneration": 1, "RequirementExtraction": 1,
                                  "Conversation": 3}},
        "requirements": [{"id": "R1", "kind": "data", "text": "Members borrow books", "covered": True,
                          "realizedBy": ["Loan", "Book"]}],
    }
    if focus:
        digest["focus"] = list(focus)
    return digest


# outcome: "trace" (a trace answer, followed by its show_trace button),
# "no_trace" (the fixed "no trace record" reply), "describe" (a streamed
# answer with no button: the describe-model body).
SCENARIOS = [
    # Without a digest: rationale stays a real answer, explicit trace gets the fixed reply.
    {"name": "rationale_no_digest", "msg": "why does Book have an Author association?", "digest": None,
     "accept": {"describe"}, "forbid": {"no_trace"}},
    {"name": "exists_no_digest", "msg": "why does the Loan class exist?", "digest": None,
     "accept": {"describe"}, "forbid": {"no_trace"}},
    {"name": "code_no_digest", "msg": "which code was generated from Book?", "digest": None,
     "accept": {"no_trace"}, "forbid": set()},
    {"name": "stale_no_digest", "msg": "what is out of date since the last generation?", "digest": None,
     "accept": {"no_trace"}, "forbid": set()},
    # With a digest naming the element: rationale and provenance come from the trace.
    {"name": "exists_with_digest", "msg": "why does the Loan class exist?",
     "digest": _digest(_focus("a-loan", "Loan")), "accept": {"trace"}, "forbid": {"no_trace"}},
    {"name": "code_with_digest", "msg": "which code was generated from Book?",
     "digest": _digest(_focus("a-book", "Book")), "accept": {"trace"}, "forbid": {"no_trace"}},
    # What the model contains stays describe_model, record or not.
    {"name": "contents_with_digest", "msg": "what classes do I have?", "digest": _digest(),
     "accept": {"describe"}, "forbid": {"trace", "no_trace"}},
]

PASS_THRESHOLD = 0.5


async def _send(ws, sid, text, digest):
    context = {"activeDiagramType": "ClassDiagram",
               "projectSnapshot": {"name": "Library", "diagrams": {"ClassDiagram": [{"model": MODEL}]}}}
    if digest is not None:
        context["traceDigest"] = digest
    inner = {"action": "user_message", "protocolVersion": "2.0", "clientMode": "widget",
             "sessionId": sid, "message": text, "context": context}
    await ws.send(json.dumps({"action": "user_message", "user_id": sid, "message": json.dumps(inner)}))


async def _outcome(ws) -> str:
    """Read the turn's frames and name its outcome."""
    deadline = time.monotonic() + REPLY_TIMEOUT
    answered = False
    while True:
        timeout = (BUTTON_GRACE if answered else deadline - time.monotonic())
        if timeout <= 0:
            return "timeout"
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return "describe" if answered else "timeout"
        frame = _unwrap(raw)
        if not frame:
            continue
        action = frame.get("action")
        if action == "show_trace":
            return "trace"
        if action == "stream_done":
            answered = True
            continue
        if action == "assistant_message":
            text = frame.get("message") or ""
            if "trace record" in text and "Traceability" in text:
                return "no_trace"
            return "other:" + text[:40]
        if action in ("agent_error", "error"):
            return "error:" + (frame.get("message") or "")[:40]


async def _probe_once(scenario) -> str:
    sid = f"traceprobe_{uuid.uuid4().hex[:8]}"
    async with agent_ws_connect(AGENT_WS_URL) as ws:
        await _send(ws, sid, scenario["msg"], scenario["digest"])
        return await _outcome(ws)


async def run_scenario(scenario) -> dict:
    detected = []
    for _ in range(REPEATS):
        try:
            detected.append(await _probe_once(scenario))
        except Exception as e:  # noqa: BLE001 - report, don't crash the matrix
            detected.append(f"exc:{type(e).__name__}")
    hits = sum(1 for d in detected if d in scenario["accept"])
    forbidden_hit = [d for d in detected if d in scenario["forbid"]]
    passed = not forbidden_hit and hits / max(1, REPEATS) >= PASS_THRESHOLD
    return {"name": scenario["name"], "msg": scenario["msg"], "detected": detected,
            "accept": sorted(scenario["accept"]), "forbidden_hit": forbidden_hit, "passed": passed}


async def run_matrix() -> list[dict]:
    return [await run_scenario(s) for s in SCENARIOS]


def _print_report(results) -> bool:
    print(f"\nTrace-routing matrix @ {AGENT_WS_URL}  (REPEATS={REPEATS})")
    print("=" * 78)
    for r in results:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['name']:<22} {r['msg']!r}")
        print(f"        detected={r['detected']} accept={r['accept']} forbidden_hit={r['forbidden_hit']}")
    print("=" * 78)
    ok = all(r["passed"] for r in results)
    print("RESULT:", "ALL PASSED" if ok else "FAILURES ABOVE")
    return ok


def main() -> int:
    if websockets is None:
        print("the 'websockets' package is required to run the live probe")
        return 2
    return 0 if _print_report(asyncio.run(run_matrix())) else 1


# --- pytest wrapper (skipped unless RUN_LIVE_AGENT_TESTS=1) -----------------
def test_trace_routing_matrix():
    import pytest
    if not os.environ.get("RUN_LIVE_AGENT_TESTS"):
        pytest.skip("live agent test — set RUN_LIVE_AGENT_TESTS=1 to run")
    if not AGENT_WS_URL:
        pytest.skip("live agent test — set AGENT_WS_URL to the agent's WebSocket URL")
    if websockets is None:
        pytest.skip("websockets package not installed")
    failures = [r for r in asyncio.run(run_matrix()) if not r["passed"]]
    assert not failures, "trace-routing scenarios failed:\n" + "\n".join(
        f"  {r['name']}: detected={r['detected']} accept={r['accept']} forbidden_hit={r['forbidden_hit']}"
        for r in failures
    )


if __name__ == "__main__":
    sys.exit(main())
