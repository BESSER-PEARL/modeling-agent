"""Live NL-generation regression matrix (WebSocket probe against a running agent).

WHY THIS EXISTS
---------------
The modeling-agent routes a natural-language request ("generate a database",
"generate django", "build me a full app", ...) to a generator/route via an
**LLM classifier**. Unit tests pin the *handler/dispatch* logic deterministically
(see ``tests/test_generation_handler.py``), but they can't catch **classifier
drift** — e.g. the real bug where "generate a database" was answered with Django
project questions. This suite drives the *live* agent over its WebSocket with
the exact NL phrasings a user types and asserts each one routes to an acceptable
generator (and NEVER to a forbidden one, e.g. database-y requests must never hit
``django``).

Because the classifier is non-deterministic, each scenario is probed ``REPEATS``
times and must meet a pass threshold; any single hit on a *forbidden* generator
fails the scenario outright.

HOW TO RUN
----------
Standalone (prints a table, exits non-zero on failure — use as a deploy gate)::

    python -m tests.live.test_nl_generation_scenarios
    AGENT_WS_URL=wss://experimental.besser-pearl.org/agent REPEATS=3 \
        python tests/live/test_nl_generation_scenarios.py

As pytest (skipped unless explicitly enabled, since it needs a live agent)::

    RUN_LIVE_AGENT_TESTS=1 python -m pytest tests/live/test_nl_generation_scenarios.py

Env:
  AGENT_WS_URL         default wss://experimental.besser-pearl.org/agent
  REPEATS              probes per scenario (default 2)
  RUN_LIVE_AGENT_TESTS gate for the pytest wrapper (unset => skipped)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

try:
    import websockets
except Exception:  # pragma: no cover - only needed for the live run
    websockets = None

AGENT_WS_URL = os.environ.get(
    "AGENT_WS_URL", "wss://experimental.besser-pearl.org/agent"
)
REPEATS = int(os.environ.get("REPEATS", "2"))
BUILD_TIMEOUT = 180
GEN_TIMEOUT = 120


# --- scenario matrix --------------------------------------------------------
# Each scenario: a natural-language request + the generators/routes that are
# ACCEPTABLE, and any that are explicitly FORBIDDEN (a single hit fails it).
# ``build_model`` seeds a class diagram first so the request has something to
# generate from — mirroring "start from a fixed model, then generate part X".
SCENARIOS = [
    # The reported bug: a bare "database" request must yield an actual database
    # — the deterministic SQL/SQLAlchemy layer OR the Spec-Driven Agent (smart),
    # which also builds one. The ONLY wrong answer is `django`, which asks for
    # Django project settings and produces no database. "database" is genuinely
    # ambiguous, so all three DB-producing routes are accepted; django is the
    # single hard failure.
    {"name": "only_database", "msg": "generate a database",
     "accept": {"sql", "sqlalchemy", "smart"}, "forbid": {"django"}},
    {"name": "the_database", "msg": "generate the database",
     "accept": {"sql", "sqlalchemy", "smart"}, "forbid": {"django"}},
    # Explicit phrasings name their generator — the deterministic one is
    # expected (smart would be over-engineering), but django is still the bug.
    {"name": "sql_schema", "msg": "generate the SQL schema for my model",
     "accept": {"sql", "sqlalchemy"}, "forbid": {"django"}},
    {"name": "sqlalchemy", "msg": "generate the SQLAlchemy models",
     "accept": {"sqlalchemy"}, "forbid": {"django"}},
    # Backend / full-stack.
    {"name": "backend", "msg": "generate the backend",
     "accept": {"backend"}, "forbid": set()},
    {"name": "database_and_backend",
     "msg": "generate the database and the backend",
     "accept": {"backend", "smart", "sql", "sqlalchemy"}, "forbid": set()},
    # Auth + custom UI => smart (per the classifier's "extra behavior" rule).
    # django is a plausible full-web framework so it's not a hard failure here;
    # a bare SQL schema clearly under-delivers a full app, so that IS forbidden.
    {"name": "full_app",
     "msg": "build me a full web app with a UI and user authentication",
     "accept": {"smart"}, "forbid": {"sql", "sqlalchemy"}},
    # Other bare built-ins (each names its generator explicitly).
    {"name": "django", "msg": "generate django",
     "accept": {"django"}, "forbid": set()},
    {"name": "pydantic", "msg": "generate pydantic classes",
     "accept": {"pydantic"}, "forbid": set()},
    # rest_api / backend are DETERMINISTIC BESSER built-ins that emit real,
    # runnable code — a BARE request for one must route deterministically and
    # must NEVER escalate to the smart (spec-driven LLM) path, which is slower,
    # spends the user's key, and can ship non-runnable code.
    {"name": "rest_api_bare", "msg": "generate rest api",
     "accept": {"rest_api", "backend"}, "forbid": {"smart"}},
    {"name": "backend_bare", "msg": "generate the backend",
     "accept": {"backend", "rest_api"}, "forbid": {"smart"}},
]

# Fraction of REPEATS that must land in ``accept`` for a scenario to pass.
PASS_THRESHOLD = 0.5

_SEED_MODEL_REQUEST = (
    "create a class diagram for a todo application with users and tasks"
)


# --- wire protocol (double-JSON, see modeling-agent CLAUDE.md) --------------
def _unwrap(raw: str):
    try:
        inner = json.loads(raw)
    except Exception:
        return None
    for _ in range(3):
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            try:
                nxt = json.loads(inner["message"])
            except Exception:
                break
            if isinstance(nxt, dict) and "action" in nxt:
                inner = nxt
                continue
        break
    return inner if isinstance(inner, dict) else None


async def _send(ws, sid, text):
    inner = {
        "action": "user_message", "protocolVersion": "2.0",
        "clientMode": "widget", "sessionId": sid, "message": text,
        "context": {"activeDiagramType": "ClassDiagram"},
    }
    await ws.send(json.dumps(
        {"action": "user_message", "user_id": sid, "message": json.dumps(inner)}
    ))


async def _wait_meaningful(ws, timeout):
    """Return the first frame that reveals the routing decision:
    trigger_generator / trigger_smart_generator / inject_complete_system /
    a config-prompt assistant_message / an error."""
    import time
    buf = ""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        m = _unwrap(raw)
        if not m:
            continue
        act = m.get("action")
        if act == "stream_chunk":
            # The live wire uses "chunk" (session_helpers.reply_stream_chunk);
            # content/delta kept only as defensive fallbacks.
            buf += (m.get("chunk") or m.get("content") or m.get("delta") or "")
            continue
        if act == "stream_done":
            # stream_done carries the assembled text in "fullText".
            full = m.get("fullText") or buf
            if full.strip():
                return {"action": "assistant_message", "message": full,
                        "suggestedActions": m.get("suggestedActions")}
            continue
        if act in ("trigger_generator", "trigger_smart_generator",
                   "inject_complete_system", "modify_model", "auto_generate_gui",
                   "assistant_message", "agent_error", "error"):
            return m
    return {"action": "timeout", "message": ""}


def detect_generator(reply: dict) -> str:
    """Infer which generator/route the agent chose from its reply."""
    act = reply.get("action")
    if act == "trigger_generator":
        return (reply.get("generatorType") or "trigger:?").lower()
    if act == "trigger_smart_generator":
        return "smart"
    if act == "inject_complete_system":
        return "modeling"
    text = (reply.get("message") or "").lower()
    # Smart-gen confirmation. The current copy no longer says "ready to run";
    # it describes generating the application from the specification with
    # BESSER's built-in generators (+ an LLM for gaps).
    if (
        "spec-driven" in text
        or "ready to run" in text
        or "from the specification" in text
        or "built-in generators" in text
    ):
        return "smart"
    # Config-collection prompts (GENERATOR_REQUIRED_FIELDS).
    if "django project" in text or ("project name" in text and "django" in text):
        return "django"
    if "dbms" in text or "sqlalchemy" in text:
        return "sqlalchemy"
    if "dialect" in text or "sql schema" in text:
        return "sql"
    if "which framework" in text or "backend framework" in text:
        return "backend"
    # Generators with NO config prompt (backend, pydantic, python, java, …)
    # trigger immediately; if this probe didn't carry a model in the request
    # context they answer "your workspace looks empty — **backend** generation
    # …". The named generator in that message still reveals the routing, which
    # is what this matrix checks. Check longer names before their substrings
    # (sqlalchemy before sql).
    if "empty" in text and ("generat" in text or "model" in text):
        for gen in ("sqlalchemy", "jsonschema", "smartdata", "backend",
                    "pydantic", "django", "python", "java", "sql"):
            if gen in text:
                return gen
    if act in ("agent_error", "error"):
        return "error:" + text[:40]
    if act == "timeout":
        return "timeout"
    return "ambiguous:" + text[:50]


async def _probe_once(scenario) -> str:
    sid = f"nlprobe_{uuid.uuid4().hex[:8]}"
    async with websockets.connect(AGENT_WS_URL, max_size=None,
                                  ping_interval=20) as ws:
        # 1. Seed a model so the generation request has something to act on.
        await _send(ws, sid, _SEED_MODEL_REQUEST)
        await _wait_meaningful(ws, BUILD_TIMEOUT)
        # 2. The actual NL generation request under test.
        await _send(ws, sid, scenario["msg"])
        reply = await _wait_meaningful(ws, GEN_TIMEOUT)
        return detect_generator(reply)


async def run_scenario(scenario) -> dict:
    detected = []
    for _ in range(REPEATS):
        try:
            detected.append(await _probe_once(scenario))
        except Exception as e:  # noqa: BLE001 - report, don't crash the matrix
            detected.append(f"exc:{type(e).__name__}")
    accept, forbid = scenario["accept"], scenario["forbid"]
    hits = sum(1 for d in detected if d in accept)
    forbidden_hit = [d for d in detected if d in forbid]
    passed = (not forbidden_hit) and (hits / max(1, REPEATS) >= PASS_THRESHOLD)
    return {"name": scenario["name"], "msg": scenario["msg"],
            "detected": detected, "accept": sorted(accept),
            "forbid": sorted(forbid), "forbidden_hit": forbidden_hit,
            "passed": passed}


async def run_matrix() -> list[dict]:
    return [await run_scenario(s) for s in SCENARIOS]


def _print_report(results) -> bool:
    print(f"\nNL-generation matrix @ {AGENT_WS_URL}  (REPEATS={REPEATS})")
    print("=" * 78)
    all_ok = True
    for r in results:
        ok = r["passed"]
        all_ok = all_ok and ok
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {r['name']:<22} {r['msg']!r}")
        print(f"        detected={r['detected']} accept={r['accept']} "
              f"forbid={r['forbid']}")
        if r["forbidden_hit"]:
            print(f"        !! hit FORBIDDEN generator(s): {r['forbidden_hit']}")
    print("=" * 78)
    print("RESULT:", "ALL PASSED" if all_ok else "FAILURES ABOVE")
    return all_ok


def main() -> int:
    if websockets is None:
        print("the 'websockets' package is required to run the live probe")
        return 2
    results = asyncio.run(run_matrix())
    return 0 if _print_report(results) else 1


# --- pytest wrapper (skipped unless RUN_LIVE_AGENT_TESTS=1) -----------------
def test_nl_generation_matrix():
    import pytest
    if not os.environ.get("RUN_LIVE_AGENT_TESTS"):
        pytest.skip("live agent test — set RUN_LIVE_AGENT_TESTS=1 to run")
    if websockets is None:
        pytest.skip("websockets package not installed")
    results = asyncio.run(run_matrix())
    failures = [r for r in results if not r["passed"]]
    assert not failures, "NL-generation scenarios failed:\n" + "\n".join(
        f"  {r['name']}: detected={r['detected']} "
        f"accept={r['accept']} forbidden_hit={r['forbidden_hit']}"
        for r in failures
    )


if __name__ == "__main__":
    sys.exit(main())
