"""Live release sweep against the deployed WME agent.

Runs many REAL generation scenarios against wss://experimental.besser-pearl.org/agent
and checks each result for the flaws users actually complain about:

  - empty / hung / non-generation reply
  - too few classes  (a thin/broken model)
  - ZERO relationships (isolated classes — the #1 "useless model" complaint)
  - duplicate association names  (the bug we just fixed — verify at scale)
  - duplicate class names
  - dangling relationship endpoints (source/target not a real class)
  - enum used as a relationship endpoint
  - web-app flow AUTO-generating instead of pausing (the pause fix — verify at scale)

Concurrency is capped low to avoid overloading the single live agent.

Usage:
  AGENT_WS_URL=wss://experimental.besser-pearl.org/agent REPEAT=1 CONC=3 \
      python tests/live/wme_release_sweep.py
"""
import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AGENT_WS_URL", "wss://experimental.besser-pearl.org/agent")

import websockets  # noqa: E402
from test_nl_generation_scenarios import _send, _wait_meaningful, AGENT_WS_URL  # noqa: E402

REPEAT = int(os.environ.get("REPEAT", "1"))
CONC = int(os.environ.get("CONC", "3"))
GEN_TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "150"))

# Diverse domains for complete-system (class diagram) generation.
SYSTEM_DOMAINS = [
    "a hospital management system",
    "a library with books, members and loans",
    "an e-commerce store with products, orders and payments",
    "a university with students, courses and enrollments",
    "a bank with accounts, customers and transactions",
    "a restaurant with menus, orders and tables",
    "a hotel booking system",
    "a blog with posts, authors and comments",
    "a warehouse inventory system",
    "a social network with users, posts and follows",
    "a task management system with users, projects and tasks",
    "a fleet management system with vehicles, drivers and trips",
    "a gym with members, trainers and sessions",
    "an airline with flights, passengers and bookings",
    "a movie streaming service with users, movies and subscriptions",
    "a project tracker with teams, issues and sprints",
    "a clinic with patients, doctors and appointments",
    "a music app with artists, albums and playlists",
]

# Web-app flow: must PAUSE (defer) after the GUI, not auto-generate.
WEBAPP_DOMAINS = [
    "inventory management",
    "event booking",
    "a recipe sharing app",
    "customer support tickets",
    "a real estate listing",
]

# Other diagram types — routing + non-empty generation.
OTHER_SCENARIOS = [
    ("StateMachine", "create a state machine for a traffic light"),
    ("StateMachine", "create a state machine for an order lifecycle"),
    ("BPMN", "create a bpmn process for employee onboarding"),
    ("Object", "create an object diagram for a sample library"),
    ("Agent", "create an agent that answers customer FAQs"),
]


def _flaws_for_system(spec: dict) -> list:
    """Return a list of flaw strings for a generated complete-system spec."""
    flaws = []
    classes = spec.get("classes") or []
    rels = spec.get("relationships") or []
    class_names = [c.get("className") for c in classes if isinstance(c, dict) and c.get("className")]
    enum_names = {
        c.get("className")
        for c in classes
        if isinstance(c, dict) and c.get("isEnumeration") and c.get("className")
    }
    real_class_names = set(class_names) - enum_names

    if len(classes) < 3:
        flaws.append(f"thin-model({len(classes)} classes)")
    if len(rels) == 0:
        flaws.append("no-relationships")
    # duplicate class names
    lc = [n.lower() for n in class_names]
    if len(lc) != len(set(lc)):
        dupes = sorted({n for n in lc if lc.count(n) > 1})
        flaws.append(f"dup-class-names:{dupes}")
    # duplicate association names
    rel_names = [r.get("name") for r in rels if isinstance(r, dict) and r.get("name")]
    lr = [n.lower() for n in rel_names]
    if len(lr) != len(set(lr)):
        dupes = sorted({n for n in lr if lr.count(n) > 1})
        flaws.append(f"DUP-ASSOC-NAMES:{dupes}")
    # dangling endpoints + enum endpoints
    for r in rels:
        if not isinstance(r, dict):
            continue
        s, t = r.get("source"), r.get("target")
        if s not in real_class_names or t not in real_class_names:
            if s in enum_names or t in enum_names:
                flaws.append(f"enum-endpoint({s}->{t})")
            else:
                flaws.append(f"dangling-endpoint({s}->{t})")
    return flaws


async def _run_system(sem, label, prompt):
    async with sem:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                sid = "sweep_" + uuid.uuid4().hex[:6]
                await _send(ws, sid, f"create {prompt}")
                reply = await _wait_meaningful(ws, GEN_TIMEOUT)
                data = reply if isinstance(reply, dict) else {}
                action = data.get("action")
                spec = data.get("systemSpec") or {}
                if action != "inject_complete_system" or not spec.get("classes"):
                    return (label, ["no-generation(action=%s)" % action])
                return (label, _flaws_for_system(spec))
        except Exception as exc:
            return (label, [f"EXC:{type(exc).__name__}"])


async def _run_webapp(sem, label, domain):
    """Verify the web-app flow PAUSES (defers) instead of auto-generating.

    Handles BOTH paths: the GUI-choice path (prompts "generate the GUI?" → we
    answer "auto" → defer) AND the fall-through path (slow model → GUI
    auto-built, no prompt → defer directly). Either way the flaw is only
    AUTO-RAN-GENERATION; a clean defer via any path is a pass.
    """
    async with sem:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                sid = "sweepwa_" + uuid.uuid4().hex[:6]
                await _send(ws, sid, f"create a web app for {domain}")
                deferred = auto_ran = answered = False
                for _ in range(8):
                    try:
                        r = await _wait_meaningful(ws, GEN_TIMEOUT)
                    except Exception:
                        break
                    low = str(r).lower()
                    act = r.get("action") if isinstance(r, dict) else ""
                    if (act == "trigger_generator" or "starting web_app" in low
                            or "web_app code generation" in low or "downloadurl" in low):
                        auto_ran = True
                        break
                    if "generate the web app" in low:
                        deferred = True
                        break
                    if "generate the gui" in low and not answered:
                        await _send(ws, sid, "auto")
                        answered = True
                flaws = []
                if auto_ran:
                    flaws.append("AUTO-RAN-GENERATION")
                elif not deferred:
                    flaws.append("no-defer")
                return (label, flaws)
        except Exception as exc:
            return (label, [f"EXC:{type(exc).__name__}"])


async def _run_other(sem, label, prompt, expect_hint):
    async with sem:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                sid = "sweepo_" + uuid.uuid4().hex[:6]
                await _send(ws, sid, prompt)
                reply = await _wait_meaningful(ws, GEN_TIMEOUT)
                s = str(reply).lower()
                # A valid generation for any diagram type produces an inject/action payload.
                if "inject" in s or "action" in s and ("system" in s or "diagram" in s or "spec" in s):
                    return (label, [])
                return (label, ["no-generation"])
        except Exception as exc:
            return (label, [f"EXC:{type(exc).__name__}"])


async def main():
    only = os.environ.get("ONLY", "").strip().lower()  # "", "system", "webapp", "other"
    sem = asyncio.Semaphore(CONC)
    tasks = []
    for _rep in range(REPEAT):
        if only in ("", "system"):
            for d in SYSTEM_DOMAINS:
                tasks.append(_run_system(sem, f"system:{d[:28]}", d))
        if only in ("", "webapp"):
            for d in WEBAPP_DOMAINS:
                tasks.append(_run_webapp(sem, f"webapp:{d[:22]}", d))
        if only in ("", "other"):
            for label, prompt in OTHER_SCENARIOS:
                tasks.append(_run_other(sem, f"other:{label}", prompt, label))

    total = len(tasks)
    print(f"=== WME live release sweep: {total} scenarios "
          f"(REPEAT={REPEAT}, CONC={CONC}) against {AGENT_WS_URL} ===\n", flush=True)

    results = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        label, flaws = await coro
        done += 1
        status = "OK  " if not flaws else "FLAW"
        if flaws:
            print(f"[{done:3}/{total}] {status} {label}  -> {', '.join(flaws)}", flush=True)
        else:
            print(f"[{done:3}/{total}] {status} {label}", flush=True)
        results.append((label, flaws))

    flawed = [(l, f) for l, f in results if f]
    print("\n===== SUMMARY =====")
    print(f"scenarios: {total}   clean: {total - len(flawed)}   with-flaws: {len(flawed)}")
    # tally flaw categories
    from collections import Counter
    cats = Counter()
    for _l, fs in flawed:
        for f in fs:
            cats[f.split("(")[0].split(":")[0]] += 1
    if cats:
        print("flaw categories:")
        for cat, n in cats.most_common():
            print(f"   {n:3}x  {cat}")
    else:
        print("NO FLAWS — clean sweep.")


if __name__ == "__main__":
    asyncio.run(main())
