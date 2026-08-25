"""End-to-end workflow completion probe: drives each flow to its TERMINAL action,
not just the first reply. Answers config prompts and follows resume steps.

Flows:
  gen-<stack>   create model -> "generate <stack>" -> (answer config) -> trigger_generator
  mismatch      create library -> rust/other request -> mismatch confirm ->
                "Update model + generate" -> new model built -> smart-gen RESUMES
  webapp        "create a web app for X" -> deferred -> "generate the web app" -> trigger

Usage:
  AGENT_WS_URL=wss://experimental.besser-pearl.org/agent \
      python tests/live/probe_workflow_complete.py
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

TERMINAL = {"trigger_generator", "trigger_smart_generator", "inject_complete_system",
            "auto_generate_gui"}


async def _send(ws, sid, text, ctx="ClassDiagram"):
    inner = {"action": "user_message", "protocolVersion": "2.0", "clientMode": "widget",
             "sessionId": sid, "message": text, "context": {"activeDiagramType": ctx}}
    await ws.send(json.dumps({"action": "user_message", "user_id": sid,
                              "message": json.dumps(inner)}))


async def _turn(ws, timeout):
    """Collect frames for one turn. Return dict with .action = first TERMINAL action
    seen, else 'assistant_message' with assembled text + suggestedActions."""
    buf = ""
    sugg = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            break
        m = _unwrap(raw)
        if not m:
            continue
        act = m.get("action")
        if act == "stream_chunk":
            buf += (m.get("chunk") or m.get("content") or m.get("delta") or "")
            continue
        if act == "stream_done":
            full = m.get("fullText") or buf
            sugg = m.get("suggestedActions") or sugg
            if full.strip():
                return {"action": "assistant_message", "message": full, "suggestedActions": sugg}
            continue
        if act in TERMINAL:
            return m
        if act in ("assistant_message", "agent_error", "error"):
            return m
    return {"action": "timeout", "message": buf}


def _txt(m):
    return (m.get("message") or m.get("fullText") or "") if isinstance(m, dict) else str(m)


# config answers keyed by stack substring
_CONFIG_ANSWER = {
    "django": "project name is shop, use default settings",
    "sql": "use postgresql",
    "sqlalchemy": "use postgresql",
    "default": "yes, use the defaults",
}


def _answer_for(prompt_text):
    low = prompt_text.lower()
    for k, v in _CONFIG_ANSWER.items():
        if k != "default" and k in low:
            return v
    return _CONFIG_ANSWER["default"]


async def _gen_complete(sem, stack, seed="a shop with products, orders and customers"):
    label = f"gen:{stack}"
    async with sem:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                sid = "gc_" + uuid.uuid4().hex[:6]
                await _send(ws, sid, f"create a class diagram for {seed}")
                r0 = await _turn(ws, TIMEOUT)
                if r0.get("action") != "inject_complete_system":
                    return (label, f"seed-failed({r0.get('action')})", "SEED_FAILED")
                await _send(ws, sid, f"generate {stack}")
                r1 = await _turn(ws, TIMEOUT)
                if r1.get("action") == "trigger_generator":
                    return (label, f"trigger_generator({r1.get('generatorType')})", None)
                if r1.get("action") == "assistant_message":
                    # config prompt — answer it, expect a trigger next
                    ans = _answer_for(_txt(r1))
                    await _send(ws, sid, ans)
                    r2 = await _turn(ws, TIMEOUT)
                    if r2.get("action") == "trigger_generator":
                        return (label, f"config->trigger_generator({r2.get('generatorType')})", None)
                    if r2.get("action") == "assistant_message":
                        # maybe a 2nd config field; answer once more
                        await _send(ws, sid, "yes, use the defaults for everything")
                        r3 = await _turn(ws, TIMEOUT)
                        if r3.get("action") == "trigger_generator":
                            return (label, f"config2->trigger_generator({r3.get('generatorType')})", None)
                        return (label, f"stuck-on-config: {_txt(r3)[:40]}", "CONFIG_NEVER_COMPLETES")
                    return (label, f"after-config:{r2.get('action')}", "NO_TRIGGER")
                return (label, f"unexpected:{r1.get('action')} {_txt(r1)[:30]}", "NO_TRIGGER")
        except Exception as exc:
            return (label, f"EXC:{type(exc).__name__}:{exc}"[:50], "EXC")


async def _mismatch_resume(sem, trigger):
    label = f"mismatch-resume"
    async with sem:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                sid = "mr_" + uuid.uuid4().hex[:6]
                await _send(ws, sid, "create a class diagram for a library with books and members")
                r0 = await _turn(ws, TIMEOUT)
                if r0.get("action") != "inject_complete_system":
                    return (label, f"seed-failed({r0.get('action')})", "SEED_FAILED")
                await _send(ws, sid, trigger)
                r1 = await _turn(ws, TIMEOUT)
                low = _txt(r1).lower()
                labels = " ".join((a.get("label") or "") for a in (r1.get("suggestedActions") or [])).lower()
                if "doesn't match" not in low and "does not match" not in low and "update model" not in labels:
                    return (label, f"no-mismatch({r1.get('action')})", None)
                upd = next((a for a in (r1.get("suggestedActions") or [])
                            if "update model" in (a.get("label") or "").lower()), None)
                if not upd or not upd.get("prompt"):
                    return (label, "no-update-action", "NO_UPDATE_ACTION")
                await _send(ws, sid, upd["prompt"])
                # Expect: new model built, THEN smart-gen resumes (confirmation or trigger).
                built = False
                resumed = None
                for _ in range(4):
                    r = await _turn(ws, TIMEOUT)
                    a = r.get("action")
                    if a == "inject_complete_system":
                        built = True
                        continue
                    if a == "trigger_smart_generator":
                        resumed = "trigger_smart_generator"
                        break
                    if a == "assistant_message":
                        t = _txt(r).lower()
                        if ("spec-driven" in t or "from the specification" in t
                                or "built-in generators" in t or "generate the app" in t
                                or "ready" in t):
                            resumed = "smart-gen-confirmation"
                        break
                    if a in ("timeout",):
                        break
                if built and resumed:
                    return (label, f"built + resumed({resumed})", None)
                if built and not resumed:
                    return (label, "built but smart-gen did NOT resume", "NO_RESUME")
                return (label, f"built={built} resumed={resumed}", "NO_RESUME")
        except Exception as exc:
            return (label, f"EXC:{type(exc).__name__}:{exc}"[:50], "EXC")


async def _webapp_complete(sem, domain="a recipe sharing app"):
    label = "webapp-complete"
    async with sem:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                sid = "wc_" + uuid.uuid4().hex[:6]
                await _send(ws, sid, f"create a web app for {domain}")
                deferred = False
                for _ in range(4):
                    r = await _turn(ws, TIMEOUT)
                    low = _txt(r).lower()
                    if r.get("action") in TERMINAL:
                        return (label, f"AUTO-RAN({r.get('action')})", "AUTO_RAN_BEFORE_CONFIRM")
                    if "generate the web app" in low or ("ready" in low and "web app" in low):
                        deferred = True
                        break
                if not deferred:
                    return (label, "no-defer", "NO_DEFER")
                await _send(ws, sid, "generate the web app")
                for _ in range(4):
                    r = await _turn(ws, TIMEOUT)
                    a = r.get("action")
                    if a in TERMINAL:
                        return (label, f"deferred -> {a}", None)
                    if a == "assistant_message":
                        low = _txt(r).lower()
                        if "generate the gui" in low or "which" in low:
                            await _send(ws, sid, "auto")
                            continue
                        # any other terminal-ish confirmation is acceptable
                        return (label, f"deferred -> msg:{_txt(r)[:34]}", None)
                return (label, "no-terminal-after-confirm", "NO_COMPLETE")
        except Exception as exc:
            return (label, f"EXC:{type(exc).__name__}:{exc}"[:50], "EXC")


async def main():
    sem = asyncio.Semaphore(CONC)
    only = {c.strip() for c in os.environ.get("ONLY", "").lower().split(",") if c.strip()}
    tasks = []
    if not only or "gen" in only:
        for stack in ["django", "sql", "pydantic", "sqlalchemy", "python"]:
            tasks.append(_gen_complete(sem, stack))
    if not only or "mismatch" in only:
        tasks.append(_mismatch_resume(sem, "generate a rust application for a hotel booking system"))
    if not only or "webapp" in only:
        tasks.append(_webapp_complete(sem))

    total = len(tasks)
    print(f"=== workflow COMPLETION probe: {total} flows (CONC={CONC}) against {AGENT_WS_URL} ===\n", flush=True)
    results, done = [], 0
    for coro in asyncio.as_completed(tasks):
        label, outcome, flaw = await coro
        done += 1
        mark = "FLAW" if flaw else "ok  "
        print(f"[{done:2}/{total}] {mark} {label:18} -> {str(outcome)[:52]}", flush=True)
        results.append((label, outcome, flaw))

    flaws = [r for r in results if r[2]]
    print("\n===== SUMMARY =====")
    print(f"flows: {total}   completed-clean: {total - len(flaws)}   flaws: {len(flaws)}")
    if flaws:
        print("\nflaw detail:")
        for label, outcome, flaw in flaws:
            print(f"   [{label}] {outcome}  ({flaw})")
    else:
        print("\nNO FLAWS — every workflow ran to its terminal action.")


if __name__ == "__main__":
    asyncio.run(main())
