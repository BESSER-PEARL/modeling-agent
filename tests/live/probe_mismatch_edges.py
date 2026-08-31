"""Adversarial edge cases for the domain-mismatch "Update model + generate"
chain and the MISMATCH_REGEN_PENDING one-shot flag (commit c3b42a7).

Each scenario runs on its own WS session and drives multiple turns, collecting
ALL frames per turn so we can see a resume confirmation that follows an inject.

Scenarios:
  full_chain        rebuild → resume confirmation → confirm → trigger_smart_generator
  generate_anyway   "Generate anyway" generates on the CURRENT model, no rebuild
  cancel_then_create Cancel clears the flag; a later create does NOT resume
  unrelated_create  a DIFFERENT create right after a mismatch (flag armed) —
                    does it spuriously resume the old domain's smart-gen?
  decline_interrupt "never mind" at the confirmation cancels, does not build

Usage:
  AGENT_WS_URL=wss://experimental.besser-pearl.org/agent \
      python tests/live/probe_mismatch_edges.py
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

CONC = int(os.environ.get("CONC", "3"))
TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "180"))
TERMINAL = {"trigger_generator", "trigger_smart_generator", "inject_complete_system",
            "auto_generate_gui"}


async def _send(ws, sid, text):
    inner = {"action": "user_message", "protocolVersion": "2.0", "clientMode": "widget",
             "sessionId": sid, "message": text, "context": {"activeDiagramType": "ClassDiagram"}}
    await ws.send(json.dumps({"action": "user_message", "user_id": sid,
                              "message": json.dumps(inner)}))


async def _collect(ws, quiet=7, hard=None):
    """Collect ALL frames for a turn until `quiet` seconds pass with no new
    frame (or `hard` total seconds). Returns list of (action, text, [labels])."""
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
                        [x.get("label") for x in (m.get("suggestedActions") or [])]))
            buf = ""
            continue
        if a in TERMINAL or a in ("assistant_message", "agent_error", "error"):
            out.append((a, (m.get("message") or ""),
                        [x.get("label") for x in (m.get("suggestedActions") or [])]))
    return out


def _has_action(frames, action):
    return any(f[0] == action for f in frames)


def _is_resume(frames):
    """A smart-gen resume confirmation ('Model rebuilt and ready …')."""
    for a, t, _labels in frames:
        low = (t or "").lower()
        if a == "assistant_message" and (
            "model rebuilt and ready" in low
            or "from the specification" in low
            or "from your model" in low
            or "built-in generators" in low
        ):
            return True
    return False


def _is_mismatch(frames):
    for a, t, labels in frames:
        low = (t or "").lower()
        lab = " ".join(labels or []).lower()
        if "doesn't match" in low or "does not match" in low or "update model" in lab:
            return True
    return False


def _built(frames):
    return _has_action(frames, "inject_complete_system")


SEED = "create a class diagram for a library with books and members"
MISMATCH_TRIGGER = "generate a rust application for a hotel booking system"


async def _seed_and_mismatch(ws, sid):
    await _send(ws, sid, SEED)
    r0 = await _collect(ws)
    if not _built(r0):
        return None, "seed-failed"
    await _send(ws, sid, MISMATCH_TRIGGER)
    r1 = await _collect(ws)
    if not _is_mismatch(r1):
        return None, "no-mismatch"
    return r1, None


_SEM = None


async def _run(label, fn):
    async with _SEM:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                sid = label[:3] + "_" + uuid.uuid4().hex[:6]
                return await fn(ws, sid)
        except asyncio.TimeoutError:
            return (label, "HANG", "HANG")
        except Exception as exc:
            return (label, f"EXC:{type(exc).__name__}", "EXC")


# ── Scenarios ───────────────────────────────────────────────────────────────
async def full_chain(ws, sid):
    r1, err = await _seed_and_mismatch(ws, sid)
    if err:
        return ("full_chain", err, None if err == "no-mismatch" else "SETUP")
    await _send(ws, sid, "create a class diagram for a hotel booking system")
    r2 = await _collect(ws)
    if not (_built(r2) and _is_resume(r2)):
        return ("full_chain", f"rebuild built={_built(r2)} resume={_is_resume(r2)}", "NO_RESUME")
    # confirm the resumed smart-gen via the real "Continue" button prompt →
    # expect trigger_smart_generator
    await _send(ws, sid, "generate anyway with my current model")
    r3 = await _collect(ws)
    if _has_action(r3, "trigger_smart_generator"):
        return ("full_chain", "rebuild → resume → trigger_smart_generator", None)
    return ("full_chain", f"no-trigger: {[f[0] for f in r3]}", "NO_TRIGGER")


async def generate_anyway(ws, sid):
    r1, err = await _seed_and_mismatch(ws, sid)
    if err:
        return ("generate_anyway", err, None if err == "no-mismatch" else "SETUP")
    await _send(ws, sid, "generate anyway with my current model")
    r2 = await _collect(ws)
    # Should NOT rebuild the model; should reach a smart-gen path on current model.
    if _built(r2):
        return ("generate_anyway", "REBUILT (should keep current)", "UNEXPECTED_REBUILD")
    reached = _is_resume(r2) or _has_action(r2, "trigger_smart_generator") or \
        any("from the specification" in (t or "").lower() or "from your model" in (t or "").lower() or "generate" in (t or "").lower()
            for _a, t, _l in r2)
    return ("generate_anyway", f"no-rebuild, smart-path={reached}", None if reached else "NO_SMART_PATH")


async def cancel_then_create(ws, sid):
    r1, err = await _seed_and_mismatch(ws, sid)
    if err:
        return ("cancel_then_create", err, None if err == "no-mismatch" else "SETUP")
    await _send(ws, sid, "cancel the generation")
    await _collect(ws, quiet=5)
    await _send(ws, sid, "create a class diagram for a zoo with animals and keepers")
    r3 = await _collect(ws)
    if _is_resume(r3):
        return ("cancel_then_create", "SPURIOUS resume after cancel", "SPURIOUS_RESUME")
    return ("cancel_then_create", f"zoo built={_built(r3)}, no resume", None if _built(r3) else "NO_BUILD")


async def unrelated_create(ws, sid):
    r1, err = await _seed_and_mismatch(ws, sid)
    if err:
        return ("unrelated_create", err, None if err == "no-mismatch" else "SETUP")
    # Do NOT click the button — create something entirely different.
    await _send(ws, sid, "create a class diagram for a zoo with animals and keepers")
    r2 = await _collect(ws)
    if _is_resume(r2):
        # The armed flag resumed the OLD (rust hotel) smart-gen on the zoo model.
        return ("unrelated_create", "spurious resume on unrelated create", "SPURIOUS_RESUME")
    return ("unrelated_create", f"zoo built={_built(r2)}, no spurious resume", None)


async def decline_interrupt(ws, sid):
    r1, err = await _seed_and_mismatch(ws, sid)
    if err:
        return ("decline_interrupt", err, None if err == "no-mismatch" else "SETUP")
    await _send(ws, sid, "never mind")
    r2 = await _collect(ws)
    if _built(r2):
        return ("decline_interrupt", "BUILT on 'never mind'", "BUILT_ON_DECLINE")
    return ("decline_interrupt", f"no build; frames={[f[0] for f in r2]}", None)


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(CONC)
    scenarios = [full_chain, generate_anyway, cancel_then_create,
                 unrelated_create, decline_interrupt]
    tasks = [_run(fn.__name__, fn) for fn in scenarios]
    total = len(tasks)
    print(f"=== mismatch EDGE probe: {total} scenarios (CONC={CONC}) against {AGENT_WS_URL} ===\n", flush=True)
    results, done = [], 0
    for coro in asyncio.as_completed(tasks):
        label, outcome, flaw = await coro
        done += 1
        mark = "FLAW" if flaw else "ok  "
        print(f"[{done}/{total}] {mark} {label:20} -> {str(outcome)[:56]}", flush=True)
        results.append((label, outcome, flaw))

    flaws = [r for r in results if r[2] and r[2] not in ("SETUP",)]
    setups = [r for r in results if r[2] == "SETUP"]
    print("\n===== SUMMARY =====")
    print(f"scenarios: {total}   clean: {total - len(flaws) - len(setups)}   "
          f"flaws: {len(flaws)}   setup-skips: {len(setups)}")
    if flaws:
        print("\nflaw detail:")
        for label, outcome, flaw in flaws:
            print(f"   [{label}] {outcome}  ({flaw})")
    else:
        print("\nNO FLAWS — edge cases behave correctly.")


if __name__ == "__main__":
    asyncio.run(main())
