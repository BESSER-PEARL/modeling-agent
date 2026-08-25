"""Focused live probe for the two fixes shipped in commit 16354b9:

  1. decline_intent  — bare opt-out messages ("nothing", "no thanks", "never
     mind", novel phrasings) must be ACKNOWLEDGED, never routed into a create
     (which would pop the replace/keep prompt on the user's model). A message
     that merely CONTAINS such a word alongside a real request is NOT a decline.

  2. mismatch-loop  — when an existing class diagram doesn't match a
     generate/smart request, the agent offers "Update model + generate". That
     button used to re-send "...and generate the code", which re-hit the same
     mismatch check and looped. It now sends a pure "create a class diagram for
     X"; clicking it must NOT re-show the mismatch confirmation.

Reuses the WS client from test_nl_generation_scenarios (double-JSON protocol).

Usage:
  AGENT_WS_URL=wss://experimental.besser-pearl.org/agent \
      python tests/live/probe_decline_mismatch.py
"""
import asyncio
import os
import sys
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AGENT_WS_URL", "wss://experimental.besser-pearl.org/agent")

import websockets  # noqa: E402
from test_nl_generation_scenarios import _send, _wait_meaningful, AGENT_WS_URL  # noqa: E402

CONC = int(os.environ.get("CONC", "4"))
TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "150"))


def _text(reply) -> str:
    if isinstance(reply, dict):
        v = reply.get("message") or reply.get("fullText") or ""
        if isinstance(v, str):
            return v
    return str(reply)


def _actions(reply) -> list:
    sa = reply.get("suggestedActions") if isinstance(reply, dict) else None
    out = []
    for a in (sa or []):
        if isinstance(a, dict):
            out.append(a)
    return out


def _is_mismatch(reply) -> bool:
    low = _text(reply).lower()
    labels = " ".join((a.get("label") or "") for a in _actions(reply)).lower()
    return ("doesn't match" in low or "does not match" in low
            or "update model + generate" in labels)


# ── Decline scenarios: (prompt, must_not_build) ─────────────────────────────
DECLINE_YES = [  # bare opt-outs — must acknowledge, never build
    "nothing", "no thanks", "never mind", "nah I'm good", "not right now",
    "that's all for now", "I think that's it", "no", "nope", "I'm good",
    "maybe later", "cancel", "stop", "nvm", "that's all",
]
DECLINE_NO = [  # contain a decline word but ARE real requests — must not decline
    ("nothing fancy, just a todo app with users and tasks", "build-or-clarify"),
]


async def _probe_decline(sem, prompt):
    label = prompt[:32]
    async with sem:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                await _send(ws, "d_" + uuid.uuid4().hex[:6], prompt)
                r = await _wait_meaningful(ws, TIMEOUT)
                act = r.get("action") if isinstance(r, dict) else ""
                low = _text(r).lower()
                built = act in ("inject_complete_system", "modify_model", "auto_generate_gui")
                acked = ("no problem" in low or "whenever you'd like" in low
                         or "here whenever" in low)
                if built:
                    return ("decline", label, f"BUILT({act})", "BUILT_ON_DECLINE")
                # acceptable: a plain acknowledgement (ideally the decline ack)
                tag = "acked" if acked else "text-noack"
                return ("decline", label, tag, None)
        except asyncio.TimeoutError:
            return ("decline", label, "HANG", "HANG")
        except Exception as exc:
            return ("decline", label, f"EXC:{type(exc).__name__}", "EXC")


async def _probe_nondecline(sem, prompt, _kind):
    label = prompt[:32]
    async with sem:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                await _send(ws, "nd_" + uuid.uuid4().hex[:6], prompt)
                r = await _wait_meaningful(ws, TIMEOUT)
                act = r.get("action") if isinstance(r, dict) else ""
                low = _text(r).lower()
                # Must NOT be swallowed as a decline ack.
                declined = ("no problem" in low and "whenever you'd like" in low)
                if declined:
                    return ("nondecline", label, "DECLINE_ACK", "MISFIRED_DECLINE")
                return ("nondecline", label, act or "text", None)
        except Exception as exc:
            return ("nondecline", label, f"EXC:{type(exc).__name__}", "EXC")


# Turn-2 triggers that should route to generation→smart against the existing
# (library) model for a DIFFERENT domain, firing the mismatch confirmation.
# Unsupported-stack ("rust") forces the smart route (as in the original bug).
MISMATCH_TRIGGERS = [
    "generate a rust application for a hotel booking system",
    "generate a rust app about a hospital with patients and doctors",
    "generate a full web app with user authentication and a custom dashboard for a restaurant",
    "build the code for a rust scholarship management application",
]


async def _probe_mismatch(sem, trigger):
    """2-3 turn: build library model → ask to generate a DIFFERENT domain app →
    expect mismatch confirmation → click 'Update model + generate' → must NOT
    re-show the mismatch confirmation (the loop bug)."""
    label = trigger[:34]
    async with sem:
        try:
            async with websockets.connect(AGENT_WS_URL, max_size=None, ping_interval=20) as ws:
                sid = "mm_" + uuid.uuid4().hex[:6]
                # Turn 1 — build a library class diagram.
                await _send(ws, sid, "create a class diagram for a library with books and members")
                r0 = await _wait_meaningful(ws, TIMEOUT)
                if (r0.get("action") if isinstance(r0, dict) else "") != "inject_complete_system":
                    return ("mismatch", label, f"seed-failed({r0.get('action')})", "SEED_FAILED")
                # Turn 2 — smart/generate ask for a clearly different domain.
                await _send(ws, sid, trigger)
                r1 = await _wait_meaningful(ws, TIMEOUT)
                if not _is_mismatch(r1):
                    # Not a hard fail — classifier may accept; but we can't test the loop.
                    return ("mismatch", label, f"no-mismatch({r1.get('action')})", None)
                acts = _actions(r1)
                upd = next((a for a in acts if "update model" in (a.get("label") or "").lower()), None)
                if not upd or not upd.get("prompt"):
                    return ("mismatch", label, "no-update-action", "NO_UPDATE_ACTION")
                upd_prompt = upd["prompt"]
                # Turn 3 — click "Update model + generate" (sends its prompt).
                await _send(ws, sid, upd_prompt)
                r2 = await _wait_meaningful(ws, TIMEOUT)
                if _is_mismatch(r2):
                    return ("mismatch", label, f"LOOPED: {_text(r2)[:34]}", "MISMATCH_LOOP")
                act2 = r2.get("action") if isinstance(r2, dict) else ""
                return ("mismatch", label, f"broke-loop -> {act2 or 'text'}", None)
        except asyncio.TimeoutError:
            return ("mismatch", label, "HANG", "HANG")
        except Exception as exc:
            return ("mismatch", label, f"EXC:{type(exc).__name__}", "EXC")


async def main():
    sem = asyncio.Semaphore(CONC)
    only = {c.strip() for c in os.environ.get("ONLY", "").lower().split(",") if c.strip()}
    tasks = []
    if not only or "decline" in only:
        tasks += [_probe_decline(sem, p) for p in DECLINE_YES]
        tasks += [_probe_nondecline(sem, p, k) for p, k in DECLINE_NO]
    if not only or "mismatch" in only:
        tasks += [_probe_mismatch(sem, t) for t in MISMATCH_TRIGGERS]

    total = len(tasks)
    print(f"=== decline + mismatch probe: {total} scenarios (CONC={CONC}) against {AGENT_WS_URL} ===\n", flush=True)
    results, done = [], 0
    for coro in asyncio.as_completed(tasks):
        cat, label, outcome, flaw = await coro
        done += 1
        mark = "FLAW" if flaw else "ok  "
        print(f"[{done:2}/{total}] {mark} {cat:11} {label:34} -> {str(outcome)[:44]}", flush=True)
        results.append((cat, label, outcome, flaw))

    flaws = [r for r in results if r[3]]
    print("\n===== SUMMARY =====")
    print(f"scenarios: {total}   clean: {total - len(flaws)}   flaws: {len(flaws)}")
    if flaws:
        print("\nflaw detail:")
        for cat, label, outcome, flaw in flaws:
            print(f"   [{cat}] {label} -> {outcome}  ({flaw})")
    else:
        print("\nNO FLAWS — decline + mismatch fixes behave correctly live.")


if __name__ == "__main__":
    asyncio.run(main())
