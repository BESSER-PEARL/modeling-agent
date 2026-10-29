"""Any-target probe: the user names what to build (mobile, Windows, MVC, ...)
at any point of the conversation, in any language, and every reply and the
final ``trigger_smart_generator`` honour it.

Each scenario is a scripted conversation driven like the editor does it: a
project id on every turn, the injected model carried back, quick actions
clicked by sending their ``prompt``. Checks run on the replies:

  * ``lang``   replies are in the expected language (stop-word count)
  * ``names``  a reply / the trigger instructions name the target
  * ``route``  the expected action arrives (never a silent cancel)

Usage:
  AGENT_WS_URL=ws://localhost:8790 REPEATS=3 python tests/live/probe_any_target.py
  SCENARIOS=win,web python tests/live/probe_any_target.py   # a subset
  REPORT=out.json  writes every turn's replies for review
"""
import asyncio
import json
import os
import re
import sys
import time
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))

from _agent_ws import connect as agent_ws_connect  # noqa: E402
from probe_workflow_complete import _SHOP_MODEL  # noqa: E402
from test_nl_generation_scenarios import _unwrap, AGENT_WS_URL  # noqa: E402

REPEATS = int(os.environ.get("REPEATS", "3"))
CONC = int(os.environ.get("CONC", "4"))
TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "240"))
_REPLY_ACTIONS = {"inject_complete_system", "trigger_smart_generator", "trigger_generator",
                  "assistant_message", "agent_error", "auto_generate_gui", "modify_model"}

_STOP = {
    "es": {"de", "la", "el", "que", "tu", "una", "para", "con", "los", "las", "puedes", "modelo", "aplicación"},
    "en": {"the", "and", "your", "you", "with", "for", "can", "model", "will", "this", "application"},
}


def _model_from_spec(spec):
    """The class model the editor holds after injecting *spec* (classes and
    their attribute rows; enough for the agent's workspace context)."""
    elements = {}
    for i, cls in enumerate((spec or {}).get("classes") or []):
        cid = f"c{i}"
        attr_ids = []
        for j, attr in enumerate(cls.get("attributes") or []):
            aid = f"{cid}a{j}"
            attr_ids.append(aid)
            elements[aid] = {"id": aid, "type": "ClassAttribute", "owner": cid,
                             "name": f"{attr.get('name')}: {attr.get('type', 'str')}",
                             "bounds": {"x": 0, "y": 0, "width": 200, "height": 30}}
        elements[cid] = {"id": cid, "type": "Class", "owner": None, "name": cls.get("className"),
                         "attributes": attr_ids, "methods": [],
                         "bounds": {"x": 250 * i, "y": 0, "width": 200, "height": 100}}
    if not elements:
        return None
    return {"version": "3.0.0", "type": "ClassDiagram", "size": {"width": 1200, "height": 800},
            "elements": elements, "relationships": {}, "interactive": {"elements": {}, "relationships": {}},
            "assessments": {}}


def lang_of(text):
    words = re.findall(r"[a-záéíóúñü]+", (text or "").lower())
    scores = {k: sum(w in v for w in words) for k, v in _STOP.items()}
    return max(scores, key=scores.get) if any(scores.values()) else "?"


async def _send(ws, sid, pid, text, model, *, raw=None):
    context = {"activeDiagramType": "ClassDiagram",
               "projectSnapshot": {"id": pid, "name": "AnyTarget",
                                   "diagrams": {"ClassDiagram": [{"model": model}]} if model else {}}}
    if model:
        context["activeModel"] = model
    inner = raw or {"action": "user_message", "message": text, "context": context}
    inner = {"protocolVersion": "2.0", "clientMode": "workspace", "sessionId": sid, **inner}
    await ws.send(json.dumps({"action": "user_message", "user_id": sid, "message": json.dumps(inner)}))


async def _turn(ws):
    """Frames of one turn: the first reply action (streams assembled), plus progress text."""
    buf, progress, t0 = "", [], time.monotonic()
    while time.monotonic() - t0 < TIMEOUT:
        try:
            m = _unwrap(await asyncio.wait_for(ws.recv(), timeout=TIMEOUT))
        except asyncio.TimeoutError:
            break
        if not m:
            continue
        act = m.get("action")
        if act == "progress":
            progress.append(m.get("message") or "")
        elif act == "stream_chunk":
            buf += m.get("chunk") or ""
        elif act == "stream_done":
            return {"action": "assistant_message", "message": m.get("fullText") or buf,
                    "progress": progress}
        elif act in _REPLY_ACTIONS:
            return {**m, "progress": progress}
    return {"action": "timeout", "message": buf, "progress": progress}


# Steps: ("say", text) | ("click", None) sends the first quick action's prompt |
# ("smart_done", None) reports a finished smart run like the editor does.
# expect: per step, (actions allowed, lang or None, words of which one must appear or None)
_TARGET_WIN = ("windows",)
_TARGET_MOBILE = ("móvil", "movil", "mobile", "android", "ios")
_TARGET_NATIVE = ("nativ",)
SCENARIOS = {
    # The reviewer's three prompts, each driven to the trigger.
    "win": [
        (("say", "quiero una app de windows 11 local para gestionar el inventario de mi tienda"),
         ({"inject_complete_system"}, "es", _TARGET_WIN)),
        (("click", None), ({"assistant_message"}, "es", _TARGET_WIN)),
        (("click", None), ({"trigger_smart_generator"}, None, _TARGET_WIN)),
    ],
    "mobile": [
        (("say", "quiero una app móvil para controlar mis gastos"),
         ({"inject_complete_system"}, "es", _TARGET_MOBILE)),
        (("click", None), ({"assistant_message"}, "es", _TARGET_MOBILE)),
        (("click", None), ({"trigger_smart_generator"}, None, _TARGET_MOBILE)),
    ],
    "mvc": [
        (("say", "crea una aplicación con arquitectura MVC para gestionar una biblioteca"),
         ({"inject_complete_system"}, "es", None)),
        (("say", "generate the application"), ({"assistant_message"}, "es", ("mvc",))),
        (("say", "no, una app nativa, no una web app"), ({"assistant_message"}, "es", _TARGET_NATIVE)),
        (("click", None), ({"trigger_smart_generator"}, None, _TARGET_NATIVE + ("mvc",))),
    ],
    # Table C row 2: a model already there, target named on the generate request.
    "row2": [
        (("say_with_model", "now generate a Windows app from my model"),
         ({"assistant_message"}, "en", _TARGET_WIN)),
        (("click", None), ({"trigger_smart_generator"}, None, _TARGET_WIN)),
    ],
    # Row 3: changed mid-confirmation; latest statement wins.
    "row3": [
        (("say", "I want a mobile app to track my expenses"),
         ({"inject_complete_system"}, "en", _TARGET_MOBILE)),
        (("click", None), ({"assistant_message"}, "en", _TARGET_MOBILE)),
        (("say", "actually make it a desktop app"), ({"assistant_message"}, "en", ("desktop",))),
        (("click", None), ({"trigger_smart_generator"}, None, ("desktop",))),
    ],
    # Row 4: the target in turn 1, the domain in turn 2. The agent may build a
    # model from turn 1 alone; either way the target must reach the build.
    "row4": [
        (("say", "I want a mobile app"), (_REPLY_ACTIONS, "en", None)),
        (("say", "for gym tracking"), (_REPLY_ACTIONS, "en", _TARGET_MOBILE)),
        (("say", "generate anyway with my current model"),
         ({"trigger_smart_generator", "assistant_message"}, None, _TARGET_MOBILE)),
    ],
    # Row 5: a hand-drawn model, no history: no target keeps develop's default
    # (the deterministic web_app route, which asks for a GUI diagram).
    "row5": [
        (("say_with_model", "generate the app"), ({"assistant_message"}, "en", ("web_app", "built-in generators"))),
        (("say_with_model", "generate a mobile app"), ({"assistant_message"}, "en", _TARGET_MOBILE)),
    ],
    # Row 6: a feature added to a generated native app stays a code change.
    "row6": [
        (("say", "quiero una app móvil para controlar mis gastos"), ({"inject_complete_system"}, "es", None)),
        (("click", None), ({"assistant_message"}, "es", None)),
        (("click", None), ({"trigger_smart_generator"}, None, _TARGET_MOBILE)),
        (("smart_done", None), ({"assistant_message"}, None, None)),
        (("say", "añade una pantalla de ajustes"), ({"assistant_message", "trigger_smart_generator"}, "es", None)),
    ],
    # Web regression: must route as on develop (compare with a develop agent).
    "web": [
        (("say", "create a web app for a todo list"), (_REPLY_ACTIONS, "en", None)),
    ],
}


async def run(name, rep):
    sid, pid = f"at_{name}_{uuid.uuid4().hex[:6]}", f"proj-{uuid.uuid4().hex[:8]}"
    model, last, log, failures = None, None, [], []
    async with agent_ws_connect(AGENT_WS_URL, max_size=None) as ws:
        for (kind, text), (allowed, lang, words) in SCENARIOS[name]:
            if kind == "click":
                actions = (last or {}).get("suggestedActions") or []
                text = actions[0]["prompt"] if actions else "generate anyway with my current model"
            if kind == "say_with_model":
                model = model or _SHOP_MODEL
            if kind == "smart_done":
                await _send(ws, sid, pid, "", model, raw={
                    "action": "frontend_event", "eventType": "generator_result", "ok": True,
                    "message": "done", "metadata": {"smart": True, "filename": "app.zip"}})
            else:
                await _send(ws, sid, pid, text, model)
            r = await _turn(ws)
            if r.get("action") == "inject_complete_system":
                model = _model_from_spec(r.get("systemSpec")) or _SHOP_MODEL
            body = r.get("message") or ""
            checked = " ".join([body, r.get("instructions") or ""]).lower()
            entry = {"sent": text, "action": r.get("action"), "message": body,
                     "labels": [a.get("label") for a in r.get("suggestedActions") or []],
                     "instructions": r.get("instructions"), "progress": r.get("progress")}
            log.append(entry)
            if r.get("action") not in allowed:
                failures.append(f"{text[:30]!r}: got {r.get('action')}")
            if lang and body and lang_of(body) != lang:
                failures.append(f"{text[:30]!r}: reply in {lang_of(body)}")
            if words and not any(w in checked for w in words):
                failures.append(f"{text[:30]!r}: target {words[0]!r} not named")
            last = r
    return name, rep, failures, log


async def main():
    names = [n for n in os.environ.get("SCENARIOS", ",".join(SCENARIOS)).split(",") if n]
    sem = asyncio.Semaphore(CONC)

    async def guarded(n, i):
        async with sem:
            try:
                return await run(n, i)
            except Exception as exc:
                return n, i, [f"EXC {type(exc).__name__}: {exc}"], []
    results = await asyncio.gather(*(guarded(n, i) for n in names for i in range(REPEATS)))
    by = {}
    for name, rep, failures, _ in results:
        by.setdefault(name, []).append(failures)
    for name in names:
        runs = by[name]
        ok = sum(1 for f in runs if not f)
        print(f"{name:7s} {ok}/{len(runs)} ok")
        for f in runs:
            for line in f:
                print(f"         - {line}")
    if os.environ.get("REPORT"):
        with open(os.environ["REPORT"], "w", encoding="utf-8") as fh:
            json.dump([{"scenario": n, "rep": i, "failures": f, "turns": log}
                       for n, i, f, log in results], fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
