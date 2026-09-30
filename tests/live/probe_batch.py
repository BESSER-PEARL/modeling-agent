"""Run the hotel prompt N times; dump each systemSpec for analysis."""
import asyncio, json, os, sys, time, uuid
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _agent_ws import connect  # noqa
from test_nl_generation_scenarios import _unwrap, AGENT_WS_URL  # noqa

PROMPT = open(os.path.join(HERE, "hotel_prompt.txt"), encoding="utf-8").read()
OUT = os.environ.get("BATCH_OUT", os.path.join(HERE, "batch"))
N = int(os.environ.get("BATCH_N", "10"))


async def one(i):
    sid = "batch%d_%s" % (i, uuid.uuid4().hex[:6])
    async with connect(AGENT_WS_URL) as ws:
        inner = {"action": "user_message", "protocolVersion": "2.0",
                 "clientMode": "widget", "sessionId": sid, "message": PROMPT,
                 "context": {"activeDiagramType": "ClassDiagram"}}
        await ws.send(json.dumps({"action": "user_message", "user_id": sid,
                                  "message": json.dumps(inner)}))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 300:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=40)
            except asyncio.TimeoutError:
                break
            m = _unwrap(raw)
            if not m:
                continue
            if m.get("action") == "inject_complete_system":
                return m.get("systemSpec")
    return None


async def main():
    os.makedirs(OUT, exist_ok=True)
    for i in range(1, N + 1):
        dest = os.path.join(OUT, f"run{i:02d}.json")
        if os.path.exists(dest) and os.path.getsize(dest) > 2:
            print(f"run {i}: already have it, skipping", flush=True)
            continue
        t0 = time.monotonic()
        try:
            spec = await one(i)
        except Exception as exc:                      # noqa: BLE001
            print(f"run {i}: ERROR {type(exc).__name__}: {exc}", flush=True)
            spec = None
        json.dump(spec, open(dest, "w", encoding="utf-8"), indent=2)
        n = len((spec or {}).get("classes", []))
        print(f"run {i}: {'ok' if spec else 'NO SPEC'} "
              f"classes={n} {time.monotonic()-t0:.0f}s", flush=True)

asyncio.run(main())
