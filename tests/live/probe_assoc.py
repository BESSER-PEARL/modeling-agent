"""Send the hotel benchmark prompt and dump the systemSpec the agent builds."""
import asyncio, json, os, sys, time, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
HERE = r"C:\Users\sulejmani\Desktop\BESSER-Experimental\modeling-agent\tests\live"
sys.path.insert(0, HERE)
import websockets
from _agent_ws import connect, AGENT_WS_ORIGIN  # noqa
from test_nl_generation_scenarios import _unwrap, AGENT_WS_URL  # noqa

PROMPT = open(os.path.join(os.path.dirname(__file__), "hotel_prompt.txt"),
              encoding="utf-8").read()

async def main():
    sid = "probe_" + uuid.uuid4().hex[:8]
    async with connect(AGENT_WS_URL) as ws:
        inner = {"action": "user_message", "protocolVersion": "2.0",
                 "clientMode": "widget", "sessionId": sid, "message": PROMPT,
                 "context": {"activeDiagramType": "ClassDiagram"}}
        await ws.send(json.dumps({"action": "user_message", "user_id": sid,
                                  "message": json.dumps(inner)}))
        t0, spec = time.monotonic(), None
        while time.monotonic() - t0 < 300:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                if spec: break
                continue
            m = _unwrap(raw)
            if not m: continue
            if m.get("action") == "inject_complete_system":
                spec = m.get("systemSpec")
                break
    out = os.path.join(os.path.dirname(__file__), "spec_out.json")
    json.dump(spec, open(out, "w", encoding="utf-8"), indent=2)
    print("saved:", out, "| got spec:", spec is not None)

asyncio.run(main())
