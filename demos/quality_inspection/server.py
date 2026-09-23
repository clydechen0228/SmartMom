"""Production quality inspection with Laya - service and dashboard.

    python demos/quality_inspection/server.py                  # http://127.0.0.1:8090
    python demos/quality_inspection/server.py --mock           # no weights: keyword stand-in
    python demos/quality_inspection/server.py --no-simulate    # wait for a real gateway

IoT edge gateways POST batches of station readings to /api/v1/telemetry. Each reading
is stored raw, then a single worker runs it through the pipeline in arrival order:

    limits + SPC + alarm table (rules)  ->  Laya on the operator note, if there is one
    -> policy gate -> auto disposition or QA review queue, line alerts, traceability record

The dashboard at / follows the pipeline live over server-sent events.
"""
import argparse
import asyncio
import collections
import csv
import hmac
import io
import json
import os
import queue
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from qi import config  # noqa: E402
from qi.config import GATEWAYS, LINE, POLICY, SPC_WINDOW, STATION_ORDER, STATIONS  # noqa: E402
from qi.gateway import SCENARIOS, EdgeGateway, LineSimulator  # noqa: E402
from qi.inspection import QUESTIONS, build_report, decide, note_text  # noqa: E402
from qi.rules import check_limits, control_limits, spc_signals  # noqa: E402
from qi.store import Store  # noqa: E402

STATIC = os.path.join(HERE, "static")
_NOTE_LANG = re.compile(r"Operator note \(([\w-]+)\):")
DEVICE_TO_STATION = {s["device"]: sid for sid, s in STATIONS.items()}


# --- wire format ---------------------------------------------------------------------
class VisionFinding(BaseModel):
    label: str = Field(max_length=120)
    location: Optional[str] = Field(default=None, max_length=120)
    confidence: float = Field(ge=0, le=1)


class OperatorNote(BaseModel):
    text: str = Field(max_length=2000)
    lang: Optional[str] = Field(default=None, max_length=16)


class Reading(BaseModel):
    device: str = Field(max_length=64)
    station: str = Field(max_length=32)
    serial: str = Field(max_length=64)
    seq: int = Field(ge=0)
    ts: str = Field(max_length=40)
    measurements: Dict[str, Optional[float]] = Field(default_factory=dict)
    vision: Optional[List[VisionFinding]] = None
    alarms: List[str] = Field(default_factory=list, max_length=20)
    operator_note: Optional[OperatorNote] = None


class Batch(BaseModel):
    gateway: str = Field(max_length=64)
    messages: List[Reading] = Field(max_length=500)


class ReviewBody(BaseModel):
    disposition: Literal["pass", "rework", "scrap"]
    reviewer: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=1000)


class ScenarioBody(BaseModel):
    name: str


# --- application state ---------------------------------------------------------------
class State:
    def __init__(self):
        self.store: Optional[Store] = None
        self.engine = None
        self.gateway: Optional[EdgeGateway] = None
        self.work: "queue.Queue[tuple]" = queue.Queue(maxsize=2000)
        self.history: Dict[tuple, collections.deque] = {}
        self.devices: Dict[str, dict] = {}
        self.subscribers: List[asyncio.Queue] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.processing: Optional[dict] = None
        self.errors = 0
        self.listeners: List = []          # in-process subscribers, e.g. the platform bridge
        self.order_of = None               # optional serial -> production order (the MES knows)


S = State()


def publish(kind: str, data: dict):
    """Fan an event out to every open dashboard; safe to call from the worker thread."""
    if not S.loop:
        return
    payload = "event: %s\ndata: %s\n\n" % (kind, json.dumps(data, default=str))

    def _put():
        for q in list(S.subscribers):
            if q.qsize() < 500:                     # a stalled tab must not grow without bound
                q.put_nowait(payload)
    S.loop.call_soon_threadsafe(_put)
    for fn in list(S.listeners):
        try:
            fn(kind, data)
        except Exception as e:              # a listener must never break the pipeline
            print("listener error: %s" % e, file=sys.stderr)


def summary(rec: dict) -> dict:
    d = rec["decision"]
    laya = rec.get("laya") or {}
    topic = (laya.get("answers") or {}).get("topic") or {}
    trace = d.get("trace") or [""]
    trace_zh = d.get("trace_zh") or trace
    held = next((i for i, t in enumerate(trace) if "HOLD" in t), len(trace) - 1)
    return {
        "why_zh": trace_zh[held] if held < len(trace_zh) else trace[held],
        "order": S.order_of(rec["serial"]) if S.order_of else None,
        "note": laya.get("input"), "topic": topic.get("choice"), "topic_conf": topic.get("confidence"),
        "why": next((t for t in trace if "HOLD" in t), trace[-1]),
        "id": rec["id"], "serial": rec["serial"], "station": rec["station"], "ts": rec["ts"],
        "disposition": rec["disposition"], "status": rec["status"], "final": rec.get("final"),
        "defect": d.get("defect"), "root_cause": d.get("root_cause"),
        "alert": (d.get("alert") or {}).get("level"),
        "failed": [f["label"] for f in rec["findings"] if f["status"] != "ok" and f["role"] == "product"],
        "latency_ms": rec.get("latency_ms"), "model": rec.get("model"),
        "lang": (_NOTE_LANG.search(rec["report"]) or [None, None])[1],
    }


def seed_history():
    for sid, st in STATIONS.items():
        for key, spec in st["characteristics"].items():
            if spec.get("spc"):
                pts = S.store.series(sid, key, SPC_WINDOW)
                S.history[(sid, key)] = collections.deque((p["value"] for p in pts), maxlen=SPC_WINDOW)


# --- the pipeline --------------------------------------------------------------------
def process(telemetry_id: int, msg: dict):
    sid = msg["station"]
    st = STATIONS[sid]
    chars = st["characteristics"]
    S.processing = {"serial": msg["serial"], "station": sid, "since": time.time()}

    findings = check_limits(chars, msg["measurements"])
    hist = {}
    for key, spec in chars.items():
        if spec.get("spc"):
            dq = S.history.setdefault((sid, key), collections.deque(maxlen=SPC_WINDOW))
            v = msg["measurements"].get(key)
            if v is not None:
                dq.append(v)
            hist[key] = list(dq) if v is not None else []
    spc = spc_signals(chars, hist)
    report = build_report(sid, st, LINE["product"], msg, findings, spc)

    # Laya reads what a person wrote. A record without a note never touches the model.
    note = note_text(msg)
    laya, engine_ok = None, True
    if note:
        # Wait out the initial model load rather than holding the first notes unread.
        while S.engine.state in ("cold", "loading"):
            time.sleep(0.2)
        try:
            if S.engine.state != "ready":
                raise RuntimeError(S.engine.error or "engine not ready")
            laya = {"input": note, **S.engine.predict(note)}
        except Exception as e:                        # a bad record must not stop the line
            S.errors += 1
            engine_ok = False
            laya = {"input": note, "error": "%s: %s" % (type(e).__name__, e)}
    answers = laya.get("answers") if laya else None

    decision = decide(findings, spc, answers, msg, chars, engine_ok=engine_ok)
    status = "auto" if decision["auto"] else "pending"
    rec = {
        "telemetry_id": telemetry_id, "serial": msg["serial"], "station": sid, "ts": msg["ts"],
        "created": time.time(), "findings": findings, "spc": spc, "report": report,
        "laya": laya, "decision": decision, "disposition": decision["disposition"],
        "status": status, "final": decision["disposition"] if status == "auto" else None,
        "latency_ms": (laya or {}).get("ms"),
        "model": ((laya or {}).get("routing") or {}).get("model"),
    }
    rec["id"] = S.store.add_inspection(rec)
    publish("inspection", summary(rec))
    if decision["alert"]:
        publish("alert", S.store.raise_alert(rec["id"], sid, decision["alert"]))
    S.processing = None


def worker():
    while True:
        tid, msg = S.work.get()
        try:
            process(tid, msg)
        except Exception as e:                        # log and keep consuming
            S.errors += 1
            S.processing = None
            print("pipeline error on %s: %s" % (msg.get("serial"), e), file=sys.stderr)


# --- app -----------------------------------------------------------------------------
ARGS: argparse.Namespace


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.loop = asyncio.get_running_loop()
    S.store = Store(ARGS.db)
    seed_history()
    # Readings accepted before a restart but not yet inspected go back on the queue first.
    for item in S.store.unprocessed():
        S.work.put(item)
    if getattr(ARGS, "engine", None) is not None:
        S.engine = ARGS.engine                  # shared with other modules by the platform
    elif ARGS.mock:
        from qi.engine import MockEngine
        S.engine = MockEngine()
    else:
        from qi.engine import LayaEngine
        S.engine = LayaEngine(ARGS.device)
        S.engine.warm_async()
    threading.Thread(target=worker, name="inspection-worker", daemon=True).start()
    S.gateway = EdgeGateway("http://%s:%d%s" % (ARGS.host, ARGS.port, getattr(ARGS, "base", "")), "edge-gw-01",
                            interval=ARGS.interval, sim=LineSimulator())
    if ARGS.simulate:
        S.gateway.start()
    yield
    if S.gateway:
        S.gateway.stop()


app = FastAPI(title="Laya quality inspection", lifespan=lifespan)


@app.post("/api/v1/telemetry", status_code=202)
def ingest(batch: Batch, authorization: str = Header(default="")):
    expected = GATEWAYS.get(batch.gateway)
    token = authorization[7:] if authorization.startswith("Bearer ") else ""
    if not expected or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(401, "unknown gateway or bad key")
    # Refuse the whole batch while the backlog is deep; the gateway keeps it and retries.
    if S.work.qsize() + len(batch.messages) > S.work.maxsize:
        raise HTTPException(429, "inspection backlog full, retry later")

    accepted, duplicates, rejected = 0, 0, []
    now = time.time()
    for m in batch.messages:
        sid = DEVICE_TO_STATION.get(m.device)
        if sid is None or sid != m.station:
            rejected.append({"device": m.device, "seq": m.seq, "error": "device not registered for station"})
            continue
        unknown = set(m.measurements) - set(STATIONS[sid]["characteristics"])
        if unknown:
            rejected.append({"device": m.device, "seq": m.seq, "error": "unknown characteristics %s" % sorted(unknown)})
            continue
        payload = m.model_dump()
        tid = S.store.add_telemetry(batch.gateway, m.device, m.seq, payload)
        dev = S.devices.setdefault(m.device, {"device": m.device, "station": sid, "count": 0})
        dev.update(last_seen=now, gateway=batch.gateway)
        dev["count"] += 1
        if tid is None:
            duplicates += 1
            continue
        S.work.put((tid, payload))
        accepted += 1
    publish("telemetry", {"accepted": accepted, "queue": S.work.qsize()})
    return {"accepted": accepted, "duplicates": duplicates, "rejected": rejected, "queue_depth": S.work.qsize()}


@app.get("/api/meta")
def meta():
    return {
        "line": LINE,
        "stations": [{"id": sid, **{k: v for k, v in STATIONS[sid].items()}} for sid in STATION_ORDER],
        "questions": QUESTIONS,
        "policy": POLICY,
        "engine": S.engine.info(),
        "scenarios": {k: {"station": v[0], "units": v[1], "label": v[2], "about": v[3]}
                      for k, v in SCENARIOS.items()},
        "simulate": bool(S.gateway and S.gateway.running),
    }


@app.get("/api/status")
def status():
    now = time.time()
    devices = []
    for sid in STATION_ORDER:
        dev = STATIONS[sid]["device"]
        d = S.devices.get(dev, {"device": dev, "station": sid, "count": 0, "last_seen": None})
        devices.append({**d, "online": bool(d.get("last_seen") and now - d["last_seen"] < config.DEVICE_TIMEOUT_S)})
    return {
        "engine": S.engine.info(),
        "queue": S.work.qsize(),
        "processing": S.processing,
        "errors": S.errors,
        "devices": devices,
        "gateway": S.gateway.status() if S.gateway else None,
        "stats": S.store.stats(),
    }


@app.get("/api/inspections")
def inspections(limit: int = 50, status: Optional[str] = None, station: Optional[str] = None,
                before: Optional[int] = None):
    rows = S.store.inspections(min(limit, 500), status, station, before)
    return [summary(r) for r in rows]


@app.get("/api/inspections/{iid}")
def inspection(iid: int):
    r = S.store.inspection(iid)
    if not r:
        raise HTTPException(404, "no such inspection")
    st = STATIONS[r["station"]]
    r["charts"] = {k: {"label": s["label"], "unit": s["unit"], "decimals": s.get("decimals", 3),
                       "limits": control_limits(s),
                       "points": S.store.series(r["station"], k, SPC_WINDOW * 2)}
                   for k, s in st["characteristics"].items() if s.get("spc")}
    r["station_name"] = st["name"]
    return r


@app.post("/api/inspections/{iid}/review")
def review(iid: int, body: ReviewBody):
    if not S.store.review(iid, body.disposition, body.reviewer.strip(), body.note.strip()):
        raise HTTPException(409, "not pending review (already reviewed, or auto-dispositioned)")
    rec = S.store.inspection(iid)
    publish("review", summary(rec))
    return summary(rec)


@app.get("/api/line")
def line():
    out = []
    for sid in STATION_ORDER:
        st = STATIONS[sid]
        key = st["key"]
        spec = st["characteristics"][key]
        last = S.store.inspections(1, station=sid)
        out.append({
            "id": sid, "name": st["name"], "device": st["device"], "sensor": st["sensor"],
            "key": key, "label": spec["label"], "unit": spec["unit"], "decimals": spec.get("decimals", 3),
            "limits": control_limits(spec), "points": S.store.series(sid, key, 30),
            "last": summary(last[0]) if last else None,
        })
    return out


@app.get("/api/alerts")
def alerts(limit: int = 30):
    return S.store.alerts(min(limit, 200))


@app.post("/api/alerts/{aid}/ack")
def ack(aid: int):
    if not S.store.ack_alert(aid):
        raise HTTPException(409, "already acknowledged or unknown")
    publish("ack", {"id": aid})
    return {"ok": True}


@app.get("/api/export.csv")
def export():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "serial", "station", "ts", "disposition", "status", "final", "defect",
                "root_cause", "alert", "failed", "model", "latency_ms", "reviewer", "review_note"])
    for r in S.store.export_rows():
        d = r["decision"]
        w.writerow([r["id"], r["serial"], r["station"], r["ts"], r["disposition"], r["status"], r["final"],
                    d.get("defect"), d.get("root_cause"), (d.get("alert") or {}).get("level"),
                    "; ".join(f["label"] for f in r["findings"] if f["status"] != "ok"),
                    r["model"], r["latency_ms"], r["reviewer"], r["review_note"]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=inspections.csv"})


# --- simulator control (only for the in-process gateway) -----------------------------
@app.get("/api/sim")
def sim_status():
    return S.gateway.status()


@app.post("/api/sim/scenario")
def sim_scenario(body: ScenarioBody):
    if body.name not in SCENARIOS:
        raise HTTPException(404, "unknown scenario")
    S.gateway.sim.inject(body.name)
    publish("sim", S.gateway.status())
    return S.gateway.status()


@app.post("/api/sim/{action}")
def sim_action(action: Literal["start", "stop", "clear"]):
    {"start": S.gateway.start, "stop": S.gateway.stop, "clear": S.gateway.sim.clear}[action]()
    publish("sim", S.gateway.status())
    return S.gateway.status()


@app.get("/api/stream")
async def stream(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    S.subscribers.append(q)

    async def gen():
        try:
            yield "retry: 3000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    yield await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            S.subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-cache"})


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--db", default=os.path.join(HERE, "data", "qi.db"))
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--mock", action="store_true", help="keyword stand-in instead of Laya (UI work only)")
    ap.add_argument("--no-simulate", dest="simulate", action="store_false",
                    help="do not run the in-process gateway; wait for a real one")
    ap.add_argument("--interval", type=float, default=2.5, help="simulated seconds between readings")
    ARGS = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.db)), exist_ok=True)

    import uvicorn
    print("quality inspection on http://%s:%d  (%s engine)" % (ARGS.host, ARGS.port, "mock" if ARGS.mock else "Laya"))
    # Open dashboards hold an event stream; do not let them stall a shutdown.
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
