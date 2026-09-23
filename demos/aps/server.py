"""Plant APS demo - service and planner workbench.

    python demos/aps/server.py                 # http://127.0.0.1:8095
    python demos/aps/server.py --mock          # keyword stand-in instead of Laya

Implements the design in "Plant APS Design": SAP-shaped orders for two lines, a CP-SAT
schedule that puts on-time delivery first, event-driven repair proposals the planner
approves, a Laya-read disruption inbox, and publish previews for the MES and SAP.
Solves run as background jobs (a full plan takes up to 60 s, a repair up to 30 s).
"""
import argparse
import os
import sys
import threading
import time
import traceback
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from aps import inbox  # noqa: E402
from aps.kpis import kpis  # noqa: E402
from aps.plant import DAY, HORIZON_DAYS, SHIFT_END, load_plant  # noqa: E402
from aps.solver import SPILL_DAYS  # noqa: E402
from aps.state import EVENT_KINDS, FREEZE_MIN, REQUIRED, EventError, Planning  # noqa: E402

STATIC = os.path.join(HERE, "static")


class EventBody(BaseModel):
    kind: str
    machine: Optional[str] = None
    order: Optional[str] = None
    material: Optional[str] = None
    hours: Optional[float] = None
    percent: Optional[float] = None
    quantity: Optional[float] = None
    due_hours: Optional[float] = None
    source: Optional[str] = Field(default=None, max_length=2000)


class TextBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class DecideBody(BaseModel):
    approve: bool
    who: str = Field(min_length=1, max_length=64)


class ClockBody(BaseModel):
    minutes: int = Field(gt=0, le=24 * 60)


class App:
    def __init__(self):
        self.pl: Optional[Planning] = None
        self.clf = None
        self.job: Optional[dict] = None
        self.lock = threading.Lock()
        self.suggestions: list = []            # events proposed by other modules (quality)
        self._sid = 0

    def suggest(self, source: str, title: str, event: dict, detail: str = "", key: Optional[str] = None,
                title_zh: str = "", detail_zh: str = "") -> dict:
        """Another module proposes an event; the planner completes and sends it. With a
        `key`, an open suggestion for the same thing (say, the same order's scrap) is
        updated instead of piling up a new one."""
        with self.lock:
            missing = [k for k in REQUIRED.get(event.get("kind"), ()) if event.get(k) in (None, "")]
            fields = {"ts": time.time(), "source": source, "title": title, "detail": detail,
                      "title_zh": title_zh or title, "detail_zh": detail_zh or detail,
                      "event": event, "missing": missing}
            if key:
                for x in self.suggestions:
                    if x.get("key") == key and x["status"] == "open":
                        x.update(fields)
                        x["updates"] = x.get("updates", 1) + 1
                        return x
            self._sid += 1
            s = {"id": self._sid, "key": key, "status": "open", "updates": 1, **fields}
            self.suggestions = (self.suggestions + [s])[-30:]
            return s

    def run_job(self, kind: str, fn):
        with self.lock:
            if self.job and self.job["running"]:
                raise HTTPException(409, "a solve is already running")
            self.job = {"kind": kind, "started": time.time(), "running": True, "error": None}

        def work():
            try:
                fn()
            except EventError as e:
                self.job["error"] = str(e)
            except Exception as e:                    # keep the service up; show the error
                traceback.print_exc()
                self.job["error"] = "%s: %s" % (type(e).__name__, e)
            finally:
                self.job["running"] = False
                self.job["ended"] = time.time()
        threading.Thread(target=work, daemon=True).start()
        return {"job": self.job}

    def close_by_key(self, key: str, status: str = "dismissed") -> Optional[dict]:
        with self.lock:
            for x in self.suggestions:
                if x.get("key") == key and x["status"] == "open":
                    x["status"] = status
                    return x
        return None


A = App()
app = FastAPI(title="Plant APS demo")


def ops_view(pl: Planning, plant, schedule, frozen_ids=()):
    meta = {op.id: (o, op) for o in plant.orders.values() for op in o.ops}
    status = {s["order"]: s["status"] for s in pl.orders_view(plant, schedule)}
    rows = []
    for k, a in {**pl.done, **schedule}.items():
        o, op = meta[k]
        state = "done" if k in pl.done else ("running" if a["start"] < pl.now else
                                               ("frozen" if k in frozen_ids else "planned"))
        rows.append({"id": k, "order": o.id, "material": o.material, "op": "%04d" % op.seq,
                     "wc": op.work_center, "machine": a["machine"], "start": a["start"], "end": a["end"],
                     "setup": a.get("setup", 0), "state": state, "order_status": status.get(o.id)})
    return rows


def blocked_view(plant):
    limit = (HORIZON_DAYS + SPILL_DAYS) * DAY
    return {m: [b for b in bl if b[0] < limit] for m, bl in plant.blocked.items()}


@app.get("/api/state")
def state():
    pl = A.pl
    if pl is None:
        raise HTTPException(503, "starting")
    with pl.lock:
        frozen = set(pl.frozen())
        prop = pl.proposal
        out = {
            "now": pl.now, "freeze_min": FREEZE_MIN, "horizon_days": HORIZON_DAYS, "shift_end": SHIFT_END,
            "day": DAY,
            "machines": [{"id": m.id, "wc": m.work_center, "line": m.line, "speed": m.speed}
                         for m in pl.plant.machines.values()],
            "orders": pl.orders_view(),
            "ops": ops_view(pl, pl.plant, pl.schedule, frozen),
            "blocked": blocked_view(pl.plant),
            "kpis": pl.versions[-1].kpis if pl.versions else None,
            "current_kpis": None,
            "versions": [{"id": v.id, "ts": v.ts, "reason": v.reason, "approver": v.approver, "kpis": v.kpis}
                         for v in pl.versions],
            "events": pl.events[-30:],
            "proposal": None,
            "job": A.job,
            "engine": A.clf.info() if A.clf else None,
            "event_kinds": EVENT_KINDS,
            "materials": sorted({o.material for o in pl.plant.orders.values()}),
            "suggestions": [x for x in A.suggestions if x["status"] == "open"],
        }
        out["current_kpis"] = kpis(pl.plant, pl.schedule, pl.done)
        if prop:
            moved = {k for k, a in prop.schedule.items() if k in pl.schedule and
                     (a["machine"] != pl.schedule[k]["machine"] or abs(a["start"] - pl.schedule[k]["start"]) > 30)}
            out["proposal"] = {
                "id": prop.id, "event": prop.event, "kpis": prop.kpis, "before": prop.before,
                "status": prop.status, "seconds": prop.seconds, "violations": prop.violations,
                "objective": prop.objective,
                "ops": ops_view(pl, prop.plant, prop.schedule, set(prop.fixed)),
                "orders": pl.orders_view(prop.plant, prop.schedule),
                "blocked": blocked_view(prop.plant),
                "moved": sorted(moved),
            }
        return out


@app.post("/api/plan", status_code=202)
def plan():
    def go():
        with A.pl.lock:
            if A.pl.proposal:
                raise EventError("approve or reject the open proposal first")
        A.pl.full_plan()
    return A.run_job("full_plan", go)


@app.post("/api/events", status_code=202)
def event(body: EventBody):
    ev = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        A.pl.validate(dict(ev))
    except EventError as e:
        raise HTTPException(400, str(e))
    if A.pl.proposal:
        raise HTTPException(409, "approve or reject the open proposal first")
    return A.run_job("repair", lambda: A.pl.propose(ev))


@app.post("/api/proposal/{pid}/decide")
def decide(pid: int, body: DecideBody):
    try:
        v = A.pl.decide(pid, body.approve, body.who.strip())
    except EventError as e:
        raise HTTPException(409, str(e))
    return {"version": v.id if v else None}


@app.post("/api/clock")
def clock(body: ClockBody):
    if A.job and A.job["running"]:
        raise HTTPException(409, "wait for the running solve")
    try:
        return A.pl.advance(body.minutes)
    except EventError as e:
        raise HTTPException(409, str(e))


@app.get("/api/inbox/samples")
def samples():
    return inbox.samples()


@app.post("/api/inbox/read")
def read(body: TextBody):
    if A.clf is None or A.clf.state != "ready":
        raise HTTPException(503, "the language model is still loading")
    return inbox.read_message(A.clf, body.text, A.pl.plant.machines, A.pl.plant.orders)


class SuggestionBody(BaseModel):
    used: bool = False


@app.post("/api/suggestions/{sid}/close")
def close_suggestion(sid: int, body: SuggestionBody):
    for x in A.suggestions:
        if x["id"] == sid and x["status"] == "open":
            x["status"] = "used" if body.used else "dismissed"
            return x
    raise HTTPException(404, "no such open suggestion")


@app.get("/api/publish")
def publish():
    pl = A.pl
    with pl.lock:
        v = pl.versions[-1] if pl.versions else None
        return {"version": v.id if v else None, "dispatch": pl.dispatch_lists(), "sap": pl.sap_updates()}


@app.post("/api/reset", status_code=202)
def reset():
    def go():
        A.pl = Planning(load_plant())
        A.pl.full_plan()
    return A.run_job("full_plan", go)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-cache"})


def start(mock: bool, device: Optional[str] = None, plan_on_start: bool = True, classifier=None):
    A.pl = Planning(load_plant())
    A.clf = classifier or (inbox.MockClassifier() if mock else inbox.LayaClassifier(device))
    A.clf.warm_async()
    if plan_on_start:
        A.run_job("full_plan", A.pl.full_plan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--mock", action="store_true", help="keyword stand-in instead of Laya")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    start(args.mock, args.device)
    import uvicorn
    print("plant APS on http://%s:%d (%s)" % (args.host, args.port, "mock" if args.mock else "Laya"))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
