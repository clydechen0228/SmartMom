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
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from aps import inbox  # noqa: E402
from aps import labels as labels_mod  # noqa: E402
from aps.explain import explain, load_by_day  # noqa: E402
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
    start_hours: Optional[float] = None
    shift_hours: Optional[float] = None
    priority: Optional[int] = Field(default=None, ge=1, le=3)
    source: Optional[str] = Field(default=None, max_length=2000)
    reading_id: Optional[str] = Field(default=None, max_length=32)   # the Laya reading it came from


class LabelBody(BaseModel):
    """A planner decision on a Laya reading, for training data."""
    reading_id: str = Field(max_length=32)
    kind: Optional[str] = None                     # inbox: the event kind the message really was
    gold: Dict[str, str] = {}                      # or answers per question id
    source: str = Field(default="planner", max_length=40)
    weak: bool = False


class ScenarioBody(EventBody):
    label: str = Field(default="", max_length=300)


class TextBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class DecideBody(BaseModel):
    approve: bool
    who: str = Field(min_length=1, max_length=64)
    comment: str = Field(default="", max_length=1000)


class RerunBody(BaseModel):
    mode: str = Field(pattern="^(narrow|protect)$")
    orders: List[str] = []


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
        self.rejections: list = []             # Laya's reading of each rejection comment
        self.notes: list = []                  # shift notes read by Laya, newest last
        self.labels = labels_mod.LabelLog()    # planner decisions on Laya readings -> training data

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
        if k not in meta:
            continue                                  # a cancelled order's finished work
        o, op = meta[k]
        state = "done" if k in pl.done else ("running" if a["start"] < pl.now else
                                               ("frozen" if k in frozen_ids else "planned"))
        rows.append({"id": k, "order": o.id, "material": o.material, "op": "%04d" % op.seq,
                     "wc": op.work_center, "machine": a["machine"], "start": a["start"], "end": a["end"],
                     "setup": a.get("setup", 0), "state": state, "order_status": status.get(o.id)})
    return rows


def with_why(pl, plant, allops, rows, holds):
    """Late and at-risk orders carry their main cause, from the schedule itself."""
    for r in rows:
        if r["status"] != "ok":
            e = explain(plant, allops, r["order"], holds, pl.now)
            r["why_en"], r["why_zh"] = e.get("summary_en"), e.get("summary_zh")
    return rows


def scenario_summary(pl, x):
    return {"id": x.id, "event": x.event, "label": x.label, "kpis": x.kpis, "before": x.before,
            "status": x.status, "seconds": x.seconds, "violations": x.violations,
            "stale": x.base_version != len(pl.versions), "affected": x.affected}


def rejection_tally():
    """How often each reason came up, counting only readings above the gate."""
    tally = {k: 0 for k in inbox.REJECT_Q["criteria"]}
    for r in A.rejections:
        if r["confidence"] >= inbox.GATE:
            tally[r["reason"]] += 1
    return {"counts": tally, "total": len(A.rejections), "recent": A.rejections[-5:]}


def machine_flags():
    """The latest note per machine; flagged ones mark the machine on the Gantt."""
    out = {}
    for n in A.notes:
        if n.get("machine"):
            out[n["machine"]] = {"flagged": n["flagged"], "p_problem": n["p_problem"], "text": n["text"], "at": n["at"]}
    return out


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
            "orders": with_why(pl, pl.plant, pl.all_ops(), pl.orders_view(), pl.holds),
            "load": load_by_day(pl.plant, pl.all_ops()),
            "scenarios": [scenario_summary(pl, x) for x in pl.scenarios],
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
            "last_rejected": pl.last_rejected,
            "rejections": rejection_tally(),
            "notes": A.notes[-8:],
            "machine_flags": machine_flags(),
            "labels": A.labels.summary(),
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
                "orders": with_why(pl, prop.plant, pl.all_ops(prop.schedule),
                                   pl.orders_view(prop.plant, prop.schedule), prop.holds),
                "load": load_by_day(prop.plant, pl.all_ops(prop.schedule)),
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
    ev = {k: v for k, v in body.model_dump().items() if v is not None and k != "reading_id"}
    try:
        A.pl.validate(dict(ev))
    except EventError as e:
        raise HTTPException(400, str(e))
    if A.pl.proposal:
        raise HTTPException(409, "approve or reject the open proposal first")
    out = A.run_job("repair", lambda: A.pl.propose(ev))
    if body.reading_id:
        A.labels.label_event(body.reading_id, ev["kind"])
    return out


@app.post("/api/scenarios", status_code=202)
def scenario(body: ScenarioBody):
    ev = {k: v for k, v in body.model_dump().items() if v not in (None, "") and k not in ("label", "reading_id")}
    try:
        A.pl.validate(dict(ev))
    except EventError as e:
        raise HTTPException(400, str(e))
    out = A.run_job("scenario", lambda: A.pl.what_if(ev, body.label))
    if body.reading_id:
        A.labels.label_event(body.reading_id, ev["kind"])
    return out


@app.get("/api/scenarios/{sid}")
def scenario_detail(sid: int):
    pl = A.pl
    with pl.lock:
        x = next((s for s in pl.scenarios if s.id == sid), None)
        if x is None:
            raise HTTPException(404, "no such scenario")
        return {**scenario_summary(pl, x), "ops": ops_view(pl, x.plant, x.schedule, set(x.fixed)),
                "orders": with_why(pl, x.plant, pl.all_ops(x.schedule), pl.orders_view(x.plant, x.schedule), x.holds),
                "blocked": blocked_view(x.plant), "load": load_by_day(x.plant, pl.all_ops(x.schedule))}


@app.post("/api/scenarios/{sid}/promote")
def promote(sid: int):
    if A.job and A.job["running"]:
        raise HTTPException(409, "wait for the running solve")
    try:
        return {"proposal": A.pl.promote(sid).id}
    except EventError as e:
        raise HTTPException(409, str(e))


@app.delete("/api/scenarios/{sid}")
def discard(sid: int):
    A.pl.discard(sid)
    return {"ok": True}


@app.get("/api/explain/{order}")
def explain_order(order: str, view: str = "current"):
    pl = A.pl
    with pl.lock:
        plant, sched, holds = pl.plant, pl.schedule, pl.holds
        if view == "proposal" and pl.proposal:
            plant, sched, holds = pl.proposal.plant, pl.proposal.schedule, pl.proposal.holds
        elif view.startswith("scenario:"):
            x = next((s for s in pl.scenarios if str(s.id) == view.split(":", 1)[1]), None)
            if x:
                plant, sched, holds = x.plant, x.schedule, x.holds
        e = explain(plant, pl.all_ops(sched), order, holds, pl.now)
    if not e.get("found"):
        raise HTTPException(404, "no such order in this plan")
    return e


@app.post("/api/proposal/{pid}/decide")
def decide(pid: int, body: DecideBody):
    try:
        v = A.pl.decide(pid, body.approve, body.who.strip(), body.comment.strip())
    except EventError as e:
        raise HTTPException(409, str(e))
    out = {"version": v.id if v else None, "rejection": None}
    comment = body.comment.strip()
    if not body.approve and comment and A.clf is not None and A.clf.state == "ready":
        r = inbox.read_rejection(A.clf, comment, A.pl.last_rejected, A.pl.plant)
        r["reading_id"] = A.labels.register("rejection", comment, {"reason": inbox.REJECT_Q},
                                            {"reason": {"choice": r["reason"], "confidence": r["confidence"],
                                                        "probabilities": r["probabilities"]}},
                                            (r.get("routing") or {}).get("model"))
        with A.lock:
            A.rejections = (A.rejections + [{"ts": time.time(), "proposal": pid, "reason": r["reason"],
                                            "confidence": r["confidence"], "comment": comment}])[-50:]
        if A.pl.last_rejected is not None:
            A.pl.last_rejected["reading"] = r
        out["rejection"] = r
    return out


@app.post("/api/rerun", status_code=202)
def rerun(body: RerunBody):
    """Run the last rejected event again, with fewer moves or with orders protected."""
    last = A.pl.last_rejected
    if not last:
        raise HTTPException(404, "nothing was rejected since the last clock step")
    if A.pl.proposal:
        raise HTTPException(409, "approve or reject the open proposal first")
    ev = {k: v for k, v in last["event"].items() if k not in ("narrow", "protect")}
    if last["event"]["kind"] == "rush_order":
        ev.pop("order", None)                     # the solver numbers a new rush order again
    if body.mode == "narrow":
        ev["narrow"] = True
    else:
        if not body.orders:
            raise HTTPException(400, "name the orders to protect")
        ev["protect"] = body.orders
    try:
        A.pl.validate(dict(ev))
    except EventError as e:
        raise HTTPException(400, str(e))
    out = A.run_job("repair", lambda: A.pl.propose(ev))
    rid = (last.get("reading") or {}).get("reading_id")
    if rid:
        A.labels.label(rid, {"reason": labels_mod.RERUN_REASON[body.mode]}, "clicked_step")
    return out


@app.get("/api/notes/samples")
def note_samples():
    return inbox.note_samples()


@app.post("/api/notes/read")
def read_note(body: TextBody):
    """An operator's shift note: Laya reads the machine's condition, rules find the machine."""
    if A.clf is None or A.clf.state != "ready":
        raise HTTPException(503, "the language model is still loading")
    r = inbox.read_note(A.clf, body.text, A.pl.plant)
    r["reading_id"] = A.labels.register("note", body.text, {"condition": inbox.NOTE_Q},
                                        {"condition": {"choice": r["condition"], "confidence": None,
                                                       "probabilities": r.get("probabilities")}},
                                        (r.get("routing") or {}).get("model"))
    r["ts"], r["at"] = time.time(), A.pl.now
    with A.lock:
        A.notes = (A.notes + [r])[-20:]
    return r


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
    return prepare(inbox.read_message(A.clf, body.text, A.pl.plant, A.pl.now))


def prepare(r: dict) -> dict:
    """Give the reading an id, so the planner's decision on it becomes a training label."""
    qs = inbox.COMMAND_QUESTIONS if "intent" in r else inbox.QUESTIONS
    cls = r.get("classification") or {}
    r["reading_id"] = A.labels.register("command" if "intent" in r else "inbox", r["text"], qs,
                                        cls.get("answers") or {}, (cls.get("routing") or {}).get("model"))
    return r


@app.post("/api/labels")
def add_label(body: LabelBody):
    """The planner's decision on a reading that no event carried: a dismissal, a picked
    intent, a clicked next step, a marked shift note."""
    gold = labels_mod.inbox_gold(body.kind) if body.kind else dict(body.gold)
    rec = A.labels.label(body.reading_id, gold, body.source, "weak" if body.weak else "strong")
    return {"stored": rec is not None}


@app.get("/api/labels/stats")
def label_stats():
    return {"path": A.labels.path, "jobs": A.labels.stats()}


URGENCY_RANK = {"today": 0, "this_week": 1, "info": 2}


@app.post("/api/inbox/triage")
def triage():
    """Read every sample message and sort: quarantined first, then by urgency."""
    if A.clf is None or A.clf.state != "ready":
        raise HTTPException(503, "the language model is still loading")
    rows = []
    for s in inbox.samples():
        r = prepare(inbox.read_message(A.clf, s["text"], A.pl.plant, A.pl.now))
        rows.append({**s, "reading": r})
    rows.sort(key=lambda x: (not x["reading"]["guard"]["flagged"], x["reading"]["event"] is None,
                             URGENCY_RANK.get(x["reading"]["urgency"], 3), -x["reading"]["confidence"]))
    return rows


@app.get("/api/command/samples")
def command_samples():
    return inbox.command_samples()


@app.post("/api/command")
def command(body: TextBody):
    """The planner's command bar: Laya reads the intent, rules and the solver do the rest."""
    if A.clf is None or A.clf.state != "ready":
        raise HTTPException(503, "the language model is still loading")
    r = prepare(inbox.read_message(A.clf, body.text, A.pl.plant, A.pl.now, command=True))
    intent, conf = r["intent"], r["intent_confidence"]
    out = {"reading": r, "intent": intent, "intent_confidence": conf}
    if r["guard"]["flagged"]:
        out["action"] = "quarantined"               # the planner picks the intent, if any
    elif conf < inbox.GATE:
        out["action"] = "choose_intent"
    elif intent == "why":
        order = r["fields"].get("order")
        if not order:
            out["action"] = "need_order"
        else:
            try:
                out["action"], out["explanation"] = "explain", explain_order(order)
            except HTTPException as e:
                out["action"], out["error"] = "need_order", e.detail
    elif intent == "what_if" and r.get("event") and not r.get("missing"):
        ev = dict(r["event"])
        try:
            A.pl.validate(dict(ev))
            A.run_job("scenario", lambda: A.pl.what_if(ev, body.text))
            out["action"] = "scenario_started"
        except (EventError, HTTPException) as e:
            out["action"], out["error"] = "complete", getattr(e, "detail", str(e))
    else:
        out["action"] = "complete"                   # the planner completes the event form
    return out


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


def start(mock: bool, device: Optional[str] = None, plan_on_start: bool = True, classifier=None,
          labels_path: Optional[str] = None, models: Optional[dict] = None):
    A.pl = Planning(load_plant())
    A.labels = labels_mod.LabelLog(labels_path)
    A.clf = classifier or (inbox.MockClassifier() if mock else inbox.LayaClassifier(device, models=models))
    A.clf.warm_async()
    if plan_on_start:
        A.run_job("full_plan", A.pl.full_plan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--mock", action="store_true", help="keyword stand-in instead of Laya")
    ap.add_argument("--device", default=None)
    ap.add_argument("--laya-english", default=None, help="checkpoint for English text: a directory or hub id")
    ap.add_argument("--laya-multilingual", default=None, help="checkpoint for other languages (中文, Deutsch …)")
    ap.add_argument("--labels", default=None, help="label log (default data/labels/labels.jsonl or $APS_LABELS)")
    args = ap.parse_args()
    models = inbox.models_from_env()
    models.update({k: v for k, v in (("english", args.laya_english), ("multilingual", args.laya_multilingual)) if v})
    start(args.mock, args.device, labels_path=args.labels, models=models or None)
    import uvicorn
    print("plant APS on http://%s:%d (%s)" % (args.host, args.port, "mock" if args.mock else "Laya"))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
