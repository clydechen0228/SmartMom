"""Smart Factory Platform: every demo in one process, one port, one Laya model.

    python demos/platform/server.py            # http://127.0.0.1:8100
    python demos/platform/server.py --mock     # no Laya weights (Laya console disabled)

    /           portal: live overview of every module (EN / 中文)
    /quality/   quality inspection        (demos/quality_inspection)
    /aps/       planning and scheduling   (demos/aps)
    /laya/      Laya console              (webui)
    /docs/      design documents          (docs/)

The modules stay independent apps; this file mounts them, gives them one shared Laya
router instead of one copy each, and connects them: a line stop raised by quality
inspection becomes a suggested machine-down event in planning, which the planner
completes (how long?) and turns into a repair proposal. Machine ids are the same in both
modules (CNC-02, CMM-01, ASM-03, LKT-04, AOI-01), which is what makes the link direct.
Units are linked to SAP orders by a simulated MES genealogy, so a held unit suggests a
quality hold on its order and scrapped units suggest a replacement order.
"""
import argparse
import importlib.util
import os
import sys
import threading
import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEMOS = os.path.dirname(HERE)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")    # see demos/quality_inspection
for p in (ROOT, os.path.join(ROOT, "examples"), os.path.join(DEMOS, "quality_inspection"),
          os.path.join(DEMOS, "aps")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402


def _load(name: str, path: str):
    """Import a module's server.py under its own name: all three are called server.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qi = _load("qi_server", os.path.join(DEMOS, "quality_inspection", "server.py"))
aps = _load("aps_server", os.path.join(DEMOS, "aps", "server.py"))
console = _load("laya_console", os.path.join(ROOT, "webui", "server.py"))

ARGS = argparse.Namespace(host="127.0.0.1", port=8100, mock=True, device=None, interval=2.5,
                          db=os.path.join(DEMOS, "quality_inspection", "data", "platform.db"),
                          simulate=True, plan_on_start=True)


class Genealogy:
    """Which production order each inspected unit belongs to.

    In a plant the MES knows this (the unit was started against an order). Here the
    platform plays MES: a unit is assigned to the earliest-due unfinished SAP order for
    the line's material, until that order's quantity is used up, then the next one.
    """

    def __init__(self, material: str):
        self.material = material
        self.unit_order = {}
        self.used = {}
        self.lock = threading.Lock()

    def order_of(self, serial: str):
        with self.lock:
            if serial in self.unit_order:
                return self.unit_order[serial]
            pl = aps.A.pl
            if pl is None:
                return None
            for o in sorted((o for o in pl.plant.orders.values() if o.material == self.material),
                            key=lambda o: (o.due, o.id)):
                finished = all(op.id in pl.done for op in o.ops)
                if not finished and self.used.get(o.id, 0) < o.quantity:
                    self.used[o.id] = self.used.get(o.id, 0) + 1
                    self.unit_order[serial] = o.id
                    return o.id
            return None


class Bridge:
    """Quality -> planning. Runs on the quality worker thread; must never block it.

    stop alert on a machine  -> suggested machine-down on the same machine (hours: planner)
    unit held for QA         -> suggested quality hold on its order (hours: planner)
    unit scrapped            -> suggested replacement order for the scrapped quantity
    QA releases the last hold of an order -> the hold suggestion closes itself
    """

    def __init__(self):
        self.feed: List[dict] = []
        self.seen = set()
        self.pending = {}                 # order -> serials held at QA
        self.scrapped = {}                # order -> serials scrapped
        self.lock = threading.Lock()

    def log(self, **item):
        item["ts"] = time.time()
        with self.lock:
            self.feed = (self.feed + [item])[-30:]

    def on_quality(self, kind: str, data: dict):
        if kind == "alert":
            self._stop(data)
        elif kind in ("inspection", "review"):
            self._unit(kind, data)

    def _stop(self, data: dict):
        if data.get("level") != "stop" or data.get("id") in self.seen:
            return                                   # folded repeats keep the same alert id
        self.seen.add(data.get("id"))
        station = data.get("station")
        if aps.A.pl is None or station not in aps.A.pl.plant.machines:
            self.log(kind="stop", station=station, action="ignored")
            return
        reasons = "; ".join(data.get("reasons") or [])
        reasons_zh = "；".join(data.get("reasons_zh") or data.get("reasons") or [])
        s = aps.A.suggest("quality", "Line stop on %s" % station,
                          {"kind": "machine_down", "machine": station, "hours": None},
                          "%s (likely cause: %s)" % (reasons, data.get("cause") or "unknown"),
                          key="stop:%s" % station, title_zh="%s 停线" % station, detail_zh=reasons_zh)
        self.log(kind="stop", station=station, action="suggested", suggestion=s["id"], detail=reasons)

    def _unit(self, kind: str, data: dict):
        order, serial = data.get("order"), data.get("serial")
        pl = aps.A.pl
        if not order or pl is None or order not in pl.plant.orders:
            return
        o = pl.plant.orders[order]
        outcome = data.get("final") if data.get("status") == "reviewed" else data.get("disposition")
        with self.lock:
            held = self.pending.setdefault(order, set())
            if data.get("status") == "pending":
                held.add(serial)
            else:
                held.discard(serial)
            n_held = len(held)
            if outcome == "scrap":
                self.scrapped.setdefault(order, set()).add(serial)
            n_scrap = len(self.scrapped.get(order, ()))
        if data.get("status") == "pending":
            s = aps.A.suggest(
                "quality", "Quality hold on order %s" % order,
                {"kind": "quality_hold", "order": order, "hours": None},
                "%d unit(s) held at QA, last %s at %s: %s" % (n_held, serial, data.get("station"), data.get("why", "")),
                key="hold:%s" % order, title_zh="订单 %s 质量冻结" % order,
                detail_zh="%d 件待 QA 复核，最近一件 %s（%s）：%s" % (n_held, serial, data.get("station"), data.get("why_zh", "")))
            self.log(kind="hold", station=data.get("station"), order=order, serial=serial,
                     action="suggested", suggestion=s["id"])
        elif kind == "review" and n_held == 0 and aps.A.close_by_key("hold:%s" % order, "released"):
            self.log(kind="release", station=data.get("station"), order=order, serial=serial, action="closed")
        if outcome == "scrap" and (kind == "review" or data.get("status") == "auto"):
            due_h = max(1.0, round((o.due - pl.now) / 60.0, 1))
            s = aps.A.suggest(
                "quality", "Replace %d scrapped unit(s) of order %s" % (n_scrap, order),
                {"kind": "rush_order", "material": o.material, "quantity": n_scrap, "due_hours": due_h},
                "Scrapped at %s, last %s. Same due date as the order." % (data.get("station"), serial),
                key="scrap:%s" % order, title_zh="补做订单 %s 报废的 %d 件" % (order, n_scrap),
                detail_zh="在 %s 报废，最近一件 %s。交期与原订单相同。" % (data.get("station"), serial))
            self.log(kind="scrap", station=data.get("station"), order=order, serial=serial,
                     action="suggested", suggestion=s["id"])


BRIDGE = Bridge()
GENEALOGY = Genealogy(qi.LINE["product"].split()[0])        # "HSG-7731 pump housing" -> HSG-7731
SHARED = {"router": None, "device": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    router = None
    if not ARGS.mock:
        from laya import Router
        from examples_common import pick_device
        device = ARGS.device or pick_device()
        router = Router(device=device, max_loaded=3)
        SHARED.update(router=router, device=device)
        console._router, console._device = router, device
    # quality: its own lifespan, told where it is mounted and which engine to use
    qi.ARGS = argparse.Namespace(host=ARGS.host, port=ARGS.port, base="/quality", db=ARGS.db,
                                 device=ARGS.device, mock=ARGS.mock, simulate=ARGS.simulate,
                                 interval=ARGS.interval, engine=None)
    if router is not None:
        from qi.engine import LayaEngine
        qi.ARGS.engine = LayaEngine(SHARED["device"], router=router)
    else:
        from qi.engine import MockEngine
        qi.ARGS.engine = MockEngine()
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(qi.lifespan(qi.app))
        qi.S.listeners.append(BRIDGE.on_quality)
        qi.S.order_of = GENEALOGY.order_of
        # planning: same router, one warm-up for everyone
        from aps import inbox as aps_inbox
        clf = aps_inbox.LayaClassifier(SHARED["device"], router=router) if router else aps_inbox.MockClassifier()
        aps.start(ARGS.mock, ARGS.device, plan_on_start=ARGS.plan_on_start, classifier=clf)
        if router is not None:
            qi.ARGS.engine.warm_async()
        yield


app = FastAPI(title="Smart Factory Platform", lifespan=lifespan)


def _engine_state() -> dict:
    e = qi.S.engine.info() if qi.S.engine else {}
    return {"kind": e.get("kind"), "state": e.get("state"), "device": e.get("device"),
            "loaded": e.get("loaded", []), "version": e.get("version"), "shared": SHARED["router"] is not None}


@app.get("/api/overview")
def overview():
    """One call for the portal: the headline numbers of every module."""
    out = {"ts": time.time(), "engine": _engine_state(), "bridge": BRIDGE.feed[-10:]}
    try:
        st = qi.status()
        out["quality"] = {"stats": st["stats"], "queue": st["queue"], "gateway": st["gateway"],
                          "devices_online": sum(d["online"] for d in st["devices"]),
                          "devices": len(st["devices"])}
    except Exception as e:                   # a module still starting must not break the portal
        out["quality"] = {"error": str(e)}
    try:
        pl = aps.A.pl
        v = pl.versions[-1] if pl and pl.versions else None
        from aps.kpis import kpis
        k = kpis(pl.plant, pl.schedule, pl.done) if pl and pl.schedule else None
        out["planning"] = {"kpis": k, "version": v.id if v else None, "now": pl.now if pl else None,
                           "proposal": bool(pl and pl.proposal), "job": aps.A.job,
                           "suggestions": sum(1 for x in aps.A.suggestions if x["status"] == "open")}
    except Exception as e:
        out["planning"] = {"error": str(e)}
    out["console"] = {"available": SHARED["router"] is not None}
    return out


@app.get("/")
def portal():
    return FileResponse(os.path.join(HERE, "static", "index.html"), headers={"Cache-Control": "no-cache"})


for prefix in ("quality", "aps", "laya", "docs"):
    app.add_api_route("/" + prefix, (lambda p=prefix: RedirectResponse("/%s/" % p)), include_in_schema=False)


@console.app.middleware("http")
async def _console_needs_model(request, call_next):
    if request.url.path.startswith("/api") and SHARED["router"] is None:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "the Laya console needs the model; start without --mock"}, status_code=503)
    return await call_next(request)


app.mount("/quality", qi.app)
app.mount("/aps", aps.app)
app.mount("/laya", console.app)
app.mount("/docs", StaticFiles(directory=os.path.join(ROOT, "docs"), html=True), name="docs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=ARGS.host)
    ap.add_argument("--port", type=int, default=ARGS.port)
    ap.add_argument("--mock", action="store_true", help="keyword stand-ins instead of Laya")
    ap.add_argument("--device", default=None)
    ap.add_argument("--interval", type=float, default=ARGS.interval, help="quality line: seconds between readings")
    ap.add_argument("--db", default=ARGS.db)
    a = ap.parse_args()
    ARGS.host, ARGS.port, ARGS.mock, ARGS.device, ARGS.interval, ARGS.db = a.host, a.port, a.mock, a.device, a.interval, a.db
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.db)), exist_ok=True)
    import uvicorn
    print("smart factory platform on http://%s:%d (%s)" % (ARGS.host, ARGS.port, "mock" if ARGS.mock else "Laya"))
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
