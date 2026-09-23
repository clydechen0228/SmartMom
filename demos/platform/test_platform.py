"""Offline tests for the platform: all modules mounted, mock engines, no Laya weights.

    python demos/platform/test_platform.py
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("APS_FULL_S", "10")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import server as platform  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


tmp = tempfile.mkdtemp()
platform.ARGS.mock, platform.ARGS.simulate, platform.ARGS.plan_on_start = True, False, False
platform.ARGS.db = os.path.join(tmp, "p.db")

with TestClient(platform.app) as c:
    aps, qi = platform.aps, platform.qi
    aps.A.pl.full_plan(time_limit=10)

    home = c.get("/")
    check("portal served", (home.status_code, "Smart Factory Platform" in home.text), (200, True))
    check("module path redirects to its slash", c.get("/aps", follow_redirects=False).status_code in (301, 302, 307), True)
    check("quality page", c.get("/quality/").status_code, 200)
    check("quality API under its prefix", c.get("/quality/api/meta").json()["line"]["id"], "L3")
    check("planning page", c.get("/aps/").status_code, 200)
    check("planning API under its prefix", len(c.get("/aps/api/state").json()["ops"]), 150)
    check("console needs the model in mock mode", c.get("/laya/api/meta").status_code, 503)
    check("console page still loads", c.get("/laya/").status_code, 200)
    check("docs served", c.get("/docs/plant-aps-design.html").status_code, 200)
    check("module pages use relative API paths",
          all('"/api/' not in c.get(u).text for u in ("/quality/", "/aps/", "/laya/")), True)

    ov = c.get("/api/overview").json()
    check("overview has every module", sorted(k for k in ov if k in ("quality", "planning", "console")),
          ["console", "planning", "quality"])
    check("overview planning KPIs", ov["planning"]["kpis"]["orders"], 36)

    # quality -> planning: a stop alert becomes one suggestion; repeats and watches do not
    qi.publish("alert", {"id": 901, "station": "LKT-04", "level": "stop",
                         "reasons": ["SPC WE2 on leak rate"], "cause": "machine"})
    qi.publish("alert", {"id": 901, "station": "LKT-04", "level": "stop", "reasons": ["again"]})
    qi.publish("alert", {"id": 902, "station": "CNC-02", "level": "watch", "reasons": ["alarm"]})
    sug = c.get("/aps/api/state").json()["suggestions"]
    check("line stop becomes one planning suggestion", len(sug), 1)
    check("suggested event is a machine-down missing its hours",
          (sug[0]["event"]["kind"], sug[0]["event"]["machine"], sug[0]["missing"]), ("machine_down", "LKT-04", ["hours"]))
    check("bridge feed shows it", c.get("/api/overview").json()["bridge"][-1]["kind"], "stop")
    ev = dict(sug[0]["event"], hours=4)
    check("planner completes it -> repair job", c.post("/aps/api/events", json=ev).status_code, 202)
    check("suggestion closed as used", c.post("/aps/api/suggestions/%d/close" % sug[0]["id"], json={"used": True}).json()["status"], "used")
    check("no open suggestions left", c.get("/aps/api/state").json()["suggestions"], [])
    deadline = time.time() + 60
    while time.time() < deadline and c.get("/aps/api/state").json()["job"]["running"]:
        time.sleep(0.5)
    prop = c.get("/aps/api/state").json()["proposal"]
    check("repair proposal is feasible", (prop is not None, prop and prop["violations"]), (True, []))

    # units -> SAP orders (simulated MES genealogy), holds and scrap -> planning
    g = platform.GENEALOGY
    o1 = g.order_of("HSG7731-T-0001")
    first = min((o for o in aps.A.pl.plant.orders.values() if o.material == "HSG-7731"), key=lambda o: (o.due, o.id))
    check("unit goes to the earliest-due open order of its material", o1, first.id)
    check("same unit, same order", g.order_of("HSG7731-T-0001"), o1)
    check("next unit fills the same order", g.order_of("HSG7731-T-0002"), o1)
    unit = {"serial": "HSG7731-T-0001", "station": "AOI-01", "order": o1, "why": "dent", "why_zh": "磕碰"}
    qi.publish("inspection", dict(unit, id=1, status="pending", disposition="hold"))
    sug = {x["key"]: x for x in c.get("/aps/api/state").json()["suggestions"]}
    hold = sug.get("hold:%s" % o1)
    check("held unit -> quality hold on its order",
          (hold and hold["event"]["kind"], hold and hold["event"]["order"], hold and hold["missing"]), ("quality_hold", o1, ["hours"]))
    check("hold suggestion has a Chinese title", bool(hold and "质量冻结" in hold["title_zh"]), True)
    qi.publish("review", dict(unit, id=1, status="reviewed", disposition="hold", final="scrap"))
    sug = {x["key"]: x for x in c.get("/aps/api/state").json()["suggestions"]}
    check("QA decides the last held unit -> hold suggestion closes", "hold:%s" % o1 in sug, False)
    scrap = sug.get("scrap:%s" % o1)
    check("scrapped unit -> replacement order", (scrap and scrap["event"]["kind"], scrap and scrap["event"]["material"],
          scrap and scrap["event"]["quantity"]), ("rush_order", "HSG-7731", 1))
    qi.publish("inspection", {"id": 2, "serial": "HSG7731-T-0002", "station": "CNC-02", "order": o1,
                              "status": "auto", "disposition": "scrap"})
    sug = {x["key"]: x for x in c.get("/aps/api/state").json()["suggestions"]}
    check("second scrap updates the same suggestion", (sug["scrap:%s" % o1]["id"], sug["scrap:%s" % o1]["event"]["quantity"]),
          (scrap["id"], 2))
    check("replacement due with the order", sug["scrap:%s" % o1]["event"]["due_hours"] > 0, True)
    ev = c.get("/api/overview").json()["bridge"]
    check("portal feed shows hold, release and scrap", {"hold", "release", "scrap"} <= {e["kind"] for e in ev}, True)
    qi.S.order_of = g.order_of
    check("quality records carry the order", c.get("/quality/api/meta").status_code, 200)

    # telemetry through the mounted quality API
    from qi.gateway import LineSimulator
    sim = LineSimulator(seed=4)
    r = c.post("/quality/api/v1/telemetry", json={"gateway": "edge-gw-01", "messages": [sim.next_message() for _ in range(5)]},
               headers={"Authorization": "Bearer demo-gateway-key"})
    check("telemetry accepted through the platform", (r.status_code, r.json()["accepted"]), (202, 5))
    deadline = time.time() + 10
    while time.time() < deadline and len(c.get("/quality/api/inspections").json()) < 5:
        time.sleep(0.1)
    rows = c.get("/quality/api/inspections").json()
    check("quality inspected them", len(rows), 5)
    check("each inspection names its SAP order", all(r["order"] for r in rows), True)
    check("Chinese reason on every summary", all("why_zh" in r for r in rows), True)

print("%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("FAIL", f)
sys.exit(1 if FAIL else 0)
