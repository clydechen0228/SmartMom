"""Offline tests for the APS demo: no Laya weights, short solver limits (about a minute).

    python demos/aps/test_aps.py
"""
import copy
import os
import sys
import time

os.environ.setdefault("APS_FULL_S", "15")
os.environ.setdefault("APS_REPAIR_S", "10")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from aps import inbox  # noqa: E402
from aps.checker import check  # noqa: E402
from aps.kpis import kpis  # noqa: E402
from aps.plant import DAY, load_plant, order_from_sap, routings  # noqa: E402
from aps.solver import Problem, dispatch_edd, solve  # noqa: E402
from aps.state import EventError, Planning  # noqa: E402

PASS, FAIL = [], []


def check_(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


# --- master data and the SAP adapter -------------------------------------------------------
plant = load_plant()
check_("36 SAP orders", len(plant.orders), 36)
check_("8 machines on 2 lines", (len(plant.machines), sorted({m.line for m in plant.machines.values()})), (8, ["L1", "L3"]))
o = order_from_sap({"ManufacturingOrder": "1", "Material": "HSG-7731", "TotalQuantity": 100, "DueDay": 1,
                    "DueTime": "14:00", "MaterialAvailDay": 0, "MaterialAvailTime": "10:00"}, routings())
check_("routing minutes = std x qty", [op.minutes for op in o.ops], [200, 50, 120, 110, 30])
check_("due and release in plant minutes", (o.due, o.release), (DAY + 8 * 60, 4 * 60))
check_("slower machine takes longer", plant.duration(o.ops[0], "CNC-03"), 250)

# --- baseline, solver and the independent checker ------------------------------------------
p = Problem(plant)
edd = dispatch_edd(p)
check_("EDD is feasible", check(plant, edd), [])
r = solve(p, time_limit=15)
check_("CP-SAT is feasible", check(plant, r.schedule), [])
k_edd, k_cp = kpis(plant, edd), kpis(plant, r.schedule)
check_("CP-SAT never worse than EDD on weighted lateness", k_cp["weighted_tardiness_h"] <= k_edd["weighted_tardiness_h"], True)
check_("CP-SAT at least as many on-time orders", k_cp["on_time"] >= k_edd["on_time"], True)

bad = copy.deepcopy(r.schedule)
a, b = sorted((k for k, v in bad.items() if v["machine"] == "LKT-04"), key=lambda k: bad[k]["start"])[:2]
bad[b]["start"], bad[b]["end"] = bad[a]["start"], bad[a]["start"] + (bad[b]["end"] - bad[b]["start"])
check_("checker catches overlap", any("overlap" in v for v in check(plant, bad)), True)
bad = copy.deepcopy(r.schedule)
first = next(iter(plant.orders.values())).ops
bad[first[1].id]["start"] = bad[first[0].id]["start"]
check_("checker catches routing order", any("previous operation" in v for v in check(plant, bad)), True)
bad = copy.deepcopy(r.schedule)
k0 = next(iter(bad))
d = bad[k0]["end"] - bad[k0]["start"]
bad[k0]["start"], bad[k0]["end"] = 16 * 60 + 10, 16 * 60 + 10 + d
check_("checker catches night work", any("blocked" in v for v in check(plant, bad)), True)

# --- planning state: clock, events, proposals, versions --------------------------------------
pl = Planning(load_plant())
v1 = pl.full_plan()
check_("full plan versioned", (v1.id, v1.approver), (1, "system"))
adv = pl.advance(180)
check_("clock advances; MES confirmations recorded", (adv["now"], adv["confirmed"] == len(pl.done)), (180, True))
frozen_before = pl.frozen()

prop = pl.propose({"kind": "machine_down", "machine": "LKT-04", "hours": 6})
check_("machine-down repair is feasible", prop.violations, [])
lkt = [a for a in prop.schedule.values() if a["machine"] == "LKT-04"]
check_("nothing runs on LKT-04 while it is down", any(a["start"] < 180 + 360 and a["end"] > 180 for a in lkt), False)
kept = [k for k, f in frozen_before.items() if f["machine"] != "LKT-04" and k in prop.fixed]
check_("frozen work elsewhere does not move",
       all((prop.schedule[k]["start"], prop.schedule[k]["machine"]) == (frozen_before[k]["start"], frozen_before[k]["machine"]) for k in kept), True)
v2 = pl.decide(prop.id, True, "planner-a")
check_("approval creates version 2", (v2.id, pl.proposal), (2, None))

try:
    pl.propose({"kind": "machine_down", "machine": "CNC-99", "hours": 2})
    check_("unknown machine rejected", False, True)
except EventError:
    check_("unknown machine rejected", True, True)
try:
    pl.propose({"kind": "quality_hold", "order": "10004405"})
    check_("missing field rejected", False, True)
except EventError as e:
    check_("missing field rejected", "hours" in str(e), True)

prop = pl.propose({"kind": "rush_order", "material": "CVR-1100", "quantity": 60, "due_hours": 20})
rush = prop.event["order"]
check_("rush order added with priority 3", (rush in prop.plant.orders, prop.plant.orders[rush].weight), (True, 3))
check_("rush repair feasible", prop.violations, [])
pl.decide(prop.id, False, "planner-a")
check_("rejected proposal leaves the plan unchanged", rush in pl.plant.orders, False)

prop = pl.propose({"kind": "quality_hold", "order": "10004420", "hours": 8})
held = [prop.schedule[op.id] for op in prop.plant.orders["10004420"].ops if op.id in prop.schedule and op.id not in prop.fixed]
check_("held order waits for the hold", all(a["start"] >= pl.now + 480 for a in held), True)
check_("hold repair feasible", prop.violations, [])
pl.decide(prop.id, True, "planner-b")
check_("dispatch list per machine", sorted(pl.dispatch_lists()) == sorted(pl.plant.machines), True)
check_("SAP write-back row per unfinished order", len(pl.sap_updates()) >= 30, True)

# --- inbox: extraction rules and the mock classifier ------------------------------------------
fields = {s["id"]: inbox.extract(s["text"], plant.machines, plant.orders) for s in inbox.SAMPLES}
check_("extract machine + hours", fields["m1"], {"machine": "LKT-04", "hours": 6.0})
check_("extract zh machine + percent", fields["m2"], {"machine": "CNC-02", "percent": 30.0})
check_("extract de order + Std.", fields["m3"], {"order": "10004430", "hours": 24.0})
check_("extract material, qty, hours", fields["m4"], {"material": "HSG-7731", "hours": 30.0, "quantity": 80})
check_("extract zh order + 小时", fields["m5"], {"order": "10004405", "hours": 8.0})
check_("unknown ids are ignored", inbox.extract("CNC-77 and order 10009999", plant.machines, plant.orders), {})
mock = inbox.MockClassifier()
got = {s["id"]: inbox.read_message(mock, s["text"], plant.machines, plant.orders) for s in inbox.SAMPLES}
check_("mock reads every sample as expected", all(got[s["id"]]["event_type"] == s["expect"] for s in inbox.SAMPLES), True)
check_("complete event needs no planner", (got["m1"]["event"], got["m1"]["needs_planner"]),
       ({"kind": "machine_down", "machine": "LKT-04", "hours": 6.0}, False))
r2 = inbox.read_message(mock, "CNC-01 is broken", plant.machines, plant.orders)
check_("missing hours goes to the planner", (r2["missing"], r2["needs_planner"]), (["hours"], True))


class Unsure(inbox.MockClassifier):
    def classify(self, text):
        out = super().classify(text)
        out["answers"]["event_type"]["confidence"] = 0.3
        return out


check_("below the gate goes to the planner", inbox.read_message(Unsure(), inbox.SAMPLES[0]["text"], plant.machines, plant.orders)["needs_planner"], True)

# --- API ----------------------------------------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402

server.start(mock=True, plan_on_start=False)
server.A.pl.full_plan()
c = TestClient(server.app)
st = c.get("/api/state").json()
check_("state has ops, orders, machines", (len(st["ops"]), len(st["orders"]), len(st["machines"])), (150, 36, 8))
rd = c.post("/api/inbox/read", json={"text": inbox.SAMPLES[4]["text"]}).json()
check_("inbox endpoint builds the event", rd["event"], {"kind": "quality_hold", "order": "10004405", "hours": 8.0})
check_("bad event -> 400", c.post("/api/events", json={"kind": "machine_down", "machine": "X"}).status_code, 400)
check_("event accepted -> 202", c.post("/api/events", json=rd["event"]).status_code, 202)
check_("second solve while running -> 409", c.post("/api/plan").status_code, 409)
deadline = time.time() + 60
while time.time() < deadline and c.get("/api/state").json()["job"]["running"]:
    time.sleep(0.5)
st = c.get("/api/state").json()
check_("proposal ready with no violations", (st["proposal"] is not None, st["proposal"]["violations"] if st["proposal"] else None), (True, []))
pid = st["proposal"]["id"]
check_("clock blocked while a proposal is open", c.post("/api/clock", json={"minutes": 60}).status_code, 409)
check_("approve", c.post("/api/proposal/%d/decide" % pid, json={"approve": True, "who": "qa"}).json()["version"], 2)
pub = c.get("/api/publish").json()
check_("publish preview", (pub["version"], len(pub["dispatch"])), (2, 8))

print("%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("FAIL", f)
sys.exit(1 if FAIL else 0)
