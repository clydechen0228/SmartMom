"""Measure the APS demo end to end with default time limits and the real Laya checkpoints.

    python demos/aps/evaluate.py            # about 5 minutes on a laptop CPU
    python demos/aps/evaluate.py --mock     # classifier stand-in

Writes data/eval_results.json: full plan vs the due-date rule, each disruption's repair
from the same approved plan (time, status, KPIs before and after, operations moved,
feasibility), and how Laya read every sample message. The design document reports these.
"""
import argparse
import copy
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from aps import inbox  # noqa: E402
from aps.plant import load_plant  # noqa: E402
from aps.state import FULL_LIMIT_S, REPAIR_LIMIT_S, Planning  # noqa: E402

SCENARIO = [
    {"kind": "machine_down", "machine": "LKT-04", "hours": 6},
    {"kind": "rush_order", "material": "HSG-7731", "quantity": 80, "due_hours": 30},
    {"kind": "quality_hold", "order": "10004405", "hours": 8},
    {"kind": "machine_degrading", "machine": "CNC-02", "percent": 30},
    {"kind": "material_late", "order": "10004430", "hours": 24},
]
KEYS = ("otd", "on_time", "late_orders", "weighted_tardiness_h", "setup_h")


def pick(k):
    return {x: k[x] for x in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "machine": platform.machine(), "processor": platform.processor(),
           "cpus": os.cpu_count(), "limits_s": {"full": FULL_LIMIT_S, "repair": REPAIR_LIMIT_S}}

    pl = Planning(load_plant())
    v = pl.full_plan()
    out["full_plan"] = {"cp_sat": pick(v.kpis), "edd": pick(v.kpis["edd_baseline"]),
                        "seconds": v.kpis["solve_s"], "status": v.kpis["status"],
                        "orders": v.kpis["orders"], "operations": len(pl.schedule)}
    print("full plan", out["full_plan"])

    # Each disruption is repaired from the same approved plan at Mon 09:00, so the numbers
    # compare events, not the order they happened to arrive in.
    pl.advance(180)
    base = copy.deepcopy(pl)
    reps = []
    for ev in SCENARIO:
        trial = copy.deepcopy(base)
        p = trial.propose(ev)
        row = {"event": ev["kind"], "detail": {k: v for k, v in ev.items() if k != "kind"},
               "seconds": p.seconds, "status": p.status, "before": pick(p.before), "after": pick(p.kpis),
               "moved": p.kpis["moved_ops"], "affected": trial.events[-1]["affected"],
               "violations": len(p.violations)}
        reps.append(row)
        print(ev["kind"], row["seconds"], "s", row["status"], "otd", row["before"]["otd"], "->", row["after"]["otd"],
              "wt", row["before"]["weighted_tardiness_h"], "->", row["after"]["weighted_tardiness_h"],
              "moved", row["moved"], "affected", row["affected"], "violations", row["violations"])
    out["repairs"] = reps

    clf = inbox.MockClassifier() if args.mock else inbox.LayaClassifier()
    if hasattr(clf, "warm"):
        clf.warm()
    rows = []
    for s in inbox.SAMPLES:
        r = inbox.read_message(clf, s["text"], pl.plant.machines, pl.plant.orders)
        rows.append({"id": s["id"], "lang": s["lang"], "expect": s["expect"], "got": r["event_type"],
                     "confidence": round(r["confidence"], 3), "model": r["classification"]["routing"].get("model"),
                     "ms": r["classification"]["ms"], "planner": r["needs_planner"],
                     "correct": r["event_type"] == s["expect"]})
        print(s["id"], s["lang"], s["expect"], "->", r["event_type"], round(r["confidence"], 2), "planner" if r["needs_planner"] else "auto")
    out["inbox"] = {"engine": clf.info()["kind"], "rows": rows,
                    "correct": sum(r["correct"] for r in rows), "total": len(rows),
                    "confident_errors": sum(1 for r in rows if not r["correct"] and not r["planner"])}
    path = os.path.join(HERE, "data", "eval_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("inbox %(correct)d/%(total)d correct, %(confident_errors)d confident errors" % out["inbox"])
    print("wrote", path)


if __name__ == "__main__":
    main()
