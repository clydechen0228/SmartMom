"""KPIs for a schedule, as defined in the design (section 10)."""
from typing import Dict

from .plant import HORIZON_DAYS, SHIFT_END, clock_label

AT_RISK_SLACK = 4 * 60


def order_status(plant, allops: Dict[str, dict]) -> Dict[str, dict]:
    out = {}
    for o in plant.orders.values():
        ends = [allops[op.id]["end"] for op in o.ops if op.id in allops]
        finish = max(ends) if ends else None
        slack = None if finish is None else o.due - finish
        status = "late" if slack is not None and slack < 0 else ("risk" if slack is not None and slack < AT_RISK_SLACK else "ok")
        out[o.id] = {"order": o.id, "material": o.material, "quantity": o.quantity, "customer": o.customer,
                     "weight": o.weight, "due": o.due, "due_label": clock_label(o.due),
                     "release": o.release, "finish": finish,
                     "finish_label": clock_label(finish) if finish is not None else None,
                     "slack": slack, "status": status}
    return out


def kpis(plant, schedule: Dict[str, dict], done: Dict[str, dict] = None, previous: Dict[str, dict] = None) -> dict:
    allops = {**(done or {}), **schedule}
    st = order_status(plant, allops)
    n = len(st)
    late = [s for s in st.values() if s["status"] == "late"]
    wt = sum(plant.orders[s["order"]].weight * max(0, -s["slack"]) for s in late)
    setup = sum(a.get("setup", 0) for a in schedule.values())
    avail = HORIZON_DAYS * SHIFT_END
    util = {}
    for wc, ms in plant.work_centers.items():
        busy = 0
        for a in allops.values():
            if a["machine"] in ms:
                busy += max(0, min(a["end"], HORIZON_DAYS * 24 * 60) - a["start"])
        util[wc] = round(busy / (avail * len(ms)), 3)
    moved = None
    if previous:
        moved = sum(1 for k, a in schedule.items() if k in previous and
                    (a["machine"] != previous[k]["machine"] or abs(a["start"] - previous[k]["start"]) > 30))
    return {"orders": n, "on_time": n - len(late), "otd": round((n - len(late)) / n, 4) if n else None,
            "late_orders": len(late), "weighted_tardiness_h": round(wt / 60, 1),
            "setup_h": round(setup / 60, 1), "utilization": util,
            "bottleneck": max(util, key=util.get) if util else None, "moved_ops": moved}
