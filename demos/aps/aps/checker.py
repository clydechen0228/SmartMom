"""Independent feasibility check of a schedule. Shares no code with the solver on purpose:
a bug in the model should not be able to hide itself here."""
from typing import Dict, List


def check(plant, schedule: Dict[str, dict], now: int = 0, done: Dict[str, dict] = None,
          fixed: Dict[str, dict] = None, holds: Dict[str, int] = None, blocked: Dict[str, list] = None) -> List[str]:
    done, fixed, holds = done or {}, fixed or {}, holds or {}
    blocked = blocked if blocked is not None else plant.blocked
    allops = {**done, **schedule}
    bad: List[str] = []
    for o in plant.orders.values():
        prev_end = None
        for op in o.ops:
            a = allops.get(op.id)
            if a is None:
                bad.append("%s not scheduled" % op.id)
                continue
            if a["machine"] not in plant.work_centers[op.work_center]:
                bad.append("%s on ineligible machine %s" % (op.id, a["machine"]))
            if op.id not in done and op.id not in fixed:
                want = max(1, int(round(op.minutes / plant.machines[a["machine"]].speed)))
                if a["end"] - a["start"] != want:
                    bad.append("%s lasts %d min, expected %d" % (op.id, a["end"] - a["start"], want))
                if a["start"] < now:
                    bad.append("%s starts before now" % op.id)
                if o.id in holds and a["start"] < holds[o.id]:
                    bad.append("%s starts during the quality hold" % op.id)
            if prev_end is not None and a["start"] < prev_end:
                bad.append("%s starts before its previous operation ends" % op.id)
            if op is o.ops[0] and op.id not in done and op.id not in fixed and a["start"] < o.release:
                bad.append("%s starts before material is available" % op.id)
            prev_end = a["end"]
    for op_id, f in fixed.items():
        a = schedule.get(op_id)
        if a and (a["machine"], a["start"], a["end"]) != (f["machine"], f["start"], f["end"]):
            bad.append("%s is frozen but moved" % op_id)
    by_m: Dict[str, list] = {}
    for op_id, a in allops.items():
        by_m.setdefault(a["machine"], []).append((a["start"], a["end"], op_id, a.get("setup", 0)))
    for mc, items in by_m.items():
        items.sort()
        for (s0, e0, i0, _), (s1, e1, i1, su1) in zip(items, items[1:]):
            if s1 < e0 + (su1 or 0):
                bad.append("%s and %s overlap on %s (incl. %d min setup)" % (i0, i1, mc, su1 or 0))
        for s0, e0, i0, _ in items:
            if i0 in done or i0 in fixed:
                continue
            for b0, b1 in blocked.get(mc, []):
                if s0 < b1 and e0 > b0:
                    bad.append("%s on %s runs into blocked time" % (i0, mc))
    return bad
