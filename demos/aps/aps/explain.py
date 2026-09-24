"""Why an order is late or at risk, and how loaded each work centre is: from the schedule.

No model and no generated text: every minute between an operation becoming ready and it
starting is attributed to a cause the schedule can show (another order on the machine,
the unmanned night, downtime or maintenance, a quality hold, material not yet there),
and the causes are ranked by minutes. Laya's job ends at understanding the question.
"""
from typing import Dict, List, Optional

from .plant import DAY, HORIZON_DAYS, SHIFT_END, SHIFT_START, clock_label

CAUSE_TEXT = {
    "machine_busy": ("waiting for %s, busy with other orders", "等待 %s（被其他订单占用）"),
    "night": ("unmanned nights", "夜间无人值守"),
    "downtime": ("downtime or maintenance on %s", "%s 停机或保养"),
    "hold": ("quality hold", "质量冻结"),
    "material": ("material not yet available", "物料尚未到货"),
    "setup": ("setup changes on %s", "%s 换型"),
    "process": ("processing time", "加工时间"),
}


def _night(t: int) -> bool:
    r = t % DAY
    return r >= SHIFT_END or r < SHIFT_START


def _overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def explain(plant, allops: Dict[str, dict], order_id: str, holds: Optional[Dict[str, int]] = None,
            t0: int = 0) -> dict:
    """Minutes of delay by cause for one order, biggest first, in English and Chinese."""
    holds = holds or {}
    o = plant.orders.get(order_id)
    if o is None:
        return {"order": order_id, "found": False}
    ops = [(op, allops[op.id]) for op in o.ops if op.id in allops]
    if not ops:
        return {"order": order_id, "found": False}
    by_machine: Dict[str, List[tuple]] = {}
    for k, a in allops.items():
        by_machine.setdefault(a["machine"], []).append((a["start"], a["end"], k))
    causes: Dict[tuple, int] = {}

    def add(key, minutes):
        if minutes > 0:
            causes[key] = causes.get(key, 0) + minutes

    prev_end = None
    for i, (op, a) in enumerate(ops):
        if i == 0:
            # from the start of the plan until the material arrives, the order cannot start
            add(("material", None), max(0, min(a["start"], o.release) - t0))
            ready = min(a["start"], max(t0, o.release))
        else:
            ready = prev_end
        if o.id in holds:
            add(("hold", None), _overlap(ready, a["start"], ready, holds[o.id]))
        wait0, wait1 = ready, a["start"] - a.get("setup", 0)
        if wait1 > wait0:
            night = sum(1 for t in range(wait0, wait1, 15) if _night(t)) * 15
            blocked = sum(_overlap(wait0, wait1, b0, b1) for b0, b1 in plant.blocked.get(a["machine"], [])
                          if (b1 - b0) != DAY - SHIFT_END + SHIFT_START)
            busy = sum(_overlap(wait0, wait1, s0, e0) for s0, e0, k in by_machine.get(a["machine"], []) if k != op.id)
            add(("night", None), night)
            add(("downtime", a["machine"]), blocked)
            add(("machine_busy", a["machine"]), max(0, min(busy, wait1 - wait0 - night - blocked)))
        add(("setup", a["machine"]), a.get("setup", 0))
        add(("process", None), a["end"] - a["start"])
        prev_end = a["end"]

    finish = ops[-1][1]["end"]
    ranked = sorted(((k, v) for k, v in causes.items() if k[0] != "process"), key=lambda kv: -kv[1])
    items = []
    for (kind, target), minutes in ranked:
        en, zh = CAUSE_TEXT[kind]
        items.append({"cause": kind, "target": target, "hours": round(minutes / 60, 1),
                      "en": en % target if "%s" in en else en, "zh": zh % target if "%s" in zh else zh})
    late_min = finish - o.due
    status = "late" if late_min > 0 else ("risk" if late_min > -240 else "ok")
    top = items[0] if items else None
    if status == "ok":
        summary_en = "On time with %.1f h to spare." % (-late_min / 60)
        summary_zh = "按时完成，余量 %.1f 小时。" % (-late_min / 60)
    else:
        lead_en = "%.1f h late" % (late_min / 60) if late_min > 0 else "only %.1f h of slack" % (-late_min / 60)
        lead_zh = "延误 %.1f 小时" % (late_min / 60) if late_min > 0 else "余量仅 %.1f 小时" % (-late_min / 60)
        summary_en = "%s; biggest cause: %s (%.1f h)." % (lead_en.capitalize(), top["en"], top["hours"]) if top else lead_en + "."
        summary_zh = "%s；主要原因：%s（%.1f 小时）。" % (lead_zh, top["zh"], top["hours"]) if top else lead_zh + "。"
    return {"order": o.id, "found": True, "status": status, "due": o.due, "due_label": clock_label(o.due),
            "finish": finish, "finish_label": clock_label(finish), "late_h": round(late_min / 60, 1),
            "process_h": round(causes.get(("process", None), 0) / 60, 1), "causes": items,
            "summary_en": summary_en, "summary_zh": summary_zh}


def load_by_day(plant, allops: Dict[str, dict], days: int = HORIZON_DAYS) -> Dict[str, List[float]]:
    """Busy share of each work centre's shift time, per day of the horizon."""
    out = {}
    for wc, ms in plant.work_centers.items():
        row = []
        for d in range(days):
            d0, d1 = d * DAY + SHIFT_START, d * DAY + SHIFT_END
            busy = sum(_overlap(a["start"], a["end"], d0, d1) for a in allops.values() if a["machine"] in ms)
            row.append(round(busy / ((d1 - d0) * len(ms)), 3))
        out[wc] = row
    return out
