"""Measure the APS demo end to end with default time limits and the real Laya checkpoints.

    python demos/aps/evaluate.py            # about 5 minutes on a laptop CPU
    python demos/aps/evaluate.py --mock     # classifier stand-in
    python demos/aps/evaluate.py --laya-only   # re-measure the language side, keep the solver rows

Writes data/eval_results.json: full plan vs the due-date rule, each disruption's repair
from the same approved plan (time, status, KPIs before and after, operations moved,
feasibility), how Laya read every sample message (guard, two readings, gate, facts) and
every planner command (intent). The design document reports these.
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
    {"kind": "machine_down", "machine": "CNC-02", "hours": 4},
    {"kind": "maintenance", "machine": "CNC-01", "start_hours": 20, "hours": 3},
    {"kind": "order_cancel", "order": "10004409"},
    {"kind": "due_change", "order": "10004404", "shift_hours": -48},
    {"kind": "quantity_change", "order": "10004413", "quantity": 60},
    {"kind": "priority_change", "order": "10004403", "priority": 3},
    # the follow-ups a rejection offers, for the first event
    {"kind": "machine_down", "machine": "LKT-04", "hours": 6, "narrow": True},
    {"kind": "machine_down", "machine": "LKT-04", "hours": 6, "protect": ["10004405"]},
]
KEYS = ("otd", "on_time", "late_orders", "weighted_tardiness_h", "setup_h")


def pick(k):
    return {x: k[x] for x in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--laya-only", action="store_true", help="keep the solver results already in the file")
    args = ap.parse_args()
    path = os.path.join(HERE, "data", "eval_results.json")
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "machine": platform.machine(), "processor": platform.processor(),
           "cpus": os.cpu_count(), "limits_s": {"full": FULL_LIMIT_S, "repair": REPAIR_LIMIT_S}}

    pl = Planning(load_plant())
    if args.laya_only:
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
        out.update({k: prev[k] for k in ("full_plan", "repairs", "limits_s")})
        run_laya(args, out, pl, path)
        return
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
    run_laya(args, out, pl, path)


def run_laya(args, out, pl, path):
    clf = inbox.MockClassifier() if args.mock else inbox.LayaClassifier()
    if hasattr(clf, "warm"):
        clf.warm()
    rows = []
    for s in inbox.SAMPLES:
        r = inbox.read_message(clf, s["text"], pl.plant, pl.now)
        flagged = r["guard"]["flagged"]
        # an injection is caught when the guard flags it; any other message is read correctly
        # when the event kind is right (a false guard alarm only sends it to the planner)
        correct = flagged if s["expect"] == "flagged" else r["event_type"] == s["expect"]
        auto = not r["needs_planner"]
        # the event that would apply itself must also be the right event with the right facts
        rows.append({"id": s["id"], "lang": s["lang"], "expect": s["expect"], "got": r["event_type"], "flagged": flagged,
                     "confidence": round(r["confidence"], 3), "second": r["second_reading"]["event_type"],
                     "second_confidence": round(r["second_reading"]["confidence"], 3), "agree": r["agree"],
                     "guard": {k: round(v, 3) for k, v in r["guard"]["scores"].items()},
                     "urgency": r["urgency"], "reasons": r["reasons"], "event": r.get("event"),
                     "model": r["classification"]["routing"].get("model"), "ms": r["classification"]["ms"],
                     "auto": auto, "correct": correct,
                     # one reading alone, gated at 50 %: what the old single-question design would do
                     "single_1_wrong_confident": (s["expect"] != "flagged" and r["event_type"] != s["expect"] and r["confidence"] >= inbox.GATE),
                     "single_2_wrong_confident": (s["expect"] != "flagged" and r["second_reading"]["event_type"] != s["expect"]
                                                  and r["second_reading"]["confidence"] >= inbox.GATE)})
        print(s["id"], s["lang"], s["expect"], "->", rows[-1]["got"] + (" [flagged]" if flagged else ""), round(r["confidence"], 2), "/", r["second_reading"]["event_type"],
              round(r["second_reading"]["confidence"], 2), "auto" if auto else "planner:" + ",".join(r["reasons"]))
    out["inbox"] = {"engine": clf.info()["kind"], "rows": rows,
                    "correct": sum(r["correct"] for r in rows), "total": len(rows),
                    "auto": sum(r["auto"] for r in rows),
                    "auto_correct": sum(1 for r in rows if r["auto"] and r["correct"]),
                    "confident_errors": sum(1 for r in rows if r["auto"] and not r["correct"]),
                    "single_reading_confident_errors": [sum(r["single_1_wrong_confident"] for r in rows),
                                                        sum(r["single_2_wrong_confident"] for r in rows)],
                    "guard_false_alarms": sum(1 for r in rows if r["expect"] != "flagged" and r["flagged"]),
                    "injections_caught": sum(1 for r in rows if r["expect"] == "flagged" and r["flagged"])}
    cmds = []
    for s in inbox.COMMAND_SAMPLES:
        r = inbox.read_message(clf, s["text"], pl.plant, pl.now, command=True)
        cmds.append({"text": s["text"], "lang": s["lang"], "expect": s["expect"], "got": r["intent"],
                     "confidence": round(r["intent_confidence"], 3), "event": r.get("event"),
                     "correct": r["intent"] == s["expect"], "gated": r["intent_confidence"] >= inbox.GATE})
        print("cmd", s["lang"], s["expect"], "->", r["intent"], round(r["intent_confidence"], 2), r.get("event"))
    sets = json.load(open(os.path.join(HERE, "data", "eval_sets.json"), encoding="utf-8"))
    cue_rows = [{"id": m["id"], "lang": m["lang"], "hedged": m["hedged"], "consequence": m["consequence"],
                 "got": inbox.cues(m["text"])} for m in sets["cues_test"]]
    out["cues"] = {"total": len(cue_rows),
                   "hedged_right": sum(r["got"]["hedged"] == r["hedged"] for r in cue_rows),
                   "consequence_right": sum(r["got"]["consequence"] == r["consequence"] for r in cue_rows),
                   "misses": [r["id"] for r in cue_rows if r["got"]["hedged"] != r["hedged"] or r["got"]["consequence"] != r["consequence"]]}
    print("cues", {k: v for k, v in out["cues"].items()})
    for name, setname, key in (("rejections", "reject_test", "reason"), ("notes", "notes_test", "condition")):
        rows = []
        for m in sets[setname]:
            if name == "rejections":
                r = inbox.read_rejection(clf, m["text"], None, pl.plant)
                rows.append({"id": m["id"], "lang": m["lang"], "expect": m[key], "got": r["reason"],
                             "confidence": round(r["confidence"], 3), "confident": r["confident"]})
            else:
                r = inbox.read_note(clf, m["text"], pl.plant)
                rows.append({"id": m["id"], "lang": m["lang"], "expect": m[key], "got": r["condition"],
                             "p_problem": r["p_problem"], "flagged": r["flagged"], "machine": r["machine"]})
        if name == "rejections":
            summ = {"right": sum(r["got"] == r["expect"] for r in rows), "total": len(rows),
                    "confident": sum(r["confident"] for r in rows),
                    "confident_wrong": [r["id"] for r in rows if r["confident"] and r["got"] != r["expect"]]}
        else:
            bad = [r for r in rows if r["expect"] != "ok"]
            summ = {"right": sum(r["got"] == r["expect"] for r in rows), "total": len(rows),
                    "flagged": sum(r["flagged"] for r in rows),
                    "false_flags": [r["id"] for r in rows if r["flagged"] and r["expect"] == "ok"],
                    "stops_caught": sum(1 for r in bad if r["expect"] == "stop" and r["flagged"]),
                    "stops": sum(1 for r in bad if r["expect"] == "stop"),
                    "watch_caught": sum(1 for r in bad if r["expect"] == "watch" and r["flagged"]),
                    "watch": sum(1 for r in bad if r["expect"] == "watch"),
                    "machine_found": sum(1 for r in rows if r["machine"])}
        out[name] = {"rows": rows, **summ}
        print(name, summ)
    out["commands"] = {"rows": cmds, "correct": sum(c["correct"] for c in cmds), "total": len(cmds),
                       "confident_errors": sum(1 for c in cmds if c["gated"] and not c["correct"])}
    print("commands %(correct)d/%(total)d, %(confident_errors)d confident errors" % out["commands"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("inbox %(correct)d/%(total)d correct, %(auto)d auto (%(auto_correct)d right), %(confident_errors)d confident errors, "
          "single readings alone %(single_reading_confident_errors)s, injections %(injections_caught)d, false alarms %(guard_false_alarms)d" % out["inbox"])
    print("wrote", path)


if __name__ == "__main__":
    main()
