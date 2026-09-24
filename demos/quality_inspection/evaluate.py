"""Acceptance run with the real Laya checkpoints: every scenario through the full pipeline.

    python demos/quality_inspection/evaluate.py            # a minute or two on a laptop CPU
    python demos/quality_inspection/evaluate.py --mock

Prints what Laya answered and what the policy decided for each record, and checks the
outcome each scenario is built to produce. Run it after changing the questions, the
note handling or a threshold - it is the regression test for the parts that a model
decides.
"""
import argparse
import collections
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qi.config import SPC_WINDOW, STATIONS  # noqa: E402
from qi.gateway import BENIGN_NOTES, LineSimulator  # noqa: E402
from qi.inspection import decide, note_text  # noqa: E402
from qi.rules import check_limits, spc_signals  # noqa: E402

# scenario -> (units to run, {station: expected disposition(s) on the affected units})
EXPECT = {
    None: (3, {sid: {"pass"} for sid in STATIONS}),
    "torque_gun": (3, {"ASM-03": {"rework", "scrap"}}),
    "bad_lot": (2, {"LKT-04": {"scrap"}}),
    "gauge_drift": (2, {"CMM-01": {"hold"}}),
    "tray_scratch": (2, {"AOI-01": {"rework"}}),
    "operator_dent": (1, {"AOI-01": {"hold"}}),
    "sensor_dropout": (1, {"LKT-04": {"hold"}}),
    "tool_wear": (10, {}),                  # judged on alerts below, not per unit
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    if args.mock:
        from qi.engine import MockEngine
        engine = MockEngine()
    else:
        from qi.engine import LayaEngine
        engine = LayaEngine(args.device)
        engine.warm()
        if engine.state != "ready":
            sys.exit("Laya failed to load: %s" % engine.error)

    history = collections.defaultdict(lambda: collections.deque(maxlen=SPC_WINDOW))
    failures, lat = [], []
    stop_seen_at = None

    for scenario, (units, expect) in EXPECT.items():
        sim = LineSimulator(seed=args.seed)
        if scenario:
            sim.inject(scenario)
        affected = scenario and {sid for sid in expect} or set()
        print("\n== %s" % (scenario or "normal production"))
        for _ in range(units * len(STATIONS)):
            msg = sim.next_message()
            sid = msg["station"]
            if scenario and sid not in affected and scenario != "tool_wear":
                continue                         # only the station the fault lives on
            if scenario == "tool_wear" and sid != "CNC-02":
                continue
            st = STATIONS[sid]
            chars = st["characteristics"]
            findings = check_limits(chars, msg["measurements"])
            hist = {}
            for k, spec in chars.items():
                if spec.get("spc") and msg["measurements"].get(k) is not None:
                    history[(scenario, sid, k)].append(msg["measurements"][k])
                    hist[k] = list(history[(scenario, sid, k)])
            spc = spc_signals(chars, hist)
            note = note_text(msg)
            a, laya = None, "-"
            if note:
                t0 = time.time()
                res = engine.predict(note)
                lat.append((time.time() - t0) * 1000)
                a = res["answers"]
                laya = "%-19s %.2f  %-13s %.2f  %s" % (
                    a["topic"]["choice"], a["topic"]["confidence"],
                    a["defect_kind"]["choice"], a["defect_kind"]["confidence"], res["routing"]["model"])
            d = decide(findings, spc, a, msg, chars)
            print("  %-7s %-6s %-4s %-5s %-6s | %s | %s" % (
                sid, d["disposition"], "auto" if d["auto"] else "HOLD",
                (d["alert"] or {}).get("level", "-"), d["root_cause"] or "-", laya, note[:48]))
            want = expect.get(sid)
            if want and d["disposition"] not in want:
                failures.append("%s %s: %s, expected %s" % (scenario or "normal", sid, d["disposition"], "/".join(sorted(want))))
                for line in d["trace"]:
                    print("      " + line)
            if scenario == "tool_wear" and d["alert"] and d["alert"]["level"] == "stop" and stop_seen_at is None:
                stop_seen_at = msg["measurements"]["bore_d"]

    # Benign shift notes on a good part: every one Laya is unsure about costs a person a look.
    print("\n== benign notes on a good part")
    clean = check_limits(STATIONS["AOI-01"]["characteristics"], {"defect_score": 0.04})
    holds = 0
    for lang, text in BENIGN_NOTES:
        msg = {"alarms": [], "operator_note": {"lang": lang, "text": text}}
        res = engine.predict(text)
        a = res["answers"]
        d = decide(clean, [], a, msg, STATIONS["AOI-01"]["characteristics"])
        holds += d["disposition"] == "hold"
        print("  %-3s %-5s %-19s %.2f  %-12s | %s" % (lang, d["disposition"], a["topic"]["choice"],
              a["topic"]["confidence"], res["routing"]["model"], text))
    print("  false holds: %d of %d" % (holds, len(BENIGN_NOTES)))

    if stop_seen_at is None:
        failures.append("tool_wear: no line stop raised")
    else:
        print("\ntool_wear: line stop first raised at bore %.4f mm (USL 42.050)" % stop_seen_at)
    lat.sort()
    if lat:
        print("\nLaya latency p50 %.0f ms, p95 %.0f ms over %d notes (%s)" % (
            lat[len(lat) // 2], lat[min(len(lat) - 1, int(len(lat) * 0.95))], len(lat), engine.info()["device"]))
    print("%d expectation(s) failed" % len(failures))
    for f in failures:
        print("FAIL", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
