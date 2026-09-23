"""Offline tests for the quality inspection demo. No checkpoint and no Hub download.

Rules, SPC and the policy are pure functions; the API runs against the mock engine and a
temporary database, with the in-process gateway switched off.

    python demos/quality_inspection/test_quality_inspection.py
"""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qi.config import STATIONS  # noqa: E402
from qi.gateway import SCENARIOS, LineSimulator  # noqa: E402
from qi.inspection import build_report, decide, note_text  # noqa: E402
from qi.rules import check_limits, spc_signals, western_electric  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def answers(topic="nothing", t_conf=0.9, kind="none", k_conf=0.9):
    return {"topic": {"choice": topic, "confidence": t_conf},
            "defect_kind": {"choice": kind, "confidence": k_conf}}


# --- limits ----------------------------------------------------------------------------
cnc = STATIONS["CNC-02"]["characteristics"]
f = {x["key"]: x for x in check_limits(cnc, {"bore_d": 42.071, "spindle_vib": 3.0})}
check("bore above usl is out", f["bore_d"]["status"], "out")
check("excess is signed distance past the limit", round(f["bore_d"]["excess"], 3), 0.021)
check("process param in limit", f["spindle_vib"]["status"], "ok")
f = {x["key"]: x for x in check_limits(cnc, {"bore_d": 41.949})}
check("below lsl is out", f["bore_d"]["status"], "out")
check("absent reading is missing", f["spindle_vib"]["status"], "missing")
f = {x["key"]: x for x in check_limits(cnc, {"bore_d": 42.050, "spindle_vib": 6.0})}
check("value on the limit is in tolerance", (f["bore_d"]["status"], f["spindle_vib"]["status"]), ("ok", "ok"))

# --- SPC -------------------------------------------------------------------------------
check("in control -> no signal", western_electric([0.1, -0.2, 0.3, -0.1, 0.0], 0, 1), None)
check("WE1 beyond 3 sigma", western_electric([0, 0, 3.2], 0, 1)["rule"], "WE1")
check("WE2 two of three beyond 2 sigma", western_electric([2.2, 0.1, 2.5], 0, 1)["rule"], "WE2")
check("WE2 needs the same side", western_electric([-2.2, 0.1, 2.5], 0, 1), None)
check("WE3 four of five beyond 1 sigma", western_electric([1.2, 1.5, 0.2, 1.1, 1.3], 0, 1)["rule"], "WE3")
check("WE4 eight on one side", western_electric([0.5, 0.2, 0.4, 0.1, 0.3, 0.6, 0.2, 0.1], 0, 1)["rule"], "WE4")
check("trend of six rising", western_electric([-0.9, -0.5, -0.2, 0.1, 0.4, 0.8], 0, 1)["rule"], "TREND")
check("five rising is not yet a trend", western_electric([-0.5, -0.2, 0.1, 0.4, 0.8], 0, 1), None)
sig = spc_signals(cnc, {"bore_d": [42.0, 42.001, 42.035]})
check("spc_signals reports per characteristic", [(s["key"], s["rule"]) for s in sig], [("bore_d", "WE1")])
check("non-SPC characteristics are skipped", spc_signals(cnc, {"spindle_vib": [99.0]}), [])

# --- report ----------------------------------------------------------------------------
msg = {"serial": "S1", "measurements": {"bore_d": 42.071, "spindle_vib": 7.9}, "alarms": ["SPINDLE_LOAD_HIGH"],
       "operator_note": {"lang": "zh", "text": "刀具磨损"}}
findings = check_limits(cnc, msg["measurements"])
rep = build_report("CNC-02", STATIONS["CNC-02"], "HSG-7731", msg, findings, sig)
check("report states the tolerance violation in words", "OUT OF TOLERANCE by 0.021 mm" in rep, True)
check("report states the process-limit violation", "OUT OF LIMIT by 1.9 mm/s" in rep, True)
check("report carries SPC, alarm and note",
      all(s in rep for s in ("Statistical process control", "Machine alarm: SPINDLE_LOAD_HIGH", "Operator note (zh): 刀具磨损")),
      True)
check("Laya reads the note alone", note_text(msg), "刀具磨损")
check("blank note is no note", note_text({"operator_note": {"text": "  "}}), "")

# --- policy ----------------------------------------------------------------------------
ok_f = check_limits(cnc, {"bore_d": 42.0, "spindle_vib": 3.0})
quiet = {"alarms": [], "operator_note": None}
noted = {"alarms": [], "operator_note": {"text": "small dent on the flange"}}

d = decide(ok_f, [], None, quiet, cnc)
check("clean, no note -> auto pass, no alert", (d["disposition"], d["auto"], d["alert"]), ("pass", True, None))
d = decide(ok_f, [], answers(), noted, cnc)
check("clean, note says nothing wrong -> pass", (d["disposition"], d["auto"]), ("pass", True))
d = decide(ok_f, [], answers("part_defect", kind="surface"), noted, cnc)
check("clean numbers, note reports damage -> hold", (d["disposition"], d["auto"], d["defect"]), ("hold", False, "surface"))
d = decide(ok_f, [], answers("part_defect", kind="surface", k_conf=0.2), noted, cnc)
check("unsure defect kind is recorded as unclassified", d["defect"], "unclassified")
d = decide(ok_f, [], answers("part_defect", t_conf=0.3), noted, cnc)
check("Laya unsure about a note -> hold", (d["disposition"], d["auto"]), ("hold", False))
d = decide(ok_f, [], None, noted, cnc, engine_ok=False)
check("note unread because engine down -> hold", (d["disposition"], d["auto"]), ("hold", False))
d = decide(ok_f, [], answers("material_problem"), noted, cnc)
check("suspect lot on a good part -> hold", (d["disposition"], d["root_cause"]), ("hold", "material"))
d = decide(ok_f, [], answers("machine_problem"), noted, cnc)
check("machine problem on a good part -> pass + watch",
      (d["disposition"], d["alert"]["level"], d["root_cause"]), ("pass", "watch", "machine"))

bad_f = check_limits(cnc, {"bore_d": 42.071, "spindle_vib": 3.0})
d = decide(bad_f, [], None, quiet, cnc)
check("out of tolerance -> reaction plan (scrap), auto", (d["disposition"], d["auto"], d["defect"]), ("scrap", True, "dimensional"))
d = decide(bad_f, [], answers("nothing", t_conf=0.99), noted, cnc)
check("a note cannot overrule a failed measurement", d["disposition"], "scrap")
d = decide(bad_f, [], answers("measurement_problem"), noted, cnc)
check("suspect gauge -> hold instead of scrap", (d["disposition"], d["auto"]), ("hold", False))
d = decide(bad_f, [], answers("measurement_problem", t_conf=0.3), noted, cnc)
check("unsure gauge doubt does not stop the reaction", d["disposition"], "scrap")

asm = STATIONS["ASM-03"]["characteristics"]
tq = check_limits(asm, {"torque_1": 10, "torque_2": 10, "torque_3": 8.2, "torque_4": 10})
d = decide(tq, [], None, {"alarms": ["NUTRUNNER_SPINDLE_3_NOK"]}, asm)
check("rework reaction, cause from the alarm table", (d["disposition"], d["root_cause"], d["alert"]["level"]),
      ("rework", "machine", "watch"))

miss = check_limits(STATIONS["LKT-04"]["characteristics"], {})
d = decide(miss, [], None, quiet, STATIONS["LKT-04"]["characteristics"])
check("missing reading -> hold", (d["disposition"], d["auto"]), ("hold", False))

trend = [{"key": "bore_d", "label": "Bore diameter", "rule": "TREND", "text": "..."}]
chatter = {"alarms": [], "operator_note": {"text": "chatter noise, tool worn"}}
d = decide(ok_f, trend, answers("machine_problem"), chatter, cnc)
check("SPC drift + operator reports machine problem -> stop", d["alert"]["level"], "stop")
d = decide(ok_f, trend, None, quiet, cnc)
check("SPC drift alone -> watch", d["alert"]["level"], "watch")
d = decide(ok_f, [], answers("machine_problem"), chatter, cnc)
check("words alone -> watch, never stop", d["alert"]["level"], "watch")
d = decide(ok_f, trend, answers("machine_problem", t_conf=0.3), chatter, cnc)
check("unsure words do not stop the line", d["alert"]["level"], "watch")

# --- simulator -------------------------------------------------------------------------
sim = LineSimulator(seed=7)
msgs = [sim.next_message() for _ in range(10)]
check("units walk the stations in order", [m["station"] for m in msgs[:5]], ["CNC-02", "CMM-01", "ASM-03", "LKT-04", "AOI-01"])
check("one serial per unit", len({m["serial"] for m in msgs}), 2)
seqs = [m["seq"] for m in msgs if m["device"] == "cnc-02-plc"]
check("device sequence strictly increases", seqs == sorted(set(seqs)), True)
for name, (station, units, _, _) in SCENARIOS.items():
    s = LineSimulator(seed=1)
    s.inject(name)
    for _ in range(5 * units):
        s.next_message()
    check("scenario %s ends after %d units" % (name, units), name in s.active, False)
s = LineSimulator(seed=1)
s.inject("tool_wear")
bores = [m["measurements"]["bore_d"] for m in (s.next_message() for _ in range(50)) if m["station"] == "CNC-02"]
check("tool wear drifts the bore out of tolerance", bores[7] > 42.05, True)

# --- API (mock engine, temp db, no gateway) ----------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402

tmp = tempfile.mkdtemp()
server.ARGS = type("A", (), {"db": os.path.join(tmp, "t.db"), "mock": True, "device": None,
                             "simulate": False, "interval": 1.0, "host": "127.0.0.1", "port": 1})()
KEY = {"Authorization": "Bearer demo-gateway-key"}
with TestClient(server.app) as c:
    sim = LineSimulator(seed=3)
    sim.inject("operator_dent")
    batch = {"gateway": "edge-gw-01", "messages": [sim.next_message() for _ in range(5)]}
    check("bad key -> 401", c.post("/api/v1/telemetry", json=batch, headers={"Authorization": "Bearer x"}).status_code, 401)
    r = c.post("/api/v1/telemetry", json=batch, headers=KEY)
    check("batch accepted", (r.status_code, r.json()["accepted"]), (202, 5))
    r = c.post("/api/v1/telemetry", json=batch, headers=KEY)
    check("resend is deduplicated", (r.json()["accepted"], r.json()["duplicates"]), (0, 5))
    wrong = dict(batch["messages"][0], device="lkt-04-tester", seq=1)
    r = c.post("/api/v1/telemetry", json={"gateway": "edge-gw-01", "messages": [wrong]}, headers=KEY)
    check("device/station mismatch rejected", len(r.json()["rejected"]), 1)

    deadline = time.time() + 10
    while time.time() < deadline and len(c.get("/api/inspections").json()) < 5:
        time.sleep(0.05)
    rows = c.get("/api/inspections").json()
    check("every reading inspected", len(rows), 5)
    aoi = next(r for r in rows if r["station"] == "AOI-01")
    check("dent note holds the AOI unit", (aoi["disposition"], aoi["status"]), ("hold", "pending"))
    detail = c.get("/api/inspections/%d" % aoi["id"]).json()
    check("detail keeps report, answers and trace",
          all(k in detail for k in ("report", "laya", "decision", "findings")) and bool(detail["decision"]["trace"]), True)
    r = c.post("/api/inspections/%d/review" % aoi["id"], json={"disposition": "rework", "reviewer": "qa1", "note": "dent confirmed"})
    check("review closes the hold", (r.status_code, r.json()["final"], r.json()["status"]), (200, "rework", "reviewed"))
    check("second review conflicts", c.post("/api/inspections/%d/review" % aoi["id"],
                                            json={"disposition": "pass", "reviewer": "qa1"}).status_code, 409)
    st = c.get("/api/status").json()["stats"]
    check("stats count the unit and the review", (st["units"], st["reviewed"], st["pending"], st["rework"]), (1, 1, 0, 1))
    check("only the note was sent to the engine", detail["laya"]["input"], batch["messages"][4]["operator_note"]["text"])
    check("line view has every station", len(c.get("/api/line").json()), 5)
    check("csv export", c.get("/api/export.csv").text.count("\n"), 6)

    # Store-and-forward: readings buffer while the service is unreachable, then drain in
    # order once it is back, and the resend of anything already delivered is dropped.
    from qi.gateway import EdgeGateway  # noqa: E402
    gw = EdgeGateway("http://127.0.0.1:9", sim=LineSimulator(seed=9))   # nothing listens on :9
    for _ in range(3):
        gw.buffer.append(gw.sim.next_message())
    check("link down -> flush fails, readings kept", (gw.flush(), len(gw.buffer), gw.last_error is not None), (False, 3, True))
    check("backoff scheduled", gw._next_try > time.time(), True)
    gw.client, gw.url = c, "/api/v1/telemetry"                           # link restored
    check("link back -> buffer drains", (gw.flush(), len(gw.buffer), gw.sent), (True, 0, 3))

# --- alert folding ------------------------------------------------------------------------
from qi.store import Store  # noqa: E402

st = Store(os.path.join(tmp, "a.db"))
a1 = st.raise_alert(1, "CNC-02", {"level": "stop", "reasons": ["r1"], "cause": "machine"})
a2 = st.raise_alert(2, "CNC-02", {"level": "stop", "reasons": ["r2"], "cause": None})
check("repeat alert folds into the open one", (a2["id"], a2["count"], a2["reasons"], a2["cause"]), (a1["id"], 2, ["r2"], "machine"))
check("other level opens its own", st.raise_alert(3, "CNC-02", {"level": "watch", "reasons": []})["id"] != a1["id"], True)
st.ack_alert(a1["id"])
check("acknowledged alert is not reused", st.raise_alert(4, "CNC-02", {"level": "stop", "reasons": []})["id"] != a1["id"], True)

print("%d passed, %d failed" % (len(PASS), len(FAIL)))
for line in FAIL:
    print("FAIL", line)
sys.exit(1 if FAIL else 0)
