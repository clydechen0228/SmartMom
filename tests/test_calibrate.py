"""Per-bucket temperature fitting. No model weights are loaded.

Records are CPU tuples (qtype, logits, target, k). The fitter is the notebook's NLL+LBFGS
on log T, grouped by `temp_bucket`.
"""
import inspect
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent, load  # noqa: E402
from laya.calibrate import (  # noqa: E402
    TEMP_HI,
    TEMP_LO,
    apply_calibration_payload,
    calibration_payload,
    fit_one_temperature,
    fit_temperature_map,
    fit_temperatures,
    records_from_labeled,
)
from laya.common import QTYPES, temp_bucket  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def peaked(k, idx, mag):
    z = np.full(k, -mag, dtype=np.float32)
    z[idx] = mag
    t = np.zeros(k, dtype=np.float32)
    t[idx] = 1.0
    return z, t


def overconfident_records(n, k, qtype, seed=0, acc=0.6, mag=8.0):
    rng = np.random.RandomState(seed)
    recs = []
    for i in range(n):
        y = int(rng.randint(0, k))
        pred = y if rng.rand() < acc else int((y + 1) % k)
        z = np.full(k, -mag, dtype=np.float32)
        z[pred] = mag
        t = np.zeros(k, dtype=np.float32)
        t[y] = 1.0
        recs.append((qtype, z, t, k))
    return recs


# --------------------------------------------------------------- overconfident -> T > 1
choice_k2 = overconfident_records(40, 2, QTYPES["choice"], seed=1, acc=0.6, mag=8.0)
t_choice = fit_one_temperature([(z, t) for _, z, t, _ in choice_k2])
check_true("overconfident/T>1", t_choice > 1.0, "T=%r" % t_choice)
check_true("overconfident/T<=10", t_choice <= TEMP_HI, "T=%r" % t_choice)


# --------------------------------------------------------------- bucket keys + omit n<10
noul_k2 = overconfident_records(40, 2, QTYPES["noul"], seed=2, acc=0.6, mag=8.0)
choice_k12 = overconfident_records(40, 12, QTYPES["choice"], seed=3, acc=0.6, mag=8.0)
tiny_score = overconfident_records(5, 4, QTYPES["score"], seed=4, acc=0.6, mag=8.0)
mixed = choice_k2 + noul_k2 + choice_k12 + tiny_score
fitted = fit_temperature_map(mixed, compute_ece=True)
keys = set(fitted["temperature_by_options"])
check_true("keys/choice:2", "choice:2" in keys, keys)
check_true("keys/noul:2", "noul:2" in keys, keys)
check_true("keys/choice:11+", "choice:11+" in keys, keys)
check("keys/temp_bucket choice:2", temp_bucket(QTYPES["choice"], 2), "choice:2")
check("keys/temp_bucket noul:2", temp_bucket(QTYPES["noul"], 2), "noul:2")
check("keys/temp_bucket choice:11+", temp_bucket(QTYPES["choice"], 12), "choice:11+")
check_true("omit/n<10 score:3-5 not in map", "score:3-5" not in keys, keys)
check("omit/n_by_bucket still counts tiny", fitted["n_by_bucket"].get("score:3-5"), 5)
check_true("omit/not NaN", all(np.isfinite(v) for v in fitted["temperature_by_options"].values()))
check_true("alias/fit_temperatures is fit_temperature_map", fit_temperatures is fit_temperature_map)


# --------------------------------------------------------------- clamp 0.1-10
# 50% accurate, extreme logits: NLL wants a large T, clamped at 10.
rng = np.random.RandomState(7)
hi_pairs = []
for i in range(40):
    y = int(rng.randint(0, 2))
    pred = y if i % 2 == 0 else 1 - y
    z, t = peaked(2, pred, 80.0)
    t = np.zeros(2, dtype=np.float32)
    t[y] = 1.0
    hi_pairs.append((z, t))
t_hi = fit_one_temperature(hi_pairs)
check_true("clamp/high in [0.1, 10]", TEMP_LO <= t_hi <= TEMP_HI, "T=%r" % t_hi)
check_true("clamp/high near 10", t_hi >= 5.0, "T=%r" % t_hi)

# Always-correct, mild logits: NLL wants a small T, clamped at 0.1.
lo_pairs = []
for i in range(40):
    z, t = peaked(2, i % 2, 0.3)
    lo_pairs.append((z, t))
t_lo = fit_one_temperature(lo_pairs)
check_true("clamp/low in [0.1, 10]", TEMP_LO <= t_lo <= TEMP_HI, "T=%r" % t_lo)
check_true("clamp/low near 0.1", t_lo <= 0.15, "T=%r" % t_lo)

src = inspect.getsource(fit_one_temperature)
check_true("clamp/source uses 0.1", "0.1" in src or "TEMP_LO" in src)
check_true("clamp/n<10 returns 1.0", fit_one_temperature(lo_pairs[:5]) == 1.0)


# --------------------------------------------------------------- ECE after < before
report = fitted["report"]
check_true("ece/after < before", report["ece_after"] < report["ece_before"],
           "before=%r after=%r" % (report["ece_before"], report["ece_after"]))
check_true("ece/before finite", np.isfinite(report["ece_before"]))
check_true("ece/after finite", np.isfinite(report["ece_after"]))


# --------------------------------------------------------------- live Agent map + JSON round-trip
agent = Agent.__new__(Agent)
agent.temperature = [1.0, 1.0, 1.0]
agent.temperature_by_options = {}
live = agent.fit_temperatures(mixed, compute_ece=True)
check_true("agent/live temperature_by_options updates",
           "choice:2" in agent.temperature_by_options, agent.temperature_by_options)
check("agent/live matches result", agent.temperature_by_options, live["temperature_by_options"])
check_true("agent/type-level choice T>1", agent.temperature[0] > 1.0, agent.temperature)

td = tempfile.mkdtemp()
calib_path = os.path.join(td, "calibration.json")
agent.save_calibration(calib_path)
check_true("save/no model.safetensors", not os.path.exists(os.path.join(td, "model.safetensors")))
check_true("save/only the json file", os.listdir(td) == ["calibration.json"], os.listdir(td))
with open(calib_path) as f:
    payload = json.load(f)
check("save/keys", sorted(payload.keys()), ["temperature", "temperature_by_options"])
check_true("save/no weights key", "model.safetensors" not in json.dumps(payload))

other = Agent.__new__(Agent)
other.load_calibration(calib_path)
check("load/temperature", other.temperature, agent.temperature)
check("load/by_options", other.temperature_by_options, agent.temperature_by_options)

# module helpers round-trip on a stub
stub = type("Stub", (), {})()
apply_calibration_payload(stub, calibration_payload([1.2, 1.1, 1.3], {"choice:2": 1.4}))
check("stub/temperature", stub.temperature, [1.2, 1.1, 1.3])
check("stub/by_options", stub.temperature_by_options, {"choice:2": 1.4})


# --------------------------------------------------------------- constructor wiring (no Hub download)
init_src = inspect.getsource(Agent.__init__)
check_true("init/calibration kwarg", "calibration: Optional[str] = None" in init_src)
check_true("init/load_calibration call", "self.load_calibration(calibration)" in init_src)
check_true("init/does not write safetensors", "save_file" not in init_src)
load_src = inspect.getsource(load)
check_true("load/calibration kwarg", "calibration" in load_src)
check_true("load/forwards calibration", "calibration=calibration" in load_src)

fwd = inspect.getsource(Agent._forward_logits)
check_true("forward/returns logits before temperature", "return ids, items, logits, act, n_tokens" in fwd)
check_true("forward/no temp_bucket divide", "temp_bucket" not in fwd)
rec_src = inspect.getsource(records_from_labeled)
check_true("records_from_labeled/_forward_logits", "_forward_logits" in rec_src)
sys_one = inspect.getsource(Agent.system_one)
check_true("system_one/uses _forward_logits", "_forward_logits" in sys_one)
check_true("system_one/does not fit", "fit_temperature" not in sys_one)

# public exports
import laya as _laya  # noqa: E402
for name in ("fit_temperatures", "fit_one_temperature", "fit_temperature_map"):
    check_true("export/%s in __all__" % name, name in _laya.__all__)
    check_true("export/%s attr" % name, hasattr(_laya, name))


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
if not FAIL:
    print("synthetic ECE before=%.4f after=%.4f  choice:2 T=%.3f  noul:2 T=%.3f  choice:11+ T=%.3f" % (
        report["ece_before"], report["ece_after"],
        fitted["temperature_by_options"]["choice:2"],
        fitted["temperature_by_options"]["noul:2"],
        fitted["temperature_by_options"]["choice:11+"],
    ))
sys.exit(1 if FAIL else 0)
