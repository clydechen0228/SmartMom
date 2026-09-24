"""Tests for laya_train.

    python laya_train/test_laya_train.py            # data handling, calibration maths (seconds)
    LAYA_TRAIN_SLOW=1 python laya_train/test_laya_train.py
        # also evaluate, calibrate and a 2-step head-only fine-tune on the cached
        # multilingual checkpoint, on CPU (about 2 minutes, ~3 GB RAM)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from laya_train.calibrate import fit, fit_temperature  # noqa: E402
from laya_train.records import (load_records, option_keys, overlap, save_records, split_of,  # noqa: E402
                                target_vector)
from laya_train.scoring import metrics, temperature_for  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(name if got == want else "%s: got %r, want %r" % (name, got, want))


CHOICE = {"type": "choice", "instructions": "How is the machine?", "criteria": {"ok": "normal", "watch": "worse", "stop": "down"}}
NOUL = {"type": "noul", "instructions": "Is it broken?"}
SCORE = {"type": "score", "instructions": "How bad?", "criteria": ["fine", "minor", "major"]}

# --- gold answers -> target vectors ---------------------------------------------------------
check("choice keys in scoring order", option_keys(CHOICE), ["ok", "watch", "stop"])
check("choice gold -> one-hot", target_vector(CHOICE, "stop"), [0.0, 0.0, 1.0])
check("unknown option -> unusable", target_vector(CHOICE, "exploded"), None)
check("noul accepts booleans", target_vector(NOUL, True), [0.0, 1.0])
check("score gold by level", target_vector(SCORE, 1), [0.0, 1.0, 0.0])
check("soft target normalised", target_vector(CHOICE, {"probabilities": {"ok": 1, "stop": 3}}), [0.25, 0.0, 0.75])

# --- files, splits, leakage ---------------------------------------------------------------
tmp = tempfile.mkdtemp()
recs = [{"id": "a%d" % i, "state": "note %d" % i, "questions": {"c": CHOICE}, "gold": {"c": "ok"}} for i in range(200)]
path = os.path.join(tmp, "r.jsonl")
check("save and load round trip", (save_records(recs, path), len(load_records(path))), (200, 200))
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"state": "x", "questions": {}}) + "\n")
try:
    load_records(path)
    check("record without gold refused", False, True)
except ValueError as e:
    check("record without gold refused", "gold" in str(e), True)
share = sum(split_of(r) == "dev" for r in recs) / len(recs)
check("split near the asked share", 0.1 < share < 0.3, True)
check("split is stable", [split_of(r) for r in recs[:20]] == [split_of(r) for r in recs[:20]], True)
check("same text, same split", split_of({"state": "Same Text"}) == split_of({"state": "Same Text", "id": "other"}), True)
check("leakage found across files", overlap(recs[:5], [{"state": "note 3"}, {"state": "NOTE   4"}]), ["a3", "a4"])

# --- calibration maths ----------------------------------------------------------------------
rng = np.random.default_rng(0)
pairs = []
for _ in range(400):                     # logits 3x too sharp for how often they are right
    true = rng.integers(3)
    z = rng.normal(0, 1, 3)
    z[true] += 1.0
    pairs.append((z * 3.0, [1.0 if i == true else 0.0 for i in range(3)]))
t = fit_temperature(pairs)
check("fitted temperature undoes over-sharp logits", 2.0 < t < 4.5, True)
items = [{"qtype": 0, "target": tg, "label": int(np.argmax(tg)), "qid": "c", "lang": "en"} for _, tg in pairs]
logits = [z for z, _ in pairs]
f = fit(items, logits, min_count=20)
check("fit fills the bucket", list(f["buckets"]), ["choice:3-5"])
check("too few answers -> no bucket", fit(items[:10], logits[:10], min_count=20)["buckets"], {})
before = metrics(items, logits, {"temperature": [1.0, 1.0, 1.0]})["all"]
after = metrics(items, logits, {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": f["buckets"]})["all"]
check("calibration lowers log loss and ECE", (after["nll"] < before["nll"], after["ece"] < before["ece"]), (True, True))
check("accuracy unchanged by a temperature", after["accuracy"], before["accuracy"])
check("bucket beats the per-type value", temperature_for({"temperature": [2.0, 1, 1], "temperature_by_options": {"choice:3-5": 3.0}}, 0, 3), 3.0)
check("temperatures clamped like laya.Agent", temperature_for({"temperature": [0.1, 1, 1]}, 0, 3), 0.5)

# --- run records and the report ------------------------------------------------------------
from laya_train.report import render  # noqa: E402
from laya_train.runlog import data_summary, load_runs, module_of, save_run  # noqa: E402
check("network parts are readable", [module_of(n) for n in ("encoder.layers.3.attn.Wqkv.weight", "encoder.embeddings.tok_embeddings.weight",
                                                          "head.layers.1.linear1.weight", "scorer.1.weight")],
      ["encoder.layers.3", "encoder.embeddings", "head.layers.1", "scorer"])
ds = data_summary(recs[:3] + [{"id": "x", "state": "y", "questions": {"c": CHOICE}, "gold": {"c": "nope"}}])
check("data summary counts usable answers only", (ds["records"], ds["answers"], ds["questions"]), (4, 3, {"c": {"ok": 3}}))
run_dir = os.path.join(tmp, "runs", "demo")
os.makedirs(run_dir)
save_run(run_dir, {"kind": "finetune", "init": "multilingual", "data": {"train": ds}, "note": "</script><b>x</b>"})
runs = load_runs(os.path.join(tmp, "runs"))
check("runs found under a directory", [r["name"] for r in runs], ["demo"])
page = render(runs)
check("report embeds the runs as data, safely", ("</script><b>" not in page, '"name": "demo"' in page or '"name":"demo"' in page), (True, True))

# --- the real thing (opt-in) ----------------------------------------------------------------
if os.environ.get("LAYA_TRAIN_SLOW"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import laya
    from laya_train import TrainConfig, calibrate, evaluate, finetune
    notes = [("CNC-01 ran the whole shift without problems.", "ok"), ("ASM-03 本班运行正常。", "ok"),
             ("CNC-03 液压系统故障，已停机。", "stop"), ("AOI-01 camera dead since 14:00, nothing inspected.", "stop"),
             ("CNC-02 spindle getting louder towards the end of the shift.", "watch"),
             ("LKT-04 密封圈有轻微漏气，还能用，需要关注。", "watch")]
    recs = [{"id": "n%d" % i, "state": s, "questions": {"c": CHOICE}, "gold": {"c": g}} for i, (s, g) in enumerate(notes)]
    m = evaluate("multilingual", recs, device="cpu")
    check("evaluate scores every answer", m["all"]["n"], 6)
    rep = calibrate("multilingual", recs, os.path.join(tmp, "cal"), min_count=5, device="cpu")
    check("calibrated checkpoint loads", "choice:3-5" in laya.load(os.path.join(tmp, "cal"), device="cpu").temperature_by_options, True)
    r = finetune("multilingual", recs, os.path.join(tmp, "ft"), dev_records=recs, device="cpu",
                 cfg=TrainConfig(epochs=1, max_steps=2, micro_batch=3, grad_accum=1, freeze_encoder=True), log=lambda *a: None)
    ans = laya.load(os.path.join(tmp, "ft"), device="cpu").system_one(notes[2][0], {"c": CHOICE})["answers"]["c"]
    check("fine-tuned checkpoint loads and answers", ans["choice"] in ("ok", "watch", "stop"), True)
    check("history starts at epoch 0 and records the steps", (r["history"][0]["epoch"], r["history"][-1]["steps"]), (0, 2))
    check("run.json written with curves", os.path.exists(os.path.join(tmp, "ft", "run.json")), True)

print("%d passed, %d failed" % (len(PASS), len(FAIL)))
for x in FAIL:
    print("FAIL", x)
sys.exit(1 if FAIL else 0)
