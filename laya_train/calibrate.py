"""Fit calibration temperatures on labelled data, without training the model.

For every bucket `laya.Agent` uses (question type x option count, e.g. "choice:3-5"), one
temperature T is fitted so that softmax(logits / T) minimises the log loss on the
labelled answers. Buckets with too few answers keep the checkpoint's value. The new
checkpoint links to the old weights; only `rl_agent_config.json` changes.

Fit on data the model was not trained on, and judge the result on yet another set.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from laya.common import QTYPE_NAMES, TEMP_MAX, TEMP_MIN, temp_bucket

from .checkpoint import add_history, load_for_training, resolve, write
from .records import build_items
from .runlog import data_summary, save_run, temperatures
from .scoring import metrics, raw_logits


def fit_temperature(pairs: Sequence[Tuple[np.ndarray, Sequence[float]]]) -> float:
    """One temperature for (logits, target) pairs: minimise mean log loss (LBFGS on log T)."""
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    T = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, :len(z)] = torch.tensor(z, dtype=torch.float32)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), TEMP_MIN, TEMP_MAX).item())


def fit(items: Sequence[dict], logits: Sequence[np.ndarray], min_count: int = 20) -> Dict[str, object]:
    """Temperatures per bucket and per question type, from labelled items."""
    by_bucket: Dict[str, list] = defaultdict(list)
    by_type: Dict[int, list] = defaultdict(list)
    for it, z in zip(items, logits):
        by_bucket[temp_bucket(it["qtype"], len(z))].append((z, it["target"]))
        by_type[it["qtype"]].append((z, it["target"]))
    buckets = {b: round(fit_temperature(p), 4) for b, p in by_bucket.items() if len(p) >= min_count}
    types = {qt: round(fit_temperature(p), 4) for qt, p in by_type.items() if len(p) >= min_count}
    counts = {b: len(p) for b, p in by_bucket.items()}
    return {"buckets": buckets, "types": types, "counts": counts}


def calibrate(ckpt: str, records: Sequence[dict], out_dir: str, eval_records: Optional[Sequence[dict]] = None,
              min_count: int = 20, device: Optional[str] = None, gate: float = 0.5, link: bool = True) -> dict:
    """Fit on `records`, write a calibrated checkpoint to `out_dir`, report before and after."""
    src = resolve(ckpt)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_for_training(src)
    model.to(device)
    items = build_items(records, tok, cfg)
    if not items:
        raise ValueError("no usable labelled answers in the calibration records")
    logits = raw_logits(model, tok, items, device)
    fitted = fit(items, logits, min_count)
    if not fitted["buckets"] and not fitted["types"]:
        raise ValueError("no bucket has %d answers; label more data or lower min_count (counts: %s)"
                         % (min_count, fitted["counts"]))

    new = dict(cfg)
    temps = list(cfg.get("temperature") or [1.0, 1.0, 1.0])
    for qt, t in fitted["types"].items():
        temps[qt] = t
    new["temperature"] = temps
    new["temperature_by_options"] = {**(cfg.get("temperature_by_options") or {}), **fitted["buckets"]}
    report = {"fit_on": len(items), "counts": fitted["counts"], "fitted_buckets": fitted["buckets"],
              "fitted_types": {QTYPE_NAMES[k]: v for k, v in fitted["types"].items()},
              "fit_before": metrics(items, logits, cfg, gate)["all"], "fit_after": metrics(items, logits, new, gate)["all"]}
    if eval_records:
        ev_items = build_items(eval_records, tok, cfg)
        ev_logits = raw_logits(model, tok, ev_items, device)
        report["eval_before"] = metrics(ev_items, ev_logits, cfg, gate)
        report["eval_after"] = metrics(ev_items, ev_logits, new, gate)
    new = add_history(new, {"step": "calibrate", "from": ckpt, "answers": len(items), "buckets": fitted["buckets"]})
    write(out_dir, new, source=src, link=link)
    report["out_dir"] = out_dir
    run = {"kind": "calibrate", "init": ckpt, "init_dir": src, "config": {"min_count": min_count, "gate": gate},
           "data": {"calib": data_summary(records), "test": data_summary(eval_records or [])},
           "temperatures": {"before": temperatures(cfg), "after": temperatures(new)},
           "calibration": {"answers": len(items), "buckets": fitted["buckets"], "counts": fitted["counts"]},
           "fit": {"before": report["fit_before"], "after": report["fit_after"]}}
    if eval_records:
        run["test"] = {"before": report["eval_before"], "after": report["eval_after"]}
    save_run(out_dir, run)
    return report
