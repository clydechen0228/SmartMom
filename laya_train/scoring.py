"""Raw scores from a checkpoint and the numbers that say whether it got better.

Metrics per question, per language and overall:
  accuracy      top answer equals the gold answer
  nll, brier    proper scores of the probabilities (lower is better)
  ece           calibration error of the top probability (15 bins; lower is better)
  gated         answers whose Laya confidence clears the gate, and how many of those are wrong:
                what an application that acts on confident answers would get wrong
"""
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from laya.common import (QTYPE_NAMES, clamp_temperature, collate_items, confidence_from_probs,
                         ece_score, temp_bucket)


@torch.no_grad()
def raw_logits(model, tok, items: Sequence[dict], device: str = "cpu", batch: int = 16) -> List[np.ndarray]:
    """Unscaled option logits per item (temperature is applied later, per bucket)."""
    model.eval()
    out = []
    for i in range(0, len(items), batch):
        chunk = [{k: it[k] for k in ("ids", "markers", "qtype")} for it in items[i:i + batch]]
        b = collate_items([chunk], tok.pad_token_id)
        logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                          b["marker_mask"].to(device), b["qtype"].to(device))
        logits = logits.float().cpu().numpy()
        for r, it in enumerate(chunk):
            out.append(logits[r, :len(it["markers"])].copy())
    return out


def temperature_for(cfg: dict, qtype: int, k: int) -> float:
    """The temperature `laya.Agent` would apply (clamped, bucket first)."""
    by_opt = cfg.get("temperature_by_options", {}) or {}
    b = temp_bucket(qtype, k)
    if b in by_opt:
        return clamp_temperature(by_opt[b])
    return clamp_temperature((cfg.get("temperature") or [1.0, 1.0, 1.0])[qtype])


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def reported_confidence(qtype: int, p: np.ndarray) -> float:
    """The `confidence` field `laya.Agent` returns, which applications gate on."""
    if QTYPE_NAMES[qtype] == "noul":
        return float(max(p[1], 1 - p[1]))
    return confidence_from_probs(p, len(p))


def metrics(items: Sequence[dict], logits: Sequence[np.ndarray], cfg: dict, gate: float = 0.5,
            cfgs: Optional[Sequence[dict]] = None) -> dict:
    """`cfgs` gives each item its own checkpoint config (routed evaluation)."""
    groups: Dict[str, list] = defaultdict(list)
    for n, (it, z) in enumerate(zip(items, logits)):
        p = softmax(z / temperature_for(cfgs[n] if cfgs else cfg, it["qtype"], len(z)))
        t = np.asarray(it["target"])
        row = {"correct": float(p.argmax() == it["label"]), "top": float(p.max()),
               "nll": float(-(t * np.log(np.clip(p, 1e-12, 1))).sum()),
               "brier": float(((p - t) ** 2).sum()), "conf": reported_confidence(it["qtype"], p)}
        for g in ("all", "q:" + it["qid"], "lang:" + str(it.get("lang") or "?")):
            groups[g].append(row)
    out = {}
    for g, rows in groups.items():
        c = np.array([r["correct"] for r in rows])
        top = np.array([r["top"] for r in rows])
        gated = [r for r in rows if r["conf"] >= gate]
        out[g] = {"n": len(rows), "accuracy": round(float(c.mean()), 4),
                  "nll": round(float(np.mean([r["nll"] for r in rows])), 4),
                  "brier": round(float(np.mean([r["brier"] for r in rows])), 4),
                  "ece": round(ece_score(top, c), 4),
                  "gated": len(gated), "gated_wrong": sum(1 for r in gated if not r["correct"])}
    return out


def evaluate(ckpt: str, records: Sequence[dict], device: Optional[str] = None, gate: float = 0.5,
             cfg_override: Optional[dict] = None) -> dict:
    """Metrics for one checkpoint on labelled records."""
    from .checkpoint import load_for_training
    from .records import build_items
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_for_training(ckpt)
    model.to(device)
    items = build_items(records, tok, cfg)
    logits = raw_logits(model, tok, items, device)
    return metrics(items, logits, cfg_override or cfg, gate)


def route_records(records: Sequence[dict]) -> Dict[str, List[dict]]:
    """Split records the way laya.Router would: English text vs. everything else."""
    from laya import Router
    r = Router()                                   # routing only; nothing is loaded
    out: Dict[str, List[dict]] = {"english": [], "multilingual": []}
    for rec in records:
        key = r.route(rec["state"])["model"]
        out["multilingual" if key != "english" else "english"].append(rec)
    return out


def evaluate_routed(english: str, multilingual: str, records: Sequence[dict], device: Optional[str] = None,
                    gate: float = 0.5) -> dict:
    """Metrics for a pair of checkpoints used as an application uses them: each text goes
    to the checkpoint the router picks for its language."""
    from .checkpoint import load_for_training
    from .records import build_items
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    items, logits, cfgs = [], [], []
    for key, ckpt in (("english", english), ("multilingual", multilingual)):
        part = route_records(records)[key]
        if not part:
            continue
        model, tok, cfg = load_for_training(ckpt)
        model.to(device)
        its = build_items(part, tok, cfg)
        items += its
        logits += raw_logits(model, tok, its, device)
        cfgs += [cfg] * len(its)
        del model
    return metrics(items, logits, {}, gate, cfgs=cfgs)


def table(results: Dict[str, dict], keys: Sequence[str] = ("all",)) -> str:
    """A plain-text comparison of several checkpoints: {name: evaluate() result}."""
    lines = ["%-22s %-26s %5s %8s %7s %7s %7s %10s" % ("checkpoint", "group", "n", "accuracy", "nll", "brier", "ece",
                                                        "gated/wrong")]
    groups = sorted({g for r in results.values() for g in r}, key=lambda g: (g != "all", g))
    for g in groups:
        if keys and not any(g == k or g.startswith(k) for k in keys):
            continue
        for name, r in results.items():
            if g in r:
                m = r[g]
                lines.append("%-22s %-26s %5d %8.3f %7.3f %7.3f %7.3f %6d/%-3d" % (
                    name[:22], g[:26], m["n"], m["accuracy"], m["nll"], m["brier"], m["ece"], m["gated"], m["gated_wrong"]))
    return "\n".join(lines)
