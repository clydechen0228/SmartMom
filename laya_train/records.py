"""Labelled records: the one data format for evaluation, calibration and fine-tuning.

A record is one text with the typed questions asked about it and the right answers, one
JSON object per line (JSONL):

    {"id": "m1",
     "state": "LKT-04 pressure sensor failed, the leak tester is stopped.",
     "questions": {"kind": {"type": "choice", "instructions": "What does the message report?",
                            "criteria": {"normal": "...", "stopped": "...", ...}}},
     "gold": {"kind": "stopped"},
     "meta": {"lang": "en", "source": "aps.inbox", "split": "train"}}

`gold` holds, per question id, either a plain answer (a choice key, `true`/`false` for
noul, a level index for score) or `{"probabilities": {...}}` for a soft target, the shape
the typed-decisions dataset uses. Questions without a gold answer are skipped. It is the
same shape as `LocalLLaMA/typed-decisions`, so records from either source mix freely.
"""
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

from laya.common import QTYPES, build_sequence, render_options


def load_records(paths: Sequence[str]) -> List[dict]:
    out = []
    for path in ([paths] if isinstance(paths, str) else paths):
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError("%s:%d is not JSON: %s" % (path, n, e))
                for key in ("state", "questions", "gold"):
                    if key not in rec:
                        raise ValueError("%s:%d has no %r" % (path, n, key))
                rec.setdefault("id", "%s:%d" % (os.path.basename(path), n))
                rec.setdefault("meta", {})
                out.append(rec)
    return out


def save_records(records: Iterable[dict], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def to_internal(qdef: dict) -> dict:
    """A public question definition in the form `build_sequence` takes (as `Agent` does)."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def option_keys(qdef: dict) -> List[str]:
    """Answer keys in the order the model scores them."""
    q = to_internal(qdef)
    if q["t"] == "choice":
        return list(q["crit"].keys())
    if q["t"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(q["crit"]))]


def target_vector(qdef: dict, gold) -> Optional[List[float]]:
    """The gold answer as a probability vector over the options, or None if unusable."""
    keys = option_keys(qdef)
    if isinstance(gold, dict) and "probabilities" in gold:
        probs = {str(k): float(v) for k, v in gold["probabilities"].items()}
        vec = [max(0.0, probs.get(k, 0.0)) for k in keys]
    else:
        if isinstance(gold, bool):
            gold = "true" if gold else "false"
        gold = str(gold)
        if gold not in keys:
            return None
        vec = [1.0 if k == gold else 0.0 for k in keys]
    s = sum(vec)
    return [v / s for v in vec] if s > 0 else None


def build_items(records: Sequence[dict], tok, cfg: dict) -> List[dict]:
    """One model input per (record, question) that has a usable gold answer."""
    items = []
    max_len, head_max_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
    for rec in records:
        for qid, qdef in rec["questions"].items():
            if qid not in rec["gold"]:
                continue
            target = target_vector(qdef, rec["gold"][qid])
            if target is None:
                continue
            q = to_internal(qdef)
            seq, markers = build_sequence(tok, rec["state"], q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                continue                     # options did not fit head_max_len
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": target,
                          "label": target.index(max(target)), "rid": rec["id"], "qid": qid,
                          "lang": rec.get("meta", {}).get("lang")})
    return items


def split_of(rec: dict, dev_share: float = 0.2, seed: str = "laya") -> str:
    """A stable split from the text itself, so the same text always lands on the same side."""
    if rec.get("meta", {}).get("split"):
        return rec["meta"]["split"]
    h = int(hashlib.sha1((seed + "|" + json.dumps(rec["state"], ensure_ascii=False)).encode()).hexdigest()[:8], 16)
    return "dev" if (h % 1000) / 1000 < dev_share else "train"


def text_key(state) -> str:
    return " ".join(json.dumps(state, ensure_ascii=False).lower().split())


def overlap(a: Sequence[dict], b: Sequence[dict]) -> List[str]:
    """Record ids in `a` whose text also appears in `b`: leakage between train and test."""
    seen = {text_key(r["state"]) for r in b}
    return [r["id"] for r in a if text_key(r["state"]) in seen]
