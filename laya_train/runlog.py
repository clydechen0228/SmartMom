"""What a run did, saved next to the checkpoint it wrote (`run.json`), for the report.

A run record holds: the kind (finetune / calibrate), where it started, the data it saw
(counts and the texts themselves), every setting, the curves (per step and per epoch),
scores before and after on held-out data, the temperatures before and after, how much
each part of the network moved, and the checkpoint's lineage (`history`).
"""
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, Optional, Sequence

import torch

from laya.common import TEMP_MAX, TEMP_MIN

from .records import option_keys, target_vector

RUN_FILE = "run.json"


def data_summary(records: Sequence[dict], keep_texts: int = 400) -> dict:
    """Counts per question, answer and language, and the records themselves (capped)."""
    per_q: Dict[str, Counter] = defaultdict(Counter)
    langs = Counter()
    rows = []
    for r in records:
        langs[r.get("meta", {}).get("lang") or "?"] += 1
        gold = {}
        for qid, g in r["gold"].items():
            q = r["questions"].get(qid)
            v = target_vector(q, g) if q else None
            if v is None:
                continue
            ans = option_keys(q)[v.index(max(v))]
            per_q[qid][ans] += 1
            gold[qid] = ans
        if gold and len(rows) < keep_texts:
            rows.append({"id": r["id"], "text": r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False),
                         "lang": r.get("meta", {}).get("lang"), "source": r.get("meta", {}).get("source"), "gold": gold})
    return {"records": len(records), "answers": sum(sum(c.values()) for c in per_q.values()),
            "languages": dict(langs), "questions": {q: dict(c) for q, c in per_q.items()}, "rows": rows}


def module_of(name: str) -> str:
    """Group parameter names into parts of the network a person can read."""
    m = re.match(r"^(encoder\.layers\.\d+|encoder\.embeddings|head\.layers\.\d+|type_emb|scorer|act_head)", name)
    if m:
        return m.group(1)
    return "encoder.other" if name.startswith("encoder.") else name.split(".")[0]


@torch.no_grad()
def weight_changes(model: torch.nn.Module, init_path: str) -> list:
    """Relative change ||w - w0|| / ||w0|| per part of the network, in network order."""
    from safetensors.torch import load_file
    w0 = load_file(init_path)
    num, den, count, order = defaultdict(float), defaultdict(float), Counter(), []
    for name, w in model.named_parameters():        # parameters only: buffers are not trained
        if name not in w0 or not w.is_floating_point():
            continue
        g = module_of(name)
        if g not in num:
            order.append(g)
        a, b = w.detach().float().cpu(), w0[name].float()
        num[g] += float(((a - b) ** 2).sum())
        den[g] += float((b ** 2).sum())
        count[g] += a.numel()
    return [{"module": g, "params": count[g], "relative_change": round((num[g] ** 0.5) / max(den[g] ** 0.5, 1e-12), 6)}
            for g in order]


def temperatures(cfg: dict) -> dict:
    """The temperatures a checkpoint applies, as laya.Agent clamps them."""
    t = cfg.get("temperature") or [1.0, 1.0, 1.0]
    clamp = lambda v: round(min(TEMP_MAX, max(TEMP_MIN, float(v))), 4)  # noqa: E731
    return {"by_type": {"choice": clamp(t[0]), "score": clamp(t[1]), "noul": clamp(t[2])},
            "by_options": {k: clamp(v) for k, v in sorted((cfg.get("temperature_by_options") or {}).items())}}


def save_run(out_dir: str, run: dict) -> str:
    run = dict(run)
    run.setdefault("finished", time.strftime("%Y-%m-%d %H:%M"))
    run.setdefault("name", os.path.basename(os.path.abspath(out_dir)))
    path = os.path.join(out_dir, RUN_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1, ensure_ascii=False, default=str)
    return path


def load_runs(root: str) -> list:
    """Every run.json under `root` (one level of checkpoint directories), newest first."""
    runs = []
    if not os.path.isdir(root):
        return runs
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, RUN_FILE)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                run = json.load(f)
            run.setdefault("name", d)
            run["dir"] = os.path.join(root, d)
            cfg_path = os.path.join(root, d, "rl_agent_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, encoding="utf-8") as f:
                    run["lineage"] = json.load(f).get("history", [])
            runs.append(run)
    runs.sort(key=lambda r: r.get("finished", ""), reverse=True)
    return runs
