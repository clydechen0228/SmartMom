"""Fit and persist per-bucket temperatures for Laya logits.

Ports the notebook LBFGS fitter (NLL on softmax(z/T), clamp 0.1-10, skip n<10) and groups
records by `temp_bucket(qtype, k)` so the runtime's `temperature_by_options` lookup has a writer.
Fitting is CPU-side: records are `(qtype, logits, target, k)` and do not load Hub weights.
"""
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .common import ece_score, temp_bucket

MIN_BUCKET_N = 10
TEMP_LO, TEMP_HI = 0.1, 10.0
N_QTYPES = 3

# (qtype:int, logits:1d, target:1d, k:int)
Record = Tuple[int, Any, Any, int]


def _vec(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _pairs_to_tensors(pairs: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
    kmax = max(len(_vec(z)) for z, _ in pairs)
    z_mat = torch.full((len(pairs), kmax), -1e4)
    t_mat = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        z = _vec(z)
        t = _vec(t)
        n = min(len(z), len(t))
        z_mat[i, :n] = torch.from_numpy(np.ascontiguousarray(z[:n]))
        t_mat[i, :n] = torch.from_numpy(np.ascontiguousarray(t[:n]))
    return z_mat, t_mat


def fit_one_temperature(pairs: Sequence) -> float:
    """Fit one scalar T by NLL + LBFGS on log T. Returns 1.0 when n < 10."""
    sel = list(pairs)
    if len(sel) < MIN_BUCKET_N:
        return 1.0
    z_mat, t_mat = _pairs_to_tensors(sel)
    with torch.enable_grad():
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = -(t_mat * torch.log_softmax(z_mat / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss

        opt.step(closure)
    return float(torch.clamp(log_t.exp(), TEMP_LO, TEMP_HI).item())


def _iter_records(records: Iterable) -> List[Tuple[int, np.ndarray, np.ndarray, int]]:
    out = []
    for rec in records:
        if len(rec) == 4:
            qtype, logits, target, k = rec
        elif len(rec) == 3:
            qtype, logits, target = rec
            k = len(_vec(logits))
        else:
            raise ValueError("record must be (qtype, logits, target[, k])")
        logits = _vec(logits)
        target = _vec(target)
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        out.append((int(qtype), logits[:k], target[:k], k))
    return out


def _softmax(z, t_scale: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t_scale))
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _ece_report(recs, temperature, temperature_by_options) -> Dict[str, float]:
    conf_b, cor_b, conf_a, cor_a = [], [], [], []
    for qt, z, t, k in recs:
        y = int(np.argmax(t[:k]))
        p0 = _softmax(z[:k], 1.0)
        conf_b.append(float(p0.max()))
        cor_b.append(float(int(p0.argmax()) == y))
        t_scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        p1 = _softmax(z[:k], t_scale)
        conf_a.append(float(p1.max()))
        cor_a.append(float(int(p1.argmax()) == y))
    return {
        "ece_before": ece_score(np.asarray(conf_b), np.asarray(cor_b)),
        "ece_after": ece_score(np.asarray(conf_a), np.asarray(cor_a)),
        "n": float(len(recs)),
    }


def fit_temperature_map(records: Iterable, compute_ece: bool = False) -> Dict[str, Any]:
    """Fit type-level scalars and per-bucket temperatures.

    Buckets with fewer than 10 examples are omitted; the type-level scalar covers them.
    """
    recs = _iter_records(records)
    temperature = [1.0] * N_QTYPES
    by_type = {qt: [] for qt in range(N_QTYPES)}
    by_bucket: Dict[str, list] = {}
    for qt, z, t, k in recs:
        if qt in by_type:
            by_type[qt].append((z, t))
        key = temp_bucket(qt, k)
        by_bucket.setdefault(key, []).append((z, t))

    n_by_bucket = {key: len(pairs) for key, pairs in by_bucket.items()}
    for qt in range(N_QTYPES):
        if by_type[qt]:
            temperature[qt] = fit_one_temperature(by_type[qt])

    temperature_by_options = {}
    for key, pairs in by_bucket.items():
        if len(pairs) < MIN_BUCKET_N:
            continue
        temperature_by_options[key] = fit_one_temperature(pairs)

    out: Dict[str, Any] = {
        "temperature": temperature,
        "temperature_by_options": temperature_by_options,
        "n_by_bucket": n_by_bucket,
    }
    if compute_ece:
        out["report"] = _ece_report(recs, temperature, temperature_by_options)
    return out


fit_temperatures = fit_temperature_map


def records_from_labeled(agent, pairs: Sequence) -> List[Record]:
    """Collect CPU records from `(state, questions, targets)` via `agent._forward_logits`.

    `targets` maps question id -> 1-d target distribution. Optional for callers who have a
    loaded agent; tests stay weight-free by passing synthetic records to `fit_temperature_map`.
    """
    records: List[Record] = []
    for state, questions, targets in pairs:
        ids, items, logits, _act, _n_tokens = agent._forward_logits(state, questions)
        for r, qid in enumerate(ids):
            k = len(items[r]["markers"])
            qt = int(items[r]["qtype"])
            tgt = _vec(targets[qid])[:k]
            records.append((qt, logits[r, :k], tgt, k))
    return records


def calibration_payload(temperature, temperature_by_options) -> Dict[str, Any]:
    """JSON body written by `Agent.save_calibration` (no weights)."""
    return {
        "temperature": [float(x) for x in temperature],
        "temperature_by_options": {
            str(k): float(v) for k, v in dict(temperature_by_options).items()
        },
    }


def apply_calibration_payload(obj, payload: Dict[str, Any]) -> None:
    temps = payload.get("temperature")
    if temps is None or len(temps) != N_QTYPES:
        raise ValueError("calibration JSON must contain temperature: [3 floats]")
    obj.temperature = [float(x) for x in temps]
    raw = payload.get("temperature_by_options") or {}
    obj.temperature_by_options = {str(k): float(v) for k, v in raw.items()}
