"""Fine-tune a Laya checkpoint on labelled records with RLCD.

The training step is the one in notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb:
for each batch, sample G noisy versions of the model's logits, score each with a strictly
proper scoring rule against the gold answer (log + spherical score, plus the ranked
probability score for `score` questions), and push the logits towards the better samples
(a GRPO-style policy gradient), together with a soft cross-entropy term.

Differences from the notebook:
  * one process on any device (cuda, mps, cpu), or several GPUs under `torchrun`;
  * `freeze_encoder` trains only the decision head: far less memory, fine for a CPU smoke
    test or very little data;
  * with a dev set, it is scored after every epoch and the best epoch is kept (small plant
    datasets overfit quickly);
  * temperatures are fitted on held-out calibration records, never on the training data
    (the notebook fits them on training items, which makes confidence look better than it is).
"""
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from laya.common import proper_reward

from .checkpoint import add_history, load_for_training, resolve, write
from .records import build_items
from .runlog import data_summary, save_run, temperatures, weight_changes
from .scoring import metrics, raw_logits


@dataclass
class TrainConfig:
    epochs: int = 4
    micro_batch: int = 8
    grad_accum: int = 4
    group_size: int = 4            # noisy samples per item for the policy-gradient baseline
    lr_encoder: float = 2.5e-5
    lr_head: float = 1.0e-4
    weight_decay: float = 0.01
    sigma_start: float = 0.4       # exploration noise on the logits, annealed per epoch
    sigma_end: float = 0.1
    ce_weight: float = 1.0         # soft cross-entropy guidance next to the RL loss
    w_sph: float = 0.75
    w_rps: float = 1.0
    freeze_encoder: bool = False
    max_steps: int = 0             # stop after this many optimiser steps (0 = no limit)
    seed: int = 42
    max_len: Optional[int] = None  # override the checkpoint's context lengths
    head_max_len: Optional[int] = None
    save_dtype: str = "fp16"
    log_every: int = 20
    gate: float = 0.5


def _collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    b = {"input_ids": torch.full((n, L), pad_id, dtype=torch.long), "attention_mask": torch.zeros((n, L), dtype=torch.long),
         "marker_pos": torch.zeros((n, kmax), dtype=torch.long), "marker_mask": torch.zeros((n, kmax), dtype=torch.bool),
         "target": torch.zeros((n, kmax)), "qtype": torch.tensor([it["qtype"] for it in items])}
    for i, it in enumerate(items):
        k = len(it["markers"])
        b["input_ids"][i, :len(it["ids"])] = torch.tensor(it["ids"])
        b["attention_mask"][i, :len(it["ids"])] = 1
        b["marker_pos"][i, :k] = torch.tensor(it["markers"])
        b["marker_mask"][i, :k] = True
        b["target"][i, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return b


def _dist():
    """(rank, world_size, local_rank) under torchrun, else a single process."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return 0, 1, 0
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return dist.get_rank(), ws, int(os.environ.get("LOCAL_RANK", "0"))


def finetune(init: str, train_records: Sequence[dict], out_dir: str, dev_records: Optional[Sequence[dict]] = None,
             calib_records: Optional[Sequence[dict]] = None, cfg: Optional[TrainConfig] = None,
             device: Optional[str] = None, log=print, test_records: Optional[Sequence[dict]] = None) -> dict:
    """Train, keep the best epoch (by dev log loss if a dev set is given), calibrate, save.
    With `test_records`, the starting and the final checkpoint are both scored on them."""
    tc = cfg or TrainConfig()
    rank, world, local_rank = _dist()
    if device is None:
        device = ("cuda:%d" % local_rank if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    dev_type = torch.device(device).type
    if dev_type == "cuda":
        torch.cuda.set_device(torch.device(device))
    random.seed(tc.seed + rank)
    torch.manual_seed(tc.seed)
    say = log if rank == 0 else (lambda *a, **k: None)

    src = resolve(init)
    model, tok, mcfg = load_for_training(src)
    if tc.max_len:
        mcfg["max_len"] = tc.max_len
    if tc.head_max_len:
        mcfg["head_max_len"] = tc.head_max_len
    items = build_items(train_records, tok, mcfg)
    if not items:
        raise ValueError("no usable labelled answers in the training records")
    dev_items = build_items(dev_records, tok, mcfg) if dev_records else []
    test_items = build_items(test_records, tok, mcfg) if test_records else []
    cfg0 = dict(mcfg)
    started = time.strftime("%Y-%m-%d %H:%M")
    say("training on %d answers from %d records (%s); dev %d answers; device %s x%d"
        % (len(items), len(train_records), src, len(dev_items), device, world))

    if tc.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    if dev_type == "cuda" and not tc.freeze_encoder:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device).train()
    net = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local_rank] if dev_type == "cuda" else None, find_unused_parameters=True)

    enc = [p for n, p in net.named_parameters() if "encoder." in n and p.requires_grad]
    head = [p for n, p in net.named_parameters() if "encoder." not in n and p.requires_grad]
    groups = ([{"params": enc, "lr": tc.lr_encoder}] if enc else []) + [{"params": head, "lr": tc.lr_head}]
    opt = torch.optim.AdamW(groups, weight_decay=tc.weight_decay)
    mine = items[rank::world]
    per_epoch = max(1, math.ceil(len(mine) / (tc.micro_batch * tc.grad_accum)))
    total = per_epoch * tc.epochs if not tc.max_steps else min(per_epoch * tc.epochs, tc.max_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total), eta_min=1e-6)
    use_amp = dev_type == "cuda"
    # torch >= 2.3 has torch.amp.GradScaler; 2.2 (the newest for Intel Macs) only the cuda one
    scaler = (torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_amp))

    history, best, steps, t0, curve = [], None, 0, time.time(), []
    test_before = None
    if rank == 0:
        if test_items:
            test_before = metrics(test_items, raw_logits(model, tok, test_items, device), cfg0, tc.gate)
        if dev_items:                                # epoch 0: the model as it started
            history.append({"epoch": 0, "steps": 0, "seconds": 0.0,
                            "dev": metrics(dev_items, raw_logits(model, tok, dev_items, device), mcfg, tc.gate)["all"]})
        model.train()
    best_dir = os.path.join(out_dir, "_best")
    for epoch in range(tc.epochs):
        random.shuffle(mine)
        sigma = tc.sigma_start + (tc.sigma_end - tc.sigma_start) * (epoch / max(1, tc.epochs - 1))
        net.train()
        run_loss, run_r, n_b, accum = 0.0, 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(mine), tc.micro_batch):
            b = _collate(mine[i:i + tc.micro_batch], tok.pad_token_id)
            with torch.autocast(dev_type, dtype=torch.float16, enabled=use_amp):
                logits, act = net(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                                  b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            qtype = b["qtype"].to(device)
            eps = torch.randn((tc.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask              # zero-mean over the options
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=tc.w_sph, w_rps=tc.w_rps)
                adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + tc.ce_weight * loss_ce) / tc.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            accum += 1
            run_loss += loss.item() * tc.grad_accum
            run_r += r.mean().item()
            n_b += 1
            last = i + tc.micro_batch >= len(mine)
            if accum % tc.grad_accum == 0 or last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
                curve.append({"step": steps, "epoch": epoch + 1, "loss": round(run_loss / n_b, 5),
                              "reward": round(run_r / n_b, 5), "lr": sched.get_last_lr()[-1]})
                if steps % tc.log_every == 0:
                    say("  epoch %d step %d  loss %.4f  reward %.3f  lr %.2e"
                        % (epoch + 1, steps, run_loss / n_b, run_r / n_b, sched.get_last_lr()[-1]))
                if tc.max_steps and steps >= tc.max_steps:
                    break
        row = {"epoch": epoch + 1, "steps": steps, "train_loss": round(run_loss / max(1, n_b), 4),
               "reward": round(run_r / max(1, n_b), 4), "seconds": round(time.time() - t0, 1)}
        if rank == 0 and dev_items:
            m = metrics(dev_items, raw_logits(model, tok, dev_items, device), mcfg, tc.gate)["all"]
            row["dev"] = m
            if best is None or m["nll"] < best["dev"]["nll"]:
                best = row
                write(best_dir, mcfg, model=model, tok=tok, dtype=tc.save_dtype)
        say("epoch %d done: %s" % (epoch + 1, {k: v for k, v in row.items() if k != "dev"})
            + ("  dev acc %.3f nll %.3f ece %.3f" % (row["dev"]["accuracy"], row["dev"]["nll"], row["dev"]["ece"]) if "dev" in row else ""))
        history.append(row)
        if tc.max_steps and steps >= tc.max_steps:
            break
    if world > 1:
        import torch.distributed as dist
        dist.barrier()

    report = {"init": init, "config": asdict(tc), "answers": len(items), "history": history}
    trained_params = sum(p.numel() for g in groups for p in g["params"])
    if rank == 0:
        from safetensors.torch import load_file
        start = history[0] if history and history[0]["epoch"] == 0 else None
        if start is not None and (best is None or start["dev"]["nll"] <= best["dev"]["nll"]):
            # no epoch beat the model it started from: keep the starting weights
            model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(src, "model.safetensors")).items()})
            report["kept_epoch"] = 0
            say("no epoch improved dev log loss over the starting model; its weights are kept")
        elif best is not None:                       # reload the best epoch before calibrating
            model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(best_dir, "model.safetensors")).items()})
            report["kept_epoch"] = best["epoch"]
        mcfg = add_history(mcfg, {"step": "finetune", "from": init, "answers": len(items),
                                  "epochs": len(history), "kept_epoch": report.get("kept_epoch")})
        if calib_records:
            from .calibrate import fit
            c_items = build_items(calib_records, tok, mcfg)
            c_logits = raw_logits(model, tok, c_items, device)
            fitted = fit(c_items, c_logits)
            temps = list(mcfg.get("temperature") or [1.0, 1.0, 1.0])
            for qt, t in fitted["types"].items():
                temps[qt] = t
            mcfg["temperature"] = temps
            mcfg["temperature_by_options"] = {**(mcfg.get("temperature_by_options") or {}), **fitted["buckets"]}
            report["calibration"] = {"answers": len(c_items), "buckets": fitted["buckets"], "counts": fitted["counts"]}
            if not fitted["buckets"] and not fitted["types"]:
                say("calibration skipped: fewer than 20 answers per bucket in the calibration records")
        else:
            say("no calibration records: the checkpoint keeps its old temperatures; calibrate before gating on confidence")
        mcfg["fine_tuned"] = True
        write(out_dir, mcfg, model=model, tok=tok, dtype=tc.save_dtype)
        run = {"kind": "finetune", "init": init, "init_dir": src, "started": started, "device": "%s x%d" % (device, world),
               "seconds": round(time.time() - t0, 1), "config": asdict(tc), "kept_epoch": report.get("kept_epoch"),
               "trained": "head only" if tc.freeze_encoder else "encoder and head",
               "params": {"total": sum(p.numel() for p in model.parameters()), "trained": trained_params},
               "data": {"train": data_summary(train_records), "dev": data_summary(dev_records or []),
                        "calib": data_summary(calib_records or []), "test": data_summary(test_records or [])},
               "epochs": history, "steps": curve, "calibration": report.get("calibration"),
               "temperatures": {"before": temperatures(cfg0), "after": temperatures(mcfg)},
               "weight_changes": weight_changes(model, os.path.join(src, "model.safetensors"))}
        if test_items:
            run["test"] = {"before": test_before,
                           "after": metrics(test_items, raw_logits(model, tok, test_items, device), mcfg, tc.gate)}
        save_run(out_dir, run)
        report["run"] = os.path.join(out_dir, "run.json")
        if os.path.isdir(best_dir):
            import shutil
            shutil.rmtree(best_dir)
        report["out_dir"] = out_dir
        say("saved %s" % out_dir)
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
    return report
