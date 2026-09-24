"""Command line for laya_train.

    python -m laya_train check     --data labels.jsonl --against heldout.jsonl
    python -m laya_train split     --data labels.jsonl --train train.jsonl --dev dev.jsonl --exclude heldout.jsonl
    python -m laya_train evaluate  --ckpt multilingual --ckpt ckpt/plant-v1 --data heldout.jsonl
    python -m laya_train calibrate --ckpt multilingual --data calib.jsonl --eval heldout.jsonl --out ckpt/ml-cal
    python -m laya_train finetune  --init multilingual --train train.jsonl --dev dev.jsonl \\
                                   --calib calib.jsonl --out ckpt/plant-v1 [--epochs 4 --freeze-encoder ...]
    torchrun --nproc_per_node=2 -m laya_train finetune ...        # several GPUs

Checkpoints are names ("english", "multilingual", "typed-decisions"), hub ids or local
directories written by this tool.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import fields

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")   # torch 2.2 forks compile workers otherwise


def _records(paths):
    from .records import load_records
    return load_records(paths) if paths else []


def cmd_check(a):
    from .records import option_keys, overlap, target_vector
    recs = _records(a.data)
    per_q, per_lang, bad = defaultdict(Counter), Counter(), 0
    for r in recs:
        per_lang[r.get("meta", {}).get("lang") or "?"] += 1
        for qid, g in r["gold"].items():
            q = r["questions"].get(qid)
            if q is None or target_vector(q, g) is None:
                bad += 1
                continue
            v = target_vector(q, g)
            per_q[qid][option_keys(q)[v.index(max(v))]] += 1
    print("%d records; languages %s" % (len(recs), dict(per_lang)))
    for qid, c in sorted(per_q.items()):
        print("  %-16s %4d answers  %s" % (qid, sum(c.values()), dict(c.most_common())))
    if bad:
        print("  %d gold answers not usable (unknown question or option)" % bad)
    if a.against:
        leak = overlap(recs, _records(a.against))
        print("texts also in %s: %d%s" % (", ".join(a.against), len(leak), (" e.g. %s" % leak[:5]) if leak else ""))


def cmd_split(a):
    from .records import overlap, save_records, split_of
    recs = _records(a.data)
    if a.exclude:
        drop = set(overlap(recs, _records(a.exclude)))
        if drop:
            print("dropped %d records whose text is in the held-out set" % len(drop))
        recs = [r for r in recs if r["id"] not in drop]
    train = [r for r in recs if split_of(r, a.dev_share) == "train"]
    dev = [r for r in recs if split_of(r, a.dev_share) != "train"]
    print("train %d -> %s, dev %d -> %s" % (save_records(train, a.train), a.train, save_records(dev, a.dev), a.dev))


def cmd_evaluate(a):
    from .scoring import evaluate, evaluate_routed, table
    recs = _records(a.data)
    results = {c: evaluate(c, recs, device=a.device, gate=a.gate) for c in (a.ckpt or [])}
    for en, ml in (a.routed or []):
        name = "routed:%s|%s" % (os.path.basename(en.rstrip("/")), os.path.basename(ml.rstrip("/")))
        results[name] = evaluate_routed(en, ml, recs, device=a.device, gate=a.gate)
    if not results:
        raise SystemExit("give --ckpt and/or --routed")
    print(table(results, keys=("all", "q:", "lang:") if a.detail else ("all",)))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


def cmd_calibrate(a):
    from .calibrate import calibrate
    r = calibrate(a.ckpt, _records(a.data), a.out, eval_records=_records(a.eval), min_count=a.min_count,
                  device=a.device, gate=a.gate, link=not a.copy)
    print("answers per bucket:", r["counts"])
    print("fitted:", r["fitted_buckets"], r["fitted_types"])
    for part in ("fit", "eval"):
        if part + "_before" in r:
            b, af = r[part + "_before"], r[part + "_after"]
            b, af = b.get("all", b), af.get("all", af)
            print("%-5s ece %.3f -> %.3f   nll %.3f -> %.3f   gated %d/%d wrong -> %d/%d wrong" % (
                part, b["ece"], af["ece"], b["nll"], af["nll"], b["gated"], b["gated_wrong"], af["gated"], af["gated_wrong"]))
    print("wrote", r["out_dir"])


def cmd_finetune(a):
    from .finetune import TrainConfig, finetune
    tc = TrainConfig(**{f.name: getattr(a, f.name) for f in fields(TrainConfig)})
    r = finetune(a.init, _records(a.train), a.out, dev_records=_records(a.dev), calib_records=_records(a.calib),
                 cfg=tc, device=a.device, test_records=_records(a.test))
    if a.json and "out_dir" in r:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)


def cmd_report(a):
    from .report import write_report
    path, n = write_report(a.runs, a.out)
    print("wrote %s (%d runs from %s)" % (path, n, a.runs))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m laya_train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="count labels per question and language; find leakage")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--against", nargs="*", help="held-out files to check for shared texts")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("split", help="stable train/dev split by text; drops held-out texts")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--dev-share", type=float, default=0.2)
    p.add_argument("--exclude", nargs="*", help="held-out files; their texts never go into train or dev")
    p.set_defaults(fn=cmd_split)

    p = sub.add_parser("report", help="an HTML page with every run: data, settings, curves, before/after")
    p.add_argument("--runs", default="ckpt", help="directory holding the checkpoints (default: %(default)s)")
    p.add_argument("--out", default="ckpt/training-report.html")
    p.set_defaults(fn=cmd_report)

    for name, fn, h in (("evaluate", cmd_evaluate, "score checkpoints on labelled records"),
                        ("calibrate", cmd_calibrate, "fit temperatures; write a calibrated checkpoint"),
                        ("finetune", cmd_finetune, "train with RLCD; keep the best epoch; calibrate")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--device", default=None)
        p.add_argument("--gate", type=float, default=0.5, help="confidence gate for the 'gated' count")
        p.set_defaults(fn=fn)
        if name == "evaluate":
            p.add_argument("--ckpt", action="append", help="score every record with this checkpoint")
            p.add_argument("--routed", nargs=2, action="append", metavar=("ENGLISH", "MULTILINGUAL"),
                           help="a pair used as applications use it: each text to the checkpoint its language routes to")
            p.add_argument("--data", nargs="+", required=True)
            p.add_argument("--detail", action="store_true", help="also per question and per language")
            p.add_argument("--json")
        elif name == "calibrate":
            p.add_argument("--ckpt", required=True)
            p.add_argument("--data", nargs="+", required=True, help="labelled records to fit on")
            p.add_argument("--eval", nargs="*", help="other labelled records to judge the result on")
            p.add_argument("--out", required=True)
            p.add_argument("--min-count", type=int, default=20)
            p.add_argument("--copy", action="store_true", help="copy weights instead of linking them")
        else:
            from .finetune import TrainConfig
            p.add_argument("--init", required=True)
            p.add_argument("--train", nargs="+", required=True)
            p.add_argument("--dev", nargs="*")
            p.add_argument("--calib", nargs="*")
            p.add_argument("--test", nargs="*", help="held-out records: scored before and after training")
            p.add_argument("--out", required=True)
            p.add_argument("--json")
            for f in fields(TrainConfig):
                if f.name == "gate":
                    continue
                flag = "--" + f.name.replace("_", "-")
                if f.type in (bool, "bool"):
                    p.add_argument(flag, action="store_true", default=f.default)
                else:
                    typ = {"int": int, "float": float, "str": str}.get(getattr(f.type, "__name__", str(f.type)), None)
                    if typ is None:                       # Optional[int]
                        typ = int
                    p.add_argument(flag, type=typ, default=f.default)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
