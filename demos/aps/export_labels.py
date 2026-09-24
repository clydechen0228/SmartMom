"""Turn the planner's decisions into Laya training data.

    python demos/aps/export_labels.py stats
    python demos/aps/export_labels.py export  --out work/aps_labels.jsonl [--include-weak]
    python demos/aps/export_labels.py heldout --out work/aps_heldout.jsonl

`export` writes the labels collected by the running workbench (data/labels/labels.jsonl,
or $APS_LABELS) as laya_train records, leaving out every text of the hand-labelled
measurement sets. `heldout` writes those measurement sets, for evaluating and
calibrating checkpoints. Then:

    python -m laya_train split     --data work/aps_labels.jsonl --train work/train.jsonl --dev work/dev.jsonl \\
                                   --exclude work/aps_heldout.jsonl
    python -m laya_train finetune  --init multilingual --train work/train.jsonl --dev work/dev.jsonl ...
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from aps import labels  # noqa: E402
from laya_train.records import save_records  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("stats", "export", "heldout"))
    ap.add_argument("--log", default=labels.DEFAULT_PATH, help="label log (default: %(default)s)")
    ap.add_argument("--out")
    ap.add_argument("--include-weak", action="store_true", help="also dismissed messages as 'no action'")
    a = ap.parse_args(argv)
    if a.cmd == "stats":
        s = labels.LabelLog(a.log).stats()
        if not s:
            print("no labels yet in", a.log)
        for job, v in sorted(s.items()):
            print("%-10s %4d labels, Laya right on %d (%.0f%%), %d weak" % (
                job, v["labels"], v["laya_right"], 100 * v["laya_right"] / max(1, v["labels"]), v["weak"]))
        return
    if not a.out:
        ap.error("--out is required")
    held = labels.heldout()
    if a.cmd == "heldout":
        print("wrote %d records to %s" % (save_records(held, a.out), a.out))
        return
    recs = labels.export(a.log, include_weak=a.include_weak, exclude_texts=[r["state"] for r in held])
    print("wrote %d records to %s (held-out texts left out)" % (save_records(recs, a.out), a.out))


if __name__ == "__main__":
    main()
