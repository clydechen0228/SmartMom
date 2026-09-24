"""Evaluate, calibrate and fine-tune Laya checkpoints on your own labelled data.

    from laya_train import load_records, evaluate, calibrate, finetune, TrainConfig

    dev = load_records("data/dev.jsonl")
    print(evaluate("multilingual", dev)["all"])
    calibrate("multilingual", load_records("data/calib.jsonl"), "ckpt/multilingual-cal", eval_records=dev)
    finetune("multilingual", load_records("data/train.jsonl"), "ckpt/plant-v1",
             dev_records=dev, calib_records=load_records("data/calib.jsonl"))

The command line does the same: `python -m laya_train --help`.
"""
from .calibrate import calibrate
from .finetune import TrainConfig, finetune
from .records import load_records, save_records, split_of
from .scoring import evaluate, table

__all__ = ["load_records", "save_records", "split_of", "evaluate", "table", "calibrate", "finetune", "TrainConfig"]
