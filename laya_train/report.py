"""The training report: every run under a directory, as one self-contained HTML page.

    python -m laya_train report --runs ckpt --out ckpt/training-report.html

The platform serves the same page live at /training/. For each run it shows the data it
saw, every setting, the curves (loss and reward per step; dev scores per epoch, starting
from the untrained model), held-out scores before and after, the temperatures before and
after, how much each part of the network moved, and the checkpoint's lineage.
"""
import json
import os
from typing import List, Tuple

from .runlog import load_runs

HERE = os.path.dirname(os.path.abspath(__file__))


def render(runs: List[dict], title: str = "Laya training runs") -> str:
    with open(os.path.join(HERE, "report_template.html"), encoding="utf-8") as f:
        tpl = f.read()
    data = json.dumps(runs, ensure_ascii=False, default=str).replace("</", "<\\/")
    return tpl.replace("/*__RUNS__*/[]", data).replace("__TITLE__", title)


def write_report(root: str, out: str) -> Tuple[str, int]:
    runs = load_runs(root)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(runs))
    return out, len(runs)
