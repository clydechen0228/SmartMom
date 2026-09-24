# laya_train: evaluate, calibrate and fine-tune Laya on your own data

The `laya` package runs checkpoints; it has no training API. Its building blocks
(`build_model`, `build_sequence`, `proper_reward`) are used by the fine-tuning notebook in
`notebooks/`. This module packages that loop as a library and a command line, adds
calibration and evaluation, and reads one simple data format.

```bash
python -m laya_train check     --data labels.jsonl --against heldout.jsonl
python -m laya_train split     --data labels.jsonl --train train.jsonl --dev dev.jsonl --exclude heldout.jsonl
python -m laya_train evaluate  --routed english multilingual --data heldout.jsonl --detail
python -m laya_train calibrate --ckpt multilingual --data calib.jsonl --eval heldout.jsonl --out ckpt/ml-cal
python -m laya_train finetune  --init multilingual --train train.jsonl --dev dev.jsonl \
                               --calib calib.jsonl --out ckpt/plant-v1
torchrun --nproc_per_node=2 -m laya_train finetune ...      # several GPUs
```

```python
from laya_train import load_records, evaluate, calibrate, finetune, TrainConfig
```

Run it from the repo root (it is not part of the pip package). Tests:
`python laya_train/test_laya_train.py` (seconds), `LAYA_TRAIN_SLOW=1 …` (real checkpoint,
about 75 s on a laptop CPU).

## Data: one labelled text per line

```json
{"id": "n7", "state": "CNC-03 液压系统故障，已停机。",
 "questions": {"condition": {"type": "choice", "instructions": "How is the machine?",
               "criteria": {"ok": "normal, no findings", "watch": "…", "stop": "out of service"}}},
 "gold": {"condition": "stop"},
 "meta": {"lang": "zh", "source": "aps.note"}}
```

`gold` holds a choice key, `true`/`false` for a yes/no question, a level for a score
question, or `{"probabilities": {...}}` for a soft target. Ask the questions exactly as
your application asks them: the wording is part of the input. The format matches the
`LocalLLaMA/typed-decisions` dataset, so the two can be mixed.

## Three levels, cheapest first

| Step | What changes | Needs | Hardware |
|---|---|---|---|
| **Evaluate** | Nothing: measures accuracy, calibration error (ECE), log loss and how many confident answers are wrong | Labelled held-out records | CPU |
| **Calibrate** | One temperature per question type and option count, in `rl_agent_config.json`; the weights are linked, not copied | A few hundred labelled answers per bucket (`choice:3-5`, `choice:6-10`, …) | CPU, minutes |
| **Fine-tune** | The weights (RLCD, as in the notebook), then calibration on separate records | Hundreds to thousands of labelled answers per question | A GPU; head-only (`--freeze-encoder`) runs on a CPU for small tests |

`laya-multilingual`, which reads every non-English text, ships with **no fitted
temperatures** (`[1.0, 1.0, 1.0]`). Its confidence is uncalibrated until you calibrate it,
so calibration is the first step whenever an application gates on confidence in other
languages.

## Rules that keep the numbers honest

- **Three separate sets.** Train on one, pick the epoch on a dev set, calibrate and report
  on others. `split --exclude` drops every text that is in a held-out file;
  `check --against` finds leaks.
- **Calibrate on data the model was not trained on.** `finetune` fits temperatures only on
  `--calib` records. The notebook fits them on training items, which makes confidence look
  better than it is.
- **Compare against the published checkpoints** with `evaluate --routed`: each text goes to
  the checkpoint its language routes to, as in an application.
- **Too little data shows.** On 27 APS messages, calibration improved the fit set
  (ECE 0.286 → 0.183) and made the other 20 worse (0.097 → 0.215). Buckets with fewer
  than `--min-count` (default 20) answers keep their old temperature.

## Fine-tuning notes

- Defaults follow the notebook: 4 epochs, encoder learning rate 2.5e-5, head 1e-4, 4 noisy
  samples per answer, noise 0.4 → 0.1, soft cross-entropy weight 1.0.
- With `--dev`, the best epoch by dev log loss is kept. Small plant datasets overfit fast.
- `--freeze-encoder` trains only the decision head: about 3 GB of RAM for the multilingual
  checkpoint on CPU. The English checkpoint (ModernBERT-large) needs a GPU for full training.
- The notebook reports 4–5 h on Kaggle's free 2×T4 for about 30k questions. Training data
  that holds plant text should not leave the plant; use your own GPU for real data.
- Every checkpoint written here records what was done to it in `history` in
  `rl_agent_config.json`.

## Run records and the report

Every `calibrate` and `finetune` run writes `run.json` next to its checkpoint:

- **Data:** counts per question, answer and language, and the records themselves, for
  train, dev, calibration and test.
- **Settings:** every training setting, which parts were trained, and how many parameters.
- **Curves:** loss and proper-score reward per step; dev accuracy, log loss and calibration
  error per epoch, starting from the untrained model (epoch 0); the epoch that was kept.
- **Before and after** on the `--test` records: overall, per question and per language.
- **What was adjusted:** temperatures before and after, and how far each part of the
  network moved (relative weight change per encoder layer, head layer and scorer).
- **Lineage:** the `history` chain in the checkpoint's config.

```bash
python -m laya_train report --runs ckpt --out ckpt/training-report.html   # one self-contained page
```

The platform serves the same page live at `/training/` (runs from `ckpt/`, or `$LAYA_RUNS`).

**Worked example** (in `ckpt/aps-ml-run1`, not committed): the multilingual checkpoint,
fine-tuned on this laptop's CPU (encoder and head, 6 epochs, 11 minutes, 7 GB RAM) on 41
APS dev messages, and scored on 50 test messages it never saw (81 answers):

| | Before | After (epoch 1 kept) |
|---|---|---|
| Accuracy | 63.0% | 77.8% |
| Log loss | 1.214 | 0.509 |
| Calibration error | 0.224 | 0.082 |
| Confident but wrong | 14 of 55 | 3 of 53 |

The dev curve shows why the kept epoch matters: after epoch 1 the training loss fell to 0
while dev log loss rose from 0.95 to 4.2, which is the model memorising 41 messages. With
this little data, treat the numbers as a demonstration of the loop, not as a model to
deploy.

## Using a checkpoint

`laya.load("ckpt/plant-v1")` loads it like a published one. In the APS workbench:

```bash
./run.sh restart platform --aps-laya-multilingual ckpt/plant-v1   # planning inside the platform
./run.sh restart aps --laya-multilingual ckpt/plant-v1            # planning alone
LAYA_MULTILINGUAL=ckpt/plant-v1 ./run.sh restart aps              # same, by environment
```

A custom checkpoint replaces the published one for planning only. Quality inspection, the
Laya console and the injection guard keep the published checkpoints.
