# Examples

Run them from the repo root, in that order. `01` needs nothing but the package;
`02` and `03` download a checkpoint from the Hub on first run.

| script | what it shows | cost |
|---|---|---|
| `01_routing.py` | `Router.route()` — which checkpoint, and why | no weights, ~1 ms |
| `02_predict.py` | real forward passes over EN / ZH / HI / technical states | downloads one checkpoint |
| `03_presets.py` | `guard_questions()`, `triage_questions()`, `email_state()` | reuses that checkpoint |
| `04_full_router.py` | `preload=True` — all three resident, automatic routing | needs all three (~4.7 GB) |
| `05_typed_decisions.py` | `auto_task_detection=True` and the four workflow signatures | needs typed-decisions |

```bash
python examples/01_routing.py
python examples/02_predict.py                 # defaults to the multilingual checkpoint
python examples/02_predict.py english         # or typed-decisions
python examples/03_presets.py
python examples/04_full_router.py
python examples/05_typed_decisions.py
```

## Reading a result

Each question type puts its answer under a different key:

```python
res["answers"]["department"]["choice"]         # "billing"  — plus ["probabilities"], ["confidence"]
res["answers"]["urgency"]["score"]             # 1.88 — an expectation over ["legend"], not an index
res["answers"]["refund_requested"]["noul"]     # 0.94 — P(yes). There is no ["answer"] boolean.
```

## Device

`pick_device()` in `examples_common.py` honours `$LAYA_DEVICE`, else prefers cuda, then
mps, then cpu. It downgrades mps to cpu when `torch.autocast` has no mps support: Laya
would otherwise pick mps on any Mac with a Metal GPU, but that autocast path only exists
from torch 2.3, and Intel Macs top out at torch 2.2.2 — PyPI ships no newer macOS
x86_64 wheels — where a forward pass dies with `unsupported autocast device_type 'mps'`.

```bash
LAYA_DEVICE=cpu python examples/02_predict.py
```
