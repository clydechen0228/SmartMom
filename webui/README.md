# Local web UI

Laya ships no CLI and no interface — it is a library. This is a small local one:
a FastAPI server holding the checkpoints resident, and a single static page.

```bash
python webui/server.py                    # http://127.0.0.1:8000
python webui/server.py --preload          # all three checkpoints resident at startup
python webui/server.py --port 8077 --device cpu
```

Needs `fastapi` and `uvicorn` on top of the package:

```bash
pip install fastapi uvicorn
```

## The page

- **State** — plain text, or JSON (an object, or a list of conversation turns). The sample
  chips cover English, Chinese, Hindi and a prompt-injection attempt.
- **Questions** — a preset, or any question JSON you type. Presets are the shipped
  `guard_questions()`, `triage_questions()`, `moderation_questions()`, `router_questions()`,
  `email_questions()`, plus the `customer_service` typed-decisions workflow.
- **Model** — `auto` routes by script and language; the rest force a checkpoint.
- **Predict** runs a forward pass. **Route only** answers which checkpoint would handle it
  and why, without loading weights.

Results show the routing reason, the timings, and one card per question with the full
probability distribution — not just the argmax.

## Endpoints

| route | body | returns |
|---|---|---|
| `GET /api/meta` | — | version, device, models, presets, samples, workflow signatures |
| `POST /api/route` | `{state, questions?}` | the `RouteDecision`, no weights loaded |
| `POST /api/predict` | `{state, questions, model?}` | answers, routing, usage, timings |

`/api/predict` resolves and loads the checkpoint *before* timing the forward pass, so a
cold load is reported separately as `cold_load` instead of being billed to `ms`. Without
that split the first call reads as ~18 s of "inference" when the pass itself is ~200 ms.

## Cost of the first call

Checkpoints load lazily. The first request that needs one pays 12–40 s to build it, then
it stays resident (`max_loaded=3`, so switching languages does not evict). `--preload`
moves that cost to startup. Measured on CPU (2019 i7): cold load 12–16 s per checkpoint,
then 50–190 ms per question.

## Device

`--device`, else `$LAYA_DEVICE`, else `examples_common.pick_device()`: cuda, then mps,
then cpu — downgrading mps to cpu when `torch.autocast` has no mps support, which is the
case on Intel Macs (torch tops out at 2.2.2 there). See `examples/README.md`.
