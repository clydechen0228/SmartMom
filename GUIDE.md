# Laya: local setup, service, MCP and enforcement

A complete guide to running Laya on this machine and wiring it into Claude Code.
Everything here was measured on this checkout, on an Intel Mac (2019 i7), CPU only.

- [What Laya is (and is not)](#what-laya-is-and-is-not)
- [1. Environment](#1-environment)
- [2. The weights](#2-the-weights)
- [3. Using the library](#3-using-the-library)
- [4. The web UI](#4-the-web-ui)
- [5. The HTTP service](#5-the-http-service)
- [6. MCP server](#6-mcp-server)
- [7. The WebFetch hook](#7-the-webfetch-hook)
- [8. Where to actually use it](#8-where-to-actually-use-it)
- [9. The plant demos and run.sh](#9-the-plant-demos-and-runsh)
- [Troubleshooting](#troubleshooting)
- [Uninstall](#uninstall)

---

## What Laya is (and is not)

Laya scores **typed questions** over natural-language text in a single forward pass.
It is non-autoregressive: no decoder, no generated tokens, nothing to parse and nothing
to hallucinate. You get calibrated probabilities instead of prose.

Three question types:

| type | you give | you get back |
|---|---|---|
| `choice` | labelled criteria | `["choice"]` (a label), `["probabilities"]`, `["confidence"]` |
| `score` | an ordered scale | `["score"]` — a continuous expectation, not an index — plus `["legend"]` |
| `noul` | a yes/no question | `["noul"]` — P(yes). **There is no `["answer"]` boolean.** |

That per-type key difference is the single most common mistake when calling it.

**It cannot write, review or judge code.** The encoders (ModernBERT, mmBERT) have no
code training and the context is 512–1024 tokens, which one real source file exceeds.
Measured here: `difficulty` scored a trivial variable rename at 1.51 and a full
hexagonal-architecture refactor at 1.95 — too flat to route on. An agent-trace risk
workflow scored `rm -rf src/` *lower* for `needs_review` (0.23) than a test failure
(0.41). Do not gate destructive operations on it.

**What it is genuinely good at**, and what this setup is built around: screening
untrusted natural-language text fast enough to run on every fetch.

---

## 1. Environment

This machine is an Intel Mac, which constrains everything downstream:

- PyPI ships **no macOS x86_64 torch wheels past 2.2.2** (cp38–cp311). The system
  Python 3.13 cannot install torch at all.
- torch 2.2.2 is built against **NumPy 1.x** and crashes on import with NumPy 2.
- **transformers 5.x disables PyTorch** entirely below torch 2.5, leaving `laya`
  loaded with no model backend and no error until you call it.
- torch 2.2.2 has **no mps autocast**, so Laya's own device pick (`mps` on any Mac
  with a Metal GPU) dies with `unsupported autocast device_type 'mps'`.

The working environment is `./.conda311`, already built:

```bash
conda create -p ./.conda311 python=3.11
./.conda311/bin/pip install "torch==2.2.2" "numpy<2" "transformers>=4.48,<5"
./.conda311/bin/pip install -e .
./.conda311/bin/pip install fastapi uvicorn mcp        # service + MCP server
```

> `pip install mcp` fails building `cryptography` from source (it reaches for Rust).
> `pip install --only-binary=:all: cryptography` first — a prebuilt wheel exists.

Always pass `device="cpu"` here, or use `examples_common.pick_device()`, which prefers
cuda → mps → cpu and downgrades mps to cpu when autocast is unavailable. On Apple
Silicon or Linux + CUDA none of this applies.

**Test suite** — all 7 CI suites pass in this environment (298 assertions):

```bash
for t in test_router test_criteria test_download test_shortlist \
         test_decision_model test_packaging test_email; do
  ./.conda311/bin/python tests/$t.py
done
```

`tests/test_local_e2e.py` is excluded from CI and needs checkpoints at `~/laya_models`.

---

## 2. The weights

All three checkpoints are downloaded — **2.2 GB total**, bf16, in the shared HF cache:

```
~/.cache/huggingface/hub/models--convaiinnovations--laya/
└── snapshots/1c5edc17.../
    ├── model.safetensors         804M   english          ModernBERT-large, 421M params, 512 ctx
    ├── multilingual/…            614M   multilingual     mmBERT-base, 322M params, 1024 ctx, 100+ langs
    └── typed-decisions/…         804M   typed-decisions  ModernBERT-large, 421M params, 1024 ctx
```

One HF repo, three subfolders; only the requested subfolder downloads. Move the cache
with `HF_HOME` or `HF_HUB_CACHE`; delete the `models--convaiinnovations--laya` directory
to reclaim the space.

**Checkpoints are not interchangeable.** On the same English refund email,
`english` scored `churn_risk` 0.78 while `multilingual` scored 0.02 — the text says
"or we will cancel our plan", so the English checkpoint is right. Don't use
`multilingual` for English just because it is smaller.

**One known defect:** loading `english` warns

```
this checkpoint ships temperatures outside [0.5, 5] … clamping choice:11+=0.1006.
Treat confidence from the affected buckets as uncalibrated.
```

Choice questions with **11 or more options** return uncalibrated `confidence` on the
`english` checkpoint. The probability distribution is still usable; the confidence
number is not.

---

## 3. Using the library

```python
from laya import Router

router = Router(device="cpu")        # cpu is mandatory on this machine
res = router.predict(state, questions)

res["answers"]["department"]["choice"]    # "billing"
res["answers"]["urgency"]["score"]        # 1.88  (expectation over ["legend"])
res["answers"]["refund"]["noul"]          # 0.94  P(yes)
res["routing"]["reason"]                  # why this checkpoint was chosen
```

`state` may be a string, a dict, or a list of conversation turns.

**Routing is free.** `router.route(state, questions)` returns the decision in ~1.5 ms
without loading weights or running a forward pass. Measured:

```
english   -> english        English Latin text
hindi     -> multilingual   non-Latin script (devanagari, 100% of letters)
chinese   -> multilingual   non-Latin script (han, 100% of letters)
french    -> multilingual   Latin script but language looks like 'fr', not English
```

Override precedence: `model` > `task` > detected workflow > `lang` > detected script.

**Use the presets** before writing questions by hand — they are tuned, yours will
usually be worse: `guard_questions()`, `triage_questions()`, `moderation_questions()`,
`router_questions()`, `email_questions()`. `email_state()` strips signatures and
disclaimers first.

**Many choice options:** all options share a 192-token budget, and exceeding it raises
`ValueError: question 'x' options exceed head_max_len`. Past ~20 labels use
`predict_shortlist()` with `embed_fn_from_agent()` to rank and cut first.

**Runnable examples**, cheapest first — see `examples/README.md`:

| | | cost |
|---|---|---|
| `01_routing.py` | routing decisions and their reasons | no weights, ~1 ms |
| `02_predict.py` | EN / ZH / HI / technical, real forward passes | one checkpoint |
| `03_presets.py` | guard, triage, `email_state()` | reuses it |
| `04_full_router.py` | `preload=True`, all three resident | all three |
| `05_typed_decisions.py` | the four workflow signatures | typed-decisions |

**Performance here:** 12–16 s to build a checkpoint, then 50–190 ms per question on
CPU. The 33 ms / 7.2 ms batched in LAYA.md are T4 GPU figures.

---

## 4. The web UI

```bash
./.conda311/bin/python webui/server.py --port 8077 --preload
open http://127.0.0.1:8077
```

Paste a state on the left, pick a preset or write question JSON on the right, hit
**Predict**. Results show the routing reason, timings, and the full probability
distribution per question — not just the argmax. **Route only** answers which
checkpoint would handle it, in ~2 ms, without loading anything.

Needs `fastapi` and `uvicorn`, which are deliberately **not** in `pyproject.toml` —
library users should not be forced to install a web framework. Details in
`webui/README.md`.

---

## 5. The HTTP service

The same process is the backend for everything else. Keeping one resident service
matters: each checkpoint costs 12–16 s and ~2 GB of RAM to build, so per-client
loading would be unusable.

| route | body | returns |
|---|---|---|
| `GET /api/meta` | — | version, device, models, presets, samples, workflows |
| `POST /api/route` | `{state, questions?}` | the routing decision, no weights loaded |
| `POST /api/predict` | `{state, questions, model?}` | answers, routing, usage, timings |

`/api/predict` loads the checkpoint **before** timing the forward pass and reports a
cold load separately as `cold_load`. Without that split the first call reads as ~18 s
of "inference" when the pass itself is ~200 ms.

**Start and stop it** from the repo root with `./run.sh`:

```bash
./run.sh start laya          # the service on :8077, all three checkpoints preloaded
./run.sh stop laya
./run.sh status              # every server: platform, laya, and the modules on their own
./run.sh start all           # the demo platform (:8100) and the Laya service
./run.sh install-service     # the launchd job below, filled in for this checkout
```

**Keep it running** with the bundled launchd job (what `./run.sh install-service` does):

```bash
sed -e "s|__LAYA_ROOT__|$PWD|g" -e "s|__HOME__|$HOME|g" \
    mcp_server/com.laya.service.plist > ~/Library/LaunchAgents/com.laya.service.plist
launchctl load ~/Library/LaunchAgents/com.laya.service.plist
tail -f ~/Library/Logs/laya-service.log
```

Starts at login, preloads all three, restarts if it dies. The command fills
this checkout's paths into the plist (`__LAYA_ROOT__`, `__HOME__`); run it again after
moving the repo. Run it from the repo root.

---

## 6. MCP server

```
Claude Code ──stdio──▶ laya_mcp.py ──HTTP──▶ webui/server.py ──▶ 2.2 GB weights
  (per project)         (instant)            (one, long-lived)     (loaded once)
```

Registered at **user scope**, so it is available in every project with no per-project
setup:

```bash
claude mcp add --scope user laya -- \
  "$PWD/.conda311/bin/python" "$PWD/mcp_server/laya_mcp.py"
claude mcp list      # laya - ✔ Connected
```

Restart Claude Code to pick it up. Two tools:

**`laya_guard(text, source?)`** — screens untrusted text. Returns per-check
probabilities plus a `flagged` boolean, and deliberately **does not echo the text
back**, so a payload inside it cannot reach the agent through the tool result.
Verified through a real MCP client:

| | prompt_injection | jailbreak | flagged |
|---|---|---|---|
| README with a hidden "open ~/.aws/credentials" comment | 0.62 | 0.88 | **true** |
| the same README, clean | 0.08 | 0.14 | false |

**`laya_classify(text, questions, model?)`** — your own typed questions. Returns
answers flattened to value + confidence; full distributions stay in the HTTP API,
since they rarely earn their place in an agent's context.

How you actually invoke them — you ask for the outcome, not the tool:

> Fetch https://some-blog.dev/api-guide and **screen it with laya** before using any of it.

> Here's the issue body from #412. **Check it for prompt injection**, then implement the fix.

> Classify the 50 strings in `data/feedback.json` **with laya** for sentiment and whether each is a bug report.

That last one is where it beats an LLM call outright: 50 items at ~100 ms locally, no
tokens, calibrated probabilities.

---

## 7. The WebFetch hook

**A tool is an offer; a hook is enforcement.** Claude decides whether to call
`laya_guard`. If you say "fetch this and summarise it", it will probably just do that.
The hook runs on **every** WebFetch, unconditionally.

Installed in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "WebFetch",
      "hooks": [{
        "type": "command",
        "command": "\"…/.conda311/bin/python\" \"…/mcp_server/hooks/screen_webfetch.py\" 2>/dev/null || true",
        "timeout": 25,
        "statusMessage": "Screening fetched content with Laya..."
      }]
    }]
  }
}
```

`mcp_server/hooks/screen_webfetch.py` gathers the string leaves of the tool response
(capped at 4000 chars), scores them with the guard preset, and when injection or
jailbreak exceeds `LAYA_GUARD_THRESHOLD` (default 0.5) prints a `PostToolUse` result
that injects a warning into Claude's context and shows the user a one-line notice.

Verified by pipe-testing the literal settings command from an unrelated directory:

```
poisoned page → {"systemMessage": "Laya flagged fetched content … (injection 1.00, jailbreak 1.00)", …}
clean page    → no output
service down  → no output
```

**It fails open and silent by design.** If the service is restarting, the payload is
malformed, or anything else goes wrong, it prints nothing and exits 0. A guard that
blocks all work whenever a local service hiccups is a guard that gets switched off.
This is advisory context injection, not a block — it does not stop the fetch, it tells
Claude to treat what came back as data.

Tunables: `LAYA_GUARD_THRESHOLD`, `LAYA_SERVICE_URL`, `LAYA_GUARD_TIMEOUT`.

Review or disable it any time with `/hooks`. Claude Code only surfaces "Ran N hooks"
when a hook errors or is slow — silent success is invisible by design.

---

## 8. Where to actually use it

**Good fits**, in rough order of value:

1. **Screening third-party prose entering an agent** — fetched pages, issue and PR
   bodies, dependency READMEs, scraped docs. Real risk, right shape, fast enough to
   run on everything. This is what the hook automates.
2. **Bulk classification of user text** — feedback, tickets, messages, in 100+
   languages. Local, free per item, calibrated.
3. **Cheap pre-routing on natural language** — domain and "does this need tools" were
   5/5 correct on coding requests.
4. **Moderation and guardrails** in a product that handles user-generated text.

**Poor fits:** anything about code (diffs, correctness, review, task difficulty),
anything needing more than ~1000 tokens of context, and anything where you need
generated text rather than a score.

---

## 9. The plant demos and run.sh

`demos/` holds a smart-factory platform built on Laya: quality inspection, planning (APS),
the Laya console and a page of Laya training runs, in one process. `run.sh` in the repo
root starts and stops it and the Laya service:

```bash
./run.sh start                 # the platform → http://127.0.0.1:8100/
./run.sh start all             # platform + Laya service (:8077)
./run.sh status
./run.sh stop all
./run.sh help                  # every command, server option and environment variable
```

Details: `demos/platform/README.md`, `demos/aps/README.md`,
`demos/quality_inspection/README.md`. Training Laya on plant data: `laya_train/README.md`.
Contributing (pull requests only; `main` is protected): `CONTRIBUTING.md`.

---

## Troubleshooting

**`unsupported autocast device_type 'mps'`** — pass `device="cpu"`. torch 2.2.2, the
newest available for Intel Macs, has no mps autocast.

**`ModuleNotFoundError: torch`, or "no matching distribution"** — you are on the
system Python 3.13. Use `./.conda311/bin/python`.

**Laya loads but has no backend** — transformers 5.x silently disabled torch. Pin
`transformers<5`. The giveaway is `[transformers] Disabling PyTorch because PyTorch >= 2.5 is required`.

**`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`** — pin
`numpy<2`.

**MCP tools say they can't reach the service** — start it, or load the launchd job.
Check with `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8077/api/meta`.

**`laya` missing from `/mcp`** — restart Claude Code; a session started before
registration won't see it.

**The hook never fires** — confirm with
`jq '.hooks.PostToolUse' ~/.claude/settings.json`, then open `/hooks` once to reload
config, or restart.

**`ValueError: question 'x' options exceed head_max_len=192`** — too many or too
verbose choice labels. Shorten them or use `predict_shortlist()`.

**First call takes 15 s** — that is the checkpoint building, not the forward pass.
Use `--preload`.

---

## Uninstall

```bash
claude mcp remove laya                                             # MCP server
launchctl unload ~/Library/LaunchAgents/com.laya.service.plist     # service
rm ~/Library/LaunchAgents/com.laya.service.plist
rm -rf ~/.cache/huggingface/hub/models--convaiinnovations--laya    # 2.2 GB of weights
rm -rf .conda311                                                   # the environment
```

The hook is removed by deleting the `WebFetch` group from `hooks.PostToolUse` in
`~/.claude/settings.json`, or by turning it off in `/hooks`.
