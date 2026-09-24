# Contributing to SmartMom

Contributions come in as pull requests. `main` changes only through a reviewed pull
request, and the repository owner merges.

## How to contribute

1. **Fork** the repository (or, if you are a collaborator, create a branch).
2. Make your change on a branch named for what it does, e.g. `fix/aps-night-overlap`.
3. **Run the tests** that cover what you changed (below).
4. **Open a pull request** against `main`. The template asks what changed, how you tested
   it and which documents you updated.
5. A reviewer approves, and the owner merges.

## What the branch rules allow

GitHub enforces these rules; they are not conventions.

| | Contributors | Owner |
|---|---|---|
| Open, update and review pull requests | yes | yes |
| Push to your own branch or fork | yes | yes |
| Push directly to `main` | no | no: pull requests only |
| Merge a pull request into `main` | no | yes |
| Force-push to any branch | no | yes |
| Delete any branch | no | yes |

A pull request needs **one approval**:
- the approval must come after the last push;
- a new push dismisses earlier approvals;
- every review conversation must be resolved first.

## Tests to run

The CI on each pull request builds the `laya` package and runs its own tests. The
platform's tests run locally:

| You changed | Run |
|---|---|
| `laya/` (the library) | the suites in `tests/` (CI runs them too) |
| `demos/quality_inspection/` | `python demos/quality_inspection/test_quality_inspection.py` |
| `demos/aps/` | `python demos/aps/test_aps.py` (a few minutes) |
| `demos/platform/`, or a module it mounts | `python demos/platform/test_platform.py` |
| `laya_train/` | `python laya_train/test_laya_train.py`, and `LAYA_TRAIN_SLOW=1 …` for model changes |
| `run.sh` | `bash -n run.sh`, then `./run.sh status`, `start`, `stop` on the server you touched |

Use the project's Python (`./.conda311/bin/python`) or any Python 3.11 environment with
`pip install -e . fastapi uvicorn httpx "ortools==9.10.4067" "numpy<2"`. See `GUIDE.md`
for the environment.

## Rules for changes

- **Numbers come from measurements.** A figure in a document is produced by a script in
  the repository (`evaluate.py`, `laya_train`) and re-measured when the code it describes
  changes. Say how it was measured, and say it plainly when a result is weak.
- **Measure a Laya question before building on it.** Choose wordings on a dev set and
  report on a held-out set; never tune on the held-out messages in `demos/aps/data/`.
- **Keep the documents in step.** The repository's documents are the maintained version.
  If a change alters behaviour, commands or measured results, update the READMEs, `GUIDE.md`
  and the files in `docs/` in the same pull request. Pages with English and 中文 get both.
- **No plant data in git.** Label logs (`demos/aps/data/labels/`), trained checkpoints
  (`ckpt/`), work files (`work/`) and logs (`var/`) are ignored on purpose: they can hold
  plant text. Never commit them, secrets, tokens or personal paths.
- **Commit messages** say what changed and why in the first line; details go in the body.
