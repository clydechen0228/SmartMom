# Plant APS demo

A working advanced planning and scheduling (APS) demo for a whole plant, built from the
design in *Plant APS Design* (English / 中文). The scheduler optimises on-time delivery first.
It reads orders shaped like SAP S/4HANA data, repairs the schedule when something goes
wrong, uses Laya to turn written disruptions into typed events, and previews what the
custom MES and SAP would receive.

```bash
pip install -e . fastapi uvicorn "ortools==9.10.4067" "numpy<2"   # from the repo root
python demos/aps/server.py                                       # http://127.0.0.1:8095
python demos/aps/server.py --mock                                # no Laya weights
```

`ortools` 9.10 is pinned because later releases require numpy 2, and torch 2.2 (the
newest on Intel Macs) needs numpy 1.x. On a machine with a newer torch, recent ortools works.

The first full plan takes up to 60 s. The workbench (EN / 中文 toggle) shows the result
as soon as it lands. Set `APS_FULL_S` and `APS_REPAIR_S` to change the time limits.

## The plant

| Line | Work centre | Machines | Notes |
|---|---|---|---|
| L1 | WC-CNC | CNC-01, CNC-02, CNC-03 (0.8× speed) | family setups 15–60 min |
| L1 | WC-CMM | CMM-01 | |
| L3 | WC-ASM | ASM-03, ASM-05 (1.1×) | |
| L3 | WC-LKT | LKT-04 | bottleneck, about 81% loaded |
| L3 | WC-AOI | AOI-01 | |

There are 36 production orders for 4 materials (150 operations) over a five-day, two-shift
week, with nights unmanned. `data/sap/*.json` holds the SAP-shaped orders, routings and
work centres, which `aps/sample_data.py` regenerates. `data/setup_matrix.json` is the
APS-maintained setup matrix, since SAP has no standard object for it.

## How it decides

- **Full plan:** a CP-SAT flexible job shop with shifts, sequence-dependent setups and
  material release dates. It is lexicographic: weighted tardiness is minimised and locked,
  then setups are minimised.
- **Repair** (after an event): only a *neighbourhood* is re-optimised. That covers operations
  on the disrupted work centre or order, plus every operation of an order a
  sequence-keeping repair would make late. Everything else keeps its machine and its order
  on that machine. Weighted tardiness is minimised and locked first, then the start-time
  change against the approved plan. Setups are respected but not re-optimised; a full
  re-plan does that.
- **Frozen window:** operations running or starting within 2 h never move.
- **Proposals, not edits:** every repair is a proposal with before and after KPIs. A named
  planner approves it (which creates a schedule version) or rejects it. An independent
  checker (`aps/checker.py`, sharing no code with the solver) validates every schedule.
  A proposal with violations cannot be approved.
- **Disruption inbox:** Laya answers two choice questions (event type, urgency). Rules
  pull machine, order, material, hours, percent and quantity out of the text and check them
  against master data. Below 50% confidence, or with a field missing, the planner completes
  the event.

## Measured

`python demos/aps/evaluate.py` wrote `data/eval_results.json` on a 2019 Intel laptop CPU
(12 threads), with the default limits of 60 s full and 30 s repair:

| | Due-date rule | CP-SAT full plan |
|---|---|---|
| On-time orders | 28 of 36 (77.8%) | **35 of 36 (97.2%)** |
| Weighted lateness | 62.4 h | **3.3 h** |
| Setup time | 28.9 h | **7.2 h** |

Each disruption was repaired from the same approved plan at Mon 09:00:

| Event | Solve | On-time before → after | Ops moved > 30 min | Violations |
|---|---|---|---|---|
| LKT-04 down 6 h | 30.1 s, tardiness optimal | 97.2% → 91.7% | 59 | 0 |
| Rush order, 80 × HSG-7731 in 30 h | 13.3 s, optimal | 97.2% → 91.9% | 62 | 0 |
| Quality hold on 10004405, 8 h | 24.8 s, optimal | 97.2% → 94.4% | 83 | 0 |
| CNC-02 30% slower | 30.1 s, tardiness optimal | 97.2% → 97.2% | 88 | 0 |
| Castings for 10004430 late 24 h | 1.0 s, optimal | 97.2% → 97.2% | 0 | 0 |

"Optimal" means optimal *within the repair neighbourhood*. A full re-plan can still do
better, at the price of moving more work. CP-SAT runs 8 threads and results vary between
runs. In repeated trials of the same event, most repairs reached the same optimum, but
some runs stopped at the time limit with a feasible, checked proposal.

Inbox, real Laya checkpoints: **6 of 8** sample messages (EN / 中文 / DE) classified
correctly. **No confident errors**: both misses scored below the gate (0.10, 0.49) and went
to the planner. The option wording matters. An earlier wording read a Chinese "camera
cleaned, no impact" note as a late delivery at 0.86 confidence.

## Files

| Path | What |
|---|---|
| `aps/plant.py` | Internal model and the SAP-shaped adapter |
| `aps/solver.py` | Due-date rule, repair heuristic, CP-SAT model |
| `aps/checker.py` | Independent feasibility check |
| `aps/state.py` | Clock, events, proposals, versions, MES and SAP payloads |
| `aps/inbox.py` | Laya questions, extraction rules, sample messages |
| `server.py`, `static/index.html` | API and the bilingual workbench |
| `test_aps.py` | 46 offline tests (mock classifier, short limits, about 1 min) |
| `evaluate.py` | The measurements above |

## Not in the demo

Live SAP (OData or IDoc) and MES connections, the master-plan level, multiple plants,
setups that avoid unmanned nights, and preemption. The design document lists these as the
next steps.
