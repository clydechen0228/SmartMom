# Plant APS demo

A working advanced planning and scheduling (APS) demo for a whole plant, built from the
design in *Plant APS Design* (English / 中文). The scheduler optimises on-time delivery first.
It reads orders shaped like SAP S/4HANA data, repairs the schedule when something goes
wrong, uses Laya to turn written disruptions and planner questions into typed events,
runs what-if scenarios side by side, explains why an order is late, and previews what the
custom MES and SAP would receive.

```bash
pip install -e . fastapi uvicorn "ortools==9.10.4067" "numpy<2"   # from the repo root
python demos/aps/server.py                                       # http://127.0.0.1:8095
python demos/aps/server.py --mock                                # no Laya weights
```

`ortools` 9.10 is pinned because later releases require numpy 2, and torch 2.2 (the
newest on Intel Macs) needs numpy 1.x. On a machine with a newer torch, recent ortools works.

How to use the workbench, in English and 中文: [`docs/aps-workbench-guide.html`](../../docs/aps-workbench-guide.html)
(also served by the platform at `/docs/aps-workbench-guide.html`).

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
- **Ten event kinds:** machine down, machine degrading, planned maintenance (a future
  window), material late, rush order, order cancelled, due date moved, quantity changed,
  quality hold and priority changed. Any order event can also carry a new priority.
- **What-if scenarios:** any event can run as a scenario instead of a proposal. Up to six
  sit side by side with their KPIs; one click views a scenario on the Gantt, and another
  turns it into the proposal. A scenario computed before the plan changed is marked stale
  and cannot be promoted.
- **Why is it late:** `aps/explain.py` splits the wait of every late or at-risk order into
  causes read off the schedule itself (material not yet available, machine busy with other
  orders, unmanned nights, downtime, quality hold, setups) and ranks them, in English and
  Chinese. The orders table shows the biggest cause; clicking an order shows all of them.
- **Load heatmap:** busy share of each work centre's shift time per day, for the current
  plan, the proposal or a scenario.

## How Laya is used

Laya reads language, rules find facts, the solver decides. Every job below was measured
on messages written for it (`data/eval_sets.json`, `data/inbox_dev.json`) before it was
built, and one idea was dropped because Laya read it poorly.

| Job | Laya reads | Rules add | What happens |
|---|---|---|---|
| Disruption inbox | Event kind, twice with different wording; `guard_questions()` for injection | Machines, orders, hours, start times, date shifts, quantities; hedge words; what a delay costs the customer | Applies itself only if both readings agree above 50%, the guard is quiet, every fact is found and the sender is sure; otherwise the planner confirms the prefilled event |
| Command bar | Intent: what-if, report, why | The same facts | A what-if runs a scenario; "why" shows the ranked causes |
| Rejected proposal | Why the planner rejected it (too many changes, a customer promise, people or tools missing, the event was wrong) | Orders and machines named in the comment; orders the proposal made late | Offers the next step: re-run with only the disrupted machine and order free, re-run protecting orders, block a machine, or correct the event |
| Shift notes | Machine condition (ok, watch, stop) | The machine | A likely breakdown marks the machine on the Gantt and offers a machine-down event |

**Rules, not Laya, for three cues.** Measured on 32 new messages, Laya could not tell a
confirmed report from a possibility, how critical an order is to the customer, or how
urgent a message is. These cues are carried by a few words, so rules read them:

- *Unsure sender* ("might", "not confirmed", 可能, 也许, eventuell, möglicherweise): the event
  never applies itself, and the workbench suggests running it as a what-if first.
- *The customer's line would stop or a penalty applies* ("line stops", 停线, 罚款,
  Vertragsstrafe): the event carries priority 3 for that order. *No hurry* ("no rush",
  不急, keine Eile): a rush order gets priority 1.
- *Urgency* follows: today for a machine down or a quality hold, or when the customer's
  line would stop; otherwise this week.

**Command bar examples:**

- *"What if LKT-04 goes down for 8 hours?"* → a scenario is solved; nothing changes.
- *"订单 10004407 为什么有延误风险？"* → the ranked causes for that order.
- *"CNC-01 just broke down, repair takes 3 hours."* → the event form, prefilled.

Below the gate, or when the guard flags the text, the planner picks the intent.

## Measured

`python demos/aps/evaluate.py` wrote `data/eval_results.json` on a 2019 Intel laptop CPU
(12 threads), with the default limits of 60 s full and 30 s repair:

| | Due-date rule | CP-SAT full plan |
|---|---|---|
| On-time orders | 28 of 36 (77.8%) | **35 of 36 (97.2%)** |
| Weighted lateness | 62.4 h | **3.3 h** |
| Setup time | 28.9 h | **8.5 h** |

Each event was repaired from the same approved plan at Mon 09:00 (35 of 36 on time,
weighted lateness 3.3 h):

| Event | Solve | On time after | Weighted lateness after | Ops moved > 30 min | Violations |
|---|---|---|---|---|---|
| LKT-04 down 6 h | 25.6 s, optimal | 35 of 36 | 3.3 h | 19 | 0 |
| CNC-02 down 4 h | 30.1 s, tardiness optimal | 35 of 36 | 3.3 h | 67 | 0 |
| CNC-02 30% slower | 30.1 s, tardiness optimal | 35 of 36 | 3.3 h | 62 | 0 |
| CNC-01 maintenance in 20 h for 3 h | 7.7 s, optimal | 35 of 36 | 3.3 h | 5 | 0 |
| Rush order, 80 × HSG-7731 in 30 h | 20.1 s, optimal | 33 of 37 | 10.0 h | 51 | 0 |
| Quality hold on 10004405, 8 h | 0.7 s, optimal | 35 of 36 | 3.3 h | 4 | 0 |
| Castings for 10004430 late 24 h | 0.7 s, optimal | 35 of 36 | 3.3 h | 0 | 0 |
| Order 10004409 cancelled | 0.6 s, optimal | 34 of 35 | 3.3 h | 0 | 0 |
| Order 10004404 due 48 h earlier | 0.9 s, optimal | 33 of 36 | 40.9 h | 30 | 0 |
| Order 10004413 cut to 60 pieces | 0.6 s, optimal | 35 of 36 | 3.3 h | 0 | 0 |
| Order 10004403 raised to priority 3 | 0.5 s, optimal | 35 of 36 | 5.0 h | 0 | 0 |
| LKT-04 down 6 h, re-run with only LKT-04 free | 1.2 s, optimal | 35 of 36 | 3.3 h | 21 | 0 |
| LKT-04 down 6 h, re-run protecting 10004405 | 24.2 s, optimal | 35 of 36 | 3.3 h | 18 | 0 |

"Optimal" means optimal *within the repair neighbourhood*. A full re-plan can still do
better, at the price of moving more work. CP-SAT runs 8 threads and results vary between
runs. Moving 10004404 two days earlier cannot be met: the repair keeps the rest on time
and shows the planner the cost. 10004403 is late because its material arrives late;
raising its priority cannot fix that, and weighted lateness rises with its weight. The
re-run with only the disrupted machine free did not move fewer operations here (the normal
repair already minimises moves), but it solved in 1.2 s instead of 18 s. With a shorter
time limit, where the normal repair stops before minimising moves, it moved far fewer.

Two solver fixes came out of the new events. Downtime or maintenance that overlaps a
night made the model infeasible (blocked intervals are now merged). The repair warm start
could break the "keep your order on your machine" constraints and was discarded, and one
CNC-02 repair ended at 2 of 36 on time (the warm start now keeps that order).

**Laya, real checkpoints, 18 messages in English, 中文 and Deutsch plus 2 injections:**

| | Result |
|---|---|
| Event kind read correctly | 12 of 18, and 2 of 2 injections flagged |
| Applied by itself | 6, all correct |
| **Confident errors (a wrong event applied by itself)** | **0** |
| Reading 1 alone, 50% gate | 3 confident errors |
| Reading 2 alone, 50% gate | 2 confident errors |
| Guard false alarms | 3 genuine messages flagged (sent to the planner, reading kept) |
| Command intents | 7 of 8, 0 confident errors |
| Rejection reasons (12 held-out comments) | 9 of 12; 8 above the gate, 2 of them wrong |
| Shift notes (12 held-out notes) | 5 flagged, none healthy: 4 of 4 breakdowns, 1 of 4 early warnings |
| Cue rules (20 held-out messages) | unsure sender 20 of 20, customer consequence 19 of 20 |
| Latency | 0.6–3.9 s per message for the four questions on the laptop CPU (median 1.8 s), plus the guard |

The readings fail on different messages, which is why the agreement rule removes the
confident errors. The cost is that 12 of 18 messages go to the planner, prefilled. The
weak spots are the maintenance and machine-problem wordings (a "Maintenance estimates 6
hours" note reads as planned maintenance, but then misses its start time and stops) and
date changes (m14 and m15). Better option wording, measured with `evaluate.py
--laya-only`, did not help: on 32 new messages a rewritten wording read the same
55–60% as the current one, so the current wording stays.

Two confident mistakes out of eight on rejection comments is why a rejection only
*suggests* a next step: all four stay on screen and the planner clicks one. Shift notes
catch breakdowns but not early warnings ("vibration still within limits", "grip slips
now and then"), so a quiet note is not evidence that a machine is fine.

## Files

| Path | What |
|---|---|
| `aps/plant.py` | Internal model and the SAP-shaped adapter |
| `aps/solver.py` | Due-date rule, repair heuristic, CP-SAT model |
| `aps/checker.py` | Independent feasibility check |
| `aps/state.py` | Clock, events, proposals, versions, MES and SAP payloads |
| `aps/inbox.py` | Laya questions, the two-reading gate, guard, extraction rules, samples |
| `aps/explain.py` | Why-late causes and the load by work centre and day |
| `data/inbox_dev.json`, `data/eval_sets.json` | Messages for choosing wordings (dev) and measuring (test) |
| `server.py`, `static/index.html` | API and the bilingual workbench |
| `test_aps.py` | 101 offline tests (mock classifier, short limits, a few minutes) |
| `evaluate.py` | The measurements above |

## Not in the demo

Live SAP (OData or IDoc) and MES connections, the master-plan level, multiple plants,
setups that avoid unmanned nights, and preemption. The design document lists these as the
next steps.
