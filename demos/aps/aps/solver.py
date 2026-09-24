"""Scheduling: a due-date dispatching baseline and the CP-SAT model.

Both take a `Problem` and return a schedule: {op_id: {"machine", "start", "end", "setup"}}.

Full plan, solved level by level and locked before the next (lexicographic):
    1. total weighted tardiness      sum_o  w_o * max(0, C_o - d_o)
    2. total setup minutes           (subject to level 1 <= best found)

Repair (after a disruption), on the affected neighbourhood only:
    1. total weighted tardiness      (ops outside the affected set keep their machine)
    2. total start-time change       (subject to level 1 <= best found)

Simplifications, stated so nobody mistakes them for features: setups may run through
an unmanned night (only operations are kept out of blocked time), and an interrupted
operation restarts from the beginning.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

from .plant import DAY, HORIZON_DAYS, SHIFT_END, SHIFT_START, Plant

SPILL_DAYS = 5                                   # schedule may run past the horizon
MAX_T = (HORIZON_DAYS + SPILL_DAYS) * DAY


@dataclass
class Problem:
    plant: Plant
    now: int = 0
    done: Dict[str, dict] = field(default_factory=dict)      # op -> recorded assignment, finished
    fixed: Dict[str, dict] = field(default_factory=dict)     # op -> assignment that must not move
    holds: Dict[str, int] = field(default_factory=dict)      # order -> earliest restart
    previous: Dict[str, dict] = field(default_factory=dict)  # last approved schedule (repair)
    affected: Optional[Set[str]] = None                      # repair: ops free to move anywhere

    def pending_ops(self):
        for o in self.plant.orders.values():
            for op in o.ops:
                if op.id not in self.done:
                    yield o, op

    def blocked(self, machine: str) -> List[Tuple[int, int]]:
        """Plant blocks plus nights for the spill days after the horizon."""
        extra = [(d * DAY + SHIFT_END, (d + 1) * DAY + SHIFT_START)
                 for d in range(HORIZON_DAYS, HORIZON_DAYS + SPILL_DAYS)]
        merged: List[Tuple[int, int]] = []
        # merged, because a downtime or maintenance window can overlap a night, and the
        # solver puts blocked time in a no-overlap constraint
        for b0, b1 in sorted(self.plant.blocked.get(machine, []) + extra):
            if merged and b0 <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b1))
            else:
                merged.append((b0, b1))
        return merged

    def machine_tail(self, machine: str) -> Tuple[int, Optional[str]]:
        """End and setup family of the last finished op on a machine."""
        end, fam = 0, None
        for op_id, a in self.done.items():
            if a["machine"] == machine and a["end"] >= end:
                end, fam = a["end"], a["family"]
        return end, fam


def _earliest(p: Problem, o, op) -> int:
    lb = p.now
    if op.seq == o.ops[0].seq:
        lb = max(lb, o.release)
    if o.id in p.holds:
        lb = max(lb, p.holds[o.id])
    for prev in o.ops:
        if prev.seq < op.seq and prev.id in p.done:
            lb = max(lb, p.done[prev.id]["end"])
    return lb


def _fits(blocks, start, dur) -> int:
    """Earliest start >= `start` whose [s, s+dur) avoids every blocked interval."""
    s = start
    moved = True
    while moved:
        moved = False
        for b0, b1 in blocks:
            if s < b1 and s + dur > b0:
                s, moved = b1, True
    return s


def _with_setups(p: Problem, sched: Dict[str, dict]) -> Dict[str, dict]:
    """Recompute each op's setup minutes from its machine predecessor."""
    fam = {op.id: op.family for _, op in p.pending_ops()}
    wc = {op.id: op.work_center for _, op in p.pending_ops()}
    by_m: Dict[str, List[str]] = {}
    for op_id, a in sched.items():
        by_m.setdefault(a["machine"], []).append(op_id)
    for m, ids in by_m.items():
        ids.sort(key=lambda i: sched[i]["start"])
        _, last = p.machine_tail(m)
        for i in ids:
            sched[i]["setup"] = p.plant.setup_minutes(wc[i], last, fam[i])
            sched[i]["family"] = fam[i]
            last = fam[i]
    return sched


# ---------------------------------------------------------------------------------------
def list_schedule(p: Problem, priority, prefer=None, stick: int = 60, pinned=None) -> Dict[str, dict]:
    """Serial list scheduling. `priority(order, op)` ranks ready operations (lowest first);
    `prefer(op_id)` names a machine to keep unless another finishes `stick` minutes sooner."""
    plant = p.plant
    free, last = {}, {}
    for m in plant.machines:
        free[m], last[m] = p.machine_tail(m)
        free[m] = max(free[m], p.now)
    sched: Dict[str, dict] = {}
    for op_id, a in sorted(p.fixed.items(), key=lambda kv: kv[1]["start"]):
        sched[op_id] = dict(a)
        if a["end"] >= free[a["machine"]]:
            free[a["machine"]], last[a["machine"]] = a["end"], a["family"]
    todo = {op.id: (o, op) for o, op in p.pending_ops() if op.id not in sched}
    # Pinned ops also keep their previous order on their machine (the repair model demands
    # it); otherwise the warm start is infeasible and the solver starts from nothing.
    chain_prev: Dict[str, str] = {}
    if pinned and p.previous:
        by_m: Dict[str, List[str]] = {}
        for k in pinned:
            if k in todo and k in p.previous:
                by_m.setdefault(p.previous[k]["machine"], []).append(k)
        for ids in by_m.values():
            ids.sort(key=lambda i: p.previous[i]["start"])
            chain_prev.update(zip(ids[1:], ids))
    while todo:
        cands = []
        for strict in (True, False):
            for op_id, (o, op) in todo.items():
                preds = [x for x in o.ops if x.seq < op.seq and x.id not in p.done]
                if any(x.id not in sched for x in preds):
                    continue
                if strict and chain_prev.get(op_id) in todo:
                    continue
                r = max([_earliest(p, o, op)] + [sched[x.id]["end"] for x in preds])
                cands.append((priority(o, op, r), r, op_id))
            if cands:
                break
        cands.sort()
        _, r, op_id = cands[0]
        o, op = todo.pop(op_id)
        options = []
        want = prefer(op_id) if prefer else None
        machines = [want] if pinned and op_id in pinned and want in plant.eligible(op) else plant.eligible(op)
        for m in machines:
            dur = plant.duration(op, m)
            su = plant.setup_minutes(op.work_center, last[m], op.family)
            s = _fits(p.blocked(m), max(r, free[m] + su), dur)
            options.append((s + dur, m, s, dur))
        options.sort()
        choice = options[0]
        for opt in options:
            if opt[1] == want and opt[0] <= choice[0] + stick:
                choice = opt
        _, m, s, dur = choice
        sched[op_id] = {"machine": m, "start": s, "end": s + dur, "family": op.family}
        free[m], last[m] = s + dur, op.family
    return _with_setups(p, sched)


def dispatch_edd(p: Problem) -> Dict[str, dict]:
    """Earliest-due-date list scheduling: the rule a planner would use by hand."""
    return list_schedule(p, lambda o, op, r: (o.due, -o.weight, op.seq))


def repair_heuristic(p: Problem) -> Dict[str, dict]:
    """Keep the previous plan's order and machines where still possible; slot new or
    displaced operations in by due date. Always feasible, close to what the floor knows."""
    prev = p.previous or {}

    def prio(o, op, ready):
        # Rank by when the op actually can start, not only where it used to be: a held or
        # late op then waits its turn instead of reserving its machine and stalling the
        # queue behind it.
        want = prev[op.id]["start"] if op.id in prev else max(p.now, o.due - 8 * 60)
        return (max(want, ready), op.seq)

    aff = p.affected

    def prefer(k):
        return prev.get(k, {}).get("machine")

    # unaffected ops keep their machine whatever it costs, so the warm start obeys the
    # neighbourhood constraints of the repair model
    return list_schedule(p, prio, prefer, stick=10 ** 6 if aff is not None else 60,
                         pinned=None if aff is None else {k for k in prev if k not in aff})


def weighted_tardiness(p: Problem, sched: Dict[str, dict]) -> int:
    total = 0
    for o in p.plant.orders.values():
        ends = [sched[op.id]["end"] for op in o.ops if op.id in sched] + \
               [p.done[op.id]["end"] for op in o.ops if op.id in p.done]
        total += o.weight * max(0, max(ends) - o.due)
    return total


def best_start(p: Problem, repair: bool) -> Dict[str, dict]:
    """The better of the dispatching rule and the sequence-keeping repair heuristic, by
    weighted tardiness: the solver's warm start and its fallback if it finds nothing."""
    if repair and p.previous and p.affected is not None:
        return repair_heuristic(p)
    cands = [dispatch_edd(p)] + ([repair_heuristic(p)] if repair and p.previous else [])
    return min(cands, key=lambda sch: weighted_tardiness(p, sch))


# ---------------------------------------------------------------------------------------
@dataclass
class SolveResult:
    schedule: Dict[str, dict]
    status: List[str]
    objective: Dict[str, float]
    seconds: float


def solve(p: Problem, time_limit: float = 60.0, repair: bool = False, workers: int = 8,
          hint: Optional[Dict[str, dict]] = None) -> SolveResult:
    t0 = time.time()
    plant = p.plant
    m = cp_model.CpModel()
    start, end, pres, tard = {}, {}, {}, {}
    per_machine: Dict[str, List[str]] = {k: [] for k in plant.machines}
    intervals: Dict[str, list] = {k: [] for k in plant.machines}
    fam, wc = {}, {}

    for o, op in p.pending_ops():
        fam[op.id], wc[op.id] = op.family, op.work_center
        lb = _earliest(p, o, op)
        if op.id in p.fixed:
            f = p.fixed[op.id]
            s = m.NewIntVar(f["start"], f["start"], "s" + op.id)
            e = m.NewIntVar(f["end"], f["end"], "e" + op.id)
            start[op.id], end[op.id] = s, e
            lit = m.NewBoolVar("x%s@%s" % (op.id, f["machine"]))
            m.Add(lit == 1)
            pres[(op.id, f["machine"])] = lit
            per_machine[f["machine"]].append(op.id)
            intervals[f["machine"]].append(m.NewIntervalVar(f["start"], f["end"] - f["start"], f["end"], ""))
            continue
        s = m.NewIntVar(lb, MAX_T, "s" + op.id)
        e = m.NewIntVar(lb, MAX_T, "e" + op.id)
        start[op.id], end[op.id] = s, e
        lits = []
        for mc in plant.eligible(op):
            dur = plant.duration(op, mc)
            lit = m.NewBoolVar("x%s@%s" % (op.id, mc))
            iv = m.NewOptionalIntervalVar(s, dur, e, lit, "")
            pres[(op.id, mc)] = lit
            per_machine[mc].append(op.id)
            intervals[mc].append(iv)
            lits.append(lit)
        m.AddExactlyOne(lits)

    # routing precedence
    for o in plant.orders.values():
        ops = [op for op in o.ops if op.id not in p.done]
        for a, b in zip(ops, ops[1:]):
            m.Add(start[b.id] >= end[a.id])

    # machines: no overlap with each other or with blocked time; setups via a circuit
    setup_terms = []
    arc_lits: Dict[str, Dict[tuple, object]] = {}
    for mc, ids in per_machine.items():
        for b0, b1 in p.blocked(mc):
            if b1 > p.now:
                intervals[mc].append(m.NewIntervalVar(max(b0, p.now), b1 - max(b0, p.now), b1, ""))
        m.AddNoOverlap(intervals[mc])
        if not ids or not plant.setup.get(plant.machines[mc].work_center):
            continue
        tail_end, tail_fam = p.machine_tail(mc)
        arcs = []
        empty = m.NewBoolVar("")
        arcs.append((0, 0, empty))
        node = {a: i for i, a in enumerate(ids, 1)}
        arc_lits[mc] = {"node": node, "empty": empty}
        for i, a in enumerate(ids, 1):
            lit = pres[(a, mc)]
            arcs.append((i, i, lit.Not()))
            first = m.NewBoolVar("")
            arcs.append((0, i, first))
            arc_lits[mc][(0, i)] = first
            su0 = plant.setup_minutes(wc[a], tail_fam, fam[a])
            if su0:
                m.Add(start[a] >= tail_end + su0).OnlyEnforceIf(first)
                setup_terms.append(su0 * first)
            last_arc = m.NewBoolVar("")
            arcs.append((i, 0, last_arc))
            arc_lits[mc][(i, 0)] = last_arc
            for j, b in enumerate(ids, 1):
                if i == j:
                    continue
                arc = m.NewBoolVar("")
                arcs.append((i, j, arc))
                arc_lits[mc][(i, j)] = arc
                su = plant.setup_minutes(wc[a], fam[a], fam[b])
                m.Add(start[b] >= end[a] + su).OnlyEnforceIf(arc)
                if su:
                    setup_terms.append(su * arc)
        m.AddCircuit(arcs)
        # an empty machine may only take the depot loop
        for a in ids:
            m.AddImplication(empty, pres[(a, mc)].Not())

    # repair neighbourhood: outside the affected set, keep machine and sequence
    if repair and p.previous and p.affected is not None:
        keep: Dict[str, List[str]] = {}
        for op_id, a in p.previous.items():
            if op_id in start and op_id not in p.fixed and op_id not in p.affected \
                    and (op_id, a["machine"]) in pres:
                m.Add(pres[(op_id, a["machine"])] == 1)
                keep.setdefault(a["machine"], []).append(op_id)
        for mc, ids in keep.items():
            ids.sort(key=lambda i: p.previous[i]["start"])
            for x, y in zip(ids, ids[1:]):
                m.Add(start[y] >= end[x])
            # Kept ops keep their order, so most sequence arcs between them are impossible:
            # x can only be followed directly by the next kept op or by a free op. Fixing
            # those arcs to 0 removes most of the circuit and makes repairs converge.
            lits = arc_lits.get(mc)
            if lits:
                node, rank = lits["node"], {k: i for i, k in enumerate(ids)}
                for x in ids:
                    for y in ids:
                        if x != y and rank[y] != rank[x] + 1:
                            arc = lits.get((node[x], node[y]))
                            if arc is not None:
                                m.Add(arc == 0)

    # tardiness
    wt_terms = []
    for o in plant.orders.values():
        ops = [op for op in o.ops if op.id not in p.done]
        if not ops:
            continue
        t = m.NewIntVar(0, MAX_T, "t" + o.id)
        m.Add(t >= end[ops[-1].id] - o.due)
        tard[o.id] = t
        wt_terms.append(o.weight * t)
    wt = sum(wt_terms)
    setup_total = sum(setup_terms) if setup_terms else 0

    # warm start: a complete hint (machine choice, times, sequence arcs, tardiness) from the
    # last approved schedule or the dispatching rule, so the solver starts from a known plan
    base = dict(hint or best_start(p, repair))
    for op_id, f in p.fixed.items():
        base[op_id] = f
    if all(k in base for k in start):
        for op_id in start:
            a = base[op_id]
            m.AddHint(start[op_id], a["start"])
            m.AddHint(end[op_id], a["end"])
            for mc in plant.machines:
                if (op_id, mc) in pres:
                    m.AddHint(pres[(op_id, mc)], 1 if mc == a["machine"] else 0)
        for mc, lits in arc_lits.items():
            node = lits["node"]
            seq = sorted((a for a in node if base[a]["machine"] == mc), key=lambda a: base[a]["start"])
            on = set()
            if seq:
                on.add((0, node[seq[0]]))
                on.update((node[x], node[y]) for x, y in zip(seq, seq[1:]))
                on.add((node[seq[-1]], 0))
            m.AddHint(lits["empty"], 0 if seq else 1)
            for key, lit in lits.items():
                if isinstance(key, tuple):
                    m.AddHint(lit, 1 if key in on else 0)
        for o in plant.orders.values():
            if o.id in tard:
                last = [op for op in o.ops if op.id in start][-1]
                m.AddHint(tard[o.id], max(0, base[last.id]["end"] - o.due))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    status, obj = [], {}
    budgets = [0.7, 0.3]

    def run(frac):
        # A level gets its share of the budget, or all that is left if earlier levels
        # finished early (a proven optimum rarely needs its whole share).
        left = time_limit - (time.time() - t0)
        share = time_limit * frac if frac < 1.0 else left
        solver.parameters.max_time_in_seconds = max(1.0, min(left, share) if frac < 1.0 else left)
        r = solver.Solve(m)
        status.append(solver.StatusName(r))
        return r

    snap: Dict[str, dict] = {}

    def keep_solution():
        """Hint the next level with this one, and keep it in case the next finds nothing."""
        m.ClearHints()
        allv = list(start.values()) + list(end.values()) + list(pres.values()) + list(tard.values())
        for lits in arc_lits.values():
            allv += [v for k, v in lits.items() if k != "node"]
        for v in allv:
            m.AddHint(v, solver.Value(v))
        for o, op in p.pending_ops():
            mc = next(mc for (i, mc), lit in pres.items() if i == op.id and solver.Value(lit))
            snap[op.id] = {"machine": mc, "start": solver.Value(start[op.id]),
                           "end": solver.Value(end[op.id]), "family": op.family}

    if repair and p.previous:
        # Repair, still on-time first: level 1 minimises weighted tardiness with every op
        # outside the affected set kept on its machine; level 2 minimises how far start
        # times move, with tardiness locked at its best value. Setups are respected but
        # not re-optimised here; a full re-plan does that.
        m.Minimize(wt)
        r = run(0.7)
        if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return SolveResult(best_start(p, repair), status + ["FALLBACK_HEURISTIC"], {}, time.time() - t0)
        best_wt = int(solver.ObjectiveValue())
        obj["weighted_tardiness"] = best_wt
        keep_solution()
        m.Add(wt <= best_wt)
        devs = []
        for op_id, a in p.previous.items():
            if op_id in start and op_id not in p.fixed:
                d = m.NewIntVar(0, MAX_T, "")
                m.AddAbsEquality(d, start[op_id] - a["start"])
                devs.append(d)
        if devs:
            m.Minimize(sum(devs))
            r = run(1.0)
            if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                obj["start_change_minutes"] = int(solver.ObjectiveValue())
                keep_solution()
        sched = {k: dict(v) for k, v in snap.items()}
        return SolveResult(_with_setups(p, sched), status, obj, time.time() - t0)

    # full plan: strictly lexicographic
    m.Minimize(wt)
    r = run(budgets[0])
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SolveResult(best_start(p, repair), status + ["FALLBACK_HEURISTIC"], {}, time.time() - t0)
    best_wt = int(solver.ObjectiveValue())
    obj["weighted_tardiness"] = best_wt
    obj["weighted_tardiness_bound"] = solver.BestObjectiveBound()
    keep_solution()
    m.Add(wt <= best_wt)
    m.Minimize(setup_total)
    r = run(1.0)
    if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        obj["setup_minutes"] = int(solver.ObjectiveValue())
        keep_solution()

    sched = {k: dict(v) for k, v in snap.items()}
    return SolveResult(_with_setups(p, sched), status, obj, time.time() - t0)

