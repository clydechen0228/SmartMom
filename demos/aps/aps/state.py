"""Planning state: the approved schedule, the clock, events, proposals and versions.

A disruption never edits the approved schedule directly. It is applied to a copy of the
plant, the copy is repaired, and the result is a *proposal* the planner approves or
rejects. Approval makes the copy current and records a schedule version.
"""
import copy
import itertools
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .checker import check
from .kpis import kpis, order_status
from .plant import Plant, clock_label, load_plant, order_from_sap, routings
from .solver import Problem, dispatch_edd, solve

FREEZE_MIN = 120                  # operations starting within this window do not move
FULL_LIMIT_S = float(os.environ.get("APS_FULL_S", 60))       # full plan time limit
REPAIR_LIMIT_S = float(os.environ.get("APS_REPAIR_S", 30))   # repair time limit

EVENT_KINDS = ("machine_down", "machine_degrading", "material_late", "rush_order", "quality_hold")
REQUIRED = {
    "machine_down": ("machine", "hours"),
    "machine_degrading": ("machine", "percent"),
    "material_late": ("order", "hours"),
    "rush_order": ("material", "quantity", "due_hours"),
    "quality_hold": ("order", "hours"),
}


class EventError(ValueError):
    pass


@dataclass
class Proposal:
    id: int
    event: dict
    plant: Plant
    holds: Dict[str, int]
    fixed: Dict[str, dict]
    schedule: Dict[str, dict]
    kpis: dict
    before: dict
    status: List[str]
    objective: dict
    seconds: float
    violations: List[str]


@dataclass
class Version:
    id: int
    ts: float
    reason: str
    approver: str
    kpis: dict
    schedule: Dict[str, dict] = field(repr=False)


class Planning:
    def __init__(self, plant: Optional[Plant] = None):
        self.lock = threading.RLock()
        self.plant = plant or load_plant()
        self.now = 0
        self.done: Dict[str, dict] = {}
        self.holds: Dict[str, int] = {}
        self.schedule: Dict[str, dict] = {}
        self.baseline: Dict[str, dict] = {}
        self.proposal: Optional[Proposal] = None
        self.versions: List[Version] = []
        self.events: List[dict] = []
        self.job: Optional[dict] = None
        self._ids = itertools.count(1)
        self._next_order = max(int(o) for o in self.plant.orders) + 1

    def __deepcopy__(self, memo):
        """Copy the planning state (for what-if trials); the copy gets its own lock."""
        new = Planning.__new__(Planning)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "lock":
                new.lock = threading.RLock()
            elif k == "_ids":
                new._ids = itertools.count(len(self.events) + 1)
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new

    # --- helpers ------------------------------------------------------------------
    def frozen(self, schedule=None, now=None) -> Dict[str, dict]:
        """Running ops, and ops starting inside the frozen window."""
        schedule = self.schedule if schedule is None else schedule
        now = self.now if now is None else now
        return {k: dict(a) for k, a in schedule.items()
                if k not in self.done and a["start"] < now + FREEZE_MIN}

    def problem(self, plant=None, holds=None, fixed=None, previous=None) -> Problem:
        return Problem(plant or self.plant, self.now, self.done,
                       self.frozen() if fixed is None else fixed,
                       self.holds if holds is None else holds,
                       self.schedule if previous is None else previous)

    # --- full plan ----------------------------------------------------------------
    def full_plan(self, time_limit: Optional[float] = None, approver: str = "system") -> Version:
        # Inputs are copied under the lock and the solve runs without it, so the
        # workbench can keep reading state during a 60 s solve.
        with self.lock:
            p = Problem(copy.deepcopy(self.plant), self.now, dict(self.done), self.frozen(),
                        dict(self.holds), {})
        baseline = dispatch_edd(p)
        r = solve(p, time_limit=time_limit or FULL_LIMIT_S, hint=baseline)
        viol = check(p.plant, r.schedule, p.now, p.done, p.fixed, p.holds)
        if viol:
            raise RuntimeError("full plan infeasible: %s" % viol[:3])
        with self.lock:
            self.baseline = baseline
            self.schedule = r.schedule
            v = self._version("full plan (%s)" % "/".join(r.status), approver)
            v.kpis.update(solve_s=round(r.seconds, 1), status=r.status)
            v.kpis["edd_baseline"] = kpis(p.plant, baseline, p.done)
            return v

    def _version(self, reason: str, approver: str) -> Version:
        v = Version(len(self.versions) + 1, time.time(), reason, approver,
                    kpis(self.plant, self.schedule, self.done), copy.deepcopy(self.schedule))
        self.versions.append(v)
        return v

    # --- events -------------------------------------------------------------------
    def validate(self, ev: dict) -> dict:
        kind = ev.get("kind")
        if kind not in EVENT_KINDS:
            raise EventError("unknown event kind %r" % kind)
        missing = [k for k in REQUIRED[kind] if ev.get(k) in (None, "")]
        if missing:
            raise EventError("missing %s" % ", ".join(missing))
        if "machine" in REQUIRED[kind] and ev["machine"] not in self.plant.machines:
            raise EventError("unknown machine %s" % ev["machine"])
        if "order" in REQUIRED[kind] and ev["order"] not in self.plant.orders:
            raise EventError("unknown order %s" % ev["order"])
        if "material" in REQUIRED[kind] and ev["material"] not in routings():
            raise EventError("unknown material %s" % ev["material"])
        for k in ("hours", "percent", "quantity", "due_hours"):
            if k in ev and ev[k] not in (None, ""):
                ev[k] = float(ev[k])
                if ev[k] <= 0:
                    raise EventError("%s must be positive" % k)
        if kind == "machine_degrading" and ev["percent"] >= 90:
            raise EventError("percent must be below 90")
        return ev

    def apply(self, ev: dict, plant: Plant, holds: Dict[str, int], fixed: Dict[str, dict]):
        """Change a copy of the plant (and holds / frozen set) for one event."""
        kind, now = ev["kind"], self.now

        def release_order(order_id):
            for op in plant.orders[order_id].ops:
                fixed.pop(op.id, None)

        if kind == "machine_down":
            mc, until = ev["machine"], now + int(ev["hours"] * 60)
            plant.blocked[mc] = sorted(plant.blocked[mc] + [(now, until)])
            for k in [k for k, a in fixed.items() if a["machine"] == mc]:
                fixed.pop(k)                         # interrupted work restarts after repair
        elif kind == "machine_degrading":
            mc = ev["machine"]
            plant.machines[mc].speed = round(plant.machines[mc].speed * (1 - ev["percent"] / 100), 4)
        elif kind == "material_late":
            o = plant.orders[ev["order"]]
            first = o.ops[0].id
            if first in self.done or (first in fixed and fixed[first]["start"] < now):
                raise EventError("order %s has already started" % o.id)
            o.release = max(o.release, now + int(ev["hours"] * 60))
            release_order(o.id)
        elif kind == "rush_order":
            oid = str(self._next_order)
            rec = {"ManufacturingOrder": oid, "Material": ev["material"], "TotalQuantity": int(ev["quantity"]),
                   "Priority": 3, "SoldToParty": ev.get("customer", "RUSH"), "DueDay": 0, "DueTime": "06:00"}
            o = order_from_sap(rec, routings())
            o.due, o.release = now + int(ev["due_hours"] * 60), now
            plant.orders[oid] = o
            ev["order"] = oid
        elif kind == "quality_hold":
            o = plant.orders[ev["order"]]
            holds[o.id] = now + int(ev["hours"] * 60)
            release_order(o.id)

    def propose(self, ev: dict, time_limit: Optional[float] = None) -> Proposal:
        with self.lock:
            ev = self.validate(dict(ev))
            plant, holds, fixed = copy.deepcopy(self.plant), dict(self.holds), self.frozen()
            self.apply(ev, plant, holds, fixed)
            self._unfreeze_successors(plant, fixed)
            p = Problem(plant, self.now, dict(self.done), fixed, holds, copy.deepcopy(self.schedule))
            p.affected = self.affected(ev, plant, p)
            if ev["kind"] == "rush_order":
                self._next_order += 1
        r = solve(p, time_limit=time_limit or REPAIR_LIMIT_S, repair=True)
        viol = check(plant, r.schedule, p.now, p.done, fixed, holds, plant.blocked)
        with self.lock:
            prop = Proposal(next(self._ids), ev, plant, holds, fixed, r.schedule,
                            kpis(plant, r.schedule, self.done, self.schedule),
                            kpis(self.plant, self.schedule, self.done),
                            r.status, r.objective, round(r.seconds, 1), viol)
            self.proposal = prop
            self.events.append({"ts": time.time(), "at": self.now, "event": ev, "proposal": prop.id,
                                "outcome": "proposed", "affected": len(p.affected)})
            return prop

    def _unfreeze_successors(self, plant: Plant, fixed: Dict[str, dict]):
        """A frozen op cannot stay put once an earlier op of its order is free to move."""
        for o in plant.orders.values():
            released = False
            for op in o.ops:
                if op.id in self.done:
                    continue
                if op.id not in fixed:
                    released = True
                elif released and fixed[op.id]["start"] >= self.now:
                    fixed.pop(op.id)

    def affected(self, ev: dict, plant: Plant, p: Problem) -> set:
        """Operations a repair may re-optimise freely: those on the disrupted work
        centre or order, plus every op of an order the disruption puts at risk."""
        aff = set()
        wcs, orders = set(), set()
        if ev.get("machine"):
            wcs.add(plant.machines[ev["machine"]].work_center)
        if ev.get("order"):
            orders.add(ev["order"])
        for o in plant.orders.values():
            for op in o.ops:
                if op.work_center in wcs or o.id in orders or op.id not in self.schedule:
                    aff.add(op.id)
        # orders that a sequence-keeping repair would make late get re-optimised too
        from .solver import repair_heuristic
        quick = repair_heuristic(Problem(p.plant, p.now, p.done, p.fixed, p.holds, p.previous))
        for o in plant.orders.values():
            ends = [quick[op.id]["end"] for op in o.ops if op.id in quick]
            if ends and max(ends) > o.due:
                aff.update(op.id for op in o.ops)
        return aff

    def decide(self, proposal_id: int, approve: bool, who: str) -> Optional[Version]:
        with self.lock:
            if not self.proposal or self.proposal.id != proposal_id:
                raise EventError("no such open proposal")
            prop, self.proposal = self.proposal, None
            for e in self.events:
                if e.get("proposal") == prop.id:
                    e["outcome"] = "approved" if approve else "rejected"
                    e["by"] = who
            if not approve:
                return None
            if prop.violations:
                raise EventError("proposal has violations and cannot be approved")
            self.plant, self.holds, self.schedule = prop.plant, prop.holds, prop.schedule
            return self._version("%s approved" % prop.event["kind"], who)

    # --- clock --------------------------------------------------------------------
    def advance(self, minutes: int) -> dict:
        """Move the clock; the MES confirms everything that finished meanwhile."""
        with self.lock:
            if self.proposal:
                raise EventError("approve or reject the open proposal first")
            self.now += int(minutes)
            confirmed = 0
            for k, a in self.schedule.items():
                if k not in self.done and a["end"] <= self.now:
                    self.done[k] = dict(a)
                    confirmed += 1
            for k in list(self.schedule):
                if k in self.done:
                    del self.schedule[k]
            return {"now": self.now, "now_label": clock_label(self.now), "confirmed": confirmed}

    # --- views --------------------------------------------------------------------
    def all_ops(self, schedule=None) -> Dict[str, dict]:
        return {**self.done, **(self.schedule if schedule is None else schedule)}

    def orders_view(self, plant=None, schedule=None) -> List[dict]:
        plant = plant or self.plant
        st = order_status(plant, self.all_ops(schedule))
        return sorted(st.values(), key=lambda s: (s["due"], s["order"]))

    def dispatch_lists(self) -> Dict[str, list]:
        """What the MES receives on publish: the sequence per machine, not yet done."""
        out: Dict[str, list] = {m: [] for m in self.plant.machines}
        ops = {op.id: (o, op) for o in self.plant.orders.values() for op in o.ops}
        for k, a in sorted(self.schedule.items(), key=lambda kv: kv[1]["start"]):
            o, op = ops[k]
            out[a["machine"]].append({"order": o.id, "operation": "%04d" % op.seq, "material": o.material,
                                      "quantity": o.quantity, "planned_start": clock_label(a["start"]),
                                      "planned_end": clock_label(a["end"]), "setup_min": a.get("setup", 0)})
        return out

    def sap_updates(self) -> List[dict]:
        """Scheduled start and finish per production order, as they would be written back."""
        rows = []
        for s in self.orders_view():
            if s["finish"] is None:
                continue
            starts = [self.all_ops()[op.id]["start"] for op in self.plant.orders[s["order"]].ops]
            rows.append({"ManufacturingOrder": s["order"], "MfgOrderScheduledStartDate": clock_label(min(starts)),
                         "MfgOrderScheduledEndDate": s["finish_label"]})
        return rows

