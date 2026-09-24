"""Plant 1 master data as the APS sees it, and the adapter from SAP-shaped records.

The demo reads JSON files shaped like SAP S/4HANA production-order, routing and work
centre records (`data/sap/`). Field names follow the style of the S/4HANA APIs; the real
service names, fields and authorisations must be confirmed with the SAP team. Everything
after `load_plant()` works on the internal model below, so replacing the files with live
OData or IDoc calls changes this module only.

Time is in minutes from the start of the planning horizon (Monday 06:00).
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

DAY = 24 * 60
SHIFT_START, SHIFT_END = 0, 16 * 60          # 06:00-22:00, two shifts; nights unmanned
HORIZON_DAYS = 5
HORIZON = HORIZON_DAYS * DAY


@dataclass
class Machine:
    id: str
    work_center: str
    line: str
    speed: float = 1.0                        # >1 faster, <1 slower than standard


@dataclass
class Operation:
    id: str                                   # "<order>-<seq>"
    order: str
    seq: int
    work_center: str
    minutes: int                              # standard duration at speed 1.0
    family: str                               # setup family on this work centre


@dataclass
class Order:
    id: str                                   # SAP production order number
    material: str
    quantity: int
    due: int                                  # minutes from horizon start
    release: int = 0                          # material available from
    weight: int = 1                           # 1 normal, 2 key customer, 3 rush
    customer: str = ""
    ops: List[Operation] = field(default_factory=list)


@dataclass
class Plant:
    machines: Dict[str, Machine]
    work_centers: Dict[str, List[str]]        # work centre -> machine ids
    orders: Dict[str, Order]
    setup: Dict[str, Dict[str, Dict[str, int]]]   # work centre -> from family -> to family -> minutes
    blocked: Dict[str, List[Tuple[int, int]]]     # machine -> [(start, end)] unavailable

    def setup_minutes(self, wc: str, a: Optional[str], b: str) -> int:
        if a is None:
            return 0
        return self.setup.get(wc, {}).get(a, {}).get(b, 0)

    def duration(self, op: Operation, machine: str) -> int:
        return max(1, int(round(op.minutes / self.machines[machine].speed)))

    def eligible(self, op: Operation) -> List[str]:
        return self.work_centers[op.work_center]


def night_blocks() -> List[Tuple[int, int]]:
    """22:00-06:00 every night of the horizon, plus the tail after the horizon."""
    blocks = [(d * DAY + SHIFT_END, (d + 1) * DAY + SHIFT_START) for d in range(HORIZON_DAYS)]
    return blocks


def _read(name: str) -> dict:
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def _minutes(iso_day: int, hhmm: str) -> int:
    h, m = hhmm.split(":")
    return iso_day * DAY + (int(h) - 6) * 60 + int(m)


def order_from_sap(rec: dict, routings: Dict[str, dict]) -> Order:
    """One SAP-shaped production order record -> internal Order with its operations."""
    o = Order(
        id=rec["ManufacturingOrder"],
        material=rec["Material"],
        quantity=int(rec["TotalQuantity"]),
        due=_minutes(rec["DueDay"], rec["DueTime"]),
        release=_minutes(rec.get("MaterialAvailDay", 0), rec.get("MaterialAvailTime", "06:00")),
        weight=int(rec.get("Priority", 1)),
        customer=rec.get("SoldToParty", ""),
    )
    for step in routings[o.material]["operations"]:
        o.ops.append(Operation(
            id="%s-%s" % (o.id, step["Operation"]),
            order=o.id, seq=int(step["Operation"]),
            work_center=step["WorkCenter"],
            minutes=int(round(float(step["StdMinPerUnit"]) * o.quantity)),
            family=step.get("SetupFamily", o.material),
        ))
    return o


def load_plant() -> Plant:
    wcs = _read("sap/work_centers.json")["value"]
    machines, work_centers = {}, {}
    for wc in wcs:
        work_centers[wc["WorkCenter"]] = []
        for res in wc["Capacities"]:
            machines[res["Resource"]] = Machine(res["Resource"], wc["WorkCenter"], wc["Line"],
                                                float(res.get("SpeedFactor", 1.0)))
            work_centers[wc["WorkCenter"]].append(res["Resource"])
    routings = {r["Material"]: r for r in _read("sap/routings.json")["value"]}
    orders = {}
    for rec in _read("sap/production_orders.json")["value"]:
        o = order_from_sap(rec, routings)
        orders[o.id] = o
    setup = _read("setup_matrix.json")
    blocked = {m: night_blocks() for m in machines}
    return Plant(machines, work_centers, orders, setup, blocked)


def routings() -> Dict[str, dict]:
    return {r["Material"]: r for r in _read("sap/routings.json")["value"]}


def clock_label(minute: int) -> str:
    """'Mon 07:30' style label for a horizon minute."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d, rem = divmod(minute, DAY)
    h, m = divmod(rem + 6 * 60, 60)
    if h >= 24:
        d, h = d + 1, h - 24
    return "%s %02d:%02d" % (days[d % 7], h, m)
