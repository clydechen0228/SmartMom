"""Write the demo's SAP-shaped master and order data to data/sap/ (deterministic).

    python -m aps.sample_data            # from demos/aps; regenerates the JSON files

Plant 1 has two lines. L1 machines housings, valve bodies and covers; L3 assembles,
leak-tests and inspects them. LKT-04 is the intended bottleneck.
"""
import json
import os
import random

from .plant import DATA

WORK_CENTERS = [
    {"WorkCenter": "WC-CNC", "Line": "L1", "Description": "CNC machining",
     "Capacities": [{"Resource": "CNC-01", "SpeedFactor": 1.0},
                    {"Resource": "CNC-02", "SpeedFactor": 1.0},
                    {"Resource": "CNC-03", "SpeedFactor": 0.8}]},
    {"WorkCenter": "WC-CMM", "Line": "L1", "Description": "Coordinate measurement",
     "Capacities": [{"Resource": "CMM-01", "SpeedFactor": 1.0}]},
    {"WorkCenter": "WC-ASM", "Line": "L3", "Description": "Assembly",
     "Capacities": [{"Resource": "ASM-03", "SpeedFactor": 1.0},
                    {"Resource": "ASM-05", "SpeedFactor": 1.1}]},
    {"WorkCenter": "WC-LKT", "Line": "L3", "Description": "Leak test",
     "Capacities": [{"Resource": "LKT-04", "SpeedFactor": 1.0}]},
    {"WorkCenter": "WC-AOI", "Line": "L3", "Description": "Final vision inspection",
     "Capacities": [{"Resource": "AOI-01", "SpeedFactor": 1.0}]},
]

# Standard minutes per unit by operation. SetupFamily drives the setup matrix.
ROUTINGS = [
    {"Material": "HSG-7731", "Description": "Pump housing", "operations": [
        {"Operation": "0010", "WorkCenter": "WC-CNC", "StdMinPerUnit": 2.0, "SetupFamily": "HSG-A"},
        {"Operation": "0020", "WorkCenter": "WC-CMM", "StdMinPerUnit": 0.5, "SetupFamily": "HSG"},
        {"Operation": "0030", "WorkCenter": "WC-ASM", "StdMinPerUnit": 1.2, "SetupFamily": "HSG-7731"},
        {"Operation": "0040", "WorkCenter": "WC-LKT", "StdMinPerUnit": 1.1, "SetupFamily": "HSG"},
        {"Operation": "0050", "WorkCenter": "WC-AOI", "StdMinPerUnit": 0.3, "SetupFamily": "ALL"}]},
    {"Material": "HSG-7735", "Description": "Pump housing, long", "operations": [
        {"Operation": "0010", "WorkCenter": "WC-CNC", "StdMinPerUnit": 2.4, "SetupFamily": "HSG-B"},
        {"Operation": "0020", "WorkCenter": "WC-CMM", "StdMinPerUnit": 0.6, "SetupFamily": "HSG"},
        {"Operation": "0030", "WorkCenter": "WC-ASM", "StdMinPerUnit": 1.4, "SetupFamily": "HSG-7735"},
        {"Operation": "0040", "WorkCenter": "WC-LKT", "StdMinPerUnit": 1.1, "SetupFamily": "HSG"},
        {"Operation": "0050", "WorkCenter": "WC-AOI", "StdMinPerUnit": 0.3, "SetupFamily": "ALL"}]},
    {"Material": "VLV-2200", "Description": "Valve body", "operations": [
        {"Operation": "0010", "WorkCenter": "WC-CNC", "StdMinPerUnit": 1.6, "SetupFamily": "VLV"},
        {"Operation": "0020", "WorkCenter": "WC-CMM", "StdMinPerUnit": 0.4, "SetupFamily": "VLV"},
        {"Operation": "0030", "WorkCenter": "WC-LKT", "StdMinPerUnit": 1.25, "SetupFamily": "VLV"},
        {"Operation": "0040", "WorkCenter": "WC-AOI", "StdMinPerUnit": 0.3, "SetupFamily": "ALL"}]},
    {"Material": "CVR-1100", "Description": "Cover", "operations": [
        {"Operation": "0010", "WorkCenter": "WC-CNC", "StdMinPerUnit": 0.9, "SetupFamily": "CVR"},
        {"Operation": "0020", "WorkCenter": "WC-AOI", "StdMinPerUnit": 0.2, "SetupFamily": "ALL"}]},
]

# Minutes to change over between setup families, per work centre (APS-maintained;
# SAP has no standard object for sequence-dependent setups).
SETUP = {
    "WC-CNC": {
        "HSG-A": {"HSG-B": 15, "VLV": 60, "CVR": 45},
        "HSG-B": {"HSG-A": 15, "VLV": 60, "CVR": 45},
        "VLV": {"HSG-A": 60, "HSG-B": 60, "CVR": 40},
        "CVR": {"HSG-A": 45, "HSG-B": 45, "VLV": 40},
    },
    "WC-CMM": {"HSG": {"VLV": 20}, "VLV": {"HSG": 20}},
    "WC-ASM": {"HSG-7731": {"HSG-7735": 25}, "HSG-7735": {"HSG-7731": 25}},
    "WC-LKT": {"HSG": {"VLV": 30}, "VLV": {"HSG": 30}},
}

CUSTOMERS = [("C-1001", 2), ("C-1002", 1), ("C-1003", 1), ("C-1004", 2), ("C-1005", 1)]
MIX = [("HSG-7731", 0.35, (60, 180)), ("HSG-7735", 0.2, (40, 140)),
       ("VLV-2200", 0.25, (60, 200)), ("CVR-1100", 0.2, (80, 260))]
DUE_TIMES = ["14:00", "21:00"]


def production_orders(n: int = 36, seed: int = 20260923) -> list:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        r, acc = rng.random(), 0.0
        for mat, p, (lo, hi) in MIX:
            acc += p
            if r <= acc:
                break
        cust, weight = rng.choice(CUSTOMERS)
        qty = rng.randrange(lo, hi + 1, 10)
        due_day = min(4, 1 + i * 4 // n + rng.choice([0, 0, 1]))
        rec = {
            "ManufacturingOrder": str(10004400 + i + 1),
            "Material": mat, "ProductionPlant": "1000", "TotalQuantity": qty,
            "SoldToParty": cust, "Priority": weight,
            "DueDay": due_day, "DueTime": rng.choice(DUE_TIMES),
        }
        if rng.random() < 0.2:                     # material arrives later in the week
            rec["MaterialAvailDay"] = rng.choice([0, 1])
            rec["MaterialAvailTime"] = rng.choice(["10:00", "14:00"])
        out.append(rec)
    return out


def write():
    os.makedirs(os.path.join(DATA, "sap"), exist_ok=True)
    files = {
        "sap/work_centers.json": {"value": WORK_CENTERS},
        "sap/routings.json": {"value": ROUTINGS},
        "sap/production_orders.json": {"value": production_orders()},
        "setup_matrix.json": SETUP,
    }
    for name, body in files.items():
        with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
            json.dump(body, f, indent=1, ensure_ascii=False)
            f.write("\n")
    print("wrote", ", ".join(files))


if __name__ == "__main__":
    write()
