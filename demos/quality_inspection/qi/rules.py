"""Deterministic checks on the numbers: tolerance limits and statistical process control.

These run before Laya and are authoritative for measurements. Laya is a language model
over text; it does not compare 42.071 against 42.050, and it should never be the thing
that decides whether a measured value is in tolerance.
"""
from typing import Dict, List, Optional, Sequence


def check_limits(characteristics: Dict[str, dict], measurements: Dict[str, float]) -> List[dict]:
    """One finding per configured characteristic: ok, out, or missing."""
    findings = []
    for key, spec in characteristics.items():
        value = measurements.get(key)
        f = {
            "key": key, "label": spec["label"], "unit": spec["unit"], "role": spec["role"],
            "value": value, "nominal": spec["nominal"], "lsl": spec["lsl"], "usl": spec["usl"],
            "decimals": spec.get("decimals", 3), "status": "ok", "excess": 0.0,
        }
        if value is None:
            f["status"] = "missing"
        elif spec["usl"] is not None and value > spec["usl"]:
            f["status"], f["excess"] = "out", round(value - spec["usl"], 6)
        elif spec["lsl"] is not None and value < spec["lsl"]:
            f["status"], f["excess"] = "out", round(value - spec["lsl"], 6)
        findings.append(f)
    return findings


def western_electric(values: Sequence[float], center: float, sigma: float) -> Optional[dict]:
    """The first Western Electric / Nelson rule the newest point completes, or None.

    Only rules that end on the newest point are reported, so a signal fires once when it
    appears instead of on every later point that still sits inside the window.
    """
    if sigma <= 0 or not values:
        return None
    z = [(v - center) / sigma for v in values]
    last = z[-1]
    side = 1 if last > 0 else -1

    if abs(last) > 3:
        return {"rule": "WE1", "text": "point beyond the 3-sigma control limit"}
    tail3 = z[-3:]
    if len(tail3) == 3 and sum(1 for x in tail3 if x * side > 2) >= 2 and last * side > 2:
        return {"rule": "WE2", "text": "2 of the last 3 points beyond 2 sigma on the same side"}
    tail5 = z[-5:]
    if len(tail5) == 5 and sum(1 for x in tail5 if x * side > 1) >= 4 and last * side > 1:
        return {"rule": "WE3", "text": "4 of the last 5 points beyond 1 sigma on the same side"}
    tail8 = z[-8:]
    if len(tail8) == 8 and all(x * side > 0 for x in tail8):
        return {"rule": "WE4", "text": "8 points in a row on the same side of the centre line"}
    tail6 = values[-6:]
    if len(tail6) == 6:
        steps = [b - a for a, b in zip(tail6, tail6[1:])]
        if all(s > 0 for s in steps):
            return {"rule": "TREND", "text": "6 points in a row steadily increasing"}
        if all(s < 0 for s in steps):
            return {"rule": "TREND", "text": "6 points in a row steadily decreasing"}
    return None


def spc_signals(characteristics: Dict[str, dict], history: Dict[str, Sequence[float]]) -> List[dict]:
    """Run the control rules for every SPC characteristic, over history ending in this part."""
    out = []
    for key, spec in characteristics.items():
        if not spec.get("spc"):
            continue
        values = list(history.get(key) or [])
        sig = western_electric(values, spec["nominal"], spec["sigma"])
        if sig:
            out.append({"key": key, "label": spec["label"], **sig})
    return out


def control_limits(spec: dict) -> dict:
    return {
        "center": spec["nominal"],
        "ucl": spec["nominal"] + 3 * spec["sigma"],
        "lcl": spec["nominal"] - 3 * spec["sigma"],
        "usl": spec["usl"], "lsl": spec["lsl"],
    }
