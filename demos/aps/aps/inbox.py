"""The disruption inbox: written messages -> typed scheduling events.

Two parts with separate jobs, as in the design:
  * Laya classifies what the message is about (event type, urgency), with calibrated
    confidence, in whatever language it is written.
  * Rules find the concrete fields: machine and order numbers have fixed formats and are
    checked against master data; hours, percentages and quantities are patterns.
Below the confidence gate, or with a required field missing, the planner completes the
event before anything is re-planned.
"""
import os
import re
import sys
import threading
import time
from typing import Dict, List, Optional

from .state import REQUIRED

GATE = 0.50

# Option names and wording measured on the sample inbox (evaluate.py): this wording got
# 6 of 8 right with both misses below the gate; an earlier wording read "camera cleaned,
# no impact" (zh) as a late delivery at 0.86 confidence. Keys are what Laya reads, so
# they are plain words; KIND maps them onto the scheduler's event kinds.
QUESTIONS = {
    "event_type": {
        "type": "choice",
        "instructions": "What does the message report?",
        "criteria": {
            "normal": "everything is normal, no problem, nothing to do",
            "machine_stopped": "a machine has stopped or broken down",
            "machine_slow": "a machine is running slower or noisy and needs maintenance",
            "delivery_late": "a supplier delivery arrives late",
            "urgent_order": "a customer wants an extra urgent order",
            "quality_block": "parts failed a quality test and are blocked",
        },
    },
    "urgency": {
        "type": "choice",
        "instructions": "How soon does production need to react?",
        "criteria": {
            "today": "now or within this shift",
            "this_week": "in the coming days",
            "info": "no reaction needed",
        },
    },
}
KIND = {"normal": "no_action", "machine_stopped": "machine_down", "machine_slow": "machine_degrading",
        "delivery_late": "material_late", "urgent_order": "rush_order", "quality_block": "quality_hold"}

# Sample messages for the demo, with the event each should become.
SAMPLES = [
    {"id": "m1", "lang": "en", "from": "maintenance",
     "text": "LKT-04 pressure sensor failed, the leak tester is stopped. Maintenance estimates 6 hours.",
     "expect": "machine_down"},
    {"id": "m2", "lang": "zh", "from": "设备科",
     "text": "CNC-02 主轴异响，振动越来越大，目前速度降低约30%，建议本周安排维修。",
     "expect": "machine_degrading"},
    {"id": "m3", "lang": "de", "from": "Einkauf",
     "text": "Die Gussteile für Auftrag 10004430 kommen erst in 24 Std., Lieferverzug beim Lieferanten.",
     "expect": "material_late"},
    {"id": "m4", "lang": "en", "from": "sales",
     "text": "Customer C-1001 needs 80 extra HSG-7731 housings within 30 hours, please add a rush order.",
     "expect": "rush_order"},
    {"id": "m5", "lang": "zh", "from": "质量部",
     "text": "订单 10004405 的产品泄漏测试不合格，质量部已冻结该批次，预计8小时后给出结论。",
     "expect": "quality_hold"},
    {"id": "m6", "lang": "de", "from": "Schichtleitung",
     "text": "CNC-03 läuft wieder normal, keine Maßnahmen nötig.",
     "expect": "no_action"},
    {"id": "m7", "lang": "en", "from": "HR",
     "text": "Reminder: the canteen closes early on Friday.",
     "expect": "no_action"},
    {"id": "m8", "lang": "zh", "from": "班组长",
     "text": "AOI-01 相机镜头已清洁完毕，不影响生产。",
     "expect": "no_action"},
]

_MACHINE = re.compile(r"\b((?:CNC|CMM|ASM|LKT|AOI)-\d{2})\b")
_ORDER = re.compile(r"(?<!\d)(1000\d{4})(?!\d)")
_MATERIAL = re.compile(r"\b(HSG-7731|HSG-7735|VLV-2200|CVR-1100)\b")
_HOURS = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:h\b|hours?|hrs?|小时|个小时|Std\.?|Stunden)", re.I)
_PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_QTY = re.compile(r"(\d+)\s*(?:extra|pcs|pieces|units|Stück|件|个)?\s*(?:HSG|VLV|CVR)", re.I)
_TOMORROW = re.compile(r"tomorrow|明天|morgen", re.I)


def _num(x: str) -> float:
    return float(x.replace(",", "."))


def extract(text: str, machines, orders) -> Dict[str, object]:
    """Fields found by pattern, kept only when they exist in master data."""
    out: Dict[str, object] = {}
    m = [x for x in _MACHINE.findall(text) if x in machines]
    if m:
        out["machine"] = m[0]
    o = [x for x in _ORDER.findall(text) if x in orders]
    if o:
        out["order"] = o[0]
    mat = _MATERIAL.findall(text)
    if mat:
        out["material"] = mat[0]
    h = _HOURS.findall(text)
    if h:
        out["hours"] = _num(h[0])
    elif _TOMORROW.search(text):
        out["hours"] = 24.0
    pc = _PERCENT.findall(text)
    if pc:
        out["percent"] = _num(pc[0])
    q = _QTY.findall(text)
    if q:
        out["quantity"] = int(q[0])
    return out


def to_event(kind: str, fields: Dict[str, object]) -> dict:
    """Map extracted fields onto the event's required fields; report what is missing."""
    ev = {"kind": kind}
    if kind == "rush_order":
        ev.update({k: fields.get(k) for k in ("material", "quantity")})
        ev["due_hours"] = fields.get("hours")
    elif kind in REQUIRED:
        ev.update({k: fields.get(k) for k in REQUIRED[kind]})
    missing = [k for k in REQUIRED.get(kind, ()) if ev.get(k) in (None, "")]
    return {"event": ev, "missing": missing}


# ---------------------------------------------------------------------------------------
class LayaClassifier:
    kind = "laya"

    def __init__(self, device: Optional[str] = None, router=None):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")   # see quality demo
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        for pth in (root, os.path.join(root, "examples")):
            if pth not in sys.path:
                sys.path.insert(0, pth)
        import laya
        from laya import Router
        from examples_common import pick_device
        self.version = laya.__version__
        self.device = device or pick_device()
        self.router = router or Router(device=self.device, max_loaded=3)
        self.state, self.error = "cold", None

    def warm(self):
        self.state = "loading"
        try:
            for m in ("english", "multilingual"):
                self.router.load(m)
            self.state = "ready"
        except Exception as e:
            self.state, self.error = "error", "%s: %s" % (type(e).__name__, e)

    def warm_async(self):
        threading.Thread(target=self.warm, daemon=True).start()

    def classify(self, text: str) -> dict:
        t0 = time.time()
        res = self.router.predict(text, QUESTIONS)
        return {"answers": res["answers"], "routing": dict(res.get("routing") or {}),
                "ms": round((time.time() - t0) * 1000, 1)}

    def info(self):
        return {"kind": self.kind, "version": self.version, "device": self.device,
                "state": self.state, "error": self.error}


_KEYWORDS = [
    ("machine_down", r"stopped|failed|broken|down\b|停机|故障|停止|steht|ausgefallen"),
    ("machine_degrading", r"slower|noisy|vibration|降低|异响|振动|langsamer|Geräusch"),
    ("material_late", r"delay|late|延迟|推迟|Verzug|kommen erst"),
    ("rush_order", r"rush|urgent|extra|插单|加急|Eilauftrag"),
    ("quality_hold", r"quality|failed .*test|冻结|不合格|Qualität|gesperrt"),
]


class MockClassifier:
    """Keyword stand-in in Laya's output format, for UI work without the weights."""
    kind, version, device, state, error = "mock", "mock", "none", "ready", None

    def warm_async(self):
        pass

    def classify(self, text: str) -> dict:
        low = text.lower()
        winner = "no_action"
        for label, pat in _KEYWORDS:
            if re.search(pat, low, re.I):
                winner = label
                break
        if re.search(r"normal|keine maßnahmen|不影响|nothing", low):
            winner = "no_action"
        key = {v: k for k, v in KIND.items()}[winner]
        probs = {o: (0.75 if o == key else 0.05) for o in QUESTIONS["event_type"]["criteria"]}
        return {"answers": {"event_type": {"choice": key, "probabilities": probs, "confidence": 0.7},
                            "urgency": {"choice": "info" if winner == "no_action" else "today",
                                        "probabilities": {}, "confidence": 0.6}},
                "routing": {"model": "mock", "reason": "mock classifier - keyword heuristics, not Laya"},
                "ms": 1.0}

    def info(self):
        return {"kind": "mock", "version": "mock", "device": "none", "state": "ready", "error": None}


def read_message(classifier, text: str, machines, orders) -> dict:
    """Classify + extract + map to an event, with the gate applied."""
    res = classifier.classify(text)
    et = res["answers"]["event_type"]
    fields = extract(text, machines, orders)
    kind, conf = KIND.get(et["choice"], et["choice"]), et["confidence"]
    out = {"text": text, "classification": res, "event_type": kind, "confidence": conf,
           "fields": fields, "gate": GATE, "needs_planner": conf < GATE}
    if kind == "no_action":
        out.update(event=None, missing=[])
    else:
        mapped = to_event(kind, fields)
        out.update(mapped)
        out["needs_planner"] = out["needs_planner"] or bool(mapped["missing"])
    return out


def samples() -> List[dict]:
    return [dict(s) for s in SAMPLES]
