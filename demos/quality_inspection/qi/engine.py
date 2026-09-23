"""The Laya side: one Router kept resident, reading each operator note.

`MockEngine` is a keyword stand-in with the same output shape, for running the UI on a
machine without the weights. It is labelled as such everywhere it shows up; nothing it
produces should be read as Laya's behaviour.
"""
import os
import re
import sys
import threading
import time
from typing import Dict, Optional

from .inspection import QUESTIONS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class LayaEngine:
    kind = "laya"

    def __init__(self, device: Optional[str] = None):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("USE_TF", "0")
        # torch 2.2 forks a compile worker per core when a checkpoint loads, although
        # Laya never compiles. In a server those forks inherit the listening socket and
        # outlive a crash, keeping the port bound. One thread means no pool.
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        sys.path.insert(0, os.path.join(ROOT, "examples"))
        import laya
        from laya import Router
        from examples_common import pick_device

        self.version = laya.__version__
        self.device = device or pick_device()
        # max_loaded=3: a note in Hindi routes to the multilingual checkpoint, and that
        # must not evict the English one the next record needs.
        self.router = Router(device=self.device, max_loaded=3)
        self.state = "cold"
        self.error: Optional[str] = None
        self.load_ms: Dict[str, float] = {}

    def warm(self, models=("english", "multilingual")):
        """Load both checkpoints up front: notes arrive in any language, and a cold load
        on the first German note would be a 10-15 s outlier."""
        self.state = "loading"
        try:
            for m in models:
                t0 = time.time()
                self.router.load(m)
                self.load_ms[m] = round((time.time() - t0) * 1000)
            self.state = "ready"
        except Exception as e:             # no network, no cache, out of memory
            self.state, self.error = "error", "%s: %s" % (type(e).__name__, e)

    def warm_async(self):
        threading.Thread(target=self.warm, name="laya-warm", daemon=True).start()

    def predict(self, text: str) -> dict:
        route = self.router.route(text, QUESTIONS)
        cold = None
        if route.model not in self.router.loaded:
            t0 = time.time()
            self.router.load(route.model)
            cold = round((time.time() - t0) * 1000)
        t0 = time.time()
        res = self.router.predict(text, QUESTIONS)
        return {
            "answers": res["answers"],
            "routing": dict(res.get("routing") or route),
            "ms": round((time.time() - t0) * 1000, 1),
            "cold_load_ms": cold,
        }

    def info(self) -> dict:
        return {"kind": self.kind, "version": self.version, "device": self.device,
                "state": self.state, "error": self.error, "loaded": self.router.loaded,
                "load_ms": self.load_ms}


# --------------------------------------------------------------------------------------
_KEYWORDS = {
    "topic": [
        ("measurement_problem", r"calibrat|probe|gauge|suspect"),
        ("material_problem", r"lot\b|charge|batch|supplier|lieferant|casting|gie(ß|ss)erei"),
        ("handling_problem", r"tray|bandeja|packag|transport"),
        ("machine_problem", r"tool|spindle|chatter|wear|fixture|打滑|拧紧枪"),
        ("part_defect", r"dent|scratch|raya|porosit|damage|crack|burr|扭矩不足"),
    ],
    "defect_kind": [
        ("leak", r"leak|porosit|seal"),
        ("assembly", r"torque|bolt|螺栓|loose|missing"),
        ("surface", r"dent|scratch|raya|burr|mark"),
        ("dimensional", r"diameter|bore|flatness|size"),
        ("contamination", r"oil|chips|dirt|particle"),
    ],
}


def _dist(options, winner, peak):
    rest = (1 - peak) / (len(options) - 1)
    return {o: round(peak if o == winner else rest, 4) for o in options}


class MockEngine:
    """Keyword heuristics in Laya's output format. For UI work only."""
    kind = "mock"
    version = "mock"
    device = "none"
    state = "ready"
    error = None

    def warm_async(self):
        pass

    def predict(self, text: str) -> dict:
        t0 = time.time()
        low = text.lower()
        answers = {}
        for qid, fallback in (("topic", "nothing"), ("defect_kind", "none")):
            options = list(QUESTIONS[qid]["criteria"])
            winner, peak = fallback, 0.8
            for label, pat in _KEYWORDS[qid]:
                if re.search(pat, low):
                    winner, peak = label, 0.75
                    break
            probs = _dist(options, winner, peak)
            answers[qid] = {"type": "choice", "choice": winner, "probabilities": probs,
                            "confidence": round(peak - sorted(probs.values())[-2], 4)}
        time.sleep(0.05)
        return {"answers": answers,
                "routing": {"model": "mock", "reason": "mock engine - keyword heuristics, not Laya"},
                "ms": round((time.time() - t0) * 1000, 1), "cold_load_ms": None}

    def info(self) -> dict:
        return {"kind": "mock", "version": "mock", "device": "none", "state": "ready",
                "error": None, "loaded": [], "load_ms": {}}
