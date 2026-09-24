"""Labels from normal use: what Laya read, next to what the planner actually decided.

Every Laya reading gets an id. When the planner acts on it, the decision becomes a label:

  job        reading                         label comes from                     strength
  inbox      event kind (two wordings)       the event the planner submitted      strong
             "                               dismissing the message              weak (no_action)
  command    intent                          the intent the planner picked       strong
  rejection  reason                          the next step the planner clicked   strong
  note       machine condition               the planner's ok / watch / stop     strong
             "                               "Report machine down"               strong (stop)

Each label is written as a training record (laya_train.records format: state, questions,
gold) plus what Laya answered, so `export` only has to filter. The log holds plant text:
it stays out of git (data/labels/ is ignored) and should stay inside the plant.
"""
import hashlib
import json
import os
import threading
import time
import uuid
from typing import Dict, List, Optional

from . import inbox

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.environ.get("APS_LABELS", os.path.join(HERE, "data", "labels", "labels.jsonl"))

# event kind -> the answer each inbox question should have given
KIND_ANSWER = {v: k for k, v in inbox.KIND_MAP.items()}
DOMAIN_ANSWER = {"no_action": "normal", "machine_down": "machine", "machine_degrading": "machine", "maintenance": "machine",
                 "material_late": "supply", "rush_order": "customer", "order_cancel": "customer", "due_change": "customer",
                 "quantity_change": "customer", "quality_hold": "quality"}
MACHINE_ANSWER = {v: k for k, v in inbox.MACHINE_MAP.items()}
ORDER_ANSWER = {v: k for k, v in inbox.ORDER_MAP.items()}
RERUN_REASON = {"narrow": "too_many_changes", "protect": "customer_promise"}


def inbox_gold(kind: str) -> Dict[str, str]:
    """What every inbox question should have answered for this event kind."""
    gold = {}
    if kind in KIND_ANSWER:
        gold["kind"] = KIND_ANSWER[kind]
    if kind in DOMAIN_ANSWER:
        gold["domain"] = DOMAIN_ANSWER[kind]
    if kind in MACHINE_ANSWER:
        gold["machine_issue"] = MACHINE_ANSWER[kind]
    if kind in ORDER_ANSWER:
        gold["order_change"] = ORDER_ANSWER[kind]
    return gold


class LabelLog:
    def __init__(self, path: Optional[str] = None, keep: int = 500):
        self.path = path or DEFAULT_PATH
        self.keep = keep
        self.readings: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self._summary: Optional[dict] = None     # running totals for the workbench

    # --- readings --------------------------------------------------------------------
    def register(self, job: str, text: str, questions: dict, answers: dict, model: Optional[str] = None) -> str:
        """Remember what Laya was asked and answered; returns the id a later label refers to."""
        rid = uuid.uuid4().hex[:12]
        slim = {q: {"choice": a.get("choice"), "confidence": a.get("confidence"), "probabilities": a.get("probabilities")}
                for q, a in answers.items() if q in questions}
        with self.lock:
            self.readings[rid] = {"job": job, "text": text, "questions": questions, "answers": slim, "model": model,
                                  "ts": time.time()}
            while len(self.readings) > self.keep:
                self.readings.pop(next(iter(self.readings)))
        return rid

    # --- labels ----------------------------------------------------------------------
    def label(self, reading_id: str, gold: Dict[str, str], source: str, strength: str = "strong",
              who: Optional[str] = None) -> Optional[dict]:
        """Write one training record for a reading. Unknown ids and empty gold are ignored."""
        with self.lock:
            r = self.readings.get(reading_id)
        gold = {q: g for q, g in (gold or {}).items() if r and q in r["questions"]
                and g in (r["questions"][q].get("criteria") or {})}
        if not r or not gold:
            return None
        rec = {"id": "%s-%s" % (reading_id, "-".join(sorted(gold))), "state": r["text"],
               "questions": {q: r["questions"][q] for q in gold}, "gold": gold,
               "laya": {q: r["answers"].get(q) for q in gold},
               "meta": {"job": r["job"], "source": source, "strength": strength, "model": r["model"],
                        "reading_id": reading_id, "ts": round(time.time(), 1), "by": who,
                        "laya_right": all((r["answers"].get(q) or {}).get("choice") == g for q, g in gold.items())}}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.summary()
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._summary["total"] += 1
            self._summary["laya_right"] += bool(rec["meta"]["laya_right"])
        return rec

    def summary(self) -> dict:
        """Totals for the workbench, read from the file once, then kept in memory."""
        if self._summary is None:
            recs = read(self.path)
            self._summary = {"total": len(recs), "laya_right": sum(bool(r.get("meta", {}).get("laya_right")) for r in recs)}
        return dict(self._summary)

    def label_event(self, reading_id: str, kind: str, who: Optional[str] = None) -> Optional[dict]:
        return self.label(reading_id, inbox_gold(kind), "planner_submitted", who=who)

    def stats(self) -> dict:
        """Labels so far, per job, and how often Laya's reading was right."""
        out: Dict[str, dict] = {}
        for rec in read(self.path):
            m = rec.get("meta", {})
            s = out.setdefault(m.get("job", "?"), {"labels": 0, "laya_right": 0, "weak": 0})
            s["labels"] += 1
            s["laya_right"] += bool(m.get("laya_right"))
            s["weak"] += m.get("strength") == "weak"
        return out


def read(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- export -------------------------------------------------------------------------------
def export(path: str, include_weak: bool = False, exclude_texts=()) -> List[dict]:
    """Training records from the log: the latest label per (text, question), weak labels
    only if asked, texts in `exclude_texts` (the held-out sets) never."""
    excl = {" ".join(str(t).lower().split()) for t in exclude_texts}
    latest: Dict[tuple, dict] = {}
    for rec in read(path):
        if rec.get("meta", {}).get("strength") == "weak" and not include_weak:
            continue
        key_text = " ".join(str(rec["state"]).lower().split())
        if key_text in excl:
            continue
        for qid in rec["gold"]:
            latest[(key_text, qid)] = rec
    merged: Dict[str, dict] = {}
    for (key_text, qid), rec in latest.items():
        m = merged.setdefault(key_text, {"id": rec["id"], "state": rec["state"], "questions": {}, "gold": {},
                                         "meta": {"lang": guess_lang(rec["state"]), "source": "aps." + rec["meta"]["job"]}})
        m["questions"][qid] = rec["questions"][qid]
        m["gold"][qid] = rec["gold"][qid]
    return list(merged.values())


def guess_lang(text: str) -> str:
    if any("一" <= ch <= "鿿" for ch in text):
        return "zh"
    if any(ch in "äöüßÄÖÜ" for ch in text) or any(w in text.lower().split() for w in ("der", "die", "das", "und", "ist", "nicht")):
        return "de"
    return "en"


def heldout() -> List[dict]:
    """Every hand-labelled APS message as records: the measurement sets. Never train on these."""
    here = os.path.join(HERE, "data")
    out = []
    for s in inbox.SAMPLES:
        if s["expect"] != "flagged":
            out.append({"id": "inbox-" + s["id"], "state": s["text"], "questions": dict(inbox.QUESTIONS),
                        "gold": inbox_gold(s["expect"]), "meta": {"lang": s["lang"], "source": "aps.samples"}})
    with open(os.path.join(here, "inbox_dev.json"), encoding="utf-8") as f:
        for s in json.load(f):
            out.append({"id": "inbox-" + s["id"], "state": s["text"], "questions": dict(inbox.QUESTIONS),
                        "gold": inbox_gold(s["expect"]), "meta": {"lang": s["lang"], "source": "aps.inbox_dev"}})
    for s in inbox.COMMAND_SAMPLES:
        out.append({"id": "cmd-" + hashlib.sha1(s["text"].encode()).hexdigest()[:8], "state": s["text"], "questions": {"intent": inbox.INTENT_Q},
                    "gold": {"intent": s["expect"]}, "meta": {"lang": s["lang"], "source": "aps.commands"}})
    with open(os.path.join(here, "eval_sets.json"), encoding="utf-8") as f:
        sets = json.load(f)
    for name in ("reject_dev", "reject_test"):
        for s in sets[name]:
            out.append({"id": s["id"], "state": s["text"], "questions": {"reason": inbox.REJECT_Q},
                        "gold": {"reason": s["reason"]}, "meta": {"lang": s["lang"], "source": "aps." + name}})
    for name in ("notes_dev", "notes_test"):
        for s in sets[name]:
            out.append({"id": s["id"], "state": s["text"], "questions": {"condition": inbox.NOTE_Q},
                        "gold": {"condition": s["condition"]}, "meta": {"lang": s["lang"], "source": "aps." + name}})
    return out
