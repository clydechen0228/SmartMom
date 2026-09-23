"""From one telemetry message to a disposition.

    telemetry -> check_limits + spc_signals -> Laya on the operator note -> decide

The split of responsibility is the point of the demo:

* Sensors and machines speak in numbers and codes. Numbers are judged by rules
  (rules.py); alarm codes and vision labels come from fixed vocabularies and are mapped
  by table (config.ALARMS). Deterministic, auditable, never a model.
* People speak in words, in whatever language the shift speaks. The operator's note is
  what Laya reads: one forward pass, typed questions, calibrated probabilities, routed
  to the English or multilingual checkpoint by the note's script and language.
* The policy below combines the two and decides what a machine may do on its own and
  what goes to a person.

Why Laya does not read the whole record: measured on this line, a report that mixes
"Leak rate 0.61 ml/min, within tolerance" with the note made Laya answer 'leak' for
healthy parts and miss defects the note described. Given the note alone it classifies
reliably, in 0.2-0.7 s instead of 2.5 s on a laptop CPU. evaluate.py keeps the numbers.
"""
from typing import Dict, List, Optional

from .config import ALARMS, POLICY

# Asked of the operator note, in one forward pass. Choice questions only: on the
# multilingual checkpoint, yes/no (noul) questions read near 0 for notes that plainly
# report a problem, while the same content as a choice is classified correctly.
QUESTIONS = {
    "topic": {
        "type": "choice",
        "instructions": "What does the operator report?",
        "criteria": {
            "nothing": "normal operation, no problem",
            "part_defect": "a defect or damage on the part",
            "machine_problem": "a tool, machine or equipment problem",
            "material_problem": "a supplier or material lot problem",
            "measurement_problem": "a gauge, probe or measurement problem",
            "handling_problem": "damage from trays, packaging or transport",
        },
    },
    "defect_kind": {
        "type": "choice",
        "instructions": "What kind of defect is described?",
        "criteria": {
            "none": "no defect",
            "dimensional": "wrong size, diameter or shape",
            "surface": "scratch, dent or mark on the surface",
            "assembly": "loose, missing or wrongly tightened part",
            "leak": "leak, porosity or seal problem",
            "contamination": "dirt, oil or chips",
        },
    },
}

TOPIC_CAUSE = {"machine_problem": "machine", "material_problem": "material",
               "measurement_problem": "measurement", "handling_problem": "handling"}
DISPOSITIONS = ("pass", "rework", "scrap", "hold")
NOTE_CHARS = 600           # well inside the 512-token English context


def _fmt(v: Optional[float], decimals: int) -> str:
    return "n/a" if v is None else ("%.*f" % (decimals, v))


def _limits_text(f: dict) -> str:
    d, u = f["decimals"], (" " + f["unit"]) if f["unit"] else ""
    if f["lsl"] is not None and f["usl"] is not None:
        return "%s to %s%s" % (_fmt(f["lsl"], d), _fmt(f["usl"], d), u)
    if f["usl"] is not None:
        return "at most %s%s" % (_fmt(f["usl"], d), u)
    return "at least %s%s" % (_fmt(f["lsl"], d), u)


def note_text(msg: dict) -> str:
    """What Laya reads: the operator's note, trimmed; empty when there is none."""
    return ((msg.get("operator_note") or {}).get("text") or "").strip()[:NOTE_CHARS]


def build_report(station_id: str, station: dict, product: str, msg: dict,
                 findings: List[dict], spc: List[dict]) -> str:
    """The whole record in one paragraph, for the audit trail and the reviewer."""
    lines = ["Station %s %s, part %s, serial %s." % (station_id, station["name"], product, msg["serial"])]
    for f in findings:
        d, u = f["decimals"], (" " + f["unit"]) if f["unit"] else ""
        kind = "process limit" if f["role"] == "process" else "tolerance"
        if f["status"] == "missing":
            lines.append("%s: NO READING received from the sensor." % f["label"])
        elif f["status"] == "out":
            lines.append("%s measured %s%s against a %s of %s: OUT OF %s by %s%s." % (
                f["label"], _fmt(f["value"], d), u, kind, _limits_text(f),
                "TOLERANCE" if f["role"] == "product" else "LIMIT",
                _fmt(abs(f["excess"]), d), u))
        else:
            lines.append("%s %s%s, within %s." % (f["label"], _fmt(f["value"], d), u, kind))
    for s in spc:
        lines.append("Statistical process control on %s: %s." % (s["label"].lower(), s["text"]))
    vision = msg.get("vision")
    if vision is not None:
        lines.append("Vision system: " + ("; ".join(
            "%s on %s, confidence %.2f" % (v["label"], v.get("location") or "part", v["confidence"])
            for v in vision) if vision else "no defect detected") + ".")
    for a in msg.get("alarms") or []:
        lines.append("Machine alarm: %s." % a)
    note = msg.get("operator_note") or {}
    text = note_text(msg)
    lines.append("Operator note%s: %s" % (" (%s)" % note["lang"] if note.get("lang") else "", text)
                 if text else "Operator note: none.")
    return " ".join(lines)


def decide(findings: List[dict], spc: List[dict], answers: Optional[dict], msg: dict,
           characteristics: Dict[str, dict], policy: Dict[str, float] = POLICY,
           engine_ok: bool = True) -> dict:
    """Combine rule results and Laya's reading of the note into a disposition.

    `answers` is None when there was no note to read (or Laya failed - then pass
    engine_ok=False). Returns {disposition, auto, defect, root_cause, alert, trace};
    `auto` False means the unit is held for a person, and `trace` lists every step.
    """
    trace: List[str] = []
    tau = policy["min_confidence"]
    failed = [f for f in findings if f["role"] == "product" and f["status"] == "out"]
    missing = [f for f in findings if f["role"] == "product" and f["status"] == "missing"]
    process_out = [f for f in findings if f["role"] == "process" and f["status"] == "out"]
    alarms = [(a, ALARMS.get(a)) for a in msg.get("alarms") or []]
    has_note = bool(note_text(msg))

    topic = t_conf = kind = None
    k_conf = 0.0
    if answers:
        topic, t_conf = answers["topic"]["choice"], answers["topic"]["confidence"]
        kind, k_conf = answers["defect_kind"]["choice"], answers["defect_kind"]["confidence"]
    sure = answers is not None and t_conf >= tau
    note_cause = TOPIC_CAUSE.get(topic) if sure else None
    alarm_cause = next((m["cause"] for _, m in alarms if m and m.get("cause")), None)

    if answers:
        trace.append("Laya read the operator note: '%s' at %.0f%% confidence%s." % (
            topic, t_conf * 100, "" if sure else " - below the %.0f%% gate" % (tau * 100)))
    elif has_note and not engine_ok:
        trace.append("Laya unavailable - the operator note could not be read.")
    else:
        trace.append("No operator note - Laya not called; the record is numbers and codes only.")

    # --- disposition -------------------------------------------------------------
    if missing:
        disposition, auto = "hold", False
        trace.append("Rules: no reading for %s. A unit without its measurement is never released."
                     % ", ".join(f["label"] for f in missing))
    elif failed:
        reaction = "scrap" if any(characteristics[f["key"]].get("reaction") == "scrap" for f in failed) else "rework"
        disposition, auto = reaction, True
        trace.append("Rules: %s out of tolerance -> reaction plan: %s."
                     % (", ".join(f["label"] for f in failed), reaction.upper()))
        if note_cause == "measurement":
            disposition, auto = "hold", False
            trace.append("The operator reports a measurement problem, so the failing reading itself is "
                         "suspect. HOLD for gauge verification instead of %s." % reaction.upper())
    else:
        trace.append("Rules: every product characteristic within tolerance.")
        if not has_note:
            disposition, auto = "pass", True
        elif answers is None:
            disposition, auto = "hold", False
            trace.append("A written note nobody has read cannot be ignored. HOLD.")
        elif not sure:
            disposition, auto = "hold", False
            trace.append("Laya is unsure what the note says. HOLD for a person to read it.")
        elif topic == "nothing":
            disposition, auto = "pass", True
            trace.append("The note reports nothing wrong -> PASS.")
        elif topic in ("part_defect", "handling_problem"):
            disposition, auto = "hold", False
            trace.append("The operator reports damage that no sensor measured. HOLD to confirm.")
        elif topic in ("material_problem", "measurement_problem"):
            disposition, auto = "hold", False
            trace.append("The note puts the %s in doubt, so passing readings do not clear the part. HOLD."
                         % ("material lot" if topic == "material_problem" else "measurement"))
        else:                                        # machine_problem on a good part
            disposition, auto = "pass", True
            trace.append("A machine problem on a part that measures good: PASS the part, alert the line.")

    # --- defect and cause for the record -------------------------------------------
    if failed:
        defect = characteristics[failed[0]["key"]].get("defect")
    elif answers and sure and topic in ("part_defect", "handling_problem") and kind != "none" and k_conf >= tau:
        defect = kind
    elif answers and sure and topic in ("part_defect", "handling_problem"):
        defect = "unclassified"
    else:
        defect = None
    cause = note_cause or alarm_cause

    # --- line alert (independent of this unit's disposition) ---------------------
    reasons = ["SPC %s on %s" % (s["rule"], s["label"].lower()) for s in spc]
    reasons += ["%s above process limit" % f["label"].lower() for f in process_out]
    reasons += ["alarm %s" % a for a, _ in alarms]
    if note_cause:
        reasons.append("operator reports a %s problem" % note_cause)
    alert = None
    if reasons:
        # Stop the line only when the numbers show drift AND a person on the line
        # independently reports an equipment problem. Either alone is a watch.
        drift = bool(spc) or bool(process_out)
        stop = drift and note_cause == "machine"
        alert = {"level": "stop" if stop else "watch", "reasons": reasons, "cause": cause}
        trace.append("Line alert %s: %s." % (alert["level"].upper(), "; ".join(reasons)))

    return {"disposition": disposition, "auto": auto, "defect": defect, "root_cause": cause,
            "alert": alert, "trace": trace}
