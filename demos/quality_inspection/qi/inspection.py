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


# Chinese names for what the trace mentions (the English names come from config).
ZH_LABEL = {
    "bore_d": "镗孔直径", "spindle_vib": "主轴振动", "flatness": "密封面平面度", "position": "螺栓孔位置度",
    "torque_1": "1 号螺栓扭矩", "torque_2": "2 号螺栓扭矩", "torque_3": "3 号螺栓扭矩", "torque_4": "4 号螺栓扭矩",
    "leak_rate": "泄漏率", "defect_score": "视觉缺陷评分",
}
ZH_TOPIC = {"nothing": "无异常", "part_defect": "零件缺陷", "machine_problem": "设备问题",
            "material_problem": "物料问题", "measurement_problem": "测量问题", "handling_problem": "搬运损伤"}
ZH_CAUSE = {"machine": "设备", "material": "物料", "measurement": "测量", "handling": "搬运"}
ZH_REACTION = {"scrap": "报废", "rework": "返工"}


def decide(findings: List[dict], spc: List[dict], answers: Optional[dict], msg: dict,
           characteristics: Dict[str, dict], policy: Dict[str, float] = POLICY,
           engine_ok: bool = True) -> dict:
    """Combine rule results and Laya's reading of the note into a disposition.

    `answers` is None when there was no note to read (or Laya failed - then pass
    engine_ok=False). Returns {disposition, auto, defect, root_cause, alert, trace,
    trace_zh}; `auto` False means the unit is held for a person, and `trace` lists every
    step (trace_zh is the same steps in Chinese, for the bilingual dashboard).
    """
    trace: List[str] = []
    trace_zh: List[str] = []

    def say(en: str, zh: str):
        trace.append(en)
        trace_zh.append(zh)

    def zh_labels(fs):
        return "、".join(ZH_LABEL.get(f["key"], f["label"]) for f in fs)

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
        say("Laya read the operator note: '%s' at %.0f%% confidence%s." % (
                topic, t_conf * 100, "" if sure else " - below the %.0f%% gate" % (tau * 100)),
            "Laya 识别操作员留言：“%s”，置信度 %.0f%%%s。" % (
                ZH_TOPIC.get(topic, topic), t_conf * 100, "" if sure else "，低于 %.0f%% 阈值" % (tau * 100)))
    elif has_note and not engine_ok:
        say("Laya unavailable - the operator note could not be read.",
            "Laya 不可用，无法识别操作员留言。")
    else:
        say("No operator note - Laya not called; the record is numbers and codes only.",
            "无操作员留言，未调用 Laya；该记录只有数值与代码。")

    # --- disposition -------------------------------------------------------------
    if missing:
        disposition, auto = "hold", False
        say("Rules: no reading for %s. A unit without its measurement is never released."
            % ", ".join(f["label"] for f in missing),
            "规则：%s 无读数。缺少测量值的产品绝不放行。" % zh_labels(missing))
    elif failed:
        reaction = "scrap" if any(characteristics[f["key"]].get("reaction") == "scrap" for f in failed) else "rework"
        disposition, auto = reaction, True
        say("Rules: %s out of tolerance -> reaction plan: %s."
            % (", ".join(f["label"] for f in failed), reaction.upper()),
            "规则：%s 超差 → 反应计划：%s。" % (zh_labels(failed), ZH_REACTION[reaction]))
        if note_cause == "measurement":
            disposition, auto = "hold", False
            say("The operator reports a measurement problem, so the failing reading itself is "
                "suspect. HOLD for gauge verification instead of %s." % reaction.upper(),
                "操作员报告测量问题，超差读数本身可疑。冻结待量具核查，而不是%s。" % ZH_REACTION[reaction])
    else:
        say("Rules: every product characteristic within tolerance.", "规则：所有产品特性均在公差内。")
        if not has_note:
            disposition, auto = "pass", True
        elif answers is None:
            disposition, auto = "hold", False
            say("A written note nobody has read cannot be ignored. HOLD.", "无人阅读的书面留言不能忽略。冻结。")
        elif not sure:
            disposition, auto = "hold", False
            say("Laya is unsure what the note says. HOLD for a person to read it.",
                "Laya 无法确定留言内容。冻结，由人工阅读。")
        elif topic == "nothing":
            disposition, auto = "pass", True
            say("The note reports nothing wrong -> PASS.", "留言未报告异常 → 放行。")
        elif topic in ("part_defect", "handling_problem"):
            disposition, auto = "hold", False
            say("The operator reports damage that no sensor measured. HOLD to confirm.",
                "操作员报告了传感器未测到的损伤。冻结待确认。")
        elif topic in ("material_problem", "measurement_problem"):
            disposition, auto = "hold", False
            say("The note puts the %s in doubt, so passing readings do not clear the part. HOLD."
                % ("material lot" if topic == "material_problem" else "measurement"),
                "留言使%s存疑，合格读数不足以放行该零件。冻结。"
                % ("物料批次" if topic == "material_problem" else "测量结果"))
        else:                                        # machine_problem on a good part
            disposition, auto = "pass", True
            say("A machine problem on a part that measures good: PASS the part, alert the line.",
                "零件测量合格但设备有问题：放行零件，向产线告警。")

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
    reasons_zh = ["%s 触发 SPC %s" % (ZH_LABEL.get(s["key"], s["label"]), s["rule"]) for s in spc]
    reasons += ["%s above process limit" % f["label"].lower() for f in process_out]
    reasons_zh += ["%s 超出过程限值" % ZH_LABEL.get(f["key"], f["label"]) for f in process_out]
    reasons += ["alarm %s" % a for a, _ in alarms]
    reasons_zh += ["报警 %s" % a for a, _ in alarms]
    if note_cause:
        reasons.append("operator reports a %s problem" % note_cause)
        reasons_zh.append("操作员报告%s问题" % ZH_CAUSE.get(note_cause, note_cause))
    alert = None
    if reasons:
        # Stop the line only when the numbers show drift AND a person on the line
        # independently reports an equipment problem. Either alone is a watch.
        drift = bool(spc) or bool(process_out)
        stop = drift and note_cause == "machine"
        alert = {"level": "stop" if stop else "watch", "reasons": reasons, "reasons_zh": reasons_zh, "cause": cause}
        say("Line alert %s: %s." % (alert["level"].upper(), "; ".join(reasons)),
            "产线告警（%s）：%s。" % ("停线" if stop else "关注", "；".join(reasons_zh)))

    return {"disposition": disposition, "auto": auto, "defect": defect, "root_cause": cause,
            "alert": alert, "trace": trace, "trace_zh": trace_zh}
