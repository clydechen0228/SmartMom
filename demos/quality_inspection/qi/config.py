"""The production line this demo inspects: stations, characteristics and their limits.

Everything a quality engineer would own lives here, not in code: tolerances, which
characteristics are product characteristics (a violation fails the part) and which are
process parameters (a violation raises an alert but does not fail the part), what a
failure's reaction is, and which characteristics run statistical process control.

`sigma` is the short-term process sigma the control limits are built from. In a real
plant it comes from a capability study; here it is chosen so a healthy process sits at
a Cpk of roughly 1.67 against the tolerance.
"""
import os

LINE = {
    "id": "L3",
    "name": "Line 3",
    "product": "HSG-7731 pump housing",
    "serial_prefix": "HSG7731",
}

# role:     "product" -> out of limits fails the part; "process" -> alert only
# reaction: what a failed product characteristic calls for on its own
# defect:   the defect_type a failure of this characteristic is expected to be
STATIONS = {
    "CNC-02": {
        "name": "Bore milling",
        "device": "cnc-02-plc",
        "sensor": "in-process bore gauge + spindle accelerometer",
        "characteristics": {
            "bore_d": {"label": "Bore diameter", "unit": "mm", "role": "product",
                       "nominal": 42.000, "lsl": 41.950, "usl": 42.050, "sigma": 0.010,
                       "reaction": "scrap", "defect": "dimensional", "spc": True, "decimals": 3},
            "spindle_vib": {"label": "Spindle vibration", "unit": "mm/s", "role": "process",
                            "nominal": 3.0, "lsl": None, "usl": 6.0, "sigma": 0.5,
                            "spc": False, "decimals": 1},
        },
        "key": "bore_d",
    },
    "CMM-01": {
        "name": "Coordinate measurement",
        "device": "cmm-01",
        "sensor": "touch-probe CMM",
        "characteristics": {
            "flatness": {"label": "Sealing face flatness", "unit": "mm", "role": "product",
                         "nominal": 0.008, "lsl": None, "usl": 0.020, "sigma": 0.0024,
                         "reaction": "rework", "defect": "dimensional", "spc": True, "decimals": 3},
            "position": {"label": "Bolt-hole true position", "unit": "mm", "role": "product",
                         "nominal": 0.012, "lsl": None, "usl": 0.040, "sigma": 0.005,
                         "reaction": "scrap", "defect": "dimensional", "spc": False, "decimals": 3},
        },
        "key": "flatness",
    },
    "ASM-03": {
        "name": "Cover bolting",
        "device": "asm-03-nutrunner",
        "sensor": "4-spindle nutrunner torque transducers",
        "characteristics": {
            "torque_%d" % i: {"label": "Bolt %d torque" % i, "unit": "N·m", "role": "product",
                              "nominal": 10.0, "lsl": 9.0, "usl": 11.0, "sigma": 0.2,
                              "reaction": "rework", "defect": "assembly",
                              "spc": i == 1, "decimals": 2}
            for i in range(1, 5)
        },
        "key": "torque_1",
    },
    "LKT-04": {
        "name": "Leak test",
        "device": "lkt-04-tester",
        "sensor": "pressure-decay leak tester, 2.5 bar",
        "characteristics": {
            "leak_rate": {"label": "Leak rate", "unit": "ml/min", "role": "product",
                          "nominal": 0.6, "lsl": None, "usl": 2.0, "sigma": 0.25,
                          "reaction": "scrap", "defect": "leak", "spc": True, "decimals": 2},
        },
        "key": "leak_rate",
    },
    "AOI-01": {
        "name": "Final vision",
        "device": "aoi-01-camera",
        "sensor": "12 MP line-scan camera, defect classifier",
        "characteristics": {
            "defect_score": {"label": "Vision defect score", "unit": "", "role": "product",
                             "nominal": 0.05, "lsl": None, "usl": 0.50, "sigma": 0.05,
                             "reaction": "rework", "defect": "surface", "spc": False, "decimals": 2},
        },
        "key": "defect_score",
    },
}

STATION_ORDER = ["CNC-02", "CMM-01", "ASM-03", "LKT-04", "AOI-01"]

# Edge gateways allowed to post telemetry, and their bearer keys. Set real keys through
# the environment; the default exists only so the demo runs out of the box.
GATEWAYS = {
    "edge-gw-01": os.environ.get("QI_GATEWAY_KEY", "demo-gateway-key"),
}

# PLC / tester alarm codes. A fixed vocabulary, so a table maps them - no model needed.
ALARMS = {
    "SPINDLE_LOAD_HIGH": {"cause": "machine", "text": "spindle load above limit"},
    "NUTRUNNER_SPINDLE_3_NOK": {"cause": "machine", "text": "nutrunner spindle 3 not OK"},
    "PRESSURE_SENSOR_TIMEOUT": {"cause": "measurement", "text": "leak tester pressure sensor timeout"},
}

# Gating. Laya's probabilities are calibrated (trained against proper scoring rules), so
# a threshold on its confidence is a threshold on how often it is wrong.
POLICY = {
    "min_confidence": 0.50,          # below this, Laya's reading of a note goes to a person
}

SPC_WINDOW = 25                      # points kept per characteristic for the control chart
DEVICE_TIMEOUT_S = 15                # a device with no message for this long reads offline
