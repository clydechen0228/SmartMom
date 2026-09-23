"""IoT edge gateway simulator.

Stands in for the box on the shop floor that reads the PLCs, gauges, nutrunners, leak
tester and camera, stamps each reading with a device sequence number, and forwards it
to the inspection service over HTTPS. Like a real gateway it store-and-forwards: when
the service is unreachable it keeps buffering and resends in order once it is back.
Sequence numbers make the resend idempotent - the service drops duplicates.

    python -m qi.gateway --url http://127.0.0.1:8090            # run against a server
    python -m qi.gateway --url ... --scenario tool_wear         # start with a fault

The server can also run one of these in-process (the default), which is what the
dashboard's scenario buttons drive.
"""
import argparse
import collections
import datetime as dt
import random
import threading
import time
from typing import Dict, List, Optional

from .config import GATEWAYS, LINE, STATION_ORDER, STATIONS

BENIGN_NOTES = [
    ("en", "Shift handover, line running normally."),
    ("de", "Schichtübergabe, keine Auffälligkeiten."),
    ("zh", "交接班，设备运行正常。"),
    ("es", "Cambio de turno, todo en orden."),
    ("hi", "शिफ्ट बदली, लाइन सामान्य चल रही है।"),
    ("en", "Coolant topped up, parts look good."),
]

# name -> (station, units it lasts, label shown in the dashboard, what it demonstrates)
SCENARIOS = {
    "tool_wear": ("CNC-02", 10, "Tool wear on CNC-02",
                  "Bore diameter drifts up as tool T14 wears. SPC flags the trend before parts go "
                  "out of tolerance; the operator's chatter note makes Laya read it as systemic -> line stop."),
    "torque_gun": ("ASM-03", 3, "Nutrunner slip (Chinese note)",
                   "Bolt 3 under-torqued. The operator writes in Chinese; Laya classifies assembly / machine."),
    "bad_lot": ("LKT-04", 4, "Porous casting lot (German note)",
                "Leak test fails. The note blames supplier lot L-8812 in German; Laya reads leak / material."),
    "gauge_drift": ("CMM-01", 3, "CMM probe drift",
                    "Flatness reads out of tolerance, but the note says the probe is drifting. Laya's "
                    "root cause 'measurement' holds the parts for gauge verification instead of reworking them."),
    "tray_scratch": ("AOI-01", 4, "Scratches from new trays (Spanish note)",
                     "Camera flags scratches; the Spanish note traces them to new plastic trays."),
    "operator_dent": ("AOI-01", 1, "Dent the camera missed",
                      "Every measurement passes, but the operator reports a dent. Laya catches the "
                      "defect in the words and holds the unit."),
    "sensor_dropout": ("LKT-04", 1, "Leak sensor timeout",
                       "No leak reading arrives. A unit without its measurement is held, never passed."),
}


def _iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


class LineSimulator:
    """Units move through the stations in order; one message per station visit."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        # Lot code from the start time keeps serials unique across restarts.
        self.lot = dt.datetime.now().strftime("%y%m%d%H%M")
        self.unit = 1
        self.stage = 0
        self.seq: Dict[str, int] = collections.defaultdict(int)
        self.active: Dict[str, int] = {}          # scenario -> step reached
        self.lock = threading.Lock()

    def inject(self, name: str):
        if name not in SCENARIOS:
            raise KeyError(name)
        with self.lock:
            self.active[name] = 0

    def clear(self):
        with self.lock:
            self.active.clear()

    def _normal(self, spec: dict) -> float:
        v = self.rng.gauss(spec["nominal"], spec["sigma"])
        if spec["lsl"] is None:
            v = abs(v)                               # one-sided characteristics are >= 0
        return round(v, spec.get("decimals", 3) + 1)

    def next_message(self) -> dict:
        with self.lock:
            sid = STATION_ORDER[self.stage]
            st = STATIONS[sid]
            serial = "%s-%s-%04d" % (LINE["serial_prefix"], self.lot, self.unit)
            m = {k: self._normal(s) for k, s in st["characteristics"].items()}
            msg = {
                "device": st["device"], "station": sid, "serial": serial, "ts": _iso(),
                "measurements": m, "alarms": [], "operator_note": None,
            }
            if sid == "AOI-01":
                msg["vision"] = []
            if self.rng.random() < 0.08:
                lang, text = self.rng.choice(BENIGN_NOTES)
                msg["operator_note"] = {"lang": lang, "text": text}

            for name in list(self.active):
                station, units, _, _ = SCENARIOS[name]
                if station != sid:
                    continue
                step = self.active[name]
                getattr(self, "_s_" + name)(msg, step)
                if step + 1 >= units:
                    del self.active[name]
                else:
                    self.active[name] = step + 1

            # Millisecond clock, forced monotonic: survives a gateway restart without
            # colliding with sequence numbers the service has already stored.
            seq = max(self.seq[st["device"]] + 1, int(time.time() * 1000))
            self.seq[st["device"]] = msg["seq"] = seq
            self.stage += 1
            if self.stage == len(STATION_ORDER):
                self.stage, self.unit = 0, self.unit + 1
            return msg

    # --- fault scenarios: each edits one message at its station ----------------------
    def _s_tool_wear(self, msg, step):
        m = msg["measurements"]
        m["bore_d"] = round(42.004 + 0.009 * step + self.rng.gauss(0, 0.002), 4)
        m["spindle_vib"] = round(3.4 + 0.5 * step + self.rng.gauss(0, 0.2), 2)
        if step >= 3:
            msg["operator_note"] = {"lang": "en", "text": (
                "Chatter noise from the spindle since the last parts, tool T14 is close to end of "
                "life. Bores keep getting bigger.")}
        if step >= 6:
            msg["alarms"] = ["SPINDLE_LOAD_HIGH"]

    def _s_torque_gun(self, msg, step):
        msg["measurements"]["torque_3"] = round(self.rng.gauss(8.3, 0.15), 3)
        msg["alarms"] = ["NUTRUNNER_SPINDLE_3_NOK"]
        msg["operator_note"] = {"lang": "zh", "text": "3号螺栓拧紧枪打滑，扭矩不足，需要重新拧紧。"}

    def _s_bad_lot(self, msg, step):
        msg["measurements"]["leak_rate"] = round(self.rng.gauss(3.6, 0.5), 3)
        msg["operator_note"] = {"lang": "de", "text": (
            "Porosität an der Gusswand sichtbar, alle Teile aus Gießerei-Charge L-8812. "
            "Lieferant informieren.")}

    def _s_gauge_drift(self, msg, step):
        msg["measurements"]["flatness"] = round(self.rng.gauss(0.026, 0.0015), 4)
        msg["operator_note"] = {"lang": "en", "text": (
            "The CMM probe drifted 0.02 mm on the reference block this morning and recalibration "
            "is requested. Readings are suspect, the parts look fine by hand.")}

    def _s_tray_scratch(self, msg, step):
        msg["measurements"]["defect_score"] = round(self.rng.uniform(0.78, 0.93), 3)
        msg["vision"] = [{"label": "scratch 4 mm", "location": "sealing face",
                          "confidence": msg["measurements"]["defect_score"]}]
        msg["operator_note"] = {"lang": "es", "text": (
            "Rayas en la cara de sellado, vienen de las bandejas de plástico nuevas. "
            "Ya van varias piezas esta hora.")}

    def _s_operator_dent(self, msg, step):
        msg["operator_note"] = {"lang": "en", "text": (
            "Small dent on the flange edge, the camera did not flag it. Please check before shipping.")}

    def _s_sensor_dropout(self, msg, step):
        msg["measurements"].pop("leak_rate", None)
        msg["alarms"] = ["PRESSURE_SENSOR_TIMEOUT"]


class EdgeGateway:
    """Generates readings on a fixed cadence and forwards them in batches, with backoff."""

    def __init__(self, url: str, gateway_id: str = "edge-gw-01", key: Optional[str] = None,
                 interval: float = 2.5, sim: Optional[LineSimulator] = None, batch: int = 50):
        import httpx
        self.url = url.rstrip("/") + "/api/v1/telemetry"
        self.gateway_id = gateway_id
        self.key = key or GATEWAYS.get(gateway_id, "")
        self.interval = interval
        self.sim = sim or LineSimulator()
        self.batch = batch
        self.buffer: collections.deque = collections.deque(maxlen=10000)
        self.client = httpx.Client(timeout=5.0)
        self.running = False
        self.sent = 0
        self.last_error: Optional[str] = None
        self._backoff = 0.0
        self._next_try = 0.0
        self._thread: Optional[threading.Thread] = None

    def status(self) -> dict:
        return {"gateway": self.gateway_id, "running": self.running, "interval": self.interval,
                "buffered": len(self.buffer), "sent": self.sent, "last_error": self.last_error,
                "scenarios": dict(self.sim.active)}

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="edge-gateway", daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        next_read = time.time()
        while self.running:
            now = time.time()
            if now >= next_read:
                self.buffer.append(self.sim.next_message())
                next_read = now + self.interval
            if self.buffer and now >= self._next_try:
                self.flush()
            time.sleep(0.05)

    def flush(self) -> bool:
        items: List[dict] = [self.buffer[i] for i in range(min(self.batch, len(self.buffer)))]
        try:
            r = self.client.post(self.url, json={"gateway": self.gateway_id, "messages": items},
                                 headers={"Authorization": "Bearer " + self.key})
            if r.status_code == 401:
                raise RuntimeError("401 unauthorized - check QI_GATEWAY_KEY")
            if r.status_code == 429:
                raise RuntimeError("429 service busy")
            r.raise_for_status()
        except Exception as e:                       # network down, server restarting, 5xx
            self.last_error = "%s: %s" % (type(e).__name__, e)
            self._backoff = min(30.0, max(1.0, self._backoff * 2))
            self._next_try = time.time() + self._backoff
            return False
        for _ in items:
            self.buffer.popleft()
        self.sent += len(items)
        self.last_error, self._backoff, self._next_try = None, 0.0, 0.0
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--gateway", default="edge-gw-01")
    ap.add_argument("--key", default=None, help="bearer key (default: $QI_GATEWAY_KEY or the demo key)")
    ap.add_argument("--interval", type=float, default=2.5, help="seconds between readings")
    ap.add_argument("--scenario", action="append", default=[], choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    gw = EdgeGateway(args.url, args.gateway, args.key, args.interval, LineSimulator(args.seed))
    for s in args.scenario:
        gw.sim.inject(s)
    gw.start()
    print("gateway %s -> %s every %.1f s (Ctrl-C to stop)" % (args.gateway, gw.url, args.interval))
    try:
        while True:
            time.sleep(5)
            st = gw.status()
            print("sent %(sent)d  buffered %(buffered)d  %(last_error)s" % st)
    except KeyboardInterrupt:
        gw.stop()


if __name__ == "__main__":
    main()
