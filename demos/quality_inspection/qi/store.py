"""SQLite traceability store: raw telemetry, inspection records, reviews, alerts.

Every disposition keeps the exact report Laya read, Laya's full answers, the rule
findings and the policy trace, so any unit can be reconstructed after the fact.
"""
import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gateway TEXT NOT NULL,
    device TEXT NOT NULL,
    seq INTEGER NOT NULL,
    received REAL NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE (device, seq)
);
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telemetry_id INTEGER NOT NULL REFERENCES telemetry(id),
    serial TEXT NOT NULL,
    station TEXT NOT NULL,
    ts TEXT NOT NULL,
    created REAL NOT NULL,
    findings TEXT NOT NULL,
    spc TEXT NOT NULL,
    report TEXT NOT NULL,
    laya TEXT,
    decision TEXT NOT NULL,
    disposition TEXT NOT NULL,
    status TEXT NOT NULL,              -- auto | pending | reviewed
    final TEXT,                        -- disposition after review (= disposition when auto)
    reviewer TEXT,
    review_note TEXT,
    reviewed REAL,
    latency_ms REAL,
    model TEXT
);
CREATE INDEX IF NOT EXISTS ix_insp_station ON inspections(station, id);
CREATE INDEX IF NOT EXISTS ix_insp_status ON inspections(status);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER REFERENCES inspections(id),
    station TEXT NOT NULL,
    created REAL NOT NULL,
    level TEXT NOT NULL,
    reasons TEXT NOT NULL,
    cause TEXT,
    acked REAL,
    count INTEGER NOT NULL DEFAULT 1,  -- units that raised it while it stayed open
    last_inspection_id INTEGER,
    updated REAL
);
"""

_JSON_COLS = ("findings", "spc", "laya", "decision", "reasons", "payload")


def _row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    for k in _JSON_COLS:
        if k in d and d[k] is not None:
            d[k] = json.loads(d[k])
    return d


class Store:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        # Databases created before alerts were folded lack these columns.
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(alerts)")}
        for col, ddl in (("count", "INTEGER NOT NULL DEFAULT 1"), ("last_inspection_id", "INTEGER"),
                         ("updated", "REAL")):
            if col not in have:
                self.db.execute("ALTER TABLE alerts ADD COLUMN %s %s" % (col, ddl))
        self.db.execute("UPDATE alerts SET last_inspection_id=inspection_id WHERE last_inspection_id IS NULL")
        self.lock = threading.Lock()

    # --- telemetry -----------------------------------------------------------------
    def add_telemetry(self, gateway: str, device: str, seq: int, payload: dict) -> Optional[int]:
        """Insert a message; None when (device, seq) was already received (a retry)."""
        with self.lock:
            try:
                cur = self.db.execute(
                    "INSERT INTO telemetry (gateway, device, seq, received, payload) VALUES (?,?,?,?,?)",
                    (gateway, device, seq, time.time(), json.dumps(payload)))
                return cur.lastrowid
            except sqlite3.IntegrityError:
                return None

    def unprocessed(self) -> List[tuple]:
        """Telemetry stored but never inspected - what a crash left in the work queue."""
        with self.lock:
            rows = self.db.execute(
                "SELECT t.id, t.payload FROM telemetry t LEFT JOIN inspections i ON i.telemetry_id=t.id "
                "WHERE i.id IS NULL ORDER BY t.id").fetchall()
        return [(r["id"], json.loads(r["payload"])) for r in rows]

    # --- inspections ---------------------------------------------------------------
    def add_inspection(self, rec: Dict[str, Any]) -> int:
        cols = ("telemetry_id", "serial", "station", "ts", "created", "findings", "spc", "report",
                "laya", "decision", "disposition", "status", "final", "latency_ms", "model")
        vals = [json.dumps(rec[c]) if c in _JSON_COLS else rec[c] for c in cols]
        with self.lock:
            cur = self.db.execute("INSERT INTO inspections (%s) VALUES (%s)"
                                  % (",".join(cols), ",".join("?" * len(cols))), vals)
            return cur.lastrowid

    def inspection(self, iid: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            r = self.db.execute("SELECT * FROM inspections WHERE id=?", (iid,)).fetchone()
        return _row(r) if r else None

    def inspections(self, limit: int = 50, status: Optional[str] = None,
                    station: Optional[str] = None, before: Optional[int] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM inspections WHERE 1=1", []
        if status:
            q += " AND status=?"
            args.append(status)
        if station:
            q += " AND station=?"
            args.append(station)
        if before:
            q += " AND id<?"
            args.append(before)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            return [_row(r) for r in self.db.execute(q, args).fetchall()]

    def review(self, iid: int, final: str, reviewer: str, note: str) -> bool:
        with self.lock:
            cur = self.db.execute(
                "UPDATE inspections SET status='reviewed', final=?, reviewer=?, review_note=?, reviewed=? "
                "WHERE id=? AND status='pending'", (final, reviewer, note, time.time(), iid))
            return cur.rowcount == 1

    def series(self, station: str, key: str, limit: int) -> List[Dict[str, Any]]:
        """Recent values of one characteristic, oldest first, for control charts."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id, serial, ts, findings, disposition FROM inspections "
                "WHERE station=? ORDER BY id DESC LIMIT ?", (station, limit)).fetchall()
        out = []
        for r in reversed(rows):
            f = next((f for f in json.loads(r["findings"]) if f["key"] == key), None)
            if f and f["value"] is not None:
                out.append({"id": r["id"], "serial": r["serial"], "ts": r["ts"],
                            "value": f["value"], "status": f["status"], "disposition": r["disposition"]})
        return out

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            tot = self.db.execute(
                "SELECT COUNT(*) n, "
                "SUM(status='auto') auto, SUM(status='pending') pending, SUM(status='reviewed') reviewed, "
                "SUM(COALESCE(final, disposition)='pass') pass, SUM(COALESCE(final, disposition)='rework') rework, "
                "SUM(COALESCE(final, disposition)='scrap') scrap "
                "FROM inspections").fetchone()
            lat = [r[0] for r in self.db.execute(
                "SELECT latency_ms FROM inspections WHERE latency_ms IS NOT NULL ORDER BY id DESC LIMIT 200")]
            # First-pass yield per unit: a serial counts only if every station it has
            # reached passed it outright. A hold later released by a person is not first pass.
            units = self.db.execute(
                "SELECT serial, MIN(disposition='pass') ok FROM inspections GROUP BY serial").fetchall()
            alerts = self.db.execute("SELECT COUNT(*) FROM alerts WHERE acked IS NULL").fetchone()[0]
        lat.sort()
        d = {k: (tot[k] or 0) for k in tot.keys()}
        d["units"] = len(units)
        d["fpy"] = (sum(1 for u in units if u["ok"]) / len(units)) if units else None
        d["auto_rate"] = (d["auto"] / d["n"]) if d["n"] else None
        d["latency_p50"] = lat[len(lat) // 2] if lat else None
        d["latency_p95"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None
        d["open_alerts"] = alerts
        return d

    # --- alerts --------------------------------------------------------------------
    def raise_alert(self, inspection_id: int, station: str, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Open an alert, or fold this unit into the open one for the same station and level.

        A drifting machine raises the same alarm on every part until someone acts; one
        alert with a count is what a line lead can work with, twenty identical ones are not.
        """
        now = time.time()
        with self.lock:
            row = self.db.execute(
                "SELECT id FROM alerts WHERE station=? AND level=? AND acked IS NULL ORDER BY id DESC LIMIT 1",
                (station, alert["level"])).fetchone()
            if row:
                self.db.execute(
                    "UPDATE alerts SET count=count+1, last_inspection_id=?, updated=?, reasons=?, cause=COALESCE(?, cause) "
                    "WHERE id=?", (inspection_id, now, json.dumps(alert["reasons"]), alert.get("cause"), row["id"]))
                aid = row["id"]
            else:
                aid = self.db.execute(
                    "INSERT INTO alerts (inspection_id, station, created, level, reasons, cause, last_inspection_id, updated) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (inspection_id, station, now, alert["level"], json.dumps(alert["reasons"]), alert.get("cause"),
                     inspection_id, now)).lastrowid
        return self.alert(aid)

    def alert(self, aid: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            r = self.db.execute(
                "SELECT a.*, i.serial FROM alerts a LEFT JOIN inspections i ON i.id=a.last_inspection_id "
                "WHERE a.id=?", (aid,)).fetchone()
        return _row(r) if r else None

    def alerts(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self.lock:
            return [_row(r) for r in self.db.execute(
                "SELECT a.*, i.serial FROM alerts a LEFT JOIN inspections i ON i.id=a.last_inspection_id "
                "ORDER BY COALESCE(a.updated, a.created) DESC LIMIT ?", (limit,)).fetchall()]

    def ack_alert(self, aid: int) -> bool:
        with self.lock:
            return self.db.execute("UPDATE alerts SET acked=? WHERE id=? AND acked IS NULL",
                                   (time.time(), aid)).rowcount == 1

    def export_rows(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [_row(r) for r in self.db.execute("SELECT * FROM inspections ORDER BY id").fetchall()]
