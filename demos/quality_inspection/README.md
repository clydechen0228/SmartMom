# Production quality inspection with Laya

A working demo of Laya in a manufacturing quality loop. Station data arrives from an IoT
edge gateway. Rules judge the numbers, Laya reads what the operators write in any language,
and a policy decides which units can be released automatically and which go to a QA
engineer, with a traceable reason for every unit.

```bash
pip install -e . fastapi uvicorn httpx          # from the repo root
./run.sh start                                  # the whole platform; quality at http://127.0.0.1:8100/quality/
./run.sh start quality                          # quality alone → http://127.0.0.1:8090
./run.sh help quality                           # every option (--mock, --no-simulate, --interval …)
```

Without `run.sh`: `python demos/quality_inspection/server.py`.

The first start loads the English and multilingual checkpoints (about 10–15 s each on
CPU, cached after the first download). Records without an operator note are processed
while they load. Use `--mock` to run the UI without the weights. Mock mode replaces Laya
with keyword heuristics, and the dashboard says so in a banner.

## The line

Pump housing HSG-7731, five stations, one message per station per unit:

| station | sensor | characteristics |
|---|---|---|
| CNC-02 bore milling | in-process gauge, spindle accelerometer | bore Ø 42.000 ± 0.050 mm (SPC) · spindle vibration ≤ 6 mm/s (process) |
| CMM-01 coordinate measurement | touch-probe CMM | sealing-face flatness ≤ 0.020 (SPC) · true position ≤ 0.040 |
| ASM-03 cover bolting | 4-spindle nutrunner | 4 × torque 10 ± 1 N·m |
| LKT-04 leak test | pressure decay, 2.5 bar | leak rate ≤ 2.0 ml/min (SPC) |
| AOI-01 final vision | line-scan camera + classifier | defect score ≤ 0.50 |

Tolerances, reaction plans (rework or scrap), SPC sigma, alarm codes and the confidence
gate all live in `qi/config.py`, not in code.

## Who decides what

```
 sensors, PLCs ─► edge gateway ─► POST /api/v1/telemetry ─► raw store ─► worker
                 (seq, buffer,     (auth, validate,                          │
                  retry)            dedupe, 429)            ┌────────────────┴────────────────┐
                                                            │ rules: limits, reaction plan,   │
                                                            │ Western Electric SPC, alarm map │
                                                            ├─────────────────────────────────┤
                                                            │ Laya: the operator note, if any │
                                                            ├─────────────────────────────────┤
                                                            │ policy gate                     │
                                                            └──────┬───────────────┬──────────┘
                                                   auto disposition│               │hold → QA queue
                                                                   ▼               ▼
                                                     traceability DB, line alerts, dashboard (SSE)
```

**Rules** own everything that is a number or a code. That covers whether a measurement is in
tolerance, what the reaction plan says, SPC drift, and what an alarm code means. They are
deterministic and auditable. No note can overrule a failed measurement.

**Laya** reads what a person wrote at the station. It asks two typed questions in one
forward pass (*what does the operator report* and *what kind of defect*), with the note
routed to the English or multilingual checkpoint by script and language. Its answers
change outcomes in four ways:

| the note says | the rules say | outcome |
|---|---|---|
| a defect the sensors missed ("small dent on the flange edge") | pass | **hold**: a person confirms |
| the gauge is suspect ("probe drifted on the reference block") | fail | **hold** for gauge verification instead of scrapping possibly good parts |
| a machine problem ("chatter noise, tool T14 near end of life") | SPC drift | **line stop**, before the first part goes out of tolerance |
| nothing wrong ("Schichtübergabe, keine Auffälligkeiten") | pass | **pass**, automatically |

Below 50% confidence, Laya's reading goes to a person. Laya's probabilities are trained
against proper scoring rules, so the gate threshold corresponds to an actual error rate.
Records without a note never reach the model, which on this line is about 90% of them.

## Measured, not assumed

`evaluate.py` runs every fault scenario through the full pipeline with the real
checkpoints and checks the outcome each one is built to produce:

```bash
python demos/quality_inspection/evaluate.py
```

On a 2019 Intel laptop CPU (torch 2.2.2):

- every scenario produces its expected disposition
- the tool-wear line stop fires at bore 42.029 mm, three units before the first
  out-of-tolerance part (USL 42.050)
- Laya takes 456 ms per note at the median and 575 ms at p95
- 2 of 6 benign shift notes are held for review. The Spanish note is routed to the English
  checkpoint and read at 37%, and the Hindi one is read correctly but at 48%, just under the
  gate. Both mistakes make the system more cautious, never more permissive.

Three design choices come from these measurements, not from taste:

1. **Laya reads the note alone, not the whole record.** When the note came with the
   rule results ("Leak rate 0.61 ml/min, within tolerance …"), Laya answered *leak* for
   healthy parts and missed the defect in the operator's dent note. Reading the note alone
   is also five times faster.
2. **Choice questions, not yes/no.** On the multilingual checkpoint, yes/no (`noul`)
   questions read near 0 for Chinese, German and Spanish notes that plainly report a
   problem. The same content asked as a `choice` is classified correctly.
3. **`TORCHINDUCTOR_COMPILE_THREADS=1`.** With torch 2.2, loading a checkpoint forks a
   compile worker per CPU core, even though Laya never compiles. In a server those forks
   inherit the listening socket and keep the port bound after the parent dies. The engine
   sets the variable before torch loads.

## Try it

The **Inject a fault** panel drives the in-process gateway:

| fault | station | what to watch |
|---|---|---|
| Tool wear | CNC-02 | the bore trends up, SPC fires, the chatter note turns a watch into a **stop** |
| Nutrunner slip (Chinese note) | ASM-03 | rework, cause from the alarm table, note read by the multilingual checkpoint |
| Porous casting lot (German note) | LKT-04 | scrap on the measurement |
| CMM probe drift | CMM-01 | flatness "fails", Laya reads *measurement problem*, the part is **held**, not reworked |
| Scratches from new trays (Spanish note) | AOI-01 | rework on the vision score |
| Dent the camera missed | AOI-01 | every number passes, the note alone **holds** the unit |
| Leak sensor timeout | LKT-04 | no reading means hold, never pass |

Click any row for the full record: decision trace, measurements against spec, control
chart, the note Laya read with its routing reason, and the probability of every option
against the gate. Held units wait in the QA queue until a named reviewer decides.

## Connecting real equipment

Run the service with `--no-simulate` and have your gateway (Node-RED, Kepware, an
OPC UA or MQTT bridge, a PLC's HTTP client) POST batches:

```bash
curl -X POST http://127.0.0.1:8090/api/v1/telemetry \
  -H "Authorization: Bearer $QI_GATEWAY_KEY" -H "Content-Type: application/json" -d '{
  "gateway": "edge-gw-01",
  "messages": [{
    "device": "aoi-01-camera", "station": "AOI-01", "serial": "HSG7731-000123",
    "seq": 1695456000123, "ts": "2026-09-23T08:00:00Z",
    "measurements": {"defect_score": 0.04},
    "vision": [], "alarms": [],
    "operator_note": {"lang": "en", "text": "Small dent on the flange edge, camera did not flag it."}
  }]}'
```

- `seq` must increase per device. A resend of `(device, seq)` is accepted and dropped,
  so gateways can retry without creating duplicates.
- A device must be registered for the station it reports, and unknown characteristics
  are rejected.
- When the backlog is full the whole batch gets `429`. The gateway keeps it and retries.
- Readings stored but not yet inspected when the service stops are re-queued at start.

`python -m qi.gateway --url http://host:8090` (from this directory) runs the simulator
as a separate process, as a real gateway would.

## API

| route | purpose |
|---|---|
| `POST /api/v1/telemetry` | gateway ingestion (bearer key per gateway) |
| `GET /api/stream` | server-sent events: `inspection`, `alert`, `review`, `ack`, `sim` |
| `GET /api/inspections[?status=&station=&before=]` · `GET /api/inspections/{id}` | feed and full record |
| `POST /api/inspections/{id}/review` | `{disposition, reviewer, note}`, held units only |
| `GET /api/alerts` · `POST /api/alerts/{id}/ack` | line alerts. A repeat folds into the open alert for that station and level |
| `GET /api/status` · `GET /api/line` · `GET /api/meta` | KPIs, devices, engine, station series, configuration |
| `GET /api/export.csv` | traceability export |

## Tests

```bash
python demos/quality_inspection/test_quality_inspection.py   # offline, no weights, ~2 s
python demos/quality_inspection/evaluate.py                  # real checkpoints
```

## What a production deployment would change

This is a single-process demo. For a real line you would add TLS in front and move keys
to a secret store (per device, not per gateway). You would move SQLite to Postgres or a
historian, and run one inspection worker per station on a GPU box, where Laya is about 33 ms
per call instead of 0.5 s. The reaction plans and alarm table would come from the MES rather
than a Python file. And before any auto-release, you would run `evaluate.py` against a few
hundred of your own historical notes, labelled by QA, to set the gate where your own error
tolerance is.
