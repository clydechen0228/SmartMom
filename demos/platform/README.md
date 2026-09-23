# Smart Factory Platform

Every demo in one process on one port, with one shared Laya model and a live link between
quality and planning.

```bash
python demos/platform/server.py           # http://127.0.0.1:8100
python demos/platform/server.py --mock    # no Laya weights; the Laya console is disabled
```

| Path | Module | Source |
|---|---|---|
| `/` | Portal: live overview of every module (EN / 中文) | `demos/platform/static/index.html` |
| `/quality/` | Quality inspection: IoT gateway, rules, SPC, Laya on operator notes, QA review | `demos/quality_inspection` |
| `/aps/` | Planning and scheduling: SAP orders, CP-SAT, repairs, disruption inbox (EN / 中文) | `demos/aps` |
| `/laya/` | Laya console: ask your own typed questions | `webui` |
| `/docs/` | The design documents | `docs/` |

Each module still runs on its own (`demos/quality_inspection/server.py` on :8090,
`demos/aps/server.py` on :8095, `webui/server.py`). The platform imports the same apps and
mounts them. Their pages use relative API paths, so they work both at `/` and under a
prefix, and they show a "← Platform" link only when mounted.

## What the platform adds

- **One Laya model.** A single `Router` is shared by the quality engine, the planning
  inbox and the console. On a CPU laptop the whole platform uses about 3.7 GB, against one
  copy of the checkpoints per module when run separately.
- **Quality → planning.** Three signals from quality inspection become suggested events
  in planning, each one completed and approved by the planner:

  | Quality sees | Planning gets | The planner adds |
  |---|---|---|
  | A *stop* alert on a machine (SPC drift plus an operator reporting a machine problem) | Machine down on the same machine | How long |
  | A unit held for QA | Quality hold on that unit's SAP order | How long |
  | Units scrapped (automatically or by QA) | A replacement order for the scrapped quantity, due with the original | Nothing; review it |

  When QA releases the last held unit of an order, its hold suggestion closes on its own.
  Repeats update one open suggestion per machine or order instead of piling up.
- **Units know their order.** In a plant the MES records which production order each
  unit was started against. The platform plays that role: a unit is assigned to the
  earliest-due unfinished SAP order for its material until that order's quantity is used
  up. The order shows in the quality feed, the QA queue and the unit detail.
- **One overview.** `/api/overview` returns the headline figures of every module, which
  the portal shows every 3 s.

## Try the links

1. **Line stop:** in `/quality/`, click **Tool wear on CNC-02**. Within about a minute the SPC
   trend and the operator's chatter note raise a stop. `/aps/` shows a suggested machine
   down; click **Review**, enter the hours, **Create proposal**.
2. **Hold:** click **Dent the camera missed**. The unit is held, and `/aps/` suggests a
   quality hold on its order. Release it in the QA queue and the suggestion closes.
3. **Scrap:** scrap a held unit in the QA queue, or run **Porous casting lot**. `/aps/`
   suggests a replacement order for that quantity.

The portal's *Quality → planning* card lists every hand-over as it happens.

The portal, quality and planning pages switch between English and 中文 (one setting for all).

## Tests

```bash
python demos/platform/test_platform.py    # 34 checks, mock engines, about 20 s
```
