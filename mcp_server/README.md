# Laya as an MCP server

Exposes Laya to Claude Code (and any MCP client) as two tools. Two processes:

```
Claude Code ──stdio──▶ laya_mcp.py ──HTTP──▶ webui/server.py ──▶ 2.2 GB of weights
   (per project)        (instant)            (one, long-lived)      (loaded once)
```

The split matters. A stdio MCP server is spawned per client, and building the
checkpoints costs 12–16 s and ~2 GB of RAM each. Keeping them in one shared HTTP
service means the MCP server starts instantly, every project reuses the same resident
models, and a tool call is a local round trip.

## Setup

```bash
pip install mcp httpx                       # on top of laya

# 1. the model service — leave it running
python webui/server.py --port 8077 --preload

# 2. register, once, for every project
claude mcp add --scope user laya -- /abs/path/to/python /abs/path/to/mcp_server/laya_mcp.py
claude mcp list                             # laya - ✔ Connected
```

`--scope user` is what makes it available from any directory. Use `--scope project` to
commit it to one repo's `.mcp.json` instead. Remove with `claude mcp remove laya`.

### Keeping the service up

The service must be running for the tools to work. To start it at login and restart it if
it dies, install the bundled launchd job:

```bash
sed -e "s|__LAYA_ROOT__|$PWD|g" -e "s|__HOME__|$HOME|g" \
    mcp_server/com.laya.service.plist > ~/Library/LaunchAgents/com.laya.service.plist
launchctl load ~/Library/LaunchAgents/com.laya.service.plist
tail -f ~/Library/Logs/laya-service.log          # watch it preload
```

Stop it with `launchctl unload ~/Library/LaunchAgents/com.laya.service.plist`. Run the
install from the repo root; it fills this checkout's paths into the plist, so run it again
if you move the repo.

`LAYA_SERVICE_URL` (default `http://127.0.0.1:8077`) points it at another host or port.
`LAYA_TIMEOUT` (default 120 s) covers a cold load if you skipped `--preload`.

If the service is down, both tools return a message saying how to start it rather than
failing silently.

## Tools

### `laya_guard(text, source?)`

Screens untrusted text for prompt injection, jailbreak attempts and sensitive data.
Returns probabilities plus a `flagged` boolean, and deliberately **does not echo the
text back**, so a payload inside it cannot reach the agent through the tool result.

Measured on a poisoned README (an HTML comment telling the agent to read
`~/.aws/credentials`) versus a clean one:

| | prompt_injection | jailbreak | flagged |
|---|---|---|---|
| poisoned | 0.62 | 0.88 | **true** |
| clean | 0.08 | 0.14 | false |

### `laya_classify(text, questions, model?)`

Your own typed questions over natural-language text — `choice`, `score`, `noul` — in one
forward pass, with calibrated probabilities. Returns the answers flattened to value plus
confidence; the full distributions stay in the HTTP API, since they rarely earn their
place in an agent's context.

## What this is not for

Laya reads natural language, not code. The encoders (ModernBERT, mmBERT) have no code
training, and the context is 512–1024 tokens, which one real source file exceeds. It
cannot review a diff, judge correctness, or estimate how hard a coding task is — a
measured `difficulty` spread of 1.46 (trivia) to 1.95 (architectural refactor) is too
flat to route on.

The job it is actually good at is the one above: screening third-party prose before an
agent acts on it, fast enough to run on every fetch.

## A tool is an offer, not a guarantee

Claude decides whether to call `laya_guard`. If you want *enforcement* — every fetched
page screened, no exceptions — a `PostToolUse` hook on WebFetch that shells out to the
HTTP API is the mechanism that actually guarantees it. This server makes the capability
available; it does not make it mandatory.
