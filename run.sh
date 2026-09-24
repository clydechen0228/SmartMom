#!/usr/bin/env bash
# Start and stop the demo servers from the project root.  Full help: ./run.sh help
#
#   ./run.sh start   [name] [server options…]   # in the background
#   ./run.sh fg      [name] [server options…]   # in this terminal (Ctrl+C stops)
#   ./run.sh stop    [name|all]
#   ./run.sh restart [name] [server options…]
#   ./run.sh status
#   ./run.sh logs    [name]                     # follow the log (Ctrl+C stops watching)
#   ./run.sh install-service | uninstall-service   # Laya service at login (macOS launchd)
#
# Names: platform (default, :8100, every module in one process), laya (the Laya service,
# :8077, used by the Laya MCP tools and the WebFetch guard hook), and the modules on their
# own: aps (:8095), quality (:8090), console (:8000). "start all" = platform + laya;
# "stop all" stops everything.
#
# Server options go straight to the server, e.g.
#   ./run.sh start platform --mock
#   ./run.sh start platform --aps-laya-multilingual ckpt/aps-ml-run1
#   ./run.sh start aps --laya-multilingual ckpt/aps-ml-run1
#
# Logs and process ids live in var/run/ (ignored by git). When the launchd job is
# installed, start/stop/logs for "laya" go through launchd (it restarts a killed process).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_DIR="$ROOT/var/run"
mkdir -p "$RUN_DIR"

if [[ -x "$ROOT/.conda311/bin/python" ]]; then
  PY="$ROOT/.conda311/bin/python"
elif [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
else
  PY="$(command -v python3 || command -v python)"
fi
export TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=false

NAMES="platform laya aps quality console"
LAUNCHD_LABEL="com.laya.service"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/$LAUNCHD_LABEL.plist"
LAUNCHD_LOG="$HOME/Library/Logs/laya-service.log"

script_of() {
  case "$1" in
    platform) echo "demos/platform/server.py" ;;
    aps)      echo "demos/aps/server.py" ;;
    quality)  echo "demos/quality_inspection/server.py" ;;
    console)  echo "webui/server.py" ;;
    laya)     echo "webui/server.py" ;;
    *) echo "unknown server '$1' (choose: $NAMES)" >&2; exit 2 ;;
  esac
}

port_of() {
  # the port the server will use: --port N among its options, else its default
  local name="$1"; shift
  local prev=""
  for a in "$@"; do
    if [[ "$prev" == "--port" ]]; then echo "$a"; return; fi
    case "$a" in --port=*) echo "${a#--port=}"; return ;; esac
    prev="$a"
  done
  case "$name" in platform) echo 8100 ;; laya) echo 8077 ;; aps) echo 8095 ;; quality) echo 8090 ;; console) echo 8000 ;; esac
}

default_args() {
  # the Laya service: fixed port, all three checkpoints loaded at start (what the MCP tools expect)
  [[ "$1" == "laya" ]] && echo "--port 8077 --preload" || true
}

launchd_installed() { [[ -f "$LAUNCHD_PLIST" ]] && command -v launchctl >/dev/null; }
launchd_loaded() { launchctl list 2>/dev/null | grep -q "$LAUNCHD_LABEL"; }

pid_of() {
  local f="$RUN_DIR/$1.pid"
  [[ -f "$f" ]] || return 1
  local pid; pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then echo "$pid"; else rm -f "$f"; return 1; fi
}

port_busy() {
  command -v lsof >/dev/null || return 1
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1
}

wait_up() {
  # wait until the port answers or the process dies (the first Laya load can take a while)
  local pid="$1" port="$2" name="$3"
  for _ in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited during start-up; last lines of the log:" >&2
      tail -n 25 "$RUN_DIR/$name.log" >&2
      rm -f "$RUN_DIR/$name.pid"
      return 1
    fi
    if curl -s -o /dev/null "http://127.0.0.1:$port/" 2>/dev/null; then return 0; fi
    sleep 1
  done
  echo "$name is still starting; check ./run.sh logs $name" >&2
}

cmd_start() {
  local name="${1:-platform}"; [[ $# -gt 0 ]] && shift
  if [[ "$name" == "all" ]]; then cmd_start platform "$@"; cmd_start laya; return; fi
  local script; script="$(script_of "$name")"
  if [[ "$name" == "laya" ]]; then
    [[ $# -eq 0 ]] && set -- $(default_args laya)
    export LAYA_DEVICE="${LAYA_DEVICE:-cpu}"
    if launchd_installed; then
      launchd_loaded || launchctl load -w "$LAUNCHD_PLIST"
      echo "laya service is managed by launchd ($LAUNCHD_LABEL); log: $LAUNCHD_LOG"
      for _ in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:8077/api/meta && { echo "laya is up: http://127.0.0.1:8077/"; return 0; }; sleep 1; done
      echo "laya is still loading its checkpoints; ./run.sh logs laya" >&2; return 0
    fi
  fi
  local port; port="$(port_of "$name" "$@")"
  if pid="$(pid_of "$name")"; then
    echo "$name is already running (pid $pid) on http://127.0.0.1:$port/"; return 0
  fi
  if busy="$(port_busy "$port")"; then
    echo "port $port is in use by pid $busy ($(ps -o comm= -p "$busy" 2>/dev/null || echo '?'))." >&2
    echo "Stop it, or start on another port: ./run.sh start $name --port <n>" >&2
    exit 1
  fi
  nohup "$PY" "$ROOT/$script" "$@" > "$RUN_DIR/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_DIR/$name.pid"
  echo "starting $name (pid $pid), log: var/run/$name.log"
  if wait_up "$pid" "$port" "$name"; then
    echo "$name is up: http://127.0.0.1:$port/"
  fi
}

cmd_fg() {
  local name="${1:-platform}"; [[ $# -gt 0 ]] && shift
  local script; script="$(script_of "$name")"
  if [[ "$name" == "laya" ]]; then [[ $# -eq 0 ]] && set -- $(default_args laya); export LAYA_DEVICE="${LAYA_DEVICE:-cpu}"; fi
  local port; port="$(port_of "$name" "$@")"
  if busy="$(port_busy "$port")"; then
    echo "port $port is in use by pid $busy; stop it first (./run.sh stop $name)" >&2; exit 1
  fi
  echo "$name on http://127.0.0.1:$port/ (Ctrl+C stops)"
  exec "$PY" "$ROOT/$script" "$@"
}

stop_one() {
  local name="$1"
  if [[ "$name" == "laya" ]] && launchd_installed && launchd_loaded; then
    launchctl unload -w "$LAUNCHD_PLIST"          # a plain kill would be restarted by launchd
    echo "stopped laya (launchd job unloaded; ./run.sh start laya loads it again)"
    return
  fi
  if pid="$(pid_of "$name")"; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    rm -f "$RUN_DIR/$name.pid"
    echo "stopped $name (pid $pid)"
  else
    # also stop a copy started some other way (e.g. by hand): whatever listens on its port,
    # if it is this server's script (never a browser or another program on that port)
    local script port busy; script="$(script_of "$name")"; port="$(port_of "$name")"
    busy="$(port_busy "$port" || true)"
    if [[ -n "$busy" ]] && ps -o command= -p "$busy" 2>/dev/null | grep -q "$script"; then
      kill "$busy" 2>/dev/null || true
      echo "stopped $name started outside run.sh (pid $busy)"
    else
      echo "$name is not running"
    fi
  fi
}

cmd_stop() {
  local name="${1:-platform}"
  if [[ "$name" == "all" ]]; then
    for n in $NAMES; do stop_one "$n"; done
  else
    script_of "$name" >/dev/null
    stop_one "$name"
  fi
}

cmd_status() {
  for n in $NAMES; do
    local port; port="$(port_of "$n")"
    if [[ "$n" == "laya" ]] && launchd_installed; then
      if launchd_loaded && port_busy "$port" >/dev/null; then echo "laya      running  (launchd, starts at login)   log $LAUNCHD_LOG"
      elif launchd_loaded; then echo "laya      loading  (launchd, starts at login)   log $LAUNCHD_LOG"
      else echo "laya      stopped  (launchd job installed but unloaded)"; fi
      continue
    fi
    if pid="$(pid_of "$n")"; then
      echo "$(printf '%-9s' "$n") running  pid $(printf '%-6s' "$pid") http://127.0.0.1:$port/   log var/run/$n.log"
    elif busy="$(port_busy "$port")"; then
      echo "$(printf '%-9s' "$n") port $port in use by pid $busy (not started by run.sh)"
    else
      echo "$(printf '%-9s' "$n") stopped"
    fi
  done
}

cmd_logs() {
  local name="${1:-platform}"
  script_of "$name" >/dev/null
  if [[ "$name" == "laya" ]] && launchd_installed; then tail -n 40 -f "$LAUNCHD_LOG"; return; fi
  [[ -f "$RUN_DIR/$name.log" ]] || { echo "no log yet for $name" >&2; exit 1; }
  tail -n 40 -f "$RUN_DIR/$name.log"
}

cmd_install_service() {
  command -v launchctl >/dev/null || { echo "launchd is macOS only; use ./run.sh start laya instead" >&2; exit 1; }
  if pid="$(pid_of laya)"; then stop_one laya; fi       # hand over from run.sh to launchd
  mkdir -p "$(dirname "$LAUNCHD_PLIST")" "$(dirname "$LAUNCHD_LOG")"
  sed -e "s|__LAYA_ROOT__|$ROOT|g" -e "s|__HOME__|$HOME|g" mcp_server/com.laya.service.plist > "$LAUNCHD_PLIST"
  launchd_loaded && launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
  launchctl load -w "$LAUNCHD_PLIST"
  echo "installed $LAUNCHD_PLIST: the Laya service starts at login and restarts if it dies"
  echo "log: $LAUNCHD_LOG   (it takes a minute to load the three checkpoints)"
}

cmd_uninstall_service() {
  if [[ -f "$LAUNCHD_PLIST" ]]; then
    launchctl unload -w "$LAUNCHD_PLIST" 2>/dev/null || true
    rm -f "$LAUNCHD_PLIST"
    echo "removed $LAUNCHD_PLIST; ./run.sh start laya still runs it by hand"
  else
    echo "the launchd job is not installed"
  fi
}

# ---------------------------------------------------------------------------------------
# help: ./run.sh help [topic]
# ---------------------------------------------------------------------------------------
B=""; U=""; R=""
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then B=$'\033[1m'; U=$'\033[4m'; R=$'\033[0m'; fi

help_main() {
  cat <<EOF
${B}run.sh${R} - start, stop and watch the demo servers and the Laya service

${B}USAGE${R}
  ./run.sh <command> [name] [server options...]
  ./run.sh help [command | name | topic]

  Run it from anywhere; it always works in the project root.

${B}COMMANDS${R}
  start   [name] [opts]   Start in the background; waits until it answers
  fg      [name] [opts]   Run in this terminal; Ctrl+C stops it
  stop    [name | all]    Stop one server, or every one
  restart [name] [opts]   Stop, then start with the given options
  status                  What is running, with URLs and log files
  logs    [name]          Follow a server's log; Ctrl+C stops watching
  install-service         Run the Laya service at login (macOS launchd)
  uninstall-service       Remove that login job
  help    [topic]         This page, or the page for one topic

  The name defaults to ${B}platform${R}. "start all" = platform + laya.
  Each command has its own page: ./run.sh help <command>, or ./run.sh <command> --help.

${B}SERVERS${R}
  name       port   what                                              Laya memory
  platform   8100   Every module in one process: portal, quality,       ~3.7 GB
                    planning (APS), Laya console, training runs, docs
  laya       8077   The Laya service used by Claude Code's Laya tools   ~4 GB
                    (MCP) and the WebFetch guard hook
  aps        8095   Planning (APS) alone                                ~2.5 GB
  quality    8090   Quality inspection alone                            ~2.5 GB
  console    8000   Laya console alone (loads models on first use)      up to ~4 GB

  Use ${B}platform${R} for the demo. aps, quality and console are for working on one
  module: a second copy does not share data with the platform. Details: help <name>.

EOF
  help_commands
  echo
  cat <<EOF
${B}MORE HELP${R}
  ./run.sh help commands       the list above
  ./run.sh help platform | laya | aps | quality | console    options of each server
  ./run.sh help examples       common tasks, copy and paste
  ./run.sh help env            environment variables
  ./run.sh help files          logs, process ids, labels, checkpoints
  ./run.sh help troubleshoot   port in use, slow start, memory, launchd
  ./run.sh <command> --help    same as help <command>
EOF
}

help_commands() {
  cat <<EOF
${B}ALL COMMANDS${R}

  ${U}Platform${R} (every module in one process, :8100)
  ./run.sh start                   # start the platform (same as: start platform)
  ./run.sh start platform --mock   # no Laya weights; starts in seconds
  ./run.sh stop                    # stop the platform
  ./run.sh restart                 # stop and start again
  ./run.sh logs                    # follow its log (Ctrl+C stops watching)
  ./run.sh fg                      # run it in this terminal (Ctrl+C stops it)

  ${U}Laya service${R} (:8077, used by Claude Code's Laya tools and the WebFetch guard)
  ./run.sh start laya              # about 50 s: loads all three checkpoints
  ./run.sh stop laya
  ./run.sh restart laya
  ./run.sh logs laya
  ./run.sh fg laya

  ${U}Everything${R}
  ./run.sh start all               # platform (:8100) + Laya service (:8077)
  ./run.sh stop all                # everything that's running
  ./run.sh restart all             # stop everything, start platform + Laya service
  ./run.sh status                  # what is running, with URLs and log files

  ${U}One module on its own${R} (for development; a separate copy from the platform)
  ./run.sh start aps               # planning, :8095
  ./run.sh stop aps
  ./run.sh restart aps
  ./run.sh logs aps
  ./run.sh start quality           # quality inspection, :8090
  ./run.sh stop quality
  ./run.sh restart quality
  ./run.sh logs quality
  ./run.sh start console           # Laya console, :8000
  ./run.sh stop console
  ./run.sh restart console
  ./run.sh logs console

  ${U}Laya service at login${R} (macOS)
  ./run.sh install-service         # start at every login, restart if it dies
  ./run.sh uninstall-service       # remove that

  ${U}Help${R}
  ./run.sh help                    # overview
  ./run.sh help commands           # this list
  ./run.sh help <topic>            # start fg stop restart status logs service
                                   # platform laya aps quality console
                                   # env files examples troubleshoot
  ./run.sh <command> --help        # the page for one command
EOF
}

help_topic() {
  case "$1" in
  commands|all|list) help_commands ;;
  start) cat <<EOF
${B}./run.sh start [name] [server options...]${R}

Starts a server in the background and waits until it answers on its port (up to two
minutes; loading Laya takes 15-60 s). Output goes to var/run/<name>.log and the process
id to var/run/<name>.pid.

  ./run.sh start                          # platform
  ./run.sh start all                      # platform, then the Laya service
  ./run.sh start platform --mock          # no Laya weights; starts in seconds
  ./run.sh start aps --port 8200          # any server option passes through

Refuses to start when:
  - it is already running (prints its URL instead)
  - another process holds the port (names it; use --port or stop that process)
If the server dies during start-up, the last 25 log lines are printed.

For ${B}laya${R}: with no options it runs with --port 8077 --preload. If the launchd job is
installed (help service), start loads the job instead of starting a second copy.
EOF
  ;;
  fg) cat <<EOF
${B}./run.sh fg [name] [server options...]${R}

Runs a server in this terminal, with its output on screen. Ctrl+C stops it. Nothing is
written to var/run/, so status shows it as "not started by run.sh".

  ./run.sh fg                             # platform in the foreground
  ./run.sh fg aps --mock
  ./run.sh fg aps --help                  # the server's own list of options
EOF
  ;;
  stop) cat <<EOF
${B}./run.sh stop [name | all]${R}

Stops a server started by run.sh (asks politely, then forces it after 5 s). If it was
started some other way, whatever listens on its port is stopped too, but only when that
process is this server's own script: a browser or another program on the port is left
alone.

  ./run.sh stop                           # platform
  ./run.sh stop laya
  ./run.sh stop all                       # every server, including the Laya service

For ${B}laya${R} under launchd, stop unloads the job (a plain kill would be restarted).
It stays installed: ./run.sh start laya loads it again, and it starts at the next login.
EOF
  ;;
  restart) cat <<EOF
${B}./run.sh restart [name | all] [server options...]${R}

Stop, then start. The new start uses only the options given now; earlier options are
not remembered.

  ./run.sh restart                                             # platform, defaults
  ./run.sh restart platform --aps-laya-multilingual ckpt/aps-ml-run1
  ./run.sh restart all
EOF
  ;;
  status) cat <<EOF
${B}./run.sh status${R}

One line per server: running (pid, URL, log), stopped, or "port in use (not started by
run.sh)" when something else holds its port. For laya under launchd: running, loading
(still reading its checkpoints) or stopped.
EOF
  ;;
  logs) cat <<EOF
${B}./run.sh logs [name]${R}

Shows the last 40 lines of a server's log and follows it. Ctrl+C stops watching; the
server keeps running. Logs are rewritten at each start.

  var/run/<name>.log                      servers started by run.sh
  ~/Library/Logs/laya-service.log         the Laya service under launchd
EOF
  ;;
  service|install-service|uninstall-service) cat <<EOF
${B}./run.sh install-service${R}  /  ${B}./run.sh uninstall-service${R}      (macOS)

install-service fills this checkout's paths into mcp_server/com.laya.service.plist, copies
it to ~/Library/LaunchAgents/com.laya.service.plist and loads it. From then on the Laya
service starts at login, preloads all three checkpoints and is restarted if it dies. A
copy started by run.sh is stopped first, so the two do not fight over port 8077.

While it is installed, start / stop / status / logs for laya go through launchd.

uninstall-service unloads and removes the job. Run install-service again after moving
the project folder: the job holds absolute paths.
EOF
  ;;
  platform) cat <<EOF
${B}platform${R}  - every module in one process            http://127.0.0.1:8100/

  /            portal: live overview of every module (EN / 中文)
  /quality/    quality inspection: IoT gateway, rules, SPC, Laya on operator notes
  /aps/        planning: SAP orders, CP-SAT schedule, what-if, Laya inbox and commands
  /laya/       Laya console: ask your own typed questions
  /training/   Laya training runs: data, settings, improvement curves
  /docs/       design documents and the APS workbench guide

One Laya model is shared by every module (~3.7 GB). The first plan takes about a minute.

Options:
  --mock                          keyword stand-ins instead of Laya (console disabled)
  --port N                        default 8100
  --host ADDR                     default 127.0.0.1 (use 0.0.0.0 to reach it from others)
  --device cpu|cuda|mps           default: automatic
  --interval S                    quality line: seconds between simulated readings (2.5)
  --db PATH                       quality database
                                  (default demos/quality_inspection/data/platform.db)
  --aps-laya-english PATH         planning only: a trained English checkpoint
  --aps-laya-multilingual PATH    planning only: a trained checkpoint for other languages
EOF
  ;;
  laya) cat <<EOF
${B}laya${R}  - the Laya service                          http://127.0.0.1:8077/

webui/server.py with --port 8077 --preload: all three checkpoints (english, multilingual,
typed-decisions) stay loaded, ~4 GB, 15-60 s to start. Claude Code's Laya tools (MCP
server mcp_server/laya_mcp.py) and the WebFetch guard hook call it; they point elsewhere
with LAYA_SERVICE_URL. The MCP server itself is started by Claude Code, not here.

If the service is down, the Laya tools say they cannot reach it and the guard hook lets
everything through unchecked (it fails open by design).

Options (replace the defaults, so repeat what you need):
  --port N  --preload  --device cpu|cuda|mps  --host ADDR
Environment: LAYA_DEVICE (default cpu here).  Start at login: help service.
API: GET /api/meta, POST /api/route, POST /api/predict (see GUIDE.md).
EOF
  ;;
  aps) cat <<EOF
${B}aps${R}  - planning (APS) on its own                  http://127.0.0.1:8095/

Options:
  --mock                          keyword stand-in instead of Laya
  --port N  --host ADDR  --device cpu|cuda|mps
  --laya-english PATH             a trained checkpoint for English text
  --laya-multilingual PATH        a trained checkpoint for 中文, Deutsch ...
  --labels PATH                   where planner decisions are saved as training labels
                                  (default demos/aps/data/labels/labels.jsonl)
Environment: APS_FULL_S (full-plan time limit, 60), APS_REPAIR_S (repair limit, 30),
LAYA_ENGLISH / LAYA_MULTILINGUAL (same as the checkpoint options), APS_LABELS.
EOF
  ;;
  quality) cat <<EOF
${B}quality${R}  - quality inspection on its own          http://127.0.0.1:8090/

Options:
  --mock                          keyword stand-in instead of Laya (UI work only)
  --no-simulate                   no built-in gateway; wait for a real one to post data
  --interval S                    seconds between simulated readings (2.5)
  --db PATH                       database (demos/quality_inspection/data/qi.db)
  --port N  --host ADDR  --device cpu|cuda|mps
Environment: QI_GATEWAY_KEY (the key a gateway sends; demo default demo-gateway-key).
EOF
  ;;
  console) cat <<EOF
${B}console${R}  - the Laya console on its own            http://127.0.0.1:8000/

Paste a text, pick or write typed questions, see Laya's answers and routing. Models load
on first use (15-40 s the first time).

Options:
  --preload                       load all three checkpoints at start
  --port N  --host ADDR  --device cpu|cuda|mps
The platform includes the same console at /laya/.
EOF
  ;;
  env) cat <<EOF
${B}Environment variables${R}

  PYTHON              interpreter when .conda311/ is missing (default python3)
  NO_COLOR            plain help text, no bold
  LAYA_DEVICE         device for the Laya service (run.sh sets cpu)
  LAYA_SERVICE_URL    where the Laya MCP tools and guard hook find the service (8077)
  LAYA_ENGLISH        planning: trained English checkpoint (like --laya-english)
  LAYA_MULTILINGUAL   planning: trained multilingual checkpoint
  LAYA_RUNS           where /training/ looks for runs (default ckpt/)
  APS_LABELS          planning: label log file
  APS_FULL_S          planning: full-plan time limit in seconds (60)
  APS_REPAIR_S        planning: repair time limit in seconds (30)
  QI_GATEWAY_KEY      quality: key a gateway sends with its data
  HF_TOKEN            Hugging Face token, if downloads need one
  HF_HUB_OFFLINE=1    use only the checkpoints already downloaded

Set one for a single start:  APS_REPAIR_S=10 ./run.sh start platform
EOF
  ;;
  files) cat <<EOF
${B}Files${R}

  var/run/<name>.log                  log of a server started by run.sh
  var/run/<name>.pid                  its process id (removed when it stops)
  ~/Library/Logs/laya-service.log     Laya service log under launchd
  ~/Library/LaunchAgents/com.laya.service.plist   the launchd job (install-service)
  demos/aps/data/labels/labels.jsonl  planner decisions saved as training labels
  demos/quality_inspection/data/      quality databases
  ckpt/                               trained checkpoints and their run.json files

var/, ckpt/, work/ and the label log are not in git. The label log holds plant text:
keep it inside the plant.
EOF
  ;;
  examples) cat <<EOF
${B}Examples${R}

  Demo on a laptop                    ./run.sh start
  Everything, incl. Claude's tools    ./run.sh start all
  UI work, no model                   ./run.sh start platform --mock
  Watch it start                      ./run.sh logs
  Try a trained checkpoint            ./run.sh restart platform \\
                                        --aps-laya-multilingual ckpt/aps-ml-run1
  Faster planning repairs             APS_REPAIR_S=10 ./run.sh restart
  Planning alone, in this terminal    ./run.sh fg aps
  A second platform next to the first ./run.sh fg platform --port 8101 --mock
  Laya service at every login         ./run.sh install-service
  Free the memory                     ./run.sh stop all
EOF
  ;;
  troubleshoot|troubleshooting) cat <<EOF
${B}Troubleshooting${R}

"port N is in use by pid P"
    Something already listens there. ./run.sh status shows whether it is one of these
    servers; ./run.sh stop <name> stops it. Otherwise: lsof -nP -iTCP:N -sTCP:LISTEN
    and stop that program, or start on another port with --port.

"<name> is still starting"
    The first start downloads or loads the Laya checkpoints. Watch: ./run.sh logs <name>.

"<name> exited during start-up"
    The log tail printed above says why. Common: a missing package (pip install -e .
    fastapi uvicorn "ortools==9.10.4067" "numpy<2"), or not enough memory.

The Mac slows down or the server is killed
    Each server loads its own Laya model. Run one platform, not the modules next to it;
    ./run.sh stop all frees everything. --mock needs no model.

The Laya tools in Claude Code say they cannot reach the service
    ./run.sh start laya (or install-service for every login).

laya restarts right after "stop"
    It is under launchd: use ./run.sh stop laya (it unloads the job), not kill.
EOF
  ;;
  *) echo "no help for '$1'. Topics: commands start fg stop restart status logs service platform laya" \
          "aps quality console env files examples troubleshoot" >&2; return 2 ;;
  esac
}

# "<command> --help" and "<command> -h" show that command's page
if [[ $# -ge 2 && ( "$2" == "--help" || "$2" == "-h" ) ]]; then help_topic "$1"; exit $?; fi

case "${1:-}" in
  start)   shift; cmd_start "$@" ;;
  fg)      shift; cmd_fg "$@" ;;
  stop)    shift; cmd_stop "$@" ;;
  restart) shift; n="${1:-platform}"; cmd_stop "$n"; sleep 1; cmd_start "$@" ;;
  status)  cmd_status ;;
  logs)    shift; cmd_logs "$@" ;;
  install-service)   cmd_install_service ;;
  uninstall-service) cmd_uninstall_service ;;
  ""|-h|--help) help_main ;;
  help)    shift; if [[ $# -gt 0 ]]; then help_topic "$1"; else help_main; fi ;;
  *) echo "unknown command '$1'; ./run.sh help lists them" >&2; exit 2 ;;
esac
