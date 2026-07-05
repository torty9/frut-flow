#!/usr/bin/env bash
# Supervisor for früt Flow. Kept alive by launchd (KeepAlive). Every ~12s, if
# flow.py isn't running, relaunch it by opening frutflow.app. Launching THROUGH
# the .app bundle makes frutflow the TCC "responsible process", so the python
# child inherits frutflow's own Microphone + Input Monitoring + Accessibility
# grants (no Terminal needed). The app runs as a menu-bar item. Deterministic
# self-heal, independent of launchd StartInterval timing.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
FLOW_SCRIPT="$SCRIPT_DIR/flow.py"
APP="$HOME/Applications/frutflow.app"
FLOWDICTATE_DIR="$HOME/.flowdictate"
LOG="$FLOWDICTATE_DIR/watchdog.log"

prepare_flowdictate_dir() {
  umask 077
  mkdir -p "$FLOWDICTATE_DIR"
  chmod 700 "$FLOWDICTATE_DIR"
  : >> "$LOG"
  chmod 600 "$LOG"
}

process_cwd_matches_repo() {
  local pid="$1"
  local cwd
  cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)"
  [ "$cwd" = "$SCRIPT_DIR" ]
}

flow_process_matches() {
  local pid="$1"
  local args="$2"

  if [[ "$args" == *python*" $FLOW_SCRIPT"* ]]; then
    return 0
  fi

  if [[ "$args" == *python*" flow.py"* || "$args" == *python*" ./flow.py"* ]]; then
    process_cwd_matches_repo "$pid"
    return
  fi

  return 1
}

find_flow_pids() {
  local pid args
  ps -axww -o pid= -o args= | while read -r pid args; do
    [ -n "$pid" ] || continue
    case "$pid" in
      *[!0-9]*) continue ;;
    esac
    if flow_process_matches "$pid" "$args"; then
      printf '%s\n' "$pid"
    fi
  done
}

flow_is_running() {
  [ -n "$(find_flow_pids | head -n 1)" ]
}

prepare_flowdictate_dir
while true; do
  if ! flow_is_running; then
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] flow.py not running — relaunching frutflow.app" >> "$LOG"
    /usr/bin/open -g "$APP"
    sleep 8
  fi
  sleep 12
done
