#!/usr/bin/env bash
# Restart früt Flow (use after changing ~/.flowdictate/config.json).
# Stops the running instance and relaunches it as frutflow.app (menu-bar app).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
FLOW_SCRIPT="$SCRIPT_DIR/flow.py"
FLOWDICTATE_DIR="$HOME/.flowdictate"
APP="$HOME/Applications/frutflow.app"
LOG="$FLOWDICTATE_DIR/flow.log"

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

stop_flow_processes() {
  local pid
  find_flow_pids | while read -r pid; do
    kill "$pid" 2>/dev/null || true
  done
}

prepare_flowdictate_dir
stop_flow_processes
sleep 1
/usr/bin/open -g "$APP"
sleep 6
echo "früt Flow restarted (menu-bar app). Recent log:"
{ grep -v "\[flow\]\[debug\]" "$LOG" 2>/dev/null || true; } | tail -6
