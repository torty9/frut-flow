#!/bin/bash
# Double-click to FULLY stop frutflow.
# The watchdog would relaunch the app within ~20s, so we must disable the
# auto-start/watchdog FIRST, then kill the running app. frutflow will stay
# stopped until you log in again (or run the re-enable line below).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
FLOW_SCRIPT="$SCRIPT_DIR/flow.py"
FLOWDICTATE_DIR="$HOME/.flowdictate"

prepare_flowdictate_dir() {
  umask 077
  mkdir -p "$FLOWDICTATE_DIR"
  chmod 700 "$FLOWDICTATE_DIR"
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
/bin/launchctl bootout "gui/$(id -u)/com.frutflow.dictation" 2>/dev/null || true
stop_flow_processes
echo "frutflow stopped — and it will NOT auto-restart."
echo
echo "To turn frutflow back on now (without logging out):"
echo "  launchctl bootstrap gui/\$(id -u) \"\$HOME/Library/LaunchAgents/com.frutflow.dictation.plist\""
echo "…or just log in again, or double-click frutflow.app in ~/Applications."
