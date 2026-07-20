#!/usr/bin/env bash
# Restart früt Flow (use after changing ~/.flowdictate/config.json).
# Stops the running instance and relaunches it as frutflow.app (menu-bar app).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
FLOWDICTATE_DIR="$HOME/.flowdictate"
APP="$HOME/Applications/frutflow.app"
LOG="$FLOWDICTATE_DIR/flow.log"
MAX_LOG_BYTES=$((5 * 1024 * 1024))

rotate_log_if_needed() {
  local size
  [ -f "$LOG" ] || return 0
  size="$(/usr/bin/wc -c < "$LOG" 2>/dev/null | /usr/bin/tr -d '[:space:]')"
  case "$size" in
    ""|*[!0-9]*) return 0 ;;
  esac
  if [ "$size" -ge "$MAX_LOG_BYTES" ]; then
    /bin/mv -f "$LOG" "$LOG.1" 2>/dev/null || return 0
  fi
}

prepare_flowdictate_dir() {
  umask 077
  mkdir -p "$FLOWDICTATE_DIR"
  chmod 700 "$FLOWDICTATE_DIR"
  : >> "$LOG"
  chmod 600 "$LOG"
}

process_cwd_matches_flow() {
  local pid="$1"
  local cwd
  cwd="$(LC_ALL=UTF-8 lsof -a -p "$pid" -d cwd -Fn 2>/dev/null |
    sed -n 's/^n//p' | head -n 1)"
  [ -n "$cwd" ] || return 1
  # Compare directory identity so NFC/NFD spellings of "früt" are equivalent.
  # Accept this script's folder AND the installed instance's code directory,
  # so restart works from any copy of the script, not only the one sitting
  # next to the running code.
  [ "$cwd" -ef "$SCRIPT_DIR" ] && return 0
  local code_dir=""
  if [ -f "$FLOWDICTATE_DIR/code_dir" ]; then
    code_dir="$(head -n 1 "$FLOWDICTATE_DIR/code_dir" 2>/dev/null || true)"
  fi
  [ -n "$code_dir" ] && [ -d "$code_dir" ] && [ "$cwd" -ef "$code_dir" ]
}

flow_process_matches() {
  local pid="$1"
  local args="$2"

  # Avoid comparing the full non-ASCII argv path: ps locale escaping and Unicode
  # normalization can change its bytes. The cwd/inode check establishes identity.
  if [[ "$args" == *python*"/flow.py"* ||
        "$args" == *python*" flow.py"* ||
        "$args" == *python*" ./flow.py"* ]]; then
    # The installed .app instance is identifiable by its ASCII bundle path.
    if [[ "$args" == *"frutflow.app/Contents/MacOS/"* ]]; then
      return 0
    fi
    process_cwd_matches_flow "$pid"
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
rotate_log_if_needed
/usr/bin/open -g "$APP"
sleep 6
echo "früt Flow restarted (menu-bar app). Recent log:"
{ grep -v "\[flow\]\[debug\]" "$LOG" 2>/dev/null || true; } | tail -6
