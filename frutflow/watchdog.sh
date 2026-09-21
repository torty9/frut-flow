#!/usr/bin/env bash
# Supervisor for früt Flow. Kept alive by launchd (KeepAlive). Every ~12s, if
# flow.py isn't running, relaunch it by opening frutflow.app. Launching THROUGH
# the .app bundle makes frutflow the TCC "responsible process", so the python
# child inherits frutflow's own Microphone + Input Monitoring + Accessibility
# grants (no Terminal needed). The app runs as a menu-bar item. Deterministic
# self-heal, independent of launchd StartInterval timing.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
APP="$HOME/Applications/frutflow.app"
FLOWDICTATE_DIR="$HOME/.flowdictate"
LOG="$FLOWDICTATE_DIR/watchdog.log"
FLOW_LOG="$FLOWDICTATE_DIR/flow.log"
MAX_LOG_BYTES=$((1024 * 1024))
MAX_FLOW_LOG_BYTES=$((5 * 1024 * 1024))

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

rotate_flow_log_if_needed() {
  local size
  [ -f "$FLOW_LOG" ] || return 0
  size="$(/usr/bin/wc -c < "$FLOW_LOG" 2>/dev/null | /usr/bin/tr -d '[:space:]')"
  case "$size" in
    ""|*[!0-9]*) return 0 ;;
  esac
  if [ "$size" -ge "$MAX_FLOW_LOG_BYTES" ]; then
    /bin/mv -f "$FLOW_LOG" "$FLOW_LOG.1" 2>/dev/null || return 0
  fi
}

prepare_flowdictate_dir() {
  umask 077
  mkdir -p "$FLOWDICTATE_DIR"
  chmod 700 "$FLOWDICTATE_DIR"
  rotate_log_if_needed
  : >> "$LOG"
  chmod 600 "$LOG"
}

process_cwd_matches_repo() {
  local pid="$1"
  local cwd code_dir
  cwd="$(LC_ALL=UTF-8 lsof -a -p "$pid" -d cwd -Fn 2>/dev/null |
    sed -n 's/^n//p' | head -n 1)"
  [ -n "$cwd" ] || return 1
  # Compare filesystem identity, not path bytes. macOS accepts both NFC and NFD
  # spellings of "früt"; argv and pwd can therefore name the same directory
  # with different Unicode byte sequences.
  [ "$cwd" -ef "$SCRIPT_DIR" ] && return 0
  # Also accept the directory the app itself launches from (restart.sh and
  # Quit frutflow.command already do). Without this, a copy started from an
  # unzipped folder was "not running" to us and got a futile relaunch — each
  # one loading the full model before exiting on the single-instance lock.
  code_dir="$(cat "$FLOWDICTATE_DIR/code_dir" 2>/dev/null || true)"
  [ -n "$code_dir" ] && [ "$cwd" -ef "$code_dir" ]
}

flow_process_matches() {
  local pid="$1"
  local args="$2"

  # Only the ASCII script basename is inspected in argv. The cwd/inode check is
  # the authoritative repository match and is immune to locale escaping and
  # Unicode-normalization differences in the full path.
  if [[ "$args" == *python*"/flow.py"* ||
        "$args" == *python*" flow.py"* ||
        "$args" == *python*" ./flow.py"* ]]; then
    # An instance launched through the bundle is ours regardless of its cwd.
    if [[ "$args" == *"frutflow.app/Contents/MacOS/"* ]]; then
      return 0
    fi
    process_cwd_matches_repo "$pid"
    return
  fi

  return 1
}

find_flow_pids() {
  # pgrep -fl prints pid + raw argv bytes. ps is NOT usable here: under
  # launchd's C locale it vis-encodes non-ASCII paths ("früt" -> "frM-CM-<t"),
  # so full-script-path comparisons never matched and the watchdog re-`open`ed
  # the already-running app every cycle (each open fires a reopen event).
  local pid args
  pgrep -fl 'flow\.py' 2>/dev/null | while read -r pid args; do
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

# Backoff guard: a deterministically-crashing flow.py (corrupt venv, bad
# upgrade) would otherwise be relaunched every ~20s forever — menu-bar
# flicker, battery drain, log churn. After 5 consecutive relaunches with no
# healthy run in between, retry only every 5 minutes (self-heal never fully
# stops; a SUSTAINED healthy run resets the counter).
#
# "Healthy" means seen running continuously for HEALTHY_RESET_SECONDS. The
# counter used to reset on a single sighting 20s after the relaunch — but a
# crash that comes after a slow model load (or any deterministic failure later
# than that) was then "healthy" every time and never reached the backoff.
HEALTHY_RESET_SECONDS="${FRUTFLOW_HEALTHY_RESET_SECONDS:-120}"
consecutive_relaunches=0
healthy_since=0   # epoch second the current unbroken running streak began; 0 = none

relaunch_app() { /usr/bin/open -g "$APP"; }
pause() { sleep "$1"; }

# One supervision step at epoch second $1. Kept separate from the loop (and the
# loop guarded at the bottom) so tests can drive it with stubbed
# flow_is_running / relaunch_app / pause and a fake clock.
watchdog_tick() {
  local now="$1" ts
  if ! flow_is_running; then
    healthy_since=0
    consecutive_relaunches=$((consecutive_relaunches + 1))
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    rotate_log_if_needed
    rotate_flow_log_if_needed
    : >> "$LOG"
    chmod 600 "$LOG"
    echo "[$ts] flow.py not running — relaunching frutflow.app (attempt $consecutive_relaunches)" >> "$LOG"
    relaunch_app
    pause 8
    if [ "$consecutive_relaunches" -ge 5 ]; then
      ts="$(date '+%Y-%m-%d %H:%M:%S')"
      echo "[$ts] $consecutive_relaunches relaunches without a healthy run — backing off to 5-minute retries. See $FLOW_LOG for the crash." >> "$LOG"
      pause 300
    fi
  else
    [ "$healthy_since" -ne 0 ] || healthy_since="$now"
    if [ $((now - healthy_since)) -ge "$HEALTHY_RESET_SECONDS" ]; then
      consecutive_relaunches=0
    fi
  fi
}

main() {
  prepare_flowdictate_dir
  while true; do
    if [ ! -d "$APP" ]; then
      # The user dragged frutflow.app to the Trash — the Mac way to uninstall.
      # Stop supervising; the LaunchAgent's KeepAlive/PathState (see the
      # installer) then leaves us stopped instead of respawning this loop.
      ts="$(date '+%Y-%m-%d %H:%M:%S')"
      echo "[$ts] $APP is gone — watchdog exiting. (Re-run Install frut Flow.command to reinstall.)" >> "$LOG"
      exit 0
    fi
    watchdog_tick "$(date +%s)"
    pause 12
  done
}

# Run only when executed (launchd: /bin/bash watchdog.sh), not when sourced by
# the test-suite.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main
fi
