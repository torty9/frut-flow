#!/usr/bin/env bash
# Supervisor for früt Flow. Kept alive by launchd (KeepAlive). Every ~12s, if
# flow.py isn't running, relaunch it by opening frutflow.app. Launching THROUGH
# the .app bundle makes frutflow the TCC "responsible process", so the python
# child inherits frutflow's own Microphone + Input Monitoring + Accessibility
# grants (no Terminal needed). The app runs as a menu-bar item. Deterministic
# self-heal, independent of launchd StartInterval timing.
set -uo pipefail
APP="$HOME/Applications/frutflow.app"
LOG="$HOME/.flowdictate/watchdog.log"
mkdir -p "$HOME/.flowdictate"
while true; do
  if ! pgrep -f "flow\.py" >/dev/null 2>&1; then
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] flow.py not running — relaunching frutflow.app" >> "$LOG"
    open -g "$APP"
    sleep 8
  fi
  sleep 12
done
