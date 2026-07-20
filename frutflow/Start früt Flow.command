#!/bin/bash
# Double-click this file to start früt Flow. Keep the Terminal window it opens
# running while you dictate. Close the window (or press Ctrl-C) to stop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
FLOWDICTATE_DIR="$HOME/.flowdictate"
LOG="$FLOWDICTATE_DIR/flow.log"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

cd "$SCRIPT_DIR"
clear || true
echo "Starting früt Flow — loading the speech model (a few seconds)…"
echo "Keep this window open. Hold the Right Option key to dictate."
echo
# Create the log owner-only (0600): it can contain dictated text, so no other
# local user should be able to read it. APPEND + rotate at 5 MB like the
# watchdog and restart.sh do — truncating here would destroy the history the
# rotation scheme exists to preserve.
umask 077
mkdir -p "$FLOWDICTATE_DIR"
chmod 700 "$FLOWDICTATE_DIR"
if [ -f "$LOG" ] && [ "$(/usr/bin/wc -c < "$LOG" | /usr/bin/tr -d '[:space:]')" -ge $((5 * 1024 * 1024)) ] 2>/dev/null; then
  /bin/mv -f "$LOG" "$LOG.1" 2>/dev/null || true
fi
: >> "$LOG"
chmod 600 "$LOG"
if [ ! -x "$VENV_PY" ]; then
  echo "Missing virtualenv. Run ./run.sh once to set up dependencies."
  exit 1
fi
# force arm64 so the native speech wheels load; mirror output to the log so
# status (permissions) can be reviewed after the fact.
exec > >(tee -a "$LOG") 2>&1
exec /usr/bin/arch -arm64 "$VENV_PY" "$SCRIPT_DIR/flow.py"
