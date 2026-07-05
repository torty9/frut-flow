#!/usr/bin/env bash
# Lean runner used by the login agent and for manual restarts.
# (No reinstall — assumes ./run.sh has already set up .venv once.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
FLOWDICTATE_DIR="$HOME/.flowdictate"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

umask 077
mkdir -p "$FLOWDICTATE_DIR"
chmod 700 "$FLOWDICTATE_DIR"

cd "$SCRIPT_DIR"
if [ ! -x "$VENV_PY" ]; then
  echo "[start] missing virtualenv. Run ./run.sh once to set up dependencies." >&2
  exit 1
fi
exec "$VENV_PY" "$SCRIPT_DIR/flow.py"
