#!/usr/bin/env bash
# One-shot launcher: creates a virtualenv, installs deps, and starts früt Flow.
# Usage:  ./run.sh          (run the app)
#         ./run.sh --setup  (write config + show permissions)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
FLOWDICTATE_DIR="$HOME/.flowdictate"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
REQ="$SCRIPT_DIR/requirements.txt"
REQ_STAMP="$VENV_DIR/.requirements.sha256"

cd "$SCRIPT_DIR"

# PortAudio is the native library sounddevice binds to.
if ! brew list portaudio >/dev/null 2>&1; then
  echo "[run] installing portaudio via Homebrew..."
  brew install portaudio
fi

if [ ! -x "$VENV_PY" ]; then
  if [ -d "$VENV_DIR" ]; then
    echo "[run] existing virtualenv is stale; recreating..."
    rm -rf "$VENV_DIR"
  fi
  echo "[run] creating virtualenv..."
  python3 -m venv "$VENV_DIR"
fi

REQ_HASH="$(/usr/bin/shasum -a 256 "$REQ" | /usr/bin/awk '{print $1}')"
if [ ! -f "$REQ_STAMP" ] || [ "$(cat "$REQ_STAMP")" != "$REQ_HASH" ]; then
  echo "[run] installing python deps..."
  "$VENV_PY" -m pip install --quiet --upgrade pip
  "$VENV_PY" -m pip install --quiet -r "$REQ"
  printf '%s\n' "$REQ_HASH" > "$REQ_STAMP"
else
  echo "[run] python deps already match requirements.txt"
fi

umask 077
mkdir -p "$FLOWDICTATE_DIR"
chmod 700 "$FLOWDICTATE_DIR"

exec "$VENV_PY" "$SCRIPT_DIR/flow.py" "$@"
