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

# Same floors the installers enforce; failing here is friendlier than a
# cryptic pip resolution error minutes later. mlx publishes wheels only for
# macOS 14+ (Sonoma), and numpy/parakeet floors need Python 3.10+.
MACOS_MAJOR="$(/usr/bin/sw_vers -productVersion | cut -d. -f1)"
if [ "${MACOS_MAJOR:-0}" -lt 14 ]; then
  echo "[run] früt Flow needs macOS 14 (Sonoma) or newer — this Mac runs" \
       "$(/usr/bin/sw_vers -productVersion)." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "[run] Homebrew is not installed. Install it from https://brew.sh," \
       "or use 'Install frut Flow.command' which sets everything up." >&2
  exit 1
fi

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
  # Reject a too-old interpreter BEFORE building the venv: a 3.9 venv fails
  # later with a baffling "no matching distribution" from pip instead.
  if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "[run] python3 is $(python3 -V 2>&1 | cut -d' ' -f2) but früt Flow needs 3.10+." >&2
    echo "      brew install python@3.12   (then re-run ./run.sh)" >&2
    exit 1
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
