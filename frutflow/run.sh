#!/usr/bin/env bash
# One-shot launcher: creates a virtualenv, installs deps, and starts früt Flow.
# Usage:  ./run.sh          (run the app)
#         ./run.sh --setup  (write config + show permissions)
set -euo pipefail
cd "$(dirname "$0")"

# PortAudio is the native library sounddevice binds to.
if ! brew list portaudio >/dev/null 2>&1; then
  echo "[run] installing portaudio via Homebrew..."
  brew install portaudio
fi

if [ ! -d .venv ]; then
  echo "[run] creating virtualenv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[run] installing python deps..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

python flow.py "$@"
