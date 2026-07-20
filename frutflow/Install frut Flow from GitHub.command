#!/bin/bash
# Single-file installer: send this file to someone, and it downloads the app
# from GitHub before running the normal macOS installer.
set -euo pipefail

REPO_ZIP_URL="${FRUTFLOW_REPO_ZIP_URL:-https://github.com/torty9/frut-flow/archive/refs/heads/main.zip}"
TMP_ROOT=""

cleanup() {
  if [ -n "$TMP_ROOT" ] && [ -d "$TMP_ROOT" ]; then
    rm -rf "$TMP_ROOT"
  fi
}
trap cleanup EXIT

pause() {
  echo
  read -r -p "Press Return to close this window."
}

fail() {
  echo
  echo "Install failed: $*"
  pause
  exit 1
}

load_homebrew_path() {
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
}

python_is_new_enough() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

ensure_homebrew() {
  load_homebrew_path
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi

  echo "Homebrew is required to install PortAudio and Python packages."
  echo "This will run the official installer from https://brew.sh."
  echo
  read -r -p "Install Homebrew now? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) fail "Homebrew was not installed." ;;
  esac

  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  load_homebrew_path
  command -v brew >/dev/null 2>&1 || fail "Homebrew installed, but brew is not on PATH."
}

ensure_python() {
  if python_is_new_enough; then
    return 0
  fi

  echo "Python 3.10 or newer is required."
  echo "Installing Python with Homebrew..."
  echo
  brew install python
  load_homebrew_path
  python_is_new_enough || fail "Python 3.10 or newer is still not available."
}

clear || true
echo "frut Flow one-file installer"
echo

if [ "$(uname -m)" != "arm64" ]; then
  fail "This version is built for Apple Silicon Macs."
fi

ensure_homebrew
ensure_python

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/frut-flow-install.XXXXXX")"
ZIP_PATH="$TMP_ROOT/frut-flow.zip"
EXTRACT_DIR="$TMP_ROOT/source"
mkdir -p "$EXTRACT_DIR"

echo
echo "Downloading frut Flow from GitHub..."
echo "  $REPO_ZIP_URL"
echo
curl -fL "$REPO_ZIP_URL" -o "$ZIP_PATH" || fail "Could not download the GitHub zip."

ditto -x -k "$ZIP_PATH" "$EXTRACT_DIR" || fail "Could not unpack the GitHub zip."
SOURCE_DIR="$(find "$EXTRACT_DIR" -maxdepth 1 -type d -name 'frut-flow-*' -print -quit)"
[ -n "$SOURCE_DIR" ] || fail "Could not find the app folder inside the GitHub zip."

INSTALLER="$SOURCE_DIR/frutflow/Install frut Flow.command"
if [ ! -f "$INSTALLER" ]; then
  # Backward-compatible fallback for archives that put the app directly at root.
  INSTALLER="$SOURCE_DIR/Install frut Flow.command"
fi
[ -f "$INSTALLER" ] || fail "The downloaded repo does not contain the macOS installer."
chmod +x "$INSTALLER"

export FRUTFLOW_ENABLE_LOCAL_REPAIR="${FRUTFLOW_ENABLE_LOCAL_REPAIR:-1}"
export FRUTFLOW_PRELOAD_MODELS="${FRUTFLOW_PRELOAD_MODELS:-1}"

echo "Starting the frut Flow installer..."
echo
/bin/bash "$INSTALLER"
