#!/bin/bash
# Double-click installer for people receiving frut Flow as a zip.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd -P)"
INSTALL_DIR="$HOME/Applications/frut-flow"
APP="$HOME/Applications/frutflow.app"
FLOWDICTATE_DIR="$HOME/.flowdictate"
CODE_PTR="$FLOWDICTATE_DIR/code_dir"
AGENT="$HOME/Library/LaunchAgents/com.frutflow.dictation.plist"
LABEL="com.frutflow.dictation"
ENABLE_LOCAL_REPAIR="${FRUTFLOW_ENABLE_LOCAL_REPAIR:-1}"
PRELOAD_MODELS="${FRUTFLOW_PRELOAD_MODELS:-1}"

enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

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

  echo "Homebrew is needed to install the local audio and Python dependencies."
  echo "This will run the official installer from https://brew.sh."
  echo
  read -r -p "Install Homebrew now? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) fail "Homebrew was not installed." ;;
  esac

  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  load_homebrew_path
  command -v brew >/dev/null 2>&1 ||
    fail "Homebrew installed, but brew is not available on PATH."
}

ensure_python() {
  if python_is_new_enough; then
    return 0
  fi

  echo "Installing Python 3.10 or newer with Homebrew..."
  echo
  brew install python
  load_homebrew_path
  python_is_new_enough || fail "Python 3.10 or newer is still not available."
}

clear || true
echo "frut Flow installer"
echo

if [ "$(uname -m)" != "arm64" ]; then
  fail "This version is built for Apple Silicon Macs."
fi

# mlx publishes wheels only for macOS 14+; gating here beats a cryptic pip
# "no matching distribution" failure after minutes of Homebrew setup.
MACOS_MAJOR="$(/usr/bin/sw_vers -productVersion | cut -d. -f1)"
if [ "${MACOS_MAJOR:-0}" -lt 14 ]; then
  fail "frut Flow needs macOS 14 (Sonoma) or newer — this Mac runs $(/usr/bin/sw_vers -productVersion)."
fi

ensure_homebrew
ensure_python

mkdir -p "$HOME/Applications"

if [ "$SOURCE_DIR" != "$INSTALL_DIR" ]; then
  echo "Installing app files to:"
  echo "  $INSTALL_DIR"
  echo
  mkdir -p "$INSTALL_DIR"
  rsync -a \
    --exclude ".DS_Store" \
    --exclude ".venv" \
    --exclude "__pycache__" \
    --exclude "dist" \
    "$SOURCE_DIR"/ "$INSTALL_DIR"/
else
  echo "App files are already in $INSTALL_DIR"
  echo
fi

cd "$INSTALL_DIR"
chmod +x ./*.sh ./*.command 2>/dev/null || true

umask 077
mkdir -p "$FLOWDICTATE_DIR"
chmod 700 "$FLOWDICTATE_DIR"
printf "%s\n" "$INSTALL_DIR" > "$CODE_PTR"
chmod 600 "$CODE_PTR"

echo "Installing dependencies and writing the default config..."
echo
setup_args=(--setup)
if enabled "$ENABLE_LOCAL_REPAIR"; then
  setup_args+=(--enable-local-repair)
fi
"$INSTALL_DIR/run.sh" "${setup_args[@]}"

if enabled "$PRELOAD_MODELS"; then
  echo
  echo "Downloading and warming local models..."
  echo "This can take a while the first time. Future launches use the local cache."
  echo
  "$INSTALL_DIR/run.sh" --preload-models
fi

create_app_bundle() {
  # An existing, working venv may use an older Python than the current PATH.
  # The embedded interpreter must match its native extension ABI, not whichever
  # python3 Homebrew installed most recently.
  local venv_py="$INSTALL_DIR/.venv/bin/python"
  local want have python_path
  want="$("$venv_py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  python_path="$("$venv_py" -c 'import sys; print(sys._base_executable)')"
  if [ -d "$APP" ]; then
    # Keep the permissioned bundle, but refresh its embedded Python when the
    # system Python moved on: after a brew upgrade the old copied binary would
    # otherwise run against a NEWER venv's site-packages — an ABI mismatch
    # that crashes at launch and feeds the watchdog's relaunch loop.
    have="$("$APP/Contents/MacOS/python3" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo missing)"
    if [ "$have" != "$want" ]; then
      echo "Updating the app bundle's Python ($have -> $want)..."
      cp -f "$python_path" "$APP/Contents/MacOS/python3"
      chmod +x "$APP/Contents/MacOS/python3"
      if command -v codesign >/dev/null 2>&1; then
        codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
      fi
      echo "(If macOS asks for the Microphone/Accessibility/Input Monitoring"
      echo " permissions again, re-allow frutflow in Privacy & Security.)"
    else
      echo "Using existing app bundle:"
      echo "  $APP"
    fi
    echo
    return 0
  fi

  echo "Creating app bundle:"
  echo "  $APP"
  echo

  mkdir -p "$APP/Contents/MacOS"

  cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>frutflow</string>
  <key>CFBundleExecutable</key>
  <string>frutflow</string>
  <key>CFBundleIdentifier</key>
  <string>com.frutflow.dictation</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>frutflow</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>LSMultipleInstancesProhibited</key>
  <true/>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>frutflow records your voice to transcribe it into text on-device.</string>
</dict>
</plist>
PLIST

  cat > "$APP/Contents/MacOS/frutflow" <<'LAUNCHER'
#!/bin/bash
# frutflow.app launcher. The bundle is the macOS permission owner; flow.py and
# the virtualenv live outside the bundle so updating the app code does not
# require re-signing the permissioned bundle.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_PY="$HERE/python3"
FLOWDICTATE_DIR="$HOME/.flowdictate"
LOG="$FLOWDICTATE_DIR/flow.log"
PTR="$FLOWDICTATE_DIR/code_dir"
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

umask 077
mkdir -p "$FLOWDICTATE_DIR"
chmod 700 "$FLOWDICTATE_DIR"
rotate_log_if_needed
: >> "$LOG"
chmod 600 "$LOG"

CODE_DIR=""
if [ -f "$PTR" ]; then
  d="$(cat "$PTR")"
  [ -n "$d" ] && [ -f "$d/flow.py" ] && CODE_DIR="$d"
fi

if [ -z "$CODE_DIR" ]; then
  echo "[frutflow] Cannot find flow.py. Re-run Install frut Flow.command." >> "$LOG"
  exit 1
fi

SP=""
for cand in "$CODE_DIR"/.venv/lib/python*/site-packages; do
  [ -d "$cand" ] && SP="$cand" && break
done

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
export PYTHONPATH="${SP:+$SP:}${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

cd "$CODE_DIR" || exit 1
exec /usr/bin/arch -arm64 "$BUNDLE_PY" "$CODE_DIR/flow.py" >> "$LOG" 2>&1
LAUNCHER

  chmod +x "$APP/Contents/MacOS/frutflow"
  cp "$python_path" "$APP/Contents/MacOS/python3"
  chmod +x "$APP/Contents/MacOS/python3"

  if command -v codesign >/dev/null 2>&1; then
    codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
  fi
}

create_launch_agent() {
  mkdir -p "$HOME/Library/LaunchAgents"
  python3 - "$AGENT" "$INSTALL_DIR/watchdog.sh" <<'PY'
import plistlib
import sys
from pathlib import Path

agent_path = Path(sys.argv[1])
watchdog = sys.argv[2]
plist = {
    "Label": "com.frutflow.dictation",
    "ProgramArguments": ["/bin/bash", watchdog],
    "RunAtLoad": True,
    "KeepAlive": True,
}
with agent_path.open("wb") as f:
    plistlib.dump(plist, f)
PY

  chmod 644 "$AGENT"
  /bin/launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
  /bin/launchctl bootstrap "gui/$(id -u)" "$AGENT" >/dev/null 2>&1 || true
  /bin/launchctl enable "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
}

create_app_bundle

if [ -x "$INSTALL_DIR/.venv/bin/python" ] && [ -f "$INSTALL_DIR/assets/frut-flow.icns" ]; then
  "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/set-icon.py" >/dev/null 2>&1 || true
fi

create_launch_agent

echo
echo "Setup finished."
echo
echo "One-time permission step:"
echo "Open System Settings > Privacy & Security and allow frutflow for:"
echo "- Microphone"
echo "- Accessibility"
echo "- Input Monitoring"
echo
echo "If frutflow is not listed yet, press Return below to launch it once,"
echo "then open Privacy & Security again."
echo
read -r -p "Press Return to start frut Flow now."

/usr/bin/open -g "$APP"

echo
echo "frut Flow is installed."
echo "You can start it later from:"
echo "  $APP"
echo
read -r -p "Press Return to close this window."
