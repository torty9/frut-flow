#!/usr/bin/env bash
# Build a clean, shareable macOS installer zip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
DIST_DIR="$ROOT/dist"
PACKAGE_DIR_NAME="frut-flow"
STAGE="$DIST_DIR/$PACKAGE_DIR_NAME"
DATE_STAMP="${1:-$(date +%Y-%m-%d)}"
ZIP_PATH="$DIST_DIR/frut-flow-mac-share-$DATE_STAMP.zip"

cd "$ROOT"

required=(
  "flow.py"
  "run.sh"
  "restart.sh"
  "watchdog.sh"
  "set-icon.py"
  "requirements.txt"
  "config.example.json"
  "README.md"
  "LICENSE"
  "Install frut Flow.command"
  "Install frut Flow from GitHub.command"
  "Start früt Flow.command"
  "Teach a Word.command"
  "Quit frutflow.command"
)

for path in "${required[@]}"; do
  if [ ! -e "$ROOT/$path" ]; then
    echo "Missing required package file: $path" >&2
    exit 1
  fi
done

rm -rf "$STAGE"
mkdir -p "$STAGE"

for path in "${required[@]}"; do
  cp -p "$ROOT/$path" "$STAGE/"
done

# Ship runtime assets only. Logo/wordmark source files and the repair-model
# eyeballing harness remain useful to developers but do not belong in an end-user
# installer archive.
mkdir -p "$STAGE/assets/phosphor"
cp -p "$ROOT/assets/frut-flow.icns" "$STAGE/assets/"
rsync -a --exclude ".DS_Store" --exclude "microphone-stage-fill.png" \
  "$ROOT/assets/phosphor/" "$STAGE/assets/phosphor/"

cat > "$STAGE/README_FIRST.txt" <<'README'
frut Flow - quick setup

This is a local macOS voice dictation app. It runs on your Mac and does not
send audio or dictated text to a cloud service.

Requirements:
- Apple Silicon Mac
- macOS 11 or newer
- Homebrew installed from https://brew.sh
- Python 3.10 or newer

First run:
1. Unzip this folder.
2. Double-click:
   Install frut Flow.command
3. Follow the messages in the Terminal window.

The installer copies the app files to:
~/Applications/frut-flow

It also creates:
~/Applications/frutflow.app

The first install downloads the local Python packages and model weights once:
Parakeet for speech recognition and Qwen for on-device repair cleanup.

If macOS says Apple cannot verify the file:
1. Control-click "Install frut Flow.command".
2. Choose Open.
3. Click Open again in the warning dialog.

If there is no Open button, go to System Settings > Privacy & Security, scroll
to Security, and click Open Anyway for "Install frut Flow.command". Only do this
for a zip you received from someone you trust.

One-time macOS permissions:
Open System Settings > Privacy & Security and allow frutflow for:
- Microphone
- Accessibility
- Input Monitoring

After changing permissions, fully quit and reopen frutflow.

Manual fallback:
1. Open Terminal.
2. cd into the unzipped "frut-flow" folder.
3. Run:
   chmod +x *.sh *.command
   ./run.sh --setup
   ./run.sh
README

chmod +x "$STAGE"/*.sh "$STAGE"/*.command
find "$STAGE" -name ".DS_Store" -delete
rm -f "$ZIP_PATH"

(
  cd "$DIST_DIR"
  /usr/bin/zip -qry "$ZIP_PATH" "$PACKAGE_DIR_NAME" \
    -x "*.DS_Store" \
    -x "__MACOSX/*" \
    -x "*/__pycache__/*" \
    -x "*/.venv/*"
)

echo "Built installer zip:"
echo "  $ZIP_PATH"
