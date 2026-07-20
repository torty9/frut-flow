#!/usr/bin/env bash
# Build an unsigned, shareable macOS disk image.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
DIST_DIR="$ROOT/dist"
DATE_STAMP="${1:-$(date +%Y-%m-%d)}"
PAYLOAD_DIR="$DIST_DIR/frut-flow"
STAGE="$DIST_DIR/.dmg-stage"
DMG_PATH="$DIST_DIR/frut-flow-mac-installer-$DATE_STAMP.dmg"
INSTALLER_NAME="Install frut Flow.command"

cleanup() {
  rm -rf "$STAGE"
}
trap cleanup EXIT

command -v hdiutil >/dev/null 2>&1 || {
  echo "hdiutil is required to build the macOS disk image." >&2
  exit 1
}

# Reuse the zip packager as the single source of truth for the end-user payload.
"$ROOT/package-mac.sh" "$DATE_STAMP"

rm -rf "$STAGE"
mkdir -p "$STAGE/.frut-flow"
rsync -a "$PAYLOAD_DIR/" "$STAGE/.frut-flow/"

cat > "$STAGE/$INSTALLER_NAME" <<'INSTALLER'
#!/bin/bash
# Visible DMG entry point. The complete installer payload is kept in the hidden
# .frut-flow folder so the disk image presents one obvious file to double-click.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -P)"
INSTALLER="$HERE/.frut-flow/Install frut Flow.command"

if [ ! -f "$INSTALLER" ]; then
  echo "The frut Flow installer payload is missing."
  echo "Please download a fresh copy of the disk image."
  echo
  read -r -p "Press Return to close this window."
  exit 1
fi

exec /bin/bash "$INSTALLER"
INSTALLER

cat > "$STAGE/READ ME FIRST.txt" <<'README'
frut Flow - installation

Double-click:
  Install frut Flow.command

The installer downloads and configures everything needed, including:
- Homebrew and Python when they are not already installed
- Python dependencies and PortAudio
- Parakeet on-device speech recognition
- Qwen on-device cleanup

This disk image is not signed with an Apple Developer ID. If macOS blocks it:
1. Try to open "Install frut Flow.command" once.
2. Open System Settings > Privacy & Security.
3. Scroll to Security and click Open Anyway.
4. Confirm by clicking Open.

After installation, allow frutflow in Privacy & Security for:
- Microphone
- Accessibility
- Input Monitoring

Requirements:
- Apple Silicon Mac
- macOS 11 or newer
- Internet access during installation
README

chmod +x "$STAGE/$INSTALLER_NAME"
find "$STAGE" -name ".DS_Store" -delete
xattr -cr "$STAGE" 2>/dev/null || true
rm -f "$DMG_PATH"

hdiutil create \
  -volname "frut Flow Installer" \
  -srcfolder "$STAGE" \
  -fs HFS+ \
  -format UDZO \
  -ov \
  "$DMG_PATH"

hdiutil verify "$DMG_PATH"

echo
echo "Built unsigned installer DMG:"
echo "  $DMG_PATH"
