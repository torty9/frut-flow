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
früt Flow — READ ME FIRST
=========================

Thanks for trying früt Flow! It is a free voice-dictation app for your Mac.
Hold a key, talk, let go, and your words are typed wherever your cursor is.
Everything runs on your own Mac: no account, no subscription, and your voice
never leaves your computer.


BEFORE YOU START
----------------
You need:

- A Mac with an Apple chip (M1, M2, M3, M4 or newer). Intel Macs are not
  supported. Apple menu > About This Mac shows which chip you have.
- macOS 14 (Sonoma) or newer.
- Internet during setup and about 3 to 5 GB of free disk space. Setup
  downloads the speech models once; after that, dictation works offline.
- Your Mac login password. The installer needs it once if it has to install
  a helper tool called Homebrew.

Set aside 15 to 30 minutes. Almost all of it is waiting for downloads.


INSTALL
-------
1. Open the disk image (double-click the .dmg file) if it is not open yet.
   A window appears with this note and "Install frut Flow.command".

2. Double-click "Install frut Flow.command".

   macOS will most likely say it cannot verify the file, or that it could
   not check it for malware. That is expected: this app was shared with you
   by a friend, not downloaded from the App Store. To allow it:

     a. Close the warning with "Done" or "OK". Do NOT choose "Move to Trash".
     b. Open System Settings > Privacy & Security.
     c. Scroll down to the "Security" section. It says
        "Install frut Flow.command" was blocked. Click "Open Anyway".
     d. Confirm with "Open". macOS may ask for your password or Touch ID.

   Shortcut on macOS 14 (Sonoma): Control-click the file, choose "Open",
   then click "Open" again in the dialog.

3. A Terminal window opens and does the rest. Follow what it says:

   - If it asks "Install Homebrew now? [y/N]", type  y  and press Return.
     Homebrew asks for your Mac password (nothing shows while you type;
     press Return when done) and asks you to press Return to continue.
     It may also install Apple's command line tools, which can take a
     while.
   - The installer then sets up the app and downloads the speech models.
     The window can look frozen for several minutes at a time. Leave it
     alone and do not close it.
   - When you see "Setup finished", press Return. früt Flow starts and a
     microphone icon appears in the menu bar at the top right of the screen.

4. Grant the three permissions it needs. A Welcome window opens the first
   time and walks you through them, with a button for each. Or do it
   yourself: open System Settings > Privacy & Security and turn on
   "frutflow" under:

     - Microphone         so it can hear you
     - Accessibility      so it can type into other apps
     - Input Monitoring   so it can notice you holding the dictation key

   Afterwards, click the microphone icon in the menu bar and choose
   "Restart" so the new permissions take effect.

5. Done. You can close the Terminal window and eject the disk image.
   Everything has been copied to your Mac.


USING IT
--------
1. Click into any text box: a message, an email, a document, a search field.
2. Press and HOLD the Right Option (⌥) key, speak, then let go.
3. About a second later, your words appear at the cursor, cleaned up and
   punctuated.

Handy extras:

- Say "period", "comma", "question mark", "new paragraph", or
  "quote ... end quote" to punctuate.
- Said something you did not mean? Hold the key and say "never mind". The
  last dictation is deleted.
- If it misspells a name, just fix it in the text. It learns from your edit
  and gets it right next time.
- The menu-bar icon has History, Teach a Word, Settings (change the key, the
  language, and more), Restart, and Quit.
- früt Flow starts by itself whenever you log in. To stop it, click the
  menu-bar icon and choose "Quit früt Flow".


IF SOMETHING GOES WRONG
-----------------------
- Holding the key does nothing, or the menu-bar icon shows a warning sign:
  turn on Input Monitoring for frutflow, then Restart from the menu-bar icon.
- Nothing gets typed: turn on Accessibility for frutflow, then Restart.
- Lost the Welcome window? Click the menu-bar icon and choose
  "Welcome / Setup..." to open it again.
- No microphone icon in the menu bar: in Finder choose Go > Home, open the
  Applications folder there, and double-click frutflow. (This is the
  Applications folder inside your home folder, not the main one.)
- The install stopped with an error: simply double-click
  "Install frut Flow.command" again. It is safe to re-run and it picks up
  where it left off.
- Still stuck? Click the menu-bar icon, choose "Open Log", and send that
  log file to the person who gave you this.


PRIVACY
-------
Speech recognition runs on your Mac's own chip. No audio and no text is ever
sent anywhere. The one-time setup downloads open-source software and model
files from the internet; after that, dictation works with Wi-Fi turned off.

früt Flow is free and open source (MIT license).
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
