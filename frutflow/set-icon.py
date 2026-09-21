#!/usr/bin/env python3
"""Apply the früt Flow icon to frutflow.app WITHOUT re-signing the bundle.

Re-signing the .app resets its macOS permission grants (Microphone / Accessibility /
Input Monitoring). So instead of baking the icon into Contents/Resources + Info.plist
(which would change the code signature), this uses the Finder custom-icon mechanism
(NSWorkspace.setIcon), which attaches the icon to the bundle without touching the seal.

This route also matters on macOS 26 (Tahoe): the glossy badge does not fill the
system's rounded icon tile, so when it is served from Contents/Resources the Dock
draws it shrunken on a grey backing plate. A Finder custom icon is drawn as-is,
badge, glow and all. If the grey plate ever comes back, re-run this script and
then `killall Dock`.

Usage:  ./.venv/bin/python3 set-icon.py [path/to/icon.icns]
        (defaults to assets/frut-flow.icns next to this script)
"""
import sys
from pathlib import Path

from AppKit import NSImage, NSWorkspace

APP = str(Path.home() / "Applications" / "frutflow.app")
DEFAULT_ICON = Path(__file__).resolve().parent / "assets" / "frut-flow.icns"


def main() -> int:
    icon = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ICON
    if not icon.exists():
        print(f"icon not found: {icon}")
        return 1
    img = NSImage.alloc().initWithContentsOfFile_(str(icon))
    if img is None:
        print(f"could not load image: {icon}")
        return 1
    ok = NSWorkspace.sharedWorkspace().setIcon_forFile_options_(img, APP, 0)
    print(f"applied {icon.name} to {APP}: {'ok' if ok else 'FAILED'}")
    if ok:
        # nudge Finder to refresh the icon. Only on success: touch() on a
        # MISSING app would create a stray empty file squatting on the exact
        # path the watchdog and installer expect the real bundle at.
        Path(APP).touch()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
