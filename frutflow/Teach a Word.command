#!/bin/bash
# Double-click to teach früt Flow a correction — pops two simple boxes, no typing
# in Terminal. Takes effect on your next dictation (no restart needed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
FLOWDICTATE_DIR="$HOME/.flowdictate"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

cd "$SCRIPT_DIR"

# NOTE: user-entered words are passed to osascript as `on run argv` ARGUMENTS,
# never interpolated into the AppleScript source. That keeps a stray quote (or a
# crafted "…" & (do shell script …) payload) as inert text instead of executable
# AppleScript. Do not "simplify" this back to string interpolation.

heard=$(/usr/bin/osascript <<'OSA' 2>/dev/null
try
  set r to text returned of (display dialog "Teach früt Flow a fix." & return & return & "What did it type WRONG?  (the word it got wrong)" default answer "" with title "Teach früt Flow" buttons {"Cancel", "Next"} default button "Next")
  return r
on error
  return ""
end try
OSA
)
[ -z "$heard" ] && exit 0

correct=$(/usr/bin/osascript - "$heard" <<'OSA' 2>/dev/null
on run argv
  set heardWord to item 1 of argv
  try
    set r to text returned of (display dialog "What should it have typed instead of \"" & heardWord & "\"?" default answer "" with title "Teach früt Flow" buttons {"Cancel", "Save"} default button "Save")
    return r
  on error
    return ""
  end try
end run
OSA
)
[ -z "$correct" ] && exit 0

umask 077
mkdir -p "$FLOWDICTATE_DIR"
chmod 700 "$FLOWDICTATE_DIR"

if [ ! -x "$VENV_PY" ] || ! "$VENV_PY" "$SCRIPT_DIR/flow.py" --correct "$heard" "$correct" >/dev/null 2>&1; then
  /usr/bin/osascript <<'OSA' >/dev/null 2>&1
display dialog "Could not save the correction. Run ./run.sh once to set up früt Flow, then try again." with title "früt Flow" buttons {"OK"} default button "OK"
OSA
  exit 1
fi

/usr/bin/osascript - "$correct" "$heard" <<'OSA' >/dev/null 2>&1
on run argv
  set c to item 1 of argv
  set h to item 2 of argv
  display dialog "Saved. früt Flow will now type \"" & c & "\" instead of \"" & h & "\" from your next dictation on." with title "früt Flow" buttons {"Great"} default button "Great"
end run
OSA
