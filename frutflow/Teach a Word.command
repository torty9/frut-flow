#!/bin/bash
# Double-click to teach Wispr DIY a correction — pops two simple boxes, no typing
# in Terminal. Takes effect on your next dictation (no restart needed).
cd "/Users/thorstenpfeiffer/diy wispr/frutflow" || exit 1

heard=$(osascript <<'OSA' 2>/dev/null
try
  set r to text returned of (display dialog "Teach Wispr DIY a fix." & return & return & "What did it type WRONG?  (the word it got wrong)" default answer "" with title "Teach Wispr DIY" buttons {"Cancel", "Next"} default button "Next")
  return r
on error
  return ""
end try
OSA
)
[ -z "$heard" ] && exit 0

correct=$(osascript <<OSA 2>/dev/null
try
  set r to text returned of (display dialog "What should it have typed instead of \"$heard\"?" default answer "" with title "Teach Wispr DIY" buttons {"Cancel", "Save"} default button "Save")
  return r
on error
  return ""
end try
OSA
)
[ -z "$correct" ] && exit 0

./.venv/bin/python flow.py --correct "$heard" "$correct" >/dev/null 2>&1
osascript -e "display dialog \"Saved. Wispr DIY will now type \\\"$correct\\\" instead of \\\"$heard\\\" from your next dictation on.\" with title \"Wispr DIY\" buttons {\"Great\"} default button \"Great\"" >/dev/null 2>&1
