#!/bin/bash
# Double-click this file to start früt Flow. Keep the Terminal window it opens
# running while you dictate. Close the window (or press Ctrl-C) to stop.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
clear
echo "Starting früt Flow — loading the speech model (a few seconds)…"
echo "Keep this window open. Hold the Right Option key to dictate."
echo
# Create the log owner-only (0600): it can contain dictated text, so no other
# local user should be able to read it.
umask 077
mkdir -p "$HOME/.flowdictate"
# force arm64 so the native speech wheels load; mirror output to the log so
# status (permissions) can be reviewed after the fact.
exec arch -arm64 python flow.py 2>&1 | tee "$HOME/.flowdictate/flow.log"
