#!/bin/bash
# Double-click this file to start Wispr DIY. Keep the Terminal window it opens
# running while you dictate. Close the window (or press Ctrl-C) to stop.
cd "/Users/thorstenpfeiffer/diy wispr/frutflow" || exit 1
source .venv/bin/activate
clear
echo "Starting Wispr DIY — loading the speech model (a few seconds)…"
echo "Keep this window open. Hold the Right Option key to dictate."
echo
# force arm64 so the native speech wheels load; mirror output to the log so
# status (permissions, transcripts) can be reviewed after the fact.
exec arch -arm64 python flow.py 2>&1 | tee "$HOME/.flowdictate/flow.log"
