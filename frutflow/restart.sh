#!/usr/bin/env bash
# Restart Wispr DIY (use after changing ~/.flowdictate/config.json).
# Stops the running instance and relaunches it as frutflow.app (menu-bar app).
pkill -f "flow.py" 2>/dev/null
sleep 1
open -g "/Users/thorstenpfeiffer/Applications/frutflow.app"
sleep 6
echo "Wispr DIY restarted (menu-bar app). Recent log:"
grep -v "\[flow\]\[debug\]" "$HOME/.flowdictate/flow.log" 2>/dev/null | tail -6
