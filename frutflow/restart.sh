#!/usr/bin/env bash
# Restart früt Flow (use after changing ~/.flowdictate/config.json).
# Stops the running instance and relaunches it as frutflow.app (menu-bar app).
pkill -f "flow.py" 2>/dev/null
sleep 1
open -g "$HOME/Applications/frutflow.app"
sleep 6
echo "früt Flow restarted (menu-bar app). Recent log:"
grep -v "\[flow\]\[debug\]" "$HOME/.flowdictate/flow.log" 2>/dev/null | tail -6
