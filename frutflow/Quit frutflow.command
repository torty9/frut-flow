#!/bin/bash
# Double-click to FULLY stop frutflow.
# The watchdog would relaunch the app within ~20s, so we must disable the
# auto-start/watchdog FIRST, then kill the running app. frutflow will stay
# stopped until you log in again (or run the re-enable line below).
launchctl bootout "gui/$(id -u)/com.wisprdiy.dictation" 2>/dev/null
pkill -f "flow\.py" 2>/dev/null
echo "frutflow stopped — and it will NOT auto-restart."
echo
echo "To turn frutflow back on now (without logging out):"
echo "  launchctl bootstrap gui/\$(id -u) \"\$HOME/Library/LaunchAgents/com.wisprdiy.dictation.plist\""
echo "…or just log in again, or double-click frutflow.app in ~/Applications."
