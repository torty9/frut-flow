#!/usr/bin/env bash
# Lean runner used by the login agent and for manual restarts.
# (No reinstall — assumes ./run.sh has already set up .venv once.)
cd "$(dirname "$0")"
source .venv/bin/activate
exec python flow.py
