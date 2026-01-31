#!/usr/bin/env bash
# Run ALFRED evaluation with Xvfb (virtual display + GLX) to avoid:
#   X Error: BadValue (integer parameter out of range) Major opcode 152 (GLX) Minor opcode 3 (X_GLXCreateContext)
# Use this on headless servers when startx.py (NVIDIA) is not available or fails.

set -e
DISPLAY_NUM="${DISPLAY_NUM:-1}"
# Start Xvfb with GLX support (required by AI2-THOR/Unity)
if ! kill -0 $(cat /tmp/.X${DISPLAY_NUM}-lock 2>/dev/null) 2>/dev/null; then
  Xvfb :${DISPLAY_NUM} +extension GLX +iglx -screen 0 1024x768x24 &
  XVFB_PID=$!
  sleep 2
fi
export DISPLAY=:${DISPLAY_NUM}
# Override config so the evaluator uses this display
export ALFRED_X_DISPLAY="${DISPLAY_NUM}"
cd "$(dirname "$0")/.."
python scripts/evaluate.py "$@"
