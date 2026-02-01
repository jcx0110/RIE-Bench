#!/usr/bin/env bash
# Run ALFRED evaluation with a virtual display (GLX) to avoid:
#   X Error: BadValue ... Major opcode 152 (GLX) Minor opcode 3 (X_GLXCreateContext)
# Use on headless servers when no physical display is available.

set -e
cd "$(dirname "$0")/.."

# Force Mesa software GL so GLX context works on Xvfb (Unity/THOR often fails with default)
export MESA_GL_VERSION_OVERRIDE=3.3
export __GLX_VENDOR_LIBRARY_NAME=mesa

# Prefer xvfb-run: auto-picks free display and passes GLX args
if command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +iglx +extension RENDER" python scripts/evaluate.py "$@"
fi

# Fallback: start our own Xvfb on :99 (avoid :1 - often taken)
DISPLAY_NUM="${DISPLAY_NUM:-99}"
LOCK="/tmp/.X${DISPLAY_NUM}-lock"
if [ -f "$LOCK" ]; then
  OLD_PID=$(cat "$LOCK" 2>/dev/null)
  if ! kill -0 "$OLD_PID" 2>/dev/null; then
    rm -f "$LOCK"
  fi
fi
if [ ! -f "$LOCK" ] || ! kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  Xvfb :${DISPLAY_NUM} +extension GLX +iglx +extension RENDER -screen 0 1024x768x24 &
  sleep 2
fi
export DISPLAY=:${DISPLAY_NUM}
exec python scripts/evaluate.py "$@"
