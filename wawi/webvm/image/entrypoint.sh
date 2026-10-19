#!/usr/bin/env bash
# Start the screen, the VNC server, the noVNC bridge, then the control API.
# Secrets come in as env vars and are never printed: the VNC password goes to
# a 0600 file x11vnc reads AND deletes (rm:), then leaves the environment.
set -euo pipefail

: "${WEBVM_TOKEN:?WEBVM_TOKEN is required}"
: "${WEBVM_VNC_PASSWORD:?WEBVM_VNC_PASSWORD is required}"

W="${WEBVM_WIDTH:-1280}"
H="${WEBVM_SCREEN_HEIGHT:-890}"
export DISPLAY=:99

Xvfb :99 -screen 0 "${W}x${H}x24" -nolisten tcp -dpi 96 >/tmp/xvfb.log 2>&1 &
for _ in $(seq 1 50); do
  [ -e /tmp/.X11-unix/X99 ] && break
  sleep 0.1
done

umask 077
PWFILE="$(mktemp /tmp/vncpw.XXXXXX)"
printf '%s\n' "$WEBVM_VNC_PASSWORD" >"$PWFILE"
unset WEBVM_VNC_PASSWORD
x11vnc -display :99 -rfbport 5900 -localhost -forever -shared -noxdamage \
       -passwdfile "rm:$PWFILE" -quiet >/tmp/x11vnc.log 2>&1 &

websockify --web /usr/share/novnc 0.0.0.0:6080 localhost:5900 >/tmp/websockify.log 2>&1 &

exec python3 /opt/webvm/server.py
