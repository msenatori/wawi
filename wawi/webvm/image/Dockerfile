# Wawi's own little computer (wawi/webvm): a headed Chromium under Xvfb,
# driven by Playwright through a small token-protected HTTP API (server.py),
# and watchable through noVNC (x11vnc + websockify).
#
# Built on the Docker host by `scripts/wawictl webvm build`. NO secret is ever baked in:
# the API token and the VNC password arrive as environment variables at
# `docker run` time (wawi/webvm/lifecycle.py) and live in the Mac's Keychain.
FROM mcr.microsoft.com/playwright/python:v1.57.0-noble

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Xvfb (the screen), x11vnc (to watch it), noVNC + websockify (to watch it
# from a web view), tini (PID 1 that reaps), a couple of fonts so pages in
# Spanish and emoji render.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      xvfb x11vnc novnc websockify tini \
      fonts-dejavu-core fonts-noto-color-emoji \
 && rm -rf /var/lib/apt/lists/*

# The image carries the browsers (/ms-playwright) but not the Python package:
# the same version as the image tag, so they match.
RUN pip install --no-cache-dir --break-system-packages playwright==1.57.0

COPY server.py entrypoint.sh /opt/webvm/
RUN chmod 0755 /opt/webvm/entrypoint.sh /opt/webvm/server.py

# The image ships a non-root user; the browser runs as it.
USER pwuser
WORKDIR /home/pwuser

LABEL wawi.webvm="1"
EXPOSE 8800 6080
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/webvm/entrypoint.sh"]
