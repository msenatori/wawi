"""Tiny stdlib HTTP helper shared by the network sensors.

stdlib urllib on purpose: three sensors need "GET with a timeout and a size
cap", which is not worth a dependency. Tests replace the sensors' ``fetch``
callables, so nothing here is ever called without a network.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

USER_AGENT = "Wawi/1.0 (desktop pet status check)"

#: Bodies are capped: a status check must not download a video by accident,
#: and a feed is never legitimately larger than this.
MAX_BODY = 2_000_000


@dataclass(frozen=True)
class Response:
    status: int | None  # None = never got an HTTP answer
    latency_ms: int
    body: bytes = b""
    error: str = ""


def fetch(url: str, *, timeout: float = 10.0, method: str = "GET", max_bytes: int = MAX_BODY) -> Response:
    """Never raises: every failure becomes a Response with *error* set."""
    start = time.monotonic()
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user-configured URL
            body = resp.read(max_bytes) if method != "HEAD" else b""
            return Response(resp.status, _ms(start), body)
    except urllib.error.HTTPError as exc:
        # 4xx/5xx still mean the server answered; the caller decides.
        return Response(exc.code, _ms(start), b"", f"HTTP {exc.code}")
    except Exception as exc:  # URLError, timeout, ssl, connection reset...
        reason = getattr(exc, "reason", None) or exc
        return Response(None, _ms(start), b"", str(reason)[:200] or type(exc).__name__)


def fetch_json(url: str, *, timeout: float = 10.0) -> Any:
    """GET and decode JSON; raises on any failure so the runner backs off."""
    resp = fetch(url, timeout=timeout)
    if resp.status != 200:
        raise OSError(f"{url}: {resp.error or resp.status}")
    return json.loads(resp.body.decode("utf-8"))


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
