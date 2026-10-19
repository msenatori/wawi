"""HTTP client for the control API inside her little computer
(wawi/webvm/image/server.py). Stdlib only; the transport is injectable so the
tests run without a network.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class WebVMError(RuntimeError):
    """Something about her computer failed. The message is Spanish and safe to
    say out loud: it never carries a secret."""


class Unreachable(WebVMError):
    """Nobody answers at the API address (container down, its server off)."""


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes


#: ``transport(method, url, headers, data, timeout) -> Response``.
Transport = Callable[[str, str, dict, "bytes | None", float], Response]


def _urllib_transport(method: str, url: str, headers: dict, data: bytes | None,
                      timeout: float) -> Response:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - LAN API
            return Response(resp.status, resp.read())
    except urllib.error.HTTPError as exc:
        return Response(exc.code, exc.read() or b"")
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as exc:
        raise Unreachable(f"No llego a mi compu ({getattr(exc, 'reason', exc)}).") from None


class Client:
    """One method per endpoint; each returns the decoded JSON dict."""

    def __init__(self, base_url: str, token: str, transport: Transport = _urllib_transport,
                 timeout: float = 45.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._transport = transport
        self.timeout = timeout

    def __repr__(self) -> str:  # never the token
        return f"Client({self.base_url!r})"

    def _call(self, method: str, path: str, body: dict | None = None,
              query: dict | None = None, timeout: float | None = None) -> dict[str, Any]:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        data = None
        if method == "POST":
            data = json.dumps(body or {}).encode("utf-8")
            headers["Content-Type"] = "application/json"
        resp = self._transport(method, url, headers, data, timeout or self.timeout)
        try:
            payload = json.loads(resp.body.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if resp.status == 401:
            raise WebVMError("Mi compu no aceptó la clave (token).")
        if resp.status == 403:
            raise WebVMError("Esa dirección no la puedo abrir desde mi compu.")
        if resp.status >= 400:
            detail = str(payload.get("error") or f"HTTP {resp.status}")[:160]
            raise WebVMError(f"Mi compu respondió con un error: {detail}")
        return payload

    # -- endpoints ------------------------------------------------------------
    def health(self, timeout: float = 4.0) -> dict:
        return self._call("GET", "/health", timeout=timeout)

    def info(self) -> dict:
        return self._call("GET", "/info")

    def open(self, url: str) -> dict:
        return self._call("POST", "/open", {"url": url}, timeout=60.0)

    def screenshot(self, quality: int = 65) -> dict:
        return self._call("GET", "/screenshot", query={"quality": int(quality)})

    def click(self, x: float, y: float, double: bool = False) -> dict:
        return self._call("POST", "/click", {"x": x, "y": y, "double": bool(double)})

    def type(self, text: str, submit: bool = False) -> dict:
        return self._call("POST", "/type", {"text": text, "submit": bool(submit)})

    def key(self, name: str) -> dict:
        return self._call("POST", "/key", {"name": name})

    def scroll(self, dx: float = 0, dy: float = 0) -> dict:
        return self._call("POST", "/scroll", {"dx": dx, "dy": dy})

    def back(self) -> dict:
        return self._call("POST", "/back")

    def forward(self) -> dict:
        return self._call("POST", "/forward")

    def text(self, max_chars: int = 6000) -> dict:
        return self._call("GET", "/text", query={"max": int(max_chars)})

    def element_at(self, x: float, y: float) -> dict:
        return self._call("GET", "/element", query={"x": x, "y": y})

    def focused(self) -> dict:
        return self._call("GET", "/focused")

    def new_tab(self, url: str = "") -> dict:
        return self._call("POST", "/new_tab", {"url": url} if url else {})

    def close_tab(self, index: int | None = None) -> dict:
        return self._call("POST", "/close_tab", {} if index is None else {"index": index})
