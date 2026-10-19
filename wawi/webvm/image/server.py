#!/usr/bin/env python3
"""Control API for Wawi's own little computer (runs INSIDE the container).

A headed Chromium on the Xvfb screen, driven by Playwright's sync API. The
HTTP server is single-threaded on purpose: Playwright's sync objects belong to
the thread that made them, so every request runs on this one thread, one at a
time. Every endpoint needs ``Authorization: Bearer $WEBVM_TOKEN``.

Endpoints (JSON in, JSON out):

  GET  /health                       url, title, tabs, idle seconds
  GET  /info                         url, title, tabs
  POST /open        {url}            navigate the current tab
  GET  /screenshot?quality=          JPEG (base64) + clickable elements
  POST /click       {x, y, double?}  click in screenshot pixels
  POST /type        {text, submit?}  type into the focused element
  POST /key         {name}           press a key (Enter, Tab, PageDown…)
  POST /scroll      {dx, dy}         mouse wheel
  POST /back /forward
  GET  /text?max=                    readable text of the page, truncated
  GET  /element?x=&y=                what sits at (x, y) (safety checks)
  GET  /focused                      the focused element (safety checks)
  POST /new_tab     {url?}           POST /close_tab {index?}

Safety that lives here, whatever the caller asks: only http(s) navigation,
no requests to loopback/private/link-local hosts, downloads refused, JS
dialogs dismissed, and the process exits by itself after
WEBVM_IDLE_EXIT_MIN minutes without a request (the container is --rm).
"""

from __future__ import annotations

import base64
import hmac
import ipaddress
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright

TOKEN = os.environ.pop("WEBVM_TOKEN", "")
PORT = int(os.environ.get("WEBVM_PORT", "8800"))
WIDTH = int(os.environ.get("WEBVM_WIDTH", "1280"))
SCREEN_HEIGHT = int(os.environ.get("WEBVM_SCREEN_HEIGHT", "890"))
LOCALE = os.environ.get("WEBVM_LOCALE", "es-AR")
TIMEZONE = os.environ.get("WEBVM_TIMEZONE", "America/Argentina/Buenos_Aires")
IDLE_EXIT_S = float(os.environ.get("WEBVM_IDLE_EXIT_MIN", "30")) * 60.0
PROFILE_DIR = os.environ.get("WEBVM_PROFILE_DIR", "")
ALLOW_LAN = os.environ.get("WEBVM_ALLOW_LAN", "") == "1"
NAV_TIMEOUT_MS = 30000
MAX_TEXT = 20000

_BLOCKED_NAMES = {"localhost", "host.docker.internal", "gateway.docker.internal",
                  "metadata.google.internal"}


def host_blocked(host: str) -> bool:
    """A host her browser must never reach: this machine, the LAN, link-local."""
    if ALLOW_LAN:
        return False
    host = (host or "").strip("[]").lower().rstrip(".")
    if not host:
        return False
    if host in _BLOCKED_NAMES or host.endswith(".local") or host.endswith(".localhost") \
            or host.endswith(".lan") or host.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved \
        or ip.is_multicast or ip.is_unspecified


def url_allowed(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme in ("about", "data", "blob"):
        return True
    if parts.scheme not in ("http", "https"):
        return False
    return not host_blocked(parts.hostname or "")


# What a page looks like to the caller. Runs inside the page.
_ELEMENTS_JS = r"""
() => {
  const out = [];
  const seen = new Set();
  const sel = 'a[href],button,input,select,textarea,[role=button],[role=link],[role=tab],' +
              '[role=menuitem],[role=option],[role=checkbox],[role=combobox],[onclick],summary,label';
  const vw = window.innerWidth, vh = window.innerHeight;
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    if (r.bottom < 0 || r.right < 0 || r.top > vh || r.left > vw) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') continue;
    const x = Math.round(Math.min(Math.max(r.left + r.width / 2, 0), vw - 1));
    const y = Math.round(Math.min(Math.max(r.top + r.height / 2, 0), vh - 1));
    const key = x + ',' + y;
    if (seen.has(key)) continue;
    seen.add(key);
    let label = (el.getAttribute('aria-label') || el.innerText || el.value ||
                 el.getAttribute('placeholder') || el.getAttribute('title') ||
                 el.getAttribute('alt') || el.getAttribute('name') || '').trim();
    label = label.replace(/\s+/g, ' ').slice(0, 60);
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    out.push({x, y, tag, type, text: label});
    if (out.length >= 60) break;
  }
  return out;
}
"""

_DESCRIBE_JS = r"""
(el) => {
  if (!el || el === document.body || el === document.documentElement) return null;
  const pick = (e) => {
    let n = e;
    for (let i = 0; n && i < 6; i++, n = n.parentElement) {
      const t = n.tagName ? n.tagName.toLowerCase() : '';
      if (['a','button','input','select','textarea','summary','label'].includes(t)) return n;
      const role = n.getAttribute && n.getAttribute('role');
      if (role && ['button','link','menuitem','tab','checkbox','option'].includes(role)) return n;
    }
    return e;
  };
  el = pick(el);
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || (tag === 'button' ? 'submit' : '')).toLowerCase();
  const text = (el.getAttribute('aria-label') || el.innerText || el.value ||
                el.getAttribute('placeholder') || el.getAttribute('title') || '')
               .replace(/\s+/g, ' ').trim().slice(0, 120);
  const form = el.form || el.closest('form');
  let f = null;
  if (form) {
    const inputs = Array.from(form.querySelectorAll('input,textarea,select'));
    const submits = Array.from(form.querySelectorAll('button,input[type=submit]'))
      .map(b => (b.innerText || b.value || b.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim().slice(0, 40))
      .filter(Boolean).slice(0, 6);
    f = {
      action: (form.getAttribute('action') || '').slice(0, 200),
      method: (form.getAttribute('method') || 'get').toLowerCase(),
      has_password: inputs.some(i => (i.type || '').toLowerCase() === 'password'),
      input_types: Array.from(new Set(inputs.map(i => (i.type || i.tagName).toLowerCase()))).slice(0, 20),
      autocomplete: Array.from(new Set(inputs.map(i => (i.getAttribute('autocomplete') || '').toLowerCase())
                                       .filter(Boolean))).slice(0, 20),
      names: Array.from(new Set(inputs.map(i => (i.getAttribute('name') || i.id || '').toLowerCase())
                                .filter(Boolean))).slice(0, 30),
      submits,
    };
  }
  return {
    tag, type, text,
    role: (el.getAttribute('role') || '').toLowerCase(),
    name: (el.getAttribute('name') || '').toLowerCase(),
    id: (el.id || '').toLowerCase(),
    autocomplete: (el.getAttribute('autocomplete') || '').toLowerCase(),
    href: (el.href || '').slice(0, 300),
    editable: !!(el.isContentEditable || ['input','textarea'].includes(tag)),
    form: f,
  };
}
"""

_TEXT_JS = r"""
() => {
  const root = document.querySelector('main, article, [role=main]') || document.body;
  return root ? root.innerText : '';
}
"""


class Browser:
    """The one browser and the tab being driven."""

    def __init__(self) -> None:
        self._pw = sync_playwright().start()
        args = [
            f"--window-size={WIDTH},{SCREEN_HEIGHT}",
            "--window-position=0,0",
            f"--lang={LOCALE}",
            "--disable-infobars",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-dev-shm-usage",
            "--password-store=basic",
        ]
        common = dict(no_viewport=True, locale=LOCALE, timezone_id=TIMEZONE,
                      accept_downloads=False)
        ignore = ["--enable-automation"]
        if PROFILE_DIR:
            self._browser = None
            self.context = self._pw.chromium.launch_persistent_context(
                PROFILE_DIR, headless=False, args=args, ignore_default_args=ignore, **common)
        else:
            self._browser = self._pw.chromium.launch(headless=False, args=args,
                                                     ignore_default_args=ignore)
            self.context = self._browser.new_context(**common)
        self.context.set_default_navigation_timeout(NAV_TIMEOUT_MS)
        self.context.set_default_timeout(10000)
        self.context.route(lambda url: not url_allowed(url), lambda route: route.abort("blockedbyclient"))
        self.context.on("page", self._adopt)
        self.pages: list = []
        for page in self.context.pages:
            self._adopt(page)
        if not self.pages:
            self._adopt(self.context.new_page())
        self.active = self.pages[-1]

    def _adopt(self, page) -> None:
        if page in self.pages:
            return
        page.on("dialog", lambda d: _quietly(d.dismiss))
        page.on("download", lambda d: _quietly(d.cancel))
        page.on("close", lambda _p: self._forget(page))
        self.pages.append(page)
        self.active = page  # a popup the page opened becomes the one she looks at

    def _forget(self, page) -> None:
        if page in self.pages:
            self.pages.remove(page)
        if self.active is page:
            self.active = self.pages[-1] if self.pages else None

    def page(self):
        if self.active is None or self.active.is_closed():
            self.pages = [p for p in self.pages if not p.is_closed()]
            if not self.pages:
                self._adopt(self.context.new_page())
            self.active = self.pages[-1]
        return self.active

    def info(self) -> dict:
        page = self.page()
        try:
            title = page.title()
        except PWError:
            title = ""
        return {"url": page.url, "title": title,
                "tabs": [{"index": i, "url": p.url, "active": p is page}
                         for i, p in enumerate(self.pages)]}

    def settle(self) -> None:
        try:
            self.page().wait_for_load_state("domcontentloaded", timeout=8000)
        except PWError:
            pass
        time.sleep(0.4)


def _quietly(fn) -> None:
    try:
        fn()
    except Exception:
        pass


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _num(body: dict, key: str, default: float | None = None) -> float:
    value = body.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ApiError(400, f"{key} must be a number") from None


class Api:
    def __init__(self, browser: Browser) -> None:
        self.b = browser
        self.started = time.monotonic()
        self.last = time.monotonic()

    # Each handler returns a JSON-able dict.
    def health(self, q: dict, body: dict) -> dict:
        return {"ok": True, "uptime_s": round(time.monotonic() - self.started),
                **self.b.info()}

    def get_info(self, q: dict, body: dict) -> dict:
        return {"ok": True, **self.b.info()}

    def open(self, q: dict, body: dict) -> dict:
        url = str(body.get("url", "")).strip()
        if not url:
            raise ApiError(400, "url is required")
        if "://" not in url:
            url = "https://" + url
        if not url_allowed(url):
            raise ApiError(403, "url not allowed")
        page = self.b.page()
        status = None
        try:
            resp = page.goto(url, wait_until="domcontentloaded")
            status = resp.status if resp is not None else None
        except PWError as exc:
            raise ApiError(502, f"navigation failed: {str(exc).splitlines()[0][:200]}") from None
        self.b.settle()
        return {"ok": True, "status": status, **self.b.info()}

    def screenshot(self, q: dict, body: dict) -> dict:
        quality = int(q.get("quality", ["65"])[0])
        quality = max(30, min(quality, 90))
        page = self.b.page()
        raw = page.screenshot(type="jpeg", quality=quality, timeout=15000)
        size = page.evaluate("() => [window.innerWidth, window.innerHeight]")
        try:
            elements = page.evaluate(_ELEMENTS_JS)
        except PWError:
            elements = []
        return {"ok": True, "image_b64": base64.b64encode(raw).decode("ascii"),
                "width": size[0], "height": size[1], "elements": elements, **self.b.info()}

    def click(self, q: dict, body: dict) -> dict:
        x, y = _num(body, "x"), _num(body, "y")
        page = self.b.page()
        page.mouse.click(x, y, click_count=2 if body.get("double") else 1)
        self.b.settle()
        return {"ok": True, **self.b.info()}

    def type(self, q: dict, body: dict) -> dict:
        text = str(body.get("text", ""))
        if len(text) > 2000:
            raise ApiError(400, "text too long")
        page = self.b.page()
        page.keyboard.type(text, delay=15)
        if body.get("submit"):
            page.keyboard.press("Enter")
            self.b.settle()
        return {"ok": True, **self.b.info()}

    def key(self, q: dict, body: dict) -> dict:
        name = str(body.get("name", "")).strip()
        if not name or len(name) > 40:
            raise ApiError(400, "name is required")
        try:
            self.b.page().keyboard.press(name)
        except PWError as exc:
            raise ApiError(400, f"bad key: {str(exc).splitlines()[0][:120]}") from None
        self.b.settle()
        return {"ok": True, **self.b.info()}

    def scroll(self, q: dict, body: dict) -> dict:
        page = self.b.page()
        page.mouse.move(WIDTH / 2, 400)
        page.mouse.wheel(_num(body, "dx", 0), _num(body, "dy", 0))
        time.sleep(0.35)
        return {"ok": True, **self.b.info()}

    def back(self, q: dict, body: dict) -> dict:
        _quietly(lambda: self.b.page().go_back(wait_until="domcontentloaded"))
        self.b.settle()
        return {"ok": True, **self.b.info()}

    def forward(self, q: dict, body: dict) -> dict:
        _quietly(lambda: self.b.page().go_forward(wait_until="domcontentloaded"))
        self.b.settle()
        return {"ok": True, **self.b.info()}

    def text(self, q: dict, body: dict) -> dict:
        limit = max(200, min(int(q.get("max", ["6000"])[0]), MAX_TEXT))
        try:
            raw = self.b.page().evaluate(_TEXT_JS) or ""
        except PWError:
            raw = ""
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        clean = "\n".join(line for line in lines if line)
        return {"ok": True, "text": clean[:limit], "truncated": len(clean) > limit,
                "chars": len(clean), **self.b.info()}

    def element(self, q: dict, body: dict) -> dict:
        x = float(q.get("x", ["-1"])[0])
        y = float(q.get("y", ["-1"])[0])
        page = self.b.page()
        handle = page.evaluate_handle("([x, y]) => document.elementFromPoint(x, y)", [x, y])
        try:
            info = page.evaluate(_DESCRIBE_JS, handle)
        finally:
            _quietly(handle.dispose)
        return {"ok": True, "element": info, **self.b.info()}

    def focused(self, q: dict, body: dict) -> dict:
        page = self.b.page()
        handle = page.evaluate_handle("() => document.activeElement")
        try:
            info = page.evaluate(_DESCRIBE_JS, handle)
        finally:
            _quietly(handle.dispose)
        return {"ok": True, "element": info, **self.b.info()}

    def new_tab(self, q: dict, body: dict) -> dict:
        page = self.b.context.new_page()
        self.b._adopt(page)
        if body.get("url"):
            return self.open(q, body)
        return {"ok": True, **self.b.info()}

    def close_tab(self, q: dict, body: dict) -> dict:
        pages = self.b.pages
        index = body.get("index")
        page = pages[int(index)] if index is not None and 0 <= int(index) < len(pages) else self.b.page()
        if len(pages) <= 1:
            page.goto("about:blank")
        else:
            page.close()
        return {"ok": True, **self.b.info()}


ROUTES = {
    ("GET", "/health"): "health",
    ("GET", "/info"): "get_info",
    ("POST", "/open"): "open",
    ("GET", "/screenshot"): "screenshot",
    ("POST", "/click"): "click",
    ("POST", "/type"): "type",
    ("POST", "/key"): "key",
    ("POST", "/scroll"): "scroll",
    ("POST", "/back"): "back",
    ("POST", "/forward"): "forward",
    ("GET", "/text"): "text",
    ("GET", "/element"): "element",
    ("GET", "/focused"): "focused",
    ("POST", "/new_tab"): "new_tab",
    ("POST", "/close_tab"): "close_tab",
}


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        server_version = "wawi-webvm/1"

        def log_message(self, fmt, *args):  # path only; never headers
            sys.stderr.write("%s %s\n" % (self.command, self.path.split("?")[0]))

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, method: str) -> None:
            parts = urlsplit(self.path)
            name = ROUTES.get((method, parts.path))
            if name is None:
                return self._send(404, {"ok": False, "error": "not found"})
            auth = self.headers.get("Authorization", "")
            if not TOKEN or not hmac.compare_digest(auth.encode(), f"Bearer {TOKEN}".encode()):
                return self._send(401, {"ok": False, "error": "unauthorized"})
            body: dict = {}
            if method == "POST":
                length = min(int(self.headers.get("Content-Length") or 0), 64 * 1024)
                if length:
                    try:
                        body = json.loads(self.rfile.read(length) or b"{}")
                    except ValueError:
                        return self._send(400, {"ok": False, "error": "bad json"})
                    if not isinstance(body, dict):
                        return self._send(400, {"ok": False, "error": "bad json"})
            api.last = time.monotonic()
            try:
                result = getattr(api, name)(parse_qs(parts.query), body)
            except ApiError as exc:
                return self._send(exc.status, {"ok": False, "error": str(exc)})
            except PWError as exc:
                return self._send(500, {"ok": False, "error": str(exc).splitlines()[0][:200]})
            except Exception as exc:  # noqa: BLE001 - one request must not kill the server
                return self._send(500, {"ok": False, "error": type(exc).__name__})
            api.last = time.monotonic()
            self._send(200, result)

        def do_GET(self):  # noqa: N802
            self._handle("GET")

        def do_POST(self):  # noqa: N802
            self._handle("POST")

    return Handler


def _idle_guard(api: Api) -> None:
    while True:
        time.sleep(20)
        if time.monotonic() - api.last > IDLE_EXIT_S:
            sys.stderr.write("idle for too long; exiting\n")
            os._exit(0)


def main() -> None:
    if not TOKEN:
        sys.stderr.write("WEBVM_TOKEN missing\n")
        sys.exit(2)
    browser = Browser()
    api = Api(browser)
    threading.Thread(target=_idle_guard, args=(api,), daemon=True).start()
    server = HTTPServer(("0.0.0.0", PORT), make_handler(api))
    server.timeout = 60
    sys.stderr.write(f"wawi-webvm listening on {PORT}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
