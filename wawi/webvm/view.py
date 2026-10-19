"""The windows onto her little computer: a small one beside her while she
browses, and a big one on demand where the owner can watch or take over.

* MINI: a borderless, non-activating card in her paper-and-crayon style
  (like the notes STOP card, wawi/notes_ui.py), placed next to the mascot and
  following her, with the live screen (noVNC in a WKWebView, view-only,
  scaled) and two buttons: ⤢ (expand) and ✕ (hide). It fades in and out.
* BIG: an ordinary resizable window (~1000x650) with the same screen and a
  «Tomar el control» switch: off, it only watches; on, the owner's mouse and
  keyboard drive her browser (for a login, a captcha, a password).

:meth:`WebVMView.tick` runs on the panel's render timer (main thread) and is
the only place AppKit is touched. Placement, fading and the viewer page are
pure functions below, tested without a window.

The VNC password only ever lives in memory: it goes into the page as a JS
string (never in a URL, so it is in no server log), and the page is loaded
from a string with the noVNC server as its base URL.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

log = logging.getLogger(__name__)

#: The live screen in the small card (points); 1280x800 scaled down.
SCREEN_W = 320.0
SCREEN_H = 200.0
PAD = 6.0
HEADER = 22.0
#: The card.
WIDTH = SCREEN_W + 2 * PAD
HEIGHT = SCREEN_H + HEADER + 2 * PAD
#: Gap to her window.
GAP = 8.0
#: The big window's preferred size.
BIG_W = 1000.0
BIG_H = 650.0
#: Fade in/out duration (seconds).
FADE_S = 0.35
#: Hidden this long, the small view drops its VNC connection.
UNLOAD_AFTER_S = 20.0
#: The header buttons (flipped coordinates of the card: x from the left,
#: y from the top): (x, y, w, h).
EXPAND_BUTTON = (WIDTH - 2 * 22.0 - PAD, 2.0, 20.0, 20.0)
CLOSE_BUTTON = (WIDTH - 22.0 - PAD + 2.0, 2.0, 20.0, 20.0)

Rect = tuple[float, float, float, float]


def place_mini(anchor: Rect | None, visible: Rect | None,
               size: tuple[float, float] = (WIDTH, HEIGHT)) -> tuple[float, float]:
    """Bottom-left origin of the small card (y up): to her right, its top at
    her top, flipped to her left when it would leave the screen, and always
    inside *visible*. No anchor: the screen's bottom-right corner."""
    w, h = size
    if visible is None:
        visible = (0.0, 0.0, 1440.0, 900.0)
    vx, vy, vw, vh = visible
    if anchor is None:
        x, y = vx + vw - w - 24.0, vy + 24.0
    else:
        ax, ay, aw, ah = anchor
        x = ax + aw + GAP
        if x + w > vx + vw - 4.0:
            x = ax - GAP - w
        y = ay + ah - h
    x = min(max(x, vx + 4.0), vx + vw - w - 4.0)
    y = min(max(y, vy + 4.0), vy + vh - h - 4.0)
    return x, y


def place_big(visible: Rect | None) -> Rect:
    """The big window's frame: BIG_W x BIG_H (smaller on a small screen),
    centred on *visible*."""
    if visible is None:
        visible = (0.0, 0.0, 1440.0, 900.0)
    vx, vy, vw, vh = visible
    w = min(BIG_W, max(400.0, vw - 40.0))
    h = min(BIG_H, max(300.0, vh - 40.0))
    return vx + (vw - w) / 2.0, vy + (vh - h) / 2.0, w, h


def hit(x: float, y: float) -> str | None:
    """``"expand"``, ``"close"`` or None for a click at (x, y) in the card's
    flipped coordinates."""
    for name, (bx, by, bw, bh) in (("expand", EXPAND_BUTTON), ("close", CLOSE_BUTTON)):
        if bx - 3 <= x <= bx + bw + 3 and by - 3 <= y <= by + bh + 3:
            return name
    return None


def fade(alpha: float, target: float, dt: float, duration: float = FADE_S) -> float:
    """*alpha* one step of *dt* seconds closer to *target* (0..1)."""
    if duration <= 0:
        return target
    step = max(0.0, dt) / duration
    if alpha < target:
        return min(target, alpha + step)
    return max(target, alpha - step)


def _js(value: Any) -> str:
    """*value* as a JS literal that cannot close the <script> it sits in."""
    return (json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))


def viewer_html(host: str, port: int, password: str, *, view_only: bool, big: bool) -> str:
    """The page that shows her screen: noVNC's RFB client, loaded with the
    noVNC server as base URL. Values are JSON-encoded into the script."""
    toolbar = ""
    if big:
        toolbar = (
            '<div id="bar"><span id="state">Conectando…</span>'
            '<label><input type="checkbox" id="control"> Tomar el control</label></div>'
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
html,body{{margin:0;height:100%;background:#1d232a;overflow:hidden;
  font:13px "Chalkboard SE",-apple-system,sans-serif;color:#e8eadf}}
#screen{{position:absolute;left:0;right:0;bottom:0;top:{30 if big else 0}px}}
#bar{{height:30px;display:flex;align-items:center;gap:16px;padding:0 12px;
  background:#2a3440;border-bottom:1px solid #3d4a57}}
#bar label{{cursor:pointer}}
#wait{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  color:#9aa5a0;font-size:{14 if big else 11}px;pointer-events:none}}
</style></head>
<body>{toolbar}<div id="screen"></div><div id="wait">Prendiendo mi compu…</div>
<script type="module">
import RFB from './core/rfb.js';
const url = 'ws://' + {_js(host)} + ':' + {int(port)} + '/websockify';
const password = {_js(password)};
let rfb = null;
function setState(t) {{
  const s = document.getElementById('state'); if (s) s.textContent = t;
  document.title = t;
}}
function connect() {{
  rfb = new RFB(document.getElementById('screen'), url, {{credentials: {{password}}}});
  rfb.viewOnly = {_js(bool(view_only))};
  rfb.scaleViewport = true;
  rfb.resizeSession = false;
  rfb.background = '#1d232a';
  rfb.addEventListener('connect', () => {{
    document.getElementById('wait').style.display = 'none';
    setState(rfb.viewOnly ? 'Mirando' : 'Con el control');
  }});
  rfb.addEventListener('disconnect', () => {{
    document.getElementById('wait').style.display = 'flex';
    setState('Desconectada');
    setTimeout(connect, 2500);
  }});
  rfb.addEventListener('credentialsrequired', () => rfb.sendCredentials({{password}}));
}}
const box = document.getElementById('control');
if (box) box.addEventListener('change', () => {{
  if (rfb) {{ rfb.viewOnly = !box.checked; setState(box.checked ? 'Con el control' : 'Mirando');
             if (box.checked) rfb.focus(); }}
}});
connect();
</script></body></html>"""


try:  # AppKit, only in the real app ------------------------------------------------
    import objc  # type: ignore[import]
    from AppKit import (  # type: ignore[import]
        NSApp,
        NSAppearanceNameDarkAqua,
        NSAttributedString,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSFloatingWindowLevel,
        NSMakePoint,
        NSMakeRect,
        NSPanel,
        NSScreen,
        NSView,
        NSWindow,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskMiniaturizable,
        NSWindowStyleMaskNonactivatingPanel,
        NSWindowStyleMaskResizable,
        NSWindowStyleMaskTitled,
    )
    from Foundation import NSURL, NSObject  # type: ignore[import]

    _HAS_APPKIT = True
except Exception:  # pragma: no cover - not macOS / no PyObjC
    _HAS_APPKIT = False


if _HAS_APPKIT:

    class WawiWebVMPanel(NSPanel):  # type: ignore[misc]
        """Borderless, non-activating: a click never steals focus."""

        def canBecomeKeyWindow(self):  # noqa: N802
            return False

        def canBecomeMainWindow(self):  # noqa: N802
            return False

    class WawiWebVMCard(NSView):  # type: ignore[misc]
        """The paper card behind the screen, with its two header buttons."""

        def initWithFrame_host_(self, frame, host):  # noqa: N802
            self = objc.super(WawiWebVMCard, self).initWithFrame_(frame)
            if self is None:
                return None
            self._host = host
            return self

        def isFlipped(self):  # noqa: N802
            return True

        def isOpaque(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def mouseDown_(self, event):  # noqa: N802
            try:
                p = self.convertPoint_fromView_(event.locationInWindow(), None)
                what = hit(float(p.x), float(p.y))
                if what:
                    self._host._clicked(what)
            except Exception:
                log.debug("webvm: card click failed", exc_info=True)

        def drawRect_(self, _rect):  # noqa: N802
            try:
                self._host._paint()
            except Exception:
                log.debug("webvm: card drawing failed", exc_info=True)

    class WawiWebVMWindowDelegate(NSObject):  # type: ignore[misc]
        def initWithHost_(self, host):  # noqa: N802
            self = objc.super(WawiWebVMWindowDelegate, self).init()
            if self is None:
                return None
            self._host = host
            return self

        def windowWillClose_(self, _note):  # noqa: N802
            try:
                self._host._big_closed()
            except Exception:
                log.debug("webvm: big window close failed", exc_info=True)


def _web_view(frame: Any) -> Any:
    from WebKit import WKWebView, WKWebViewConfiguration, WKWebsiteDataStore  # type: ignore[import]

    config = WKWebViewConfiguration.alloc().init()
    # Nothing of the session survives: no cookies, no cache on disk.
    config.setWebsiteDataStore_(WKWebsiteDataStore.nonPersistentDataStore())
    web = WKWebView.alloc().initWithFrame_configuration_(frame, config)
    try:
        web.setValue_forKey_(False, "drawsBackground")
    except Exception:
        pass
    return web


class WebVMView:
    """Owns both windows. One per panel; main thread only."""

    def __init__(self) -> None:
        self.window: Any = None
        self.card: Any = None
        self.web: Any = None
        self.big: Any = None
        self.big_web: Any = None
        self._delegate: Any = None
        self._alpha = 0.0
        self._last_tick = time.monotonic()
        self._hidden_since: float | None = None
        self._loaded = False
        self._origin: tuple[float, float] | None = None
        self._dark = False
        self._service: Any = None

    # -- main thread ------------------------------------------------------
    def tick(self, service: Any, anchor_window: Any = None) -> None:
        """Show, move, fade or hide the windows for *service*. Never raises."""
        if not _HAS_APPKIT:
            return
        try:
            self._tick(service, anchor_window)
        except Exception:
            log.debug("webvm: view tick failed", exc_info=True)

    def _tick(self, service: Any, anchor_window: Any) -> None:
        now = time.monotonic()
        dt, self._last_tick = now - self._last_tick, now
        self._service = service
        if service is not None:
            service.poll()
        running = service is not None and service.running
        if service is not None and service.expand_requested and running:
            service.expand_requested = False
            self.open_big()
        if self.big is not None and not running:
            self.big.close()  # its delegate clears self.big
        wanted = running and bool(service.view_wanted) and self.big is None
        if self.window is None and not wanted:
            return
        if self.window is None:
            self._build()
        target = 1.0 if wanted else 0.0
        if wanted and not self._loaded:
            self._load(service)
        if wanted:
            origin = place_mini(*self._geometry(anchor_window))
            if origin != self._origin:
                self._origin = origin
                self.window.setFrameOrigin_(NSMakePoint(*origin))
            if not self.window.isVisible():
                self._dark = self._read_dark()
                self.card.setNeedsDisplay_(True)
                self.window.setAlphaValue_(self._alpha)
                self.window.orderFrontRegardless()
            self._hidden_since = None
        alpha = fade(self._alpha, target, dt)
        if alpha != self._alpha:
            self._alpha = alpha
            self.window.setAlphaValue_(alpha)
        if alpha <= 0.0 and self.window.isVisible():
            self.window.orderOut_(None)
            self._hidden_since = now
        if not wanted and self._loaded and self._hidden_since is not None and \
                now - self._hidden_since > UNLOAD_AFTER_S:
            self._unload()

    def open_big(self) -> None:
        """The big window (or bring it to front). Main thread."""
        service = self._service
        if service is None:
            return
        if self.big is None:
            screen = NSScreen.mainScreen()
            visible = None
            if screen is not None:
                v = screen.visibleFrame()
                visible = (v.origin.x, v.origin.y, v.size.width, v.size.height)
            x, y, w, h = place_big(visible)
            style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                     | NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable)
            big = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(x, y, w, h), style, NSBackingStoreBuffered, False)
            big.setTitle_("La compu de Wawi")
            big.setReleasedWhenClosed_(False)
            big.setMinSize_((420, 300))
            self._delegate = WawiWebVMWindowDelegate.alloc().initWithHost_(self)
            big.setDelegate_(self._delegate)
            web = _web_view(NSMakeRect(0, 0, w, h))
            web.setAutoresizingMask_(2 | 16)  # width + height sizable
            big.setContentView_(web)
            settings = service.settings
            web.loadHTMLString_baseURL_(
                viewer_html(settings.host, settings.view_port, service.view_password(),
                            view_only=True, big=True),
                NSURL.URLWithString_(settings.view_url))
            self.big, self.big_web = big, web
        try:
            NSApp().activateIgnoringOtherApps_(True)
        except Exception:
            pass
        self.big.makeKeyAndOrderFront_(None)

    def close(self) -> None:
        for window in (self.big, self.window):
            if window is not None:
                try:
                    window.orderOut_(None)
                    window.close()
                except Exception:
                    log.debug("webvm: closing a window failed", exc_info=True)
        self.window = self.card = self.web = self.big = self.big_web = None
        self._loaded = False

    # -- internals --------------------------------------------------------
    def _big_closed(self) -> None:
        web = self.big_web
        if web is not None:
            try:
                web.loadHTMLString_baseURL_("", None)  # drop the VNC connection
            except Exception:
                pass
        self.big = self.big_web = None

    def _build(self) -> None:
        rect = NSMakeRect(0, 0, WIDTH, HEIGHT)
        window = WawiWebVMPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        window.setLevel_(NSFloatingWindowLevel)
        window.setOpaque_(False)
        from wawi.games.ui import _rgba

        window.setBackgroundColor_(_rgba((0, 0, 0, 0)))
        window.setHasShadow_(True)
        window.setHidesOnDeactivate_(False)
        window.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                      | NSWindowCollectionBehaviorFullScreenAuxiliary)
        card = WawiWebVMCard.alloc().initWithFrame_host_(rect, self)
        window.setContentView_(card)
        # The card is flipped: the screen sits under the header.
        web = _web_view(NSMakeRect(PAD, HEADER + PAD, SCREEN_W, SCREEN_H))
        card.addSubview_(web)
        window.setAlphaValue_(0.0)
        self.window, self.card, self.web = window, card, web

    def _load(self, service: Any) -> None:
        settings = service.settings
        try:
            password = service.view_password()
        except Exception:
            log.warning("webvm: no VNC password; the view stays empty", exc_info=True)
            return
        self.web.loadHTMLString_baseURL_(
            viewer_html(settings.host, settings.view_port, password, view_only=True, big=False),
            NSURL.URLWithString_(settings.view_url))
        self._loaded = True

    def _unload(self) -> None:
        if self.web is not None:
            self.web.loadHTMLString_baseURL_("", None)
        self._loaded = False

    def _clicked(self, what: str) -> None:
        service = self._service
        if service is None:
            return
        if what == "expand":
            service.expand()
        elif what == "close":
            service.show(False, by_owner=True)

    @staticmethod
    def _geometry(anchor_window: Any) -> tuple[Rect | None, Rect | None]:
        screen = None
        anchor = None
        if anchor_window is not None:
            try:
                if anchor_window.isVisible():
                    f = anchor_window.frame()
                    anchor = (f.origin.x, f.origin.y, f.size.width, f.size.height)
                    screen = anchor_window.screen()
            except Exception:
                anchor = None
        screen = screen or NSScreen.mainScreen()
        visible = None
        if screen is not None:
            v = screen.visibleFrame()
            visible = (v.origin.x, v.origin.y, v.size.width, v.size.height)
        return anchor, visible

    @staticmethod
    def _read_dark() -> bool:
        try:
            app = NSApp()
            appearance = app.effectiveAppearance() if app is not None else None
            return bool(appearance and appearance.bestMatchFromAppearancesWithNames_(
                [NSAppearanceNameDarkAqua]) == NSAppearanceNameDarkAqua)
        except Exception:
            return False

    def _paint(self) -> None:
        from wawi.games.ui import _attrs, _rgba

        paper = (0.13, 0.19, 0.24, .98) if self._dark else (1.0, .985, .94, .98)
        ink = (.88, .92, .83, 1) if self._dark else (.16, .20, .42, 1)
        card = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(1, 1, WIDTH - 2, HEIGHT - 2), 12, 12)
        _rgba(paper).setFill()
        card.fill()
        _rgba(ink).setStroke()
        card.setLineWidth_(1.6)
        card.stroke()
        title = NSAttributedString.alloc().initWithString_attributes_("Mi compu", _attrs(12.5, ink))
        title.drawAtPoint_(NSMakePoint(PAD + 4, 3))
        for glyph, (bx, by, bw, bh) in (("⤢", EXPAND_BUTTON), ("✕", CLOSE_BUTTON)):
            pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(bx, by, bw, bh), 6, 6)
            _rgba(ink).setStroke()
            pill.setLineWidth_(1.1)
            pill.stroke()
            mark = NSAttributedString.alloc().initWithString_attributes_(glyph, _attrs(12, ink))
            size = mark.size()
            mark.drawAtPoint_(NSMakePoint(bx + (bw - size.width) / 2, by + (bh - size.height) / 2))
