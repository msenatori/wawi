"""Thread-safe confirmation queue and a non-activating AppKit presenter.

Producers may run on worker threads. ``tick`` is called by the panel on the
main thread and is the only place where AppKit is touched.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)
TIMEOUT_S = 180.0


@dataclass(frozen=True)
class Confirmation:
    id: str
    title: str
    detail: str
    origin: str
    accept: Callable[[], None]
    cancel: Callable[[], None]
    expires_at: float


class Registry:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 presenter: Any = None) -> None:
        self.clock = clock
        self.presenter = presenter
        self._lock = threading.RLock()
        self._items: list[Confirmation] = []
        self._shown: str | None = None

    def add(self, title: str, detail: str, origin: str,
            accept: Callable[[], None], cancel: Callable[[], None],
            *, timeout: float = TIMEOUT_S) -> str:
        item = Confirmation(uuid.uuid4().hex, title, detail, origin, accept, cancel,
                            self.clock() + timeout)
        with self._lock:
            self._items.append(item)
        return item.id

    def resolve(self, identifier: str, accepted: bool) -> bool:
        with self._lock:
            item = next((i for i in self._items if i.id == identifier), None)
            if item is None or self.clock() >= item.expires_at:
                return False
            self._items.remove(item)
        try:
            (item.accept if accepted else item.cancel)()
        except Exception:
            log.warning("confirmation callback failed: %s", item.origin, exc_info=True)
        return True

    def discard(self, identifier: str) -> None:
        """Remove a superseded/expired request without running either callback."""
        with self._lock:
            self._items = [i for i in self._items if i.id != identifier]

    def pending(self) -> list[Confirmation]:
        with self._lock:
            return list(self._items)

    def tick(self) -> None:
        """Expire and reconcile the single visible panel; call on AppKit's main thread."""
        with self._lock:
            now = self.clock()
            expired = [i for i in self._items if now >= i.expires_at]
            if expired:
                self._items = [i for i in self._items if now < i.expires_at]
            item = self._items[0] if self._items else None
            shown = self._shown
            self._shown = item.id if item else None
        for old in expired:
            log.info("confirmation expired: %s (%s)", old.title, old.origin)
            try:
                old.cancel()
            except Exception:
                log.warning("confirmation expiry callback failed", exc_info=True)
        if self.presenter is None or shown == self._shown:
            return
        if item is None:
            self.presenter.hide()
        else:
            self.presenter.show(item)


REGISTRY = Registry()


def install_presenter(presenter: Any) -> None:
    REGISTRY.presenter = presenter


def tick() -> None:
    REGISTRY.tick()


class AppKitPresenter:
    """Paper/chalk confirmation card anchored beside Wawi's mascot."""

    def __init__(self, panel: Any) -> None:
        self.panel = panel
        self.window: Any = None
        self.view: Any = None
        self.item: Confirmation | None = None
        self.dark = False

    def show(self, item: Confirmation) -> None:
        from AppKit import (NSPanel, NSView, NSWindowStyleMaskBorderless,
                            NSWindowStyleMaskNonactivatingPanel, NSBackingStoreBuffered,
                            NSFloatingWindowLevel, NSWindowCollectionBehaviorCanJoinAllSpaces,
                            NSWindowCollectionBehaviorFullScreenAuxiliary, NSMakeRect)
        from wawi.games.ui import _attrs, _rgba  # shared hand-drawn visual vocabulary

        self.item = item
        from AppKit import NSApp, NSAppearanceNameDarkAqua
        app = NSApp()
        appearance = app.effectiveAppearance() if app is not None else None
        self.dark = bool(appearance and appearance.bestMatchFromAppearancesWithNames_(
            [NSAppearanceNameDarkAqua]) == NSAppearanceNameDarkAqua)
        if self.window is None:
            host = self
            class NonactivatingPanel(NSPanel):
                def canBecomeKeyWindow(self):
                    return False
                def canBecomeMainWindow(self):
                    return False
            class Card(NSView):
                def isFlipped(self):
                    return True
                def mouseDown_(self, event):
                    p = self.convertPoint_fromView_(event.locationInWindow(), None)
                    if 104 <= p.y <= 137:
                        ident = host.item.id if host.item else None
                        if ident:
                            threading.Thread(target=REGISTRY.resolve,
                                             args=(ident, p.x < 157),
                                             name="wawi-confirm-click", daemon=True).start()
                def drawRect_(self, rect):
                    host._draw()
            self.window = NonactivatingPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 306, 148),
                NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
                NSBackingStoreBuffered, False)
            self.window.setLevel_(NSFloatingWindowLevel)
            self.window.setOpaque_(False)
            self.window.setBackgroundColor_(_rgba((0, 0, 0, 0)))
            self.window.setHasShadow_(True)
            self.window.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces |
                                               NSWindowCollectionBehaviorFullScreenAuxiliary)
            self.view = Card.alloc().initWithFrame_(NSMakeRect(0, 0, 306, 148))
            self.window.setContentView_(self.view)
        mascot = getattr(self.panel, '_mascot_window', None)
        anchor = mascot.frame() if mascot is not None else self.panel._window.frame()
        x = anchor.origin.x + anchor.size.width + 8
        y = anchor.origin.y + max(0, anchor.size.height - 148)
        screen = (mascot or self.panel._window).screen()
        if screen is not None:
            vis = screen.visibleFrame()
            x = min(max(x, vis.origin.x + 4), vis.origin.x + vis.size.width - 310)
            y = min(max(y, vis.origin.y + 4), vis.origin.y + vis.size.height - 152)
        self.window.setFrame_display_(NSMakeRect(x, y, 306, 148), True)
        self.view.setNeedsDisplay_(True)
        self.window.orderFrontRegardless()

    def hide(self) -> None:
        self.item = None
        if self.window is not None:
            self.window.orderOut_(None)

    def _draw(self) -> None:
        from AppKit import NSAttributedString, NSBezierPath, NSMakePoint, NSMakeRect
        from wawi.games.ui import _attrs, _rgba
        item = self.item
        if item is None:
            return
        paper = (0.13, 0.19, 0.24, .98) if self.dark else (1.0, .985, .94, .98)
        ink = (.88, .92, .83, 1) if self.dark else (.16, .20, .42, 1)
        button = (.22, .31, .34, 1) if self.dark else (.93, .90, .80, 1)
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(2, 2, 302, 144), 12, 12)
        _rgba(paper).setFill(); path.fill()
        _rgba(ink).setStroke(); path.setLineWidth_(1.7); path.stroke()
        for text, y, size in ((item.title[:45], 16, 16), (item.detail[:95], 49, 12)):
            NSAttributedString.alloc().initWithString_attributes_(text, _attrs(size, ink)).drawInRect_(
                NSMakeRect(15, y, 276, 43))
        for label, x in (("Aceptar", 18), ("Cancelar", 158)):
            pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(x, 104, 130, 33), 9, 9)
            _rgba(button).setFill(); pill.fill()
            _rgba(ink).setStroke(); pill.setLineWidth_(1.3); pill.stroke()
            title = NSAttributedString.alloc().initWithString_attributes_(label, _attrs(13, ink))
            size = title.size()
            title.drawAtPoint_(NSMakePoint(x + (130 - size.width) / 2, 110))
