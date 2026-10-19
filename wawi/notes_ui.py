"""The giant STOP beside Wawi while she takes notes (wawi/notes.py).

A borderless, non-activating floating panel in her paper-and-crayon style
(the same vocabulary as wawi/confirm.py): a big red rounded square that says
STOP, with the time recorded and the lines written under it. A click anywhere
on it ends the session. It never takes focus, so clicking it in the middle of
typing does not steal the keyboard from the app you are in.

:meth:`StopButton.tick` is called by the panel's render timer on the main
thread, and is the only place AppKit is touched. Everything it decides with
is a pure function below, so the tests do not need a window.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from wawi.notes import format_elapsed

log = logging.getLogger(__name__)

#: The card, in points.
WIDTH = 150.0
HEIGHT = 158.0
#: The red button inside it (flipped coordinates: from the top-left).
BUTTON = (14.0, 12.0, 122.0, 100.0)
#: Gap to her window.
GAP = 8.0


def label(elapsed: float, lines: int) -> str:
    """The line under the button: ``● 03:12 · 5 líneas``."""
    what = "1 línea" if lines == 1 else f"{lines} líneas"
    return f"● {format_elapsed(elapsed)} · {what}"


def place(anchor: tuple[float, float, float, float] | None,
          visible: tuple[float, float, float, float] | None) -> tuple[float, float]:
    """Bottom-left origin of the card: right of *anchor* (her window, x y w h,
    y up), top-aligned with it, flipped to her left if it would leave the
    screen, and always inside *visible*. No anchor: the screen's bottom-right."""
    if visible is None:
        visible = (0.0, 0.0, 1440.0, 900.0)
    vx, vy, vw, vh = visible
    if anchor is None:
        x, y = vx + vw - WIDTH - 24.0, vy + 24.0
    else:
        ax, ay, aw, ah = anchor
        x = ax + aw + GAP
        if x + WIDTH > vx + vw - 4.0:
            x = ax - GAP - WIDTH
        y = ay + max(0.0, ah - HEIGHT)
    x = min(max(x, vx + 4.0), vx + vw - WIDTH - 4.0)
    y = min(max(y, vy + 4.0), vy + vh - HEIGHT - 4.0)
    return x, y


def hit(x: float, y: float) -> bool:
    """A click at (*x*, *y*) (flipped view coordinates) is on the button.
    Generous: the whole card counts, minus nothing -- it is a STOP."""
    return 0.0 <= x <= WIDTH and 0.0 <= y <= HEIGHT


try:  # AppKit, only in the real app ------------------------------------------------
    import objc  # type: ignore[import]
    from AppKit import (  # type: ignore[import]
        NSApp,
        NSAppearanceNameDarkAqua,
        NSAttributedString,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSFloatingWindowLevel,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSMakePoint,
        NSMakeRect,
        NSPanel,
        NSScreen,
        NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
    )

    _HAS_APPKIT = True
except Exception:  # pragma: no cover - not macOS / no PyObjC
    _HAS_APPKIT = False


if _HAS_APPKIT:

    class WawiNotesStopPanel(NSPanel):  # type: ignore[misc]
        """Borderless, non-activating: a click never steals focus."""

        def canBecomeKeyWindow(self):  # noqa: N802
            return False

        def canBecomeMainWindow(self):  # noqa: N802
            return False

    class WawiNotesStopView(NSView):  # type: ignore[misc]
        def initWithFrame_host_(self, frame, host):  # noqa: N802
            self = objc.super(WawiNotesStopView, self).initWithFrame_(frame)
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
                if hit(float(p.x), float(p.y)):
                    self._host._clicked()
            except Exception:
                log.debug("notes: STOP click failed", exc_info=True)

        def drawRect_(self, _rect):  # noqa: N802
            try:
                self._host._paint()
            except Exception:
                log.debug("notes: STOP drawing failed", exc_info=True)


class StopButton:
    """The STOP card. *on_stop* is called on a thread of its own when clicked."""

    def __init__(self, on_stop: Callable[[], None] | None = None) -> None:
        self.on_stop = on_stop
        self.window: Any = None
        self.view: Any = None
        self._drawn: tuple | None = None
        self._text = ""
        self._dark = False
        self._pressed = False

    # -- main thread ------------------------------------------------------
    def tick(self, session: Any, anchor_window: Any = None) -> None:
        """Show, move, refresh or hide the card for *session* (None = no
        note being taken). Main thread only; never raises."""
        if not _HAS_APPKIT:
            return
        try:
            if session is None:
                self._pressed = False
                # Whether or not a frame was drawn since: a click resets
                # _drawn (to repaint pressed) and the session can end before
                # the next tick, and then the card stayed up for good.
                if self.window is not None and self.window.isVisible():
                    self.window.orderOut_(None)
                self._drawn = None
                return
            if self.window is None:
                self._build()
            text = label(session.elapsed(), session.lines)
            origin = place(*self._geometry(anchor_window))
            state = (text, origin, self._pressed)
            if state == self._drawn:
                return
            first = self._drawn is None or not self.window.isVisible()
            self._drawn = state
            self._text = text
            self.window.setFrameOrigin_(NSMakePoint(*origin))
            self.view.setNeedsDisplay_(True)
            if first:
                self._dark = self._read_dark()
                self.window.orderFrontRegardless()
        except Exception:
            log.debug("notes: STOP tick failed", exc_info=True)

    def close(self) -> None:
        if self.window is not None:
            try:
                self.window.orderOut_(None)
                self.window.close()
            except Exception:
                log.debug("notes: closing STOP failed", exc_info=True)
        self.window = self.view = None
        self._drawn = None

    # -- internals --------------------------------------------------------
    def _build(self) -> None:
        rect = NSMakeRect(0, 0, WIDTH, HEIGHT)
        window = WawiNotesStopPanel.alloc().initWithContentRect_styleMask_backing_defer_(
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
        view = WawiNotesStopView.alloc().initWithFrame_host_(rect, self)
        window.setContentView_(view)
        self.window, self.view = window, view

    @staticmethod
    def _geometry(anchor_window: Any) -> tuple[tuple | None, tuple | None]:
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

    def _clicked(self) -> None:
        if self._pressed:
            return  # one click is enough; the session is already ending
        self._pressed = True
        self._drawn = None  # repaint pressed
        if self.view is not None:
            self.view.setNeedsDisplay_(True)
        callback = self.on_stop
        if callback is None:
            return
        threading.Thread(target=callback, name="wawi-notes-stop", daemon=True).start()

    def _paint(self) -> None:
        from wawi.games.ui import _attrs, _rgba

        paper = (0.13, 0.19, 0.24, .98) if self._dark else (1.0, .985, .94, .98)
        ink = (.88, .92, .83, 1) if self._dark else (.16, .20, .42, 1)
        red = (.62, .16, .14, 1) if self._pressed else (.87, .24, .20, 1)
        card = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(2, 2, WIDTH - 4, HEIGHT - 4), 14, 14)
        _rgba(paper).setFill()
        card.fill()
        _rgba(ink).setStroke()
        card.setLineWidth_(1.8)
        card.stroke()

        bx, by, bw, bh = BUTTON
        button = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(bx, by, bw, bh), 18, 18)
        _rgba(red).setFill()
        button.fill()
        _rgba(ink).setStroke()
        button.setLineWidth_(3.0)
        button.stroke()
        # The stop square, hand-drawn, above the word.
        square = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(bx + bw / 2 - 11, by + 14, 22, 22), 4, 4)
        _rgba((1, .97, .92, 1)).setFill()
        square.fill()

        attrs = _attrs(30, (1, .97, .92, 1))
        bold = NSFont.fontWithName_size_("ChalkboardSE-Bold", 30)
        if bold is not None:
            attrs = {NSFontAttributeName: bold,
                     NSForegroundColorAttributeName: _rgba((1, .97, .92, 1))}
        word = NSAttributedString.alloc().initWithString_attributes_("STOP", attrs)
        size = word.size()
        word.drawAtPoint_(NSMakePoint(bx + (bw - size.width) / 2, by + 42))

        # The REC dot in red, the rest in ink.
        dot_text, _, rest = self._text.partition(" ")
        dot = NSAttributedString.alloc().initWithString_attributes_(
            dot_text + " ", _attrs(12.5, (.87, .24, .20, 1)))
        line = NSAttributedString.alloc().initWithString_attributes_(rest, _attrs(12.5, ink))
        dw, lw = dot.size().width, line.size().width
        x0 = (WIDTH - dw - lw) / 2
        dot.drawAtPoint_(NSMakePoint(x0, by + bh + 14))
        line.drawAtPoint_(NSMakePoint(x0 + dw, by + bh + 14))
