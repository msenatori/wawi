"""The buzón "Wawi investiga": a little mailbox you drop things in and TYPE
the order for.

Dropping something on Wawi herself makes her catch it and open the
microphone (wawi/drops.py, Orchestrator._handle_drop). The buzón is the
quiet half of the same thing: a small always-on-top window, drawn in her
paper-and-crayon style, that accepts the same drags (files, the screenshot
thumbnail, pictures, text, links). On a drop she HOLDS the items exactly as
if they had been dropped on her (a spoken order still carries them), and a
small text field pops up beside the buzón: "¿Qué investigo?". Enter sends
the order as a typed turn (Orchestrator.submit_typed), Esc closes it and the
items stay held for their usual hold_s. A double-click on the buzón opens
the same field with nothing in it, for a plain typed question; ⌘V in the
field with a picture (or copied files) on the clipboard adds it as an item.

It can be dragged anywhere; where it was left, and the menu switch
("Buzón"), are kept in ``private/inbox_local.json`` and win over the config
(``inbox.enabled``).

:class:`Inbox` is what the panel (wawi/panel.py) talks to: :meth:`Inbox.tick`
from its main-thread render timer, :meth:`Inbox.set_enabled` from the menu.
Everything it decides with is a pure function or plain state, so the tests
drive it without a window; AppKit is only touched in the ``_show``/``_build``
methods and the view classes at the bottom.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from wawi import drops

log = logging.getLogger(__name__)

#: Where the switch and the position are kept (inside private/).
LOCAL_FILE = "inbox_local.json"
#: Menu item title and NSMenuItem tag (wawi/panel.py tags start at 4242).
TITLE = "Buzón"
TAG = 4250
#: The field's placeholder, and the order sent for an empty Enter with items.
PLACEHOLDER = "¿Qué investigo?"
DEFAULT_ORDER = "Investigá esto."
#: The line above the field with nothing dropped.
HINT = "Escribí y Enter · Esc cierra"
#: A turn is in flight: the typed text stays in the field.
BUSY_TEXT = "Estoy en otra, dame un segundo."
#: The buzón's tooltip.
TOOLTIP = "Wawi investiga: tirá cosas acá · doble clic para escribirle"

#: The mailbox is drawn on a 96 x 92 grid and shrunk by SCALE; its name
#: lives in the tooltip (too small to read under the box at this size).
SCALE = 0.5
#: The buzón window, in points.
WIDTH = 96.0 * SCALE
HEIGHT = 92.0 * SCALE
#: The field's window, in points.
FIELD_WIDTH = 300.0
FIELD_HEIGHT = 74.0
#: Gap between the buzón and its field.
GAP = 6.0
#: The longest summary line above the field (characters).
LINE_MAX = 46

Rect = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Settings and the local file
# --------------------------------------------------------------------------- #


def enabled_from_config(raw: Any) -> bool:
    """``inbox.enabled`` from config.yaml (default True)."""
    section = raw if isinstance(raw, dict) else {}
    value = section.get("enabled", True)
    return value if isinstance(value, bool) else True


def load_state(path: Path) -> dict:
    """The local file's contents, or {} (missing, unreadable, not a dict)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(path: Path, state: dict) -> None:
    """Write *state* atomically. A failure costs the memory, never the app."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        log.warning("inbox: could not save %s", path, exc_info=True)


def _origin(raw: Any) -> tuple[float, float] | None:
    try:
        x, y = raw
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# What the field says and sends
# --------------------------------------------------------------------------- #

_PLURAL = {
    "imagen": "imágenes", "PDF": "PDF", "documento": "documentos", "planilla": "planillas",
    "código": "archivos de código", "audio": "audios", "carpeta": "carpetas",
    "archivo comprimido": "archivos comprimidos", "archivo": "archivos", "link": "links",
    "texto": "textos",
}


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


def _label(item: drops.DropItem) -> str:
    return drops._KIND_LABEL.get(item.kind, "archivo")


def items_line(items: list[drops.DropItem]) -> str:
    """The one line above the field: what was dropped (never its contents
    beyond a short peek at a text). Nothing dropped: the hint."""
    if not items:
        return HINT
    if len(items) == 1:
        item = items[0]
        label = _label(item)
        if item.kind == "text":
            what = f"«{item.text}»"
        elif item.kind == "url":
            what = item.text
        else:
            what = item.name or Path(item.path or "").name
        head = label[0].upper() + label[1:]
        return _clip(f"{head}: {what}" if what else head, LINE_MAX)
    counts: dict[str, int] = {}
    for item in items:
        label = _label(item)
        counts[label] = counts.get(label, 0) + 1
    parts = [label if n == 1 else f"{n} {_PLURAL.get(label, label)}" for label, n in counts.items()]
    return _clip(f"{len(items)} cosas: " + ", ".join(parts), LINE_MAX)


def order_text(typed: str, items: list[drops.DropItem]) -> str | None:
    """What Enter sends: the typed text; with items and nothing typed,
    :data:`DEFAULT_ORDER`; with neither, None (the field just closes)."""
    text = (typed or "").strip()
    if text:
        return text
    return DEFAULT_ORDER if items else None


def paste_items(pb: drops.Pasteboard, *, picture_dir: Path | None = None) -> list[drops.DropItem]:
    """What ⌘V in the field adds as items: copied files, or a picture with no
    text beside it. A text on the clipboard is NOT an item (it is pasted as
    text, which is what ⌘V means in a text field): []."""
    try:
        if pb.file_paths():
            return drops.classify(pb, picture_dir=picture_dir)
        types = set(pb.types() or ())
        if drops.STRING in types and (pb.string(drops.STRING) or "").strip():
            return []
        if types & {drops.PNG, drops.JPEG, drops.TIFF}:
            return [i for i in drops.classify(pb, picture_dir=picture_dir) if i.kind == "image"]
    except Exception:
        log.debug("inbox: could not read the clipboard", exc_info=True)
    return []


# --------------------------------------------------------------------------- #
# Geometry (x y w h, y up, like AppKit's screen coordinates)
# --------------------------------------------------------------------------- #

_DEFAULT_VISIBLE: Rect = (0.0, 0.0, 1440.0, 900.0)


def home(visible: Rect | None) -> tuple[float, float]:
    """Where the buzón starts: bottom-right of the screen, above the Dock."""
    vx, vy, vw, _vh = visible or _DEFAULT_VISIBLE
    return vx + vw - WIDTH - 24.0, vy + 96.0


def on_screen(origin: tuple[float, float], screens: list[Rect]) -> bool:
    """Most of a buzón at *origin* is on one of *screens*."""
    x, y = origin
    cx, cy = x + WIDTH / 2, y + HEIGHT / 2
    return any(sx <= cx <= sx + sw and sy <= cy <= sy + sh for sx, sy, sw, sh in screens)


def field_origin(box: Rect, visible: Rect | None) -> tuple[float, float]:
    """Bottom-left of the field: centred above the buzón, below it when there
    is no room above, and always inside *visible*."""
    vx, vy, vw, vh = visible or _DEFAULT_VISIBLE
    bx, by, bw, bh = box
    x = bx + bw / 2 - FIELD_WIDTH / 2
    y = by + bh + GAP
    if y + FIELD_HEIGHT > vy + vh - 4.0:
        y = by - GAP - FIELD_HEIGHT
    x = min(max(x, vx + 4.0), vx + vw - FIELD_WIDTH - 4.0)
    y = min(max(y, vy + 4.0), vy + vh - FIELD_HEIGHT - 4.0)
    return x, y


# --------------------------------------------------------------------------- #
# The buzón
# --------------------------------------------------------------------------- #


class Inbox:
    """The buzón and its field. Every method runs on the main thread.

    *on_drop(items) -> bool* holds the items (True; False: they went
    elsewhere, e.g. into a note) and must not block. *on_submit(text) -> bool*
    starts the typed turn (True) or refuses it (False: a turn is in flight)
    and must not block either.
    """

    def __init__(self, *, enabled_default: bool = True,
                 on_drop: Callable[[list], bool] | None = None,
                 on_submit: Callable[[str], bool] | None = None,
                 local_file: Path | None = None) -> None:
        self.on_drop = on_drop
        self.on_submit = on_submit
        # None: private/inbox_local.json, looked up on use (a test or
        # WAWI_PRIVATE_DIR may move the folder after import).
        self._local_file = local_file
        state = load_state(self.local_file)
        saved = state.get("enabled")
        self.enabled = saved if isinstance(saved, bool) else bool(enabled_default)
        self.origin = _origin(state.get("origin"))
        #: What the open field is about (the items held for it).
        self.items: list[drops.DropItem] = []
        self.field_open = False
        #: The line above the field (what was dropped, or "busy").
        self.line = HINT
        #: A drag is over the buzón right now.
        self.hover = False
        # AppKit objects, built on first show.
        self.window: Any = None
        self.view: Any = None
        self.field_window: Any = None
        self.field: Any = None
        self._line_view: Any = None
        self._field_delegate: Any = None
        self._drawn: tuple | None = None

    @property
    def local_file(self) -> Path:
        if self._local_file is not None:
            return self._local_file
        from wawi import paths

        return paths.private(LOCAL_FILE)

    def _save(self) -> None:
        state: dict[str, Any] = {"enabled": self.enabled}
        if self.origin is not None:
            state["origin"] = [round(self.origin[0], 1), round(self.origin[1], 1)]
        save_state(self.local_file, state)

    # -- the switch and the position ---------------------------------------

    def set_enabled(self, on: bool) -> None:
        """The menu switch. Persisted; off also closes the field."""
        self.enabled = bool(on)
        if not self.enabled:
            self.cancel()
        self._save()
        log.info("Buzón %s", "on" if self.enabled else "off")

    def moved(self, x: float, y: float) -> None:
        """The owner dragged the buzón to (*x*, *y*): remembered."""
        self.origin = (float(x), float(y))
        self._save()

    # -- drops, the field and the order ------------------------------------

    def dropped(self, items: list[drops.DropItem]) -> bool:
        """*items* arrived (a drag, or ⌘V). With the field already open they
        join what it holds. True when someone took them."""
        if not items:
            return False
        combined = (self.items + list(items)) if self.field_open else list(items)
        callback = self.on_drop
        if callback is None:
            return False
        try:
            held = bool(callback(combined))
        except Exception:
            log.warning("inbox drop callback raised; ignoring", exc_info=True)
            return False
        if held:
            self.items = combined
            self.open_field()
        return True

    def drop_pasteboard(self, pasteboard: Any) -> bool:
        """A drag ended on the buzón. Never raises."""
        self.hover = False
        try:
            items = drops.classify(pasteboard)
        except Exception:
            log.warning("inbox: could not read what was dropped", exc_info=True)
            return False
        if not items:
            log.info("Buzón: nothing she understands on the pasteboard")
            return False
        return self.dropped(items)

    def paste(self, pasteboard: Any) -> bool:
        """⌘V in the field: True when the clipboard was taken as items (a
        picture, files); False lets the field paste the text."""
        items = paste_items(pasteboard)
        if not items:
            return False
        self.dropped(items)
        return True

    def double_clicked(self) -> None:
        """A plain typed question: the field, with nothing in it."""
        if self.field_open:
            self._show_field()
            return
        self.items = []
        self.open_field()

    def open_field(self) -> None:
        self.field_open = True
        self.line = items_line(self.items)
        self._show_field()

    def submit(self, typed: str) -> bool:
        """Enter. True when the field closed (sent, or nothing to send)."""
        order = order_text(typed, self.items)
        if order is None:
            self.close_field()
            return True
        callback = self.on_submit
        try:
            taken = bool(callback(order)) if callback is not None else False
        except Exception:
            log.warning("inbox submit callback raised; ignoring", exc_info=True)
            taken = False
        if not taken:
            # Busy: the text stays where it was typed.
            self.line = BUSY_TEXT
            self._refresh_line()
            return False
        self.items = []
        self.close_field()
        return True

    def cancel(self) -> None:
        """Esc. The items stay held by the orchestrator for their hold_s."""
        self.items = []
        self.close_field()

    def close_field(self) -> None:
        self.field_open = False
        self.line = HINT
        self._hide_field()

    # -- AppKit (main thread) -----------------------------------------------

    def tick(self) -> None:
        """Show, refresh or hide the buzón. Main thread only; never raises."""
        if not _HAS_APPKIT:
            return
        try:
            if not self.enabled:
                for window in (self.window, self.field_window):
                    if window is not None and window.isVisible():
                        window.orderOut_(None)
                self._drawn = None
                return
            if self.window is None:
                self._build()
            if not self.window.isVisible():
                self.window.setFrameOrigin_(self._start_origin())
                self.window.orderFrontRegardless()
                self._drawn = None
            state = (self.hover, bool(self.items), self.field_open)
            if state != self._drawn:
                self._drawn = state
                self.view.setNeedsDisplay_(True)
        except Exception:
            log.debug("inbox: tick failed", exc_info=True)

    def _start_origin(self) -> tuple[float, float]:
        screens: list[Rect] = []
        visible: Rect | None = None
        try:
            for screen in NSScreen.screens() or ():
                f = screen.visibleFrame()
                screens.append((f.origin.x, f.origin.y, f.size.width, f.size.height))
            main = NSScreen.mainScreen()
            if main is not None:
                v = main.visibleFrame()
                visible = (v.origin.x, v.origin.y, v.size.width, v.size.height)
        except Exception:
            pass
        if self.origin is not None and on_screen(self.origin, screens):
            return self.origin
        return home(visible)

    def _build(self) -> None:
        rect = NSMakeRect(0, 0, WIDTH, HEIGHT)
        window = WawiInboxPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        window.setLevel_(NSFloatingWindowLevel)
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setHasShadow_(False)
        window.setHidesOnDeactivate_(False)
        window.setReleasedWhenClosed_(False)
        window.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                      | NSWindowCollectionBehaviorFullScreenAuxiliary)
        view = WawiInboxView.alloc().initWithFrame_host_(rect, self)
        view.setToolTip_(TOOLTIP)
        window.setContentView_(view)
        self.window, self.view = window, view

    def _build_field(self) -> None:
        rect = NSMakeRect(0, 0, FIELD_WIDTH, FIELD_HEIGHT)
        window = WawiInboxFieldPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        window._host = self
        window.setLevel_(NSFloatingWindowLevel)
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setHasShadow_(True)
        window.setHidesOnDeactivate_(False)
        window.setReleasedWhenClosed_(False)
        window.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                      | NSWindowCollectionBehaviorFullScreenAuxiliary)
        card = WawiInboxCard.alloc().initWithFrame_(rect)
        window.setContentView_(card)
        line = NSTextField.alloc().initWithFrame_(NSMakeRect(14, 44, FIELD_WIDTH - 28, 18))
        line.setEditable_(False)
        line.setSelectable_(False)
        line.setBordered_(False)
        line.setDrawsBackground_(False)
        line.setFont_(NSFont.fontWithName_size_("Chalkboard SE", 11.5)
                      or NSFont.systemFontOfSize_(11.5))
        line.setTextColor_(NSColor.secondaryLabelColor())
        line.cell().setLineBreakMode_(4)  # NSLineBreakByTruncatingTail
        card.addSubview_(line)
        field = NSTextField.alloc().initWithFrame_(NSMakeRect(12, 12, FIELD_WIDTH - 24, 26))
        field.setPlaceholderString_(PLACEHOLDER)
        field.setFont_(NSFont.systemFontOfSize_(14))
        field.setBezeled_(True)
        field.setBezelStyle_(1)  # NSTextFieldRoundedBezel
        delegate = WawiInboxFieldDelegate.alloc().initWithHost_(self)
        field.setDelegate_(delegate)
        card.addSubview_(field)
        self.field_window, self.field, self._line_view = window, field, line
        self._field_delegate = delegate  # the field holds its delegate weakly

    def _show_field(self) -> None:
        if not _HAS_APPKIT:
            return
        try:
            if self.window is None:
                self.tick()
            if self.field_window is None:
                self._build_field()
            self._refresh_line()
            self._place_field()
            try:
                NSApp().activateIgnoringOtherApps_(True)
            except Exception:
                pass
            self.field_window.makeKeyAndOrderFront_(None)
            self.field_window.makeFirstResponder_(self.field)
        except Exception:
            log.debug("inbox: could not show the field", exc_info=True)

    def _place_field(self) -> None:
        if self.field_window is None or self.window is None:
            return
        f = self.window.frame()
        box = (f.origin.x, f.origin.y, f.size.width, f.size.height)
        visible = None
        screen = self.window.screen() or NSScreen.mainScreen()
        if screen is not None:
            v = screen.visibleFrame()
            visible = (v.origin.x, v.origin.y, v.size.width, v.size.height)
        self.field_window.setFrameOrigin_(NSMakePoint(*field_origin(box, visible)))

    def _refresh_line(self) -> None:
        if self._line_view is not None:
            try:
                self._line_view.setStringValue_(self.line)
            except Exception:
                log.debug("inbox: could not update the line", exc_info=True)

    def _hide_field(self) -> None:
        if self.field_window is None:
            return
        try:
            was_key = bool(self.field_window.isKeyWindow())
            self.field.setStringValue_("")
            self.field_window.orderOut_(None)
            if was_key:
                # Typing is over: the keyboard goes back to the app the owner
                # was in (opening the field activated Wawi to get it).
                NSApp().deactivate()
        except Exception:
            log.debug("inbox: could not hide the field", exc_info=True)

    def _paste_clipboard(self) -> bool:
        """⌘V reached the field window."""
        from wawi.panel import _AppKitPasteboard

        return self.paste(_AppKitPasteboard(NSPasteboard.generalPasteboard()))

    def _paint(self, dark: bool) -> None:
        """The mailbox, in her doubled navy crayon line. A drag over it fills
        it warm and raises the flag; held items stick out of the slot."""
        ink = (.88, .92, .83, 1) if dark else (.16, .20, .42, 1)
        paper = (.13, .19, .24, .97) if dark else (1.0, .985, .94, .97)
        warm = (.33, .40, .27, .98) if dark else (1.0, .89, .56, .98)
        red = (.87, .24, .20, 1)
        up = self.hover or self.field_open

        def rgba(c):
            return NSColor.colorWithSRGBRed_green_blue_alpha_(*c)

        shrink = NSAffineTransform.transform()
        shrink.scaleBy_(SCALE)
        shrink.translateXBy_yBy_(0, -12)
        shrink.concat()
        # The post.
        post = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(43, 16, 10, 20), 2, 2)
        rgba(paper).setFill()
        post.fill()
        rgba(ink).setStroke()
        post.setLineWidth_(2.2)
        post.stroke()
        # The box: flat bottom, arched top.
        body = NSBezierPath.bezierPath()
        body.moveToPoint_((14, 34))
        body.lineToPoint_((82, 35))
        body.lineToPoint_((82, 64))
        body.curveToPoint_controlPoint1_controlPoint2_((14, 64), (81, 90), (15, 91))
        body.closePath()
        rgba(warm if self.hover else paper).setFill()
        body.fill()
        rgba(ink).setStroke()
        body.setLineWidth_(3.0)
        body.stroke()
        rgba(ink[:3] + (.36,)).setStroke()
        body.setLineWidth_(1.0)
        body.stroke()
        # A paper sticking out of the slot while items are held.
        if self.items:
            sheet = NSBezierPath.bezierPathWithRect_(NSMakeRect(33, 58, 30, 14))
            rgba((1, 1, 1, 1)).setFill()
            sheet.fill()
            rgba(ink).setStroke()
            sheet.setLineWidth_(1.4)
            sheet.stroke()
        # The slot.
        slot = NSBezierPath.bezierPath()
        slot.moveToPoint_((30, 58))
        slot.lineToPoint_((66, 58))
        slot.setLineCapStyle_(1)
        slot.setLineWidth_(4.0)
        rgba(ink).setStroke()
        slot.stroke()
        # The flag: down at rest, up when something is coming.
        pole = NSBezierPath.bezierPath()
        pole.moveToPoint_((82, 50))
        pole.lineToPoint_((89, 50) if not up else (88, 74))
        pole.setLineWidth_(2.2)
        pole.setLineCapStyle_(1)
        pole.stroke()
        flag = NSBezierPath.bezierPath()
        if up:
            flag.moveToPoint_((88, 74))
            flag.lineToPoint_((94, 70))
            flag.lineToPoint_((88, 66))
        else:
            flag.moveToPoint_((89, 50))
            flag.lineToPoint_((92, 44))
            flag.lineToPoint_((86, 44))
        flag.closePath()
        rgba(red).setFill()
        flag.fill()
        flag.setLineWidth_(1.4)
        flag.stroke()


try:  # AppKit, only in the real app ------------------------------------------------
    import objc  # type: ignore[import]
    from AppKit import (  # type: ignore[import]
        NSAffineTransform,
        NSDragOperationCopy,
        NSApp,
        NSAppearanceNameAqua,
        NSAppearanceNameDarkAqua,
        NSAttributedString,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSColor,
        NSDragOperationCopy,
        NSEvent,
        NSEventModifierFlagCommand,
        NSFloatingWindowLevel,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSMakePoint,
        NSMakeRect,
        NSPanel,
        NSPasteboard,
        NSScreen,
        NSTextField,
        NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSObject  # type: ignore[import]

    _HAS_APPKIT = True
except Exception:  # pragma: no cover - not macOS / no PyObjC
    _HAS_APPKIT = False


def _is_dark(view: Any) -> bool:
    try:
        return view.effectiveAppearance().bestMatchFromAppearancesWithNames_(
            [NSAppearanceNameAqua, NSAppearanceNameDarkAqua]) == NSAppearanceNameDarkAqua
    except Exception:
        return False


#: ⌘-keys the field handles itself: with no Edit menu in the app (it is an
#: accessory, menu-bar-only process), nothing else would turn them into the
#: text actions.
_KEY_ACTIONS = {"v": "paste:", "c": "copy:", "x": "cut:", "a": "selectAll:", "z": "undo:"}


def selector_name(selector: Any) -> str:
    """A selector as PyObjC hands it (bytes or str) as a str."""
    return selector.decode() if isinstance(selector, bytes) else str(selector)


if _HAS_APPKIT:

    class WawiInboxPanel(NSPanel):  # type: ignore[misc]
        """Borderless, non-activating: a click or a drag never steals focus."""

        def canBecomeKeyWindow(self):  # noqa: N802
            return False

        def canBecomeMainWindow(self):  # noqa: N802
            return False

    class WawiInboxFieldPanel(NSPanel):  # type: ignore[misc]
        """The field's window: it DOES become key, or nothing could be typed."""

        def canBecomeKeyWindow(self):  # noqa: N802
            return True

        def canBecomeMainWindow(self):  # noqa: N802
            return False

        def performKeyEquivalent_(self, event):  # noqa: N802
            try:
                if event.modifierFlags() & NSEventModifierFlagCommand:
                    key = str(event.charactersIgnoringModifiers() or "").lower()
                    host = getattr(self, "_host", None)
                    if key == "v" and host is not None and host._paste_clipboard():
                        return True
                    action = _KEY_ACTIONS.get(key)
                    if action and NSApp().sendAction_to_from_(action, None, self):
                        return True
            except Exception:
                log.debug("inbox: key equivalent failed", exc_info=True)
            return objc.super(WawiInboxFieldPanel, self).performKeyEquivalent_(event)

    class WawiInboxFieldDelegate(NSObject):  # type: ignore[misc]
        """Enter sends, Esc closes."""

        def initWithHost_(self, host):  # noqa: N802
            self = objc.super(WawiInboxFieldDelegate, self).init()
            if self is None:
                return None
            self._host = host
            return self

        def control_textView_doCommandBySelector_(self, control, _text_view, selector):  # noqa: N802
            try:
                name = selector_name(selector)
                if name == "insertNewline:":
                    self._host.submit(str(control.stringValue() or ""))
                    return True
                if name in ("cancelOperation:", "complete:"):
                    self._host.cancel()
                    return True
            except Exception:
                log.debug("inbox: field command failed", exc_info=True)
            return False

    class WawiInboxCard(NSView):  # type: ignore[misc]
        """The field's paper card."""

        def isOpaque(self):  # noqa: N802
            return False

        def drawRect_(self, _rect):  # noqa: N802
            try:
                dark = _is_dark(self)
                b = self.bounds()
                card = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(2, 2, b.size.width - 4, b.size.height - 4), 12, 13)
                NSColor.colorWithSRGBRed_green_blue_alpha_(
                    *((.13, .19, .24, .98) if dark else (1.0, .985, .94, .98))).setFill()
                card.fill()
                NSColor.colorWithSRGBRed_green_blue_alpha_(
                    *((.88, .92, .83, 1) if dark else (.16, .20, .42, 1))).setStroke()
                card.setLineWidth_(2.2)
                card.stroke()
            except Exception:
                log.debug("inbox: card drawing failed", exc_info=True)

    class WawiInboxView(NSView):  # type: ignore[misc]
        """The mailbox: a drag destination, dragged to move, double-clicked
        to type."""

        def initWithFrame_host_(self, frame, host):  # noqa: N802
            self = objc.super(WawiInboxView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._host = host
            self._press = None
            self._from = None
            self._moved = False
            try:
                self.registerForDraggedTypes_(list(drops.DRAG_TYPES))
            except Exception:
                log.debug("inbox: the buzón does not accept drops", exc_info=True)
            return self

        def isOpaque(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        # -- NSDraggingDestination ----------------------------------------

        def draggingEntered_(self, _sender):  # noqa: N802
            self._host.hover = True
            self.setNeedsDisplay_(True)
            return NSDragOperationCopy

        def draggingUpdated_(self, _sender):  # noqa: N802
            return NSDragOperationCopy

        def draggingExited_(self, _sender):  # noqa: N802
            self._host.hover = False
            self.setNeedsDisplay_(True)

        def prepareForDragOperation_(self, _sender):  # noqa: N802
            return True

        def performDragOperation_(self, sender):  # noqa: N802
            try:
                from wawi.panel import _AppKitPasteboard

                return bool(self._host.drop_pasteboard(
                    _AppKitPasteboard(sender.draggingPasteboard())))
            except Exception:
                log.debug("inbox: drop failed", exc_info=True)
                return False

        def concludeDragOperation_(self, _sender):  # noqa: N802
            self._host.hover = False
            self.setNeedsDisplay_(True)

        # -- the mouse -----------------------------------------------------

        def mouseDown_(self, event):  # noqa: N802
            try:
                if event.clickCount() >= 2:
                    self._from = None
                    self._host.double_clicked()
                    return
                self._from = NSEvent.mouseLocation()
                self._press = self.window().frame().origin
                self._moved = False
            except Exception:
                log.debug("inbox: click failed", exc_info=True)

        def mouseDragged_(self, _event):  # noqa: N802
            if self._from is None or self._press is None:
                return
            try:
                p = NSEvent.mouseLocation()
                dx, dy = p.x - self._from.x, p.y - self._from.y
                if not self._moved and abs(dx) < 3 and abs(dy) < 3:
                    return
                self._moved = True
                self.window().setFrameOrigin_(NSMakePoint(self._press.x + dx, self._press.y + dy))
                if self._host.field_open:
                    self._host._place_field()
            except Exception:
                log.debug("inbox: move failed", exc_info=True)

        def mouseUp_(self, _event):  # noqa: N802
            try:
                if self._moved:
                    o = self.window().frame().origin
                    self._host.moved(o.x, o.y)
            except Exception:
                log.debug("inbox: could not remember where it was left", exc_info=True)
            self._from = self._press = None
            self._moved = False

        def drawRect_(self, _rect):  # noqa: N802
            try:
                self._host._paint(_is_dark(self))
            except Exception:
                log.debug("inbox: drawing failed", exc_info=True)
