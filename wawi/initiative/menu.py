"""The "Iniciativa ▸" submenu: the master switch, one checkmark per act kind
(with how she does it now: "con un globito", "en voz alta"…), and the
calendar opt-in for the morning summary. Rebuilt every time it opens (a menu
delegate), so what learning changed shows without a refresh from the panel.

The panel inserts :func:`menu_item` once and knows nothing else about it.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: Suffix per level in the menu.
SUFFIX = {"quiet": " (en silencio)", "say": " (en voz alta)", "auto": " (directo)"}
TITLE = "Iniciativa"
MASTER = "Hacer cosas sola"
CALENDAR = "Agenda en el resumen"


def entries(engine: Any) -> list[tuple[str, str, bool, bool]]:
    """(title, represented kind / action, checked, enabled) rows. Pure."""
    from wawi.initiative.core import KINDS

    on = bool(engine.enabled)
    rows = [(MASTER, "@master", on, True), ("-", "", False, False)]
    for name, kind in KINDS.items():
        level = engine.level(name) if on else engine.store.level(name, kind.default)
        title = kind.title[0].upper() + kind.title[1:] + SUFFIX.get(level, "")
        rows.append((title, name, level != "off", on))
    rows.append(("-", "", False, False))
    rows.append((CALENDAR, "@calendar", bool(engine.store.calendar), on))
    return rows


def click(engine: Any, what: str) -> None:
    """A row was clicked. Main thread; quick."""
    if what == "@master":
        engine.set_enabled(not engine.enabled)
    elif what == "@calendar":
        turn_on = not engine.store.calendar
        engine.store.calendar = turn_on
        engine.store.save()
        if turn_on:
            from wawi.initiative import calendar

            calendar.request_access()
    elif what:
        engine.set_kind(what, engine.level(what) == "off")


try:
    import objc  # type: ignore[import]
    from AppKit import NSMenu, NSMenuItem  # type: ignore[import]
    from Foundation import NSObject  # type: ignore[import]

    class WawiInitiativeMenuTarget(NSObject):  # type: ignore[misc]
        def initWithEngine_(self, engine):  # noqa: N802
            self = objc.super(WawiInitiativeMenuTarget, self).init()
            if self is None:
                return None
            self._engine = engine
            return self

        def pick_(self, sender):  # noqa: N802
            try:
                click(self._engine, str(sender.representedObject() or ""))
            except Exception:
                log.debug("initiative: menu click failed", exc_info=True)

        def menuNeedsUpdate_(self, menu):  # noqa: N802
            try:
                _fill(menu, self, self._engine)
            except Exception:
                log.debug("initiative: menu refresh failed", exc_info=True)

    def _fill(menu: Any, target: Any, engine: Any) -> None:
        menu.removeAllItems()
        for title, what, checked, enabled in entries(engine):
            if title == "-":
                menu.addItem_(NSMenuItem.separatorItem())
                continue
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "pick:", "")
            item.setTarget_(target)
            item.setRepresentedObject_(what)
            item.setState_(1 if checked else 0)
            item.setEnabled_(bool(enabled))
            menu.addItem_(item)

    _TARGETS: list[Any] = []  # menus hold their targets weakly

    def build_item(engine: Any) -> Any:
        target = WawiInitiativeMenuTarget.alloc().initWithEngine_(engine)
        _TARGETS.append(target)
        sub = NSMenu.alloc().initWithTitle_(TITLE)
        sub.setAutoenablesItems_(False)
        sub.setDelegate_(target)
        _fill(sub, target, engine)
        top = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(TITLE, None, "")
        top.setSubmenu_(sub)
        return top

except Exception:  # no AppKit (tests on another OS): no menu
    def build_item(engine: Any) -> Any:  # type: ignore[misc]
        return None
