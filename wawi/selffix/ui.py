"""The "Autoarreglo ▸" submenu of the menu-bar item.

:class:`SelffixMenu` is plain Python (what each entry says and does), so it is
tested without AppKit; :meth:`SelffixMenu.menu_item` binds it to an NSMenu.
Clicks that touch git or start agents run off the main thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

#: (action, title)
ENTRY_TOGGLE = "toggle"
ENTRY_FIX = "fix"
ENTRY_UNDO = "undo"
ENTRY_HISTORY = "history"
ENTRY_APPLY = "apply"
ENTRY_DISCARD = "discard"


class SelffixMenu:
    def __init__(self, service: Any, *, notify: Callable[[str], None] | None = None,
                 run: Callable[[Callable[[], None]], None] | None = None) -> None:
        self.service = service
        self.notify = notify or service.notify
        self.run = run or (lambda fn: threading.Thread(target=fn, name="wawi-selffix-menu",
                                                       daemon=True).start())
        self._items: dict[str, Any] = {}
        self._target = None

    def entries(self) -> list[tuple[str, str, bool, bool]]:
        """(action, title, checked, visible), in menu order."""
        pending = self.service.store["pending"]
        asking = bool(pending) and not pending.get("approved")
        return [
            (ENTRY_TOGGLE, "Autoarreglo activado", self.service.enabled, True),
            (ENTRY_FIX, "Arreglarme ahora", False, True),
            (ENTRY_APPLY, "Aplicar el arreglo pendiente", False, asking),
            (ENTRY_DISCARD, "Descartar el arreglo pendiente", False, asking),
            (ENTRY_UNDO, "Deshacer el último arreglo", False, True),
            (ENTRY_HISTORY, "Ver historial", False, True),
        ]

    def click(self, action: str) -> None:
        service = self.service
        if action == ENTRY_TOGGLE:
            service.set_enabled(not service.enabled)
            self.notify("Autoarreglo prendido." if service.enabled else "Autoarreglo apagado.")
        elif action == ENTRY_HISTORY:
            service.show_history()
        elif action == ENTRY_FIX:
            self.run(lambda: self.notify(service.fix_now()[1]))
        elif action == ENTRY_UNDO:
            self.run(lambda: self.notify(service.undo_last()[1]))
        elif action in (ENTRY_APPLY, ENTRY_DISCARD):
            self.run(lambda: self.notify(service.decide_pending(action == ENTRY_APPLY)[1]))
        self.refresh()

    # -- AppKit -------------------------------------------------------------
    def menu_item(self) -> Any:
        """The "Autoarreglo" NSMenuItem with its submenu."""
        from AppKit import NSMenu, NSMenuItem

        target = _target_class().alloc().initWithMenu_(self)
        self._target = target  # menus hold their targets weakly
        sub = NSMenu.alloc().initWithTitle_("Autoarreglo")
        sub.setAutoenablesItems_(False)
        sub.setDelegate_(target)  # menuNeedsUpdate_: fresh checkmark and pending items
        for action, title, _checked, _visible in self.entries():
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "clicked:", "")
            item.setTarget_(target)
            item.setRepresentedObject_(action)
            sub.addItem_(item)
            self._items[action] = item
        top = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Autoarreglo", None, "")
        top.setSubmenu_(sub)
        self.refresh()
        return top

    def refresh(self) -> None:
        for action, _title, checked, visible in self.entries():
            item = self._items.get(action)
            if item is None:
                continue
            try:
                item.setState_(1 if checked else 0)
                item.setHidden_(not visible)
            except Exception:
                pass


_TARGET = None


def _target_class() -> Any:
    global _TARGET
    if _TARGET is None:
        from Foundation import NSObject

        class WawiSelffixMenuTarget(NSObject):  # type: ignore[misc]
            def initWithMenu_(self, menu):  # noqa: N802
                self = self.init()
                if self is not None:
                    self._menu = menu
                return self

            def clicked_(self, sender):  # noqa: N802
                try:
                    self._menu.click(str(sender.representedObject()))
                except Exception:
                    log.warning("selffix: menu click failed", exc_info=True)

            def menuNeedsUpdate_(self, _menu):  # noqa: N802
                self._menu.refresh()

        _TARGET = WawiSelffixMenuTarget
    return _TARGET
