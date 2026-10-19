"""The "Su compu ▸" submenu: show / hide the small view, the big window, and
switch her computer off. Rebuilt every time it opens (a menu delegate). The
panel inserts :func:`build_item` once and knows nothing else about it.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)

TITLE = "Su compu"
#: NSMenuItem tag of the "Su compu ▸" item (wawi/panel.py tags start at 4242).
TAG = 4248


def entries(service: Any) -> list[tuple[str, str, bool]]:
    """(title, action, enabled) rows. Pure."""
    running = bool(service.running)
    page = service.page or {}
    where = (page.get("title") or page.get("url") or "").strip()
    state = f"Prendida: {where[:40]}" if running and where else ("Prendida" if running else "Apagada")
    showing = running and bool(service.view_wanted)
    return [
        (state, "", False),
        ("-", "", False),
        ("Ocultar" if showing else "Mostrar", "hide" if showing else "show", True),
        ("Ventana grande", "big", True),
        ("Apagar", "stop", running),
    ]


def click(service: Any, action: str) -> None:
    """A row was clicked. Main thread; quick (slow work goes to a thread)."""
    if action == "show":
        service.open_from_menu(big=False)
    elif action == "big":
        service.open_from_menu(big=True)
    elif action == "hide":
        service.show(False, by_owner=True)
    elif action == "stop":
        threading.Thread(target=service.stop, name="wawi-webvm-stop", daemon=True).start()


try:
    import objc  # type: ignore[import]
    from AppKit import NSMenu, NSMenuItem  # type: ignore[import]
    from Foundation import NSObject  # type: ignore[import]

    class WawiWebVMMenuTarget(NSObject):  # type: ignore[misc]
        def initWithService_(self, service):  # noqa: N802
            self = objc.super(WawiWebVMMenuTarget, self).init()
            if self is None:
                return None
            self._service = service
            return self

        def pick_(self, sender):  # noqa: N802
            try:
                click(self._service, str(sender.representedObject() or ""))
            except Exception:
                log.debug("webvm: menu click failed", exc_info=True)

        def menuNeedsUpdate_(self, menu):  # noqa: N802
            try:
                _fill(menu, self, self._service)
            except Exception:
                log.debug("webvm: menu refresh failed", exc_info=True)

    def _fill(menu: Any, target: Any, service: Any) -> None:
        menu.removeAllItems()
        for title, action, enabled in entries(service):
            if title == "-":
                menu.addItem_(NSMenuItem.separatorItem())
                continue
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, "pick:" if action else None, "")
            item.setTarget_(target)
            item.setRepresentedObject_(action)
            item.setEnabled_(bool(enabled))
            menu.addItem_(item)

    _TARGETS: list[Any] = []  # menus hold their targets weakly

    def build_item(service: Any) -> Any:
        if service is None:
            return None
        target = WawiWebVMMenuTarget.alloc().initWithService_(service)
        _TARGETS.append(target)
        sub = NSMenu.alloc().initWithTitle_(TITLE)
        sub.setAutoenablesItems_(False)
        sub.setDelegate_(target)
        _fill(sub, target, service)
        top = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(TITLE, None, "")
        top.setTag_(TAG)
        top.setSubmenu_(sub)
        return top

except Exception:  # no AppKit (tests on another OS): no menu
    def build_item(service: Any) -> Any:  # type: ignore[misc]
        return None
