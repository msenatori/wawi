"""The "Personalidad ▸" submenu: one row per personality in personalities/,
a check on the one she has. Rebuilt every time it opens (a menu delegate), so
a file dropped in the folder shows up without a restart. The panel inserts
:func:`build_item` once and knows nothing else about it.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

TITLE = "Personalidad"
#: NSMenuItem tag of the "Personalidad ▸" item (wawi/panel.py tags start at 4242).
TAG = 4249


def entries() -> list[tuple[str, str, bool, str]]:
    """(title, name, checked, tooltip) rows. Pure but for the folder read."""
    from wawi.pet import api, personality

    current = api.personality()
    return [(name.replace("_", " ").capitalize(), name, name == current, desc)
            for name, desc in personality.summaries()]


def click(name: str) -> None:
    from wawi.pet import api

    if name:
        api.set_personality(name)


try:
    import objc  # type: ignore[import]
    from AppKit import NSMenu, NSMenuItem  # type: ignore[import]
    from Foundation import NSObject  # type: ignore[import]

    class WawiPersonalityMenuTarget(NSObject):  # type: ignore[misc]
        def pick_(self, sender):  # noqa: N802
            try:
                click(str(sender.representedObject() or ""))
            except Exception:
                log.debug("personality: menu click failed", exc_info=True)

        def menuNeedsUpdate_(self, menu):  # noqa: N802
            try:
                _fill(menu, self)
            except Exception:
                log.debug("personality: menu refresh failed", exc_info=True)

    def _fill(menu: Any, target: Any) -> None:
        menu.removeAllItems()
        for title, name, checked, tip in entries():
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "pick:", "")
            item.setTarget_(target)
            item.setRepresentedObject_(name)
            item.setState_(1 if checked else 0)
            if tip:
                item.setToolTip_(tip)
            menu.addItem_(item)

    _TARGETS: list[Any] = []  # menus hold their targets weakly

    def build_item() -> Any:
        target = WawiPersonalityMenuTarget.alloc().init()
        _TARGETS.append(target)
        sub = NSMenu.alloc().initWithTitle_(TITLE)
        sub.setAutoenablesItems_(False)
        sub.setDelegate_(target)
        _fill(sub, target)
        top = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(TITLE, None, "")
        top.setTag_(TAG)
        top.setSubmenu_(sub)
        return top

except Exception:  # no AppKit (tests on another OS): no menu
    def build_item() -> Any:  # type: ignore[misc]
        return None
