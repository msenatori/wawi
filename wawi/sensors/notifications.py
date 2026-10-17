"""TICKET (not implemented): react to macOS notifications.

Idea: when a mail or a chat message arrives, Wawi fetches it (``mail``
behaviour) or perks up, without reading it aloud.

Why it is not built:

- Notification Center has no public API to *read* other apps' notifications.
  The known routes are (a) reading its private SQLite database under
  ``~/Library/Group Containers/group.com.apple.usernoted/``, which needs Full
  Disk Access and breaks between macOS releases, or (b) the Accessibility API
  on the notification banners, which needs Accessibility permission and sees
  the full message text.
- Both hand a desktop pet the content of every private message on the
  machine. That is far more than "a mail arrived" requires.

A safer future shape: per-app opt-in integrations that only report *counts*
(e.g. Mail's unread count via AppleScript, with an Automation prompt the user
explicitly accepts), emitted as ``mail.new`` events with no subject or body.

This module deliberately registers nothing; it exists so the decision is
written down next to the sensors that were built.
"""

from __future__ import annotations

from wawi.reactions import Event
from wawi.sensors.base import Sensor


class NotificationsSensor(Sensor):  # pragma: no cover - placeholder
    name = "notifications"

    def poll(self) -> list[Event]:
        return []
