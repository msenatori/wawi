"""She notices what the owner is doing: a silent gesture now and then.

The ONLY input is the front window's owner — the app's name, as the window
list already gives it (wawi/pet/desktop.py; no Screen Recording, no titles).
Never a window title, a URL or anything on screen, and the logs only ever
see the category. The gestures are silent (the owner may be in a call) and
rare: one every half hour or so at most (the personality's context section),
only while she rests awake, habituated like any small stimulus
(wawi/pet/dog.py), and held back in meetings, full screen, quiet mode, at
night and while a turn, a game, the board or notes are on. A glance or an
ear flick, not a performance.

Pure: the clock comes in with every call.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

#: App names (lower case, as the window list shows them) per category.
#: Matching is by prefix too, so "PyCharm CE" or "Code - Insiders" count.
APPS: dict[str, tuple[str, ...]] = {
    "code": ("cursor", "code", "visual studio code", "vscodium", "xcode", "iterm2",
             "iterm", "terminal", "warp", "ghostty", "zed", "pycharm", "intellij idea",
             "webstorm", "goland", "rustrover", "clion", "android studio", "sublime text",
             "nova", "bbedit", "alacritty", "kitty", "wezterm", "windsurf"),
    "video": ("quicktime player", "iina", "vlc", "tv", "netflix", "infuse", "plex",
              "disney+", "prime video", "youtube"),
    "chat": ("slack", "messages", "whatsapp", "discord", "telegram", "signal",
             "microsoft teams", "mensajes"),
    "design": ("figma", "sketch", "photoshop", "adobe photoshop", "illustrator",
               "adobe illustrator", "affinity designer", "affinity photo", "pixelmator pro",
               "procreate", "blender", "framer"),
}

#: What she does for each, strung from clips she already has (no new art).
#: "glance" is a look toward the owner's side (added by the caller).
GESTURES: dict[str, tuple[str, ...]] = {
    # You at work: a look over.
    "code": ("glance",),
    "video": ("glance",),
    # Something pinged: ears up, a look over.
    "chat": ("ear_twitch", "glance"),
    "design": ("glance",),
    # You switched apps: a quick look at what came up.
    "switch": ("glance",),
}
#: Now and then (the personality's context.laptop_chance) for code: busy like
#: you, at her little laptop for a moment.
LAPTOP: tuple[str, ...] = ("glance", "laptop")

#: A new front app must stay front this long before it counts (a cmd-tab
#: through several apps is one switch, to where you stopped).
SETTLE = 1.5
#: A switch older than this is old news: no glance for it any more.
STALE = 10.0
#: The same app front this long: she may do its category's gesture.
DWELL = 90.0


def classify(owner: str | None) -> str | None:
    """The category of an app name, or None (most apps have none)."""
    name = (owner or "").strip().lower()
    if not name:
        return None
    for category, apps in APPS.items():
        for app in apps:
            if name == app or (len(app) > 3 and name.startswith(app + " ")):
                return category
    return None


@dataclass
class ContextGestures:
    """When she may do a context gesture, and which category it is for.

    *gap*: seconds between gestures, drawn in the range (the rate limit).
    """

    gap: tuple[float, float]
    rng: random.Random = field(default_factory=random.Random)
    app: str | None = None
    since: float = 0.0
    switched: bool = False
    next_at: float = 0.0
    started: bool = False

    def sample(self, now: float, owner: str | None, allowed: bool) -> str | None:
        """Feed the front app (every tick is fine); the category to gesture
        for now ("switch" for a plain app switch), or None. *allowed* False
        (a meeting, the night, busy…) only tracks the app."""
        if not self.started:
            # No gesture for whatever happened to be open at launch.
            self.started = True
            self.app, self.since = owner, now
            self.next_at = now + self.rng.uniform(*self.gap) * 0.5
            return None
        if owner != self.app:
            self.app, self.since, self.switched = owner, now, owner is not None
        if self.switched and now - self.since > STALE:
            self.switched = False
        if not allowed or now < self.next_at or not owner:
            return None
        category = classify(owner)
        if self.switched and now - self.since >= SETTLE:
            self.switched = False
            return self._consume(now, category or "switch")
        if category is not None and not self.switched and now - self.since >= DWELL:
            return self._consume(now, category)
        return None

    def _consume(self, now: float, category: str) -> str:
        self.next_at = now + self.rng.uniform(*self.gap)
        return category
