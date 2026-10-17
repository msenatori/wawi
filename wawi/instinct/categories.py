"""What kind of app is in front: a small built-in table, plus what is unknown.

Only the app's bundle id and its name are ever looked at -- never a window
title, never what is on screen. An app the table does not know becomes
``other:<name>`` (lowercased, trimmed), so "the owner opens Linear at 10"
can still be learned without a table entry for every app on earth.
"""

from __future__ import annotations

import re

#: Categories whose long stretches count as a focus block.
FOCUS = frozenset({"code", "terminal", "design", "writing"})

#: Bundle id (exact or prefix) → category. Checked before names.
BUNDLES: dict[str, str] = {
    "com.microsoft.vscode": "code", "com.visualstudio.code": "code",
    "com.todesktop.230313mzl4w4u92": "code",  # Cursor
    "dev.zed.zed": "code", "com.jetbrains.": "code", "com.apple.dt.xcode": "code",
    "com.sublimetext.": "code", "com.panic.nova": "code", "org.vim.macvim": "code",
    "com.apple.terminal": "terminal", "com.googlecode.iterm2": "terminal",
    "dev.warp.warp": "terminal", "net.kovidgoyal.kitty": "terminal",
    "com.github.wez.wezterm": "terminal", "io.alacritty": "terminal",
    "com.mitchellh.ghostty": "terminal",
    "com.apple.safari": "browser", "com.google.chrome": "browser",
    "org.mozilla.firefox": "browser", "company.thebrowser.browser": "browser",
    "com.brave.browser": "browser", "com.microsoft.edgemac": "browser",
    "com.tinyspeck.slackmacgap": "chat", "com.hnc.discord": "chat",
    "net.whatsapp.whatsapp": "chat", "ru.keepcoder.telegram": "chat",
    "com.apple.mobilesms": "chat",
    "us.zoom.xos": "meeting", "com.microsoft.teams": "meeting",
    "com.microsoft.teams2": "meeting", "com.apple.facetime": "meeting",
    "com.cisco.webexmeetingsapp": "meeting",
    "com.spotify.client": "music", "com.apple.music": "music",
    "com.figma.desktop": "design", "com.bohemiancoding.sketch3": "design",
    "com.adobe.": "design", "com.pixelmatorteam.": "design",
    "com.apple.mail": "mail", "com.microsoft.outlook": "mail",
    "com.readdle.smartemail-mac": "mail",
    "com.apple.ical": "calendar", "com.flexibits.fantastical2.mac": "calendar",
    "md.obsidian": "writing", "com.apple.notes": "writing",
    "com.microsoft.word": "writing", "com.apple.iwork.pages": "writing",
    "notion.id": "writing",
    "com.apple.finder": "files",
}

#: App name (lowercase, substring) → category, for apps without a known id.
NAMES: dict[str, str] = {
    "code": "code", "cursor": "code", "xcode": "code", "pycharm": "code",
    "intellij": "code", "webstorm": "code", "goland": "code", "zed": "code",
    "terminal": "terminal", "iterm": "terminal", "warp": "terminal",
    "ghostty": "terminal", "kitty": "terminal", "alacritty": "terminal",
    "safari": "browser", "chrome": "browser", "firefox": "browser", "arc": "browser",
    "brave": "browser", "edge": "browser",
    "slack": "chat", "discord": "chat", "whatsapp": "chat", "telegram": "chat",
    "messages": "chat", "mensajes": "chat",
    "zoom": "meeting", "teams": "meeting", "facetime": "meeting", "webex": "meeting",
    "meet": "meeting",
    "spotify": "music", "music": "music", "música": "music",
    "figma": "design", "sketch": "design", "photoshop": "design", "illustrator": "design",
    "mail": "mail", "outlook": "mail", "spark": "mail",
    "calendar": "calendar", "calendario": "calendar", "fantastical": "calendar",
    "obsidian": "writing", "notes": "writing", "notas": "writing", "notion": "writing",
    "word": "writing", "pages": "writing",
    "finder": "files",
}

#: Spanish names, for patterns and the self-model.
WORDS: dict[str, str] = {
    "code": "el editor", "terminal": "la terminal", "browser": "el navegador",
    "chat": "el chat", "meeting": "reuniones", "music": "música", "design": "diseño",
    "mail": "el mail", "calendar": "el calendario", "writing": "tus notas",
    "files": "el Finder", "idle": "nada",
}

_SAFE = re.compile(r"[^\w ._+-]")


def _other(name: str) -> str:
    clean = _SAFE.sub("", (name or "").strip().lower())[:24].strip()
    return f"other:{clean}" if clean else "other"


def categorize(bundle: str | None, name: str | None,
               overrides: dict[str, str] | None = None) -> str:
    """The category of an app, from its bundle id and/or its name."""
    b = (bundle or "").strip().lower()
    n = (name or "").strip().lower()
    for key in (n, b):
        if key and overrides and key in overrides:
            return overrides[key]
    if b:
        if b in BUNDLES:
            return BUNDLES[b]
        for prefix, cat in BUNDLES.items():
            if prefix.endswith(".") and b.startswith(prefix):
                return cat
    if n:
        if n in NAMES:
            return NAMES[n]
        for word in n.replace("-", " ").split():
            if word in NAMES:
                return NAMES[word]
    if not n and not b:
        return "unknown"
    return _other(n or b.rsplit(".", 1)[-1])


def word(category: str) -> str:
    """A category in Spanish ("other:linear" → "Linear")."""
    if category.startswith("other:"):
        return category[6:].title() or "otra app"
    return WORDS.get(category, category)
