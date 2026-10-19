"""Where everything personal lives: one gitignored folder, ``private/``.

Wawi records a room and a screen. Every file that holds any of it (the
transcripts, the logs, what she learned, the local switches, your
config.yaml) lives under ONE folder at the repository root, so nothing
personal can be published by mistake and a backup is a single copy.

``WAWI_PRIVATE_DIR`` moves the folder (tests, or a second profile). It is
read on every call, never cached, so a test's override takes effect at once.

A path from config.yaml is honoured as written, with one exception: a bare
file name (``conversation_log: conversation.log``, the way the template and
every older config spell it) means "the usual file", which now lives under
``private/``. Anything with a folder in it, absolute or relative, is yours.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

#: The env var that moves the private folder.
ENV = "WAWI_PRIVATE_DIR"
#: The folder name, relative to the working directory (the repository root).
PRIVATE = Path("private")

#: Everything that used to sit loose at the repository root and now moves
#: into ``private/`` on the first start (see :func:`migrate`). Names only.
LEGACY_NAMES: tuple[str, ...] = (
    "config.yaml",
    "conversation.log",
    "answers.log",
    "wawi.log",
    "preferences.txt",
    "recall.txt",
    "issues",
    "screenshots",
    "board_local",
    "inflight.json",
    "panel_state.json",
    "pet_state.json",
    "games_state.json",
    "instinct_state.json",
    "learn_state.json",
    "learn_brain.npz",
    "learn_life.jsonl",
    "curiosities_local.json",
    "initiative_local.json",
    "music_local.json",
    "orca_local.json",
    "plugins_local.json",
    "reminders_local.json",
    "selffix_local.json",
    "sensors_local.json",
    "shortcuts_local.json",
    "wake_local.json",
)


def private_dir() -> Path:
    """The private folder: ``$WAWI_PRIVATE_DIR`` or ``private``."""
    raw = os.environ.get(ENV, "").strip()
    return Path(raw).expanduser() if raw else PRIVATE


def private(name: str | Path) -> Path:
    """*name* inside the private folder."""
    return private_dir() / name


def under(root: Path) -> Path:
    """The private folder of the checkout at *root* (for code that knows the
    repository but not the working directory)."""
    d = private_dir()
    return d if d.is_absolute() else root / d


def resolve(value: str | Path, base: Path | None = None) -> Path:
    """A path from config: a bare file name goes in *base* (default: the
    private folder); anything else is used as written (``~`` expanded)."""
    p = Path(str(value)).expanduser()
    if p.is_absolute() or len(p.parts) != 1:
        return p
    return (base if base is not None else private_dir()) / p


def config_path() -> Path:
    """The config to load: ``private/config.yaml``, or a root ``config.yaml``
    that has not been moved yet. Neither: the private one (for the error)."""
    new = private("config.yaml")
    if new.exists():
        return new
    old = Path("config.yaml")
    return old if old.exists() else new


def local_dir(config: Path) -> Path:
    """Where a config's bare ``local_file`` names go: beside it, except for a
    root config.yaml not moved yet, whose state still belongs in private/."""
    try:
        if config.resolve().parent == Path.cwd().resolve():
            return private_dir()
    except OSError:
        pass
    return config.parent


def migrate(root: Path | None = None, dest: Path | None = None,
            names: tuple[str, ...] = LEGACY_NAMES) -> list[str]:
    """Move the personal files left at *root* into *dest* (the private folder).

    Only a name *dest* lacks is moved, so it never overwrites anything and a
    second run does nothing. Nothing else is touched or deleted. Returns the
    names moved; the caller logs them once its log file is open (wawi.log is
    one of the files that move).
    """
    root = Path(".") if root is None else root
    dest = private_dir() if dest is None else dest
    if not dest.is_absolute():
        dest = root / dest
    moved: list[str] = []
    for name in names:
        old = root / name
        new = dest / name
        if not (old.exists() or old.is_symlink()) or new.exists() or new.is_symlink():
            continue
        try:
            dest.mkdir(parents=True, exist_ok=True)
            if old.is_dir() and not old.is_symlink():
                shutil.move(str(old), str(new))
            else:
                os.replace(old, new)
        except OSError as exc:
            log.warning("could not move %s into %s: %s", name, dest, exc)
            continue
        moved.append(name)
    return moved
