"""Who she is: every number of her dog life in one :class:`Personality`.

A personality is a plain YAML file in the repo's ``personalities/`` folder
(one file per personality, shareable): a name, a description, an author, a
version and the parameters of the dog model (wawi/pet/dog.py), grouped in
sections. The file is nested for people; here it is flattened, section and
key joined by ``_`` (``sleep: {deep_median_min: 20}`` is
``sleep_deep_median_min``).

Loading never fails: a missing key takes the default personality's value
(``perra_tranquila``), a bad value is clamped into its range (or dropped)
with a warning, an unknown personality falls back to the default. Choose
one with ``pet.personality`` (a name from the folder or a path to a file);
the older ``pet.liveliness`` maps calm / normal / playful onto the three
built-ins.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

log = logging.getLogger(__name__)

#: Where personalities live: the tracked ``personalities/`` folder at the
#: repo root. The single place to drop new ones.
FOLDER = Path(__file__).resolve().parents[2] / "personalities"
SEARCH_PATH: tuple[Path, ...] = (FOLDER,)
DEFAULT = "perra_tranquila"
#: pet.liveliness (and its Spanish names) → a built-in personality.
ALIASES = {
    "calm": "perra_tranquila", "tranquila": "perra_tranquila", "calma": "perra_tranquila",
    "normal": "perra_atenta", "atenta": "perra_atenta",
    "playful": "cachorra", "inquieta": "cachorra", "juguetona": "cachorra",
}

Range = tuple[float, float]

#: Every parameter: (default, low, high). A pair default means a [lo, hi]
#: range drawn uniformly (both ends clamped, then sorted). Defaults are
#: ``perra_tranquila``'s; the file says the same (a test checks).
SPEC: dict[str, tuple[Any, float, float]] = {
    # -- sleep (rules 2–4) -------------------------------------------------
    # A deep-sleep bout: lognormal around the median, clamped to [min, max].
    "sleep_deep_median_min": (20.0, 1.0, 180.0),
    "sleep_deep_sigma": (0.4, 0.0, 2.0),
    "sleep_deep_min_min": (15.0, 1.0, 180.0),
    "sleep_deep_max_min": (45.0, 1.0, 240.0),
    # Light doze on the way down (falling asleep) and between deep bouts.
    "sleep_doze_in_min": ((2.0, 8.0), 0.2, 60.0),
    "sleep_doze_between_min": ((1.0, 5.0), 0.2, 60.0),
    # After the head lift that ends a sleep, the chance she lies back down.
    "sleep_lie_back_down": (0.4, 0.0, 0.95),
    # Owner at the desk and she was just up: she falls asleep this much sooner.
    "sleep_latency_after_activity": (0.6, 0.1, 1.0),
    # -- waking up (rule 4) ------------------------------------------------
    "wake_head_lift_s": ((3.0, 10.0), 0.5, 30.0),
    "wake_stretch": (0.7, 0.0, 1.0),
    "wake_shake": (0.3, 0.0, 1.0),
    # Once up, the chance she walks to a new spot (near the owner).
    "wake_walk": (0.25, 0.0, 1.0),
    # -- resting, awake (rules 7–8) -----------------------------------------
    "rest_minutes": ((10.0, 20.0), 0.5, 240.0),
    "rest_after_active_min": ((3.0, 8.0), 0.5, 120.0),
    "rest_glance_every_min": ((3.0, 10.0), 0.5, 240.0),
    "rest_glance_s": ((1.0, 3.0), 0.3, 10.0),
    "rest_gesture_every_min": ((10.0, 20.0), 1.0, 480.0),
    # -- up and about (rule 1) ----------------------------------------------
    # At the end of a rest, the chance she gets up (× the daily rhythm).
    "active_chance": (0.7, 0.0, 1.0),
    "active_minutes": ((3.0, 7.0), 0.2, 60.0),
    "active_something_every_s": ((30.0, 90.0), 5.0, 1800.0),
    # -- habituation (rules 5–6) --------------------------------------------
    # Each exposure multiplies novelty by this; it recovers toward 1 with this
    # time constant (≈ fully back after three of them); below the threshold
    # she does not react.
    "habituation_drop": (0.5, 0.0, 1.0),
    "habituation_recover_min": (10.0, 0.5, 240.0),
    "habituation_threshold": (0.3, 0.0, 1.0),
    "habituation_ear_flick_every_s": (120.0, 5.0, 3600.0),
    # -- greeting the owner back (rule 10) -----------------------------------
    "greeting_look_after_min": (5.0, 0.5, 600.0),
    "greeting_get_up_after_min": (30.0, 1.0, 1200.0),
    "greeting_full_after_min": (120.0, 1.0, 2400.0),
    "greeting_look_wag_s": (3.0, 0.0, 30.0),
    "greeting_get_up_wag_s": ((5.0, 10.0), 0.0, 60.0),
    "greeting_full_s": ((10.0, 20.0), 0.0, 120.0),
    # -- daily rhythm (rule 11): × on the chance to get up, × on sleep bouts --
    "rhythm_morning_hours": ((8.0, 10.0), 0.0, 24.0),
    "rhythm_morning_active": (2.0, 0.0, 10.0),
    "rhythm_morning_sleep": (0.85, 0.1, 5.0),
    "rhythm_siesta_hours": ((12.0, 16.0), 0.0, 24.0),
    "rhythm_siesta_active": (0.6, 0.0, 10.0),
    "rhythm_siesta_sleep": (1.15, 0.1, 5.0),
    "rhythm_evening_hours": ((17.0, 22.0), 0.0, 24.0),
    "rhythm_evening_active": (2.0, 0.0, 10.0),
    "rhythm_evening_sleep": (0.85, 0.1, 5.0),
    # The night (pet.rhythm night_start..night_end): mostly sleep.
    "rhythm_night_active": (0.1, 0.0, 10.0),
    "rhythm_night_sleep": (1.4, 0.1, 5.0),
    "rhythm_night_rest": (0.4, 0.05, 5.0),
    # Nobody at the Mac: she gets up less and rests shorter.
    "rhythm_alone_active": (0.3, 0.0, 10.0),
    "rhythm_alone_rest": (0.6, 0.05, 5.0),
    # -- play / zoomies (rule 12) --------------------------------------------
    "play_per_day": (1, 0, 20),
    "play_cooldown_h": (8.0, 0.0, 48.0),
    "play_chance": (0.3, 0.0, 1.0),
    "play_evening_boost": (2.0, 0.0, 10.0),
    "play_after_rest_min": (60.0, 0.0, 1440.0),
    "play_seconds": ((5.0, 20.0), 1.0, 120.0),
    # -- asking for water / food / play (rule 13) ----------------------------
    "requests_every_h": ((1.0, 3.0), 0.1, 48.0),
    "requests_tries": ((2.0, 3.0), 1.0, 10.0),
    "requests_between_min": ((1.0, 3.0), 0.1, 60.0),
    # -- what the owner is doing (rule 15) -----------------------------------
    "context_every_min": ((30.0, 60.0), 1.0, 1440.0),
    "context_laptop_chance": (0.1, 0.0, 1.0),
    # -- where she rests (rule 9) --------------------------------------------
    # Window tops vs the floor (1.0), her own level (no climb, no ladder),
    # and how much nearness to the owner (cursor / front window) counts.
    "desktop_top_weight": (0.5, 0.0, 10.0),
    "desktop_same_level_weight": (1.5, 0.0, 10.0),
    "desktop_near_owner": (1.0, 0.0, 10.0),
    # Sitting on the front window's text, she moves at most this often.
    "desktop_cover_every_min": (10.0, 1.0, 1440.0),
}
_INTS = {"play_per_day"}


@dataclass(frozen=True)
class Personality:
    """See :data:`SPEC` for every field (default, range). Times carry their
    unit in the name (``_min``, ``_s``, ``_h``); pairs are [lo, hi] ranges."""

    name: str = DEFAULT
    description: str = ""
    author: str = ""
    version: str = "1"
    sleep_deep_median_min: float = 20.0
    sleep_deep_sigma: float = 0.4
    sleep_deep_min_min: float = 15.0
    sleep_deep_max_min: float = 45.0
    sleep_doze_in_min: Range = (2.0, 8.0)
    sleep_doze_between_min: Range = (1.0, 5.0)
    sleep_lie_back_down: float = 0.4
    sleep_latency_after_activity: float = 0.6
    wake_head_lift_s: Range = (3.0, 10.0)
    wake_stretch: float = 0.7
    wake_shake: float = 0.3
    wake_walk: float = 0.25
    rest_minutes: Range = (10.0, 20.0)
    rest_after_active_min: Range = (3.0, 8.0)
    rest_glance_every_min: Range = (3.0, 10.0)
    rest_glance_s: Range = (1.0, 3.0)
    rest_gesture_every_min: Range = (10.0, 20.0)
    active_chance: float = 0.7
    active_minutes: Range = (3.0, 7.0)
    active_something_every_s: Range = (30.0, 90.0)
    habituation_drop: float = 0.5
    habituation_recover_min: float = 10.0
    habituation_threshold: float = 0.3
    habituation_ear_flick_every_s: float = 120.0
    greeting_look_after_min: float = 5.0
    greeting_get_up_after_min: float = 30.0
    greeting_full_after_min: float = 120.0
    greeting_look_wag_s: float = 3.0
    greeting_get_up_wag_s: Range = (5.0, 10.0)
    greeting_full_s: Range = (10.0, 20.0)
    rhythm_morning_hours: Range = (8.0, 10.0)
    rhythm_morning_active: float = 2.0
    rhythm_morning_sleep: float = 0.85
    rhythm_siesta_hours: Range = (12.0, 16.0)
    rhythm_siesta_active: float = 0.6
    rhythm_siesta_sleep: float = 1.15
    rhythm_evening_hours: Range = (17.0, 22.0)
    rhythm_evening_active: float = 2.0
    rhythm_evening_sleep: float = 0.85
    rhythm_night_active: float = 0.1
    rhythm_night_sleep: float = 1.4
    rhythm_night_rest: float = 0.4
    rhythm_alone_active: float = 0.3
    rhythm_alone_rest: float = 0.6
    play_per_day: int = 1
    play_cooldown_h: float = 8.0
    play_chance: float = 0.3
    play_evening_boost: float = 2.0
    play_after_rest_min: float = 60.0
    play_seconds: Range = (5.0, 20.0)
    requests_every_h: Range = (1.0, 3.0)
    requests_tries: Range = (2.0, 3.0)
    requests_between_min: Range = (1.0, 3.0)
    context_every_min: Range = (30.0, 60.0)
    context_laptop_chance: float = 0.1
    desktop_top_weight: float = 0.5
    desktop_same_level_weight: float = 1.5
    desktop_near_owner: float = 1.0
    desktop_cover_every_min: float = 10.0


_META = ("name", "description", "author", "version")


def flatten(raw: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"sleep": {"deep_sigma": 0.4}}`` → ``{"sleep_deep_sigma": 0.4}``.
    The metadata keys stay as they are."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "_"))
        else:
            out[name] = value
    return out


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _clean(key: str, value: Any, where: str) -> Any:
    """The value for *key*, clamped into its range; None to drop it."""
    default, lo, hi = SPEC[key]

    def clamp(x: float) -> float:
        y = min(hi, max(lo, x))
        if y != x:
            log.warning("Personality %s: %s=%s out of [%s, %s], using %s",
                        where, key, x, lo, hi, y)
        return y

    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            single = _number(value)
            if single is None:
                log.warning("Personality %s: %s should be [lo, hi], ignored", where, key)
                return None
            value = (single, single)
        a, b = _number(value[0]), _number(value[1])
        if a is None or b is None:
            log.warning("Personality %s: %s should be two numbers, ignored", where, key)
            return None
        a, b = clamp(a), clamp(b)
        return (min(a, b), max(a, b))
    x = _number(value)
    if x is None:
        log.warning("Personality %s: %s should be a number, ignored", where, key)
        return None
    x = clamp(x)
    return int(round(x)) if key in _INTS else x


def from_dict(raw: dict[str, Any] | None, base: Personality | None = None,
              where: str = "?") -> Personality:
    """A personality from a (nested) dict over *base* (default: the built-in
    defaults). Unknown keys are warned about and ignored; nothing raises."""
    p = base or Personality()
    if not isinstance(raw, dict):
        if raw is not None:
            log.warning("Personality %s: not a mapping, using %s", where, p.name)
        return p
    flat = flatten(raw)
    changes: dict[str, Any] = {}
    for key, value in flat.items():
        if key in _META:
            changes[key] = str(value) if value is not None else ""
        elif key in SPEC:
            v = _clean(key, value, where)
            if v is not None:
                changes[key] = v
        else:
            log.warning("Personality %s: unknown key %s, ignored", where, key)
    p = replace(p, **changes)
    if p.sleep_deep_min_min > p.sleep_deep_max_min:
        p = replace(p, sleep_deep_min_min=p.sleep_deep_max_min)
    return p


def _read(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        log.warning("Could not read personality %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        log.warning("Personality %s is not a mapping", path)
        return None
    return data


def find(name: str, search: Sequence[Path] = SEARCH_PATH) -> Path | None:
    """The file for *name*: a path to a file, or ``<name>.yaml|.yml|.json``
    in the search folders (first hit wins)."""
    text = str(name or "").strip()
    if not text:
        return None
    direct = Path(text).expanduser()
    if direct.suffix in (".yaml", ".yml", ".json") and direct.is_file():
        return direct
    for folder in search:
        for ext in (".yaml", ".yml", ".json"):
            candidate = Path(folder) / f"{text}{ext}"
            if candidate.is_file():
                return candidate
    return None


def available(search: Sequence[Path] = SEARCH_PATH) -> list[str]:
    """The personality names found in the search folders."""
    names: list[str] = []
    for folder in search:
        try:
            files: Iterable[Path] = sorted(Path(folder).iterdir())
        except OSError:
            continue
        for f in files:
            if f.suffix in (".yaml", ".yml", ".json") and f.stem not in names:
                names.append(f.stem)
    return names


def resolve_name(value: object) -> str:
    """A config value (a personality name, a path, or a liveliness alias)
    to what :func:`load` takes; empty means the default."""
    text = str(value or "").strip()
    return ALIASES.get(text.lower(), text) or DEFAULT


def load(name: object = DEFAULT, search: Sequence[Path] = SEARCH_PATH) -> Personality:
    """The personality *name* (alias, name or path) over the default one.
    Never raises: anything missing or broken falls back with a warning."""
    wanted = resolve_name(name)
    base = Personality()
    default_path = find(DEFAULT, search)
    if default_path is not None:
        base = from_dict(_read(default_path), base, where=DEFAULT)
    if wanted == DEFAULT:
        return base
    path = find(wanted, search)
    if path is None:
        log.warning("No personality %r (have: %s); using %s", wanted,
                    ", ".join(available(search)) or "none", DEFAULT)
        return base
    p = from_dict(_read(path), base, where=path.stem)
    if p.name == base.name:
        p = replace(p, name=path.stem)
    return p


def field_names() -> list[str]:
    return [f.name for f in fields(Personality) if f.name not in _META]
