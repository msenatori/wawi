"""The ``instinct:`` section of config.yaml, parsed once and validated.

Everything here is cheap: one sample every ~30 s (two Quartz calls and one
NSWorkspace property), a JSON file of a few tens of kilobytes, and a decision
every couple of minutes that is a handful of comparisons. So it is on by
default; ``observe: false`` keeps the instincts that need no observation
(hunger) and drops everything that looks at the machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Every instinct the engine knows, with its default cooldown in minutes.
#: The engine's docstring explains what triggers each one.
INSTINCTS: dict[str, float] = {
    "welcome": 120.0,
    "focus_company": 45.0,
    "focus_done": 120.0,
    "break_ball": 90.0,
    "late_night": 180.0,
    "frenzy": 180.0,
    "hungry": 120.0,
    "lunch": 360.0,
    "noticed": 240.0,
}


@dataclass(frozen=True)
class InstinctSettings:
    #: Master switch. Off = no observer, no instincts, no voice tool.
    enabled: bool = True
    #: Look at the machine (front app, input timing). Off = only the
    #: instincts that come from her own needs.
    observe: bool = True
    #: Aggregates of the owner's days (gitignored, local only).
    state_file: Path | None = Path("instinct_state.json")
    #: Seconds between samples.
    sample_s: float = 30.0
    #: Seconds between instinct decisions.
    decide_s: float = 120.0
    #: Days of history kept (rolling).
    keep_days: int = 56
    #: No input for this long = away from the keyboard.
    idle_after_s: float = 120.0
    #: Never act during a call (meeting app in front, or reflex quiet mode
    #: says meeting). Bubbles are always held back in any quiet mode.
    quiet_meetings: bool = True
    #: At least this long between any two instincts (minutes).
    min_gap_min: float = 20.0
    #: Instincts enabled (all by default).
    instincts: tuple[str, ...] = tuple(INSTINCTS)
    #: Per-instinct cooldowns in minutes (defaults in :data:`INSTINCTS`).
    cooldowns: dict[str, float] = field(default_factory=lambda: dict(INSTINCTS))
    #: Owner overrides of the app → category table: {"Linear": "browser"}.
    categories: dict[str, str] = field(default_factory=dict)
    #: Say instinct bubbles aloud too (off: bubbles are silent by default).
    speak: bool = False

    @classmethod
    def from_dict(cls, raw: Any) -> "InstinctSettings":
        """Parse the section; anything malformed falls back to its default."""
        d = cls()
        if not isinstance(raw, dict):
            return d

        def num(key: str, default: float, lo: float, hi: float) -> float:
            try:
                return min(hi, max(lo, float(raw.get(key, default))))
            except (TypeError, ValueError):
                return default

        state = d.state_file
        if "state_file" in raw:
            state = Path(str(raw["state_file"])).expanduser() if raw["state_file"] else None

        enabled = list(INSTINCTS)
        listed = raw.get("instincts")
        if isinstance(listed, dict):
            # {"frenzy": false, "lunch": true}: switch single ones off.
            enabled = [n for n in INSTINCTS if bool(listed.get(n, True))]
        elif isinstance(listed, (list, tuple)):
            enabled = [n for n in INSTINCTS if n in {str(x) for x in listed}]

        cooldowns = dict(INSTINCTS)
        for name, value in (raw.get("cooldowns") or {}).items() \
                if isinstance(raw.get("cooldowns"), dict) else ():
            if name in cooldowns:
                try:
                    cooldowns[name] = min(7 * 24 * 60.0, max(1.0, float(value)))
                except (TypeError, ValueError):
                    pass

        cats: dict[str, str] = {}
        for app, cat in (raw.get("categories") or {}).items() \
                if isinstance(raw.get("categories"), dict) else ():
            if str(app).strip() and str(cat).strip():
                cats[str(app).strip().lower()] = str(cat).strip().lower()[:24]

        return cls(
            enabled=bool(raw.get("enabled", d.enabled)),
            observe=bool(raw.get("observe", d.observe)),
            state_file=state,
            sample_s=num("sample_s", d.sample_s, 5.0, 600.0),
            decide_s=num("decide_s", d.decide_s, 15.0, 3600.0),
            keep_days=int(num("keep_days", d.keep_days, 7, 365)),
            idle_after_s=num("idle_after_s", d.idle_after_s, 30.0, 3600.0),
            quiet_meetings=bool(raw.get("quiet_meetings", d.quiet_meetings)),
            min_gap_min=num("min_gap_min", d.min_gap_min, 0.0, 24 * 60.0),
            instincts=tuple(enabled),
            cooldowns=cooldowns,
            categories=cats,
            speak=bool(raw.get("speak", d.speak)),
        )
