"""The ``pet:`` section of config.yaml, parsed once and validated."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wawi import paths as _paths
from wawi.pet import liveliness as _liveliness

DEFAULTS: dict[str, Any] = {
    # Needs, moods, desktop life and reactions. Off = the old mascot exactly.
    "enabled": True,
    # Who she is: a personality from the personalities/ folder (a name or a
    # path to a file): every number of her dog life (wawi/pet/dog.py). Empty:
    # the one pet.liveliness stands for.
    "personality": "",
    # The older knob, an alias: calm (perra_tranquila, the default), normal
    # (perra_atenta), playful (cachorra). See wawi/pet/liveliness.py.
    "liveliness": "calm",
    # Her needs, kept between runs (gitignored).
    "state_file": "pet_state.json",
    # Walking around the desktop, sitting on windows, climbing, falling.
    "wander": True,
    # Minutes between her wanderings (drawn at random in this range).
    "wander_every_min": [8, 25],
    # Ignored this long (no voice turn, no touch) and you are typing: she
    # comes over to watch. 0 turns it off.
    "lonely_after_min": 30,
    # Dropped in mid-air (a drag), she falls to whatever is below.
    "gravity_on_drop": True,
    # Speech bubbles for reactions.
    "bubbles": True,
    # Short sounds (barks) when a file for them exists in wawi/assets/sounds.
    "sounds": True,
    "rhythm": {"enabled": True, "night_start": 21, "night_end": 7},
    "requests": {"enabled": True, "interval_min": [120, 180]},
}


@dataclass(frozen=True)
class PetSettings:
    enabled: bool = True
    liveliness: str = "calm"
    personality: str = "perra_tranquila"
    state_file: Path | None = field(default_factory=lambda: _paths.private("pet_state.json"))
    wander: bool = True
    wander_every: tuple[float, float] = (8 * 60.0, 25 * 60.0)
    lonely_after: float = 30 * 60.0
    gravity_on_drop: bool = True
    bubbles: bool = True
    sounds: bool = True
    rhythm_enabled: bool = True
    night_start: float = 21.0
    night_end: float = 7.0
    requests_enabled: bool = True
    request_interval: tuple[float, float] = (7200.0, 10800.0)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PetSettings":
        """Parse the config section; anything malformed falls back to its
        default rather than stopping the app over a pet."""
        d = dict(DEFAULTS)
        d.update(raw or {})

        def num(v: Any, default: float) -> float:
            try:
                return max(0.0, float(v))
            except (TypeError, ValueError):
                return default

        every = d.get("wander_every_min")
        try:
            lo, hi = (float(every[0]), float(every[1]))  # type: ignore[index]
        except (TypeError, ValueError, IndexError, KeyError):
            lo, hi = 8.0, 25.0
        lo, hi = max(1.0, min(lo, hi)), max(1.0, max(lo, hi))
        state = d.get("state_file")
        rhythm = d.get("rhythm") if isinstance(d.get("rhythm"), dict) else {}
        requests = d.get("requests") if isinstance(d.get("requests"), dict) else {}
        intervals = requests.get("interval_min", [120, 180])
        try:
            request_interval = sorted((max(1.0, float(intervals[0])),
                                       max(1.0, float(intervals[1]))))
        except (TypeError, ValueError, IndexError, KeyError):
            request_interval = [120.0, 180.0]
        return cls(
            enabled=bool(d.get("enabled", True)),
            liveliness=_liveliness.parse(d.get("liveliness")),
            personality=(str(d.get("personality") or "").strip()
                         or _liveliness.personality_for(_liveliness.parse(d.get("liveliness")))),
            state_file=_paths.resolve(state) if state else None,
            wander=bool(d.get("wander", True)),
            wander_every=(lo * 60.0, hi * 60.0),
            lonely_after=num(d.get("lonely_after_min"), 30.0) * 60.0,
            gravity_on_drop=bool(d.get("gravity_on_drop", True)),
            bubbles=bool(d.get("bubbles", True)),
            sounds=bool(d.get("sounds", True)),
            rhythm_enabled=bool(rhythm.get("enabled", True)),
            night_start=num(rhythm.get("night_start"), 21.0) % 24,
            night_end=num(rhythm.get("night_end"), 7.0) % 24,
            requests_enabled=bool(requests.get("enabled", True)),
            request_interval=(request_interval[0] * 60, request_interval[1] * 60),
        )
