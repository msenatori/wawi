"""Local-time activity profile, independent of the desktop and clock source."""

from __future__ import annotations

from datetime import datetime

# Hour, sleep delay, idle-gap multiplier, sleeping-wake-gap multiplier,
# and weights for calm / playful / energetic behaviours.
SCHEDULE = (
    (0.0, 3.0, 2.0, 3.0, 2.2, 0.08, 0.0),
    (7.0, 9.0, 1.2, 1.5, 1.5, 0.5, 0.2),
    (9.0, 24.0, 0.7, 0.7, 0.7, 1.4, 1.3),
    (12.0, 32.0, 0.6, 0.7, 0.6, 1.8, 1.8),
    (14.0, 12.0, 1.2, 1.4, 1.4, 0.6, 0.4),
    (16.0, 26.0, 0.7, 0.8, 0.7, 1.7, 1.6),
    (19.0, 13.0, 1.0, 1.1, 1.2, 0.7, 0.4),
    (21.0, 4.0, 1.8, 2.4, 2.0, 0.12, 0.02),
    (24.0, 3.0, 2.0, 3.0, 2.2, 0.08, 0.0),
)
CALM = frozenset({"yawn", "stretch", "look_left", "look_right", "look_around",
                  "stare", "sniff"})
ENERGETIC = frozenset({"run", "tail_chase", "roll", "dig", "zoomies", "parachute",
                       "bird_chase", "jump_tries", "jump_flop", "jump", "hop",
                       "ladder"})
PLAY = frozenset({"wag", "scratch", "ball_bring", "play_chew", "play_ball"})


def _smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def profile(wall: float, *, enabled: bool = True, night_start: float = 21.0,
            night_end: float = 7.0) -> tuple[float, float, float, dict[str, float], bool]:
    """Return sleep delay, idle gap, wake gap, weights and night flag.

    The clock is supplied by the caller; interpolation prevents sharp changes
    across schedule boundaries. Night endpoints shift the evening/dawn rows.
    """
    hour = datetime.fromtimestamp(wall).hour + datetime.fromtimestamp(wall).minute / 60.0
    night = hour >= night_start or hour < night_end
    if not enabled:
        return 10.0, 1.0, 1.0, {}, night
    rows = list(SCHEDULE)
    rows[1] = (night_end, *rows[1][1:])
    rows[7] = (night_start, *rows[7][1:])
    rows.sort(key=lambda r: r[0])
    for a, b in zip(rows, rows[1:]):
        if a[0] <= hour < b[0]:
            u = _smooth((hour - a[0]) / max(0.01, b[0] - a[0]))
            values = tuple(x + (y - x) * u for x, y in zip(a[1:], b[1:]))
            break
    else:
        values = rows[-1][1:]
    sleep, idle, wake, calm, play, energy = values
    weights = {n: calm for n in CALM} | {n: energy for n in ENERGETIC}
    weights.update({name: play for name in PLAY})
    weights["moon_yawn"] = weights["late_night"] = max(0.0, 1.2 - energy) * 2.0
    weights["hungry_bowl"] = 0.0 if night else 1.0
    return sleep, idle, wake, weights, night
