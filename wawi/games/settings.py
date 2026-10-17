"""The ``games:`` section of config.yaml, parsed once and validated."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    # Games against Wawi. Off = no menu, no proposals, no voice tool.
    "enabled": True,
    # Scores, the owner's rock-paper-scissors habits and which games/times she
    # learned the owner likes (gitignored).
    "state_file": "games_state.json",
    # She proposes a game on her own now and then.
    "propose": True,
    # Minutes between two proposals, at least.
    "propose_every_min": 45,
    # At most this many proposals a day.
    "max_proposals_per_day": 4,
    # A proposal nobody answers is a "no" after this long (seconds).
    "proposal_timeout_s": 25,
    # She only proposes when bored: fun (0..100) below this…
    "bored_below": 45,
    # …or right after the owner comes back from a break this long (minutes).
    "after_break_min": 10,
    # A key pressed in the last N seconds = the owner is focused: no proposal.
    "focus_typing_s": 45,
    # Nobody touched the keyboard for this long: nobody to play with.
    "away_after_min": 15,
    # Say proposals out loud too (still only when the voice may speak).
    "speak_proposals": False,
    # Ta-te-tí: facil | normal | dificil.
    "tictactoe_level": "normal",
    # Piedra, papel o tijera: first to this many wins takes the match.
    "rps_first_to": 3,
    # Fetch: throws before she is too tired to go on.
    "fetch_max": 6,
    # Hide and seek: seconds before she gives up hiding and comes out.
    "hide_seconds": 90,
    # Trivia: questions per game.
    "trivia_questions": 5,
}

LEVELS = ("facil", "normal", "dificil")


@dataclass(frozen=True)
class GamesSettings:
    enabled: bool = True
    state_file: Path | None = Path("games_state.json")
    propose: bool = True
    propose_every: float = 45 * 60.0
    max_per_day: int = 4
    proposal_timeout: float = 25.0
    bored_below: float = 45.0
    after_break: float = 10 * 60.0
    focus_typing: float = 45.0
    away_after: float = 15 * 60.0
    speak_proposals: bool = False
    tictactoe_level: str = "normal"
    rps_first_to: int = 3
    fetch_max: int = 6
    hide_seconds: float = 90.0
    trivia_questions: int = 5

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "GamesSettings":
        """Parse the section; anything malformed falls back to its default
        rather than stopping the app over a game."""
        d = dict(DEFAULTS)
        d.update(raw or {})

        def num(key: str, lo: float, hi: float) -> float:
            try:
                return min(hi, max(lo, float(d.get(key))))
            except (TypeError, ValueError):
                return float(DEFAULTS[key])

        level = str(d.get("tictactoe_level", "normal")).strip().lower()
        level = level.replace("á", "a").replace("í", "i")
        state = d.get("state_file")
        return cls(
            enabled=bool(d.get("enabled", True)),
            state_file=Path(str(state)).expanduser() if state else None,
            propose=bool(d.get("propose", True)),
            propose_every=num("propose_every_min", 1.0, 24 * 60.0) * 60.0,
            max_per_day=int(num("max_proposals_per_day", 0, 50)),
            proposal_timeout=num("proposal_timeout_s", 5.0, 600.0),
            bored_below=num("bored_below", 0.0, 100.0),
            after_break=num("after_break_min", 1.0, 24 * 60.0) * 60.0,
            focus_typing=num("focus_typing_s", 0.0, 3600.0),
            away_after=num("away_after_min", 1.0, 24 * 60.0) * 60.0,
            speak_proposals=bool(d.get("speak_proposals", False)),
            tictactoe_level=level if level in LEVELS else "normal",
            rps_first_to=int(num("rps_first_to", 1, 10)),
            fetch_max=int(num("fetch_max", 1, 50)),
            hide_seconds=num("hide_seconds", 15.0, 900.0),
            trivia_questions=int(num("trivia_questions", 1, 20)),
        )
