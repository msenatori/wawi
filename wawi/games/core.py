"""What every game shares: what a game says back, and reading spoken Spanish.

A game is a small pure state machine. The owner acts through
:meth:`Game.hear` (a transcript), :meth:`Game.click` (a button, a cell) or
time (:meth:`Game.tick`); each returns a :class:`Reply` — the line to show
(and to say, when the owner spoke), the expression she makes, and when a
round or the whole game ends, who won. Nothing here touches AppKit or the
clock: the manager (:mod:`wawi.games.manager`) passes time in and turns
replies into bubbles and reactions.
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

#: Outcomes, always from the OWNER's side.
OWNER, WAWI, DRAW = "owner", "wawi", "draw"


@dataclass
class Reply:
    """What a game answers to one move."""

    text: str = ""
    #: A behaviour/clip she acts out (``party``, ``sad``, ``smug``,
    #: ``eyeroll``, ``shrug``, ``cheer``, ``tilt``…), or None.
    expr: str | None = None
    #: A round (or the match) just ended: who won it.
    result: str | None = None
    #: The whole game is over.
    done: bool = False
    #: Fun she got out of it (0..100 points on her ``fun`` need).
    fun: float = 0.0
    #: Energy it cost her (points, negative): fetching tires her out.
    energy: float = 0.0
    #: A short non-speech sound (``bark``, ``whine``) — a hint, a yelp.
    sound: str | None = None
    #: A second behaviour right after :attr:`expr` (her hand in piedra,
    #: papel o tijera, THEN how she takes the round).
    then: str | None = None


@dataclass
class GameEnv:
    """What the games know about this frame (built by the host from the
    pet's Env). Coordinates: AppKit global points, y up."""

    now: float = 0.0
    wall: float = 0.0
    #: Her ground point while she stands on her own; None by the panel.
    position: tuple[float, float] | None = None
    size: tuple[float, float] = (124.0, 110.0)
    screens: Sequence[Any] = ()
    windows: Sequence[Any] = ()
    cursor: tuple[float, float] | None = None
    key_idle: float = math.inf
    #: Doing something already (a voice turn, an act, a trip).
    busy: bool = False
    #: Quiet mode (meeting, full screen, quiet hours); None = unknown.
    quiet: bool | None = None
    fun: float = 80.0
    energy: float = 80.0
    mood: str = "content"
    #: The animation frames she has (empty: unknown, use the fallbacks).
    frames: frozenset[str] = frozenset()


#: Behaviours drawn with frames of their own: behaviour → (the frame it
#: needs, what to do without it — another behaviour, or None for nothing).
#: A half-drawn set never shows a stranger: no hand frame, no hand.
ART: dict[str, tuple[str, str | None]] = {
    "giggle": ("giggle_a", "tickle"),
    "smug_win": ("smug_win", "party"),
    "think_paw_chin": ("think_paw_chin", None),
    "cover_eyes": ("cover_eyes", "wag"),
    "rps_rock": ("rps_rock", None),
    "rps_paper": ("rps_paper", None),
    "rps_scissors": ("rps_scissors", None),
}


def art(behaviour: str | None, frames: frozenset[str] | set[str]) -> str | None:
    """*behaviour* if she has the art for it, else its fallback (see :data:`ART`)."""
    if behaviour is None or behaviour not in ART:
        return behaviour
    frame, fallback = ART[behaviour]
    return behaviour if frame in frames else fallback


@dataclass
class Button:
    id: str
    label: str


class Game:
    """Base class. Subclasses set :attr:`name` / :attr:`title` and override
    what they use; the defaults ignore everything."""

    name = "game"
    title = "Juego"
    #: Needs her body free on the desktop (not sitting by the panel).
    needs_body = False

    def __init__(self, rng: random.Random | None = None, **_: Any) -> None:
        self.rng = rng or random.Random()
        self.over = False
        #: The line on her game bubble right now.
        self.line = ""
        self.buttons: list[Button] = []

    def start(self) -> Reply:
        return Reply()

    def hear(self, text: str) -> Reply | None:
        """A transcript. None = not a move in this game (let the brain have it)."""
        return None

    def click(self, what: str) -> Reply | None:
        return None

    def tick(self, dt: float, env: Any) -> Reply | None:
        return None

    def stop(self) -> Reply:
        self.over = True
        return Reply("Bueno, lo dejamos acá.", "shrug", done=True)

    def state(self) -> dict[str, Any]:
        """Extra state the UI draws (a board, a ball…)."""
        return {}

    def bubble_ok(self) -> bool:
        """May her bubble show now? (Not while she hides: it would point
        right at her.)"""
        return True

    def pick(self, lines: tuple[str, ...] | list[str]) -> str:
        return self.rng.choice(list(lines))


# ---------------------------------------------------------------------------
# Reading what the owner said
# ---------------------------------------------------------------------------

def norm(text: str) -> str:
    """Lowercase, no accents, no punctuation, single spaces."""
    t = unicodedata.normalize("NFKD", str(text or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9ñ ]+", " ", t)
    return " ".join(t.split())


def words(text: str) -> list[str]:
    return norm(text).split()


#: Utterances longer than this are conversation, not a move: "piedra" inside
#: a long sentence about something else must not play a round.
MAX_MOVE_WORDS = 7

STOP_WORDS = (
    "basta", "no juego mas", "no quiero jugar", "terminamos", "termina el juego",
    "terminar el juego", "dejemos de jugar", "deja de jugar", "fin del juego", "me rindo",
    "chau juego", "no juguemos mas", "suficiente", "pará de jugar", "para de jugar",
)


def is_stop(text: str) -> bool:
    t = norm(text)
    return any(norm(s) in t for s in STOP_WORDS) or t in ("para", "listo basta", "stop")


_UNITS = {
    "cero": 0, "uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiun": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}
_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
         "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100}


def parse_number(text: str) -> int | None:
    """The first number in *text*, digits or Spanish words up to 100.

    >>> parse_number("cuarenta y dos"), parse_number("es el 7"), parse_number("hola")
    (42, 7, None)
    """
    t = norm(text)
    m = re.search(r"\b(\d{1,3})\b", t)
    if m:
        return int(m.group(1))
    ws = t.split()
    for i, w in enumerate(ws):
        if w in _TENS:
            value = _TENS[w]
            if value < 100 and i + 2 < len(ws) and ws[i + 1] == "y" and ws[i + 2] in _UNITS \
                    and _UNITS[ws[i + 2]] < 10:
                value += _UNITS[ws[i + 2]]
            return value
        if w in _UNITS and w not in ("un", "una"):
            return _UNITS[w]
    return None


def mentions(text: str, options: dict[str, tuple[str, ...]]) -> str | None:
    """Which key of *options* the text names (by any of its words), if
    exactly one does."""
    ws = set(words(text))
    hits = [k for k, alts in options.items() if any(a in ws for a in alts)]
    return hits[0] if len(hits) == 1 else None
