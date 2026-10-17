"""Adiviná el número, both ways round.

* :class:`GuessHers` — she thinks of a number from 1 to 100, the owner
  guesses out loud; she answers "más alto" / "más bajo" with a face to
  match, warm when close, and an eye-roll when a guess ignores what she
  already said.
* :class:`GuessYours` — the owner thinks of one; she finds it by binary
  search ("¿es 50?" … "más alto" / "más bajo" / "¡sí!") and notices when
  the answers stop adding up.
"""

from __future__ import annotations

import random
from typing import Any

from wawi.games.core import (
    MAX_MOVE_WORDS,
    OWNER,
    WAWI,
    Button,
    Game,
    Reply,
    norm,
    parse_number,
    words,
)

LO, HI = 1, 100
#: Guesses the owner gets before she wins.
MAX_TRIES = 10


def direction(text: str) -> str | None:
    """'up' (más alto / mayor), 'down' (más bajo / menor), 'yes' (¡ese!), or None."""
    t = norm(text)
    ws = t.split()
    if not ws or len(ws) > MAX_MOVE_WORDS:
        return None
    up = any(w in ws for w in ("alto", "mayor", "arriba"))
    down = any(w in ws for w in ("bajo", "menor", "abajo", "menos"))
    if up != down:
        return "up" if up else "down"
    if any(w in ws for w in ("si", "ese", "eso", "correcto", "exacto", "adivinaste", "bingo")):
        return "yes"
    return None


class GuessHers(Game):
    """She thinks, you guess."""

    name = "numero"
    title = "Adiviná el número"

    def __init__(self, rng: random.Random | None = None, secret: int | None = None,
                 **kw: Any) -> None:
        super().__init__(rng, **kw)
        self.secret = secret if secret is not None else self.rng.randint(LO, HI)
        self.tries = 0
        #: What she has told so far: the number is in [lo, hi].
        self.lo, self.hi = LO, HI

    def start(self) -> Reply:
        self.line = "Pensé un número del 1 al 100. ¡Adiviná! Decime un número."
        return Reply(self.line, "smug")

    def guess(self, n: int) -> Reply:
        if self.over:
            return Reply("Ya está. ¿Jugamos otra?", "tilt")
        if not LO <= n <= HI:
            return Reply("Del 1 al 100, dije.", "eyeroll")
        self.tries += 1
        if n == self.secret:
            self.over = True
            self.line = f"¡Sí! Era {n}. En {self.tries} intento{'s' if self.tries > 1 else ''}."
            if self.tries <= 4:
                self.line += " ¿Me espiaste?"
            return Reply(self.line, "party" if self.tries > 4 else "side_eye", result=OWNER,
                         done=True, fun=20)
        ignored = not (self.lo <= n <= self.hi)
        up = n < self.secret
        if up:
            self.lo = max(self.lo, n + 1)
        else:
            self.hi = min(self.hi, n - 1)
        if self.tries >= MAX_TRIES:
            self.over = True
            self.line = f"¡Se acabaron los intentos! Era {self.secret}."
            return Reply(self.line, "smug", result=WAWI, done=True, fun=20)
        word = "Más alto" if up else "Más bajo"
        if ignored:
            self.line = f"{word}… ¡te lo había dicho!"
            return Reply(self.line, "eyeroll", fun=2)
        close = abs(n - self.secret) <= 5
        self.line = f"{word}. {'¡Caliente, caliente!' if close else ''}".strip()
        return Reply(self.line, "wag" if close else ("paw_up" if up else "tilt"), fun=2)

    def hear(self, text: str) -> Reply | None:
        if len(words(text)) > MAX_MOVE_WORDS:
            return None
        n = parse_number(text)
        return None if n is None else self.guess(n)


class GuessYours(Game):
    """You think, she guesses (binary search, and a running commentary)."""

    name = "adivino"
    title = "Adivino tu número"

    def __init__(self, rng: random.Random | None = None, **kw: Any) -> None:
        super().__init__(rng, **kw)
        self.lo, self.hi = LO, HI
        self.tries = 0
        self.current = 0
        self.buttons = [Button("dir:down", "Más bajo"), Button("dir:yes", "¡Ese!"),
                        Button("dir:up", "Más alto")]

    def _ask(self) -> int:
        self.current = (self.lo + self.hi) // 2
        self.tries += 1
        return self.current

    def start(self) -> Reply:
        n = self._ask()
        self.line = f"Pensá un número del 1 al 100. Yo adivino. ¿Es {n}?"
        return Reply(self.line, "smug")

    def answer(self, d: str) -> Reply:
        if self.over:
            return Reply("Ya lo adiviné. ¿Otra?", "tilt")
        if d == "yes":
            self.over = True
            self.line = f"¡{self.current}! En {self.tries}. Te leo la mente."
            return Reply(self.line, "party" if self.tries <= 7 else "cheer", result=WAWI,
                         done=True, fun=20)
        if d == "up":
            self.lo = self.current + 1
        else:
            self.hi = self.current - 1
        if self.lo > self.hi:
            self.over = True
            self.line = "Eso no cierra… ¿me estás haciendo trampa?"
            return Reply(self.line, "eyeroll", result=OWNER, done=True, fun=10)
        n = self._ask()
        if self.lo == self.hi:
            self.line = f"Entonces es {n}, seguro. ¿Es {n}?"
            return Reply(self.line, "smug", fun=2)
        self.line = f"¿Es {n}?"
        return Reply(self.line, "tilt" if self.tries % 2 else None, fun=2)

    def hear(self, text: str) -> Reply | None:
        d = direction(text)
        if d is None:
            n = parse_number(text)
            if n is not None and n == self.current:
                d = "yes"
        return None if d is None else self.answer(d)

    def click(self, what: str) -> Reply | None:
        if what.startswith("dir:") and what[4:] in ("up", "down", "yes"):
            return self.answer(what[4:])
        return None
