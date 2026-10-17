"""Piedra, papel o tijera — and a dog who learns your habits.

People are bad at being random: they repeat a winner, switch after a loss,
cycle piedra → papel → tijera. :class:`PatternLearner` keeps a few small
frequency tables over the owner's recent moves (what they play in general,
after their last move, after their last two, after their last move given
how that round went), each decayed so recent habits weigh more. Every table
makes a prediction each round and keeps a score of how often it was right;
the best-scoring one speaks. Wawi then plays what beats the prediction —
but only part of the time: the rest she throws at random, so she stays
beatable (and against a truly random player nobody can do better than a
third each anyway). numpy only.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np

from wawi.games.core import DRAW, MAX_MOVE_WORDS, OWNER, WAWI, Button, Game, Reply, words

MOVES = ("piedra", "papel", "tijera")
#: What each move is called when spoken (plurals, the odd synonym).
SPOKEN: dict[int, tuple[str, ...]] = {
    0: ("piedra", "piedras", "roca", "puño"),
    1: ("papel", "papeles", "hoja"),
    2: ("tijera", "tijeras"),
}
EMOJI = ("✊", "✋", "✌️")
#: Her hand when she throws (frames; without them she just reacts).
HANDS = ("rps_rock", "rps_paper", "rps_scissors")


def beats(m: int) -> int:
    """The move that beats *m* (papel beats piedra, tijera beats papel…)."""
    return (m + 1) % 3


def outcome(owner: int, wawi: int) -> str:
    if owner == wawi:
        return DRAW
    return OWNER if owner == beats(wawi) else WAWI


class PatternLearner:
    """Predicts the owner's next move from their history. Pure numpy."""

    #: Per-update fading of every table: about the last 15-20 rounds count.
    DECAY = 0.92
    #: Fading of each predictor's score.
    SCORE_DECAY = 0.85
    #: Chance she trusts the prediction (scaled by how sure it is); the rest
    #: is a random throw.
    EXPLOIT = 0.8

    def __init__(self, rng: np.random.Generator | None = None, exploit: float | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.exploit = self.EXPLOIT if exploit is None else float(exploit)
        self.freq = np.zeros(3)
        self.after1 = np.zeros((3, 3))
        self.after2 = np.zeros((9, 3))
        self.after_res = np.zeros((9, 3))
        #: Score per predictor (freq, after1, after2, after_res).
        self.scores = np.zeros(4)
        self.history: list[tuple[int, int]] = []  # (owner, wawi)

    # -- predicting ---------------------------------------------------------------
    def _tables(self) -> list[np.ndarray | None]:
        h = self.history
        out: list[np.ndarray | None] = [self.freq]
        out.append(self.after1[h[-1][0]] if h else None)
        out.append(self.after2[h[-2][0] * 3 + h[-1][0]] if len(h) >= 2 else None)
        if h:
            o, w = h[-1]
            res = {OWNER: 0, WAWI: 1, DRAW: 2}[outcome(o, w)]
            out.append(self.after_res[o * 3 + res])
        else:
            out.append(None)
        return out

    def predict(self) -> tuple[np.ndarray, int | None]:
        """(probabilities of the owner's next move, the predictor used)."""
        best, best_i = None, None
        for i, row in enumerate(self._tables()):
            if row is None or row.sum() < 1e-6:
                continue
            if best_i is None or self.scores[i] > self.scores[best_i]:
                best, best_i = row, i
        if best is None:
            return np.full(3, 1.0 / 3.0), None
        p = (best + 0.1) / (best.sum() + 0.3)
        return p, best_i

    def choose(self) -> int:
        """Wawi's next move."""
        p, used = self.predict()
        confidence = float(p.max() - 1.0 / 3.0) * 1.5  # 0 (no idea) .. 1
        if used is not None and self.rng.random() < self.exploit * min(1.0, 0.35 + confidence):
            return beats(int(np.argmax(p)))
        return int(self.rng.integers(3))

    # -- learning ------------------------------------------------------------------
    def update(self, owner: int, wawi: int) -> None:
        for i, row in enumerate(self._tables()):
            if row is not None and row.sum() > 1e-6:
                hit = 1.0 if int(np.argmax(row)) == owner else -0.5
                self.scores[i] = self.scores[i] * self.SCORE_DECAY + hit
        h = self.history
        for table in (self.freq, self.after1, self.after2, self.after_res):
            table *= self.DECAY
        self.freq[owner] += 1.0
        if h:
            o, w = h[-1]
            self.after1[o, owner] += 1.0
            res = {OWNER: 0, WAWI: 1, DRAW: 2}[outcome(o, w)]
            self.after_res[o * 3 + res, owner] += 1.0
        if len(h) >= 2:
            self.after2[h[-2][0] * 3 + h[-1][0], owner] += 1.0
        h.append((owner, wawi))
        del h[:-200]

    # -- persistence -----------------------------------------------------------------
    def to_list(self, keep: int = 60) -> list[list[int]]:
        return [[o, w] for o, w in self.history[-keep:]]

    def load(self, raw: Any) -> None:
        """Replay a saved history: the tables are rebuilt from it."""
        if not isinstance(raw, list):
            return
        for item in raw:
            try:
                o, w = int(item[0]), int(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= o < 3 and 0 <= w < 3:
                self.update(o, w)


def parse_move(text: str) -> int | None:
    ws = words(text)
    if not ws or len(ws) > MAX_MOVE_WORDS:
        return None
    hits = {m for m, alts in SPOKEN.items() if any(a in ws for a in alts)}
    if len(hits) != 1:
        return None
    return hits.pop()


# What she says. Rioplatense, ironic, short enough for a bubble.
WIN_LINES = ("¡Te gané! Sos re predecible.", "Ja. Te vi venir.", "Otra para mí. ¿Seguro que querés seguir?",
             "Te leo como un libro abierto.")
LOSE_LINES = ("Bah, suerte de principiante.", "Me distraje con una mosca.", "Ok, esa te la dejé.",
              "Hmm. No me lo esperaba.")
DRAW_LINES = ("Empate. Pensamos igual, qué miedo.", "Empate. Otra.", "Mismo. Qué casualidad.")


class RPS(Game):
    name = "ppt"
    title = "Piedra, papel o tijera"

    def __init__(self, rng: random.Random | None = None, learner: PatternLearner | None = None,
                 first_to: int = 3, **kw: Any) -> None:
        super().__init__(rng, **kw)
        self.learner = learner or PatternLearner()
        self.first_to = max(1, int(first_to))
        self.owner = 0
        self.wawi = 0
        self.buttons = [Button(f"rps:{i}", f"{EMOJI[i]} {MOVES[i].capitalize()}") for i in range(3)]

    def start(self) -> Reply:
        self.line = f"¡Piedra, papel o tijera! Al que llega a {self.first_to}. Decime o tocá."
        return Reply(self.line, "wag")

    def score_text(self) -> str:
        return f"Vos {self.owner}, yo {self.wawi}"

    def play(self, owner_move: int) -> Reply:
        if self.over:
            return Reply("Ya terminamos. ¿Otra?", "tilt")
        mine = self.learner.choose()  # decided from history only, never from this move
        self.learner.update(owner_move, mine)
        res = outcome(owner_move, mine)
        said = f"{EMOJI[mine]} {MOVES[mine].capitalize()}."
        if res == WAWI:
            self.wawi += 1
            line, expr = self.pick(WIN_LINES), "smug"
        elif res == OWNER:
            self.owner += 1
            line, expr = self.pick(LOSE_LINES), self.rng.choice(("eyeroll", "sad"))
        else:
            line, expr = self.pick(DRAW_LINES), "shrug"
        if max(self.owner, self.wawi) >= self.first_to:
            self.over = True
            if self.wawi > self.owner:
                self.line = f"{said} ¡Gané el partido! {self.score_text()}."
                return Reply(self.line, HANDS[mine], result=WAWI, done=True, fun=30,
                             then="smug_win")
            self.line = f"{said} Ganaste vos… {self.score_text()}. La próxima no te la llevás."
            return Reply(self.line, HANDS[mine], result=OWNER, done=True, fun=25, then="sad")
        self.line = f"{said} {line} ({self.score_text()})"
        return Reply(self.line, HANDS[mine], fun=4, then=expr)

    def hear(self, text: str) -> Reply | None:
        m = parse_move(text)
        return None if m is None else self.play(m)

    def click(self, what: str) -> Reply | None:
        if what.startswith("rps:"):
            try:
                m = int(what[4:])
            except ValueError:
                return None
            if 0 <= m < 3:
                return self.play(m)
        return None


def simulate(learner: PatternLearner, owner_moves: Sequence[int]) -> dict[str, int]:
    """Play *owner_moves* against *learner*; tally outcomes (for tests and
    for tuning by hand)."""
    tally = {OWNER: 0, WAWI: 0, DRAW: 0}
    for m in owner_moves:
        w = learner.choose()
        learner.update(int(m), w)
        tally[outcome(int(m), w)] += 1
    return tally
