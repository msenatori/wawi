"""Ta-te-tí on a little board next to her.

The owner is X, Wawi is O. She plays perfect minimax (memoised: ta-te-tí
has a few thousand positions), and the difficulty knob decides how often
she plays a random legal move instead: on ``facil`` often enough that she
loses, on ``dificil`` never (and then nobody beats her — a draw is the best
you get). Cells are numbered 0..8, row by row from the top-left.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

from wawi.games.core import DRAW, MAX_MOVE_WORDS, OWNER, WAWI, Game, Reply, norm, parse_number

X, O, EMPTY = "X", "O", " "
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
#: Chance of a random move instead of the best one, per level.
MISTAKES = {"facil": 0.55, "normal": 0.2, "dificil": 0.0}
#: Seconds she "thinks" before her move (the board shows her turn).
THINK = 0.9


def winner(board: str) -> str | None:
    """X, O, DRAW ("=") or None while the game goes on."""
    for a, b, c in LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return "=" if EMPTY not in board else None


def winning_line(board: str) -> tuple[int, int, int] | None:
    for line in LINES:
        a, b, c = line
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return line
    return None


def free(board: str) -> list[int]:
    return [i for i, c in enumerate(board) if c == EMPTY]


def put(board: str, i: int, mark: str) -> str:
    return board[:i] + mark + board[i + 1:]


@lru_cache(maxsize=None)
def minimax(board: str, to_move: str) -> int:
    """Value of *board* for O with *to_move* playing: +10-depth O wins,
    -10+depth X wins, 0 draw (quicker wins and slower losses preferred)."""
    w = winner(board)
    depth = 9 - board.count(EMPTY)
    if w == O:
        return 10 - depth
    if w == X:
        return depth - 10
    if w == "=":
        return 0
    nxt = X if to_move == O else O
    values = [minimax(put(board, i, to_move), nxt) for i in free(board)]
    return max(values) if to_move == O else min(values)


def best_moves(board: str, mark: str = O) -> list[int]:
    """Every move that is optimal for *mark*."""
    nxt = X if mark == O else O
    scored = [(minimax(put(board, i, mark), nxt), i) for i in free(board)]
    if not scored:
        return []
    target = max(s for s, _ in scored) if mark == O else min(s for s, _ in scored)
    return [i for s, i in scored if s == target]


def choose(board: str, level: str, rng: random.Random, mark: str = O) -> int:
    moves = free(board)
    if not moves:
        raise ValueError("board is full")
    if rng.random() < MISTAKES.get(level, 0.2):
        return rng.choice(moves)
    return rng.choice(best_moves(board, mark))


# Spoken positions. Order matters: two-word places first.
_ROW = {"arriba": 0, "superior": 0, "medio": 1, "centro": 1, "abajo": 2, "inferior": 2}
_COL = {"izquierda": 0, "medio": 1, "centro": 1, "derecha": 2}


def parse_cell(text: str) -> int | None:
    """'jugá en el centro', 'arriba a la izquierda', 'abajo', 'la 7'…"""
    t = norm(text)
    ws = t.split()
    if not ws or len(ws) > MAX_MOVE_WORDS:
        return None
    n = parse_number(t)
    if n is not None and 1 <= n <= 9:
        return n - 1
    row = next((_ROW[w] for w in ws if w in ("arriba", "superior", "abajo", "inferior")), None)
    col = next((_COL[w] for w in ws if w in ("izquierda", "derecha")), None)
    centre = any(w in ("centro", "medio") for w in ws)
    if row is None and col is None:
        return 4 if centre else None
    if row is None:
        row = 1
    if col is None:
        col = 1
    return row * 3 + col


WIN_LINES = ("¡Ta-te-tí! Gané yo.", "Tres en línea. Aprendé.", "Te encerré, ¿viste?")
LOSE_LINES = ("Uh, me ganaste. Revancha.", "Bueno, bueno… estaba en fácil.", "Ok. Esta vez sí.")
DRAW_LINES = ("Empate. Como siempre en el ta-te-tí.", "Nadie gana. Típico.")


class TicTacToe(Game):
    name = "tateti"
    title = "Ta-te-tí"

    def __init__(self, rng: random.Random | None = None, level: str = "normal",
                 owner_starts: bool = True, **kw: Any) -> None:
        super().__init__(rng, **kw)
        self.level = level if level in MISTAKES else "normal"
        self.board = EMPTY * 9
        self.turn = X if owner_starts else O
        self._think = THINK if not owner_starts else 0.0

    def start(self) -> Reply:
        if self.turn == X:
            self.line = "Ta-te-tí: vos sos la X. Tocá una casilla o decime dónde."
        else:
            self.line = "Ta-te-tí: empiezo yo. Vos sos la X."
            # Paw on her chin: she thinks her opening move.
            return Reply(self.line, "think_paw_chin")
        return Reply(self.line, "tilt")

    def state(self) -> dict[str, Any]:
        return {"board": self.board, "turn": self.turn, "line": winning_line(self.board),
                "level": self.level}

    def _end(self) -> Reply | None:
        w = winner(self.board)
        if w is None:
            return None
        self.over = True
        if w == O:
            self.line = self.pick(WIN_LINES)
            return Reply(self.line, "smug_win", result=WAWI, done=True, fun=25)
        if w == X:
            self.line = self.pick(LOSE_LINES)
            return Reply(self.line, "sad", result=OWNER, done=True, fun=20)
        self.line = self.pick(DRAW_LINES)
        return Reply(self.line, "shrug", result=DRAW, done=True, fun=20)

    def owner_move(self, cell: int) -> Reply | None:
        if self.over:
            return Reply("Ya terminó. ¿Otra?", "tilt")
        if self.turn != X:
            return Reply("Pará, que estoy pensando.", "side_eye")
        if not 0 <= cell < 9 or self.board[cell] != EMPTY:
            return Reply("Ahí no se puede.", "eyeroll")
        self.board = put(self.board, cell, X)
        end = self._end()
        if end is not None:
            return end
        self.turn = O
        self._think = THINK
        self.line = "Hmm… a ver…"
        return Reply(self.line, "think_paw_chin")

    def wawi_move(self) -> Reply:
        cell = choose(self.board, self.level, self.rng)
        self.board = put(self.board, cell, O)
        end = self._end()
        if end is not None:
            return end
        self.turn = X
        self.line = "Te toca."
        return Reply(self.line, None)

    def tick(self, dt: float, env: Any = None) -> Reply | None:
        if self.over or self.turn != O:
            return None
        self._think -= dt
        if self._think > 0.0:
            return None
        return self.wawi_move()

    def hear(self, text: str) -> Reply | None:
        cell = parse_cell(text)
        return None if cell is None else self.owner_move(cell)

    def click(self, what: str) -> Reply | None:
        if what.startswith("cell:"):
            try:
                return self.owner_move(int(what[5:]))
            except ValueError:
                return None
        return None
