"""Escondidas: she hides behind a window with only her ears or tail out.

There is no clipping to compute: the host orders her window just BELOW the
window she hides behind, so that window covers her for real and only the
part sticking out past its edge shows. This module picks the spot:

* ``top`` — behind the window's top edge, the top of her head (ears) out;
* ``right`` / ``left`` — behind a side edge, her tail out (frames face left,
  so the tail is on her right; mirrored for the left edge).

With the peek frames (``peek_ears``, ``peek_tail_a/b``) she is drawn already
cut: only the head top, or only the rump and tail, with a straight edge where
the window's edge goes. :data:`ART` says where that edge sits in the frame,
and the spot puts it exactly on the window's edge. Without them (a half-drawn
set) her whole body stands behind the window with a slice of it out
(:data:`PLAIN`), and the window in front does the cutting.

A spot is good when all of her is on a screen, the part that peeks out is
not covered by any window in front, and the rest of her is entirely inside
the window she hides behind. Among good spots she prefers ones far from the
cursor (the owner is looking there). Pure: window lists in, a spot out.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from wawi.games.core import DRAW, OWNER, WAWI, Game, GameEnv, Reply

Rect = tuple[float, float, float, float]
Point = tuple[float, float]

#: How much of her peeks out: of her height over a top edge, of her width
#: past a side edge.
PEEK_TOP = 0.3
PEEK_SIDE = 0.28
MARGIN = 8.0
COUNT_TIME = 3.5
TRAVEL_TIME = 1.6


@dataclass(frozen=True)
class Peek:
    """Where the cut edge is and what shows, as fractions of her box (x from
    its left, y up from her ground line; frames face left)."""

    #: ``top``: y of the cut (the window's top edge), the visible x span and
    #: the visible top.
    top_cut: float
    top_x: tuple[float, float]
    top_y1: float
    #: ``right`` (``left`` mirrored): x of the cut (the window's side edge),
    #: the visible right end and top.
    side_cut: float
    side_x1: float
    side_y1: float


#: Her whole body behind the window, a slice of it out.
PLAIN = Peek(1.0 - PEEK_TOP, (0.0, 1.0), 1.0, 1.0 - PEEK_SIDE, 1.0, 1.0)
#: The peek frames, measured on the art (320x286 box): peek_ears' straight
#: bottom edge is at row 212 of 286, its ears span x 8..213 and reach row
#: 141; peek_tail_a/b's straight left edge is at x 151/154, the tail reaches
#: x 314 and row 92. Each cut is taken a hair inside the art, so the window
#: (she is ordered below it) hides the outline stubs past the edge.
ART = Peek(0.26, (0.025, 0.667), 0.507, 0.482, 0.981, 0.678)
#: The frame each kind of spot needs to use :data:`ART`.
ART_FRAMES = {"top": "peek_ears", "side": "peek_tail_a"}


def peek_for(kind: str, frames: Iterable[str]) -> Peek:
    """The geometry of a *kind* spot given the frames she has."""
    need = ART_FRAMES["top" if kind == "top" else "side"]
    return ART if need in frames else PLAIN


def _intersects(a: Rect, b: Rect) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _contains(outer: Rect, inner: Rect, slack: float = 0.5) -> bool:
    return (inner[0] >= outer[0] - slack and inner[1] >= outer[1] - slack
            and inner[0] + inner[2] <= outer[0] + outer[2] + slack
            and inner[1] + inner[3] <= outer[1] + outer[3] + slack)


def _visible_frames(screens: Sequence[Any]) -> list[Rect]:
    return [tuple(getattr(s, "visible", s)) for s in screens]  # type: ignore[misc]


@dataclass(frozen=True)
class Spot:
    wid: int
    kind: str  # top | right | left
    ground: Point
    facing_right: bool
    #: The part of her that shows (global rect).
    peek: Rect
    #: Her ground relative to the window's origin (to follow it if it moves).
    offset: Point
    owner: str = ""
    #: Drawn with the peek frames (:data:`ART`), not her whole body.
    art: bool = False
    #: The part of her box that must lie inside the window (global rect).
    hidden: Rect = (0.0, 0.0, 0.0, 0.0)


def spot_on(win: Any, kind: str, size: tuple[float, float], frac: float = 0.5,
            frames: Iterable[str] = ()) -> Spot:
    """The spot of *kind* on window *win*; *frac* slides it along the edge.

    The cut edge of her drawing (:class:`Peek`) lands exactly on the
    window's edge: its top edge for ``top``, its right/left edge for
    ``right``/``left``.
    """
    w, h = size
    x, y, ww, wh = win.rect
    top = y + wh
    g = peek_for(kind, frames)
    if kind == "top":
        lo, hi = x + w / 2.0 + MARGIN, x + ww - w / 2.0 - MARGIN
        gx = lo + (hi - lo) * frac
        gy = top - h * g.top_cut
        left = gx - w / 2.0
        peek = (left + g.top_x[0] * w, top, (g.top_x[1] - g.top_x[0]) * w,
                (g.top_y1 - g.top_cut) * h)
        hidden = (left, gy, w, g.top_cut * h)
        facing = False
    else:
        lo, hi = y + MARGIN, top - h - MARGIN
        gy = lo + (hi - lo) * frac
        out = (g.side_x1 - g.side_cut) * w
        if kind == "right":
            # Unmirrored: the cut is side_cut of her width from her left.
            edge = x + ww
            gx = edge + w / 2.0 - g.side_cut * w
            peek = (edge, gy, out, g.side_y1 * h)
            hidden = (gx - w / 2.0, gy, g.side_cut * w, h)
            facing = False
        else:
            # Mirrored about her centre line: the cut is side_cut from her right.
            edge = x
            gx = edge - w / 2.0 + g.side_cut * w
            peek = (edge - out, gy, out, g.side_y1 * h)
            hidden = (edge, gy, g.side_cut * w, h)
            facing = True
    return Spot(int(win.wid), kind, (gx, gy), facing, peek, (gx - x, gy - y),
                str(getattr(win, "owner", "") or ""), g is ART, hidden)


def valid(spot: Spot, win: Any, index: int, windows: Sequence[Any], size: tuple[float, float],
          screens: Sequence[Any]) -> bool:
    w, h = size
    gx, gy = spot.ground
    her = (gx - w / 2.0, gy, w, h)
    if not any(_contains(v, her) for v in _visible_frames(screens)):
        return False
    for front in windows[:index]:
        if _intersects(spot.peek, front.rect):
            return False
    # The rest of her: inside the window she hides behind.
    return _contains(win.rect, spot.hidden, slack=1.0)


def candidates(windows: Sequence[Any], screens: Sequence[Any],
               size: tuple[float, float], frames: Iterable[str] = ()) -> list[Spot]:
    w, h = size
    frames = frozenset(frames)
    out: list[Spot] = []
    for i, win in enumerate(windows):
        _x, _y, ww, wh = win.rect
        if ww < w * 1.6 or wh < h * 1.4:
            continue
        for kind in ("top", "right", "left"):
            for frac in (0.15, 0.5, 0.85):
                s = spot_on(win, kind, size, frac, frames)
                if valid(s, win, i, windows, size, screens):
                    out.append(s)
    return out


def choose_spot(windows: Sequence[Any], screens: Sequence[Any], size: tuple[float, float],
                rng: random.Random, cursor: Point | None = None,
                frames: Iterable[str] = ()) -> Spot | None:
    """A good hiding spot, or None when there is nowhere to hide."""
    spots = candidates(windows, screens, size, frames)
    if not spots:
        return None

    def score(s: Spot) -> float:
        far = 0.0
        if cursor is not None:
            far = math.hypot(s.ground[0] - cursor[0], s.ground[1] - cursor[1])
        return min(far, 1500.0) + rng.uniform(0.0, 400.0)

    return max(spots, key=score)


FOUND_LINES = ("¡Jiji! ¡Me encontraste!", "¡Ay! ¿Cómo me viste? Jiji.", "¡Noo! Jajaja, me encontraste.")


class HideAndSeek(Game):
    name = "escondidas"
    title = "Escondidas"
    needs_body = True

    def __init__(self, rng: random.Random | None = None, seconds: float = 90.0, **kw: Any) -> None:
        super().__init__(rng, **kw)
        self.seconds = float(seconds)
        #: count → travel → hidden → done
        self.phase = "count"
        self.t = 0.0
        self.spot: Spot | None = None
        self.ground: Point | None = None
        self._hinted = False

    def start(self) -> Reply:
        self.line = "¡Escondidas! Cerrá los ojos y contá hasta diez…"
        return Reply(self.line, "cover_eyes")

    def _place(self, env: GameEnv) -> bool:
        spot = choose_spot(env.windows, env.screens, env.size, self.rng, env.cursor,
                           env.frames)
        if spot is None:
            return False
        self.spot = spot
        self.ground = spot.ground
        return True

    def tick(self, dt: float, env: Any = None) -> Reply | None:
        env = env if isinstance(env, GameEnv) else GameEnv()
        if self.over:
            return None
        self.t += dt
        if self.phase == "count":
            if self.t >= COUNT_TIME and env.position is not None:
                if not self._place(env):
                    self.over = True
                    self.line = "No encuentro dónde esconderme… abrí alguna ventana."
                    return Reply(self.line, "shrug", result=DRAW, done=True)
                self.phase, self.t = "travel", 0.0
                self.line = ""
            return None
        if self.phase == "travel":
            if self.t >= TRAVEL_TIME:
                self.phase, self.t = "hidden", 0.0
            return None
        if self.phase == "hidden":
            spot = self.spot
            assert spot is not None
            win = next((w for w in env.windows if int(getattr(w, "wid", -1)) == spot.wid), None)
            if win is None:
                self.over = True
                self.line = "¡Me sacaste el escondite! Empate."
                return Reply(self.line, "shrug", result=DRAW, done=True, fun=10)
            # The window moved: she goes with it.
            self.ground = (win.rect[0] + spot.offset[0], win.rect[1] + spot.offset[1])
            if not self._hinted and self.t >= self.seconds * 0.5:
                self._hinted = True
                return Reply("", None, sound="bark")
            if self.t >= self.seconds:
                self.over = True
                where = f" atrás de {spot.owner}" if spot.owner else ""
                self.line = f"¡Gané! Estaba acá{where}."
                return Reply(self.line, "smug", result=WAWI, done=True, fun=20)
        return None

    def touched(self) -> Reply | None:
        """A click on her (only her peeking part can be clicked)."""
        if self.phase != "hidden" or self.over:
            return None
        self.over = True
        secs = int(self.t)
        self.line = f"{self.pick(FOUND_LINES)} ({secs} s)"
        return Reply(self.line, "giggle", result=OWNER, done=True, fun=25)

    def bubble_ok(self) -> bool:
        return self.phase == "count" or self.over

    def state(self) -> dict[str, Any]:
        hidden = self.phase == "hidden" and not self.over
        spot = self.spot
        return {
            "ground": self.ground if self.phase in ("hidden",) else None,
            "invisible": self.phase == "travel",
            "below": spot.wid if hidden and spot else None,
            "facing_right": spot.facing_right if hidden and spot else None,
            "peek": spot.kind if hidden and spot else None,
            "peek_art": bool(spot.art) if hidden and spot else False,
            # She covers her eyes while she counts.
            "hold": "cover_eyes" if self.phase == "count" and not self.over else None,
        }
