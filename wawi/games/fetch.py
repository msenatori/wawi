"""Buscar la pelota: drag the ball, throw it, and she fetches it.

Physics-light and pure. The ball flies under gravity and bounces off the
edges of the screen it was thrown on (its *visible* frame: never under the
Dock or the menu bar, never off screen), rolls, and comes to rest on the
floor. She runs along that screen's floor to it, picks it up, trots back to
where she was and drops it at her feet. Every fetch costs her energy; after
``fetch_max`` throws (or running low) she is too tired and lies down.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

from wawi.games.core import Game, GameEnv, Reply

Point = tuple[float, float]
Rect = tuple[float, float, float, float]

GRAVITY = 1900.0
#: Velocity kept after a bounce (vertical) and on a wall (horizontal).
BOUNCE = 0.55
WALL = 0.6
#: Rolling friction on the floor, per second.
ROLL = 2.2
#: Below these speeds on the floor it is at rest.
REST_V = 12.0
#: Fastest throw taken from a flick of the mouse (points/s).
MAX_THROW = 2600.0
BALL_R = 13.0
#: Her running speed, and how fast she drops to the floor from a window.
RUN_SPEED = 520.0
FALL_SPEED = 900.0
GRAB_TIME = 0.55
DROP_TIME = 0.9
#: Energy each fetch costs (points of 100) and the floor below which she quits.
FETCH_ENERGY = -4.0
TIRED_BELOW = 18.0
#: Where her mouth is, as a fraction of her size from her ground point
#: (frames face LEFT; mirrored when she faces right).
MOUTH = (-0.30, 0.42)


def screen_for(screens: Sequence[Any], x: float, y: float) -> Rect | None:
    """The visible frame containing (x, y), else the nearest one."""
    best, best_d = None, math.inf
    for s in screens:
        vx, vy, vw, vh = getattr(s, "visible", s)
        cx = min(max(x, vx), vx + vw)
        cy = min(max(y, vy), vy + vh)
        d = math.hypot(cx - x, cy - y)
        if d < best_d:
            best, best_d = (vx, vy, vw, vh), d
    return best


@dataclass
class Ball:
    """The ball's centre and velocity, confined to one screen's visible frame."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    r: float = BALL_R
    resting: bool = True

    def floor(self, area: Rect) -> float:
        return area[1] + self.r

    def clamp(self, area: Rect) -> None:
        vx, vy, vw, vh = area
        self.x = min(max(self.x, vx + self.r), vx + vw - self.r)
        self.y = min(max(self.y, vy + self.r), vy + vh - self.r)

    def throw(self, vx: float, vy: float) -> None:
        speed = math.hypot(vx, vy)
        if speed > MAX_THROW:
            k = MAX_THROW / speed
            vx, vy = vx * k, vy * k
        self.vx, self.vy = float(vx), float(vy)
        self.resting = False

    def step(self, dt: float, area: Rect) -> None:
        """Advance *dt* seconds inside *area* (x, y, w, h — y up)."""
        if self.resting:
            self.clamp(area)
            return
        dt = max(0.0, min(dt, 0.05))
        vx0, vy0, vw, vh = area
        self.vy -= GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        left, right = vx0 + self.r, vx0 + vw - self.r
        bottom, top = vy0 + self.r, vy0 + vh - self.r
        if self.x < left:
            self.x, self.vx = left, abs(self.vx) * WALL
        elif self.x > right:
            self.x, self.vx = right, -abs(self.vx) * WALL
        if self.y > top:
            self.y, self.vy = top, -abs(self.vy) * BOUNCE
        if self.y <= bottom:
            self.y = bottom
            if abs(self.vy) < 140.0:
                self.vy = 0.0
            else:
                self.vy = abs(self.vy) * BOUNCE
            self.vx *= max(0.0, 1.0 - ROLL * dt)
            if self.vy == 0.0 and abs(self.vx) < REST_V:
                self.vx = 0.0
                self.resting = True


def throw_velocity(samples: Sequence[tuple[float, float, float]], window: float = 0.08) -> Point:
    """Release velocity from the last drag samples ``(t, x, y)``."""
    if len(samples) < 2:
        return (0.0, 0.0)
    t1, x1, y1 = samples[-1]
    old = [s for s in samples if t1 - s[0] <= window]
    t0, x0, y0 = old[0] if len(old) >= 2 else samples[-2]
    dt = t1 - t0
    if dt <= 1e-4:
        return (0.0, 0.0)
    return ((x1 - x0) / dt, (y1 - y0) / dt)


TIRED_LINES = ("Uf… ya no doy más. Me tiro un rato.", "Estoy muerta. Otro día seguimos.")
BRING_LINES = ("¡Acá está!", "¡Otra, otra!", "¡Tirala de nuevo!", "Traje la pelota. Obvio.")


class Fetch(Game):
    name = "pelota"
    title = "Buscar la pelota"
    needs_body = True

    def __init__(self, rng: random.Random | None = None, max_fetches: int = 6, **kw: Any) -> None:
        super().__init__(rng, **kw)
        self.max_fetches = max(1, int(max_fetches))
        self.fetches = 0
        #: "wait" (for her body), "ready", "held", "flying", "grab", "return",
        #: "drop", "tired".
        self.phase = "wait"
        self.ball: Ball | None = None
        self.area: Rect | None = None
        self.dog: list[float] | None = None
        self.home_x = 0.0
        self.facing_right = False
        self.loco: str | None = None
        self.clip: str | None = None
        self.t = 0.0
        self.size = (124.0, 110.0)
        self._samples: list[tuple[float, float, float]] = []

    def start(self) -> Reply:
        self.line = "¡A buscar la pelota! Agarrala y tirala con el mouse."
        return Reply(self.line, "wag")

    # -- the owner's hand -----------------------------------------------------------
    def grab(self, x: float, y: float, t: float) -> bool:
        if self.ball is None or self.phase in ("wait", "grab", "return", "drop", "tired"):
            return False
        self.phase = "held"
        self.ball.resting = True
        self.ball.x, self.ball.y = x, y
        self._samples = [(t, x, y)]
        return True

    def drag(self, x: float, y: float, t: float) -> None:
        if self.phase != "held" or self.ball is None:
            return
        self.ball.x, self.ball.y = x, y
        self._samples.append((t, x, y))
        del self._samples[:-12]

    def release(self, x: float, y: float, t: float) -> None:
        if self.phase != "held" or self.ball is None:
            return
        self.drag(x, y, t)
        vx, vy = throw_velocity(self._samples)
        # Thrown on whatever screen the hand let go over.
        if self._screens:
            self.area = screen_for(self._screens, x, y) or self.area
        self.ball.throw(vx, vy)
        self.phase = "flying"
        self.loco = None
        self.clip = "tilt"

    _screens: Sequence[Any] = ()

    # -- time ---------------------------------------------------------------------------
    def _floor(self) -> float:
        return self.area[1] if self.area else 0.0

    def _clamp_x(self, x: float) -> float:
        if self.area is None:
            return x
        vx, _vy, vw, _vh = self.area
        w = self.size[0]
        return min(max(x, vx + w / 2.0), vx + max(w / 2.0, vw - w / 2.0))

    def mouth(self) -> Point:
        assert self.dog is not None
        w, h = self.size
        dx = MOUTH[0] * w * (-1.0 if self.facing_right else 1.0)
        return (self.dog[0] + dx, self.dog[1] + MOUTH[1] * h)

    def _run_to(self, x: float, dt: float) -> bool:
        """Run along the floor towards *x*. True once there."""
        assert self.dog is not None
        x = self._clamp_x(x)
        d = x - self.dog[0]
        if abs(d) <= 4.0:
            self.dog[0] = x
            self.loco = None
            return True
        self.facing_right = d > 0
        step = RUN_SPEED * dt
        self.dog[0] += max(-step, min(step, d))
        self.loco = "trot"
        return False

    def tick(self, dt: float, env: Any = None) -> Reply | None:
        env = env if isinstance(env, GameEnv) else GameEnv()
        self._screens = env.screens
        self.size = env.size
        self.t += dt
        if self.phase == "wait":
            if env.position is None or not env.screens:
                return None
            x, y = env.position
            self.area = screen_for(env.screens, x, y)
            self.dog = [self._clamp_x(x), max(y, self._floor())]
            self.home_x = self.dog[0]
            self.ball = Ball(self.dog[0] - self.size[0] * 0.6, self._floor() + BALL_R)
            if self.ball.x < self.area[0] + 40:  # type: ignore[index]
                self.ball.x = self.dog[0] + self.size[0] * 0.6
            self.ball.clamp(self.area)  # type: ignore[arg-type]
            self.phase = "ready"
            return None
        if self.dog is None or self.ball is None or self.area is None:
            return None
        # She drops to the floor first (from a window's top edge).
        if self.dog[1] > self._floor():
            self.dog[1] = max(self._floor(), self.dog[1] - FALL_SPEED * dt)
            self.loco = "fall" if self.dog[1] > self._floor() else None
        if self.phase in ("flying",):
            self.ball.step(dt, self.area)
            self._run_to(self.ball.x, dt)
            if self.ball.resting and abs(self.dog[0] - self._clamp_x(self.ball.x)) <= 4.0:
                self.phase, self.t = "grab", 0.0
                self.loco, self.clip = None, "ball_bring"
            return None
        if self.phase == "grab":
            if self.t >= GRAB_TIME:
                self.phase = "return"
                self.clip = None
            return None
        if self.phase == "return":
            if self._run_to(self.home_x, dt):
                self.phase, self.t = "drop", 0.0
                self.facing_right = False
                self.clip = "wag"
                self.ball.x = self.dog[0] - self.size[0] * 0.55
                if self.ball.x < self.area[0] + 30:
                    self.ball.x = self.dog[0] + self.size[0] * 0.55
                self.ball.y = self._floor() + BALL_R
                self.ball.resting = True
                self.ball.clamp(self.area)
                self.fetches += 1
                tired = self.fetches >= self.max_fetches or env.energy + FETCH_ENERGY < TIRED_BELOW
                if tired:
                    self.phase = "tired"
                    self.over = True
                    self.clip = "yawn"
                    self.line = f"{self.pick(TIRED_LINES)} ({self.fetches} pelotas)"
                    return Reply(self.line, "yawn", done=True, fun=10, energy=FETCH_ENERGY)
                self.line = f"{self.pick(BRING_LINES)} Van {self.fetches}."
                return Reply(self.line, None, fun=6, energy=FETCH_ENERGY)
            else:
                bx, by = self.mouth()
                self.ball.x, self.ball.y = bx, by
            return None
        if self.phase == "drop" and self.t >= DROP_TIME:
            self.phase = "ready"
            self.clip = None
        return None

    def carried(self) -> bool:
        return self.phase in ("grab", "return")

    def state(self) -> dict[str, Any]:
        ball = None
        if self.ball is not None and self.phase not in ("wait", "tired"):
            ball = (self.ball.x, self.ball.y)
            if self.carried() and self.dog is not None:
                ball = self.mouth()
        return {
            "ball": ball,
            "ball_r": BALL_R,
            "grabbable": self.phase in ("ready", "flying", "drop"),
            "ground": (self.dog[0], self.dog[1]) if self.dog is not None else None,
            "loco": self.loco,
            "facing_right": self.facing_right,
            "clip": self.clip,
            # Picking it up ("grab") or bringing it back ("return"): the
            # host draws it in her mouth when she has the frames for it.
            "carried": self.phase if self.carried() else None,
            "fetches": self.fetches,
        }
