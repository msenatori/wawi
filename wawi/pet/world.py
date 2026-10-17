"""Her body on the desktop: where she can stand, and how she gets there.

Everything is in AppKit global coordinates (points, origin at the bottom-left
of the primary screen, y UP). Her position is her **ground point**: the
bottom-centre of the drawing, where her paws touch whatever she stands on.

The desktop is turned into **ledges**: horizontal surfaces she can stand on.
Each screen's floor (the bottom of its *visible* frame, so above the Dock)
and the top edge of every ordinary window — only the parts of that edge not
hidden behind a window in front of it, and only where there is headroom
below the menu bar. A ledge stores the range her CENTRE may take on it.

:class:`Body` moves her along a path of legs (walk, climb), falls with
gravity when nothing is under her (the window she sat on moved, closed or
was minimised), lands, and patrols back and forth for an emergency. It is a
plain state machine with ``dt`` coming in, so the tests drive it with
synthetic window lists and no screen at all.

She uses her little wooden :class:`Ladder` only to reach a real window's top
edge. She carries it to the window, climbs rung by rung and pulls it up after
her. Climbs without a window destination never deploy it.

Calm by design: speeds are capped (walking is slower than a cursor drifting),
and she never targets a spot that covers the cursor, the caret or the text
area of the frontmost window.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Sequence

Rect = tuple[float, float, float, float]  # x, y, w, h — y up
Point = tuple[float, float]

#: Window owners that are never furniture: system overlays and chrome.
IGNORED_OWNERS = frozenset({
    "Dock", "Window Server", "Control Center", "Notification Center", "Spotlight",
    "SystemUIServer", "screencaptureui", "Screenshot", "WindowManager", "Wallpaper",
    "loginwindow", "TextInputMenuAgent", "CursorUIViewService",
})
#: Windows smaller than this (points) are popovers, tooltips, HUDs.
MIN_WINDOW = (160.0, 90.0)
#: Below this opacity a window is an overlay, not something to sit on.
MIN_ALPHA = 0.3
#: Height of a standard title bar: the part of a window that is not content.
TITLE_BAR = 28.0
#: She may hang over the end of a ledge by this fraction of her width. Zero:
#: hanging over looked like her body floating on top of the next window. The
#: end of a ledge is a wall now, and she turns round at it.
OVERHANG = 0.0
#: She keeps this fraction of her width clear of a screen's sides. Right at
#: the edge she looked stuck there, half her head in the next monitor.
EDGE_MARGIN = 0.4


def intersects(a: Rect, b: Rect) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def inflate(r: Rect, d: float) -> Rect:
    return (r[0] - d, r[1] - d, r[2] + 2 * d, r[3] + 2 * d)


def contains(outer: Rect, inner: Rect, slack: float = 0.5) -> bool:
    return (inner[0] >= outer[0] - slack and inner[1] >= outer[1] - slack
            and inner[0] + inner[2] <= outer[0] + outer[2] + slack
            and inner[1] + inner[3] <= outer[1] + outer[3] + slack)


def her_rect(x: float, y: float, size: tuple[float, float]) -> Rect:
    """The box she covers standing at ground point (*x*, *y*)."""
    w, h = size
    return (x - w / 2.0, y, w, h)


@dataclass(frozen=True)
class Win:
    """An on-screen window, front-to-back order kept by the caller's list."""

    wid: int
    rect: Rect
    pid: int = 0
    owner: str = ""
    layer: int = 0
    alpha: float = 1.0

    @property
    def top(self) -> float:
        return self.rect[1] + self.rect[3]

    @property
    def content(self) -> Rect:
        """The window minus its title bar: where the text is."""
        x, y, w, h = self.rect
        return (x, y, w, max(0.0, h - TITLE_BAR))


@dataclass(frozen=True)
class Screen:
    frame: Rect
    visible: Rect


@dataclass(frozen=True)
class Ledge:
    """Somewhere to stand: her centre may be anywhere in [x0, x1] at height y.
    *wid* is the window it is the top of, None for a screen's floor."""

    x0: float
    x1: float
    y: float
    wid: int | None = None
    screen: int = 0

    def holds(self, x: float, y: float, tol: float = 2.5) -> bool:
        return self.x0 - 0.5 <= x <= self.x1 + 0.5 and abs(y - self.y) <= tol

    def clamp(self, x: float) -> float:
        return min(max(x, self.x0), self.x1)


# ---------------------------------------------------------------------------
# Reading the desktop (pure: raw dictionaries in, typed things out)
# ---------------------------------------------------------------------------

def cg_to_appkit(rect: Rect, primary_height: float) -> Rect:
    """A Quartz rect (origin top-left of the primary screen, y DOWN) in
    AppKit coordinates (origin bottom-left, y UP)."""
    x, y, w, h = rect
    return (x, primary_height - y - h, w, h)


def parse_windows(raw: Iterable[dict], own_pid: int, primary_height: float,
                  min_size: tuple[float, float] = MIN_WINDOW) -> list[Win]:
    """Turn ``CGWindowListCopyWindowInfo`` dictionaries into furniture.

    Keeps the front-to-back order. Drops our own windows, anything not on the
    normal layer (menu bar, Dock, overlays, screen savers), system owners,
    see-through and tiny windows. Bounds need no Screen Recording permission.
    """
    out: list[Win] = []
    for d in raw:
        try:
            if int(d.get("kCGWindowLayer", 0)) != 0:
                continue
            pid = int(d.get("kCGWindowOwnerPID", -1))
            if pid == own_pid:
                continue
            owner = str(d.get("kCGWindowOwnerName", "") or "")
            if owner in IGNORED_OWNERS:
                continue
            if float(d.get("kCGWindowAlpha", 1.0)) < MIN_ALPHA:
                continue
            b = d.get("kCGWindowBounds") or {}
            r = (float(b.get("X", 0)), float(b.get("Y", 0)),
                 float(b.get("Width", 0)), float(b.get("Height", 0)))
            if r[2] < min_size[0] or r[3] < min_size[1]:
                continue
            out.append(Win(int(d.get("kCGWindowNumber", 0)), cg_to_appkit(r, primary_height),
                           pid, owner, 0, float(d.get("kCGWindowAlpha", 1.0))))
        except (TypeError, ValueError, AttributeError):
            continue  # a malformed entry is skipped, never fatal
    return out


def frontmost(windows: Sequence[Win]) -> Win | None:
    """The front window (the list is front-to-back), or None."""
    return windows[0] if windows else None


def fullscreen_active(windows: Sequence[Win], screens: Sequence[Screen]) -> bool:
    """The front window fills a whole screen (a full-screen app, a video, a
    presentation): she keeps still and out of the way."""
    front = frontmost(windows)
    if front is None:
        return False
    for s in screens:
        fx, fy, fw, fh = s.frame
        x, y, w, h = front.rect
        if abs(x - fx) <= 2 and abs(y - fy) <= 2 and abs(w - fw) <= 2 and abs(h - fh) <= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Ledges
# ---------------------------------------------------------------------------

def _subtract(segs: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    out = []
    for s0, s1 in segs:
        if b <= s0 or a >= s1:
            out.append((s0, s1))
            continue
        if a > s0:
            out.append((s0, a))
        if b < s1:
            out.append((b, s1))
    return out


def ledges(screens: Sequence[Screen], windows: Sequence[Win],
           size: tuple[float, float]) -> list[Ledge]:
    """Every surface she can stand on, floors first.

    A window's top edge counts only where no window in front of it covers
    the edge (she would be sitting inside that one), only inside a screen,
    and only with room for her below the top of that screen's visible frame.
    """
    w, h = size
    hang = OVERHANG * w
    side = w / 2.0 + EDGE_MARGIN * w
    out: list[Ledge] = []
    for i, s in enumerate(screens):
        vx, vy, vw, _vh = s.visible
        if vw >= 2.0 * side:
            out.append(Ledge(vx + side, vx + vw - side, vy, None, i))
    for idx, win in enumerate(windows):
        x, _y, ww, _wh = win.rect
        top = win.top
        segs = [(x, x + ww)]
        for front in windows[:idx]:
            fx, fy, fw, fh = front.rect
            # Covers the edge, or would cover her sitting on it.
            if fy - 2.0 <= top <= fy + fh + h * 0.5:
                segs = _subtract(segs, fx, fx + fw)
            if not segs:
                break
        for i, s in enumerate(screens):
            vx, vy, vw, vh = s.visible
            if not (vy + h * 0.5 < top and top + h <= vy + vh):
                continue
            for s0, s1 in segs:
                a, b = max(s0, vx), min(s1, vx + vw)
                lo, hi = a + w / 2.0 - hang, b - w / 2.0 + hang
                # Never past the screen's own sides.
                lo, hi = max(lo, vx + side), min(hi, vx + vw - side)
                if hi - lo >= 4.0:
                    out.append(Ledge(lo, hi, top, win.wid, i))
    return out


def support(ledge_list: Sequence[Ledge], x: float, y: float, tol: float = 2.5) -> Ledge | None:
    """The ledge she is standing on at (*x*, *y*), or None (she is in the air)."""
    best = None
    for led in ledge_list:
        if led.holds(x, y, tol) and (best is None or led.y > best.y):
            best = led
    return best


def landing(ledge_list: Sequence[Ledge], x: float, y: float) -> Ledge | None:
    """Where she lands falling straight down from (*x*, *y*): the highest
    ledge at or below her under her centre; failing that (she is between
    screens), the nearest floor."""
    best = None
    for led in ledge_list:
        if led.x0 - 0.5 <= x <= led.x1 + 0.5 and led.y <= y + 2.5:
            if best is None or led.y > best.y:
                best = led
    if best is not None:
        return best
    floors = [led for led in ledge_list if led.wid is None]
    if not floors:
        return None
    return min(floors, key=lambda led: abs(led.clamp(x) - x) + abs(led.y - y) * 0.01)


def screen_index(screens: Sequence[Screen], x: float, y: float) -> int | None:
    for i, s in enumerate(screens):
        fx, fy, fw, fh = s.frame
        if fx <= x <= fx + fw and fy <= y <= fy + fh:
            return i
    return None


def on_screen(screens: Sequence[Screen], rect: Rect) -> bool:
    """Is *rect* entirely inside one screen's visible frame?"""
    return any(contains(s.visible, rect) for s in screens)


def clamp_to_screens(screens: Sequence[Screen], x: float, y: float,
                     size: tuple[float, float]) -> Point:
    """The nearest ground point that keeps all of her on one visible frame."""
    if not screens:
        return (x, y)
    w, h = size
    best, best_d = (x, y), math.inf
    for s in screens:
        vx, vy, vw, vh = s.visible
        side = w / 2.0 + EDGE_MARGIN * w
        cx = min(max(x, vx + side), vx + max(side, vw - side))
        cy = min(max(y, vy), vy + max(0.0, vh - h))
        d = math.hypot(cx - x, cy - y)
        if d < best_d:
            best, best_d = (cx, cy), d
    return best


def avoid_rects(front: Win | None, cursor: Point | None, caret: Rect | None,
                cursor_pad: float = 40.0, caret_pad: float = 30.0) -> list[Rect]:
    """What she must never cover: the frontmost window's text area, the
    cursor and the caret (each with a margin)."""
    out: list[Rect] = []
    if front is not None and front.content[3] > 0:
        out.append(front.content)
    if cursor is not None:
        out.append((cursor[0] - cursor_pad, cursor[1] - cursor_pad, 2 * cursor_pad, 2 * cursor_pad))
    if caret is not None:
        out.append(inflate(caret, caret_pad))
    return out


def choose_spot(ledge_list: Sequence[Ledge], here: Point, size: tuple[float, float],
                avoid: Sequence[Rect], rng: random.Random,
                screens: Sequence[Screen] = (), max_dist: float = 900.0,
                min_dist: float = 60.0) -> tuple[Ledge, float] | None:
    """Somewhere to wander to: a ledge and an x on it, not too far, not
    covering anything in *avoid*, fully on a screen. Nearer spots and window
    tops (more fun than the floor) are likelier. None when nowhere fits."""
    options: list[tuple[float, Ledge, float]] = []
    for led in ledge_list:
        for _ in range(3):
            x = rng.uniform(led.x0, led.x1) if led.x1 > led.x0 else led.x0
            d = math.hypot(x - here[0], led.y - here[1])
            if d > max_dist or d < min_dist:
                continue
            r = her_rect(x, led.y, size)
            if any(intersects(r, a) for a in avoid):
                continue
            if screens and not on_screen(screens, r):
                continue
            weight = (1.6 if led.wid is not None else 1.0) / (1.0 + d / 300.0)
            options.append((weight, led, x))
    if not options:
        return None
    total = sum(o[0] for o in options)
    r = rng.random() * total
    for weight, led, x in options:
        r -= weight
        if r < 0.0:
            return led, x
    return options[-1][1], options[-1][2]


def caret_spot(caret: Rect, size: tuple[float, float], screens: Sequence[Screen],
               avoid: Sequence[Rect] = (), margin: float = 26.0,
               line_back: float = 520.0) -> Point | None:
    """Where she sits to watch you type: NEXT to the caret, never on it.

    The caret line and the text just before it (*line_back* points to its
    left) are off limits, plus *margin*. Tried in order: right of the caret on
    its line, below-right, below-left, above-right. The first that is fully on
    a screen and clear of *avoid* wins; None if none is.
    """
    cx, cy, cw, ch = caret
    w, h = size
    ch = max(ch, 12.0)
    line = (cx - line_back, cy - margin * 0.5, line_back + cw + margin, ch + margin)
    blocked = [inflate(caret, margin), line, *avoid]
    candidates = (
        (cx + cw + margin + w / 2.0, cy - 2.0),
        (cx + cw + margin + w / 2.0, cy - margin - h),
        (cx - margin - w / 2.0, cy - margin - h),
        (cx + cw + margin + w / 2.0, cy + ch + margin),
    )
    for x, y in candidates:
        r = her_rect(x, y, size)
        if screens and not on_screen(screens, r):
            continue
        if any(intersects(r, b) for b in blocked):
            continue
        return (x, y)
    return None


# ---------------------------------------------------------------------------
# The body
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    """One piece of a path: ``walk`` to x, ``climb`` to y, ``perch`` (stay in
    the air where she is, e.g. by the caret, until told otherwise);
    ``ladder_set`` (lean the ladder up to *to* first) and ``ladder_fold``
    (put it away)."""

    kind: str
    to: float = 0.0
    speed: float = 0.0
    #: ladder_set: the wall is on her right (she faces it while climbing);
    #: None = the way she is facing.
    right: bool | None = None
    #: ladder_set: rung spacing, and how far it pokes above the top (points).
    rung: float = 0.0
    head: float = 0.0


#: The ladder: seconds to lean it up (it telescopes out meanwhile), to pull
#: it in, to sit with it folded small, and for the poof it vanishes in.
LADDER_SET = 1.8
LADDER_FOLD = 1.2
LADDER_HELD = 1.1
LADDER_POOF = 0.5
#: Its rung spacing, in her widths, and how far it pokes above the top it
#: leans on, in her heights (a real ladder sticks out past the roof).
LADDER_RUNG = 0.2
LADDER_HEAD = 0.3
#: While climbing, the part of each rung-to-rung step she spends getting a
#: grip before pulling herself up (0..1): she goes up rung by rung, not on a
#: lift.
LADDER_GRIP = 0.35


def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class LadderView(NamedTuple):
    """What to draw for the ladder now (world points, y up): its centre line
    *x*, the floor it stands on (*base*), its whole length *full*, and how
    much of it shows, from *lo* to *hi* above *base* (nothing when they
    meet); and the poof it vanishes in (*poof* 0..1, -1 for none) at
    (*poof_x*, *poof_y*)."""

    x: float
    base: float
    lo: float
    hi: float
    full: float
    poof: float = -1.0
    poof_x: float = 0.0
    poof_y: float = 0.0


@dataclass
class Ladder:
    """Her little wooden ladder, leaning against a side from *bottom* to
    *top* (and *head* above it). It is set from *home* (where she stood when
    she leant it up; she goes back that way off a perch), has the wall on
    her *right* or left, and lives through ``set`` → ``up`` → ``fold`` →
    ``held`` (folded small by her paws, part of her frame) → ``poof``.
    Pure: :meth:`advance` takes ``dt``."""

    x: float
    bottom: float
    top: float
    home: float
    right: bool
    rung: float = 24.0
    head: float = 33.0
    phase: str = "set"
    t: float = 0.0
    #: Where it folds towards (she stands there): the top or the bottom,
    #: and where she is (the poof goes off by her paws).
    fold_to: float | None = None
    fold_x: float | None = None

    @property
    def full(self) -> float:
        """Its whole length, top end included (points)."""
        return self.top - self.bottom + self.head

    def advance(self, dt: float) -> bool:
        """Move its animation on; False once the poof is over (it is gone)."""
        self.t += dt
        if self.phase == "set" and self.t >= LADDER_SET:
            self.phase, self.t = "up", 0.0
        elif self.phase == "fold" and self.t >= LADDER_FOLD:
            self.phase, self.t = "held", 0.0
        elif self.phase == "held" and self.t >= LADDER_HELD:
            self.phase, self.t = "poof", 0.0
        elif self.phase == "poof" and self.t >= LADDER_POOF:
            return False
        return True

    def fold(self, toward: float, at: float | None = None) -> None:
        """Pull it in towards *toward* (where she stands, at x *at*): it
        folds small by her paws, then poof."""
        if self.phase in ("set", "up"):
            self.phase, self.t = "fold", 0.0
            self.fold_to = toward
            self.fold_x = at

    def poof(self) -> None:
        """Gone at once (she was dragged off it, or fell): just the poof."""
        if self.phase != "poof":
            if self.phase in ("set", "up"):
                self.fold_to, self.fold_x = None, None  # where it stands
            self.phase, self.t = "poof", 0.0

    def view(self) -> LadderView:
        full = self.full
        span = self.top - self.bottom
        lo, hi = 0.0, full
        poof = -1.0
        short = min(full, 2.0 * self.rung)
        if self.phase == "set":
            # Out it comes, rung after rung, at a steady pace.
            e = min(1.0, self.t / (0.85 * LADDER_SET))
            if self.home >= self.top - 0.5 and span > 0.5:
                lo = span * (1.0 - e)  # let down from the edge she is on
            else:
                hi = short + (full - short) * e
        elif self.phase == "fold":
            e = _ease(self.t / LADDER_FOLD)
            to = self.home if self.fold_to is None else self.fold_to
            if to >= self.top - 0.5 and span > 0.5:
                lo = span * e  # pulled up after her
            else:
                hi = full - (full - short) * e  # folded down at its foot
        elif self.phase in ("held", "poof"):
            if self.fold_to is None:  # poofed where it stood
                py, px = self.bottom + min(full, 2.0 * self.rung) * 0.5, self.x
            else:
                py = self.fold_to + 0.15 * self.rung
                px = self.x if self.fold_x is None else self.fold_x
            lo = hi = 0.0
            if self.phase == "poof":
                poof = min(1.0, self.t / LADDER_POOF)
            return LadderView(self.x, self.bottom, lo, hi, full, poof, px, py)
        return LadderView(self.x, self.bottom, lo, hi, full, poof)

    def step_y(self, y: float) -> float:
        """Where she shows at true height *y* while climbing: rung by rung,
        a grip at each rung and then a pull up to the next."""
        span = self.top - self.bottom
        if span <= 0.0 or self.rung <= 0.0:
            return y
        n = max(1, round(span / self.rung))
        per = span / n
        p = min(float(n), max(0.0, (y - self.bottom) / per))
        k = math.floor(p)
        if k >= n:
            return self.top
        f = p - k
        g = _ease((f - LADDER_GRIP) / (1.0 - LADDER_GRIP))
        return self.bottom + (k + g) * per


@dataclass
class Body:
    """Her position and how it changes. Pure; ``update`` takes ``dt`` and
    the current ledges.

    ``mode`` is what the animator shows: ``rest``, ``walk``, ``trot``
    (patrol), ``climb``, ``fall``, ``land``, ``perch``; with the ladder also
    ``ladder_set`` (leaning it up), ``step_off`` (off its top onto the edge)
    and ``ladder_fold`` (putting it away).
    """

    x: float
    y: float
    mode: str = "rest"
    vy: float = 0.0
    facing_right: bool = False
    legs: list[Leg] = field(default_factory=list)
    #: Patrol: back and forth between these two x (an emergency run).
    patrol: tuple[float, float] | None = None
    #: The ledge she is on (None in the air or perched).
    ledge: Ledge | None = None
    land_left: float = 0.0
    paused: bool = False
    #: Under a parachute: the next fall is a slow float down.
    chute: bool = False
    #: She has the ladder art: plan_path climbs go up a ladder.
    ladders: bool = False
    #: The ladder standing now (also while it folds away and poofs).
    ladder: Ladder | None = None

    WALK_SPEED = 42.0
    #: Sinking speed under the parachute (points/s).
    CHUTE_FALL = 75.0
    TROT_SPEED = 130.0
    CLIMB_SPEED = 30.0
    GRAVITY = 1500.0
    MAX_FALL = 950.0
    LAND_TIME = 0.35

    @property
    def pos(self) -> Point:
        return (self.x, self.y)

    #: Modes that are "on her way" and turn into ``rest`` when she stops.
    GOING = ("walk", "trot", "climb", "step_off", "ladder_set", "ladder_fold")

    @property
    def moving(self) -> bool:
        return self.mode in self.GOING + ("fall", "land")

    @property
    def carrying_ladder(self) -> bool:
        """Walking to the foot of a climb, the ladder in her mouth."""
        legs = self.legs
        return self.mode == "walk" and bool(legs) and (
            legs[0].kind == "ladder_set"
            or (len(legs) > 1 and legs[0].kind == "walk" and legs[1].kind == "ladder_set"))

    @property
    def shown_y(self) -> float:
        """Her height as drawn: on the ladder she goes rung by rung."""
        lad = self.ladder
        if self.mode == "climb" and lad is not None and lad.phase == "up":
            return lad.step_y(self.y)
        return self.y

    def ladder_view(self) -> LadderView | None:
        return self.ladder.view() if self.ladder is not None else None

    @property
    def busy(self) -> bool:
        """On her way somewhere (or falling): do not start anything else."""
        return bool(self.legs) or self.patrol is not None or self.mode in ("fall", "land")

    def go(self, legs: list[Leg]) -> None:
        self.legs = list(legs)
        self.patrol = None
        if self.mode == "perch":
            self.mode = "rest"

    def stop(self) -> None:
        """Stop where she is (she may then fall, if that is mid-air)."""
        self.legs = []
        self.patrol = None
        if self.ladder is not None and self.mode != "perch":
            self.ladder.poof()
        if self.mode in self.GOING:
            self.mode = "rest"

    def start_patrol(self, a: float, b: float) -> None:
        self.legs = []
        self.patrol = (min(a, b), max(a, b))
        self.mode = "trot"

    def leave_perch(self) -> None:
        if self.mode == "perch":
            self.mode = "rest"
            self.ledge = None
            lad = self.ladder
            if lad is not None and lad.phase == "up":
                # Back down (or up) the ladder the way she came, then fold it.
                self.legs = [Leg("climb", lad.home), Leg("ladder_fold")]

    def place(self, x: float, y: float) -> None:
        """Put her somewhere (a drag): no motion, she re-checks support."""
        self.x, self.y = float(x), float(y)
        self.legs = []
        self.patrol = None
        self.vy = 0.0
        self.mode = "rest"
        self.ledge = None
        self.chute = False
        if self.ladder is not None:
            self.ladder.poof()

    def update(self, dt: float, ledge_list: Sequence[Ledge]) -> str | None:
        """Advance *dt* seconds. Returns an event: ``landed``, ``arrived``,
        ``fell`` (support vanished under her), or None."""
        dt = max(0.0, min(dt, 0.25))
        if self.ladder is not None and not self.ladder.advance(dt):
            self.ladder = None
        if self.mode == "land":
            self.land_left -= dt
            if self.land_left <= 0.0:
                self.mode = "rest"
                return "landed"
            return None
        if self.mode == "fall":
            return self._fall(dt, ledge_list)
        if self.mode == "perch":
            return None
        leg = self.legs[0] if self.legs and not self.paused else None
        airborne = leg is not None and leg.kind in ("climb", "shift", "perch")
        if not airborne:
            led = support(ledge_list, self.x, self.y)
            if led is None:
                self.mode = "fall"
                self.vy = 0.0
                self.ledge = None
                # The path ends here; a fresh plan comes after the landing.
                self.legs = []
                self.patrol = None
                if self.ladder is not None:
                    self.ladder.poof()
                return "fell"
            self.ledge = led
        if self.paused:
            if self.mode in self.GOING:
                self.mode = "rest" if self.ledge is not None else self.mode
            return None
        if self.patrol is not None:
            return self._patrol(dt)
        if leg is None:
            if self.mode in self.GOING:
                self.mode = "rest"
            return None
        if leg.kind in ("walk", "walk_off", "shift"):
            target = leg.to
            if leg.kind == "walk" and self.ledge is not None:
                target = self.ledge.clamp(leg.to)
            speed = leg.speed or self.WALK_SPEED
            d = target - self.x
            if abs(d) > 0.01:
                self.facing_right = d > 0
            if abs(d) <= speed * dt:
                self.x = target
                self.legs.pop(0)
            else:
                self.x += math.copysign(speed * dt, d)
            self.mode = "walk" if speed <= self.WALK_SPEED * 1.5 else "trot"
            if leg.kind == "shift":
                self.ledge = None
                if self.ladder is not None and self.ladder.phase == "up":
                    self.mode = "step_off"  # off the top of the ladder
        elif leg.kind == "ladder_set":
            lad = self.ladder
            if lad is None or lad.phase in ("fold", "poof"):
                lo, hi = sorted((self.y, leg.to))
                right = self.facing_right if leg.right is None else leg.right
                lad = self.ladder = Ladder(self.x, lo, hi, self.y, right,
                                           leg.rung or 24.0, leg.head)
            self.facing_right = lad.right
            self.mode = "ladder_set"
            if lad.phase != "set":
                self.legs.pop(0)
        elif leg.kind == "ladder_fold":
            lad = self.ladder
            if lad is None or lad.phase == "poof":
                self.legs.pop(0)
            else:
                lad.fold(self.y, self.x)
                self.mode = "ladder_fold"
                # Face it: pulling it up from the edge, or folding it at its foot.
                self.facing_right = lad.x > self.x if abs(lad.x - self.x) > 1.0 else lad.right
        elif leg.kind == "climb":
            speed = leg.speed or self.CLIMB_SPEED
            d = leg.to - self.y
            self.mode = "climb"
            self.ledge = None
            if self.ladder is not None and self.ladder.phase == "up":
                self.facing_right = self.ladder.right
            if abs(d) <= speed * dt:
                self.y = leg.to
                self.legs.pop(0)
            else:
                self.y += math.copysign(speed * dt, d)
        elif leg.kind == "perch":
            self.legs.pop(0)
            self.mode = "perch"
            self.ledge = None
            return "arrived"
        else:
            self.legs.pop(0)
        if not self.legs:
            if self.mode in self.GOING:
                self.mode = "rest"
            return "arrived"
        return None

    def _patrol(self, dt: float) -> str | None:
        a, b = self.patrol  # type: ignore[misc]
        if self.ledge is not None:
            a, b = self.ledge.clamp(a), self.ledge.clamp(b)
        if b - a < 8.0:
            self.mode = "rest"
            return None
        target = b if self.facing_right else a
        d = target - self.x
        step = self.TROT_SPEED * dt
        if abs(d) <= step:
            self.x = target
            self.facing_right = not self.facing_right
        else:
            self.x += math.copysign(step, d)
            self.facing_right = d > 0
        self.mode = "trot"
        return None

    def _fall(self, dt: float, ledge_list: Sequence[Ledge]) -> str | None:
        below = landing(ledge_list, self.x, self.y)
        if self.chute:
            # The canopy is open: a gentle, steady float down.
            self.vy = max(-self.CHUTE_FALL, self.vy - self.GRAVITY * dt)
        else:
            self.vy = max(-self.MAX_FALL, self.vy - self.GRAVITY * dt)
        ny = self.y + self.vy * dt
        if below is not None and ny <= below.y:
            # Between two screens she lands on the nearest floor, inside it.
            self.x = below.clamp(self.x)
            self.y = below.y
            self.vy = 0.0
            self.ledge = below
            self.mode = "land"
            self.land_left = self.LAND_TIME
            return None
        if below is None and ny < -10_000:  # nowhere at all: stop falling
            self.vy = 0.0
            self.mode = "rest"
            return "landed"
        self.y = ny
        return None


def plan_path(body: Body, target: Point, target_ledge: Ledge | None,
              ledge_list: Sequence[Ledge], size: tuple[float, float],
              perch: bool = False) -> list[Leg]:
    """Legs from where she is to *target* (a ground point).

    * Same height: walk.
    * Higher: walk to below it and climb — for a window's top edge, up the
      window's side, then a step across onto it. *perch* climbs straight up
      and stays there in the air (watching the caret).
    * Lower: walk off the end of her ledge nearest the target and let gravity
      do the rest; the caller re-plans after the landing.
    """
    tx, ty = target
    w, _h = size
    here = body.ledge or support(ledge_list, body.x, body.y)
    legs: list[Leg] = []
    if abs(ty - body.y) <= 2.5 and (here is None or here.x0 - 0.5 <= tx <= here.x1 + 0.5):
        legs.append(Leg("walk", tx))
    elif ty > body.y:
        cx = tx
        side = target_ledge is not None and target_ledge.wid is not None and not perch
        if side:
            left = target_ledge.x0 - w * 0.7  # type: ignore[union-attr]
            right = target_ledge.x1 + w * 0.7  # type: ignore[union-attr]
            cx = left if abs(left - body.x) <= abs(right - body.x) else right
        if here is not None:
            cx = here.clamp(cx)
        legs.append(Leg("walk", cx))
        if body.ladders and side:
            wall = target_ledge.x0 > cx if side else None  # type: ignore[union-attr]
            legs.append(ladder_leg(ty, size, wall))
        legs.append(Leg("climb", ty))
        if side:
            legs.append(Leg("shift", target_ledge.clamp(cx)))  # type: ignore[union-attr]
            if body.ladders:
                legs.append(Leg("ladder_fold"))  # pulled up after her
            legs.append(Leg("walk", tx))
        elif abs(cx - tx) > 1.0:
            legs.append(Leg("shift", tx))
    else:
        if here is not None and not perch:
            # Off the end on the target's side; gravity takes it from there.
            edge = here.x0 - w * 0.9 if tx < body.x else here.x1 + w * 0.9
            legs.append(Leg("walk", here.clamp(edge)))
            legs.append(Leg("walk_off", edge))
        else:
            legs.append(Leg("walk", tx))
            if perch:
                # A spot in the air below her: climb down to it.
                legs.append(Leg("climb", ty))
    if perch:
        legs.append(Leg("perch"))
    return legs


def ladder_leg(to: float, size: tuple[float, float], right: bool | None) -> Leg:
    """Lean the ladder up (or down) to height *to*, sized for her."""
    w, h = size
    return Leg("ladder_set", to, right=right, rung=LADDER_RUNG * w, head=LADDER_HEAD * h)


def up_and_down(body: Body, height: float, size: tuple[float, float]) -> list[Leg]:
    """Legacy side look: climb bare, never deploy a ladder without a window."""
    floor = body.y
    legs = [Leg("climb", floor + height), Leg("climb", floor)]
    return legs
