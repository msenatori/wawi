"""The games on screen: a bubble with buttons, the ta-te-tí board, the ball.

:class:`GamesHost` is what the panel (``wawi/panel.py``) talks to, from its
main-thread frame tick, through a handful of one-line hooks:

* ``wants_free()`` — fetch and hide-and-seek need her body on the desktop,
  so she leaves the panel's side while they last;
* ``before_pet(env)`` / ``after_pet(env, out, pet)`` — around the body's own
  tick: while a game moves her, the body is told she is being carried (so it
  neither wanders nor fights the game for her position), then the game's
  ground point, locomotion and expression replace the body's for the frame;
* ``touched()`` — a click on her (found you!);
* ``menu_item()`` — the "Jugar ▸" submenu.

Every window here is borderless and non-activating, like her bubble: they
never take focus from what the owner is doing. Nothing here raises into the
panel; a failing game window is logged and skipped.

The pure part — what a :class:`~wawi.games.manager.View` does to her body
this frame — is :func:`apply_to_body`, which the tests drive with a real
animator and no AppKit at all.
"""

from __future__ import annotations

import logging
from typing import Any

from wawi import mascot as _m
from wawi.games.core import GameEnv

log = logging.getLogger(__name__)

#: Without the peek frames (her whole body behind the window, a slice out):
#: over a top edge her ears up, now and then a flop…
PEEK_EARS = _m.Clip("peek_ears", (_m.Step("listening", 2.4), _m.Step("ear_flop", 0.5),
                                  _m.Step("listening", 1.8), _m.Step("listening_b", 0.6)),
                    loop=True, xfade=0.2)
#: …and behind a side edge her tail, wagging.
PEEK_TAIL = _m.Clip("peek_tail", _m.CLIPS["wag"].steps, loop=True, xfade=0.06)
#: With them: drawn already cut, the cut edge on the window's edge (see
#: wawi/games/hide.py, ART). No motion overlay: a bob or a tilt would lift
#: the cut edge off the window's edge.
PEEK_EARS_ART = _m.Clip("peek_ears_cut", (_m.Step("peek_ears", 1.0),), loop=True, xfade=0.1)
PEEK_TAIL_ART = _m.Clip("peek_tail_cut", (_m.Step("peek_tail_a", 0.17),
                                          _m.Step("peek_tail_b", 0.17)), loop=True, xfade=0.06)
#: Only peek_tail_a drawn: the tail still, rather than a stranger in between.
PEEK_TAIL_ONE = _m.Clip("peek_tail_cut", (_m.Step("peek_tail_a", 1.0),), loop=True, xfade=0.06)
#: Clips held by the game (re-asserted every frame, stopped when it lets go).
HELD = frozenset({PEEK_EARS.name, PEEK_TAIL.name, PEEK_EARS_ART.name, PEEK_TAIL_ART.name,
                  "cover_eyes"})
#: How often the quiet state is re-read for proposals (seconds).
QUIET_EVERY = 20.0


def peek_clip(kind: str, art: bool, frames: Any) -> Any:
    """The clip for peeking from a *kind* spot. *art*: the spot was placed
    for the peek frames (hide.py did the geometry for them)."""
    if kind == "top":
        return PEEK_EARS_ART if art and "peek_ears" in frames else PEEK_EARS
    if art and "peek_tail_a" in frames:
        return PEEK_TAIL_ART if "peek_tail_b" in frames else PEEK_TAIL_ONE
    return PEEK_TAIL


def ball_in_mouth(state: dict, frames: Any) -> bool:
    """Is the ball drawn in her frame right now (so its own window hides)?

    Picking it up: the ``ball_bring`` frame has it. Running back: the
    ``run_ball_*`` gallop has it, all four frames or none (a gallop that
    loses the ball every other step is worse than the ball riding along).
    """
    carried = state.get("carried")
    if carried == "grab":
        return "ball_bring" in frames
    if carried == "return":
        return state.get("loco") == "trot" and all(f in frames for f in _m.RUN_BALL_CYCLE)
    return False


def apply_to_body(view: Any, out: Any, anim: Any, last_clip: str | None) -> str | None:
    """Make her body do what the game says this frame.

    Sets *out*'s ground point (and hides her while she "travels" to a hiding
    spot), her locomotion, facing and held clip on *anim*. Returns the
    one-shot clip name last started, so it is started once, not every frame.
    """
    state = view.state or {}
    if not view.free:
        return None
    frames = getattr(anim, "frames", frozenset())
    ground = state.get("ground")
    if ground is not None:
        out.ground = (float(ground[0]), float(ground[1]))
    if state.get("invisible"):
        out.hidden = True
    if "loco" in state:
        loco = state.get("loco")
        if loco == "trot" and ball_in_mouth(state, frames):
            loco = "trot_ball"
        anim.set_locomotion(loco)
    facing = state.get("facing_right")
    if facing is not None:
        anim.facing_right = bool(facing)
    peek = state.get("peek")
    if peek is not None:
        clip = peek_clip(peek, bool(state.get("peek_art")), frames)
        if anim.act_name != clip.name:
            anim.act(clip, loop=True)
        return clip.name
    hold = state.get("hold")
    if hold and anim.has_act(hold):
        if anim.act_name != hold:
            anim.act(hold, loop=True)
        return hold
    if last_clip in HELD and anim.act_name == last_clip:
        anim.stop_act()
    clip_name = state.get("clip")
    if clip_name and clip_name != last_clip and anim.locomotion is None:
        anim.act(clip_name)
        return clip_name
    return clip_name if clip_name == last_clip else None


def build_env(env: Any, pet: Any, quiet: bool | None) -> GameEnv:
    """The games' view of this frame, from the body's Env and needs."""
    fun = energy = 80.0
    mood = "content"
    busy = False
    frames: frozenset[str] = frozenset()
    if pet is not None:
        try:
            snap = pet.needs.snapshot()
            fun, energy = float(snap.fun), float(snap.energy)
            mood = str(pet.mood)
            anim = pet.anim
            frames = frozenset(anim.frames)
            busy = (anim.key != "idle" or anim.acting or anim.locomotion is not None
                    or pet.director.active is not None)
        except Exception:
            log.debug("games: reading the body failed", exc_info=True)
    full = False
    try:
        from wawi.pet.world import fullscreen_active

        full = fullscreen_active(env.windows, env.screens)
    except Exception:
        pass
    return GameEnv(
        now=env.now, wall=env.wall, position=None if env.attached else env.position,
        size=env.size, screens=env.screens, windows=env.windows, cursor=env.cursor,
        key_idle=env.key_idle, busy=busy, quiet=True if full else quiet, fun=fun,
        energy=energy, mood=mood, frames=frames,
    )


try:  # AppKit, only in the real app ------------------------------------------------
    import objc  # type: ignore[import]
    from AppKit import (  # type: ignore[import]
        NSAttributedString,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSColor,
        NSEvent,
        NSFloatingWindowLevel,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSMakePoint,
        NSMakeRect,
        NSMenu,
        NSMenuItem,
        NSNormalWindowLevel,
        NSPanel,
        NSScreen,
        NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSObject  # type: ignore[import]

    _HAS_APPKIT = True
except Exception:  # pragma: no cover - not macOS / no PyObjC
    _HAS_APPKIT = False

PAPER = (1.0, 0.985, 0.94, 0.97)
INK = (0.16, 0.20, 0.42, 0.95)
PAW = (0.86, 0.45, 0.18, 1.0)
PAD = 9.0
TAIL = 9.0
MAX_W = 230.0
BTN_H = 22.0
BTN_GAP = 6.0
CELL = 44.0
BOARD_PAD = 8.0
NSWindowBelow = -1


def _rgba(c: tuple[float, float, float, float]) -> Any:
    return NSColor.colorWithSRGBRed_green_blue_alpha_(*c)


def _font(size: float = 12.5) -> Any:
    return NSFont.fontWithName_size_("Chalkboard SE", size) or NSFont.systemFontOfSize_(size)


def _attrs(size: float = 12.5, color: tuple[float, float, float, float] = INK) -> dict:
    return {NSFontAttributeName: _font(size), NSForegroundColorAttributeName: _rgba(color)}


if _HAS_APPKIT:

    class WawiGameWindow(NSPanel):  # type: ignore[misc]
        """Borderless, non-activating: a click never steals focus."""

        def canBecomeKeyWindow(self):  # noqa: N802
            return False

        def canBecomeMainWindow(self):  # noqa: N802
            return False

    class WawiGameBubbleView(NSView):  # type: ignore[misc]
        """Her game bubble: the line, and a row of buttons under it."""

        def initWithFrame_host_(self, frame, host):  # noqa: N802
            self = objc.super(WawiGameBubbleView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._host = host
            return self

        def isFlipped(self):  # noqa: N802
            return True

        def isOpaque(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def mouseDown_(self, event):  # noqa: N802
            try:
                p = self.convertPoint_fromView_(event.locationInWindow(), None)
                self._host._bubble_click(float(p.x), float(p.y))
            except Exception:
                log.debug("games: bubble click failed", exc_info=True)

        def drawRect_(self, _rect):  # noqa: N802
            try:
                self._host._paint_bubble(self.bounds())
            except Exception:
                log.debug("games: bubble drawing failed", exc_info=True)

    class WawiBoardView(NSView):  # type: ignore[misc]
        """The ta-te-tí board: a hand-drawn grid, X (you) and paw-orange O."""

        def initWithFrame_host_(self, frame, host):  # noqa: N802
            self = objc.super(WawiBoardView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._host = host
            return self

        def isFlipped(self):  # noqa: N802
            return True

        def isOpaque(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def mouseDown_(self, event):  # noqa: N802
            try:
                p = self.convertPoint_fromView_(event.locationInWindow(), None)
                col = int((p.x - BOARD_PAD) // CELL)
                row = int((p.y - BOARD_PAD) // CELL)
                if 0 <= col < 3 and 0 <= row < 3:
                    self._host.manager.click(f"cell:{row * 3 + col}")
            except Exception:
                log.debug("games: board click failed", exc_info=True)

        def drawRect_(self, _rect):  # noqa: N802
            try:
                self._host._paint_board(self.bounds())
            except Exception:
                log.debug("games: board drawing failed", exc_info=True)

    class WawiBallView(NSView):  # type: ignore[misc]
        """The ball: grab it, drag it, let go with a flick to throw it."""

        def initWithFrame_host_(self, frame, host):  # noqa: N802
            self = objc.super(WawiBallView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._host = host
            self._held = False
            return self

        def isOpaque(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def mouseDown_(self, _event):  # noqa: N802
            loc = NSEvent.mouseLocation()
            self._held = bool(self._host.manager.ball_grab(float(loc.x), float(loc.y)))

        def mouseDragged_(self, _event):  # noqa: N802
            if self._held:
                loc = NSEvent.mouseLocation()
                self._host.manager.ball_drag(float(loc.x), float(loc.y))
                self._host._place_ball_now()

        def mouseUp_(self, _event):  # noqa: N802
            if self._held:
                self._held = False
                loc = NSEvent.mouseLocation()
                self._host.manager.ball_release(float(loc.x), float(loc.y))

        def drawRect_(self, _rect):  # noqa: N802
            try:
                self._host._paint_ball(self.bounds())
            except Exception:
                log.debug("games: ball drawing failed", exc_info=True)

    class WawiGamesMenuTarget(NSObject):  # type: ignore[misc]
        def initWithHost_(self, host):  # noqa: N802
            self = objc.super(WawiGamesMenuTarget, self).init()
            if self is None:
                return None
            self._host = host
            return self

        def playGame_(self, sender):  # noqa: N802
            try:
                self._host.manager.start(str(sender.representedObject()))
            except Exception:
                log.debug("games: menu start failed", exc_info=True)

        def stopGame_(self, _sender):  # noqa: N802
            self._host.manager.stop()

        def proposeGame_(self, _sender):  # noqa: N802
            self._host.manager.propose_now()

        def scoreLine_(self, _sender):  # noqa: N802
            pass


def _make_window(w: float, h: float, view_cls: Any, host: Any, level: int) -> tuple[Any, Any]:
    frame = NSMakeRect(0.0, 0.0, w, h)
    style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
    window = WawiGameWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, style, NSBackingStoreBuffered, False)
    window.setOpaque_(False)
    window.setBackgroundColor_(NSColor.clearColor())
    window.setHasShadow_(False)
    window.setIgnoresMouseEvents_(False)
    window.setHidesOnDeactivate_(False)
    window.setReleasedWhenClosed_(False)
    window.setLevel_(level)
    window.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorStationary
        | NSWindowCollectionBehaviorFullScreenAuxiliary)
    view = view_cls.alloc().initWithFrame_host_(frame, host)
    window.setContentView_(view)
    return window, view


class GamesHost:
    """See the module docstring. Main thread only."""

    def __init__(self, panel: Any, manager: Any) -> None:
        self.panel = panel
        self.manager = manager
        self.view: Any = None
        self._last_clip: str | None = None
        self._below: int | None = None
        self._quiet: bool | None = None
        self._quiet_at = -1e18
        self._bubble = None
        self._bubble_view = None
        self._bubble_text: str | None = None
        self._bubble_buttons: list[Any] = []
        self._button_rects: list[tuple[str, tuple[float, float, float, float]]] = []
        self._board = None
        self._board_view = None
        self._board_state: Any = None
        self._ball = None
        self._ball_view = None
        self._ball_pos: tuple[float, float] | None = None
        self._menu_target = None
        self._score_item = None
        self._was_free = False

    @classmethod
    def create(cls, panel: Any) -> "GamesHost | None":
        from wawi import games

        manager = games.get()
        if manager is None or not _HAS_APPKIT:
            return None
        return cls(panel, manager)

    # -- hooks from the panel -------------------------------------------------------------------
    def wants_free(self) -> bool:
        return bool(self.manager.wants_free())

    def before_pet(self, env: Any) -> None:
        """While a game moves her, the body is told she is being carried."""
        self.manager.body_ok = True
        if self.view is not None and self.view.free and self.view.state.get("ground") is not None:
            env.dragging = True

    def after_pet(self, env: Any, out: Any, pet: Any, dt: float) -> None:
        if env.now - self._quiet_at >= QUIET_EVERY:
            self._quiet_at = env.now
            try:
                from wawi import reflexes

                self._quiet = reflexes.is_quiet()
            except Exception:
                self._quiet = None
        genv = build_env(env, pet, self._quiet)
        view = self.manager.tick(dt, genv, pet.needs if pet is not None else None)
        self.view = view
        anim = self.panel._mascot_animator
        if view.game or view.free:
            # A game is on: her idle activities wait (wawi/mascot.ACTIVITIES).
            anim.hold_activities = True
        if view.free and not env.attached:
            self._last_clip = apply_to_body(view, out, anim, self._last_clip)
            self._was_free = True
        elif self._was_free:
            # The game let go of her: her own life takes over again.
            self._was_free = False
            # Only her held clip: a reaction that just took the stage (the
            # giggle when found) plays on.
            if self._last_clip in HELD and anim.act_name == self._last_clip:
                anim.stop_act()
            self._last_clip = None
            anim.set_locomotion(None)
        if view.bubble:
            out.bubble = None  # one bubble at a time: the game's
        self._order_below(view.state.get("below") if view.free else None, env.attached)
        self._show_bubble(view.bubble if not out.hidden else None, view.buttons)
        self._show_board(view.state if view.game == "tateti" else None)
        ball = view.state.get("ball") if view.game == "pelota" else None
        if ball is not None and view.free and ball_in_mouth(view.state, anim.frames):
            ball = None  # in her mouth, in the frame: back on the floor at the drop
        self._show_ball(ball)

    def tick_alone(self, dt: float) -> None:
        """No body (``pet.enabled: false``): the voice and board games still
        run, with her bubble and board; nothing that needs her body."""
        import time

        self.manager.body_ok = False
        try:
            view = self.manager.tick(dt, GameEnv(now=time.monotonic(), wall=time.time(), busy=True))
        except Exception:
            log.debug("games: tick failed", exc_info=True)
            return
        self.view = view
        self._show_bubble(view.bubble, view.buttons)
        self._show_board(view.state if view.game == "tateti" else None)

    def touched(self) -> bool:
        return bool(self.manager.touched())

    def close(self) -> None:
        for w in (self._bubble, self._board, self._ball):
            if w is not None:
                try:
                    w.orderOut_(None)
                    w.close()
                except Exception:
                    pass
        self._bubble = self._board = self._ball = None
        try:
            self.manager.close()
        except Exception:
            log.debug("games: close failed", exc_info=True)

    # -- the menu ------------------------------------------------------------------------------------------
    def menu_item(self) -> Any:
        """"Jugar ▸" with every game, propose, stop and the score."""
        from wawi.games.manager import GAMES

        target = WawiGamesMenuTarget.alloc().initWithHost_(self)
        self._menu_target = target  # menus hold their targets weakly
        sub = NSMenu.alloc().initWithTitle_("Jugar")
        for key, (_cls, title, _aliases) in GAMES.items():
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "playGame:", "")
            item.setTarget_(target)
            item.setRepresentedObject_(key)
            sub.addItem_(item)
        sub.addItem_(NSMenuItem.separatorItem())
        for title, sel in (("Proponeme un juego", "proposeGame:"), ("Terminar el juego", "stopGame:")):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, sel, "")
            item.setTarget_(target)
            sub.addItem_(item)
        score = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("", "scoreLine:", "")
        score.setTarget_(target)
        score.setEnabled_(False)
        sub.addItem_(score)
        self._score_item = score
        top = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Jugar", None, "")
        top.setSubmenu_(sub)
        self.refresh_menu()
        return top

    def refresh_menu(self) -> None:
        if self._score_item is not None:
            try:
                self._score_item.setTitle_(self.manager.score_line())
            except Exception:
                pass

    # -- her window's z-order (hide and seek) ---------------------------------------------------------
    def _order_below(self, wid: int | None, attached: bool) -> None:
        if wid == self._below:
            return
        window = self.panel._mascot_window
        self._below = wid
        if window is None:
            return
        try:
            if wid is not None:
                window.setLevel_(NSNormalWindowLevel)
                window.orderWindow_relativeTo_(NSWindowBelow, int(wid))
            elif not attached:
                window.setLevel_(NSFloatingWindowLevel)
                window.orderFrontRegardless()
        except Exception:
            log.debug("games: could not order her window", exc_info=True)

    # -- geometry of her ------------------------------------------------------------------------------------
    def _head(self) -> tuple[float, float, Any] | None:
        """(head x, top of her head y, screen visible frame) in global points."""
        p = self.panel
        window = p._mascot_window
        if window is None:
            return None
        f = window.frame()
        sw, sh = p._mascot_sprite_size
        pose = p._mascot_pose
        head_x = f.origin.x + f.size.width / 2.0 + pose.dx + (_m.HEAD_UV[0] - 0.5) * sw
        top = f.origin.y + p._mascot_ground + pose.dy + sh * 0.92
        screen = window.screen() or NSScreen.mainScreen()
        return head_x, top, (screen.visibleFrame() if screen is not None else None)

    # -- the bubble -------------------------------------------------------------------------------------------
    def _show_bubble(self, text: str | None, buttons: list[Any]) -> None:
        try:
            if not text:
                if self._bubble is not None and self._bubble_text is not None:
                    self._bubble_text = None
                    self._bubble.orderOut_(None)
                return
            if self._bubble is None:
                self._bubble, self._bubble_view = _make_window(
                    120.0, 40.0, WawiGameBubbleView, self, NSFloatingWindowLevel + 1)
            labels = [(b.id, b.label) for b in buttons]
            changed = text != self._bubble_text or labels != [(b.id, b.label)
                                                             for b in self._bubble_buttons]
            self._bubble_text, self._bubble_buttons = text, list(buttons)
            attrs = _attrs()
            tr = NSAttributedString.alloc().initWithString_attributes_(
                text, attrs).boundingRectWithSize_options_((MAX_W, 600.0), 1)
            widths = []
            for _id, label in labels:
                s = NSAttributedString.alloc().initWithString_attributes_(label, _attrs(11.5)).size()
                widths.append(s.width + 16.0)
            row_w = sum(widths) + BTN_GAP * max(0, len(widths) - 1)
            bw = max(min(MAX_W, tr.size.width), row_w) + 2 * PAD + 4.0
            th = tr.size.height
            bh = PAD + th + ((BTN_GAP + BTN_H) if labels else 0.0) + PAD + TAIL + 2.0
            rects = []
            x = (bw - row_w) / 2.0
            y = PAD + th + BTN_GAP
            for (bid, _label), w in zip(labels, widths):
                rects.append((bid, (x, y, w, BTN_H)))
                x += w + BTN_GAP
            self._button_rects = rects
            self._text_h = th
            head = self._head()
            if head is None:
                return
            hx, top, vis = head
            bx, by = hx - bw * 0.38, top + 2.0
            if vis is not None:
                bx = min(max(bx, vis.origin.x + 4.0), vis.origin.x + vis.size.width - bw - 4.0)
                by = min(by, vis.origin.y + vis.size.height - bh - 2.0)
            self._bubble.setFrame_display_(NSMakeRect(bx, by, bw, bh), changed)
            if changed:
                self._bubble_view.setNeedsDisplay_(True)
                self._bubble.orderFrontRegardless()
        except Exception:
            log.debug("games: bubble failed", exc_info=True)

    def _bubble_click(self, x: float, y: float) -> None:
        for bid, (bx, by, bw, bh) in self._button_rects:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.manager.click(bid)
                return
        self.manager.bubble_clicked()

    def _paint_bubble(self, bounds: Any) -> None:
        text = self._bubble_text
        if not text:
            return
        w, h = bounds.size.width, bounds.size.height
        body = NSMakeRect(2.0, 2.0, w - 4.0, h - TAIL - 3.0)  # flipped: tail at the bottom
        box = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(body, 10.0, 10.0)
        tx = min(max(w * 0.38, 18.0), w - 24.0)
        wedge = NSBezierPath.bezierPath()
        wedge.moveToPoint_(NSMakePoint(tx - 7.0, h - TAIL - 2.5))
        wedge.lineToPoint_(NSMakePoint(tx - 2.0, h - 1.0))
        wedge.lineToPoint_(NSMakePoint(tx + 8.0, h - TAIL - 2.5))
        _rgba(PAPER).setFill()
        box.fill()
        wedge.fill()
        _rgba(INK).setStroke()
        box.setLineWidth_(1.6)
        box.stroke()
        wedge.setLineWidth_(1.6)
        wedge.stroke()
        _rgba(PAPER).setFill()
        NSBezierPath.bezierPathWithRect_(NSMakeRect(tx - 5.5, h - TAIL - 3.6, 12.0, 2.6)).fill()
        NSAttributedString.alloc().initWithString_attributes_(text, _attrs()).drawInRect_(
            NSMakeRect(PAD, PAD - 1.0, w - 2 * PAD, getattr(self, "_text_h", 40.0) + 2.0))
        for bid, (bx, by, bw, bh) in self._button_rects:
            label = next((b.label for b in self._bubble_buttons if b.id == bid), "")
            pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(bx, by, bw, bh), 8.0, 8.0)
            _rgba((0.93, 0.90, 0.80, 1.0)).setFill()
            pill.fill()
            _rgba(INK).setStroke()
            pill.setLineWidth_(1.2)
            pill.stroke()
            s = NSAttributedString.alloc().initWithString_attributes_(label, _attrs(11.5))
            size = s.size()
            s.drawAtPoint_(NSMakePoint(bx + (bw - size.width) / 2.0, by + (bh - size.height) / 2.0))

    # -- the board ---------------------------------------------------------------------------------------------
    def _show_board(self, state: dict | None) -> None:
        try:
            if not state or "board" not in state:
                if self._board is not None and self._board_state is not None:
                    self._board_state = None
                    self._board.orderOut_(None)
                return
            size = 3 * CELL + 2 * BOARD_PAD
            if self._board is None:
                self._board, self._board_view = _make_window(
                    size, size, WawiBoardView, self, NSFloatingWindowLevel)
            head = self._head()
            if head is None:
                return
            changed = state != self._board_state
            first = self._board_state is None
            self._board_state = dict(state)
            if first or changed:
                hx, top, vis = head
                sw, sh = self.panel._mascot_sprite_size
                x = hx + sw * 0.75
                y = top - sh * 0.92
                if vis is not None:
                    if x + size > vis.origin.x + vis.size.width - 4.0:
                        x = hx - sw * 0.4 - size
                    x = min(max(x, vis.origin.x + 4.0), vis.origin.x + vis.size.width - size - 4.0)
                    y = min(max(y, vis.origin.y + 4.0), vis.origin.y + vis.size.height - size - 4.0)
                self._board.setFrame_display_(NSMakeRect(x, y, size, size), changed)
            if changed:
                self._board_view.setNeedsDisplay_(True)
                self._board.orderFrontRegardless()
        except Exception:
            log.debug("games: board failed", exc_info=True)

    def _paint_board(self, bounds: Any) -> None:
        state = self._board_state or {}
        board = state.get("board") or " " * 9
        w = bounds.size.width
        bg = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(1.0, 1.0, w - 2.0, w - 2.0), 12.0, 12.0)
        _rgba(PAPER).setFill()
        bg.fill()
        _rgba(INK).setStroke()
        bg.setLineWidth_(1.4)
        bg.stroke()
        grid = NSBezierPath.bezierPath()
        for i in (1, 2):
            k = BOARD_PAD + i * CELL
            grid.moveToPoint_(NSMakePoint(k, BOARD_PAD + 3))
            grid.lineToPoint_(NSMakePoint(k + 0.8, BOARD_PAD + 3 * CELL - 3))
            grid.moveToPoint_(NSMakePoint(BOARD_PAD + 3, k))
            grid.lineToPoint_(NSMakePoint(BOARD_PAD + 3 * CELL - 3, k - 0.8))
        grid.setLineWidth_(2.0)
        grid.stroke()
        line = state.get("line") or ()
        for i, c in enumerate(board):
            if c == " ":
                continue
            row, col = divmod(i, 3)
            cx = BOARD_PAD + col * CELL + CELL / 2.0
            cy = BOARD_PAD + row * CELL + CELL / 2.0
            r = CELL * 0.28
            mark = NSBezierPath.bezierPath()
            if c == "X":
                mark.moveToPoint_(NSMakePoint(cx - r, cy - r))
                mark.lineToPoint_(NSMakePoint(cx + r, cy + r))
                mark.moveToPoint_(NSMakePoint(cx + r, cy - r))
                mark.lineToPoint_(NSMakePoint(cx - r, cy + r))
                _rgba(INK).setStroke()
            else:
                mark = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx - r, cy - r, 2 * r, 2 * r))
                _rgba(PAW).setStroke()
            mark.setLineWidth_(4.0 if i in line else 3.0)
            mark.stroke()

    # -- the ball ------------------------------------------------------------------------------------------------
    def _show_ball(self, pos: tuple[float, float] | None) -> None:
        try:
            if pos is None:
                if self._ball is not None and self._ball_pos is not None:
                    self._ball_pos = None
                    self._ball.orderOut_(None)
                return
            d = 30.0
            if self._ball is None:
                self._ball, self._ball_view = _make_window(
                    d, d, WawiBallView, self, NSFloatingWindowLevel)
            first = self._ball_pos is None
            self._ball_pos = (float(pos[0]), float(pos[1]))
            self._ball.setFrameOrigin_(NSMakePoint(pos[0] - d / 2.0, pos[1] - d / 2.0))
            if first:
                self._ball_view.setNeedsDisplay_(True)
                self._ball.orderFrontRegardless()
        except Exception:
            log.debug("games: ball failed", exc_info=True)

    def _place_ball_now(self) -> None:
        """While dragged, the ball follows the hand at once, not next frame."""
        state = self.manager.game_state()
        if state and state.get("ball") is not None:
            self._show_ball(state["ball"])

    def _paint_ball(self, bounds: Any) -> None:
        image = getattr(self.panel, "_mascot_props", {}).get("prop_ball")
        w = bounds.size.width
        rect = NSMakeRect(2.0, 2.0, w - 4.0, w - 4.0)
        if image is not None:
            image.drawInRect_fromRect_operation_fraction_(rect, NSMakeRect(0, 0, 0, 0), 2, 1.0)
            return
        ball = NSBezierPath.bezierPathWithOvalInRect_(rect)
        _rgba((0.92, 0.36, 0.22, 1.0)).setFill()
        ball.fill()
        _rgba(INK).setStroke()
        ball.setLineWidth_(1.5)
        ball.stroke()

