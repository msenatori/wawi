"""When she proposes a game, which one, and what she learns from the answer.

A proposal is an interruption, so the gate is strict: never in quiet mode
(meetings, full screen, quiet hours), never while the owner is typing
(focused), never to an empty room, never more than ``max_proposals_per_day``
and never within ``propose_every_min`` of the last one. Past all that she
still needs a reason: she is bored (her ``fun`` need is low), or the owner
just came back from a break.

Then what she learned decides. A :class:`~wawi.learn.bandit.BehaviourBandit`
(the same Thompson-sampling bandit she uses for her idle behaviours) keeps,
per part of the day and mood, how often each game was accepted — and one
more arm, ``__now__``, for "is this a good moment at all". A moment the owner
keeps turning down makes her propose less then; a game they always accept
gets picked more. Accepted = +1, a spoken "no" = -1, ignored = -0.5.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from wawi.games.core import GameEnv
from wawi.games.settings import GamesSettings
from wawi.learn.bandit import BehaviourBandit, context

log = logging.getLogger(__name__)

NOW_ARM = "__now__"
#: After a return from a break, a proposal may come within this long.
RETURN_WINDOW = 240.0
#: A key press this recent means the owner is back.
BACK_IDLE = 5.0
#: The gate is evaluated at most this often (seconds).
CHECK_EVERY = 20.0
#: When the learned timing says "not now", look again this much later.
NOT_NOW_RETRY = 10 * 60.0
#: Lowest chance of going ahead however often this moment was refused —
#: tastes change, and a bandit that never tries never learns that.
MIN_CHANCE = 0.25

REWARD_YES = 1.0
REWARD_NO = -1.0
REWARD_IGNORED = -0.5


@dataclass
class Proposal:
    game: str
    at: float
    until: float
    ctx: str


class Proposer:
    def __init__(self, settings: GamesSettings, bandit: BehaviourBandit | None = None,
                 rng: random.Random | None = None) -> None:
        self.s = settings
        self.bandit = bandit or BehaviourBandit(factor_min=0.05, factor_max=20.0)
        self.rng = rng or random.Random()
        self.last_at = -1e18
        self.day = ""
        self.today = 0
        self.returned_at: float | None = None
        self._idle_peak = 0.0
        self._next_check = 0.0
        self.pending: Proposal | None = None

    # -- watching the owner ---------------------------------------------------------
    def observe(self, env: GameEnv) -> None:
        """Track breaks: a long stretch with no key press, then one."""
        idle = env.key_idle
        if idle < BACK_IDLE and self._idle_peak >= self.s.after_break:
            self.returned_at = env.now
        if idle < BACK_IDLE:
            self._idle_peak = idle
        else:
            self._idle_peak = max(self._idle_peak, idle)

    # -- the gate -----------------------------------------------------------------------
    def blocked(self, env: GameEnv, playing: bool = False) -> str | None:
        """Why she must not propose now, or None when she may."""
        s = self.s
        if not (s.enabled and s.propose) or s.max_per_day <= 0:
            return "off"
        if playing or self.pending is not None or env.busy:
            return "busy"
        if env.quiet:
            return "quiet"
        if env.now - self.last_at < s.propose_every:
            return "cooldown"
        self._roll_day(env.wall)
        if self.today >= s.max_per_day:
            return "budget"
        if env.key_idle >= s.away_after:
            return "away"
        if env.key_idle < s.focus_typing:
            return "focus"
        back = self.returned_at is not None and env.now - self.returned_at <= RETURN_WINDOW
        bored = env.fun < s.bored_below
        if not (back or bored):
            return "content"
        return None

    def _roll_day(self, wall: float) -> None:
        day = time.strftime("%Y-%m-%d", time.localtime(wall or time.time()))
        if day != self.day:
            self.day, self.today = day, 0

    def ctx(self, env: GameEnv) -> str:
        hour = time.localtime(env.wall or time.time()).tm_hour
        return context(hour, env.mood)

    # -- deciding ---------------------------------------------------------------------------
    def maybe(self, env: GameEnv, games: "Sequence[str] | Callable[[], Sequence[str]]",
              playing: bool = False) -> Proposal | None:
        """Called every frame; proposes at most once per gate check.
        *games* may be a callable: only called when the gate is open (it can
        be costly, e.g. looking for a hiding spot)."""
        self.observe(env)
        if env.now < self._next_check:
            return None
        self._next_check = env.now + CHECK_EVERY
        if self.blocked(env, playing) is not None:
            return None
        games = list(games() if callable(games) else games)
        if not games:
            return None
        ctx = self.ctx(env)
        chance = max(MIN_CHANCE, min(1.0, self.bandit.expected(NOW_ARM, ctx) * 1.6))
        if self.rng.random() > chance:
            # The owner has been saying no at this time of day: later.
            self.last_at = env.now - self.s.propose_every + NOT_NOW_RETRY
            return None
        return self.propose(env, games, ctx)

    def pick(self, games: Sequence[str], ctx: str) -> str:
        """Thompson sampling over the games: one Beta draw each, best wins."""
        names = list(dict.fromkeys(games))
        draws = {g: float(self.bandit.rng.beta(*self.bandit.posterior(g, ctx))) for g in names}
        return max(names, key=lambda g: (draws[g], -names.index(g)))

    def propose(self, env: GameEnv, games: Sequence[str], ctx: str | None = None) -> Proposal:
        ctx = ctx or self.ctx(env)
        game = self.pick(games, ctx)
        self._roll_day(env.wall)
        self.today += 1
        self.last_at = env.now
        self.returned_at = None
        self.pending = Proposal(game, env.now, env.now + self.s.proposal_timeout, ctx)
        log.info("games: proposing %s (%s)", game, ctx)
        return self.pending

    # -- the answer ----------------------------------------------------------------------------
    def answer(self, accepted: bool | None) -> Proposal | None:
        """True = yes, False = a spoken no, None = ignored (timed out)."""
        p, self.pending = self.pending, None
        if p is None:
            return None
        reward = REWARD_YES if accepted else (REWARD_NO if accepted is False else REWARD_IGNORED)
        self.bandit.update(p.game, p.ctx, reward)
        self.bandit.update(NOW_ARM, p.ctx, reward)
        return p

    def expired(self, now: float) -> bool:
        return self.pending is not None and now >= self.pending.until

    # -- persistence ---------------------------------------------------------------------------------
    def to_dict(self, wall_now: float, mono_now: float) -> dict[str, Any]:
        # Monotonic clocks do not survive a restart: keep the last proposal
        # as wall time so a restart does not reset the cooldown.
        ago = mono_now - self.last_at
        return {"bandit": self.bandit.to_dict(),
                "last_wall": (wall_now - ago) if ago < 1e12 else None,
                "day": self.day, "today": self.today}

    def load(self, raw: Any, wall_now: float, mono_now: float) -> None:
        if not isinstance(raw, dict):
            return
        self.bandit.load(raw.get("bandit"))
        last = raw.get("last_wall")
        if isinstance(last, (int, float)):
            self.last_at = mono_now - max(0.0, wall_now - float(last))
        self.day = str(raw.get("day") or "")
        try:
            self.today = int(raw.get("today", 0))
        except (TypeError, ValueError):
            self.today = 0
