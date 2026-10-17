"""The games, run: one game at a time, proposals, scores, her reactions.

:class:`GameManager` is the one object everything talks to:

* the voice turn (:meth:`hear`, worker thread) — a move, "basta", "dale" to
  a proposal, "juguemos a la pelota";
* the voice tool ``play_game`` (:meth:`start` / :meth:`stop` / :meth:`hear`);
* the menu and her windows (:meth:`start`, :meth:`click`,
  :meth:`bubble_clicked`, :meth:`touched`, the ball's grab/drag/release —
  main thread);
* the panel's frame tick (:meth:`tick`, main thread), which advances time,
  maybe proposes a game, and returns a :class:`View` for the host to draw.

One lock covers it all, so the worker and the main thread never see a half
move. Her faces and sounds go out as :class:`~wawi.reactions.Reaction` on
the bus, like any other reaction; fun and energy go into her needs on the
next tick (main thread, where the body lives).
"""

from __future__ import annotations

import dataclasses
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from wawi.games import propose as P
from wawi.games.core import (
    Button,
    Game,
    GameEnv,
    Reply,
    art,
    is_stop,
    norm,
    words,
)
from wawi.games.fetch import Fetch
from wawi.games.guess import GuessHers, GuessYours
from wawi.games.hide import HideAndSeek, choose_spot
from wawi.games.rps import RPS, PatternLearner
from wawi.games.settings import GamesSettings
from wawi.games.store import GameStore
from wawi.games.tictactoe import TicTacToe
from wawi.games.trivia import Trivia

log = logging.getLogger(__name__)

#: name → (class, title, spoken aliases). Order = the menu's.
GAMES: dict[str, tuple[type[Game], str, tuple[str, ...]]] = {
    "ppt": (RPS, "Piedra, papel o tijera",
            ("piedra papel", "papel o tijera", "piedra papel o tijera", "ppt", "cachipun")),
    "tateti": (TicTacToe, "Ta-te-tí", ("tateti", "ta te ti", "tres en raya", "tres en linea")),
    "numero": (GuessHers, "Adiviná el número",
               ("adivina el numero", "adivinar el numero", "adivinar un numero", "adivina un numero",
                "el numero")),
    "adivino": (GuessYours, "Adivino tu número",
                ("adivina mi numero", "adivinas mi numero", "adivino tu numero",
                 "adivinar mi numero")),
    "pelota": (Fetch, "Buscar la pelota", ("pelota", "buscar la pelota", "traer la pelota")),
    "escondidas": (HideAndSeek, "Escondidas", ("escondidas", "escondida", "esconderte")),
    "trivia": (Trivia, "Trivia de perros", ("trivia", "preguntas", "adivinanzas")),
}
#: How the tool / the brain may name them, beyond the keys.
ALIASES = {
    "rps": "ppt", "piedra": "ppt", "piedrapapeltijera": "ppt", "tictactoe": "tateti",
    "guess": "numero", "adivina": "numero", "number": "numero", "guessmine": "adivino",
    "fetch": "pelota", "ball": "pelota", "hide": "escondidas", "hideandseek": "escondidas",
    "dogtrivia": "trivia", "quiz": "trivia",
}
PROPOSALS = {
    "ppt": "¿Jugamos a piedra, papel o tijera?",
    "tateti": "¿Un ta-te-tí? Te dejo empezar.",
    "numero": "¿Jugamos a adivinar el número?",
    "adivino": "¿Pensás un número y yo lo adivino?",
    "pelota": "¿Me tirás la pelota? ¿Eh? ¿Eh?",
    "escondidas": "¿Jugamos a las escondidas?",
    "trivia": "¿Una trivia de perros? Sé un montón.",
}
#: The final line of a game stays up this long.
END_LINE_S = 7.0
#: A reply that is not the game's standing line shows this long.
REPLY_LINE_S = 4.0
#: "otra" right after a game: the same one again, within this long.
AGAIN_WITHIN = 90.0
AGAIN_WORDS = ("otra", "otra vez", "de nuevo", "revancha", "la revancha", "juguemos otra")
PLAY_VERBS = ("jug",)


@dataclass
class View:
    """What the host draws this frame."""

    game: str | None = None
    bubble: str | None = None
    buttons: list[Button] = field(default_factory=list)
    #: A click on the bubble means this (``accept`` a proposal), or nothing.
    bubble_action: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    #: Her body belongs to the game (she leaves the panel, wandering stops).
    free: bool = False


def resolve(name: str) -> str | None:
    """A game key from whatever the tool / the menu / a sentence calls it."""
    n = norm(name).replace(" ", "")
    if not n:
        return None
    if n in GAMES:
        return n
    if n in ALIASES:
        return ALIASES[n]
    t = norm(name)
    for key, (_cls, _title, aliases) in GAMES.items():
        if any(a in t for a in aliases):
            return key
    return None


class GameManager:
    def __init__(self, settings: GamesSettings | None = None, *, store: GameStore | None = None,
                 rng: random.Random | None = None, push: Callable[[Any], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time) -> None:
        self.settings = s = settings or GamesSettings()
        self.rng = rng or random.Random()
        self._clock = clock
        self._wall = wall
        self._lock = threading.RLock()
        self.store = store if store is not None else GameStore(s.state_file).load()
        self.learner = PatternLearner(np.random.default_rng(self.rng.randrange(2 ** 31)))
        self.learner.load(self.store.get("rps"))
        self.proposer = P.Proposer(s, rng=self.rng)
        self.proposer.bandit.rng = np.random.default_rng(self.rng.randrange(2 ** 31))
        self.proposer.load(self.store.get("proposals"), wall(), clock())
        self._push = push
        self.game: Game | None = None
        self._line: str | None = None
        self._line_until = 0.0
        self._last_game: str | None = None
        self._last_end = -1e18
        self._gains: list[tuple[float, float]] = []
        #: The host says whether a game that needs her body can run now.
        self.body_ok = False
        self._ttt_owner_starts = True
        self._env = GameEnv()

    # -- reactions out --------------------------------------------------------------------
    def _react(self, reply: Reply, speak: bool = False) -> None:
        frames = self._env.frames
        expr, then = art(reply.expr, frames), art(reply.then, frames)
        if expr is None:
            expr, then = then, None
        if not (expr or reply.sound or speak):
            return
        try:
            from wawi.reactions import BUS, Reaction

            push = self._push or BUS.push
            push(Reaction(behaviour=expr, sound=reply.sound, priority=2,
                          bubble=reply.text if speak else None, speak=speak))
            if then:
                # Same priority: it waits for the first one to finish.
                push(Reaction(behaviour=then, priority=2))
        except Exception:
            log.debug("games: reaction failed", exc_info=True)

    def _apply(self, reply: Reply | None, now: float) -> str | None:
        """Show a game's reply; book the result when the game ends."""
        if reply is None:
            return None
        g = self.game
        if reply.text:
            # The game's own line stays up while it runs; a reply that is not
            # the game's line ("ahí no se puede") shows for a moment.
            self._line = reply.text
            self._line_until = now + (END_LINE_S if reply.done else REPLY_LINE_S)
        if reply.fun or reply.energy:
            self._gains.append((reply.fun, reply.energy))
        self._react(reply)
        if reply.done and g is not None:
            if reply.result:
                self.store.record(g.name, reply.result)
            if isinstance(g, RPS):
                self.store.set("rps", self.learner.to_list())
            if isinstance(g, Fetch):
                self.store.set("fetches", int(self.store.get("fetches", 0) or 0) + g.fetches)
            self._last_game, self._last_end = g.name, now
            self.game = None
            self.store.save(force=True)
            log.info("games: %s over (%s)", g.name, reply.result or "-")
        return reply.text or None

    # -- starting and stopping -----------------------------------------------------------------
    def available(self) -> list[str]:
        """Games that can start right now."""
        return [k for k, (cls, _t, _a) in GAMES.items() if self.body_ok or not cls.needs_body]

    def start(self, name: str) -> str:
        with self._lock:
            if not self.settings.enabled:
                return "Los juegos están apagados en la configuración."
            key = resolve(name)
            if key is None:
                return "No conozco ese juego. Sé piedra, papel o tijera, ta-te-tí, adivinar " \
                       "el número, la pelota, escondidas y trivia."
            cls = GAMES[key][0]
            if cls.needs_body and not self.body_ok:
                return "Para eso tengo que estar en el escritorio, y ahora no puedo."
            now = self._clock()
            if self.proposer.pending is not None:
                # Asked for a game while one was on offer: that is a yes to
                # playing (and to now), whichever game it was.
                self.proposer.answer(True)
                self._signal("play")
            if self.game is not None:
                self.game.stop()
                self.game = None
            self.game = self._make(key)
            log.info("games: start %s", key)
            text = self._apply(self.game.start(), now) or ""
            return text

    def _make(self, key: str) -> Game:
        s = self.settings
        cls = GAMES[key][0]
        if cls is RPS:
            return RPS(self.rng, learner=self.learner, first_to=s.rps_first_to)
        if cls is TicTacToe:
            g = TicTacToe(self.rng, level=s.tictactoe_level, owner_starts=self._ttt_owner_starts)
            self._ttt_owner_starts = not self._ttt_owner_starts
            return g
        if cls is Fetch:
            return Fetch(self.rng, max_fetches=s.fetch_max)
        if cls is HideAndSeek:
            return HideAndSeek(self.rng, seconds=s.hide_seconds)
        if cls is Trivia:
            return Trivia(self.rng, count=s.trivia_questions)
        return cls(self.rng)

    def stop(self) -> str:
        with self._lock:
            if self.proposer.pending is not None and self.game is None:
                self._decline(spoken=True)
                return "Bueno, otro día."
            if self.game is None:
                return "No estamos jugando a nada."
            g, now = self.game, self._clock()
            reply = g.stop()
            self.game = None
            self._last_game, self._last_end = g.name, now
            self._line, self._line_until = reply.text, now + END_LINE_S
            self._react(reply)
            self.store.save(force=True)
            return reply.text

    @property
    def playing(self) -> str | None:
        with self._lock:
            return self.game.name if self.game is not None else None

    def wants_free(self) -> bool:
        with self._lock:
            return self.game is not None and self.game.needs_body

    # -- proposals ------------------------------------------------------------------------------
    def _signal(self, kind: str) -> None:
        try:
            from wawi import learn

            learn.signal(kind)
        except Exception:
            pass

    def accept(self) -> str | None:
        with self._lock:
            p = self.proposer.pending
            if p is None:
                return None
            self.proposer.answer(True)
            self._signal("play")
            self._save_proposals()
        return self.start(p.game)

    def _decline(self, spoken: bool) -> None:
        p = self.proposer.answer(False if spoken else None)
        self._save_proposals()
        if p is not None:
            self._line = None
            if spoken:
                self._react(Reply(expr="shrug"))

    def _save_proposals(self) -> None:
        self.store.set("proposals", self.proposer.to_dict(self._wall(), self._clock()))

    def propose_now(self, game: str | None = None) -> str | None:
        """Propose right away (the menu's test hook, and the tests')."""
        with self._lock:
            games = [game] if game in GAMES else self.available()
            if not games or self.game is not None:
                return None
            env = dataclasses.replace(self._env, now=self._clock(), wall=self._wall())
            p = self.proposer.propose(env, games)
            return self._show_proposal(p)

    def _show_proposal(self, p: P.Proposal) -> str:
        text = PROPOSALS.get(p.game, "¿Jugamos?")
        self._line = text
        self._line_until = p.until
        self._react(Reply(text, "ball_bring"), speak=self.settings.speak_proposals)
        self._save_proposals()
        return text

    # -- input ------------------------------------------------------------------------------------
    def hear(self, text: str) -> str | None:
        """A transcript. Returns what to answer, or None: not for the games."""
        if not self.settings.enabled or not text or not text.strip():
            return None
        with self._lock:
            now = self._clock()
            if self.proposer.pending is not None and self.game is None:
                from wawi import lang

                verdict = lang.affirmative(text)
                if verdict is True:
                    return self.accept()
                if verdict is False:
                    self._decline(spoken=True)
                    return "Bueno, otro día."
            g = self.game
            if g is not None:
                if is_stop(text):
                    return self.stop()
                reply = g.hear(text)
                if reply is not None:
                    return self._apply(reply, now) or ""
            t = norm(text)
            ws = words(text)
            again = self._last_game is not None and now - self._last_end <= AGAIN_WITHIN \
                and len(ws) <= 4 and any(norm(a) == t or norm(a) in t for a in AGAIN_WORDS)
            if again and g is None:
                return self.start(self._last_game or "")
            if any(w.startswith(PLAY_VERBS) for w in ws) and len(ws) <= 10:
                key = resolve(text)
                if key is not None:
                    return self.start(key)
        return None

    def click(self, what: str) -> None:
        with self._lock:
            if what == "accept":
                self.accept()
                return
            if what == "decline":
                self._decline(spoken=True)
                return
            if what == "stop":
                self.stop()
                return
            if self.game is not None:
                self._apply(self.game.click(what), self._clock())

    def bubble_clicked(self) -> bool:
        """A click on the game bubble (not on a button). True if handled."""
        with self._lock:
            if self.proposer.pending is not None and self.game is None:
                self.accept()
                return True
            if self.game is None and self._line is not None:
                self._line = None  # the final line: dismissed
                return True
            return self.game is not None

    def touched(self) -> bool:
        """A click on HER. True if a game took it (do not pet her too)."""
        with self._lock:
            g = self.game
            if isinstance(g, HideAndSeek):
                reply = g.touched()
                if reply is not None:
                    self._apply(reply, self._clock())
                    return True
            return False

    # The ball (fetch), from its own little window.
    def ball_grab(self, x: float, y: float) -> bool:
        with self._lock:
            g = self.game
            return isinstance(g, Fetch) and g.grab(x, y, self._clock())

    def ball_drag(self, x: float, y: float) -> None:
        with self._lock:
            if isinstance(self.game, Fetch):
                self.game.drag(x, y, self._clock())

    def ball_release(self, x: float, y: float) -> None:
        with self._lock:
            if isinstance(self.game, Fetch):
                self.game.release(x, y, self._clock())

    # -- time ---------------------------------------------------------------------------------------
    def tick(self, dt: float, env: GameEnv, needs: Any = None) -> View:
        with self._lock:
            self._env = env
            now = env.now
            if needs is not None and self._gains:
                for fun, energy in self._gains:
                    try:
                        needs.give({"fun": fun, "affection": fun * 0.15, "energy": energy})
                    except Exception:
                        log.debug("games: needs failed", exc_info=True)
            self._gains.clear()
            g = self.game
            if g is not None:
                self._apply(g.tick(dt, env), now)
            pr = self.proposer
            if pr.expired(now) and self.game is None:
                self._decline(spoken=False)
            elif self.settings.propose and self.game is None:
                p = pr.maybe(env, lambda: self._proposable(env),
                             playing=self._line is not None and now < self._line_until)
                if p is not None:
                    self._show_proposal(p)
            self.store.save(now=now)
            return self._view(now)

    def _proposable(self, env: GameEnv) -> list[str]:
        out = []
        for key in self.available():
            if key == "escondidas" and choose_spot(env.windows, env.screens, env.size,
                                                   self.rng, env.cursor, env.frames) is None:
                continue
            if key == "pelota" and env.energy < 35:
                continue  # too tired to run
            out.append(key)
        return out

    def _view(self, now: float) -> View:
        g = self.game
        line = self._line if self._line is not None and now < self._line_until else None
        if line is None and self._line is not None and now >= self._line_until:
            self._line = None
        if g is None:
            if self.proposer.pending is not None and line is not None:
                return View(None, line, [Button("accept", "¡Dale!"), Button("decline", "Ahora no")],
                            "accept")
            return View(None, line)
        bubble = (line or g.line or None) if g.bubble_ok() else None
        return View(g.name, bubble, list(g.buttons), None, g.state(), free=g.needs_body)

    def game_state(self) -> dict[str, Any]:
        with self._lock:
            return self.game.state() if self.game is not None else {}

    # -- reading ------------------------------------------------------------------------------------
    def score_line(self) -> str:
        s = self.store.score()
        return f"Vos {s['owner']} · Wawi {s['wawi']} · empates {s['draw']}"

    def close(self) -> None:
        with self._lock:
            self.store.set("rps", self.learner.to_list())
            self._save_proposals()
            self.store.save(force=True)
