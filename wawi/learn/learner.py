"""The learner: turns what the owner does into rewards for what Wawi did.

It sits between the body (``wawi/pet/life.py``, ``wawi/panel.py``) and the
three small models:

* the body says what Wawi just did — an idle behaviour started
  (:meth:`Learner.behaviour_started`), an alert bubble appeared
  (:meth:`Learner.alert_shown`) — and what the owner did — touched, petted,
  talked, fed, dragged her away, hid her, clicked a bubble away
  (:meth:`Learner.signal`);
* the learner matches them by time (``reward_window_s``) and settles each
  decision once: positive, negative, or neutral when nothing happened;
* settled decisions update the bandit / alert counts at once and land in the
  life log for the tiny brain, which retrains later while the machine idles.

Going the other way, :meth:`Learner.bias` re-weights the animator's idle
behaviours and :meth:`Learner.triage` tells the reflex layer which alerts to
show now, batch, or drop.

Threading: the body calls in on the main thread, the reflexes on a sensor
thread; one lock covers the shared state. Nothing here blocks: training and
the embedding model load on background threads, the state file is written at
most every couple of minutes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from wawi.learn.alerts import AlertLearner
from wawi.learn.bandit import BehaviourBandit, context
from wawi.learn.brain import Brain, LifeLog, features
from wawi.learn.relevance import BackgroundEmbedder, Relevance, model2vec_available
from wawi.learn.settings import LearnSettings

log = logging.getLogger(__name__)

#: What each owner action says about what Wawi just did.
POSITIVE = {"touch": 1.0, "pet": 1.0, "talk": 1.0, "feed": 1.0, "water": 1.0, "play": 1.0}
NEGATIVE = {"drag": -0.5, "hide": -0.5}
SAVE_EVERY_S = 120.0
TICK_EVERY_S = 5.0
#: Spanish names for the summary.
BEHAVIOUR_WORDS = {
    "look_left": "mirar para los costados", "look_right": "mirar para los costados",
    "look_around": "chusmear alrededor", "yawn": "bostezar", "scratch": "rascarme",
    "stretch": "estirarme", "sniff": "olfatear", "wag": "mover la cola", "tilt": "ladear la cabeza",
    "roll": "revolcarme", "dig": "cavar", "run": "correr", "stare": "quedarme mirándote",
    "squint": "entrecerrar los ojos", "eyeroll": "revolear los ojos", "smug": "hacerme la canchera",
    "side_eye": "mirarte de reojo", "shrug": "encogerme de hombros", "volume_knob": "jugar con la perilla",
    "tail_chase": "perseguirme la cola", "hungry_bowl": "pedirte comida con el plato",
    "ball_bring": "traerte la pelota",
}
ALERT_WORDS = {
    "rss.item": "las noticias", "weather.hot": "el calor", "weather.cold": "el frío",
    "weather.sunny": "los días lindos", "weather.rain_soon": "la lluvia",
    "break.suggest": "las pausas", "clock.morning": "el buen día", "clock.lunch": "el almuerzo",
    "clock.late_night": "cuando es tarde", "web.up": "las webs que vuelven",
    "service.up": "los servicios que vuelven", "disk.low": "el disco", "memory.high": "la memoria",
}


#: Bandit arms for the instincts (wawi/instinct) live under this prefix, next
#: to the idle behaviours: one learner, one evidence decay, one state file.
INSTINCT_ARM = "instinct:"


@dataclass
class _Pending:
    kind: str  # "b" behaviour, "a" alert, "i" instinct
    name: str
    ctx: str
    x: np.ndarray
    at: float
    event: Any = None
    title_shown: bool = False


@dataclass
class _Ctx:
    wall: float = 0.0
    mood: str = "content"
    needs: dict[str, float] = field(default_factory=dict)
    ignored_for: float = 0.0
    key_idle: float = float("inf")


class Learner:
    def __init__(
        self,
        settings: LearnSettings | None = None,
        *,
        actions: Iterable[str] = (),
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        seed: int | None = None,
        embedder: Any = "auto",
    ) -> None:
        self.settings = s = settings or LearnSettings()
        self._clock = clock
        self._wall = wall
        self._lock = threading.RLock()
        rng = np.random.default_rng(seed)
        self.bandit = BehaviourBandit(factor_min=s.factor_min, factor_max=s.factor_max,
                                      half_life_days=s.half_life_days, rng=rng)
        self.alerts = AlertLearner(half_life_days=s.half_life_days)
        if embedder == "auto":
            embedder = None
            if s.relevance and s.embed_model and model2vec_available():
                embedder = BackgroundEmbedder(s.embed_model).start()
        self.relevance = Relevance(s.interests, embedder=embedder)
        self.brain: Brain | None = None
        if s.brain:
            self.brain = Brain(
                actions,
                path=s.brain_file,
                lifelog=LifeLog(s.life_log, max_days=s.life_log_days,
                                max_bytes=s.life_log_max_bytes),
                min_samples=s.brain_min_samples,
                max_weight=s.brain_max_weight,
                retrain_every_s=s.retrain_every_h * 3600.0,
                retrain_idle_s=s.retrain_idle_s,
            )
        self._pending: list[_Pending] = []
        self._ctx = _Ctx(wall=wall())
        self._recent_alerts: deque[float] = deque(maxlen=20)
        self._dirty = False
        self._saved_at = clock()
        self._ticked_at = -1e18
        self.stats = {"rewards": 0, "positive": 0, "negative": 0}

    # -- context -----------------------------------------------------------------
    def observe(self, *, mood: str | None = None, needs: dict[str, float] | None = None,
                ignored_for: float | None = None, key_idle: float | None = None,
                wall: float | None = None) -> None:
        """What the body knows about now (called every few seconds)."""
        with self._lock:
            c = self._ctx
            c.wall = self._wall() if wall is None else wall
            if mood is not None:
                c.mood = mood
            if needs is not None:
                c.needs = dict(needs)
            if ignored_for is not None:
                c.ignored_for = ignored_for
            if key_idle is not None:
                c.key_idle = key_idle

    def _features(self, now: float) -> np.ndarray:
        c = self._ctx
        recent = sum(1 for t in self._recent_alerts if now - t < 1800.0)
        return features({"wall": c.wall or self._wall(), "mood": c.mood, "needs": c.needs,
                         "ignored_for": c.ignored_for, "key_idle": c.key_idle,
                         "recent_alerts": recent})

    def _context_key(self) -> str:
        return context(time.localtime(self._ctx.wall or self._wall()).tm_hour, self._ctx.mood)

    # -- behaviours ------------------------------------------------------------------
    def bias(self, mood_bias: dict[str, float], arms: Iterable[str]) -> dict[str, float]:
        """The animator's bias: the mood's multipliers times what was learned.

        Only *arms* are re-weighted (the behaviours that can be picked now),
        so nothing new enters the pool through here, and every factor is
        clamped, so nothing leaves it either.
        """
        out = dict(mood_bias)
        s = self.settings
        if not (s.enabled and s.behaviours):
            return out
        arms = list(arms)
        with self._lock:
            ctx = self._context_key()
            learned = self.bandit.factors(arms, ctx)
            if self.brain is not None:
                brain = self.brain.factors(self._features(self._clock()), arms)
            else:
                brain = {}
        for a in arms:
            f = learned.get(a, 1.0) * brain.get(a, 1.0)
            f = min(s.factor_max, max(s.factor_min, f))
            out[a] = float(mood_bias.get(a, 1.0)) * f
        return out

    def behaviour_started(self, name: str, now: float | None = None) -> None:
        if not (self.settings.enabled and self.settings.behaviours) or not name:
            return
        now = self._clock() if now is None else now
        with self._lock:
            self._pending.append(_Pending("b", name, self._context_key(), self._features(now), now))
            del self._pending[:-12]

    # -- alerts ------------------------------------------------------------------------
    def alert_shown(self, event: Any, bubble: str | None = None, now: float | None = None) -> None:
        """A bubble from *event* is on screen now."""
        if not (self.settings.enabled and self.settings.alerts) or event is None:
            return
        if str(getattr(event, "kind", "")).startswith("instinct."):
            return  # the instinct engine registered it already (instinct_shown)
        now = self._clock() if now is None else now
        title = str(getattr(event, "title", "") or (getattr(event, "data", {}) or {}).get("title", ""))
        with self._lock:
            # A new bubble replaces the old one: the old one was not answered.
            for p in [p for p in self._pending if p.kind == "a"]:
                self._settle(p, None)
            self._recent_alerts.append(now)
            self._pending.append(_Pending(
                "a", str(getattr(event, "kind", "")), "", self._features(now), now, event,
                title_shown=bool(title and bubble and title[:40] in bubble)))

    # -- instincts (wawi/instinct) ------------------------------------------------------------
    def instinct_shown(self, name: str, bubble: bool = False, now: float | None = None) -> None:
        """Wawi just acted on instinct *name*: what the owner does next is
        the verdict (engagement = welcome; a quick bubble click or dragging
        her away = not now)."""
        if not self.settings.enabled or not name:
            return
        now = self._clock() if now is None else now
        with self._lock:
            for p in [p for p in self._pending if p.kind == "i"]:
                self._settle(p, None)
            self._pending.append(_Pending("i", name, self._context_key(), self._features(now),
                                          now, title_shown=bool(bubble)))
            del self._pending[:-12]

    def instinct_factor(self, name: str, bubble: bool = False) -> float:
        """A multiplier for the instinct's urgency, in [factor_min, factor_max]:
        a Thompson draw from its arm against the no-evidence mean (0.5), times
        the brain's "good moment to interrupt" when it has an opinion."""
        s = self.settings
        if not s.enabled:
            return 1.0
        with self._lock:
            a, b = self.bandit.posterior(INSTINCT_ARM + name, self._context_key())
            f = float(self.bandit.rng.beta(a, b)) / 0.5
            if bubble and self.brain is not None:
                pred = self.brain.alert(self._features(self._clock()))
                if pred is not None:
                    f *= 1.0 + self.brain.weight * float(pred)
        return float(min(s.factor_max, max(s.factor_min, f)))

    def instinct_likes(self, n: int = 2) -> list[str]:
        """Instincts the owner clearly welcomes, best first."""
        with self._lock:
            known = [(a[len(INSTINCT_ARM):], self.bandit.expected(a)) for a in self.bandit.arms
                     if a.startswith(INSTINCT_ARM) and self.bandit.evidence(a) >= 2.0]
        return [k for k, v in sorted(known, key=lambda kv: -kv[1]) if v > 0.6][:n]

    def forget(self) -> None:
        """Wipe everything learned, in memory and on disk ("olvidá lo que
        aprendiste"). Settings and the embedding model stay."""
        s = self.settings
        with self._lock:
            self.bandit = BehaviourBandit(factor_min=s.factor_min, factor_max=s.factor_max,
                                          half_life_days=s.half_life_days, rng=self.bandit.rng)
            self.alerts = AlertLearner(half_life_days=s.half_life_days)
            self.relevance = Relevance(s.interests, embedder=self.relevance.embedder)
            self._pending.clear()
            self._recent_alerts.clear()
            self.stats = {"rewards": 0, "positive": 0, "negative": 0}
            self._dirty = False
            if self.brain is not None:
                self.brain.net = None
                self.brain.samples = 0
                self.brain.skill = 0.0
                self.brain.trained_at = None
                self.brain.new_since_train = 0
        paths = [s.state_file, s.brain_file, s.life_log]
        for p in [p for p in paths if p is not None]:
            for q in (p, p.with_name(p.name + ".tmp"), p.with_name(p.name + ".tmp.npz")):
                try:
                    q.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    log.warning("learn: could not delete %s", q, exc_info=True)

    # -- what the owner did ---------------------------------------------------------------
    def signal(self, kind: str, now: float | None = None) -> None:
        """touch / pet / talk / feed / water / play / drag / hide / bubble_click."""
        if not self.settings.enabled:
            return
        now = self._clock() if now is None else now
        window = self.settings.reward_window_s
        with self._lock:
            for p in list(self._pending):
                if now - p.at > window:
                    continue
                if kind in POSITIVE:
                    self._settle(p, POSITIVE[kind])
                elif kind in NEGATIVE:
                    self._settle(p, NEGATIVE[kind] if p.kind == "b" else -1.0)
                elif kind == "bubble_click" and p.kind == "i":
                    quick = now - p.at <= self.settings.dismiss_quick_s
                    self._settle(p, -1.0 if quick else 0.0)
                elif kind == "bubble_click" and p.kind == "a":
                    quick = now - p.at <= self.settings.dismiss_quick_s
                    self._settle(p, -1.0 if quick else 0.0, read=not quick)

    def _settle(self, p: _Pending, reward: float | None, read: bool = False) -> None:
        """Book one decision. *reward* None = it timed out unanswered."""
        if p in self._pending:
            self._pending.remove(p)
        r = 0.0 if reward is None else float(reward)
        wall = self._ctx.wall or self._wall()
        if p.kind == "b":
            self.bandit.update(p.name, p.ctx, r)
            if self.brain is not None:
                self.brain.record(wall, "b", p.name, r, p.x)
        elif p.kind == "i":
            self.bandit.update(INSTINCT_ARM + p.name, p.ctx, r)
            if self.brain is not None and p.title_shown:
                # A bubble on instinct is an interruption like an alert: it
                # teaches the brain's "is now a good moment" head the same way.
                self.brain.record(wall, "a", "__alert__", -0.3 if reward is None else r, p.x)
        else:
            if reward is None:
                outcome = "ignored"
            elif read:
                outcome = "read"
            else:
                outcome = "engaged" if r > 0 else "dismissed"
            self.alerts.record(p.event, outcome)
            if outcome in ("engaged", "dismissed") and p.title_shown \
                    and str(getattr(p.event, "kind", "")).startswith("rss."):
                self.relevance.learn(str(getattr(p.event, "title", "")), outcome == "engaged")
            if self.brain is not None:
                # Ignored is mildly negative for "now was a good moment".
                self.brain.record(wall, "a", "__alert__", -0.3 if reward is None else r, p.x)
        self.stats["rewards"] += 1
        if r > 0:
            self.stats["positive"] += 1
        elif r < 0:
            self.stats["negative"] += 1
        self._dirty = True

    # -- triage for the reflex layer ---------------------------------------------------------
    def triage(self, event: Any) -> str | None:
        """"now" / "later" / "ignore", or None to let the rules decide.

        Never for urgent or warning: what the owner taught Wawi about news
        must never hide a fire.
        """
        s = self.settings
        if not (s.enabled and s.alerts):
            return None
        if getattr(event, "severity", "info") not in ("info", "notice"):
            return None
        kind = str(getattr(event, "kind", ""))
        if kind == "rss.item" and s.relevance:
            title = str(getattr(event, "title", "") or (getattr(event, "data", {}) or {}).get("title", ""))
            rel, confident = self.relevance.score(title)
            if rel is not None:
                if rel >= s.own_bubble_above:
                    return "now"
                if confident and rel < s.drop_below:
                    return "ignore"
        with self._lock:
            score, evidence = self.alerts.score(event)
            if self.brain is not None:
                b = self.brain.alert(self._features(self._clock()))
                if b is not None:
                    w = self.brain.weight
                    score = (1.0 - w) * score + w * b
        if evidence >= s.alert_min_evidence and score <= s.batch_below:
            return "later"
        return None

    # -- housekeeping -------------------------------------------------------------------------
    def tick(self, now: float | None = None) -> None:
        """Settle what timed out, decay, save, maybe retrain. Cheap; the body
        calls it every frame and it does real work every few seconds."""
        if not self.settings.enabled:
            return
        now = self._clock() if now is None else now
        if now - self._ticked_at < TICK_EVERY_S:
            return
        self._ticked_at = now
        window = self.settings.reward_window_s
        with self._lock:
            for p in list(self._pending):
                if now - p.at > window:
                    self._settle(p, None)
            wall = self._ctx.wall or self._wall()
            self.bandit.decay(wall)
            self.alerts.decay(wall)
            key_idle = self._ctx.key_idle
        if self.brain is not None:
            self.brain.maybe_train(wall, key_idle)
        if self._dirty and now - self._saved_at >= SAVE_EVERY_S:
            self.save()

    # -- persistence -------------------------------------------------------------------------
    def save(self) -> None:
        path = self.settings.state_file
        self._saved_at = self._clock()
        if path is None:
            return
        with self._lock:
            data = {"version": 1, "bandit": self.bandit.to_dict(),
                    "alerts": self.alerts.to_dict(), "relevance": self.relevance.to_dict(),
                    "stats": dict(self.stats)}
            self._dirty = False
        try:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                           encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            log.warning("learn: could not save %s", path, exc_info=True)

    def load(self) -> None:
        path = self.settings.state_file
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("learn: %s unreadable; starting fresh", path)
                data = {}
            if isinstance(data, dict):
                self.bandit.load(data.get("bandit"))
                self.alerts.load(data.get("alerts"))
                self.relevance.load(data.get("relevance"))
                st = data.get("stats")
                if isinstance(st, dict):
                    for k in self.stats:
                        try:
                            self.stats[k] = int(st.get(k, 0))
                        except (TypeError, ValueError):
                            pass
        if self.brain is not None:
            self.brain.load()

    def close(self) -> None:
        """Settle nothing (an unanswered decision is not a lesson at exit);
        just keep what was learned."""
        self.save()

    # -- "¿qué aprendiste de mí?" ---------------------------------------------------------------
    def summary(self) -> str:
        """A short Spanish summary of what she learned, for the voice."""
        if not self.settings.enabled:
            return "No estoy aprendiendo nada: lo tenés apagado en la configuración."
        with self._lock:
            fav = [kv for kv in self.bandit.favourites(12)
                   if not kv[0].startswith(INSTINCT_ARM)][:2]
            least = [kv for kv in self.bandit.least(12)
                     if not kv[0].startswith(INSTINCT_ARM)][:1]
            liked = self.alerts.liked(1)
            disliked = self.alerts.disliked(2)
            brain = self.brain
            total = self.stats.get("rewards", 0)
        parts: list[str] = []
        words = lambda names: _join([BEHAVIOUR_WORDS.get(n, n) for n in names])  # noqa: E731
        if fav:
            parts.append(f"Te gusta cuando me pongo a {words([a for a, _ in fav])}")
        if least and (not fav or least[0][0] not in dict(fav)):
            parts.append(f"lo de {BEHAVIOUR_WORDS.get(least[0][0], least[0][0])} mucho no te va")
        if liked:
            parts.append(f"te interesan los avisos sobre {ALERT_WORDS.get(liked[0][0], liked[0][0])}")
        if disliked:
            parts.append("los de " + _join([ALERT_WORDS.get(k, k) for k, _ in disliked])
                         + " te los junto para después")
        if self.settings.interests:
            parts.append("de noticias te separo lo de " + _join(list(self.settings.interests[:3])))
        if brain is not None and brain.net is not None and brain.weight > 0:
            parts.append(f"y ya me sé tus horarios con {brain.samples} momentos anotados")
        if not parts:
            if total < 5:
                return "Todavía poco: recién te estoy conociendo. Haceme mimos y te digo."
            return "Te vengo mirando, pero todavía no tengo nada claro."
        text = ", ".join(parts)
        return text[0].upper() + text[1:] + "."


def _join(items: list[str]) -> str:
    items = list(dict.fromkeys(i for i in items if i))
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]
