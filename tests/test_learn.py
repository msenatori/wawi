"""What Wawi learns from her owner: the bandit, the alert scores, headline
relevance, the tiny brain, and the hooks that feed them. Everything runs on
fake clocks and seeded generators: no model, no download, no GPU."""

from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import pytest

from wawi import learn, mascot
from wawi.learn import alerts as A
from wawi.learn import brain as B
from wawi.learn.bandit import BehaviourBandit, context, part_of_day
from wawi.learn.learner import Learner
from wawi.learn.relevance import BackgroundEmbedder, Relevance, stems
from wawi.learn.settings import LearnSettings
from wawi.reactions import Event, Reaction, ReactionBus
from wawi.reflexes import QuietState, Reflexes

WALL = 1_800_000_000.0  # a Friday afternoon, local time varies by TZ: tests avoid relying on it


def settings(tmp_path, **kw):
    base = {"state_file": str(tmp_path / "learn.json"), "brain_file": str(tmp_path / "b.npz"),
            "life_log": str(tmp_path / "life.jsonl"), "embed_model": "none"}
    return LearnSettings.from_dict(base | kw)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(autouse=True)
def _no_installed_learner():
    learn.install(None)
    yield
    learn.install(None)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


class TestSettings:
    def test_defaults_are_cheap_and_on(self):
        s = LearnSettings.from_dict(None)
        assert s.enabled and s.behaviours and s.alerts and s.brain
        assert (s.factor_min, s.factor_max) == (0.5, 2.0)
        assert s.brain_max_weight <= 0.9

    def test_garbage_falls_back_and_is_clamped(self):
        s = LearnSettings.from_dict({"factor_min": "x", "factor_max": 0.2, "brain_max_weight": 5,
                                     "interests": "IA", "embed_model": "none",
                                     "state_file": None})
        assert s.factor_min == 0.5 and s.factor_max == 1.0
        assert s.brain_max_weight == 0.9
        assert s.interests == ("IA",) and s.embed_model == "" and s.state_file is None

    def test_config_carries_the_raw_section(self, tmp_path):
        from wawi import config as config_mod

        path = tmp_path / "config.yaml"
        path.write_text("learn:\n  interests: [IA]\n")
        assert config_mod.load(path).learn == {"interests": ["IA"]}


# --------------------------------------------------------------------------- #
# The bandit
# --------------------------------------------------------------------------- #

ARMS = ["wag", "roll", "yawn", "dig", "stare", "sniff"]


def simulate(bandit, ctx, likes, rounds, rng):
    """The animator's dice (uniform base weights x bandit factors) against an
    owner who reacts to *likes* 70% of the time and to the rest 5%."""
    for _ in range(rounds):
        f = bandit.factors(ARMS, ctx)
        w = np.array([f[a] for a in ARMS])
        arm = ARMS[rng.choice(len(ARMS), p=w / w.sum())]
        p = 0.7 if arm in likes else 0.05
        bandit.update(arm, ctx, 1.0 if rng.random() < p else 0.0)


class TestBandit:
    def test_context_buckets(self):
        assert part_of_day(9) == "mañana" and part_of_day(3) == "madrugada"
        assert context(15, "bored") == "tarde:bored"

    def test_converges_to_what_the_owner_enjoys(self):
        rng = np.random.default_rng(7)
        b = BehaviourBandit(rng=np.random.default_rng(1))
        simulate(b, "tarde:content", {"wag", "roll"}, 400, rng)
        means = {a: np.mean([b.factors(ARMS, "tarde:content")[a] for _ in range(200)])
                 for a in ARMS}
        assert means["wag"] > 1.4 and means["roll"] > 1.4
        assert max(means[a] for a in ARMS if a not in ("wag", "roll")) < 0.9
        assert [a for a, _ in b.favourites(2)] in (["wag", "roll"], ["roll", "wag"])

    def test_factors_are_always_clamped_and_never_zero(self):
        b = BehaviourBandit(rng=np.random.default_rng(3))
        for _ in range(200):
            b.update("wag", "tarde:content", 1.0)
            b.update("dig", "tarde:content", -1.0)
        for _ in range(50):
            f = b.factors(ARMS, "tarde:content")
            assert all(0.5 <= v <= 2.0 for v in f.values())
        assert set(f) == set(ARMS)  # nothing removed

    def test_contexts_borrow_but_stay_their_own(self):
        b = BehaviourBandit(rng=np.random.default_rng(0))
        for _ in range(40):
            b.update("wag", "mañana:happy", 1.0)
        for _ in range(20):
            b.update("wag", "noche:sleepy", -1.0)
        assert b.expected("wag", "mañana:happy") > 0.7
        assert b.expected("wag", "noche:sleepy") < 0.5
        # An unseen context inherits a little of the overall evidence.
        assert 0.5 < b.expected("wag", "tarde:bored") < b.expected("wag", "mañana:happy")

    def test_tastes_change_evidence_decays(self):
        b = BehaviourBandit(half_life_days=10)
        b.decay(WALL)
        for _ in range(20):
            b.update("wag", "tarde:content", 1.0)
        b.decay(WALL + 10 * 86400)
        assert b.arms["wag"][0] == pytest.approx(10.0)
        b.decay(WALL + 400 * 86400)
        assert "wag" not in b.arms  # forgotten entirely, not a ghost forever

    def test_persistence_round_trip_and_bad_data(self):
        b = BehaviourBandit()
        b.update("wag", "tarde:content", 1.0)
        b.update("dig", "tarde:content", 0.0)
        raw = json.loads(json.dumps(b.to_dict()))
        c = BehaviourBandit()
        c.load(raw)
        assert c.arms == b.arms and c.cells == b.cells and c.updates == 2
        c.load({"arms": {"x": ["nan?", 1]}, "cells": {"y": [-1, 2]}, "updates": "z"})
        assert c.arms == {} and c.cells == {} and c.updates == 0


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #

HOT = Event("weather.hot", "info", "weather:hot", "Calor", {"max": 35})


def news(title, feed="Diario"):
    return Event("rss.item", "info", f"rss:{abs(hash(title))}", title, {"title": title, "feed": feed})


class TestAlertScores:
    def test_no_evidence_is_no_opinion(self):
        a = A.AlertLearner()
        assert a.score(HOT) == (0.0, 0.0)

    def test_dismissals_go_negative_engagement_positive(self):
        a = A.AlertLearner()
        for _ in range(6):
            a.record(HOT, "dismissed")
        s, ev = a.score(HOT)
        assert s < -0.45 and ev >= 5
        for _ in range(20):
            a.record(HOT, "engaged")
        assert a.score(HOT)[0] > 0.3

    def test_ignoring_is_weak_and_reading_is_nothing(self):
        a = A.AlertLearner()
        for _ in range(5):
            a.record(HOT, "ignored")
            a.record(HOT, "read")
        s, _ = a.score(HOT)
        assert -0.4 < s < 0

    def test_per_feed_tastes(self):
        a = A.AlertLearner()
        for _ in range(8):
            a.record(news("x", "Deportes"), "dismissed")
            a.record(news("y", "Tecno"), "engaged")
        assert a.score(news("z", "Deportes"))[0] < 0 < a.score(news("z", "Tecno"))[0]
        assert A.source_of(news("t", "Tecno")) == "Tecno"
        assert A.source_of(HOT) == "weather:hot"

    def test_decay_and_round_trip(self):
        a = A.AlertLearner(half_life_days=1)
        a.decay(WALL)
        a.record(HOT, "dismissed")
        b = A.AlertLearner()
        b.load(json.loads(json.dumps(a.to_dict())))
        assert b.kinds == a.kinds
        a.decay(WALL + 86400)
        assert a.kinds["weather.hot"][1] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Relevance
# --------------------------------------------------------------------------- #


class FakeEmbedder:
    """Bag of topic words -> a 3-d vector (tech, sport, food)."""

    ready = True
    TOPICS = {"tech": ("modelo", "python", "ia", "program", "software", "codigo", "openai"),
              "sport": ("boca", "river", "futbol", "gol", "partido"),
              "food": ("receta", "empanada", "asado")}

    def encode(self, texts):
        out = []
        for t in texts:
            words = " ".join(stems(t)) + " " + t.lower()
            v = [sum(w in words for w in ws) for ws in self.TOPICS.values()]
            out.append(np.asarray(v, float) + 0.01)
        return np.asarray(out)


class TestRelevance:
    def test_nothing_known_is_none(self):
        assert Relevance().score("Lo que sea") == (None, False)

    def test_keywords_fold_accents_and_stem(self):
        r = Relevance(["programación", "IA"])
        assert r.score("Los programadores celebran")[0] >= 0.7
        assert r.score("La IA llega a todo")[0] >= 0.7
        score, confident = r.score("Subió la nafta")
        assert score < 0.5 and not confident  # a keyword miss alone never drops

    def test_two_letter_words_only_in_capitals(self):
        assert "ia" in stems("La IA") and stems("ya va") == []

    def test_learned_words_after_enough_labels(self):
        r = Relevance()
        for i in range(12):
            r.learn(f"Boca gana otra vez {i}", False)
            r.learn(f"Nuevo lenguaje Rust versión {i}", True)
        assert r.learned("Rust sale con mejoras") > 0.6
        assert r.learned("Boca juega mañana") < 0.4
        assert r.score("Boca juega")[1] is True

    def test_embeddings_and_learned_centroid(self):
        r = Relevance(["inteligencia artificial y software"], embedder=FakeEmbedder())
        hi, conf = r.score("OpenAI lanza un modelo de codigo")
        lo, _ = r.score("Receta de empanada")
        assert conf and hi > 0.7 > lo
        # Without interests, engaged headlines become the centroid.
        r2 = Relevance(embedder=FakeEmbedder())
        r2.learn("Gol de Boca en el partido", True)
        assert r2.embedded("River gana el partido") > 0.8

    def test_state_round_trip(self):
        r = Relevance(embedder=FakeEmbedder())
        for i in range(11):
            r.learn(f"python {i}", True)
        s = Relevance()
        s.load(json.loads(json.dumps(r.to_dict())))
        assert s.labels == 11 and s.learned("python") == pytest.approx(r.learned("python"), abs=1e-3)
        assert s.centroid is not None
        s.load({"w": {"x": 1}})
        assert s.labels == 0

    def test_background_embedder_failure_falls_back(self):
        def boom(_):
            raise ImportError("model2vec")

        emb = BackgroundEmbedder("m", loader=boom).start()
        emb._thread.join(2)
        assert emb.failed and not emb.ready
        r = Relevance(["python"], embedder=emb)
        assert r.score("python 4")[0] >= 0.7  # keywords still work

    def test_background_embedder_loads(self):
        emb = BackgroundEmbedder("m", loader=lambda _: FakeEmbedder()).start()
        emb._thread.join(2)
        assert emb.ready and emb.encode(["python"]).shape == (1, 3)


# --------------------------------------------------------------------------- #
# The learner: matching what she did with what the owner did
# --------------------------------------------------------------------------- #


def make_learner(tmp_path, clock=None, **kw):
    clock = clock or Clock()
    lr = Learner(settings(tmp_path, **kw), actions=ARMS, clock=clock, wall=lambda: WALL,
                 seed=1, embedder=None)
    return lr, clock


class TestLearner:
    def test_touch_soon_after_a_trick_is_a_reward(self, tmp_path):
        lr, clock = make_learner(tmp_path)
        lr.behaviour_started("wag")
        clock.t += 5
        lr.signal("touch")
        assert lr.bandit.arms["wag"] == [1.0, 0.0]
        assert lr.stats["positive"] == 1

    def test_nothing_happens_is_neutral_after_the_window(self, tmp_path):
        lr, clock = make_learner(tmp_path)
        lr.behaviour_started("dig")
        clock.t += 30
        lr.signal("touch")  # too late: not about the dig
        lr.tick()
        assert lr.bandit.arms["dig"] == [0.0, 0.25]

    def test_dragging_her_away_is_a_slight_negative(self, tmp_path):
        lr, clock = make_learner(tmp_path)
        lr.behaviour_started("roll")
        clock.t += 2
        lr.signal("drag")
        assert lr.bandit.arms["roll"] == [0.0, 0.5]

    def test_quick_bubble_click_is_a_dismissal_late_one_is_not(self, tmp_path):
        lr, clock = make_learner(tmp_path)
        lr.alert_shown(HOT, "Hoy hace calor")
        clock.t += 1
        lr.signal("bubble_click")
        assert lr.alerts.kinds["weather.hot"] == [0.0, 1.0]
        lr.alert_shown(HOT, "Hoy hace calor")
        clock.t += 10
        lr.signal("bubble_click")
        assert lr.alerts.kinds["weather.hot"] == [0.0, 1.0]  # read, then closed

    def test_a_new_bubble_settles_the_old_as_ignored(self, tmp_path):
        lr, clock = make_learner(tmp_path)
        lr.alert_shown(HOT, "calor")
        lr.alert_shown(news("x"), "x")
        assert lr.alerts.kinds["weather.hot"] == [0.0, 0.3]

    def test_engaged_headlines_teach_relevance(self, tmp_path):
        lr, clock = make_learner(tmp_path)
        ev = news("Python 4 anunciado")
        lr.alert_shown(ev, "Python 4 anunciado")
        clock.t += 3
        lr.signal("talk")
        assert lr.relevance.labels == 1
        lr.alert_shown(news("Boca"), "Tenés 4 noticias nuevas")  # a batch: not about one title
        clock.t += 1
        lr.signal("bubble_click")
        assert lr.relevance.labels == 1

    def test_bias_reweights_only_the_arms_and_keeps_the_mood(self, tmp_path):
        lr, _ = make_learner(tmp_path)
        for _ in range(60):
            lr.bandit.update("wag", lr._context_key(), 1.0)
        mood = {"wag": 2.0, "hungry_bowl": 3.0}
        out = lr.bias(mood, ["wag", "dig"])
        assert out["hungry_bowl"] == 3.0  # untouched: not an arm now
        assert 2.0 * 0.5 <= out["wag"] <= 2.0 * 2.0 and 0.5 <= out["dig"] <= 2.0
        assert set(out) == {"wag", "dig", "hungry_bowl"}

    def test_disabled_learns_nothing(self, tmp_path):
        lr, clock = make_learner(tmp_path, enabled=False)
        lr.behaviour_started("wag")
        lr.signal("touch")
        assert lr.bandit.arms == {} and lr.bias({"x": 1.0}, ["wag"]) == {"x": 1.0}
        assert lr.triage(HOT) is None

    def test_save_load_round_trip(self, tmp_path):
        lr, clock = make_learner(tmp_path)
        lr.behaviour_started("wag")
        lr.signal("pet")
        lr.save()
        again, _ = make_learner(tmp_path)
        again.load()
        assert again.bandit.arms == lr.bandit.arms and again.stats["positive"] == 1

    def test_corrupt_state_starts_fresh(self, tmp_path):
        (tmp_path / "learn.json").write_text("{nope")
        lr, _ = make_learner(tmp_path)
        lr.load()
        assert lr.bandit.arms == {}

    def test_tick_saves_now_and_then(self, tmp_path):
        lr, clock = make_learner(tmp_path)
        lr.behaviour_started("wag")
        lr.signal("touch")
        clock.t += 200
        lr.tick()
        assert (tmp_path / "learn.json").exists()

    def test_summary_speaks_spanish(self, tmp_path):
        lr, clock = make_learner(tmp_path, interests=["IA"])
        assert "conociendo" in Learner(settings(tmp_path, interests=[]), embedder=None).summary()
        for _ in range(6):
            lr.behaviour_started("wag")
            lr.signal("touch")
            lr.alert_shown(HOT, "calor")
            lr.signal("bubble_click")
        text = lr.summary()
        assert "mover la cola" in text and "calor" in text and "IA" in text
        assert text.endswith(".")


class TestTriage:
    def test_urgent_and_warning_never_ask(self, tmp_path):
        lr, _ = make_learner(tmp_path)
        for _ in range(20):
            lr.alerts.record(Event("web.down", "urgent"), "dismissed")
        assert lr.triage(Event("web.down", "urgent", "web:x")) is None
        assert lr.triage(Event("disk.low", "warning")) is None

    def test_brushed_away_kinds_get_batched(self, tmp_path):
        lr, _ = make_learner(tmp_path)
        assert lr.triage(HOT) is None
        for _ in range(8):
            lr.alerts.record(HOT, "dismissed")
        assert lr.triage(HOT) == "later"

    def test_news_relevance(self, tmp_path):
        lr, _ = make_learner(tmp_path, interests=["programación"])
        assert lr.triage(news("Programadores en alza")) == "now"
        assert lr.triage(news("Subió la nafta")) is None  # keyword miss: batch as before
        lr.relevance.embedder = FakeEmbedder()
        assert lr.triage(news("Receta de empanada")) == "ignore"  # confident and off-topic

    def test_module_hook_is_safe_without_a_learner(self):
        assert learn.triage(HOT) is None
        learn.signal("touch")  # no-op
        assert "apagado" in learn.summary()


# --------------------------------------------------------------------------- #
# The reflex layer's triage hook
# --------------------------------------------------------------------------- #


class FixedQuiet:
    def check(self):
        return QuietState()


def drain(bus):
    out = []
    while (r := bus.pop()) is not None:
        out.append(r)
    return out


def reflexes(verdicts):
    bus, clock = ReactionBus(), Clock()
    asked = []

    def triage(ev):
        asked.append(ev.kind)
        return verdicts.get(ev.kind)

    rx = Reflexes({}, bus=bus, clock=clock, quiet=FixedQuiet(), triage=triage)
    rx.attach()
    return rx, bus, clock, asked


class TestReflexTriage:
    def test_later_batches_a_kind_that_normally_shows(self):
        rx, bus, clock, _ = reflexes({"weather.hot": "later"})
        rx.on_event(HOT)
        assert drain(bus) == []
        clock.t += 61
        rx.tick()
        (r,) = drain(bus)
        assert r.costume == "sunglasses"  # delivered later, intact

    def test_later_bundles_several_with_a_generic_bubble(self):
        rx, bus, clock, _ = reflexes({"clock.lunch": "later"})
        for i in range(3):
            rx.on_event(Event("clock.lunch", "info", f"lunch:{i}"))
        clock.t += 61
        rx.tick()
        (r,) = drain(bus)
        assert "3 avisos" in r.bubble

    def test_now_gives_a_headline_its_own_bubble(self):
        rx, bus, _, _ = reflexes({"rss.item": "now"})
        rx.on_event(news("Python 4"))
        (r,) = drain(bus)
        assert r.bubble == "Python 4"

    def test_ignore_drops(self):
        rx, bus, clock, _ = reflexes({"rss.item": "ignore"})
        rx.on_event(news("Boca"))
        clock.t += 61
        rx.tick()
        assert drain(bus) == []

    def test_urgent_warning_and_stateful_rules_never_ask(self):
        rx, bus, _, asked = reflexes({})
        rx.on_event(Event("web.down", "urgent", "web:u", "x", {"url": "u", "host": "h"}))
        rx.on_event(Event("disk.low", "warning", "disk:/"))
        rx.on_event(Event("service.up", "notice", "svc"))  # cancel_key: stateful
        assert asked == []

    def test_a_broken_triage_is_ignored(self):
        bus = ReactionBus()

        def boom(_):
            raise RuntimeError

        rx = Reflexes({}, bus=bus, clock=Clock(), quiet=FixedQuiet(), triage=boom)
        rx.attach()
        rx.on_event(HOT)
        assert len(drain(bus)) == 1

    def test_default_hook_uses_the_installed_learner(self, tmp_path):
        lr, _ = make_learner(tmp_path)
        for _ in range(8):
            lr.alerts.record(HOT, "dismissed")
        learn.install(lr)
        bus = ReactionBus()
        rx = Reflexes({}, bus=bus, clock=Clock(), quiet=FixedQuiet())
        rx.attach()
        rx.on_event(HOT)
        assert drain(bus) == []  # batched by what she learned


# --------------------------------------------------------------------------- #
# The tiny brain
# --------------------------------------------------------------------------- #


def ctx_x(hour, typing):
    """Features for an hour of a fixed day, owner typing or not."""
    lt = time.localtime(WALL)
    midnight = WALL - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
    return B.features({"wall": midnight + hour * 3600, "mood": "content",
                       "key_idle": 2.0 if typing else 900.0})


def synthetic_log(log, n, seed=0):
    """An owner who loves 'wag' in the morning and 'roll' at night, and who
    brushes alerts away while typing."""
    rng = random.Random(seed)
    for i in range(n):
        hour = rng.choice([9, 10, 21, 22])
        typing = rng.random() < 0.5
        x = ctx_x(hour, typing)
        if rng.random() < 0.3:
            r = -1.0 if typing else 1.0
            log.append(WALL + i, "a", B.ALERT, r if rng.random() < 0.9 else -r, x)
            continue
        arm = rng.choice(ARMS)
        liked = (arm == "wag" and hour < 12) or (arm == "roll" and hour > 12)
        r = 1.0 if rng.random() < (0.85 if liked else 0.05) else 0.0
        log.append(WALL + i, "b", arm, r, x)


class TestBrain:
    def test_features_are_20_bounded_floats(self):
        x = B.features({"wall": WALL, "mood": "bored", "needs": {"food": 20},
                        "ignored_for": 1e6, "key_idle": math.inf, "recent_alerts": 99})
        assert x.shape == (B.N_FEATURES,) and np.all(np.abs(x) <= 1.0)
        assert B.features({}).shape == (B.N_FEATURES,)

    def test_no_data_no_voice(self, tmp_path):
        brain = B.Brain(ARMS, lifelog=B.LifeLog(tmp_path / "l.jsonl"))
        assert brain.weight == 0.0
        assert brain.factors(ctx_x(9, False), ARMS) == {a: 1.0 for a in ARMS}
        assert brain.alert(ctx_x(9, False)) is None
        assert brain.train(WALL) is False  # nothing to learn from

    def test_learns_synthetic_preferences_deterministically(self, tmp_path):
        log = B.LifeLog(tmp_path / "l.jsonl", max_bytes=10_000_000)
        synthetic_log(log, 1500)
        brain = B.Brain(ARMS, lifelog=log, min_samples=300, max_weight=0.5, seed=0)
        assert brain.train(WALL + 2000)
        assert brain.skill > 0.2 and brain.samples == 1500
        assert 0.05 < brain.weight <= 0.5
        morning = brain.factors(ctx_x(9, False), ARMS)
        night = brain.factors(ctx_x(22, False), ARMS)
        assert morning["wag"] > 1.0 > morning["dig"]
        assert night["roll"] > 1.0 and night["roll"] > night["wag"]
        assert all(0.5 ** 0.5 <= v <= 2.0 ** 0.5 for v in {**morning, **night}.values())
        assert brain.alert(ctx_x(10, True)) < 0 < brain.alert(ctx_x(10, False))
        # Deterministic: the same data and seed give the same brain.
        twin = B.Brain(ARMS, lifelog=log, min_samples=300, max_weight=0.5, seed=0)
        twin.train(WALL + 2000)
        assert np.allclose(twin.net.W1, brain.net.W1)

    def test_whispers_below_min_samples_and_never_exceeds_the_cap(self, tmp_path):
        brain = B.Brain(ARMS, max_weight=0.4, min_samples=300)
        brain.net, brain.skill = B.MLP(B.N_FEATURES, B.HIDDEN, len(brain.actions)), 0.9
        brain.samples = 150
        assert brain.weight == pytest.approx(0.1 * 0.4 * 0.5)
        brain.samples = 10_000
        assert brain.weight == pytest.approx(0.4)
        brain.skill = -0.2
        assert brain.weight == 0.0  # worse than averages: silent

    def test_noise_earns_no_weight(self, tmp_path):
        log = B.LifeLog(tmp_path / "l.jsonl", max_bytes=10_000_000)
        rng = random.Random(3)
        for i in range(800):
            log.append(WALL + i, "b", rng.choice(ARMS), rng.choice([0.0, 1.0]),
                       ctx_x(rng.randrange(24), rng.random() < 0.5))
        brain = B.Brain(ARMS, lifelog=log, seed=0)
        brain.train(WALL + 900)
        assert brain.weight < 0.06

    def test_log_rotates_by_size_and_age(self, tmp_path):
        log = B.LifeLog(tmp_path / "l.jsonl", max_days=1, max_bytes=20_000)
        x = ctx_x(9, False)
        for i in range(400):
            log.append(WALL + i, "b", "wag", 1.0, x)
        size = (tmp_path / "l.jsonl").stat().st_size
        assert size <= 20_000
        recs = log.records(WALL + 400)
        assert recs and recs[-1]["t"] == pytest.approx(WALL + 399)
        assert log.records(WALL + 3 * 86400) == []  # all too old to train on

    def test_log_skips_garbage_lines(self, tmp_path):
        p = tmp_path / "l.jsonl"
        p.write_text('{"t": 1}\nnot json\n{"t": %f, "h": "b", "a": "wag", "r": 1, "x": %s}\n'
                     % (WALL, json.dumps([0.0] * B.N_FEATURES)))
        assert len(B.LifeLog(p).records(WALL)) == 1

    def test_retrains_only_when_idle_rarely_and_with_new_data(self, tmp_path):
        log = B.LifeLog(tmp_path / "l.jsonl", max_bytes=10_000_000)
        synthetic_log(log, 400)
        brain = B.Brain(ARMS, lifelog=log, retrain_every_s=3600, retrain_idle_s=600)
        runs = []
        spawn = lambda fn: (runs.append(1), fn())  # noqa: E731 - run inline, deterministic
        assert not brain.maybe_train(WALL, key_idle=5.0, spawn=spawn)  # owner typing
        assert brain.maybe_train(WALL, key_idle=900.0, spawn=spawn)
        assert runs == [1] and brain.net is not None
        assert not brain.maybe_train(WALL + 60, key_idle=900.0, spawn=spawn)  # too soon
        assert not brain.maybe_train(WALL + 7200, key_idle=900.0, spawn=spawn)  # no new data
        for i in range(B.RETRAIN_NEW):
            brain.record(WALL + 7000 + i, "b", "wag", 1.0, ctx_x(9, False))
        assert brain.maybe_train(WALL + 7200, key_idle=900.0, spawn=spawn)
        assert len(runs) == 2

    def test_save_and_load(self, tmp_path):
        log = B.LifeLog(tmp_path / "l.jsonl", max_bytes=10_000_000)
        synthetic_log(log, 500)
        path = tmp_path / "b.npz"
        brain = B.Brain(ARMS, path=path, lifelog=log)
        brain.train(WALL + 600)
        again = B.Brain(ARMS, path=path)
        again.load()
        assert again.samples == 500 and np.allclose(again.net.W2, brain.net.W2)
        other = B.Brain(ARMS + ["new_trick"], path=path)
        other.load()
        assert other.net is None  # different outputs: retrain, do not misread
        path.write_bytes(b"garbage")
        broken = B.Brain(ARMS, path=path)
        broken.load()
        assert broken.net is None

    def test_learner_logs_every_settled_decision(self, tmp_path):
        lr, clock = make_learner(tmp_path)
        lr.behaviour_started("wag")
        lr.signal("touch")
        lr.alert_shown(HOT, "calor")
        clock.t += 60
        lr.tick()
        recs = lr.brain.log.records(WALL)
        assert [(r["h"], r["a"], r["r"]) for r in recs] == [("b", "wag", 1.0), ("a", B.ALERT, -0.3)]

    def test_brain_is_tiny(self, tmp_path):
        brain = B.Brain(learn.actions())
        net = B.MLP(B.N_FEATURES, B.HIDDEN, len(brain.actions))
        assert sum(p.nbytes for p in net.params()) < 10_000


# --------------------------------------------------------------------------- #
# Hooks in the body
# --------------------------------------------------------------------------- #


class Spy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kw):
            self.calls.append((name, args, kw))
            if name == "bias":
                return dict(args[0])
            return None

        return record

    def names(self, name):
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def rig(tmp_path):
    from wawi.pet import api
    from wawi.pet.life import Env, PetLife
    from wawi.pet.needs import NeedsStore
    from wawi.pet.settings import PetSettings

    api.attach(None)
    spy = Spy()
    anim = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(2))
    life = PetLife(PetSettings.from_dict({"wander": False, "lonely_after_min": 0,
                                          "state_file": str(tmp_path / "s.json")}),
                   anim, rng=random.Random(5), store=NeedsStore(tmp_path / "s.json"),
                   wall=WALL, learner=spy)
    life.start()

    def tick(t):
        return life.tick(1 / 30, Env(now=t, wall=WALL + t, attached=True))

    yield life, anim, spy, tick
    api.attach(None)


class TestBodyHooks:
    def test_touch_pet_talk_and_commands_are_signals(self, rig):
        life, anim, spy, tick = rig
        tick(1.0)
        life.touched(1.0)
        life.petted(0.1)
        life.command("feed")
        anim.key = "listening"
        tick(2.0)
        kinds = [c[1][0] for c in spy.names("signal")]
        assert kinds == ["touch", "pet", "feed", "talk"]

    def test_picked_behaviours_are_reported(self, rig):
        life, anim, spy, tick = rig
        tick(3.0)
        anim._picked("wag")
        assert spy.names("behaviour_started")[0][1] == ("wag", 3.0)

    def test_a_real_idle_pick_reaches_the_learner(self, rig):
        life, anim, spy, tick = rig
        t = 0.0
        from wawi.pet import dog as D
        from wawi.state import State

        # Her picks come from the dog model (wawi/pet/dog.py): up and about,
        # she does something within seconds.
        tick(t)
        life.dog._enter(t, D.ACTIVE, 600.0, "test")
        for _ in range(int(200 / (1 / 30))):
            t += 1 / 30
            tick(t)
            anim.update(1 / 30, State.IDLE)
            if spy.names("behaviour_started"):
                break
        assert spy.names("behaviour_started")

    def test_mood_bias_goes_through_the_learner(self, rig):
        life, anim, spy, tick = rig
        tick(10.0)
        (call,) = spy.names("bias")[:1]
        assert "wag" in call[1][1]  # the arms
        assert spy.names("observe")

    def test_event_bubbles_are_watched_and_can_be_dismissed(self, rig):
        life, anim, spy, tick = rig
        life.react(Reaction(bubble="Hoy hace calor", source=HOT))
        out = tick(1.0)
        assert out.bubble == "Hoy hace calor"
        assert spy.names("alert_shown")[0][1][:2] == (HOT, "Hoy hace calor")
        assert life.dismiss_bubble(1.5)
        assert tick(1.6).bubble is None
        assert ("signal", ("bubble_click", 1.5), {}) in spy.calls
        assert not life.dismiss_bubble(2.0)  # nothing left to dismiss

    def test_bubbles_without_an_event_are_not_alerts(self, rig):
        life, anim, spy, tick = rig
        life.react(Reaction(bubble="hola"))
        tick(1.0)
        assert spy.names("alert_shown") == []

    def test_a_broken_learner_never_breaks_her(self, rig, tmp_path):
        life, anim, spy, tick = rig

        class Broken:
            def __getattr__(self, name):
                raise RuntimeError(name)

        life.learner = Broken()
        life.touched(1.0)
        tick(1.0)
        tick(7.0)
        life.stop(WALL)

    def test_without_a_learner_she_is_exactly_as_before(self, tmp_path):
        from wawi.pet.life import Env, PetLife
        from wawi.pet.needs import NeedsStore, mood_bias
        from wawi.pet.settings import PetSettings

        anim = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(2))
        life = PetLife(PetSettings.from_dict({"state_file": str(tmp_path / "s.json")}), anim,
                       store=NeedsStore(tmp_path / "s.json"), wall=WALL)
        life.tick(1 / 30, Env(now=1.0, wall=WALL, attached=True))
        assert anim.bias == mood_bias(life.mood)


class TestBuild:
    def test_build_from_config_and_off_switch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        class Cfg:
            learn = {"embed_model": "none"}

        lr = learn.build(Cfg())
        assert isinstance(lr, Learner) and lr.relevance.embedder is None
        assert lr.brain is not None and "hungry_bowl" in lr.brain.actions

        class Off:
            learn = {"enabled": False}

        assert learn.build(Off()) is None

    def test_no_model2vec_means_keywords(self, tmp_path, monkeypatch):
        from wawi.learn import learner as L

        monkeypatch.setattr(L, "model2vec_available", lambda: False)
        lr = Learner(settings(tmp_path, embed_model="minishlab/x", interests=["IA"]))
        assert lr.relevance.embedder is None
        assert lr.triage(news("La IA avanza")) == "now"

    def test_voice_tool(self, tmp_path):
        from wawi import tools

        assert "what_learned" in {t["name"] for t in tools.MIND_TOOLS}
        r = tools.dispatch("what_learned", {})
        assert r.ok is False and "apagado" in r.speak
        lr, _ = make_learner(tmp_path)
        learn.install(lr)
        r = tools.dispatch("what_learned", {})
        assert r.ok and "conociendo" in r.speak
