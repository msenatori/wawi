"""Wawi's instincts: the activity observer, the pattern miner, the instinct
engine, the self-model and the voice tool. Whole synthetic weeks run on
seeded generators and explicit wall clocks (local time built with mktime, so
the tests do not depend on the machine's time zone)."""

from __future__ import annotations

import inspect
import json
import random
import time

import pytest

from wawi import instinct, learn, tools
from wawi.instinct import categories as C
from wawi.instinct import observer as O
from wawi.instinct import selfmodel
from wawi.instinct import tool as T
from wawi.instinct.engine import InstinctEngine
from wawi.instinct.observer import ActivityObserver, ActivityStore, Sample
from wawi.instinct.patterns import PatternMiner, confidence
from wawi.instinct.settings import INSTINCTS, InstinctSettings
from wawi.learn.learner import INSTINCT_ARM, Learner
from wawi.learn.settings import LearnSettings
from wawi.mind import local
from wawi.pet.acting import clip_for_behaviour
from wawi.reactions import Event, ReactionBus

VSCODE = ("com.microsoft.VSCode", "Code")
SLACK = ("com.tinyspeck.slackmacgap", "Slack")
CHROME = ("com.google.Chrome", "Google Chrome")
ZOOM = ("us.zoom.xos", "zoom.us")


def at(y, mo, d, h=0, mi=0, s=0) -> float:
    return time.mktime((y, mo, d, h, mi, s, 0, 0, -1))


# 2026-03-02 is a Monday.
MONDAY = (2026, 3, 2)


def day_of(n: int) -> tuple[int, int, int]:
    """The n-th day after MONDAY, as (y, m, d)."""
    t = at(*MONDAY, 12) + n * 86400
    lt = time.localtime(t)
    return lt.tm_year, lt.tm_mon, lt.tm_mday


@pytest.fixture(autouse=True)
def _nothing_installed():
    instinct.install(None)
    learn.install(None)
    yield
    instinct.install(None)
    learn.install(None)


class Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def run(obs: ActivityObserver, start: float, end: float, app=VSCODE, step: float = 30.0,
        away: bool = False, switch_every: float | None = None) -> float:
    """Feed samples from *start* to *end*: at the keyboard in *app*, or away."""
    t = start
    toggle = False
    while t < end:
        if away:
            obs.ingest(Sample(t, None, None, key_idle=1e5, mouse_idle=1e5))
        else:
            cur = app
            if switch_every is not None:
                toggle = not toggle
                cur = app if toggle else CHROME
            obs.ingest(Sample(t, cur[0], cur[1], key_idle=2.0, mouse_idle=10.0))
        t += step
    return t


def workday(obs: ActivityObserver, ymd, rng: random.Random, *, slack=True, lunch=True,
            late=False) -> None:
    """A routine weekday: 9:00 start, Slack at 10, a focus block, lunch at
    13:00, short breaks, done by 18:30 (all jittered a few minutes)."""
    j = lambda: rng.uniform(-8, 8) * 60  # noqa: E731
    t = at(*ymd, 9, 0) + j()
    t = run(obs, t, at(*ymd, 10, 0))
    if slack:
        t = run(obs, t, at(*ymd, 10, 15), app=SLACK)
    t = run(obs, t, at(*ymd, 11, 20) + j())               # focus block ~65 min
    t = run(obs, t, t + 10 * 60, away=True)                # short break
    t = run(obs, t, at(*ymd, 13, 0) + j())
    if lunch:
        t = run(obs, t, t + rng.uniform(40, 50) * 60, away=True)
    t = run(obs, t, at(*ymd, 15, 0) + j(), app=CHROME)
    t = run(obs, t, t + 10 * 60, away=True)
    t = run(obs, t, at(*ymd, 18, 30) + j())
    if late:
        y, m, d = day_of_from(ymd, 1)
        run(obs, at(*ymd, 18, 31), at(y, m, d, 1, 30))


def day_of_from(ymd, n):
    t = at(*ymd, 12) + n * 86400
    lt = time.localtime(t)
    return lt.tm_year, lt.tm_mon, lt.tm_mday


def weeks(obs: ActivityObserver, n_days: int, seed: int = 7, **kw) -> list:
    rng = random.Random(seed)
    days = []
    for i in range(n_days):
        ymd = day_of(i)
        if time.localtime(at(*ymd, 12)).tm_wday >= 5:
            continue
        workday(obs, ymd, rng, **kw)
        days.append(ymd)
    return days


def make(tmp_path=None, **kw):
    store = ActivityStore(None if tmp_path is None else tmp_path / "instinct.json")
    obs = ActivityObserver(store, **kw)
    return store, obs, PatternMiner(store, cache_s=0)


# --------------------------------------------------------------------------- #
# Categories and settings
# --------------------------------------------------------------------------- #


class TestCategories:
    def test_bundle_ids_and_names(self):
        assert C.categorize("com.microsoft.VSCode", "Code") == "code"
        assert C.categorize("com.jetbrains.pycharm", None) == "code"
        assert C.categorize(None, "iTerm2") == "terminal" or C.categorize(None, "iTerm") == "terminal"
        assert C.categorize("com.tinyspeck.slackmacgap", "Slack") == "chat"
        assert C.categorize("us.zoom.xos", "zoom.us") == "meeting"

    def test_unknown_apps_are_learned_by_a_clean_name(self):
        assert C.categorize("com.linear.app", "Linear") == "other:linear"
        assert C.categorize(None, "Wéird/App: <x>") == "other:wéirdapp x"
        assert C.categorize(None, None) == "unknown"

    def test_overrides_win(self):
        assert C.categorize("com.linear.app", "Linear", {"linear": "writing"}) == "writing"

    def test_spanish_words(self):
        assert C.word("code") == "el editor"
        assert C.word("other:linear") == "Linear"


class TestSettings:
    def test_defaults(self):
        s = InstinctSettings.from_dict(None)
        assert s.enabled and s.observe and s.quiet_meetings and not s.speak
        assert set(s.instincts) == set(INSTINCTS)
        assert s.keep_days == 56

    def test_switching_single_instincts_and_cooldowns(self):
        s = InstinctSettings.from_dict({"instincts": {"frenzy": False}, "cooldowns": {
            "welcome": 30, "nope": 5, "lunch": "x"}, "sample_s": "bad", "keep_days": 1})
        assert "frenzy" not in s.instincts and "welcome" in s.instincts
        assert s.cooldowns["welcome"] == 30 and s.cooldowns["lunch"] == INSTINCTS["lunch"]
        assert s.sample_s == 30.0 and s.keep_days == 7

    def test_list_form_and_categories(self):
        s = InstinctSettings.from_dict({"instincts": ["welcome", "hungry"],
                                        "categories": {"Linear": "Writing"}})
        assert s.instincts == ("welcome", "hungry")
        assert s.categories == {"linear": "writing"}


# --------------------------------------------------------------------------- #
# Observer
# --------------------------------------------------------------------------- #


class TestObserver:
    def test_one_day_is_folded_into_aggregates(self):
        store, obs, _ = make()
        workday(obs, MONDAY, random.Random(1))
        d = store.days[O.day_key(at(*MONDAY, 12))]
        assert 8 * 60 <= d["first"] + 240 <= 9 * 60 + 10  # ~9:00
        assert sum(d["act"]) / 3600 > 7
        assert sum(d["cats"]["chat"]) == pytest.approx(15 * 60, abs=90)
        assert d["sw"][10] >= 2
        lunch = [b for b in d["breaks"] if b[1] >= 30]
        assert len(lunch) == 1 and O.hhmm(lunch[0][0]).startswith(("12:5", "13:0"))
        assert d["focus"] and d["focus"][0][2] == "code"
        assert d["keys"][9] > 0 and d["mouse"][9] > 0

    def test_live_context(self):
        store, obs, _ = make()
        t = run(obs, at(*MONDAY, 9), at(*MONDAY, 9, 50))
        ctx = obs.context(t)
        assert ctx.active and ctx.category == "code" and ctx.focus_s >= 45 * 60
        assert ctx.since_break_s >= 49 * 60
        t = run(obs, t, t + 60 * 60, away=True)
        assert not obs.context(t).active and obs.context(t).idle_s >= 59 * 60
        t = run(obs, t, t + 60)
        ctx = obs.context(t)
        assert ctx.returned is not None and ctx.returned[1] >= 59 * 60
        assert ctx.focus_ended is not None and ctx.focus_ended[1] >= 45 * 60
        assert ctx.since_break_s < 120

    def test_switching_rate(self):
        store, obs, _ = make()
        t = run(obs, at(*MONDAY, 9), at(*MONDAY, 9, 10), switch_every=30)
        assert obs.context(t).switches >= 15

    def test_night_belongs_to_the_day_before(self):
        store, obs, _ = make()
        nxt = day_of(1)
        run(obs, at(*MONDAY, 23), at(*nxt, 1, 0))
        assert O.day_key(at(*nxt, 1, 0)) == O.day_key(at(*MONDAY, 12))
        d = store.days[O.day_key(at(*MONDAY, 12))]
        assert sum(d["act"][0:2]) >= 3500
        assert O.day_key(at(*nxt, 5)) not in store.days

    def test_touch_and_voice_hooks(self):
        store, obs, _ = make(clock=Clock(at(*MONDAY, 10)))
        obs.note("talk")
        obs.note("touch")
        obs.note("pet")
        d = store.days[O.day_key(at(*MONDAY, 10))]
        assert d["voice"] == 1 and d["touch"] == 2
        assert obs.context(at(*MONDAY, 10, 1)).since_contact_s == pytest.approx(60)

    def test_rotation_keeps_eight_weeks(self):
        store, obs, _ = make()
        for i in range(70):
            ymd = day_of(i)
            run(obs, at(*ymd, 10), at(*ymd, 10, 2))
        assert len(store.days) == 56
        assert min(store.days) == O.day_key(at(*day_of(14), 12))

    def test_save_load_round_trip_and_wipe(self, tmp_path):
        store, obs, _ = make(tmp_path)
        workday(obs, MONDAY, random.Random(2))
        store.save()
        again = ActivityStore(tmp_path / "instinct.json")
        again.load(at(*MONDAY, 20))
        assert again.days == store.days
        obs.wipe()
        assert store.days == {} and not (tmp_path / "instinct.json").exists()

    def test_a_corrupt_file_starts_fresh(self, tmp_path):
        p = tmp_path / "instinct.json"
        p.write_text("{nope")
        s = ActivityStore(p)
        s.load()
        assert s.days == {}
        p.write_text(json.dumps({"days": {"2026-03-02": {"act": [1, 2]}, "x": 5}}))
        s.load(at(*MONDAY, 12))
        assert s.days == {}


class TestPrivacy:
    def test_a_sample_has_no_room_for_content(self):
        fields = set(Sample.__dataclass_fields__)
        assert fields == {"wall", "bundle", "app", "key_idle", "mouse_idle", "locked"}

    def test_the_probe_never_asks_for_titles_or_keys(self):
        src = inspect.getsource(O.probe)
        assert "kCGWindowName" not in src
        assert "CGEventTapCreate" not in src and "addGlobalMonitor" not in src

    def test_the_state_holds_categories_and_numbers_only(self, tmp_path):
        store, obs, _ = make(tmp_path)
        workday(obs, MONDAY, random.Random(3))
        run(obs, at(*MONDAY, 19), at(*MONDAY, 19, 10),
            app=("com.secret.bank", "Mi Banco — Saldo $1.000"))
        store.save()
        raw = (tmp_path / "instinct.json").read_text()
        assert "com.microsoft" not in raw and "tinyspeck" not in raw and "secret" not in raw
        assert "Saldo" not in raw and "$" not in raw
        cats = set(store.days[O.day_key(at(*MONDAY, 12))]["cats"])
        assert cats <= {"code", "chat", "browser", "other:mi banco  saldo 1.000"}


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #


class TestPatterns:
    def test_nothing_before_a_few_days(self):
        store, obs, miner = make()
        weeks(obs, 2)
        assert miner.patterns(at(*day_of(3), 12)) == []

    def test_routines_of_two_weeks(self):
        store, obs, miner = make()
        weeks(obs, 14)
        now = at(*day_of(14), 8)
        pats = {p.key: p for p in miner.patterns(now)}
        assert {"start", "lunch", "end", "focus", "app:chat", "breaks", "hours"} <= set(pats)
        assert pats["start"].text.startswith("Los días de semana arrancás tipo 9:0") \
            or pats["start"].text.startswith("Los días de semana arrancás tipo 8:5")
        assert pats["lunch"].data["minute"] + 240 == pytest.approx(13 * 60, abs=10)
        assert 40 <= pats["lunch"].data["length"] <= 50
        assert pats["app:chat"].data["hour"] == 10
        assert "a las 10 abrís el chat" in pats["app:chat"].text
        assert pats["focus"].data["category"] == "code"
        assert pats["start"].confidence >= 0.7 and pats["lunch"].confidence >= 0.6
        assert "late" not in pats

    def test_confidence_grows_with_support(self):
        store, obs, miner = make()
        weeks(obs, 5, seed=1)  # Mon..Fri: 5 days
        few = miner.get("lunch", at(*day_of(7), 8)).confidence
        weeks(obs, 21, seed=1)
        more = miner.get("lunch", at(*day_of(21), 8)).confidence
        assert 0 < few < more
        assert confidence(0, 10) == 0 and confidence(10, 10) == 1.0
        assert confidence(10, 10, spread_min=45) == 0.5

    def test_scattered_times_are_not_a_routine(self):
        store, obs, miner = make()
        rng = random.Random(5)
        for i in range(14):
            ymd = day_of(i)
            h = rng.choice([7, 10, 13, 16])
            run(obs, at(*ymd, h), at(*ymd, h + 3))
        start = miner.get("start", at(*day_of(14), 8))
        assert start is None or start.confidence < 0.4

    def test_late_nights(self):
        store, obs, miner = make()
        weeks(obs, 10, late=True)
        late = miner.get("late", at(*day_of(10), 8))
        assert late is not None and late.confidence >= 0.3
        assert "Trasnochás" in late.text

    def test_today_is_not_part_of_usual(self):
        store, obs, miner = make()
        weeks(obs, 3)
        assert miner.patterns(at(*day_of(2), 20)) == []  # third day still going


class TestExpectations:
    def _trained(self):
        store, obs, miner = make()
        days = weeks(obs, 14)
        return store, obs, miner, day_of(14)  # a Monday after two weeks

    def test_lunch_is_expected_until_they_eat(self):
        store, obs, miner, today = self._trained()
        t = run(obs, at(*today, 9), at(*today, 13, 5))
        keys = [e.key for e in miner.expectations(obs.context(t))]
        assert "lunch" in keys
        t = run(obs, t, t + 40 * 60, away=True)
        t = run(obs, t, t + 120)
        assert "lunch" not in [e.key for e in miner.expectations(obs.context(t))]

    def test_break_overdue_by_their_own_rhythm(self):
        store, obs, miner, today = self._trained()
        every = miner.get("breaks", at(*today, 8)).data["every"]
        t = run(obs, at(*today, 14), at(*today, 14) + (every + 20) * 60)
        exp = {e.key: e for e in miner.expectations(obs.context(t))}
        assert "break" in exp and exp["break"].in_min < 0

    def test_recurring_app_is_expected(self):
        store, obs, miner, today = self._trained()
        t = run(obs, at(*today, 9), at(*today, 9, 50))
        assert "app:chat" in [e.key for e in miner.expectations(obs.context(t))]

    def test_later_than_usual(self):
        store, obs, miner, today = self._trained()
        t = run(obs, at(*today, 9), at(*today, 21, 0))
        assert "late" in [e.key for e in miner.expectations(obs.context(t))]


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class Q:
    def __init__(self, quiet=False, meeting=False):
        self.quiet, self.meeting = quiet, meeting


class FakeLearner:
    def __init__(self, factor=1.0):
        self.factor = factor
        self.shown = []

    def instinct_factor(self, name, bubble=False):
        return self.factor

    def instinct_shown(self, name, bubble=False):
        self.shown.append((name, bubble))


class Dice(random.Random):
    """random() always returns *value*: 0 = every roll passes, 1 = none."""

    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def random(self):
        return self.value

    def choice(self, seq):
        return seq[0]


def engine(obs=None, miner=None, *, roll=0.0, needs=None, quiet=None, budget=True,
           learner=None, **settings):
    bus = ReactionBus()
    s = InstinctSettings.from_dict({"min_gap_min": 0, **settings})
    e = InstinctEngine(s, observer=obs, miner=miner, bus=bus, rng=Dice(roll),
                       needs=lambda: needs or {}, quiet=lambda: quiet,
                       budget=lambda spend: budget,
                       learner=lambda: learner)
    return e, bus


def drain(bus):
    out = []
    while (r := bus.pop()) is not None:
        out.append(r)
    return out


class TestEngine:
    def test_nothing_happens_most_of_the_time(self):
        store, obs, miner = make()
        t = run(obs, at(*MONDAY, 9, 40), at(*MONDAY, 9, 50), app=CHROME)
        e, bus = engine(obs, miner)
        assert e.drives(obs.context(t), {}, t) == []
        assert e.decide(t) is None and drain(bus) == []

    def test_welcome_back_after_a_long_absence(self):
        store, obs, miner = make()
        t = run(obs, at(*MONDAY, 9), at(*MONDAY, 10))
        t = run(obs, t, t + 90 * 60, away=True)
        t = run(obs, t, t + 60)
        lr = FakeLearner()
        e, bus = engine(obs, miner, learner=lr, needs={"loneliness": 0.8})
        assert e.decide(t) == "welcome"
        r = drain(bus)[0]
        assert r.bubble == "¡Volviste! Te extrañé." and r.behaviour == "welcome"
        assert r.source.kind == "instinct.welcome" and not r.speak
        assert lr.shown == [("welcome", True)]
        # Cooldown: not again on the next return.
        t = run(obs, t, t + 60 * 60, away=True)
        t = run(obs, t, t + 60)
        assert e.decide(t) != "welcome"

    def test_first_return_of_the_day_is_a_quiet_coffee_after_the_morning_sensor(self):
        store, obs, miner = make()
        t = run(obs, at(*MONDAY, 9), at(*MONDAY, 9, 1))
        e, bus = engine(obs, miner)
        e.on_event(Event("clock.morning", at=t - 30))
        assert e.decide(t) == "welcome"
        r = drain(bus)[0]
        assert r.behaviour == "coffee" and r.bubble is None

    def test_curls_up_silently_during_focus_even_in_quiet_mode(self):
        store, obs, miner = make()
        t = run(obs, at(*MONDAY, 9), at(*MONDAY, 9, 30))
        e, bus = engine(obs, miner, quiet=Q(quiet=True), instincts=["focus_company", "break_ball"])
        assert e.decide(t) == "focus_company"
        r = drain(bus)[0]
        assert r.behaviour == "curl_focus" and r.bubble is None and r.priority == 0

    def test_nothing_at_all_in_a_meeting(self):
        store, obs, miner = make()
        t = run(obs, at(*MONDAY, 9), at(*MONDAY, 9, 30))
        e, bus = engine(obs, miner, quiet=Q(quiet=True, meeting=True))
        assert e.decide(t) is None
        store2, obs2, miner2 = make()
        t = run(obs2, at(*MONDAY, 9), at(*MONDAY, 9, 30), app=ZOOM)
        e, bus = engine(obs2, miner2, needs={"hunger": 1.0})
        assert e.decide(t) is None and drain(bus) == []

    def test_quiet_mode_holds_bubbles_back(self):
        e, bus = engine(None, None, needs={"hunger": 0.95}, quiet=Q(quiet=True))
        assert e.decide(at(*MONDAY, 12)) is None
        e, bus = engine(None, None, needs={"hunger": 0.95})
        assert e.decide(at(*MONDAY, 12)) == "hungry"
        assert drain(bus)[0].behaviour == "hungry_bowl"

    def test_the_budget_is_shared_with_the_alerts(self):
        e, bus = engine(None, None, needs={"hunger": 0.95}, budget=False)
        assert e.decide(at(*MONDAY, 12)) is None

    def test_a_sensor_that_already_said_it_wins(self):
        store, obs, miner = make()
        t = run(obs, at(*MONDAY, 9), at(*MONDAY, 11, 30), app=CHROME)
        e, bus = engine(obs, miner, instincts=["break_ball"])
        e.on_event(Event("break.suggest", at=t - 600))
        assert e.decide(t) is None
        e2, _ = engine(obs, miner, instincts=["break_ball"])
        assert e2.decide(t) == "break_ball"

    def test_break_ball_follows_their_own_rhythm(self):
        store, obs, miner = make()
        weeks(obs, 14)
        today = day_of(14)
        every = miner.get("breaks", at(*today, 8)).data["every"]
        t = run(obs, at(*today, 14), at(*today, 14) + (every - 15) * 60, app=CHROME)
        e, bus = engine(obs, miner, instincts=["break_ball"])
        assert e.decide(t) is None
        t = run(obs, t, t + 25 * 60, app=CHROME)
        assert e.decide(t) == "break_ball"
        r = drain(bus)[0]
        assert r.behaviour == "ball_bring" and "Pausa" in r.bubble

    def test_lunch_reminder_in_the_learned_window_only(self):
        store, obs, miner = make()
        weeks(obs, 14)
        today = day_of(14)
        t = run(obs, at(*today, 9), at(*today, 11, 0), app=CHROME)
        e, bus = engine(obs, miner, instincts=["lunch"])
        assert e.decide(t) is None
        t = run(obs, t, at(*today, 13, 10), app=CHROME)
        assert e.decide(t) == "lunch"
        r = drain(bus)[0]
        assert "Comiste" in r.bubble and r.behaviour == "lunch_ask"

    def test_late_night_unusual_vs_night_owl(self):
        store, obs, miner = make()
        weeks(obs, 14)
        today, tomorrow = day_of(14), day_of(15)
        t = run(obs, at(*today, 9), at(*tomorrow, 2, 5))
        e, bus = engine(obs, miner, instincts=["late_night"])
        assert e.decide(t) == "late_night"
        r = drain(bus)[0]
        assert r.bubble == "¿Seguimos? Son las 2." and r.behaviour == "late_night"

        owl_store, owl, owl_miner = make()
        weeks(owl, 14, late=True)
        t = run(owl, at(*today, 9), at(*tomorrow, 2, 5))
        e_owl, _ = engine(owl, owl_miner, instincts=["late_night"], roll=0.3)
        # 2 am is their usual hour (or close): no drive, or a much weaker one.
        assert all(d.urgency < 0.3 for d in e_owl.drives(owl.context(t), {}, t))
        assert e_owl.decide(t) is None  # a known night owl gets far fewer yawns

    def test_focus_done_and_frenzy(self):
        store, obs, miner = make()
        t = run(obs, at(*MONDAY, 9), at(*MONDAY, 10, 0))
        t = run(obs, t, t + 4 * 60, app=CHROME)
        e, bus = engine(obs, miner, instincts=["focus_done"])
        assert e.decide(t) == "focus_done"
        assert drain(bus)[0].bubble.startswith("¡Qué bloque! una hora")

        store, obs, miner = make()
        t = run(obs, at(*MONDAY, 9), at(*MONDAY, 9, 15), app=SLACK, switch_every=30)
        e, bus = engine(obs, miner, instincts=["frenzy"])
        assert e.decide(t) == "frenzy"
        assert drain(bus)[0].behaviour == "side_eye"

    def test_noticed_the_usual_app_at_the_usual_hour(self):
        store, obs, miner = make()
        weeks(obs, 14)
        today = day_of(14)
        t = run(obs, at(*today, 10, 0), at(*today, 10, 3), app=SLACK)
        e, bus = engine(obs, miner, instincts=["noticed"])
        assert e.decide(t) == "noticed"
        assert drain(bus)[0].bubble == "El chat a las 10, como siempre."

    def test_cooldowns_min_gap_and_disabled(self):
        e, bus = engine(None, None, needs={"hunger": 0.95}, min_gap_min=30)
        t = at(*MONDAY, 12)
        assert e.decide(t) == "hungry"
        assert e.decide(t + 60) is None
        assert e.decide(t + 119 * 60) is None   # hungry cooldown (120 min)
        assert e.decide(t + 121 * 60) == "hungry"
        e, _ = engine(None, None, needs={"hunger": 0.95}, instincts={"hungry": False})
        assert e.decide(t) is None
        e, _ = engine(None, None, needs={"hunger": 0.95}, enabled=False)
        assert e.decide(t) is None

    def test_maybe_decide_is_rate_limited(self):
        e, bus = engine(None, None, needs={"hunger": 0.95}, roll=1.0)
        t = at(*MONDAY, 12)
        assert e.maybe_decide(t) is None
        e.rng = Dice(0.0)
        assert e.maybe_decide(t + 10) is None   # too soon after the last look
        assert e.maybe_decide(t + 121) == "hungry"

    def test_what_the_owner_taught_tilts_the_dice(self):
        # urgency 0.5 (hunger 0.85) against a roll of 0.6.
        liked, _ = engine(None, None, needs={"hunger": 0.85}, roll=0.6, learner=FakeLearner(2.0))
        disliked, _ = engine(None, None, needs={"hunger": 0.85}, roll=0.6,
                             learner=FakeLearner(0.5))
        t = at(*MONDAY, 12)
        assert liked.decide(t) == "hungry"
        assert disliked.decide(t) is None

    def test_rare_on_average(self):
        """A realistic day: a dozen decisions or fewer, never a chatterbox."""
        store, obs, miner = make()
        weeks(obs, 14)
        today = day_of(14)
        bus = ReactionBus()
        e = InstinctEngine(InstinctSettings(), observer=obs, miner=miner, bus=bus,
                           rng=random.Random(11), needs=lambda: {"hunger": 0.3},
                           quiet=lambda: None, budget=lambda spend: None,
                           learner=lambda: None)
        t = at(*today, 9) + 37
        fired = []
        rng = random.Random(3)
        workday_end = at(*today, 18, 30)
        while t < workday_end:
            obs.ingest(Sample(t, *(VSCODE if rng.random() < 0.8 else CHROME),
                              key_idle=2.0, mouse_idle=5.0))
            name = e.maybe_decide(t)
            if name:
                fired.append(name)
            t += 30
        assert 1 <= len(fired) <= 12


# --------------------------------------------------------------------------- #
# Learner integration
# --------------------------------------------------------------------------- #


def learner(tmp_path, clock):
    s = LearnSettings.from_dict({"state_file": str(tmp_path / "learn.json"),
                                 "brain_file": str(tmp_path / "b.npz"),
                                 "life_log": str(tmp_path / "life.jsonl"),
                                 "embed_model": "none"})
    return Learner(s, actions=["wag", "yawn"], clock=clock, seed=1, embedder=None)


class TestLearnerIntegration:
    def test_engagement_after_an_instinct_is_a_reward(self, tmp_path):
        clock = Clock(1000.0)
        lr = learner(tmp_path, clock)
        lr.instinct_shown("welcome", bubble=True)
        clock.t += 5
        lr.signal("touch")
        assert lr.bandit.arms[INSTINCT_ARM + "welcome"][0] == pytest.approx(1.0)
        # The bubble also taught the brain's "good moment" head.
        rows = [json.loads(x) for x in (tmp_path / "life.jsonl").read_text().splitlines()]
        assert rows[-1]["h"] == "a" and rows[-1]["r"] == 1.0

    def test_a_quick_dismiss_or_a_drag_is_a_no(self, tmp_path):
        clock = Clock(1000.0)
        lr = learner(tmp_path, clock)
        lr.instinct_shown("frenzy", bubble=True)
        clock.t += 1
        lr.signal("bubble_click")
        lr.instinct_shown("break_ball", bubble=True)
        clock.t += 3
        lr.signal("drag")
        assert lr.bandit.arms[INSTINCT_ARM + "frenzy"][1] == pytest.approx(1.0)
        assert lr.bandit.arms[INSTINCT_ARM + "break_ball"][1] == pytest.approx(1.0)

    def test_unanswered_times_out_neutral(self, tmp_path):
        clock = Clock(1000.0)
        lr = learner(tmp_path, clock)
        lr.instinct_shown("focus_company")
        clock.t += 60
        lr.tick()
        s, f = lr.bandit.arms[INSTINCT_ARM + "focus_company"]
        assert s == 0 and 0 < f < 0.5

    def test_the_factor_follows_what_was_learned(self, tmp_path):
        clock = Clock(1000.0)
        lr = learner(tmp_path, clock)
        for _ in range(15):
            lr.instinct_shown("welcome", bubble=True)
            clock.t += 2
            lr.signal("pet")
            lr.instinct_shown("frenzy", bubble=True)
            clock.t += 1
            lr.signal("bubble_click")
            clock.t += 30
        liked = sum(lr.instinct_factor("welcome") for _ in range(50)) / 50
        disliked = sum(lr.instinct_factor("frenzy") for _ in range(50)) / 50
        assert liked > 1.3 and disliked <= 0.6
        assert 0.5 <= lr.instinct_factor("never_seen") <= 2.0
        assert lr.instinct_likes() == ["welcome"]

    def test_instincts_stay_out_of_alerts_and_behaviours(self, tmp_path):
        clock = Clock(1000.0)
        lr = learner(tmp_path, clock)
        lr.alert_shown(Event("instinct.welcome", title="¡Volviste!"), "¡Volviste!")
        assert not any(p.kind == "a" for p in lr._pending)
        for _ in range(5):
            lr.instinct_shown("welcome", bubble=True)
            clock.t += 1
            lr.signal("touch")
        assert "bienven" not in lr.summary().lower() and "instinct" not in lr.summary()
        assert all(not a.startswith(INSTINCT_ARM) for a in lr.bias({}, ["wag", "yawn"]))

    def test_forget_wipes_memory_and_files(self, tmp_path):
        clock = Clock(1000.0)
        lr = learner(tmp_path, clock)
        lr.instinct_shown("welcome", bubble=True)
        clock.t += 1
        lr.signal("touch")
        lr.save()
        assert (tmp_path / "learn.json").exists() and (tmp_path / "life.jsonl").exists()
        lr.forget()
        assert lr.bandit.arms == {} and lr.stats["rewards"] == 0
        assert not (tmp_path / "learn.json").exists() and not (tmp_path / "life.jsonl").exists()

    def test_the_real_engine_registers_with_the_real_learner(self, tmp_path):
        clock = Clock(1000.0)
        lr = learner(tmp_path, clock)
        e, bus = engine(None, None, needs={"hunger": 0.95}, learner=lr)
        assert e.decide(at(*MONDAY, 12)) == "hungry"
        assert [p.name for p in lr._pending if p.kind == "i"] == ["hungry"]


# --------------------------------------------------------------------------- #
# Self-model and the mind
# --------------------------------------------------------------------------- #


class TestSelfModel:
    def test_notes_facts_and_lines(self):
        store, obs, miner = make()
        weeks(obs, 14)
        today = day_of(14)
        t = run(obs, at(*today, 9), at(*today, 11, 40))
        sm = selfmodel.self_notes(obs, miner, t)
        assert any("sin pausa" in n for n in sm["notes"])
        assert any("concentrado" in n for n in sm["notes"])
        assert sm["facts"] and all(len(f) <= 80 for f in sm["facts"])
        assert any("¿pausa?" in ln for ln in sm["lines"])
        assert all(len(ln) <= 60 for ln in sm["lines"])
        text = selfmodel.render(sm, {"mood": "contenta"})
        assert text.startswith("Sos Wawi") and "Hoy notaste" in text

    def test_the_mind_prompt_carries_it(self):
        sm = {"notes": ["Lleva 2 horas sin pausa", "http://x.com malo", "a" * 300],
              "facts": ["Almorzás tipo 13:00"],
              "lines": ["Hoy codeaste 3 horas seguidas, ¿pausa?", "😀 no", "x" * 90]}
        s = local.summarize_state([], {"mood": "contenta"}, at(*MONDAY, 15), self_model=sm)
        text = s.render()
        assert "Hoy notaste: Lleva 2 horas sin pausa" in text
        assert "http" not in text and "Sabés de tu dueño: Almorzás tipo 13:00" in text
        assert s.lines == ("Hoy codeaste 3 horas seguidas, ¿pausa?",)
        assert "Hoy codeaste" in local.BUBBLE_SYSTEM

    def test_canned_bubbles_use_what_she_noticed(self):
        s = local.StateSummary(15, 0, {}, (), lines=("Hoy codeaste 3 horas seguidas, ¿pausa?",))
        p = local.Personality(None, rng=random.Random(0))
        seen = {p.canned(s) for _ in range(8)}
        assert "Hoy codeaste 3 horas seguidas, ¿pausa?" in seen

    def test_without_instincts_the_self_model_is_empty(self):
        assert instinct.self_notes() == {"notes": [], "facts": [], "lines": []}
        from wawi.mind import peers

        assert peers.self_model() == {"notes": [], "facts": [], "lines": []}


# --------------------------------------------------------------------------- #
# Voice tool, wiring, fallbacks
# --------------------------------------------------------------------------- #


def installed(tmp_path, n_days=14):
    s = InstinctSettings.from_dict({"state_file": str(tmp_path / "instinct.json")})
    inst = instinct.Instinct(s, probe_fn=lambda: Sample(0.0), subscribe=False)
    inst.engine.bus = ReactionBus()
    weeks(inst.observer, n_days)
    instinct.install(inst)
    return inst


class TestVoiceTool:
    def test_schema_is_terse_and_only_sent_when_installed(self, tmp_path):
        assert len(json.dumps(tools.INSTINCT_TOOLS)) < 300
        assert "about_me" not in {t["name"] for t in tools.active_tools()}
        installed(tmp_path)
        assert "about_me" in {t["name"] for t in tools.active_tools()}

    def test_what_do_you_know_about_me(self, tmp_path):
        inst = installed(tmp_path)
        r = tools.dispatch("about_me", {})
        assert r.ok and r.speak.startswith("Esto vi: ")
        assert "arrancás tipo" in r.speak and "Almorzás tipo 13" in r.speak
        assert r.speak.count(".") <= 8
        assert inst.engine.bus.pop().behaviour == "pattern_notebook"

    def test_too_early_to_tell(self, tmp_path):
        installed(tmp_path, n_days=2)
        text = T.summary()
        assert text.startswith("Todavía te estoy conociendo: llevo 2 días")

    def test_off(self):
        ok, text = T.about_me()
        assert not ok and "apagados" in text

    def test_liked_instincts_are_mentioned(self, tmp_path, monkeypatch):
        installed(tmp_path)

        class L:
            def instinct_likes(self, n):
                return ["welcome"]

        learn.install(L())
        assert T.summary().endswith("te gusta que te salude cuando volvés.")

    def test_forget_everything(self, tmp_path):
        inst = installed(tmp_path)
        inst.store.save()
        clock = Clock(1000.0)
        lr = learner(tmp_path, clock)
        lr.instinct_shown("welcome")
        clock.t += 1
        lr.signal("touch")
        lr.save()
        learn.install(lr)
        r = tools.dispatch("about_me", {"forget": True})
        assert r.ok and r.speak.startswith("Listo: me olvidé") and "solo en esta compu" in r.speak
        assert inst.store.days == {} and not (tmp_path / "instinct.json").exists()
        assert lr.bandit.arms == {} and not (tmp_path / "learn.json").exists()
        assert T.summary().startswith("Todavía nada")


class TestWiring:
    def test_build_respects_the_switch(self, tmp_path):
        class Cfg:
            instinct = {"enabled": False}

        assert instinct.build(Cfg()) is None
        Cfg.instinct = {"state_file": str(tmp_path / "i.json")}
        inst = instinct.build(Cfg())
        assert inst is not None and inst.settings.observe

    def test_step_samples_and_decides(self, tmp_path):
        clock = Clock(at(*MONDAY, 9))
        s = InstinctSettings.from_dict({"state_file": None, "instincts": ["hungry"]})
        inst = instinct.Instinct(
            s, probe_fn=lambda: Sample(clock.t, *VSCODE, key_idle=1.0, mouse_idle=1.0),
            clock=clock, subscribe=False)
        inst.engine.bus = ReactionBus()
        inst.engine._needs = lambda: {"hunger": 0.95}
        inst.engine._quiet = lambda: None
        inst.engine._budget = lambda spend: None
        inst.engine._learner = lambda: None
        inst.engine.rng = Dice(0.0)
        assert inst.step(clock.t) == "hungry"
        assert sum(inst.store.days[O.day_key(clock.t)]["act"]) == 30

    def test_observe_off_samples_nothing(self, tmp_path):
        s = InstinctSettings.from_dict({"state_file": None, "observe": False})
        calls = []
        inst = instinct.Instinct(s, probe_fn=lambda: calls.append(1) or Sample(0.0),
                                 subscribe=False)
        inst.engine._needs = lambda: {}
        inst.step(at(*MONDAY, 9))
        assert calls == [] and inst.store.days == {}
        instinct.install(inst)
        instinct.note("talk")
        assert inst.store.days == {}

    def test_the_body_notes_voice_turns_and_touches(self, tmp_path):
        inst = installed(tmp_path, n_days=0)
        from wawi.pet.life import _instinct_note

        _instinct_note("touch")
        _instinct_note("drag")
        today = inst.store.days[O.day_key(time.time())]
        assert today["touch"] == 1

    def test_config_section_is_parsed(self, tmp_path):
        from wawi import config

        p = tmp_path / "c.yaml"
        p.write_text("instinct:\n  observe: false\n")
        assert config.load(p).instinct == {"observe": False}

    def test_reflex_budget_helper(self):
        from wawi import reflexes

        assert reflexes.budget_allows() is None
        clock = Clock(1000.0)
        r = reflexes.Reflexes({"budget": {"info": 1}}, bus=ReactionBus(), clock=clock,
                              quiet=reflexes.QuietDetector({}, frontmost=lambda: None,
                                                          processes=lambda: set()))
        r.attach()
        try:
            assert reflexes.budget_allows("info", spend=True) is True
            assert reflexes.budget_allows("info") is False
        finally:
            r.detach()


class TestClipFallbacks:
    def test_new_frames_fall_back_to_calm_clips(self):
        have = {"sleep", "bark", "tilt", "yawn", "hungry_bowl", "cheer", "wake"}
        assert clip_for_behaviour("curl_focus", have) == "sleep"
        assert clip_for_behaviour("late_night", have) == "yawn"
        assert clip_for_behaviour("lunch_ask", have) == "hungry_bowl"
        assert clip_for_behaviour("welcome", have) == "cheer"
        assert clip_for_behaviour("coffee", have) == "wake"
        assert clip_for_behaviour("pattern_notebook", have) == "tilt"

    def test_the_real_frames_win_once_imported(self):
        assert clip_for_behaviour("curl_focus", {"curl_focus", "sleep"}) == "curl_focus"
