"""Iniciativa (wawi/initiative): the meeting detector and the quiet mode it
raises, the Orca watcher, the morning summary, the budget, what she learns,
the menu and the voice tool. Every macOS probe is a fake: running apps,
osascript, ps, the Orca CLI, the input idle clock."""

from __future__ import annotations

import json
import logging
import random
import time
from types import SimpleNamespace

import pytest

from wawi import initiative, tools
from wawi.initiative import core, menu, meeting as M, morning as MO, orca_watch as OW
from wawi.initiative import tool as T
from wawi.initiative.core import Act, Budget, Ctx, Hooks, Initiative, InitiativeSettings, Moment
from wawi.initiative.learning import LearningStore, step_down, step_up


def at(y, mo, d, h=0, mi=0, s=0) -> float:
    return time.mktime((y, mo, d, h, mi, s, 0, 0, -1))


MORNING = at(2026, 10, 16, 9, 0)
AFTERNOON = at(2026, 10, 16, 14, 53)
CALL = "https://meet.google.com/abc-defg-hij"


@pytest.fixture(autouse=True)
def _clean():
    initiative.install(None)
    M.set_in_meeting(False)
    yield
    initiative.install(None)
    M.set_in_meeting(False)


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Watch:
    """A watcher that returns the moments it is handed, once."""

    kinds = ("orca_done",)

    def __init__(self, act_factory=None) -> None:
        self.queue: list[Moment] = []
        self.act_factory = act_factory or (lambda m, ctx: Act(
            m.kind, "Hola", link="Abrir…", accept=lambda: self.accepted.append(1)))
        self.accepted: list = []

    def poll(self, ctx):
        out, self.queue = self.queue, []
        return out

    def act(self, m, ctx):
        return self.act_factory(m, ctx)


def engine(tmp_path=None, *, t=MORNING, idle=1.0, watchers=None, hooks=None, quiet=False,
           **settings):
    clock = Clock(t)
    box = {"idle": idle}
    store = LearningStore(None if tmp_path is None else tmp_path / "initiative.json")
    walked: list[str] = []
    e = Initiative(InitiativeSettings.from_dict(settings), store=store, hooks=hooks or Hooks(),
                   clock=clock, idle=lambda: box["idle"], quiet=lambda: quiet,
                   walk=walked.append, background=False,
                   watchers=[] if watchers is None else watchers)
    e.test_clock, e.test_idle, e.walked = clock, box, walked
    return e


# --------------------------------------------------------------------------- #
# Meeting detection
# --------------------------------------------------------------------------- #


class FakeMac:
    def __init__(self, bundles=(), tabs="", code=0, procs=(), front=None):
        self.bundles, self.tabs, self.code = set(bundles), tabs, code
        self.procs, self.front = set(procs), front
        self.scripts: list[str] = []

    def script(self, text):
        self.scripts.append(text)
        return self.code, self.tabs, "execution error: Not authorized (-1743)" if self.code else ""

    def detector(self, **kw):
        return M.MeetingDetector(bundles=lambda: self.bundles, script=self.script,
                                 processes=lambda: self.procs, front=lambda: self.front, **kw)


class TestMeetingDetection:
    def test_a_meet_call_in_a_running_chrome(self):
        mac = FakeMac({"com.google.Chrome"}, tabs=f"{CALL}\n")
        d = mac.detector()
        assert d.check(100.0) is True and d.source == "meet"
        script = mac.scripts[0]
        assert 'application id "com.google.Chrome" is running' in script
        assert "meet.google.com/" in script

    @pytest.mark.parametrize("tabs", ["https://meet.google.com/", "https://meet.google.com/landing",
                                      "https://meet.google.com/new", ""])
    def test_meet_home_or_no_tab_is_not_a_call(self, tabs):
        mac = FakeMac({"com.apple.Safari"}, tabs=tabs)
        assert mac.detector().check(100.0) is False

    def test_browsers_that_are_not_running_are_never_asked(self):
        mac = FakeMac({"com.apple.finder"}, tabs=CALL)
        assert mac.detector().check(100.0) is False
        assert mac.scripts == []

    def test_a_refusing_browser_backs_off(self):
        mac = FakeMac({"com.google.Chrome"}, code=1)
        d = mac.detector()
        assert d.check(100.0) is False
        assert d.check(130.0) is False
        assert len(mac.scripts) == 1  # not asked again for a while
        d.check(100.0 + M.REFUSED_BACKOFF_S + 1)
        assert len(mac.scripts) == 2

    def test_zoom_needs_the_call_process(self):
        assert FakeMac({M.ZOOM_BUNDLE}).detector().check(1.0) is False
        mac = FakeMac({M.ZOOM_BUNDLE}, procs={"CptHost", "zoom.us"})
        d = mac.detector()
        assert d.check(1.0) is True and d.source == "zoom"

    def test_a_meeting_app_in_front(self):
        assert FakeMac(front="FaceTime").detector().check(1.0) is True
        assert FakeMac(front="Slack").detector().check(1.0) is False

    def test_signals_can_be_switched_off(self):
        mac = FakeMac({"com.google.Chrome", M.ZOOM_BUNDLE}, tabs=CALL, procs={"CptHost"},
                      front="FaceTime")
        d = mac.detector(browsers=False, zoom=False, front_apps=False)
        assert d.check(1.0) is False and mac.scripts == []

    def test_sticky_until_a_minute_without_signal(self):
        mac = FakeMac({"com.google.Chrome"}, tabs=CALL)
        d = mac.detector(exit_after_s=60)
        d.check(0.0)
        mac.tabs = ""
        assert d.check(30.0) is True
        assert d.take_end() is None
        mac.tabs = CALL
        assert d.check(600.0) is True
        mac.tabs = ""
        assert d.check(630.0) is True
        assert d.check(661.0) is False
        end = d.take_end()
        assert end is not None and end.lasted_s == pytest.approx(600.0) and end.source == "meet"
        assert d.take_end() is None

    def test_the_meeting_code_is_never_logged(self, caplog):
        caplog.set_level(logging.DEBUG, logger="wawi.initiative")
        mac = FakeMac({"com.google.Chrome"}, tabs=CALL)
        d = mac.detector(exit_after_s=0)
        d.check(0.0)
        mac.tabs = ""
        d.check(10.0)
        assert "en reunión" in caplog.text and "terminó la reunión" in caplog.text
        assert "abc-defg-hij" not in caplog.text and "meet.google.com/abc" not in caplog.text

    def test_count_meet_calls(self):
        assert M.count_meet_calls(f"{CALL}\n{CALL}?authuser=1\nhttps://meet.google.com/\n") == 2


class TestMeetingWatcher:
    def _setup(self, tabs=CALL, start_note=True):
        mac = FakeMac({"com.google.Chrome"}, tabs=tabs)
        notes = {"on": False, "started": 0}

        def start():
            notes["started"] += 1
            notes["on"] = True

        hooks = Hooks(start_note=start if start_note else None, notes_active=lambda: notes["on"])
        e = engine(hooks=hooks)
        w = M.MeetingWatcher(e, mac.detector(exit_after_s=0))
        e.watchers = [w]
        return e, mac, notes

    def test_raises_the_quiet_flag_while_it_lasts(self):
        e, mac, _ = self._setup()
        e.step()
        assert initiative.in_meeting() and e.meeting_active
        mac.tabs = ""
        e.test_clock.t += 30
        e.step()
        assert not initiative.in_meeting()

    def test_no_quiet_flag_when_the_owner_switched_it_off(self):
        e, _, _ = self._setup()
        e.set_kind("meeting_quiet", False)
        e.step()
        assert e.meeting_active and not initiative.in_meeting()

    def test_after_a_long_meeting_she_offers_a_note(self):
        e, mac, notes = self._setup()
        e.step()
        e.test_clock.t += 20 * 60
        e.step()
        mac.tabs = ""
        e.test_clock.t += 30
        e.step()
        assert e.bubble_text() == f"{M.NOTES_TEXT}\n{M.NOTES_LINK}"
        e.click(on_link=True)
        assert notes["started"] == 1
        assert e.store.row("meeting_notes")["accepted"] == 1

    def test_saying_it_counts_as_yes(self):
        e, mac, notes = self._setup()
        e.step()
        e.test_clock.t += 20 * 60
        e.step()
        mac.tabs = ""
        e.test_clock.t += 30
        e.step()
        notes["on"] = True  # "Wawi, tomá nota"
        e.test_clock.t += 25
        e.step()
        assert e.bubble_text() is None
        assert e.store.row("meeting_notes")["accepted"] == 1

    def test_a_short_call_gets_no_offer(self):
        e, mac, _ = self._setup()
        e.step()
        mac.tabs = ""
        e.test_clock.t += 60
        e.step()
        assert e.bubble_text() is None and e.pending == []


# --------------------------------------------------------------------------- #
# Quiet mode: every speaker path consults it
# --------------------------------------------------------------------------- #


class TestQuietMode:
    def test_reflexes_quiet_state_says_meeting(self):
        from wawi.reflexes import QuietDetector

        q = QuietDetector({}, frontmost=lambda: ("Code", False), processes=lambda: set(),
                          cache_s=0)
        assert q.check().quiet is False
        M.set_in_meeting(True)
        state = q.check()
        assert state.quiet and state.meeting and state.reason == "meeting"

    def test_the_reaction_voice_goes_quiet(self, monkeypatch):
        from wawi import petvoice, reflexes

        monkeypatch.setattr(reflexes, "_ACTIVE", None)
        check = petvoice._quiet_from({"quiet_fullscreen": False, "meeting_processes": [],
                                      "meeting_apps": []})
        M.set_in_meeting(True)
        assert check() is True

    def test_no_curiosity_in_a_meeting(self, tmp_path):
        from wawi.curiosities import Curiosity, CuriosityService, CuriositySettings

        svc = CuriosityService(CuriositySettings.from_dict({"first_after_min": 0, "jitter": 0}),
                               state_path=None, now=0.0, background=False)
        svc.store.queue = [Curiosity(teaser="¿Sabías?", title="X", body="y")]
        svc.store.refilled_on = time.strftime("%Y-%m-%d", time.localtime(at(2026, 10, 16, 12)))
        M.set_in_meeting(True)
        idle = dict(idle=True, asleep=False, busy=False, channel_open=False)
        assert svc.poll(10.0, at(2026, 10, 16, 12), **idle) is None
        M.set_in_meeting(False)
        assert svc.poll(20.0, at(2026, 10, 16, 12), **idle) is not None

    def test_no_dancing_in_a_meeting(self):
        from wawi.music import dance_blocked

        kw = dict(state="IDLE", key="idle", notes=False, game=False, dragging=False,
                  hidden=False, presenting=False, working=False)
        assert dance_blocked(**kw) is None
        M.set_in_meeting(True)
        assert dance_blocked(**kw) == "meeting"

    def test_no_bark_sound_in_a_meeting(self):
        from wawi.panel import PanelObserver

        fake = SimpleNamespace(_pet_sounds=["sentinel"])
        M.set_in_meeting(True)
        PanelObserver._play_sound(fake, "/nonexistent/bark.wav")
        assert fake._pet_sounds == ["sentinel"]

    def test_spoken_reactions_wait_for_the_meeting_to_end(self):
        from wawi.orchestrator import Orchestrator
        from wawi.state import State

        orch = object.__new__(Orchestrator)
        orch.state = SimpleNamespace(state=State.IDLE)
        said = []
        orch.voice = SimpleNamespace(say=said.append)
        M.set_in_meeting(True)
        assert orch._speak_reaction("¡Se cayó la web!") is False
        assert said == []


# --------------------------------------------------------------------------- #
# Orca
# --------------------------------------------------------------------------- #


def ps(*agents, path="/w/feature", name="feature-x", repo="wawi"):
    return {"worktrees": [{"displayName": name, "repo": repo, "path": path, "isArchived": False,
                           "agents": [dict(a) for a in agents]}]}


def agent(pane="tab1:leaf1", state="working", title="Arreglar el login", outcome=None,
          since=1_000_000.0, interrupted=False, msg="Listo, PR abierto."):
    main = {"state": state, "stateStartedAt": since}
    if outcome:
        main["outcome"] = outcome
    return {"paneKey": pane, "state": state, "taskTitle": title, "mainAgent": main,
            "interrupted": interrupted, "lastAssistantMessage": msg, "stateStartedAt": since}


class FakeCLI:
    def __init__(self):
        self.result: dict = ps()
        self.calls: list[tuple] = []
        self.fail = False
        self.terminals: list[dict] = []

    def run(self, *args, timeout=0):
        self.calls.append(args)
        if self.fail:
            raise RuntimeError("runtime not reachable")
        if args[:2] == ("worktree", "ps"):
            return self.result
        if args[:2] == ("terminal", "list"):
            return {"terminals": self.terminals}
        return {}


def orca_engine(running=True, own=(), board=None):
    cli = FakeCLI()
    e = engine(hooks=Hooks(board=board))
    w = OW.OrcaWatcher(e, cli=lambda: cli, running=lambda: running, own=lambda: set(own),
                       every_s=30)
    e.watchers = [w]
    return e, w, cli


class TestOrca:
    def test_parse(self):
        agents = OW.parse_ps(ps(agent(), agent("t2:l2", "done", None, outcome="failure"),
                                name="main", repo="bruno"))
        assert agents["tab1:leaf1"].title == "Arreglar el login"
        assert agents["t2:l2"].title == "bruno" and agents["t2:l2"].failed

    def test_the_first_look_is_a_baseline(self):
        e, w, cli = orca_engine()
        cli.result = ps(agent(state="done"))
        e.step()
        assert e.pending == [] and e.bubble_text() is None
        assert w.agents is not None

    def test_working_to_done_walks_over_with_a_bubble(self):
        e, w, cli = orca_engine()
        cli.result = ps(agent())
        e.step()
        cli.result = ps(agent(state="done", since=2_000_000.0))
        e.test_clock.t += 30
        e.step()
        assert e.bubble_text() == "Terminó «Arreglar el login» ✓\nVer en Orca…"
        assert e.walked == ["orca_done"]

    def test_failed_and_waiting(self):
        e, w, cli = orca_engine()
        cli.result = ps(agent(), agent("t2:l2"))
        e.step()
        cli.result = ps(agent(state="done", outcome="failure"), agent("t2:l2", "blocked"))
        e.test_clock.t += 30
        moments = w.poll(e.context(e.test_clock.t))
        kinds = sorted((m.kind, w.act(m, None).text) for m in moments)
        assert kinds == [("orca_done", "«Arreglar el login» falló ✗"),
                         ("orca_waiting", "«Arreglar el login» te está esperando")]

    def test_interrupted_and_own_tasks_are_skipped(self):
        e, w, cli = orca_engine(own={"/w/feature"})
        cli.result = ps(agent())
        e.step()
        cli.result = ps(agent(state="done"))
        e.test_clock.t += 30
        assert w.poll(e.context(e.test_clock.t)) == []
        e2, w2, cli2 = orca_engine()
        cli2.result = ps(agent())
        e2.step()
        cli2.result = ps(agent(state="done", interrupted=True))
        e2.test_clock.t += 30
        assert w2.poll(e2.context(e2.test_clock.t)) == []

    def test_not_running_is_not_asked_and_errors_back_off(self):
        e, w, cli = orca_engine(running=False)
        e.step()
        assert cli.calls == []
        e, w, cli = orca_engine()
        cli.fail = True
        e.step()
        e.test_clock.t += 60
        e.step()
        assert len(cli.calls) == 1  # backing off

    def test_polls_on_its_own_clock(self):
        e, w, cli = orca_engine()
        e.step()
        e.test_clock.t += 10
        e.step()
        assert len(cli.calls) == 1

    def test_the_click_switches_to_the_terminal(self):
        e, w, cli = orca_engine()
        cli.terminals = [{"tabId": "tab1", "leafId": "leaf1", "handle": "term_1"}]
        w._cli = cli
        a = OW.parse_ps(ps(agent()))["tab1:leaf1"]
        import subprocess as sp

        opened = []
        orig = sp.run
        sp.run = lambda argv, **kw: opened.append(argv)
        try:
            w.focus(a)
        finally:
            sp.run = orig
        assert ("terminal", "switch", "--terminal", "term_1") in cli.calls
        assert opened == [["open", "-b", OW.ORCA_BUNDLE]]

    def test_the_click_falls_back_to_the_board(self):
        shown = []
        board = SimpleNamespace(show=lambda title, content, fmt="markdown": shown.append((title, content)))
        e, w, cli = orca_engine(board=board)
        w._cli = cli
        w.focus(OW.parse_ps(ps(agent()))["tab1:leaf1"])
        assert shown and shown[0][0] == "Arreglar el login" and "PR abierto" in shown[0][1]

    def test_said_aloud_when_the_owner_was_away(self):
        said = []
        e, w, cli = orca_engine()
        e.hooks.speak = said.append
        cli.result = ps(agent())
        e.step()
        e.test_idle["idle"] = 900.0  # away
        cli.result = ps(agent(state="done"))
        e.test_clock.t += 30
        e.step()
        assert e.bubble_text() is None and said == []  # nobody to tell
        e.test_idle["idle"] = 2.0  # back
        e.test_clock.t += 25
        e.step()
        assert said == ["Che, terminó Arreglar el login."]

    def test_overnight(self):
        now = 2_000_000.0
        w = SimpleNamespace(agents=OW.parse_ps(ps(
            agent("a:1", "working", "A"), agent("b:1", "done", "B", since=(now - 3600) * 1000),
            agent("c:1", "done", "C", since=(now - 3 * 86400) * 1000),
            agent("d:1", "waiting", "D"))))
        assert OW.overnight(w, now) == {"working": ["A"], "done": ["B"], "failed": [],
                                        "waiting": ["D"]}


# --------------------------------------------------------------------------- #
# Morning
# --------------------------------------------------------------------------- #


class TestMorning:
    def _engine(self, tmp_path, t=MORNING):
        shown = []
        board = SimpleNamespace(show=lambda title, content, fmt="markdown": shown.append((title, content)))
        e = engine(tmp_path, t=t, hooks=Hooks(board=board, notes_dir=str(tmp_path / "notas"),
                                               issues=lambda: ["El micrófono se corta"]))
        w = MO.MorningWatcher(e)
        e.watchers = [w]
        return e, w, shown

    def test_once_a_day_on_arrival(self, tmp_path, monkeypatch):
        monkeypatch.setattr(MO, "weather_line", lambda: "hoy 22° y llueve a las 15")
        e, w, shown = self._engine(tmp_path)
        e.store.last_active = MORNING - 12 * 3600  # last night
        e.step()
        assert e.bubble_text() == f"{MO.OFFER}\n{MO.LINK}"
        e.click(on_link=True)
        assert shown[0][0] == "Buen día" and "Llueve" not in shown[0][1]
        assert "Hoy 22° y llueve a las 15." in shown[0][1]
        assert "El micrófono se corta" in shown[0][1]
        # Later the same day, even after four hours away: not again.
        e.test_clock.t += 5 * 3600 - 3600 * 2
        e.store.last_active = e.test_clock.t - 5 * 3600
        e.step()
        assert e.pending == [] and e.bubble_text() is None

    def test_not_again_after_a_restart(self, tmp_path):
        e, w, _ = self._engine(tmp_path)
        e.store.last_active = MORNING - 12 * 3600
        e.step()
        again, w2, _ = self._engine(tmp_path, t=MORNING + 600)
        again.store = LearningStore(tmp_path / "initiative.json")
        again.step()
        assert again.pending == [] and again.bubble_text() is None

    def test_a_restart_is_not_an_arrival(self, tmp_path):
        e, w, _ = self._engine(tmp_path)
        e.store.last_active = MORNING - 120  # at the keyboard two minutes ago
        ctx = e.step()
        assert ctx.arrived is False and e.bubble_text() is None

    def test_outside_the_window(self, tmp_path):
        e, w, _ = self._engine(tmp_path, t=AFTERNOON)
        e.store.last_active = AFTERNOON - 12 * 3600
        assert e.step().arrived is True
        assert e.bubble_text() is None and e.store.morning_day == ""

    def test_waits_until_the_owner_is_free(self, tmp_path):
        e, w, _ = self._engine(tmp_path)
        e.store.last_active = MORNING - 12 * 3600
        e.hooks.busy = lambda: "turn"
        e.step()
        assert e.bubble_text() is None and len(e.pending) == 1
        e.hooks.busy = lambda: None
        e.test_clock.t += 25
        e.step()
        assert e.bubble_text() is not None

    def test_assembly(self):
        facts = MO.MorningFacts(
            weather="hoy 22°", orca={"working": ["A"], "done": ["B"], "failed": ["C"], "waiting": []},
            reminders=[("10:30", "llamar a Ana")], note=("Nota del 15/10/2026 a las 18:00", 4),
            issues=["<b>x</b>"], events=[("11:00", "Daily")])
        title, md = MO.assemble(facts, MORNING)
        assert title == "Buen día"
        assert "_Viernes 16 de octubre_" in md
        for bit in ("## Clima", "Hoy 22°.", "## Agenda", "**11:00** Daily", "Trabajando: «A»",
                    "Terminó: «B»", "Falló: «C»", "**10:30** llamar a Ana",
                    "Nota del 15/10/2026 a las 18:00 (4 líneas)", "&lt;b&gt;x&lt;/b&gt;"):
            assert bit in md, bit
        _, empty = MO.assemble(MO.MorningFacts(), MORNING)
        assert "Nada pendiente" in empty and "## Agenda" not in empty

    def test_last_note(self, tmp_path):
        assert MO.last_note(tmp_path) is None
        (tmp_path / "2026-10-15 18-00.md").write_text(
            "# Nota del 15/10/2026 a las 18:00\n\n- **18:00:01** hola\n- **18:00:05** chau\n")
        assert MO.last_note(tmp_path) == ("Nota del 15/10/2026 a las 18:00", 2)

    def test_reminders_today(self, monkeypatch):
        from datetime import datetime

        from wawi import reminders

        items = [{"due": datetime(2026, 10, 16, 18, 0).isoformat(), "text": "tarde"},
                 {"due": datetime(2026, 10, 16, 10, 30).isoformat(), "text": "Ana"},
                 {"due": datetime(2026, 10, 18, 9, 0).isoformat(), "text": "otro día"}]
        monkeypatch.setattr(reminders, "get", lambda: SimpleNamespace(list_items=lambda kind: items))
        assert MO.reminders_today(MORNING) == [("10:30", "Ana"), ("18:00", "tarde")]

    def test_the_calendar_only_when_opted_in(self, tmp_path, monkeypatch):
        from wawi.initiative import calendar

        asked = []
        monkeypatch.setattr(calendar, "today_events", lambda now: asked.append(now) or [("9:00", "X")])
        e, w, _ = self._engine(tmp_path)
        assert w.facts(MORNING).events is None and asked == []
        e.store.calendar = True
        assert w.facts(MORNING).events == [("9:00", "X")]

    def test_no_eventkit_means_no_calendar(self, monkeypatch):
        import builtins

        from wawi.initiative import calendar

        real = builtins.__import__

        def no_eventkit(name, *a, **kw):
            if name == "EventKit":
                raise ImportError(name)
            return real(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", no_eventkit)
        assert calendar.today_events(MORNING) is None
        assert calendar.request_access() is False


# --------------------------------------------------------------------------- #
# Budget and the engine
# --------------------------------------------------------------------------- #


def ctx(**kw):
    base = dict(now=1000.0, idle_s=1.0)
    base.update(kw)
    return Ctx(**base)


class TestBudget:
    def test_offers_per_hour(self):
        b = Budget(3, 1)
        for i in range(3):
            assert b.allows("offer", ctx(now=1000.0 + i))
            b.spend("offer", 1000.0 + i)
        assert not b.allows("offer", ctx(now=1100.0))
        assert b.allows("offer", ctx(now=1000.0 + 3601))

    def test_one_spoken_line_per_hour_and_never_at_night_or_quiet(self):
        b = Budget(3, 1)
        assert b.allows("say", ctx())
        b.spend("say", 1000.0)
        assert not b.allows("say", ctx(now=1200.0)) and b.allows("offer", ctx(now=1200.0))
        fresh = Budget(3, 1)
        assert not fresh.allows("say", ctx(night=True))
        assert not fresh.allows("say", ctx(quiet=True))

    @pytest.mark.parametrize("gate", [dict(busy="turn"), dict(busy="notes"), dict(busy="game"),
                                      dict(meeting=True), dict(idle_s=600.0)])
    def test_never_while_busy_in_a_meeting_or_away(self, gate):
        b = Budget(3, 1)
        assert not b.allows("offer", ctx(**gate))
        assert b.allows("quiet", ctx(**gate))


class TestEngine:
    def test_an_offer_nobody_clicks_is_ignored(self):
        w = Watch()
        e = engine(watchers=[w])
        w.queue = [Moment("orca_done", "k1")]
        e.step()
        assert e.bubble_text() == "Hola\nAbrir…"
        e.test_clock.t += 31
        e.step()
        assert e.bubble_text() is None
        assert e.store.row("orca_done")["ignored"] == 1

    def test_one_bubble_at_a_time_and_the_ttl(self):
        w = Watch()
        e = engine(watchers=[w])
        w.queue = [Moment("orca_done", "k1"), Moment("orca_done", "k2")]
        e.step()
        assert len(e.pending) == 1
        e.click(on_link=False)
        assert e.store.row("orca_done")["dismissed"] == 1
        e.test_clock.t += 3600  # past the second one's ttl
        e.step()
        assert e.bubble_text() is None and e.pending == []

    def test_the_same_moment_is_acted_on_once(self):
        w = Watch()
        e = engine(watchers=[w])
        w.queue = [Moment("orca_done", "same")]
        e.step()
        e.click(on_link=True)
        w.queue = [Moment("orca_done", "same")]
        e.step()
        assert e.bubble_text() is None

    def test_quiet_level_only_prepares(self):
        prepared = []
        w = Watch(lambda m, c: Act(m.kind, "x", link="y", prepare=lambda: prepared.append(1)))
        e = engine(watchers=[w])
        e.store.row("morning")["level"] = "quiet"
        w.queue = [Moment("morning", "m")]
        e.step()
        assert prepared == [1] and e.bubble_text() is None and e.pending == []

    def test_off_does_nothing_at_all(self):
        prepared = []
        w = Watch(lambda m, c: Act(m.kind, "x", prepare=lambda: prepared.append(1)))
        e = engine(watchers=[w])
        e.set_kind("orca_done", False)
        w.queue = [Moment("orca_done", "m")]
        e.step()
        assert prepared == [] and e.bubble_text() is None

    def test_auto_does_it_for_the_owner(self):
        w = Watch(lambda m, c: Act(m.kind, "Te dejé…", link="Ver…", accept=lambda: w.accepted.append(1),
                                   auto_text="Te lo abrí."))
        e = engine(watchers=[w])
        e.store.row("morning")["level"] = "auto"
        w.queue = [Moment("morning", "m")]
        e.step()
        assert w.accepted == [1] and e.bubble_text() == "Te lo abrí."

    def test_the_master_switch(self, tmp_path):
        w = Watch()
        e = engine(tmp_path, watchers=[w])
        e.set_enabled(False)
        w.queue = [Moment("orca_done", "k")]
        assert e.step() is None and e.bubble_text() is None
        assert json.loads((tmp_path / "initiative.json").read_text())["enabled"] is False


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #


class TestLearning:
    def test_steps(self):
        assert step_up("offer", "auto") == "auto" and step_up("offer", "offer") == "offer"
        assert step_up("quiet", "say") == "offer" and step_down("say") == "offer"
        assert step_down("offer") == "quiet" and step_down("quiet") == "off"

    def test_always_accepted_is_promoted(self, tmp_path):
        s = LearningStore(tmp_path / "l.json")
        for _ in range(2):
            assert s.record("morning", "accepted", default="offer", top="auto") is None
        assert s.record("morning", "accepted", default="offer", top="auto") == "auto"
        assert s.level("morning", "offer") == "auto"
        for _ in range(3):
            s.record("orca_done", "accepted", default="offer", top="say")
        assert s.level("orca_done", "offer") == "say"
        for _ in range(6):
            s.record("meeting_notes", "accepted", default="offer", top="offer")
        assert s.level("meeting_notes", "offer") == "offer"  # its ceiling

    def test_ignored_is_demoted_down_to_off(self, tmp_path):
        s = LearningStore(tmp_path / "l.json")
        for _ in range(3):
            s.record("orca_done", "ignored", default="offer", top="say")
        assert s.level("orca_done", "offer") == "quiet"
        for _ in range(3):
            s.record("orca_done", "ignored", default="offer", top="say")
        assert s.level("orca_done", "offer") == "off"

    def test_a_dismissal_counts_double_and_a_yes_resets(self, tmp_path):
        s = LearningStore(tmp_path / "l.json")
        s.record("orca_done", "dismissed", default="offer", top="say")
        s.record("orca_done", "accepted", default="offer", top="say")
        s.record("orca_done", "dismissed", default="offer", top="say")
        assert s.level("orca_done", "offer") == "offer"
        s.record("orca_done", "ignored", default="offer", top="say")
        assert s.level("orca_done", "offer") == "quiet"

    def test_the_owner_wins_and_on_forgets_a_learned_off(self, tmp_path):
        s = LearningStore(tmp_path / "l.json")
        for _ in range(3):
            s.record("morning", "accepted", default="offer", top="auto")
        s.set_owner("morning", False, "offer")
        assert s.level("morning", "offer") == "off"
        s.set_owner("morning", True, "offer")
        assert s.level("morning", "offer") == "auto"  # what it learned is kept
        for _ in range(6):
            s.record("orca_done", "ignored", default="offer", top="say")
        s.set_owner("orca_done", True, "offer")
        assert s.level("orca_done", "offer") == "offer"

    def test_persisted_and_counters_only(self, tmp_path):
        s = LearningStore(tmp_path / "l.json")
        s.record("orca_done", "accepted", default="offer", top="say")
        s.calendar, s.morning_day, s.last_active = True, "2026-10-16", 123.0
        s.save()
        again = LearningStore(tmp_path / "l.json")
        assert again.row("orca_done")["accepted"] == 1
        assert again.calendar and again.morning_day == "2026-10-16" and again.last_active == 123.0
        (tmp_path / "bad.json").write_text("{nope")
        assert LearningStore(tmp_path / "bad.json").kinds == {}

    def test_the_engine_teaches_it(self):
        w = Watch()
        e = engine(watchers=[w])
        for i in range(3):
            w.queue = [Moment("orca_done", f"k{i}")]
            e.test_clock.t += 1300  # out of the hourly budget's way
            e.step()
            e.click(on_link=True)
        assert e.level("orca_done") == "say"


# --------------------------------------------------------------------------- #
# Menu and voice
# --------------------------------------------------------------------------- #


class TestControls:
    def test_menu_rows_and_clicks(self):
        e = engine()
        rows = menu.entries(e)
        assert rows[0] == (menu.MASTER, "@master", True, True)
        titles = {what: (title, checked) for title, what, checked, _ in rows}
        assert titles["meeting_quiet"] == ("Me callo cuando estás en una reunión (en silencio)", True)
        assert titles["@calendar"][1] is False
        menu.click(e, "orca_done")
        assert e.level("orca_done") == "off"
        menu.click(e, "orca_done")
        assert e.level("orca_done") == "offer"
        menu.click(e, "@master")
        assert not e.enabled and all(not en for _, w, _, en in menu.entries(e)[2:] if w)

    def test_calendar_click_asks_for_access(self, monkeypatch):
        from wawi.initiative import calendar

        asked = []
        monkeypatch.setattr(calendar, "request_access", lambda done=None: asked.append(1) or True)
        e = engine()
        menu.click(e, "@calendar")
        assert e.store.calendar and asked == [1]
        menu.click(e, "@calendar")
        assert not e.store.calendar and asked == [1]

    def test_what_do_you_do_on_your_own(self):
        e = engine()
        ok, text, _ = T.run(e, "list")
        assert ok and text.startswith("Sola hago esto: me callo cuando estás en una reunión en silencio")
        e.set_kind("morning", False)
        assert "Apagado: te dejo el resumen de la mañana." in T.run(e, "list")[1]

    def test_dont_tell_me_about_that_any_more(self):
        w = Watch()
        e = engine(watchers=[w])
        assert T.run(e, "off")[0] is False  # nothing shown lately
        w.queue = [Moment("orca_done", "k")]
        e.step()
        ok, text, data = T.run(e, "off")
        assert ok and data == {"kind": "orca_done", "on": False}
        assert text == "Listo, ya no te aviso cuando un agente de Orca termina."
        assert e.level("orca_done") == "off" and e.bubble_text() is None
        assert T.run(e, "on", "orca_done")[0] and e.level("orca_done") == "offer"
        assert T.run(e, "off", "nada")[0] is False
        T.run(e, "off", "all")
        assert not e.enabled
        assert T.run(None, "list")[0] is False

    def test_the_tool_is_only_sent_while_installed(self):
        assert "initiative" not in {t["name"] for t in tools.active_tools()}
        initiative.install(engine())
        names = [t for t in tools.active_tools() if t["name"] == "initiative"]
        assert names and "animation" not in names[0]
        assert len(json.dumps(tools.INITIATIVE_TOOLS)) < 450
        r = tools.dispatch("initiative", {"action": "list"})
        assert r.ok and "Sola hago esto" in r.speak

    def test_the_panel_facade(self):
        w = Watch()
        e = engine(watchers=[w])
        initiative.install(e)
        assert initiative.bubble_text() is None and not initiative.owns("Hola\nAbrir…")
        w.queue = [Moment("orca_done", "k")]
        e.step()
        text = initiative.bubble_text()
        assert initiative.owns(text) and initiative.bubble_link(text) == "Abrir…"
        assert initiative.bubble_link("otra cosa\nAbrir…") == "Abrir…"
        assert initiative.bubble_link("Hola") is None
        initiative.click(True)
        assert w.accepted and initiative.bubble_text() is None


class TestBuild:
    def test_build_from_config(self, tmp_path):
        cfg = SimpleNamespace(initiative={"local_file": str(tmp_path / "i.json"),
                                          "meeting": {"exit_after_s": 30}}, pet={})
        e = initiative.build(cfg, Hooks())
        assert e is not None
        assert [type(w).__name__ for w in e.watchers] == ["MeetingWatcher", "OrcaWatcher",
                                                          "MorningWatcher"]
        assert e.settings.meeting_exit_s == 30
        assert initiative.build(SimpleNamespace(initiative={"enabled": False})) is None

    def test_config_defaults_and_local_file(self, tmp_path):
        from wawi.config import Config  # noqa: F401
        from wawi import config as C

        assert C._DEFAULTS["initiative"]["offers_per_hour"] == 3
        section = C._initiative_section({"local_file": "initiative_local.json"}, tmp_path / "config.yaml")
        assert section["local_file"] == str(tmp_path / "initiative_local.json")

    def test_settings_parse_badly_typed_values(self):
        s = InitiativeSettings.from_dict({"tick_s": "x", "offers_per_hour": 99,
                                          "morning": {"from_hour": 7}, "meeting": "nope"})
        assert s.tick_s == 25.0 and s.offers_per_hour == 30 and s.morning_from_h == 7.0


def test_the_meet_script_asks_only_for_front_tabs():
    """Live: the nested every-tab form failed (-1700) and background calls kept her quiet."""
    from wawi.initiative import meeting as M

    chrome = M._TABS_SCRIPT.format(bundle="com.google.Chrome", tab="active tab")
    assert "URL of active tab of every window" in chrome and "every tab" not in chrome
    safari = M._TABS_SCRIPT.format(bundle="com.apple.Safari", tab="current tab")
    assert "current tab" in safari
