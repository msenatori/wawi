"""Senses configured by voice, sensible defaults, the morning greeting,
Wawi's spoken reactions and her procedural barks.

No network, no microphone, no voice, no real clock: geocoding, the voice and
the quiet detector are fakes, and the runner is driven with tick().
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
import wave
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from wawi import sensors as sensors_pkg
from wawi import tools
from wawi.petvoice import PetVoice
from wawi.reactions import Event, Reaction, ReactionBus
from wawi.reflexes import QuietState, Reflexes, build_rules
from wawi.sensors import live, morning, net
from wawi.sensors.clock import ClockSensor
from wawi.sensors.runner import SensorRunner
from wawi.sensors.services import ServicesSensor
from wawi.sensors.system import SystemSensor
from wawi.sensors.weather import WeatherSensor
from wawi.sensors.web import WebSensor

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


class TestDefaults:
    def test_empty_section_gets_system_and_clock(self):
        built = sensors_pkg.build_sensors(live.effective({}, {}))
        assert sorted(s.name for s in built) == ["clock", "system"]
        system = next(s for s in built if s.name == "system")
        assert system.battery is True and system.memory is False and system.cpu is False

    def test_nothing_that_needs_personal_input_or_network(self):
        names = set(live.with_defaults({}))
        assert not names & {"web", "rss", "weather", "services"}

    def test_a_configured_sensor_replaces_the_defaults(self):
        cfg = live.with_defaults({"web": {"enabled": True, "urls": ["https://a.example"]}})
        assert set(cfg) == {"web"}

    def test_opt_outs(self):
        assert live.with_defaults({"defaults": False}) == {"defaults": False}
        assert live.effective({"enabled": False}, {"web": {"set": {"enabled": True}}}) == {"enabled": False}

    def test_example_config_documents_the_defaults(self):
        text = (ROOT / "config.yaml.example").read_text()
        assert "defaults: false" in text and "sensors_local.json" in text


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    @pytest.mark.parametrize("raw, expect", [
        ("example.com", "https://example.com"),
        ("https://example.com/health", "https://example.com/health"),
        ("http://localhost:3000", "http://localhost:3000"),
        ("localhost:3000", "http://localhost:3000"),
        ("192.168.1.10:8080/status", "http://192.168.1.10:8080/status"),
    ])
    def test_good_urls(self, raw, expect):
        assert live.clean_url(raw) == expect

    @pytest.mark.parametrize("raw", [
        "file:///etc/passwd", "ftp://example.com", "javascript:alert(1)",
        "https://user:pw@example.com", "", "no spaces.com/a b", "intranet",
        "https://example.com:99999", "mailto:a@b.com",
    ])
    def test_bad_urls(self, raw):
        with pytest.raises(live.Invalid):
            live.clean_url(raw)

    @pytest.mark.parametrize("raw, expect", [
        ("3000", "3000"), ("puerto 8080", "8080"), ("localhost:5432", "5432"),
        ("db.local:5432", "db.local:5432"), ("1", "1"), ("65535", "65535"),
    ])
    def test_good_ports(self, raw, expect):
        assert live.clean_port(raw) == expect

    @pytest.mark.parametrize("raw", ["0", "65536", "-1", "abc", "host name:80"])
    def test_bad_ports(self, raw):
        with pytest.raises(live.Invalid):
            live.clean_port(raw)

    def test_names_and_cities(self):
        assert live.clean_name("  my-api ", "contenedor") == "my-api"
        assert live.clean_name("Google Chrome", "proceso") == "Google Chrome"
        assert live.clean_city("Córdoba") == "Córdoba"
        for bad in ("rm -rf /;", "$(x)", "", "a" * 80):
            with pytest.raises(live.Invalid):
                live.clean_name(bad, "proceso")
        with pytest.raises(live.Invalid):
            live.clean_city("12345")


# --------------------------------------------------------------------------- #
# Deltas
# --------------------------------------------------------------------------- #


class TestDeltas:
    def test_add_then_remove_round_trip(self):
        base = {"web": {"enabled": True, "urls": ["https://a.example"]}}
        local: dict = {}
        live.add_item(local, "web", "urls", "https://b.example")
        merged = live.merge_local(base, local)
        assert merged["web"]["urls"] == ["https://a.example", "https://b.example"]
        # Removing something that came from config.yaml is a stored "remove".
        live.remove_item(local, "web", "urls", "https://a.example/")
        merged = live.merge_local(base, local)
        assert merged["web"]["urls"] == ["https://b.example"]
        live.remove_item(local, "web", "urls", "https://b.example")
        assert live.merge_local(base, local)["web"]["urls"] == []
        # ...and adding it back cancels the remove.
        live.add_item(local, "web", "urls", "https://a.example")
        assert live.merge_local(base, local)["web"]["urls"] == ["https://a.example"]

    def test_deltas_survive_config_edits(self):
        local: dict = {}
        live.add_item(local, "web", "urls", "https://b.example")
        later = {"web": {"enabled": True, "urls": ["https://new-in-yaml.example"]}}
        assert live.merge_local(later, local)["web"]["urls"] == [
            "https://new-in-yaml.example", "https://b.example"]

    def test_ports_compare_by_meaning(self):
        base = {"services": {"enabled": True, "ports": ["localhost:3000"]}}
        assert live.contains(base, "services", "ports", 3000)
        assert live.contains(base, "services", "ports", "127.0.0.1:3000")

    def test_weather_coordinates_replace_the_city(self):
        base = {"weather": {"enabled": True, "city": "Old"}}
        local: dict = {}
        live.set_values(local, "weather", {"lat": 1.0, "lon": 2.0})
        assert "city" not in live.merge_local(base, local)["weather"]

    def test_file_round_trip_and_broken_file(self, tmp_path):
        path = tmp_path / "sensors_local.json"
        assert live.load_local(path) == {}
        live.save_local(path, {"web": {"add": {"urls": ["https://x.example"]}}})
        assert json.loads(path.read_text())["version"] == 1
        assert live.load_local(path)["web"]["add"]["urls"] == ["https://x.example"]
        path.write_text("{not json")
        assert live.load_local(path) == {}

    def test_local_file_is_gitignored(self):
        assert "sensors_local.json" in (ROOT / ".gitignore").read_text().splitlines()


# --------------------------------------------------------------------------- #
# The runner picks changes up live
# --------------------------------------------------------------------------- #


Resp = net.Response


class TestRunnerLive:
    def test_replace_add_remove(self):
        bus = ReactionBus()
        clock = [0.0]
        runner = SensorRunner([], bus, clock=lambda: clock[0], jitter=0.0)
        web = WebSensor({"urls": ["https://a.example"]}, fetch=lambda *a, **k: Resp(200, 1))
        runner.replace(web)
        assert runner.sensors == [web]
        clock[0] = 2.0
        assert runner.tick() == 1  # a new sensor polls soon, not at its interval
        clock_only = ClockSensor({"dayparts": {}}, idle=lambda: None)
        runner.replace(clock_only)
        assert [s.name for s in runner.sensors] == ["web", "clock"]
        assert runner.remove("web") and not runner.remove("web")
        assert [s.name for s in runner.sensors] == ["clock"]

    def test_replacement_keeps_a_down_site_down(self):
        """Else the firefighter costume put on by web.down is never cancelled."""
        url = "https://a.example"
        answers = iter([None, None, 200])
        fetch = lambda *a, **k: Resp(next(answers), 1)  # noqa: E731
        old = WebSensor({"urls": [url], "fail_after": 2}, fetch=fetch)
        assert old.poll() == [] and [e.kind for e in old.poll()] == ["web.down"]
        new = WebSensor({"urls": [url, "https://b.example"], "fail_after": 2}, fetch=fetch)
        new.inherit(old)
        new.urls = [url]
        assert [e.kind for e in new.poll()] == ["web.up"]

    def test_services_inherit_only_what_is_still_watched(self):
        old = ServicesSensor({"ports": [3000, 4000]}, port_check=lambda h, p: True)
        old.poll()
        new = ServicesSensor({"ports": [3000]}, port_check=lambda h, p: True)
        new.inherit(old)
        assert set(new._deb.states()) == {"service:port:puerto 3000"}

    def test_thread_wakes_up_for_a_change(self):
        polled = threading.Event()

        class Ping(ClockSensor):
            name = "ping"

            def poll(self):
                polled.set()
                return []

        runner = SensorRunner([], ReactionBus(), jitter=0.0)
        runner.start()
        try:
            runner.replace(Ping({"dayparts": {}}, idle=lambda: None))
            assert polled.wait(5.0)  # well under the 30 s idle nap
        finally:
            runner.stop()


# --------------------------------------------------------------------------- #
# The voice tools, end to end over a real (unstarted-thread) runner
# --------------------------------------------------------------------------- #


@pytest.fixture
def running(tmp_path, monkeypatch):
    """sensors.start() with a tmp local file and no real network."""
    monkeypatch.setattr(net, "fetch", lambda *a, **k: Resp(200, 1))
    monkeypatch.setattr(SensorRunner, "start", lambda self: None)

    class Cfg:
        sensors: dict = {}
        reflexes = {"enabled": False}

    bus = ReactionBus()
    path = tmp_path / "sensors_local.json"
    runner = sensors_pkg.start(Cfg(), bus, local_path=path)
    tools.set_sensors_active(True)
    yield runner, path, bus
    tools.set_sensors_active(False)
    sensors_pkg.stop()


def names(runner):
    return sorted(s.name for s in runner.sensors)


class TestVoiceTools:
    def test_defaults_run_and_the_tool_is_offered(self, running):
        runner, _, _ = running
        assert names(runner) == ["clock", "system"]
        assert "watch" in {t["name"] for t in tools.active_tools()}

    def test_watch_a_site_live_and_persisted(self, running):
        runner, path, _ = running
        r = tools.dispatch("watch", {"action": "add", "kind": "web", "target": "example.com"})
        assert r.ok and r.speak == "Listo, vigilo example.com."
        assert names(runner) == ["clock", "system", "web"]
        assert runner.get("web").urls == ["https://example.com"]
        stored = json.loads(path.read_text())["sensors"]
        assert stored["web"]["add"]["urls"] == ["https://example.com"]
        again = tools.dispatch("watch", {"action": "add", "kind": "web", "target": "https://example.com/"})
        assert again.ok and "Ya estaba" in again.speak

    def test_localhost_dev_server_is_allowed(self, running):
        r = tools.dispatch("watch", {"action": "add", "kind": "web", "target": "localhost:3000"})
        assert r.ok and "localhost:3000" in r.speak

    def test_invalid_targets_are_refused_and_nothing_changes(self, running):
        runner, path, _ = running
        for args in (
            {"action": "add", "kind": "web", "target": "file:///etc/passwd"},
            {"action": "add", "kind": "feed", "target": "ftp://x.example/feed"},
            {"action": "add", "kind": "port", "target": "70000"},
            {"action": "add", "kind": "process", "target": "x; rm -rf /"},
            {"action": "add", "kind": "satellite", "target": "x"},
            {"action": "add", "kind": "web", "target": ""},
            {"action": "fly"},
        ):
            r = tools.dispatch("watch", args)
            assert r.ok is False and r.speak, args
        assert names(runner) == ["clock", "system"]
        assert not path.exists()

    def test_unwatch_cancels_the_costume(self, running):
        runner, path, bus = running
        tools.dispatch("watch", {"action": "add", "kind": "web", "target": "https://example.com"})
        r = tools.dispatch("watch", {"action": "remove", "kind": "web", "target": "example.com"})
        assert r.ok and "dejé de vigilar example.com" in r.speak
        assert runner.get("web").urls == []
        cancels = []
        while (x := bus.pop()) is not None:
            cancels.append(x.cancel_key)
        assert "web:https://example.com" in cancels
        r = tools.dispatch("watch", {"action": "remove", "kind": "web", "target": "example.com"})
        assert r.ok and "No estaba" in r.speak

    def test_ports_containers_processes_and_feeds(self, running):
        runner, _, _ = running
        for kind, target in (("port", "puerto 3000"), ("container", "api"),
                             ("process", "postgres"), ("feed", "https://blog.example/feed.xml")):
            assert tools.dispatch("watch", {"action": "add", "kind": kind, "target": target}).ok
        svc = runner.get("services")
        assert [p for _, p in svc.ports] == [3000]
        assert svc.docker_names == ["api"] and svc.process_names == ["postgres"]
        assert runner.get("rss").feeds == [("https://blog.example/feed.xml", "")]
        listing = tools.dispatch("watch", {"action": "list"})
        assert listing.ok
        for bit in ("el puerto 3000", "el proceso postgres", "el contenedor api",
                    "un feed de noticias", "el disco y la batería", "la hora y tus pausas"):
            assert bit in listing.speak

    def test_weather_city_is_geocoded(self, running, monkeypatch):
        runner, path, _ = running
        monkeypatch.setattr(live, "geocode", lambda city, fetch=None: {
            "name": "Córdoba", "lat": -31.41, "lon": -64.18, "country": "Argentina"})
        r = tools.dispatch("watch", {"action": "add", "kind": "weather", "target": "cordoba"})
        assert r.ok and r.speak == "Listo, te aviso del clima de Córdoba."
        weather = runner.get("weather")
        assert (weather.city, weather.lat, weather.lon) == ("Córdoba", -31.41, -64.18)
        assert "el clima de Córdoba" in tools.dispatch("watch", {"action": "list"}).speak
        assert tools.dispatch("watch", {"action": "remove", "kind": "weather"}).ok
        assert runner.get("weather") is None

    def test_weather_unknown_city_and_offline(self, running, monkeypatch):
        monkeypatch.setattr(live, "geocode", lambda city, fetch=None: None)
        r = tools.dispatch("watch", {"action": "add", "kind": "weather", "target": "Nowhere"})
        assert not r.ok and "No encontré" in r.speak

        def offline(city, fetch=None):
            raise OSError("down")

        monkeypatch.setattr(live, "geocode", offline)
        r = tools.dispatch("watch", {"action": "add", "kind": "weather", "target": "Rosario"})
        assert not r.ok and "internet" in r.speak

    def test_geocode_parses_open_meteo(self):
        payload = {"results": [{"name": "Rosario", "latitude": -32.9468, "longitude": -60.6393,
                                "country": "Argentina"}]}
        seen = []
        place = live.geocode("Rosario", lambda url: seen.append(url) or payload)
        assert place == {"name": "Rosario", "lat": -32.9468, "lon": -60.6393, "country": "Argentina"}
        assert seen[0].startswith(live.GEOCODE_URL) and "name=Rosario" in seen[0]
        assert live.geocode("x", lambda url: {}) is None

    def test_changes_survive_a_restart(self, running):
        runner, path, _ = running
        tools.dispatch("watch", {"action": "add", "kind": "web", "target": "example.com"})
        sensors_pkg.stop()

        class Cfg:
            sensors: dict = {}
            reflexes = {"enabled": False}

        runner = sensors_pkg.start(Cfg(), ReactionBus(), local_path=path)
        assert runner.get("web").urls == ["https://example.com"]

    def test_tools_say_so_when_sensors_are_off(self):
        sensors_pkg.stop()
        assert sensors_pkg.watch("web", "example.com")[0] is False
        assert "apagados" in tools.watch("list").speak


# --------------------------------------------------------------------------- #
# Morning greeting
# --------------------------------------------------------------------------- #

GB = 1000 ** 3
Usage = namedtuple("Usage", "total used free")


def weather_with(today_max=24, rain_at=None):
    w = WeatherSensor({"lat": 0, "lon": 0})
    w._last = {"temp": 15.0, "rain_at": rain_at, "today": {"max": float(today_max), "min": 10.0, "rain": 10}}
    return w


def web_with(states):
    urls = list(states)
    answers = {u: (200 if ok else None) for u, ok in states.items()}
    web = WebSensor({"urls": urls, "fail_after": 1}, fetch=lambda url, **k: Resp(answers[url], 1))
    web.poll()
    return web


class TestMorning:
    def test_everything_fine(self):
        text = morning.compose([weather_with(24), web_with({"https://a.example": True, "https://b.example": True})])
        assert text == "Buen día. Hoy 24°, todas tus webs andan."

    def test_down_site_rain_and_disk(self):
        system = SystemSensor({"battery": False}, disk_usage=lambda v: Usage(500 * GB, 0, 8 * GB))
        system.poll()
        text = morning.compose([
            weather_with(18, rain_at="15:00"),
            web_with({"https://a.example": False, "https://b.example": True}),
            system,
        ])
        assert text.startswith("Buen día. Hoy 18° y llueve a las 15:00, está caída a.example")
        assert len(text) <= 140

    def test_nothing_known_is_just_hello(self):
        assert morning.compose([]) == "¡Buen día!"
        unchecked = WebSensor({"urls": ["https://a.example"]}, fetch=lambda *a, **k: Resp(200, 1))
        assert morning.compose([unchecked, WeatherSensor({"lat": 0, "lon": 0})]) == "¡Buen día!"
        healthy = SystemSensor({"battery": False}, disk_usage=lambda v: Usage(500 * GB, 0, 200 * GB))
        healthy.poll()
        assert morning.compose([healthy]) == "¡Buen día!"

    def test_reflex_uses_the_composer_and_the_budget(self):
        bus = ReactionBus()

        class NotQuiet:
            def check(self):
                return QuietState()

        rx = Reflexes({"budget": {"info": 1}}, bus=bus, clock=lambda: 1000.0, quiet=NotQuiet())
        rx.set_composer("clock.morning", lambda e: "Buen día. Hoy 24°.")
        rx.attach()
        try:
            rx.on_event(Event("clock.morning", "info", "clock:morning:1", "morning"))
            r = bus.pop()
            assert r.bubble == "Buen día. Hoy 24°." and r.speak is False
            rx.on_event(Event("clock.morning", "info", "clock:morning:2", "morning"))
            assert bus.pop() is None  # over the info budget
        finally:
            rx.detach()

    def test_speaking_the_greeting_is_opt_in(self):
        assert build_rules({})["clock.morning"]["speak"] is False
        assert build_rules({"speak_morning": True})["clock.morning"]["speak"] is True

    def test_start_installs_the_composer(self, running):
        assert sensors_pkg._morning_bubble(None) == "¡Buen día!"


# --------------------------------------------------------------------------- #
# Spoken reactions
# --------------------------------------------------------------------------- #


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def voice(quiet=False, busy=None, **kw):
    said = []
    busy = busy if busy is not None else [False]

    def speak_now(text):
        if busy[0]:
            return False
        said.append(text)
        return True

    clock = Clock()
    pv = PetVoice(speak_now, quiet=lambda: quiet, clock=clock, start_thread=False, **kw)
    return pv, said, busy, clock


class TestPetVoice:
    def test_speaks_when_idle(self):
        pv, said, _, _ = voice()
        pv("¡Se cayó example.com!")
        assert pv.step() == "spoken" and said == ["¡Se cayó example.com!"]
        assert pv.step() == "empty"

    def test_waits_for_the_turn_then_speaks(self):
        pv, said, busy, clock = voice()
        busy[0] = True
        pv("hola")
        assert pv.step() == "busy" and said == [] and pv.pending() == ["hola"]
        clock.t = 30
        busy[0] = False
        assert pv.step() == "spoken" and said == ["hola"]

    def test_stale_lines_are_dropped(self):
        pv, said, busy, clock = voice(stale_s=60)
        busy[0] = True
        pv("vieja")
        clock.t = 61
        busy[0] = False
        assert pv.step() == "stale" and said == []

    @pytest.mark.parametrize("quiet", [True, None])
    def test_quiet_or_unknown_means_silent(self, quiet):
        pv, said, _, _ = voice(quiet=quiet)
        pv("shh")
        assert pv.step() == "quiet" and said == [] and pv.pending() == []

    def test_rate_limit_per_hour(self):
        pv, said, _, clock = voice(max_per_hour=2)
        for i in range(3):
            pv(f"línea {i}")
            pv.step()
        assert said == ["línea 0", "línea 1"]
        clock.t = 3601
        pv("después")
        assert pv.step() == "spoken"

    def test_duplicates_and_queue_bound(self):
        pv, _, busy, _ = voice()
        busy[0] = True
        for text in ("a", "a", "b", "c", "d"):
            pv(text)
        assert pv.pending() == ["b", "c", "d"]

    def test_switched_off(self):
        class Cfg:
            reflexes = {"speak_reactions": False}

        from wawi import petvoice

        assert petvoice.build(Cfg(), lambda t: True) is None

        class Cfg2:
            reflexes = {"speak_per_hour": 2, "speak_stale_s": 30}

        pv = petvoice.build(Cfg2(), lambda t: True)
        assert pv.max_per_hour == 2 and pv.stale_s == 30

    def test_thread_drains_the_queue(self):
        said = []
        pv = PetVoice(lambda t: said.append(t) or True, quiet=lambda: False, retry_s=0.01)
        try:
            pv("hola")
            deadline = time.monotonic() + 5
            while not said and time.monotonic() < deadline:
                time.sleep(0.01)
            assert said == ["hola"]
        finally:
            pv.stop()


class FakeVoice:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class FakeListener:
    def __init__(self):
        self.calls = []

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def set_speech_floor(self, level):
        self.calls.append(("floor", level))

    def set_echo_text(self, text):
        self.calls.append(("echo", text))

    def arm_follow_up(self, seconds):
        self.calls.append(("follow_up", seconds))


class FakePanel:
    channel_open = False

    def set_pet_speaker(self, speaker):
        self.speaker = speaker


class FakeCfg:
    conversation_mode = True  # would hold the channel open after an ANSWER
    wake_follow_up_s = 20.0
    wake_follow_up_idle_s = 5.0
    barge_in = False
    barge_in_level = 0.3
    wake_engine = "whisper"
    wake_phrases = ("wawi",)
    reflexes: dict = {}


def orchestrator(listener=None, cfg=None):
    from wawi.orchestrator import Orchestrator
    from wawi.state import StateMachine

    orch = object.__new__(Orchestrator)
    orch._cfg = cfg or FakeCfg()
    orch._last_response = ""
    orch._pending_confirmation = None
    orch._barge_in_armed = False
    orch._reaction_speech = False
    orch._turn_lock = threading.Lock()
    orch._stop_refresh = threading.Event()
    orch._stop_refresh.set()  # the echo-tail timer returns at once
    orch.listener = listener
    orch.state = StateMachine()
    orch.state.subscribe(orch._on_state_change)
    orch.voice = FakeVoice()
    orch.panel = FakePanel()
    return orch


class TestSpeakReaction:
    def test_same_echo_guard_as_an_answer_and_no_follow_up(self):
        from wawi.state import State

        listener = FakeListener()
        orch = orchestrator(listener)
        assert orch._speak_reaction("¡Se cayó example.com!") is True
        assert orch.voice.said == ["¡Se cayó example.com!"]
        assert listener.calls[:2] == [("echo", "¡Se cayó example.com!"), "pause"]
        assert "resume" in listener.calls
        # Nobody asked her anything: even conversation mode keeps it closed.
        assert not any(isinstance(c, tuple) and c[0] == "follow_up" for c in listener.calls)
        assert orch.state.state is State.IDLE and not orch._turn_lock.locked()
        assert orch._reaction_speech is False

    def test_never_over_a_turn(self):
        from wawi.state import State

        orch = orchestrator(FakeListener())
        orch._turn_lock.acquire()
        assert orch._speak_reaction("hola") is False
        orch._turn_lock.release()
        orch.state.set_state(State.LISTENING)  # the user just started talking
        assert orch._speak_reaction("hola") is False
        assert orch.voice.said == []

    def test_never_while_the_owner_has_the_floor(self):
        """A remark inside an open follow-up window would close it."""
        orch = orchestrator(FakeListener())
        orch.panel.channel_open = True
        assert orch._speak_reaction("Tengo hambre") is False
        assert orch.voice.said == []

    def test_barge_in_follows_the_settings(self):
        cfg = FakeCfg()
        cfg.barge_in = True
        listener = FakeListener()
        orch = orchestrator(listener, cfg)
        orch._speak_reaction("La web volvió")
        assert ("floor", 0.3) in listener.calls and "pause" not in listener.calls
        # A line that says her own name would wake her: paused instead.
        listener.calls.clear()
        orch._speak_reaction("Wawi avisa: volvió")
        assert "pause" in listener.calls

    def test_wired_to_the_panel(self):
        orch = orchestrator()
        orch._build_pet_voice(FakeCfg())
        assert isinstance(orch.panel.speaker, PetVoice)
        orch._pet_voice.stop()

    def test_quiet_helpers_follow_the_attached_engine(self):
        from wawi import reflexes

        class Meeting:
            def check(self):
                return QuietState(True, True, "meeting")

        rx = Reflexes({}, bus=ReactionBus(), quiet=Meeting())
        rx.attach()
        try:
            assert reflexes.is_quiet() is True and reflexes.quiet_state().meeting
        finally:
            rx.detach()
        assert reflexes.is_quiet() is None


class TestQuietSounds:
    def test_no_bark_during_quiet_hours(self):
        bus = ReactionBus()

        class Night:
            def check(self):
                return QuietState(True, False, "quiet_hours")

        rx = Reflexes({}, bus=bus, quiet=Night())
        rx.attach()
        try:
            rx.on_event(Event("web.down", "urgent", "web:x", "x", {"url": "x", "host": "x"}))
            r = bus.pop()
            assert r.costume == "firefighter" and r.sound is None and r.speak is False
        finally:
            rx.detach()


# --------------------------------------------------------------------------- #
# Procedural barks
# --------------------------------------------------------------------------- #


def _sounds_module():
    spec = importlib.util.spec_from_file_location("make_wawi_sounds", ROOT / "scripts" / "make_wawi_sounds.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSounds:
    def test_deterministic_short_and_gentle(self):
        m = _sounds_module()
        limits = {"bark": 0.6, "whine": 0.6, "howl": 1.5}
        for name, make in m.SOUNDS.items():
            a, b = make(), make()
            assert np.array_equal(a, b), name
            assert len(a) / m.RATE < limits[name], name
            assert float(np.max(np.abs(a))) <= 0.26, name  # about -12 dBFS at most
            assert abs(a[0]) < 1e-3 and abs(a[-1]) < 1e-3, name  # no clicks

    def test_written_files_match_the_committed_ones(self, tmp_path):
        m = _sounds_module()
        m.main(tmp_path)
        for name in ("bark", "whine", "howl"):
            fresh = (tmp_path / f"{name}.wav").read_bytes()
            committed = ROOT / "wawi" / "assets" / "sounds" / f"{name}.wav"
            assert committed.read_bytes() == fresh, name
            with wave.open(str(committed)) as w:
                assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, m.RATE)

    def test_the_body_finds_them(self):
        from wawi.pet.acting import sound_path
        from wawi.reflexes import SOUNDS

        for name in SOUNDS:
            assert sound_path(name) is not None, name
