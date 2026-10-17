"""Sensors: scheduling, edge-triggering and parsing, with every I/O faked.

Nothing here touches the network, Docker, pmset or the real clock: each
sensor takes its probes as constructor arguments precisely so these tests
can feed it a story ("up, down, down, up") and check it tells the story
once.
"""

from __future__ import annotations

import json
import os
import random
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from wawi import sensors as sensors_pkg
from wawi.reactions import Event, ReactionBus
from wawi.sensors import net
from wawi.sensors.base import Debouncer, Levels, Sensor, human_bytes
from wawi.sensors.clock import ClockSensor
from wawi.sensors.rss import RssSensor, parse_feed
from wawi.sensors.runner import SensorRunner
from wawi.sensors.services import ServicesSensor, parse_docker_ps, parse_port
from wawi.sensors.system import (
    SystemSensor,
    parse_memory_pressure,
    parse_pmset_batt,
    parse_thermal,
)
from wawi.sensors.weather import WeatherSensor, parse_forecast
from wawi.sensors.web import WebSensor

FIXTURES = Path(__file__).parent / "fixtures"


def kinds(events):
    return [e.kind for e in events]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class FakeSensor(Sensor):
    name = "fake"
    default_interval = 10.0
    min_interval = 0.0

    def __init__(self, outcomes=None, config=None):
        super().__init__(config)
        self.outcomes = list(outcomes or [])
        self.polls = 0
        self.setups = 0
        self.closed = False

    def setup(self):
        self.setups += 1

    def poll(self):
        self.polls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else []
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


class TestRunner:
    def make(self, *sensors, jitter=0.0, **kw):
        bus = ReactionBus()
        got = []
        bus.subscribe(got.append)
        runner = SensorRunner(list(sensors), bus, clock=lambda: 0.0, jitter=jitter,
                              rng=random.Random(1), **kw)
        return runner, got

    def test_polls_on_schedule_and_emits_events(self):
        s = FakeSensor([[Event("x.a")], [], [Event("x.b")]])
        runner, got = self.make(s)
        assert runner.tick(0.0) == 1
        assert runner.tick(5.0) == 0  # not due yet
        assert runner.tick(10.0) == 1
        assert runner.tick(20.0) == 1
        assert kinds(got) == ["x.a", "x.b"]
        assert s.setups == 1  # setup once, lazily, on the runner thread

    def test_jitter_stays_within_bounds(self):
        s = FakeSensor()
        runner, _ = self.make(s, jitter=0.1)
        runner.tick(100.0)
        due = runner.next_due()
        assert 109.0 <= due <= 111.0

    def test_exponential_backoff_on_errors_then_recovery(self):
        s = FakeSensor([RuntimeError("1"), RuntimeError("2"), RuntimeError("3"), []])
        runner, _ = self.make(s, max_backoff=50.0)
        runner.tick(0.0)
        assert runner.next_due() == 20.0  # 10 * 2**1
        runner.tick(20.0)
        assert runner.next_due() == 60.0  # 20 + 10 * 2**2
        runner.tick(60.0)
        assert runner.next_due() == 110.0  # capped at max_backoff (50)
        runner.tick(110.0)
        assert runner.next_due() == 120.0  # healthy again: plain interval
        assert runner._slots[0].failures == 0

    def test_failing_setup_backs_off_and_is_retried(self):
        class BadSetup(FakeSensor):
            def setup(self):
                self.setups += 1
                if self.setups == 1:
                    raise OSError("no network yet")

        s = BadSetup([[Event("ok")]])
        runner, got = self.make(s)
        runner.tick(0.0)
        assert s.polls == 0
        runner.tick(20.0)
        assert s.polls == 1 and kinds(got) == ["ok"]

    def test_one_broken_sensor_does_not_stop_the_others(self):
        bad = FakeSensor([RuntimeError("boom")])
        good = FakeSensor([[Event("good")]])
        runner, got = self.make(bad, good)
        runner.tick(0.0)
        assert kinds(got) == ["good"]

    def test_non_events_are_ignored(self):
        s = FakeSensor([["not an event", Event("ok")]])
        runner, got = self.make(s)
        runner.tick(0.0)
        assert kinds(got) == ["ok"]

    def test_hooks_run_every_tick_and_cannot_break_it(self):
        calls = []
        runner, _ = self.make(FakeSensor(), hooks=[lambda: calls.append(1), lambda: 1 / 0])
        runner.tick(0.0)
        runner.tick(1.0)
        assert len(calls) == 2

    def test_thread_starts_and_stops_cleanly(self):
        s = FakeSensor()
        runner = SensorRunner([s], ReactionBus(), jitter=0.0)
        runner.start()
        runner.stop(timeout=2.0)
        assert s.closed
        assert runner._thread is None


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_debouncer_needs_consecutive_failures(self):
        d = Debouncer(fail_after=2)
        assert d.update("k", False) is None  # baseline up
        assert d.update("k", True) is None  # one blip
        assert d.update("k", False) is None
        assert d.update("k", True) is None
        assert d.update("k", True) == "down"
        assert d.update("k", True) is None  # already down: no repeat
        assert d.update("k", False) == "up"

    def test_debouncer_silent_baseline(self):
        d = Debouncer(fail_after=1, alert_initial=False)
        assert d.update("k", True) is None
        # Starting for the first time is not "coming back".
        assert d.update("k", False) is None
        assert d.update("k", True) == "down"
        assert d.update("k", False) == "up"

    def test_levels_hysteresis(self):
        lv = Levels(lambda v: 1 if v < 10 else 0, lambda v: 1 if v < 12 else 0)
        assert lv.update("d", 50) == (None, 0)
        assert lv.update("d", 9) == (0, 1)
        assert lv.update("d", 11) == (1, 1)  # inside the margin: still low
        assert lv.update("d", 9.5) == (1, 1)
        assert lv.update("d", 13) == (1, 0)

    def test_human_bytes_uses_decimal_comma(self):
        assert human_bytes(12_345_000_000) == "12,3 GB"
        assert human_bytes(120_000_000_000) == "120 GB"
        assert human_bytes(512) == "512 B"


# --------------------------------------------------------------------------- #
# web
# --------------------------------------------------------------------------- #


class FakeFetch:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, url, timeout=10.0, method="GET"):
        self.calls.append((url, method))
        item = self.script.pop(0)
        if isinstance(item, net.Response):
            return item
        return net.Response(item, 42, b"", "" if item else "Connection refused")


class TestWeb:
    URL = "https://example.com/health"

    def test_down_after_n_failures_then_up_once(self):
        fetch = FakeFetch([200, None, 503, 503, 200, 200])
        s = WebSensor({"urls": [self.URL], "fail_after": 2}, fetch=fetch)
        seen = [kinds(s.poll()) for _ in range(6)]
        assert seen == [[], [], ["web.down"], [], ["web.up"], []]

    def test_down_event_carries_details_and_severity(self):
        s = WebSensor({"urls": [self.URL], "fail_after": 1}, fetch=FakeFetch([None]))
        (ev,) = s.poll()
        assert ev.severity == "urgent"
        assert ev.key == f"web:{self.URL}"
        assert ev.data["host"] == "example.com"
        assert ev.data["error"] == "Connection refused"
        assert ev.data["latency_ms"] == 42

    def test_4xx_is_not_an_outage(self):
        s = WebSensor({"urls": [self.URL], "fail_after": 1}, fetch=FakeFetch([404, 404]))
        assert s.poll() == [] and s.poll() == []

    def test_down_at_startup_is_reported(self):
        s = WebSensor({"urls": [self.URL], "fail_after": 2}, fetch=FakeFetch([500, 500]))
        s.poll()
        assert kinds(s.poll()) == ["web.down"]

    def test_non_http_urls_are_ignored(self):
        s = WebSensor({"urls": ["ftp://x", "example.com", {"url": self.URL}]}, fetch=FakeFetch([]))
        assert s.urls == [self.URL]

    def test_summary(self):
        s = WebSensor({"urls": [self.URL, "https://example.org"], "fail_after": 1},
                      fetch=FakeFetch([200, 200, None, 200]))
        assert s.summary() == "todavía no revisé tus webs"
        s.poll()
        assert s.summary() == "todas tus webs andan"
        s.poll()
        assert s.summary() == "está caída example.com"


# --------------------------------------------------------------------------- #
# system
# --------------------------------------------------------------------------- #

Usage = namedtuple("Usage", "total used free")
GB = 1_000_000_000


class FakeDisk:
    def __init__(self, frees, total=500 * GB):
        self.frees = list(frees)
        self.total = total

    def __call__(self, path):
        return Usage(self.total, 0, self.frees.pop(0))


class TestSystem:
    def make(self, frees=(), batt=(), mem=(), therm=(), load=(0.1, 0, 0), total=500 * GB, **cfg):
        batt, mem, therm = list(batt), list(mem), list(therm)

        def run(cmd):
            if cmd[:3] == ["pmset", "-g", "batt"]:
                return batt.pop(0) if batt else ""
            if cmd[0] == "memory_pressure":
                return mem.pop(0) if mem else ""
            if cmd[:3] == ["pmset", "-g", "therm"]:
                return therm.pop(0) if therm else ""
            return ""

        return SystemSensor(cfg, disk_usage=FakeDisk(frees, total), run=run,
                            loadavg=lambda: load, cpu_count=4)

    def test_disk_transitions_precisely(self):
        total = 500 * GB
        pct = lambda p: int(total * p / 100)  # noqa: E731
        s = self.make(frees=[pct(50), pct(9), pct(11), pct(2), pct(4), pct(6), pct(13)],
                      battery=False)
        events = [s.poll() for _ in range(7)]
        assert [kinds(e) for e in events] == [
            [], ["disk.low"], [], ["disk.critical"], [], ["disk.ok"], ["disk.ok"],
        ]
        low = events[1][0]
        assert low.severity == "warning" and low.data["volume"] == "/"
        assert low.data["free"] == "45,0 GB"
        assert events[3][0].severity == "urgent"

    def test_disk_low_by_absolute_gigabytes(self):
        # 8 GB of 50 GB is 16 % free: fine by percentage, low by size.
        s = self.make(frees=[40 * GB, 8 * GB], total=50 * GB, battery=False)
        assert s.poll() == []
        assert kinds(s.poll()) == ["disk.low"]

    def test_missing_volume_is_skipped(self):
        def boom(path):
            raise FileNotFoundError(path)

        s = SystemSensor({"volumes": ["/Volumes/Gone"], "battery": False}, disk_usage=boom)
        assert s.poll() == []

    def test_battery_low_critical_and_charging(self):
        dis = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t{}%; discharging; 1:00 remaining"
        ac = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t{}%; charging; 1:00 remaining"
        s = self.make(frees=[100 * GB] * 6, batt=[dis.format(50), dis.format(19), dis.format(18),
                                                   dis.format(6), ac.format(6), dis.format(19)])
        seen = [kinds(s.poll()) for _ in range(6)]
        assert seen == [[], ["battery.low"], [], ["battery.critical"], ["battery.charging"], ["battery.low"]]

    def test_desktop_without_battery(self):
        s = self.make(frees=[100 * GB], batt=["Now drawing from 'AC Power'\n"])
        assert s.poll() == []

    def test_memory_and_thermal(self):
        s = self.make(frees=[100 * GB] * 3, battery=False, memory=True, cpu=True,
                      mem=["System-wide memory free percentage: 40%",
                           "System-wide memory free percentage: 5%",
                           "System-wide memory free percentage: 30%"],
                      therm=["", "CPU_Speed_Limit = 70", ""])
        seen = [kinds(s.poll()) for _ in range(3)]
        assert seen == [[], ["memory.high", "cpu.hot"], ["memory.ok", "cpu.cool"]]

    def test_cpu_busy_needs_sustained_load(self):
        s = self.make(frees=[100 * GB] * 4, battery=False, cpu=True, load=(8.0, 0, 0), cpu_busy_polls=3)
        seen = [kinds(s.poll()) for _ in range(4)]
        assert seen == [[], [], ["cpu.busy"], []]

    def test_parsers(self):
        assert parse_pmset_batt("x\t54%; discharging; 3:10 remaining") == (54, False)
        assert parse_pmset_batt("Now drawing from 'AC Power'\n x\t100%; charged; 0:00") == (100, True)
        assert parse_pmset_batt("") is None
        assert parse_memory_pressure("System-wide memory free percentage: 32%") == 32
        assert parse_memory_pressure("garbage") is None
        assert parse_thermal("Note: No thermal warning level has been recorded") is False
        assert parse_thermal("CPU_Speed_Limit = 100") is False
        assert parse_thermal("Thermal warning level: 2") is True

    def test_summary(self):
        s = self.make(frees=[120 * GB], batt=["x\t54%; discharging; 3:10 remaining"])
        s.poll()
        assert s.summary() == "el disco tiene 120 GB libres; la batería está al 54%"


# --------------------------------------------------------------------------- #
# services
# --------------------------------------------------------------------------- #


class TestServices:
    def test_ports_and_processes_baseline_is_silent(self):
        ports = iter([False, True, False, False, True])
        procs = iter([{"postgres"}, {"postgres"}, set(), set(), {"postgres"}])
        s = ServicesSensor({"ports": ["localhost:3000"], "processes": ["postgres"], "fail_after": 2},
                           processes=lambda: next(procs), port_check=lambda h, p: next(ports))
        seen = [kinds(s.poll()) for _ in range(5)]
        # One failed check is not enough (fail_after: 2), and the first state
        # is a silent baseline; only the later sustained outage is news.
        assert seen == [[], [], [], ["service.down", "service.down"], ["service.up", "service.up"]]

    def test_event_payload(self):
        up = iter([True, False])
        s = ServicesSensor({"ports": [5432], "fail_after": 1}, port_check=lambda h, p: next(up))
        s.poll()
        (ev,) = s.poll()
        assert ev.kind == "service.down" and ev.severity == "warning"
        assert ev.data["name"] == "puerto 5432" and ev.data["kind_es"] == "puerto"
        assert ev.key == "service:port:puerto 5432"

    def test_docker_all_watches_what_was_running(self):
        states = iter([
            {"web": "running", "old": "exited"},
            {"web": "running", "old": "exited"},
            {"web": "exited", "old": "exited"},
            {"old": "exited"},
            {"web": "running", "old": "running"},
        ])
        s = ServicesSensor({"docker": True, "fail_after": 1}, docker=lambda: next(states))
        seen = [[(e.kind, e.data["name"]) for e in s.poll()] for _ in range(5)]
        assert seen == [[], [], [("service.down", "web")], [], [("service.up", "web")]]

    def test_docker_unavailable_is_not_an_outage(self):
        s = ServicesSensor({"docker": ["db"], "fail_after": 1}, docker=lambda: None)
        assert s.poll() == [] and s.poll() == []

    def test_parse_docker_ps(self):
        text = "\n".join([
            json.dumps({"Names": "web", "State": "running", "Status": "Up 2 hours"}),
            json.dumps({"Names": "db", "Status": "Exited (0) 3 minutes ago"}),
            "not json",
            "",
        ])
        assert parse_docker_ps(text) == {"web": "running", "db": "exited"}

    def test_parse_port(self):
        assert parse_port(3000) == ("127.0.0.1", 3000)
        assert parse_port("localhost:8080") == ("localhost", 8080)
        assert parse_port({"host": "10.0.0.2", "port": 22}) == ("10.0.0.2", 22)
        assert parse_port("nope") is None

    def test_docker_ps_uses_subprocess_and_survives_missing_cli(self, monkeypatch):
        from wawi.sensors import services

        monkeypatch.setattr(services.shutil, "which", lambda name: None)
        assert services.docker_ps() is None

        class Done:
            returncode = 0
            stdout = json.dumps({"Names": "api", "State": "running"})

        monkeypatch.setattr(services.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(services.subprocess, "run", lambda *a, **k: Done())
        assert services.docker_ps() == {"api": "running"}

    def test_port_open_uses_socket(self, monkeypatch):
        from wawi.sensors import services

        def refuse(addr, timeout):
            raise ConnectionRefusedError

        monkeypatch.setattr(services.socket, "create_connection", refuse)
        assert services.port_open("127.0.0.1", 1) is False


# --------------------------------------------------------------------------- #
# weather
# --------------------------------------------------------------------------- #


def load_forecast():
    return json.loads((FIXTURES / "open_meteo.json").read_text())


class TestWeather:
    def test_parse_fixture(self):
        f = parse_forecast(load_forecast(), hours=3, threshold=60)
        assert f["now"] == datetime(2026, 10, 15, 9, 15)
        assert f["rain_at"] == "12:00" and f["rain_prob"] == 70
        assert f["today"]["max"] == 31.2
        assert f["tomorrow"]["rain"] == 75

    def test_rain_outside_window_is_ignored(self):
        f = parse_forecast(load_forecast(), hours=2, threshold=60)
        assert f["rain_at"] is None

    def test_events_once_per_day(self):
        payload = load_forecast()
        s = WeatherSensor({"lat": -34.6, "lon": -58.38}, fetch_json=lambda url: payload)
        s.setup()
        first = s.poll()
        assert kinds(first) == ["weather.rain_soon", "weather.hot"]
        assert first[0].data["hour"] == "12:00"
        assert first[1].data["max"] == 31
        assert s.poll() == []  # same day, same rain spell: nothing new
        assert s.summary() == "afuera hay 18 °C, va a llover a las 12:00, mañana llueve"

    def test_rain_rearms_after_a_dry_window(self):
        wet = load_forecast()
        wet["current"]["weather_code"] = 61  # rainy sky: keep "sunny" out of it
        dry = json.loads(json.dumps(wet))
        dry["hourly"]["precipitation_probability"] = [0] * 8
        seq = iter([wet, dry, wet])
        s = WeatherSensor({"lat": 1, "lon": 2, "hot_c": 99}, fetch_json=lambda url: next(seq))
        s.setup()
        assert kinds(s.poll()) == ["weather.rain_soon"]
        assert s.poll() == []
        assert kinds(s.poll()) == ["weather.rain_soon"]

    def test_sunny_and_cold_wait_for_morning(self):
        payload = load_forecast()
        payload["hourly"]["precipitation_probability"] = [0] * 8
        payload["daily"]["temperature_2m_min"] = [2.0, 3.0]
        payload["current"]["time"] = "2026-10-15T06:00"
        s = WeatherSensor({"lat": 1, "lon": 2, "morning_hour": 8}, fetch_json=lambda url: payload)
        s.setup()
        assert s.poll() == []
        payload["current"]["time"] = "2026-10-15T08:30"
        assert kinds(s.poll()) == ["weather.hot", "weather.cold", "weather.sunny"]

    def test_city_is_geocoded_once_in_setup(self):
        urls = []

        def fetch(url):
            urls.append(url)
            if "geocoding" in url:
                return {"results": [{"latitude": -34.6, "longitude": -58.4}]}
            return load_forecast()

        s = WeatherSensor({"city": "Buenos Aires"}, fetch_json=fetch)
        s.setup()
        assert (s.lat, s.lon) == (-34.6, -58.4)
        s.poll()
        assert "latitude=-34.6000" in urls[-1]

    def test_unknown_city_fails_setup(self):
        s = WeatherSensor({"city": "Nowhere"}, fetch_json=lambda url: {"results": []})
        with pytest.raises(ValueError):
            s.setup()


# --------------------------------------------------------------------------- #
# rss
# --------------------------------------------------------------------------- #


class TestRss:
    def test_parse_rss_and_atom(self):
        title, items = parse_feed((FIXTURES / "feed_rss.xml").read_bytes())
        assert title == "Noticias de ejemplo"
        assert items[0] == {"id": "https://example.com/1", "title": "Primera noticia",
                            "link": "https://example.com/1"}
        assert items[1]["id"] == "https://example.com/2"  # no guid: falls back to link
        title, items = parse_feed((FIXTURES / "feed_atom.xml").read_bytes())
        assert title == "Blog de ejemplo"
        assert [i["link"] for i in items] == ["https://example.org/a", "https://example.org/b"]

    def test_first_read_marks_everything_seen(self):
        old = (FIXTURES / "feed_rss.xml").read_bytes()
        new = old.replace(b"<item>", b"<item><title>Nueva</title><guid>n1</guid></item><item>", 1)
        bodies = iter([old, old, new])
        fetch = lambda url, timeout, method: net.Response(200, 5, next(bodies))  # noqa: E731
        s = RssSensor({"feeds": [{"url": "https://example.com/feed", "name": "Ejemplo"}]}, fetch=fetch)
        assert s.poll() == []
        assert s.poll() == []
        (ev,) = s.poll()
        assert ev.kind == "rss.item" and ev.severity == "info"
        assert ev.data == {"title": "Nueva", "link": "", "feed": "Ejemplo"}
        assert s.summary() == "llegó 1 noticia nueva"

    def test_all_feeds_failing_raises_for_backoff(self):
        s = RssSensor({"feeds": ["https://example.com/feed"]},
                      fetch=lambda url, timeout, method: net.Response(None, 5, b"", "offline"))
        with pytest.raises(OSError):
            s.poll()

    def test_broken_xml_counts_as_failure(self):
        s = RssSensor({"feeds": ["https://example.com/a", "https://example.com/b"]},
                      fetch=lambda url, timeout, method: net.Response(
                          200, 5, b"<rss><channel>" if url.endswith("a") else (FIXTURES / "feed_rss.xml").read_bytes()))
        assert s.poll() == []  # one feed broken, the other fine: no raise


# --------------------------------------------------------------------------- #
# clock
# --------------------------------------------------------------------------- #


class FakeTime:
    def __init__(self, start):
        self.now = start
        self.idle = 0.0

    def advance(self, minutes, idle=None):
        self.now += timedelta(minutes=minutes)
        if idle is not None:
            self.idle = idle


class TestClock:
    def make(self, start, **cfg):
        t = FakeTime(start)
        s = ClockSensor(cfg, now=lambda: t.now, idle=lambda: t.idle)
        return s, t

    def test_daypart_fires_once_on_crossing(self):
        s, t = self.make(datetime(2026, 10, 15, 7, 58), dayparts={"morning": "08:00"})
        assert s.poll() == []
        t.advance(3)
        (ev,) = s.poll()
        assert ev.kind == "clock.morning" and ev.data["hour"] == "08:00"
        t.advance(1)
        assert s.poll() == []

    def test_daypart_past_at_startup_is_not_replayed(self):
        s, t = self.make(datetime(2026, 10, 15, 9, 0), dayparts={"morning": "08:00"})
        assert s.poll() == []
        t.advance(1)
        assert s.poll() == []

    def test_daypart_waits_for_the_user_within_grace(self):
        s, t = self.make(datetime(2026, 10, 15, 7, 59), dayparts={"morning": "08:00"}, grace_min=60)
        s.poll()
        t.advance(2, idle=1800)  # away
        assert s.poll() == []
        t.advance(20, idle=3)  # back
        assert kinds(s.poll()) == ["clock.morning"]

    def test_daypart_dropped_after_grace(self):
        s, t = self.make(datetime(2026, 10, 15, 7, 59), dayparts={"morning": "08:00"}, grace_min=30)
        s.poll()
        t.advance(2, idle=3600)
        s.poll()
        t.advance(60, idle=1)
        assert s.poll() == []

    def test_late_night_crosses_midnight(self):
        s, t = self.make(datetime(2026, 10, 15, 23, 59), dayparts={"late_night": "00:30"})
        s.poll()
        t.advance(32)
        assert kinds(s.poll()) == ["clock.late_night"]

    def test_long_session_suggests_a_break_and_resets_after_one(self):
        s, t = self.make(datetime(2026, 10, 15, 9, 0), dayparts={}, long_session_h=2,
                         break_idle_min=5, remind_every_min=60)
        s.poll()
        t.advance(119, idle=10)
        assert s.poll() == []
        t.advance(2, idle=10)
        (ev,) = s.poll()
        assert ev.kind == "break.suggest" and ev.data["duration"] == "2 h 1 min"
        assert s.summary() == "llevás 2 h 1 min sin pausa"
        t.advance(30, idle=10)
        assert s.poll() == []  # not again so soon
        t.advance(31, idle=10)
        assert kinds(s.poll()) == ["break.suggest"]  # reminder after an hour
        t.advance(10, idle=600)  # a real break
        assert s.poll() == []
        t.advance(10, idle=1)
        assert s.poll() == [] and s.summary() is None

    def test_no_idle_source_disables_long_session(self):
        t = FakeTime(datetime(2026, 10, 15, 9, 0))
        s = ClockSensor({"dayparts": {}}, now=lambda: t.now, idle=lambda: None)
        s.poll()
        t.advance(600)
        assert s.poll() == []


# --------------------------------------------------------------------------- #
# Registry, plugins, status
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_everything_is_opt_in(self):
        assert sensors_pkg.build_sensors({}) == []
        assert sensors_pkg.build_sensors({"web": {"urls": ["https://example.com"]}}) == []

    def test_enabled_sections_build(self):
        built = sensors_pkg.build_sensors({
            "web": {"enabled": True, "urls": ["https://example.com"], "interval": 30},
            "clock": True,
            "rss": {"enabled": False},
        })
        assert [s.name for s in built] == ["web", "clock"]
        assert built[0].interval == 30

    def test_master_switch(self):
        assert sensors_pkg.build_sensors({"enabled": False, "clock": True}) == []

    def test_interval_floor(self):
        (s,) = sensors_pkg.build_sensors({"clock": {"enabled": True, "interval": 0}})
        assert s.interval == ClockSensor.min_interval


PLUGIN = '''
from wawi.reactions import Event
from wawi.sensors.base import Sensor

class Hello(Sensor):
    name = "hello"
    def poll(self):
        return [Event("hello.ping", "info", "hello", self.config.get("greeting", "hola"))]

SENSOR = Hello
'''

FACTORY_PLUGIN = '''
from wawi.sensors.base import Sensor

class Other(Sensor):
    name = "other"
    def poll(self):
        return []

def make_sensor(config):
    s = Other(config)
    s.name = config.get("rename", "other")
    return s
'''


class TestUserPlugins:
    def test_loads_class_and_factory_plugins(self, tmp_path):
        (tmp_path / "hello.py").write_text(PLUGIN)
        (tmp_path / "factory.py").write_text(FACTORY_PLUGIN)
        (tmp_path / "_private.py").write_text("raise SystemExit")
        (tmp_path / "broken.py").write_text("this is not python")
        (tmp_path / "empty.py").write_text("X = 1")
        for p in tmp_path.iterdir():
            p.chmod(0o644)
        built = sensors_pkg.build_sensors({
            "user_plugins": True,
            "plugins_dir": str(tmp_path),
            "plugins": {"hello": {"greeting": "buenas"}, "factory": {"rename": "renamed"}},
        })
        names = sorted(s.name for s in built)
        assert names == ["hello", "renamed"]
        hello = next(s for s in built if s.name == "hello")
        assert hello.poll()[0].title == "buenas"

    def test_plugins_are_off_unless_asked(self, tmp_path):
        (tmp_path / "hello.py").write_text(PLUGIN)
        assert sensors_pkg.build_sensors({"plugins_dir": str(tmp_path)}) == []

    def test_disabled_and_unsafe_plugins_are_skipped(self, tmp_path):
        (tmp_path / "hello.py").write_text(PLUGIN)
        (tmp_path / "loose.py").write_text(PLUGIN)
        (tmp_path / "hello.py").chmod(0o644)
        (tmp_path / "loose.py").chmod(0o666)
        built = sensors_pkg.load_user_plugins(tmp_path, {"hello": {"enabled": False}})
        assert built == []

    def test_missing_folder(self, tmp_path):
        assert sensors_pkg.load_user_plugins(tmp_path / "nope") == []


class TestStatusAndStart:
    def test_status_without_runner(self):
        assert "apagados" in sensors_pkg.status()

    def test_status_joins_summaries(self):
        web = WebSensor({"urls": ["https://example.com"]}, fetch=FakeFetch([200]))
        web.poll()
        system = SystemSensor({"battery": False}, disk_usage=FakeDisk([120 * GB]))
        system.poll()
        runner = SensorRunner([web, system], ReactionBus())
        assert sensors_pkg.status(runner) == "Example.com anda; el disco tiene 120 GB libres."

    def test_status_with_nothing_yet(self):
        runner = SensorRunner([ClockSensor({"dayparts": {}}, idle=lambda: None)], ReactionBus())
        assert sensors_pkg.status(runner).startswith("Todavía no tengo novedades")

    def test_start_is_noop_when_switched_off(self):
        class Cfg:
            sensors = {"enabled": False, "clock": True}
            reflexes = {}

        assert sensors_pkg.start(Cfg(), ReactionBus()) is None

    def test_start_and_stop(self):
        class Cfg:
            sensors = {"clock": {"enabled": True, "dayparts": {}}}
            reflexes = {}

        bus = ReactionBus()
        runner = sensors_pkg.start(Cfg(), bus)
        try:
            assert runner is not None
            assert sensors_pkg.start(Cfg(), bus) is runner  # idempotent
            assert sensors_pkg._REFLEXES is not None
        finally:
            sensors_pkg.stop()
        assert sensors_pkg._RUNNER is None


class TestBrainHookAndConfig:
    def test_sensor_tool_only_offered_while_sensors_run(self):
        from wawi import tools

        assert "sensor_status" not in {t["name"] for t in tools.active_tools()}
        tools.set_sensors_active(True)
        try:
            assert "sensor_status" in {t["name"] for t in tools.active_tools()}
            result = tools.dispatch("sensor_status", {})
            assert result.ok and "apagados" in result.speak
        finally:
            tools.set_sensors_active(False)

    def test_config_sections_are_read_raw(self, tmp_path):
        from wawi import config as config_mod

        path = tmp_path / "config.yaml"
        path.write_text(
            "sensors:\n  web:\n    enabled: true\n    urls: [https://example.com]\n"
            "reflexes:\n  quiet_hours: '23:00-08:00'\n"
        )
        cfg = config_mod.load(path)
        assert cfg.sensors["web"]["urls"] == ["https://example.com"]
        assert cfg.reflexes["quiet_hours"] == "23:00-08:00"
        path.write_text("language: es\n")
        assert config_mod.load(path).sensors == {}
        path.write_text("sensors: [1, 2]\n")
        with pytest.raises(config_mod.ConfigError):
            config_mod.load(path)

    def test_example_config_parses_and_builds_nothing_by_default(self):
        import yaml

        example = yaml.safe_load((Path(__file__).parent.parent / "config.yaml.example").read_text())
        assert sensors_pkg.build_sensors(example.get("sensors") or {}) == []
        assert isinstance(example.get("reflexes") or {}, dict)
