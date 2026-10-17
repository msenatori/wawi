"""Reflexes: rule mapping, safe templates and the interruption budget.

The budget tests matter most: a pet that barks at every headline during a
meeting is a pet that gets uninstalled.
"""

from __future__ import annotations

from datetime import datetime, time as dtime

import pytest

from wawi import reflexes
from wawi.reactions import Event, ReactionBus
from wawi.reflexes import (
    Budget,
    QuietDetector,
    QuietState,
    Reflexes,
    build_rules,
    in_spans,
    parse_quiet_hours,
    render,
)


class FixedQuiet:
    def __init__(self, state=QuietState()):
        self.state = state

    def check(self):
        return self.state


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def drain(bus):
    out = []
    while (r := bus.pop()) is not None:
        out.append(r)
    return out


def make(cfg=None, quiet=QuietState()):
    bus = ReactionBus()
    clock = Clock()
    q = FixedQuiet(quiet)
    rx = Reflexes(cfg or {}, bus=bus, clock=clock, quiet=q)
    rx.attach()
    return rx, bus, clock, q


URL = "https://example.com/"
DOWN = Event("web.down", "urgent", f"web:{URL}", "x", {"url": URL, "host": "example.com"})
UP = Event("web.up", "notice", f"web:{URL}", "x", {"url": URL, "host": "example.com"})


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #


class TestRender:
    def test_fills_fields(self):
        assert render("¡Se cayó {host}!", {"host": "example.com"}) == "¡Se cayó example.com!"

    def test_missing_field_is_empty_not_a_crash(self):
        assert render("Quedan {free} libres", {}) == "Quedan libres"

    def test_attribute_and_index_access_is_refused(self):
        out = render("{host.__class__} {host[0]} ok", {"host": "x"})
        assert out == "ok"

    def test_positional_and_malformed_templates(self):
        assert render("a {} b", {}) == "a b"
        assert render("a {unclosed", {}) == "a {unclosed"  # kept as written, no crash
        assert render("a }", {}) == "a }"
        assert render("{x:.1f} GB", {"x": "no es número"}) == "no es número GB"
        assert render("{x:.1f} GB", {"x": 2.345}) == "2.3 GB"

    def test_none_and_empty(self):
        assert render(None, {}) is None
        assert render("{missing}", {}) is None

    def test_long_bubbles_are_clipped(self):
        out = render("{t}", {"t": "a" * 500})
        assert len(out) == reflexes.MAX_BUBBLE and out.endswith("…")


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


class TestRules:
    def test_web_down_maps_to_firefighter_run(self):
        rx, bus, *_ = make()
        rx.on_event(DOWN)
        (r,) = drain(bus)
        assert (r.behaviour, r.costume, r.sound) == ("run", "firefighter", "bark")
        assert r.bubble == "¡Se cayó example.com!"
        assert r.priority == 3
        assert r.until_key == f"web:{URL}"
        assert r.source is DOWN
        assert r.speak is False

    def test_web_up_cancels_and_celebrates(self):
        rx, bus, *_ = make()
        rx.on_event(UP)
        (r,) = drain(bus)
        assert r.cancel_key == f"web:{URL}"
        assert r.behaviour == "celebrate" and r.bubble == "example.com volvió"

    @pytest.mark.parametrize("kind,data,expect", [
        ("disk.low", {"free": "8 GB", "volume": "/"},
         ("dig", None, None, "Me estoy quedando sin lugar para mis huesos (8 GB libres)")),
        ("disk.critical", {"free": "1 GB", "volume": "/"},
         ("alarm", "alarm", "howl", "¡El disco está casi lleno! Quedan 1 GB")),
        ("battery.low", {"percent": 15}, ("sad", None, "whine", "Me estoy quedando sin pilas (15%)")),
        ("service.down", {"name": "postgres"}, ("detective", "detective", "bark", "Se cayó postgres")),
        ("service.up", {"name": "postgres"}, ("celebrate", None, None, "postgres volvió")),
        ("weather.rain_soon", {"hour": "17:00"}, (None, "umbrella", None, "Va a llover a las 17:00")),
        ("weather.hot", {"max": 33}, (None, "sunglasses", None, "Hoy hace calor: 33 °C")),
        ("weather.cold", {"min": 2}, (None, "scarf", None, "Abrigate: hoy hace 2 °C")),
        ("break.suggest", {"duration": "2 horas"},
         ("ball_bring", None, None, "¿Una pausa? Llevás 2 horas sin parar")),
    ])
    def test_default_mapping(self, kind, data, expect):
        rx, bus, *_ = make({"budget": {"info": None, "notice": None, "warning": None}})
        rx.on_event(Event(kind, "urgent" if kind == "disk.critical" else "warning", "k", "", data))
        (r,) = drain(bus)
        assert (r.behaviour, r.costume, r.sound, r.bubble) == expect

    def test_service_until_key_uses_event_key(self):
        rx, bus, *_ = make()
        rx.on_event(Event("service.down", "warning", "service:port:puerto 3000", "", {"name": "puerto 3000"}))
        (r,) = drain(bus)
        assert r.until_key == "service:port:puerto 3000"

    def test_cancel_only_rule(self):
        rx, bus, *_ = make()
        rx.on_event(Event("disk.ok", "info", "disk:/", "", {"volume": "/"}))
        (r,) = drain(bus)
        assert r.cancel_key == "disk:/" and r.behaviour is None and r.bubble is None

    def test_unknown_kinds_are_left_to_the_mind(self):
        rx, bus, *_ = make()
        rx.on_event(Event("something.new", "urgent"))
        assert drain(bus) == []

    def test_config_overrides_merge_and_disable(self):
        rules = build_rules({
            "rules": {"web.down": {"sound": None, "bogus": 1}, "rss.item": False},
            "weather.hot": {"costume": "hat"},
            "custom.thing": {"behaviour": "spaceship", "bubble": "{title}"},
        })
        assert rules["web.down"]["sound"] is None
        assert rules["web.down"]["costume"] == "firefighter"
        assert "bogus" not in rules["web.down"]
        assert "rss.item" not in rules
        assert rules["weather.hot"]["costume"] == "hat"
        assert rules["custom.thing"]["behaviour"] == "spaceship"

    def test_custom_rule_end_to_end_with_bad_numbers(self):
        rx, bus, *_ = make({"rules": {"x.y": {"behaviour": "spaceship", "bubble": "{title}",
                                              "priority": "high", "duration": "long"}}})
        rx.on_event(Event("x.y", "warning", "", "¡Despegue!"))
        (r,) = drain(bus)
        assert r.behaviour == "spaceship" and r.bubble == "¡Despegue!"
        assert r.priority == 2 and r.duration == 0.0

    def test_names_exported_for_the_body(self):
        assert "firefighter" in reflexes.COSTUMES and "umbrella" in reflexes.COSTUMES
        assert set(reflexes.SOUNDS) == {"bark", "howl", "whine"}
        assert "newspaper" in reflexes.BEHAVIOURS

    def test_detach_mutes(self):
        rx, bus, *_ = make()
        rx.detach()
        bus.emit(DOWN)
        assert drain(bus) == []

    def test_attach_subscribes_once(self):
        rx, bus, *_ = make()
        rx.attach()
        bus.emit(DOWN)
        assert len(drain(bus)) == 1


# --------------------------------------------------------------------------- #
# Budget, dedupe, batching, quiet mode
# --------------------------------------------------------------------------- #


class TestBudget:
    def test_sliding_window(self):
        b = Budget({"notice": 2})
        assert b.allow("notice", 0)
        b.spend("notice", 0)
        b.spend("notice", 10)
        assert not b.allow("notice", 100)
        assert b.allow("notice", 3600)

    def test_unlimited_and_zero(self):
        b = Budget({"urgent": None, "info": 0})
        for t in range(100):
            b.spend("urgent", t)
        assert b.allow("urgent", 100)
        assert not b.allow("info", 0)

    def test_over_budget_reactions_are_held_back(self):
        rx, bus, clock, _ = make({"budget": {"warning": 2}, "dedupe_minutes": 0})
        for i in range(4):
            rx.on_event(Event("service.down", "warning", f"service:{i}", "", {"name": str(i)}))
        assert len(drain(bus)) == 2
        clock.t += 3600
        rx.on_event(Event("service.down", "warning", "service:9", "", {"name": "9"}))
        assert len(drain(bus)) == 1

    def test_cancel_still_delivered_when_over_budget(self):
        rx, bus, *_ = make({"budget": {"notice": 0}})
        rx.on_event(UP)
        (r,) = drain(bus)
        assert r.cancel_key == f"web:{URL}"
        assert r.behaviour is None and r.bubble is None  # silent all-clear

    def test_duplicate_events_are_one_interruption(self):
        rx, bus, clock, _ = make()
        rx.on_event(DOWN)
        rx.on_event(DOWN)
        assert len(drain(bus)) == 1
        rx.on_event(UP)
        rx.on_event(DOWN)  # a real flap: down -> up -> down is news each time
        assert len(drain(bus)) == 2
        clock.t += 3600
        rx.on_event(DOWN)
        assert len(drain(bus)) == 1


def rss(n):
    return Event("rss.item", "info", f"rss:{n}", f"Titular {n}", {"title": f"Titular {n}", "feed": "f"})


class TestBatching:
    def test_single_item_after_gather(self):
        rx, bus, clock, _ = make()
        rx.on_event(rss(1))
        assert drain(bus) == []  # gathering the burst
        clock.t += 61
        rx.tick()
        (r,) = drain(bus)
        assert r.behaviour == "newspaper" and r.bubble == "Titular 1"

    def test_burst_collapses_and_rate_is_limited(self):
        rx, bus, clock, _ = make({"batch_minutes": 20})
        for i in range(4):
            rx.on_event(rss(i))
        clock.t += 61
        rx.tick()
        (r,) = drain(bus)
        assert r.bubble == "Tenés 4 noticias nuevas"
        rx.on_event(rss(10))
        rx.on_event(rss(11))
        clock.t += 120
        rx.tick()
        assert drain(bus) == []  # within the 20-minute window
        clock.t += 20 * 60
        rx.tick()
        (r,) = drain(bus)
        assert r.bubble == "Tenés 2 noticias nuevas"

    def test_batches_wait_out_quiet_mode(self):
        rx, bus, clock, q = make()
        q.state = QuietState(True, False, "fullscreen")
        rx.on_event(rss(1))
        clock.t += 300
        rx.tick()
        assert drain(bus) == []
        q.state = QuietState()
        rx.tick()
        assert drain(bus)[0].bubble == "Titular 1"

    def test_tick_is_inert_when_detached(self):
        rx, bus, clock, _ = make()
        rx.on_event(rss(1))
        rx.detach()
        clock.t += 100
        rx.tick()
        assert drain(bus) == []


class TestQuietMode:
    def test_quiet_holds_non_urgent(self):
        rx, bus, *_ = make(quiet=QuietState(True, False, "quiet_hours"))
        rx.on_event(Event("disk.low", "warning", "disk:/", "", {"free": "1 GB"}))
        assert drain(bus) == []

    def test_urgent_breaks_quiet_but_never_speaks(self):
        rx, bus, *_ = make({"rules": {"web.down": {"speak": True}}},
                           quiet=QuietState(True, False, "fullscreen"))
        rx.on_event(DOWN)
        (r,) = drain(bus)
        assert r.costume == "firefighter" and r.sound == "bark" and r.speak is False

    def test_urgent_in_a_meeting_is_silent(self):
        rx, bus, *_ = make(quiet=QuietState(True, True, "meeting"))
        rx.on_event(DOWN)
        (r,) = drain(bus)
        assert r.costume == "firefighter" and r.bubble and r.sound is None

    def test_speak_allowed_when_not_quiet(self):
        rx, bus, *_ = make({"rules": {"web.down": {"speak": True}}})
        rx.on_event(DOWN)
        assert drain(bus)[0].speak is True


class TestQuietDetector:
    def make(self, cfg=None, front=None, procs=(), now=datetime(2026, 10, 15, 12, 0)):
        t = [0.0]
        det = QuietDetector(cfg or {}, now=lambda: now, frontmost=lambda: front,
                            processes=lambda: set(procs), clock=lambda: t[0])
        return det, t

    def test_nothing_quiet(self):
        det, _ = self.make(front=("Safari", False))
        assert det.check() == QuietState()

    def test_fullscreen(self):
        det, _ = self.make(front=("VLC", True))
        assert det.check().reason == "fullscreen"

    def test_fullscreen_can_be_disabled(self):
        det, _ = self.make({"quiet_fullscreen": False}, front=("VLC", True))
        assert det.check().quiet is False

    def test_meeting_by_frontmost_app_or_process(self):
        det, _ = self.make(front=("zoom.us", False))
        assert det.check() == QuietState(True, True, "meeting")
        det, _ = self.make(front=("Safari", False), procs={"CptHost"})
        assert det.check().meeting is True
        det, _ = self.make({"quiet_meetings": False}, front=("zoom.us", False), procs={"CptHost"})
        assert det.check().meeting is False

    def test_quiet_hours(self):
        det, _ = self.make({"quiet_hours": "23:00-08:00"}, now=datetime(2026, 10, 15, 2, 0))
        assert det.check().reason == "quiet_hours"
        det, _ = self.make({"quiet_hours": ["23:00-08:00"]}, now=datetime(2026, 10, 15, 12, 0))
        assert det.check().quiet is False

    def test_result_is_cached(self):
        calls = []

        def front():
            calls.append(1)
            return ("Safari", False)

        t = [0.0]
        det = QuietDetector({}, frontmost=front, processes=set, clock=lambda: t[0], cache_s=15)
        det.check()
        det.check()
        t[0] = 16
        det.check()
        assert len(calls) == 2

    def test_parse_quiet_hours(self):
        assert parse_quiet_hours("22:30-07:00") == [(dtime(22, 30), dtime(7, 0))]
        assert parse_quiet_hours(["nope", "12:00-13:00"]) == [(dtime(12), dtime(13))]
        assert parse_quiet_hours(None) == []
        spans = parse_quiet_hours("23:00-08:00")
        assert in_spans(dtime(23, 30), spans) and in_spans(dtime(7, 59), spans)
        assert not in_spans(dtime(8, 0), spans)
        assert in_spans(dtime(12, 30), parse_quiet_hours("12:00-13:00"))
