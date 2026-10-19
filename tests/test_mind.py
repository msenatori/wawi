"""Wawi's mind: small local model, Claude for important moments, the spaceship.

Nothing here downloads a model, opens a socket or calls Claude: the local
model is a fake runner, the SDK query and urllib are replaced, and DNS is a
dict. What is tested is the part that must hold without any of them -- rate
limits, fallbacks, the order of reactions, and the guards on fetch_page.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
import urllib.error

import pytest

from wawi import mind as mind_pkg
from wawi import tools
from wawi.mind import claude as claude_mod
from wawi.mind import fetch as fetch_mod
from wawi.mind import local, peers
from wawi.mind.core import Mind
from wawi.mind.settings import MindSettings, parse
from wawi.reactions import Event, Reaction, ReactionBus

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class RecordingBus(ReactionBus):
    """A real bus that also remembers push ORDER (pop() is by priority)."""

    def __init__(self) -> None:
        super().__init__()
        self.pushed: list[Reaction] = []

    def push(self, reaction: Reaction) -> None:
        self.pushed.append(reaction)
        super().push(reaction)


class FakeRunner:
    def __init__(self, outputs=("Hola, soy Wawi y tengo hambre.",), *, delay=0.0,
                 load_error=None, gen_error=None, load_delay=0.0):
        self.outputs = list(outputs)
        self.delay = delay
        self.load_error = load_error
        self.gen_error = gen_error
        self.load_delay = load_delay
        self.loads = 0
        self.calls: list[tuple[str, str]] = []
        self.threads: set[str] = set()

    def load(self):
        self.loads += 1
        self.threads.add(threading.current_thread().name)
        time.sleep(self.load_delay)
        if self.load_error:
            raise self.load_error

    def generate(self, system, user, max_tokens, temperature):
        self.threads.add(threading.current_thread().name)
        self.calls.append((system, user))
        if self.delay:
            time.sleep(self.delay)
        if self.gen_error:
            raise self.gen_error
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]


class FakeClaude:
    def __init__(self, answer="Respuesta de Claude.", error=None, allow=True):
        self.answer = answer
        self.error = error
        self.allow = allow
        self.prompts: list[tuple[str, str]] = []

    def ask(self, prompt, system, *, timeout=None):
        self.prompts.append((prompt, system))
        if self.error:
            raise self.error
        return self.answer

    def ask_limited(self, key, prompt, system):
        if not self.allow:
            raise claude_mod.ClaudeUnavailable("rate limited")
        return self.ask(prompt, system)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_mind(settings=None, **kw):
    bus = kw.pop("bus", RecordingBus())
    m = Mind(settings or MindSettings(), bus=bus, investigate_delay_s=0, **kw)
    return m, bus


def page(text="Texto largo de la página " * 5, title="Título", url="https://example.com/"):
    return fetch_mod.Page(url=url, status=200, content_type="text/html", text=text, title=title)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


class TestSettings:
    def test_defaults_download_nothing_and_wake_claude_only_for_web_down(self):
        s = parse(None)
        assert s.local_model == "none"
        assert s.claude_events == ("web.down",)
        assert s.fetch_allow_private is False

    def test_none_spellings_disable_the_local_model(self):
        for raw in ("none", "", "off", None):
            assert parse({"local_model": raw}).local_model == "none"

    def test_a_bad_backend_is_loud(self):
        with pytest.raises(ValueError):
            parse({"local_backend": "llama.cpp"})

    def test_config_load_parses_the_mind_section(self, tmp_path):
        from wawi import config

        p = tmp_path / "config.yaml"
        p.write_text("mind:\n  local_model: mlx-community/Qwen3-1.7B-4bit\n  claude_events: []\n")
        cfg = config.load(p)
        assert cfg.mind.local_model == "mlx-community/Qwen3-1.7B-4bit"
        assert cfg.mind.claude_events == ()

    def test_config_load_rejects_a_bad_mind_section(self, tmp_path):
        from wawi import config

        p = tmp_path / "config.yaml"
        p.write_text("mind: 3\n")
        with pytest.raises(config.ConfigError):
            config.load(p)


# --------------------------------------------------------------------------- #
# Local model runner
# --------------------------------------------------------------------------- #


class TestLocalModel:
    def test_lazy_load_then_generate_on_one_worker_thread(self):
        runner = FakeRunner()
        model = local.LocalModel(runner, call_timeout_s=2)
        assert runner.loads == 0  # nothing loads at construction
        out = model.generate("s", "u")
        assert out == "Hola, soy Wawi y tengo hambre."
        assert runner.loads == 1
        assert runner.threads == {"wawi-mind-local"}
        assert threading.current_thread().name not in runner.threads
        model.generate("s", "u")
        assert runner.loads == 1
        model.close()

    def test_timeout_returns_none_quickly(self):
        model = local.LocalModel(FakeRunner(delay=1.0), call_timeout_s=0.05)
        t0 = time.monotonic()
        assert model.generate("s", "u") is None
        assert time.monotonic() - t0 < 0.8
        model.close()

    def test_a_busy_model_refuses_instead_of_queueing(self):
        model = local.LocalModel(FakeRunner(delay=0.5), call_timeout_s=0.05)
        assert model.generate("s", "u") is None  # still running on the worker
        t0 = time.monotonic()
        assert model.generate("s", "u", timeout=2) is None  # refused, not queued
        assert time.monotonic() - t0 < 0.2
        model.close()

    def test_a_failed_load_switches_it_off_for_the_run(self):
        runner = FakeRunner(load_error=RuntimeError("no weights"))
        model = local.LocalModel(runner)
        assert model.generate("s", "u") is None
        assert model.broken
        assert model.generate("s", "u") is None
        assert runner.loads == 1

    def test_slow_load_is_bounded(self):
        model = local.LocalModel(FakeRunner(load_delay=1.0), load_timeout_s=0.05)
        t0 = time.monotonic()
        assert model.generate("s", "u") is None
        assert time.monotonic() - t0 < 0.8

    def test_generation_error_is_none(self):
        model = local.LocalModel(FakeRunner(gen_error=ValueError("x")))
        assert model.generate("s", "u") is None

    def test_thinking_blocks_are_stripped(self):
        model = local.LocalModel(FakeRunner(["<think>\nhmm\n</think>\nGuau, hola."]))
        assert model.generate("s", "u") == "Guau, hola."

    def test_none_model_builds_nothing(self):
        assert local.build_local_model("none") is None

    def test_mlx_missing_degrades_to_none(self, monkeypatch):
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name, *a: None if name == "mlx_lm" else real(name, *a))
        assert local.build_local_model("mlx-community/Qwen3-1.7B-4bit", "mlx") is None

    def test_auto_backend_follows_the_brain(self):
        m = local.build_local_model("qwen3:1.7b", "auto", brain_backend="ollama")
        assert isinstance(m._runner, local._OllamaRunner)
        assert m._runner.model == "qwen3:1.7b"


# --------------------------------------------------------------------------- #
# Bubbles
# --------------------------------------------------------------------------- #

SUMMARY = local.StateSummary(16, 30, {"mood": "aburrida"}, ())


class TestParseBubble:
    @pytest.mark.parametrize("raw,expected", [
        ("Che, ¿y mi comida?", "Che, ¿y mi comida?"),
        ('"Llueve. Hoy no salgo."', "Llueve. Hoy no salgo."),
        ("Wawi: Hola che", "Hola che"),
        ("«Todo tranqui»\nSegunda línea", "Todo tranqui"),
    ])
    def test_accepts_clean_lines(self, raw, expected):
        assert local.parse_bubble(raw) == expected

    @pytest.mark.parametrize("raw", [
        None, "", "ok", "a" * 61, "Mirá https://x.com", "Hola 🐶 che", "**Hola** che",
        "Lo siento, pero no puedo.", "Como modelo de lenguaje no tengo hambre", "1234 567",
    ])
    def test_rejects_junk(self, raw):
        assert local.parse_bubble(raw) is None


class TestPersonality:
    def test_rate_limited_between_min_and_max(self):
        clock = Clock()
        p = local.Personality(None, min_interval_s=600, max_interval_s=1200, clock=clock)
        assert p.maybe_bubble(SUMMARY) is None  # not due at start
        clock.t += 599
        assert p.maybe_bubble(SUMMARY) is None
        clock.t += 602
        assert p.maybe_bubble(SUMMARY)  # due by max
        assert p.maybe_bubble(SUMMARY) is None  # re-armed
        clock.t += 1201
        assert p.maybe_bubble(SUMMARY)

    def test_never_during_quiet_mode(self):
        clock = Clock()
        p = local.Personality(None, min_interval_s=60, max_interval_s=60, clock=clock,
                              quiet=lambda: True)
        clock.t += 1000
        assert p.maybe_bubble(SUMMARY) is None

    def test_unknown_quiet_allows_a_bubble(self):
        clock = Clock()
        p = local.Personality(None, min_interval_s=60, max_interval_s=60, clock=clock,
                              quiet=lambda: None)
        clock.t += 61
        assert p.maybe_bubble(SUMMARY)

    def test_model_line_is_used_when_clean(self):
        model = local.LocalModel(FakeRunner(["Me aburro, che."]))
        p = local.Personality(model)
        assert p.line(SUMMARY) == "Me aburro, che."

    @pytest.mark.parametrize("runner", [
        FakeRunner(["Lo siento, como asistente no puedo."]),
        FakeRunner(delay=1.0),
        FakeRunner(load_error=RuntimeError()),
    ])
    def test_canned_fallback_when_the_model_is_bad_slow_or_missing(self, runner):
        model = local.LocalModel(runner, call_timeout_s=0.05)
        line = local.Personality(model, timeout_s=0.05).line(SUMMARY)
        all_canned = {ln for pool in local.CANNED.values() for ln in pool}
        assert line in all_canned
        assert len(line) <= local.BUBBLE_MAX_CHARS

    def test_canned_lines_are_valid_bubbles(self):
        for pool in local.CANNED.values():
            for line in pool:
                assert local.parse_bubble(line) == line, line

    def test_hunger_wins_over_the_hour(self):
        hungry = local.StateSummary(10, 0, {"hunger": 0.9}, ())
        p = local.Personality(None)
        assert p.canned(hungry) in local.CANNED["hambre"]

    def test_does_not_repeat_itself_back_to_back(self):
        p = local.Personality(None)
        lines = [p.canned(SUMMARY) for _ in range(4)]
        assert len(set(lines)) == 4


class TestTriage:
    def ev(self, severity, kind="rss.item"):
        return Event(kind, severity, title="algo")

    def test_urgent_always_passes_without_a_model(self):
        assert local.Triage(None).decide(self.ev("urgent")) == local.NOW

    def test_urgent_never_asks_the_model(self):
        runner = FakeRunner(["ignore"])
        local.Triage(local.LocalModel(runner)).decide(self.ev("urgent"))
        assert runner.calls == []

    def test_no_model_defers_to_rules(self):
        assert local.Triage(None).decide(self.ev("info")) is None

    def test_warning_is_left_to_rules(self):
        runner = FakeRunner(["ignore"])
        assert local.Triage(local.LocalModel(runner)).decide(self.ev("warning")) is None

    @pytest.mark.parametrize("raw,expected", [
        ("later", "later"), ("Now.", "now"), ("ignore", "ignore"), ("ahora", "now"),
        ("después", "later"), ("<think></think>later", "later"), ("maybe", None), ("", None),
    ])
    def test_parse(self, raw, expected):
        assert local.parse_triage(raw) == expected

    def test_model_answer_is_used_for_info(self):
        t = local.Triage(local.LocalModel(FakeRunner(["ignore"])))
        assert t.decide(self.ev("info")) == local.IGNORE

    def test_model_may_not_drop_a_notice(self):
        t = local.Triage(local.LocalModel(FakeRunner(["ignore"])))
        assert t.decide(self.ev("notice")) == local.LATER

    @pytest.mark.parametrize("runner", [
        FakeRunner(["no sé, depende"]), FakeRunner(delay=1.0), FakeRunner(gen_error=OSError()),
    ])
    def test_model_failure_defers_to_rules(self, runner):
        t = local.Triage(local.LocalModel(runner), timeout_s=0.05)
        assert t.decide(self.ev("info")) is None

    def test_package_triage_without_a_mind_is_none(self, monkeypatch):
        monkeypatch.setattr(mind_pkg, "_MIND", None)
        assert mind_pkg.triage(self.ev("urgent")) is None


class TestStateSummary:
    def test_renders_needs_time_and_recent_events(self):
        events = [Event("weather.rain", "info", title="Empezó a llover"),
                  Event("clock.tick", "info", title="tic")]
        s = local.summarize_state(events, {"hunger": 0.9, "mood": "contenta"},
                                  time.mktime((2026, 10, 15, 16, 30, 0, 0, 0, -1)))
        text = s.render()
        assert "16:30" in text and "tarde" in text
        assert "Hambre: alta" in text and "Ánimo: contenta" in text
        assert "Empezó a llover" in text and "tic" not in text

    def test_non_dict_pet_status_is_ignored(self):
        assert local.summarize_state([], "estoy bien").pet == {}


# --------------------------------------------------------------------------- #
# Claude plumbing
# --------------------------------------------------------------------------- #


class TestRateLimiter:
    def test_per_key_interval(self):
        clock = Clock(0)
        rl = claude_mod.RateLimiter(min_interval_s=600, max_per_hour=10, clock=clock)
        assert rl.allow("a")
        assert not rl.allow("a")
        assert rl.allow("b")
        clock.t = 601
        assert rl.allow("a")

    def test_hourly_cap_across_keys(self):
        clock = Clock(0)
        rl = claude_mod.RateLimiter(min_interval_s=0, max_per_hour=2, clock=clock)
        assert rl.allow("a") and rl.allow("b")
        assert not rl.allow("c")
        clock.t = 3601
        assert rl.allow("c")


def _sdk_messages(text, is_error=False):
    from claude_agent_sdk import AssistantMessage, ResultMessage
    from claude_agent_sdk.types import TextBlock

    msgs = []
    if text:
        msgs.append(AssistantMessage(content=[TextBlock(text=text)], model="m"))
    msgs.append(ResultMessage(subtype="error" if is_error else "success", duration_ms=1,
                              duration_api_ms=1, is_error=is_error, num_turns=1, session_id="s"))
    return msgs


class TestClaudeMind:
    def fake_query(self, monkeypatch, text, is_error=False, delay=0.0):
        seen = {}

        async def fake(prompt, options):
            seen["prompt"], seen["options"] = prompt, options
            if delay:
                await asyncio.sleep(delay)
            for m in _sdk_messages(text, is_error):
                yield m

        monkeypatch.setattr(claude_mod, "_sdk_query", fake)
        return seen

    def test_ask_uses_the_lean_subscription_path(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
        seen = self.fake_query(monkeypatch, "El sitio **devuelve** 503. https://x.com")
        cm = claude_mod.ClaudeMind(model="claude-sonnet-4-6")
        out = cm.ask("evidencia", "sistema")
        assert out == "El sitio devuelve 503. el enlace"
        opts = seen["options"]
        assert opts.model == "claude-sonnet-4-6"
        assert opts.max_turns == 1
        assert "ANTHROPIC_API_KEY" not in opts.env
        assert opts.allowed_tools == []

    def test_works_from_inside_a_running_event_loop(self, monkeypatch):
        # The brain calls tools from inside its engine loop.
        self.fake_query(monkeypatch, "Hola.")
        cm = claude_mod.ClaudeMind(model="m")

        async def inside():
            return cm.ask("p", "s")

        assert asyncio.run(inside()) == "Hola."

    def test_errors_and_timeouts_become_unavailable(self, monkeypatch):
        self.fake_query(monkeypatch, "", is_error=True)
        cm = claude_mod.ClaudeMind(model="m")
        with pytest.raises(claude_mod.ClaudeUnavailable):
            cm.ask("p", "s")
        self.fake_query(monkeypatch, "tarde", delay=1.0)
        with pytest.raises(claude_mod.ClaudeUnavailable):
            cm.ask("p", "s", timeout=0.05)

    def test_ask_limited_respects_the_limiter(self, monkeypatch):
        self.fake_query(monkeypatch, "Hola.")
        cm = claude_mod.ClaudeMind(model="m", limiter=claude_mod.RateLimiter(
            min_interval_s=600, max_per_hour=5))
        assert cm.ask_limited("k", "p", "s") == "Hola."
        with pytest.raises(claude_mod.ClaudeUnavailable):
            cm.ask_limited("k", "p", "s")


class TestRuleDiagnosis:
    @pytest.mark.parametrize("evidence,needle", [
        ({"internet": {"ok": False}, "dns": {"ok": True}, "site": {"ok": False}}, "conexión"),
        ({"internet": {"ok": True}, "dns": {"ok": False}, "site": {"ok": False}}, "DNS"),
        ({"internet": {"ok": True}, "dns": {"ok": True}, "site": {"ok": True, "status": 503}}, "503"),
        ({"internet": {"ok": True}, "dns": {"ok": True},
          "site": {"ok": False, "error": "URLError: timed out"}}, "a tiempo"),
        ({"internet": {"ok": True}, "dns": {"ok": True},
          "site": {"ok": False, "error": "Connection refused"}}, "rechaza"),
        ({"internet": {"ok": True}, "dns": {"ok": True}, "site": {"ok": True, "status": 200}}, "bien"),
    ])
    def test_each_shape_has_a_sentence(self, evidence, needle):
        assert needle in claude_mod.rule_diagnosis(evidence)

    def test_diagnose_prefers_claude_and_falls_back(self):
        ev = {"internet": {"ok": True}, "dns": {"ok": False}, "site": {}}
        assert claude_mod.diagnose(ev, FakeClaude("Es el DNS."), "k") == ("Es el DNS.", True)
        text, from_claude = claude_mod.diagnose(ev, FakeClaude(allow=False), "k")
        assert not from_claude and "DNS" in text
        assert claude_mod.diagnose(ev, None, "k")[1] is False


# --------------------------------------------------------------------------- #
# web.down investigation
# --------------------------------------------------------------------------- #

EVIDENCE = {"url": "https://x.com", "dns": {"ok": True}, "internet": {"ok": True},
            "site": {"ok": True, "status": 502}}


class TestWebDown:
    def test_detective_with_claude_diagnosis_and_voice_when_not_quiet(self):
        fc = FakeClaude("El servidor de x.com devuelve 502; es de ellos.")
        m, bus = make_mind(claude=fc, quiet=lambda: False,
                           gather=lambda url, probe_url: EVIDENCE)
        ev = Event("web.down", "urgent", key="https://x.com")
        assert m.investigate_web_down(ev).startswith("El servidor")
        (r,) = bus.pushed
        assert r.costume == "detective" and r.behaviour == "detective"
        assert r.bubble == fc.answer and r.speak is True and r.source is ev
        assert '"status": 502' in fc.prompts[0][0]

    @pytest.mark.parametrize("quiet", [True, None])
    def test_silent_in_quiet_mode_or_when_quiet_is_unknown(self, quiet):
        m, bus = make_mind(claude=FakeClaude(), quiet=lambda: quiet,
                           gather=lambda url, probe_url: EVIDENCE)
        m.investigate_web_down(Event("web.down", "urgent", key="https://x.com"))
        assert bus.pushed[0].speak is False

    def test_rules_answer_when_claude_is_rate_limited(self):
        m, bus = make_mind(claude=FakeClaude(allow=False), quiet=lambda: False,
                           gather=lambda url, probe_url: EVIDENCE)
        m.investigate_web_down(Event("web.down", "urgent", key="https://x.com"))
        assert "502" in bus.pushed[0].bubble

    def test_claude_not_used_when_web_down_is_not_opted_in(self):
        fc = FakeClaude()
        m, bus = make_mind(MindSettings(claude_events=()), claude=fc,
                           gather=lambda url, probe_url: EVIDENCE)
        m.investigate_web_down(Event("web.down", "urgent", key="https://x.com"))
        assert fc.prompts == [] and "502" in bus.pushed[0].bubble

    def test_bus_event_triggers_one_investigation_per_key(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def gather(url, probe_url):
            calls.append(url)
            started.set()
            release.wait(2)
            return EVIDENCE

        m, bus = make_mind(gather=gather, quiet=lambda: None)
        m.start()
        ev = Event("web.down", "urgent", key="https://x.com")
        bus.emit(ev)
        assert started.wait(2)
        bus.emit(ev)  # duplicate while the first is running
        release.set()
        for _ in range(100):
            if bus.pushed:
                break
            time.sleep(0.01)
        m.stop()
        assert calls == ["https://x.com"]
        assert bus.pushed[0].costume == "detective"

    def test_info_events_never_wake_claude(self):
        fc = FakeClaude()
        m, bus = make_mind(MindSettings(claude_events=("web.down", "rss.item")), claude=fc)
        m.on_event(Event("rss.item", "info", title="x"))
        time.sleep(0.05)
        assert fc.prompts == [] and bus.pushed == []
        assert m.recent_events("rss.")


# --------------------------------------------------------------------------- #
# Bubble ticker and digest
# --------------------------------------------------------------------------- #


class TestTickAndDigest:
    def test_tick_pushes_a_quiet_low_priority_bubble_when_due(self, monkeypatch):
        monkeypatch.setattr(peers, "pet_status", lambda: None)
        clock = Clock()
        m, bus = make_mind(MindSettings(bubble_min_interval_s=60, bubble_max_interval_s=60),
                           clock=clock, quiet=lambda: False)
        assert m.tick() is None
        clock.t += 61
        line = m.tick()
        assert line and bus.pushed[-1].bubble == line
        assert bus.pushed[-1].priority == 0 and not bus.pushed[-1].speak

    def test_bubbles_off(self):
        clock = Clock()
        m, bus = make_mind(MindSettings(bubbles=False), clock=clock)
        clock.t += 10_000
        assert m.tick() is None and bus.pushed == []

    def test_digest_with_claude(self, monkeypatch):
        monkeypatch.setattr(peers, "sensors_status", lambda: {"disk": "ok"})
        fc = FakeClaude("Salió Python nuevo y el disco está bien.")
        clock = Clock(time.time())
        m, _ = make_mind(claude=fc, clock=clock)
        m.on_event(Event("rss.item", "info", title="Salió Python 3.15", at=clock.t - 60))
        ok, speak, data = m.digest()
        assert ok and speak == fc.answer
        assert "Salió Python 3.15" in fc.prompts[0][0] and '"disk": "ok"' in fc.prompts[0][0]
        assert data["items"][0]["title"] == "Salió Python 3.15"

    def test_digest_without_claude_lists_titles(self, monkeypatch):
        monkeypatch.setattr(peers, "sensors_status", lambda: None)
        clock = Clock(time.time())
        m, _ = make_mind(claude=FakeClaude(error=claude_mod.ClaudeUnavailable("x")), clock=clock)
        for i in range(5):
            m.on_event(Event("rss.item", "info", title=f"Nota {i}", at=clock.t - 10))
        ok, speak, _ = m.digest()
        assert ok and "5 novedades" in speak and "Nota 0" in speak and "2 más" in speak

    def test_digest_with_nothing_new(self, monkeypatch):
        monkeypatch.setattr(peers, "sensors_status", lambda: None)
        fc = FakeClaude()
        m, _ = make_mind(claude=fc)
        assert m.digest() == (True, "No hay nada nuevo por ahora.", {"items": [], "alerts": []})
        assert fc.prompts == []


# --------------------------------------------------------------------------- #
# fetch: guards, size cap, HTML -> text
# --------------------------------------------------------------------------- #


def resolver(mapping):
    def _r(host):
        if host not in mapping:
            import socket

            raise socket.gaierror("nope")
        return mapping[host]
    return _r


PUBLIC = resolver({"example.com": ["93.184.216.34"], "evil.com": ["10.0.0.5"],
                   "v6.example": ["::1"], "mapped.example": ["::ffff:192.168.1.1"]})


class TestUrlGuards:
    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x",
                                     "javascript:alert(1)"])
    def test_only_http_and_https(self, url):
        with pytest.raises(fetch_mod.FetchError) as exc:
            fetch_mod.check_url(url, resolver=PUBLIC)
        assert exc.value.code in ("scheme", "no_host")

    @pytest.mark.parametrize("url", [
        "http://localhost:8080/", "http://127.0.0.1/", "http://192.168.0.1/",
        "http://169.254.169.254/latest/meta-data", "http://[::1]/", "https://evil.com/",
        "https://v6.example/", "https://mapped.example/", "http://printer.local/",
        "http://0.0.0.0/",
    ])
    def test_private_addresses_are_refused(self, url):
        with pytest.raises(fetch_mod.FetchError) as exc:
            fetch_mod.check_url(url, resolver=PUBLIC)
        assert exc.value.code == "private"

    def test_private_allowed_by_config(self):
        assert fetch_mod.check_url("http://192.168.0.1/", allow_private=True,
                                   resolver=PUBLIC) == "http://192.168.0.1/"

    def test_public_passes_and_bare_hosts_get_https(self):
        url = fetch_mod.normalise_url("example.com/docs")
        assert url == "https://example.com/docs"
        assert fetch_mod.check_url(url, resolver=PUBLIC) == url

    def test_unknown_host(self):
        with pytest.raises(fetch_mod.FetchError) as exc:
            fetch_mod.check_url("https://nope.invalid/", resolver=PUBLIC)
        assert exc.value.code == "dns"

    def test_redirects_are_checked_again(self):
        handler = fetch_mod._CheckedRedirects(False, PUBLIC)
        with pytest.raises(fetch_mod.FetchError) as exc:
            handler.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/admin")
        assert exc.value.code == "private"


class TestHtmlToText:
    def test_strips_scripts_styles_and_keeps_blocks(self):
        html = """<html><head><title> Mi  página </title><style>p{}</style></head>
        <body><nav>Inicio</nav><script>alert('x')</script><h1>Hola</h1>
        <p>Primer&nbsp;párrafo con <b>negrita</b>.</p><p>Segundo &amp; último</p>
        <noscript>activá JS</noscript></body></html>"""
        title, text = fetch_mod.html_to_text(html)
        assert title == "Mi página"
        assert "alert" not in text and "p{}" not in text and "activá JS" not in text
        assert text.splitlines() == ["Inicio", "Hola", "Primer párrafo con negrita.",
                                     "Segundo & último"]

    def test_omitted_head_close_does_not_hide_the_body(self):
        title, text = fetch_mod.html_to_text("<html><head><title>T</title><body><p>Hola</p>")
        assert title == "T" and text == "Hola"

    def test_broken_markup_does_not_raise(self):
        title, text = fetch_mod.html_to_text("<p>hola <b>che<div></p></i>")
        assert "hola" in text


class FakeResp:
    def __init__(self, body: bytes, ctype="text/html; charset=utf-8", status=200, url="https://example.com/"):
        self._body = io.BytesIO(body)
        self.status = status
        self._url = url
        self.headers = {"Content-Type": ctype}
        self.read_sizes: list[int] = []

    def read(self, n=-1):
        self.read_sizes.append(n)
        return self._body.read(n)

    def geturl(self):
        return self._url

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    def __init__(self, resp=None, error=None):
        self.resp, self.error, self.requests = resp, error, []

    def open(self, req, timeout=None):
        self.requests.append((req, timeout))
        if self.error:
            raise self.error
        return self.resp


class TestFetch:
    def test_size_cap_reads_at_most_max_plus_one(self):
        body = b"<p>" + b"a" * 50_000 + b"</p>"
        resp = FakeResp(body)
        p = fetch_mod.fetch("example.com", max_bytes=20_000, resolver=PUBLIC,
                            opener=FakeOpener(resp))
        assert resp.read_sizes == [20_001]
        assert p.truncated and len(p.text) <= 20_000

    def test_html_is_converted_and_charset_respected(self):
        body = "<title>Café</title><p>Año nuevo</p>".encode("latin-1")
        p = fetch_mod.fetch("https://example.com", resolver=PUBLIC,
                            opener=FakeOpener(FakeResp(body, "text/html; charset=iso-8859-1")))
        assert p.title == "Café" and p.text == "Año nuevo" and not p.truncated

    def test_binary_content_is_refused(self):
        with pytest.raises(fetch_mod.FetchError) as exc:
            fetch_mod.fetch("https://example.com/x.zip", resolver=PUBLIC,
                            opener=FakeOpener(FakeResp(b"PK..", "application/zip")))
        assert exc.value.code == "not_text"

    def test_http_error_and_timeout(self):
        err = urllib.error.HTTPError("https://example.com", 404, "nf", {}, None)
        with pytest.raises(fetch_mod.FetchError) as exc:
            fetch_mod.fetch("https://example.com", resolver=PUBLIC, opener=FakeOpener(error=err))
        assert exc.value.code == "http_404"
        with pytest.raises(fetch_mod.FetchError) as exc:
            fetch_mod.fetch("https://example.com", resolver=PUBLIC,
                            opener=FakeOpener(error=urllib.error.URLError(TimeoutError("timed out"))))
        assert exc.value.code == "timeout"

    def test_private_url_never_reaches_the_network(self):
        opener = FakeOpener(FakeResp(b"x"))
        with pytest.raises(fetch_mod.FetchError):
            fetch_mod.fetch("http://192.168.0.1/", resolver=PUBLIC, opener=opener)
        assert opener.requests == []


# --------------------------------------------------------------------------- #
# The spaceship flow
# --------------------------------------------------------------------------- #


class TestFetchPageFlow:
    def test_success_ship_then_celebrate_with_matching_keys(self):
        fc = FakeClaude("La página dice que abre a las nueve.")
        m, bus = make_mind(claude=fc, fetcher=lambda url, **kw: page())
        ok, speak, data = m.fetch_page("example.com", "¿a qué hora abre?")
        assert ok and speak == fc.answer and data["title"] == "Título"
        ship, done = bus.pushed
        assert ship.behaviour == "spaceship" and ship.until_key.startswith("fetch:")
        assert done.behaviour == "celebrate" and done.cancel_key == ship.until_key
        assert "¿a qué hora abre?" in fc.prompts[0][0]
        assert "<pagina>" in fc.prompts[0][0]

    def test_each_trip_has_its_own_key(self):
        m, bus = make_mind(claude=FakeClaude(), fetcher=lambda url, **kw: page())
        m.fetch_page("a.com")
        m.fetch_page("b.com")
        assert bus.pushed[0].until_key != bus.pushed[2].until_key

    def test_fetch_error_is_sad_and_closes_the_ship(self):
        def fail(url, **kw):
            raise fetch_mod.FetchError("private", "Esa dirección es de tu red local.")

        fc = FakeClaude()
        m, bus = make_mind(claude=fc, fetcher=fail)
        ok, speak, data = m.fetch_page("http://192.168.0.1")
        assert not ok and "red local" in speak and data["error"] == "private"
        ship, done = bus.pushed
        assert done.behaviour == "sad" and done.cancel_key == ship.until_key
        assert fc.prompts == []

    def test_claude_failure_is_sad_and_hands_the_brain_an_excerpt(self):
        m, bus = make_mind(claude=FakeClaude(error=claude_mod.ClaudeUnavailable("x")),
                           fetcher=lambda url, **kw: page())
        ok, speak, data = m.fetch_page("example.com")
        assert not ok and data["error"] == "read_failed" and data["excerpt"]
        assert [r.behaviour for r in bus.pushed] == ["spaceship", "sad"]

    def test_unexpected_crash_still_closes_the_ship_once(self):
        def boom(url, **kw):
            raise RuntimeError("bug")

        m, bus = make_mind(claude=FakeClaude(), fetcher=boom)
        ok, _, _ = m.fetch_page("example.com")
        assert not ok
        assert [r.behaviour for r in bus.pushed] == ["spaceship", "sad"]

    def test_empty_page_is_sad(self):
        m, bus = make_mind(claude=FakeClaude(), fetcher=lambda url, **kw: page(text="  "))
        ok, speak, _ = m.fetch_page("example.com")
        assert not ok and "JavaScript" in speak
        assert bus.pushed[-1].behaviour == "sad"

    def test_settings_reach_the_fetcher_and_text_is_capped(self):
        seen = {}

        def fetcher(url, **kw):
            seen.update(kw)
            return page(text="x" * 50_000)

        fc = FakeClaude()
        m, _ = make_mind(MindSettings(fetch_allow_private=True, fetch_max_chars=1000),
                         claude=fc, fetcher=fetcher)
        _, _, data = m.fetch_page("example.com")
        assert seen["allow_private"] is True
        assert fc.prompts[0][0].count("x") <= 1000 + 10
        assert data["truncated"]


# --------------------------------------------------------------------------- #
# Tools, guarded imports, prompts, wiring
# --------------------------------------------------------------------------- #


@pytest.fixture
def no_peers(monkeypatch):
    real = peers._function

    def fake(module, name):
        if module.startswith(("wawi.pet", "wawi.sensors", "wawi.reflexes")):
            return None
        return real(module, name)

    monkeypatch.setattr(peers, "_function", fake)


@pytest.fixture
def fake_peers(monkeypatch):
    calls = []
    table = {
        ("wawi.pet.api", "feed"): lambda: "¡Ñam! Gracias.",
        ("wawi.pet.api", "water"): lambda: {"speak": "Glu glu."},
        ("wawi.pet.api", "play"): lambda: {"ok": True},
        ("wawi.pet.api", "status"): lambda: {"mood": "contenta", "hunger": 0.8, "energy": 0.2},
        ("wawi.sensors", "status"): lambda: {"web": "ok", "disk": "12% libre"},
        ("wawi.reflexes", "is_quiet"): lambda: True,
    }

    def fake(module, name):
        fn = table.get((module, name))
        if fn is None:
            return None

        def wrapped(*a):
            calls.append((module, name))
            return fn(*a)
        return wrapped

    monkeypatch.setattr(peers, "_function", fake)
    return calls


@pytest.fixture
def installed_mind(monkeypatch):
    m, bus = make_mind(claude=FakeClaude("Resumen."), fetcher=lambda url, **kw: page())
    monkeypatch.setattr(tools, "_MIND", m)
    return m, bus


class TestToolsRegistration:
    def test_no_mind_no_extra_schemas(self, monkeypatch):
        monkeypatch.setattr(tools, "_MIND", None)
        assert [t["name"] for t in tools.active_tools()] == [t["name"] for t in tools.TOOLS]

    def test_mind_adds_its_tools_but_not_absent_peers(self, installed_mind, no_peers):
        names = [t["name"] for t in tools.active_tools()]
        assert "fetch_page" in names and "whats_new" in names
        assert not any(n.startswith("pet_") for n in names)

    def test_peers_present_add_pet_and_sensor_tools(self, installed_mind, fake_peers):
        names = {t["name"] for t in tools.active_tools()}
        assert {"pet_feed", "pet_water", "pet_play", "pet_status"} <= names

    def test_mind_tools_off_by_config(self, monkeypatch):
        m, _ = make_mind(MindSettings(tools=False))
        monkeypatch.setattr(tools, "_MIND", m)
        assert "fetch_page" not in {t["name"] for t in tools.active_tools()}

    def test_with_task_runner_too(self, installed_mind, no_peers, monkeypatch):
        class Runner:
            class _options:
                tools = ("Read",)
        monkeypatch.setattr(tools, "_TASK_RUNNER", Runner())
        names = [t["name"] for t in tools.active_tools()]
        assert "start_task" in names and "fetch_page" in names

    def test_new_schemas_stay_terse(self):
        import json

        # What the model is sent: without the mascot's "animation" field.
        sent = lambda specs: [tools.for_model(t) for t in specs]  # noqa: E731
        assert len(json.dumps(sent(tools.MIND_TOOLS))) < 700
        status = sent(t for t in tools.SENSOR_TOOLS if t["name"] == "sensor_status")
        watch = sent(t for t in tools.SENSOR_TOOLS if t["name"] == "watch")
        # 800 for feed/water/play/status, +150 for pet_personality.
        assert len(json.dumps(sent(tools.PET_TOOLS) + status)) < 950
        # One schema for add/remove/list of every kind, not six tools.
        assert len(json.dumps(watch)) < 450

    def test_brain_bridge_registers_them(self, installed_mind, no_peers):
        from wawi import brain

        bridge = brain._build_bridge(brain._TurnState(), None)
        assert f"mcp__{brain.MCP_SERVER_NAME}__fetch_page" in bridge.handlers


class TestToolCalls:
    def test_fetch_page_dispatch(self, installed_mind):
        _, bus = installed_mind
        result = tools.dispatch("fetch_page", {"url": "example.com", "question": "¿qué?"})
        assert result.ok and result.speak == "Resumen."
        assert [r.behaviour for r in bus.pushed] == ["spaceship", "celebrate"]

    def test_fetch_page_needs_a_url(self, installed_mind):
        assert tools.dispatch("fetch_page", {}).ok is False

    def test_mind_tools_without_mind(self, monkeypatch):
        monkeypatch.setattr(tools, "_MIND", None)
        assert tools.dispatch("fetch_page", {"url": "x.com"}).data["error"] == "mind_disabled"
        assert tools.dispatch("whats_new", {}).data["error"] == "mind_disabled"

    def test_whats_new_dispatch(self, installed_mind, no_peers):
        m, _ = installed_mind
        m.on_event(Event("rss.item", "info", title="Algo"))
        result = tools.dispatch("whats_new", {})
        assert result.ok and result.speak == "Resumen."

    @pytest.mark.parametrize("tool,needle", [
        ("pet_feed", "comer"), ("pet_water", "agua"), ("pet_play", "jugar"),
        ("pet_status", "cómo estoy"),
    ])
    def test_pet_tools_without_a_pet(self, no_peers, tool, needle):
        result = tools.dispatch(tool, {})
        assert not result.ok and needle in result.speak
        assert result.data["error"] == "pet_missing"

    def test_pet_tools_with_a_pet(self, fake_peers):
        assert tools.dispatch("pet_feed", {}).speak == "¡Ñam! Gracias."
        assert tools.dispatch("pet_water", {}).speak == "Glu glu."
        assert tools.dispatch("pet_play", {}).speak == "¡Gracias!"
        status = tools.dispatch("pet_status", {})
        assert status.speak == "Estoy contenta, tengo mucha hambre, estoy cansada."
        assert ("wawi.pet.api", "feed") in fake_peers

    def test_pet_api_that_raises(self, monkeypatch):
        def boom():
            raise RuntimeError("x")
        monkeypatch.setattr(peers, "_function", lambda m, n: boom)
        assert tools.dispatch("pet_feed", {}).data["error"] == "pet_failed"
        assert tools.dispatch("pet_status", {}).data["error"] == "pet_failed"


class TestPeers:
    def test_real_guarded_imports_report_missing(self):
        # In this checkout the body/sensors modules may or may not exist; either
        # way the lookups must answer without raising.
        assert isinstance(peers.pet_available(), bool)
        assert isinstance(peers.sensors_available(), bool)
        assert peers.call("wawi.does_not_exist", "x") is peers.MISSING
        assert peers.is_quiet() in (True, False, None)

    def test_is_quiet_from_reflexes(self, fake_peers):
        assert peers.is_quiet() is True

    def test_is_quiet_unknown(self, no_peers):
        assert peers.is_quiet() is None


class TestPromptsAndWiring:
    def test_prompt_mentions_the_tools_only_with_a_mind(self, monkeypatch):
        from wawi import prompts

        monkeypatch.setattr(tools, "_MIND", None)
        assert "fetch_page" not in prompts.system_prompt()
        m, _ = make_mind()
        monkeypatch.setattr(tools, "_MIND", m)
        assert "fetch_page" in prompts.system_prompt()
        assert "fetch_page" in prompts.system_prompt(language="en")

    def test_build_respects_enabled_and_backend(self):
        class Cfg:
            backend = "claude"
            model_on_demand = "claude-sonnet-4-6"
            ollama_url = "http://localhost:11434/api/chat"
            mind = MindSettings()

        m = mind_pkg.build(Cfg)
        assert m.claude.model == "claude-sonnet-4-6" and m.local is None
        Cfg.backend = "ollama"
        assert mind_pkg.build(Cfg).claude is None
        Cfg.mind = MindSettings(enabled=False)
        assert mind_pkg.build(Cfg) is None

    def test_orchestrator_builds_installs_and_survives_failure(self, monkeypatch):
        from wawi.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        monkeypatch.setattr(tools, "_MIND", None)

        class Cfg:
            backend = "claude"
            model_on_demand = "m"
            mind = MindSettings()

        m = orch._build_mind(Cfg)
        try:
            assert tools._MIND is m and mind_pkg.get() is m
            monkeypatch.setattr(mind_pkg, "build", lambda cfg: 1 / 0)
            assert orch._build_mind(Cfg) is None
            assert tools._MIND is None
        finally:
            mind_pkg.install(None)
            tools.set_mind(None)
