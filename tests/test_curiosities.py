"""Curiosidades: scheduling, the cache, the sources, "Ver más" and the brain."""

from __future__ import annotations

import json
import random
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from wawi import curiosities as cur
from wawi.curiosities import Curiosity, CuriositySettings, CuriosityService

TODAY = date(2026, 10, 16)
NOON = datetime(2026, 10, 16, 12, 0).timestamp()
MIDNIGHT = datetime(2026, 10, 16, 23, 30).timestamp()
DAWN = datetime(2026, 10, 16, 5, 0).timestamp()

BODY = ("La berenjena es una planta de estación cálida.\n\n"
        "En el hemisferio sur se cosecha sobre todo entre enero y marzo.")


def item(title="Berenjena", teaser="¿Sabías que la berenjena se cosecha a fines del verano?",
         **kw) -> Curiosity:
    return Curiosity(teaser=teaser, title=title, body=BODY, **kw)


def settings(**kw) -> CuriositySettings:
    base = dict(first_after_min=0.0, jitter=0.0, every_min=60.0, quiet_after_talk_s=300.0)
    base.update(kw)
    return CuriositySettings.from_dict(base)


def service(tmp_path, items=(), **kw) -> CuriosityService:
    svc = CuriosityService(settings(**kw.pop("cfg", {})), state_path=tmp_path / "c.json",
                           now=0.0, background=False, rng=random.Random(1), **kw)
    svc.store.queue = list(items)
    svc.store.refilled_on = TODAY.isoformat()  # no refill unless a test wants one
    return svc


@pytest.fixture(autouse=True)
def _no_pause(monkeypatch):
    monkeypatch.setattr(cur, "_RANDOM_PAUSE_S", 0.0)


IDLE = dict(idle=True, asleep=False, busy=False, channel_open=False)


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #


class TestSchedule:
    def test_shows_one_when_idle_in_the_day(self, tmp_path):
        svc = service(tmp_path, [item()])
        shown = svc.poll(10.0, NOON, **IDLE)
        assert shown is not None and shown.title == "Berenjena"
        assert svc.bubble_text(11.0) == f"{shown.teaser}\n{cur.LINK}"

    @pytest.mark.parametrize("wall", [MIDNIGHT, DAWN])
    def test_never_at_night(self, tmp_path, wall):
        svc = service(tmp_path, [item()])
        assert svc.poll(10.0, wall, **IDLE) is None

    def test_night_follows_the_pet_rhythm_bounds(self, tmp_path):
        svc = service(tmp_path, [item()], night_start=23.75, night_end=7.0)
        assert svc.poll(10.0, MIDNIGHT, **IDLE) is not None

    @pytest.mark.parametrize("gate", [dict(idle=False), dict(asleep=True), dict(busy=True),
                                      dict(channel_open=True)])
    def test_waits_while_not_idle(self, tmp_path, gate):
        svc = service(tmp_path, [item()])
        assert svc.poll(10.0, NOON, **{**IDLE, **gate}) is None
        # …and it is shown once the gate opens again.
        assert svc.poll(20.0, NOON, **IDLE) is not None

    def test_quiet_right_after_a_conversation(self, tmp_path):
        svc = service(tmp_path, [item()])
        svc.last_talk = 100.0
        assert svc.poll(100.0 + 120, NOON, **IDLE) is None
        assert svc.poll(100.0 + 301, NOON, **IDLE) is not None

    def test_a_turn_counts_as_talk(self, tmp_path):
        from wawi.state import State

        svc = service(tmp_path)
        svc.observe_state(State.IDLE, State.LISTENING)
        assert svc.last_talk is not None

    def test_cooldown_between_curiosities(self, tmp_path):
        svc = service(tmp_path, [item("A"), item("B")])
        assert svc.poll(10.0, NOON, **IDLE).title == "A"
        svc.dismiss()
        assert svc.poll(10.0 + 30 * 60, NOON, **IDLE) is None
        assert svc.poll(10.0 + 61 * 60, NOON, **IDLE).title == "B"

    def test_jitter_stays_within_bounds(self):
        s = settings(jitter=0.35)
        sched = cur.Schedule(s, rng=random.Random(3), now=0.0)
        gaps = []
        for _ in range(50):
            sched.fired(0.0)
            gaps.append(sched.next_at / 60.0)
        assert min(gaps) >= 60 * 0.65 and max(gaps) <= 60 * 1.35
        assert max(gaps) - min(gaps) > 5

    def test_not_right_at_start(self, tmp_path):
        svc = CuriosityService(settings(first_after_min=20.0), state_path=None, now=0.0,
                               background=False)
        svc.store.queue = [item()]
        svc.store.refilled_on = TODAY.isoformat()
        assert svc.poll(60.0, NOON, **IDLE) is None
        assert svc.poll(21 * 60.0, NOON, **IDLE) is not None

    def test_the_bubble_fades_after_show_s(self, tmp_path):
        svc = service(tmp_path, [item()], cfg={"show_s": 25})
        svc.poll(10.0, NOON, **IDLE)
        assert svc.bubble_text(34.0) is not None
        assert svc.bubble_text(36.0) is None

    def test_toggle_off_persists_and_stops_them(self, tmp_path):
        svc = service(tmp_path, [item()])
        svc.set_enabled(False)
        assert svc.poll(10.0, NOON, **IDLE) is None
        again = CuriosityService(settings(), state_path=tmp_path / "c.json", background=False)
        assert again.enabled is False
        again.set_enabled(True)
        assert CuriosityService(settings(), state_path=tmp_path / "c.json").enabled is True

    def test_logs_one_info_line_when_shown(self, tmp_path, caplog):
        svc = service(tmp_path, [item()])
        with caplog.at_level("INFO", logger="wawi.curiosities"):
            svc.poll(10.0, NOON, **IDLE)
        assert any("Curiosity shown" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# The cache and its refills
# --------------------------------------------------------------------------- #


class TestRefill:
    def test_refills_when_few_remain_once_a_day(self, tmp_path):
        calls = []

        def claude(prompt, system):
            calls.append(prompt)
            return json.dumps([{"teaser": f"¿Sabías que el dato {i} es cierto?",
                                "title": f"Tema {i}", "body_markdown": BODY,
                                "image_query": ""} for i in range(5)])

        svc = service(tmp_path, [item()], claude=claude)
        svc.store.refilled_on = ""
        svc.store.next_source = "claude"
        assert svc.maybe_refill(TODAY) is True
        assert len(calls) == 1 and svc.store.pending(TODAY) == 6
        assert svc.store.next_source == "wikipedia"  # alternates
        svc.store.queue = []
        assert svc.maybe_refill(TODAY) is False  # already refilled today
        assert svc.needs_refill(date(2026, 10, 17)) is True

    def test_no_refill_while_enough_remain(self, tmp_path):
        svc = service(tmp_path, [item(str(i)) for i in range(5)])
        svc.store.refilled_on = ""
        assert svc.needs_refill(TODAY) is False

    def test_a_failed_refill_backs_off_and_does_not_raise(self, tmp_path, caplog):
        def boom(*_a):
            raise OSError("sin red")

        svc = service(tmp_path, claude=boom, wiki_fetch=boom)
        svc.store.refilled_on = ""
        with caplog.at_level("INFO"):
            svc.maybe_refill(TODAY)
        assert svc.store.queue == []
        assert not any(r.exc_info for r in caplog.records)
        assert svc.needs_refill(TODAY) is False  # backs off for a while

    def test_stale_on_this_day_items_are_dropped(self, tmp_path):
        old = item("Efeméride", day="2026-10-15")
        svc = service(tmp_path, [old, item("Perros")])
        assert svc.store.pending(TODAY) == 1
        assert svc.store.pop(TODAY).title == "Perros"

    def test_used_titles_are_remembered_and_cache_survives_restart(self, tmp_path):
        svc = service(tmp_path, [item("A"), item("B")])
        svc.poll(10.0, NOON, **IDLE)
        again = CuriosityService(settings(), state_path=tmp_path / "c.json")
        assert [c.title for c in again.store.queue] == ["B"]
        assert "A" in again.store.used

    def test_falls_back_to_wikipedia_when_claude_fails(self, tmp_path):
        def boom(prompt, system):
            raise RuntimeError("rate limited")

        svc = service(tmp_path, claude=boom, wiki_fetch=fake_wiki(FEED))
        svc.store.refilled_on = ""
        svc.store.next_source = "claude"
        svc.maybe_refill(TODAY)
        assert svc.store.queue and all(c.source == "wikipedia" for c in svc.store.queue)

    def test_unreadable_state_file_is_ignored(self, tmp_path):
        (tmp_path / "c.json").write_text("{nope", encoding="utf-8")
        svc = CuriosityService(settings(), state_path=tmp_path / "c.json")
        assert svc.store.queue == [] and svc.enabled is True


# --------------------------------------------------------------------------- #
# Generated items
# --------------------------------------------------------------------------- #


class TestParse:
    GOOD = {"teaser": "¿Sabías que los perros sudan por las almohadillas?",
            "title": "Perros y sudor", "body_markdown": BODY, "image_query": "perro"}

    def test_valid_items_survive_code_fences(self):
        text = "```json\n" + json.dumps([self.GOOD]) + "\n```"
        items = cur.parse_items(text)
        assert len(items) == 1 and items[0].image_query == "perro"

    @pytest.mark.parametrize("bad", [
        {"teaser": "x" * 140},                    # teaser too long
        {"teaser": ""},
        {"title": ""},
        {"body_markdown": "corto"},
        {"body_markdown": None},
    ])
    def test_bad_items_are_dropped(self, bad):
        items = cur.parse_items(json.dumps([{**self.GOOD, **bad}, self.GOOD]))
        assert len(items) == 1

    def test_not_json_is_nothing(self):
        assert cur.parse_items("lo siento, no puedo") == []
        assert cur.parse_items('{"teaser": 1}') == []

    def test_repeats_and_used_titles_are_dropped(self):
        other = {**self.GOOD, "title": "Perros y Sudor!"}
        assert cur.parse_items(json.dumps([self.GOOD, other])) == cur.parse_items(
            json.dumps([self.GOOD]))
        assert cur.parse_items(json.dumps([self.GOOD]), used={"perros y sudor"}) == []

    def test_prompt_mentions_season_and_avoids_used(self):
        prompt = cur.generation_prompt(5, date(2026, 1, 10), ["Berenjena"])
        assert "verano" in prompt and "Berenjena" in prompt
        assert cur.season(date(2026, 7, 1)) == "invierno"


# --------------------------------------------------------------------------- #
# Wikipedia
# --------------------------------------------------------------------------- #

LONG = ("El ornitorrinco es un mamífero semiacuático endémico del este de Australia. "
        + "Es uno de los pocos mamíferos que ponen huevos. " * 8)


def summary(title, extract=LONG, description="", type_="standard"):
    return {"type": type_, "title": title, "extract": extract, "description": description,
            "content_urls": {"desktop": {"page": f"https://es.wikipedia.org/wiki/{title}"}}}


def fake_wiki(feed=None, randoms=()):
    randoms = list(randoms)

    def fetch(url):
        if "/feed/featured/" in url:
            if feed is None:
                raise OSError("404")
            return feed
        if "/page/random/summary" in url:
            if not randoms:
                raise OSError("no more")
            return randoms.pop(0)
        raise AssertionError(url)

    return fetch


FEED = {
    "onthisday": [{"text": "Se inaugura el Teatro Colón de Buenos Aires", "year": 1908,
                   "pages": [summary("Teatro Colón")]}],
    "tfa": summary("Ornitorrinco"),
    "mostread": {"articles": [summary("Villa Chica", description="localidad de Salta")]},
}


class TestWikipedia:
    def test_featured_feed_first(self):
        items = cur.wikipedia_batch(5, TODAY, fetch=fake_wiki(FEED))
        titles = [c.title for c in items]
        assert titles[:2] == ["Teatro Colón", "Ornitorrinco"]
        assert "Villa Chica" not in titles  # small place: a stub
        day = items[0]
        assert day.day == TODAY.isoformat()
        assert "un día como hoy" in day.body.lower() or "Un día como hoy" in day.body
        assert all(c.source == "wikipedia" and c.url for c in items)

    def test_on_this_day_teaser_is_the_fact_not_invented(self):
        entry = {"text": "Nace X", "year": 1900, "pages": [summary("X")]}
        got = cur.onthisday_item(entry, TODAY)
        assert got.teaser == "¿Sabías que un día como hoy, en 1900, nace X?"

    def test_on_this_day_keeps_names_capitalised_and_drops_asides(self):
        entry = {"text": "Henry Kissinger recibe el Nobel (compartido)", "year": 1973,
                 "pages": [{**summary("x"), "title": "Henry_Kissinger"}]}
        got = cur.onthisday_item(entry, TODAY)
        assert got.teaser == "¿Sabías que un día como hoy, en 1973, Henry Kissinger recibe el Nobel?"
        assert got.title == "Henry Kissinger"
        muere = {"text": "Muere María Antonieta", "year": 1793, "pages": [summary("x")]}
        assert "1793, muere María" in cur.onthisday_item(muere, TODAY).teaser

    def test_long_teasers_are_shortened_or_rewritten(self):
        long_lead = ("El ornitorrinco es un mamífero semiacuático endémico del este de "
                     "Australia y de la isla de Tasmania, famoso por su pico. " + LONG)
        summary_ = lambda t: summary(t, extract=long_lead)  # noqa: E731
        plain = cur.summary_item(summary_("Ornitorrinco"))
        assert len(plain.teaser) <= cur.TEASER_TARGET and plain.teaser.endswith("…")
        assert long_lead.startswith(plain.teaser[:-1])
        rewritten = cur.summary_item(summary_("Ornitorrinco"),
                                     rewrite=lambda p: "¿Sabías que el ornitorrinco pone huevos?")
        assert rewritten.teaser == "¿Sabías que el ornitorrinco pone huevos?"
        bad = cur.summary_item(summary_("Ornitorrinco"), rewrite=lambda p: "x" * 300)
        assert bad.teaser == plain.teaser

    def test_random_summaries_skip_stubs(self):
        randoms = [summary("Corto", extract="Muy corto."),
                   summary("Desamb", type_="disambiguation"),
                   summary("San Pedro", description="municipio de Jujuy"),
                   summary("Yerba mate")]
        items = cur.wikipedia_batch(1, TODAY, fetch=fake_wiki(None, randoms))
        assert [c.title for c in items] == ["Yerba mate"]

    def test_no_network_raises_for_the_fallback(self):
        def boom(url):
            raise OSError("sin red")

        with pytest.raises(OSError):
            cur.wikipedia_batch(3, TODAY, fetch=boom)

    def test_used_titles_are_skipped(self):
        items = cur.wikipedia_batch(5, TODAY, used={"teatro colon"}, fetch=fake_wiki(FEED))
        assert "Teatro Colón" not in [c.title for c in items]


# --------------------------------------------------------------------------- #
# "Ver más…"
# --------------------------------------------------------------------------- #


class FakeBoard:
    def __init__(self):
        self.shown = []

    def show(self, title, content, format="markdown"):
        from wawi.board import render_page

        self.shown.append(render_page(title, content, format))


def card(data_uri="data:image/png;base64,AAAA"):
    return SimpleNamespace(title="Foto", description="", source_url="https://commons.example/f",
                           credit="Fulano", license="CC BY-SA 4.0", data_uri=data_uri)


class TestVerMas:
    def test_opens_the_article_with_the_image_card(self, tmp_path, caplog):
        board = FakeBoard()
        svc = service(tmp_path, [item(image_query="berenjena", source="wikipedia",
                                      url="https://es.wikipedia.org/wiki/Berenjena")],
                      board=board, find_image=lambda q: card())
        svc.poll(10.0, NOON, **IDLE)
        with caplog.at_level("INFO", logger="wawi.curiosities"):
            assert svc.more() is True
        page = board.shown[-1]
        assert "image-card" in page.html and "Fulano" in page.html
        assert "Fuente: Wikipedia" in page.html and "cosecha" in page.html
        assert svc.bubble_text(11.0) is None  # the bubble goes away
        assert any("ver más" in r.message for r in caplog.records)

    def test_plain_markdown_without_an_image(self, tmp_path):
        board = FakeBoard()

        def no_image(q):
            raise OSError("nada")

        svc = service(tmp_path, [item(image_query="berenjena")], board=board, find_image=no_image)
        svc.poll(10.0, NOON, **IDLE)
        svc.more()
        page = board.shown[-1]
        assert page.format == "markdown" and "<h1>Berenjena</h1>" in page.html
        assert "<article class='image-card'>" not in page.html

    def test_generated_html_is_escaped(self):
        evil = Curiosity(teaser="¿Sabías?", title="<b>t</b>",
                         body="Hola <script>alert(1)</script>\n\nchau")
        content, fmt = cur.article_html(evil, card())
        assert fmt == "html" and "<script>" not in content and "<b>t</b>" not in content

    def test_unsafe_image_uri_falls_back_to_markdown(self):
        _, fmt = cur.article_html(item(), card("javascript:alert(1)"))
        assert fmt == "markdown"

    def test_nothing_to_open_without_a_curiosity(self, tmp_path):
        assert service(tmp_path).more() is False


# --------------------------------------------------------------------------- #
# The brain: the user turn, never the system prompt
# --------------------------------------------------------------------------- #


class TestBrainNote:
    def test_note_once_while_recent(self, tmp_path):
        svc = service(tmp_path, [item()])
        assert svc.with_note("hola", now=0.0) == "hola"
        svc.poll(10.0, NOON, **IDLE)
        noted = svc.with_note("contame más", now=20.0)
        assert noted.endswith("contame más") and "Berenjena" in noted
        assert svc.with_note("y otra cosa", now=30.0) == "y otra cosa"  # only once

    def test_no_note_once_it_is_old(self, tmp_path):
        svc = service(tmp_path, [item()], cfg={"recent_min": 20})
        svc.poll(10.0, NOON, **IDLE)
        assert svc.with_note("contame más", now=10.0 + 21 * 60) == "contame más"

    def test_module_helper_never_raises(self, monkeypatch):
        monkeypatch.setattr(cur, "_SERVICE", None)
        assert cur.with_note("hola") == "hola"

    def test_curiosities_do_not_change_the_system_prompt(self, tmp_path, monkeypatch):
        """Two curiosities, two turns, ONE live connection: the note rides in
        the user turn, so the client is never rebuilt (prompt cache kept)."""
        import claude_agent_sdk
        from wawi import brain
        from tests.test_brain import _FakeClient, _engine_ask

        _FakeClient.instances.clear()
        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClient)
        svc = service(tmp_path, [item("A", teaser="¿Sabías que A?"),
                                 item("B", teaser="¿Sabías que B?")])
        engine = brain.Engine()
        try:
            svc.poll(10.0, NOON, **IDLE)
            _engine_ask(engine, svc.with_note("contame más", now=11.0))
            svc.dismiss()
            svc.poll(10.0 + 70 * 60, NOON, **IDLE)
            _engine_ask(engine, svc.with_note("¿y eso?", now=10.0 + 70 * 60))
        finally:
            engine.close()
        assert len(_FakeClient.instances) == 1
        prompts = _FakeClient.instances[0].prompts
        assert "¿Sabías que A?" in prompts[0] and "¿Sabías que B?" in prompts[1]


def test_config_section_defaults_and_local_file(tmp_path):
    from wawi import config

    path = tmp_path / "config.yaml"
    path.write_text("language: es\n", encoding="utf-8")
    cfg = config.load(path)
    s = CuriositySettings.from_dict(cfg.curiosities)
    assert s.enabled and s.every_min == 60.0 and s.model == "claude-haiku-5-5"
    assert s.local_file == str(tmp_path / "curiosities_local.json")


class TestPanelHooks:
    """The panel's side: menu switch and the click on "Ver más…"."""

    def _panel(self, monkeypatch, svc):
        from wawi.panel import PanelObserver

        monkeypatch.setattr(cur, "_SERVICE", svc)
        return PanelObserver(enabled=False)

    def test_click_on_the_link_line_opens_the_article(self, tmp_path, monkeypatch):
        opened = []
        svc = service(tmp_path, [item()])
        svc.poll(10.0, NOON, **IDLE)
        monkeypatch.setattr(svc, "more", lambda: opened.append(True) or True)
        panel = self._panel(monkeypatch, svc)
        panel._bubble_text = svc.bubble_text(11.0)
        panel._bubble_clicked((30.0, 20.0))
        assert opened == [True]

    def test_click_elsewhere_just_dismisses(self, tmp_path, monkeypatch):
        svc = service(tmp_path, [item()])
        svc.poll(10.0, NOON, **IDLE)
        panel = self._panel(monkeypatch, svc)
        panel._bubble_text = svc.bubble_text(11.0)
        panel._bubble_clicked((30.0, 70.0))
        assert svc.bubble_text(12.0) is None

    def test_menu_switch_flips_and_persists(self, tmp_path, monkeypatch):
        svc = service(tmp_path)
        panel = self._panel(monkeypatch, svc)
        panel._handle_curiosities_click()
        assert svc.enabled is False
        assert json.loads((tmp_path / "c.json").read_text())["enabled"] is False

    def test_link_text_matches(self):
        from wawi import panel

        assert panel._CURIOSITY_LINK == cur.LINK
