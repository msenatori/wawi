"""Tests for things dragged onto Wawi (wawi/drops.py and its wiring).

No AppKit drag happens here: the pasteboard is a fake with the same four
questions the real adapter answers (panel._AppKitPasteboard), the brain is
:func:`wawi.brain.ask_sync` or the SDK seam replaced, and the orchestrator is
built without ``__init__`` like the rest of the suite.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PIL import Image

from wawi import brain, drops, mascot, tools
from wawi import orchestrator as orch_mod
from wawi.notes import NoteFile, NoteSession
from wawi.state import State
from tests.test_brain import _FakeClient, _engine_ask, _result, _assistant, fake_client  # noqa: F401
from tests.test_orchestrator import AUDIO, Recorder, make_turn_orch


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakePasteboard:
    def __init__(self, paths=(), strings=None, data=None):
        self._paths = list(paths)
        self._strings = dict(strings or {})
        self._data = dict(data or {})

    def types(self):
        out = [drops.FILE_URL] if self._paths else []
        return out + list(self._strings) + list(self._data)

    def file_paths(self):
        return list(self._paths)

    def string(self, kind):
        return self._strings.get(kind)

    def data(self, kind):
        return self._data.get(kind)


def png_bytes(size=(40, 30), mode="RGBA") -> bytes:
    import io

    buf = io.BytesIO()
    Image.new(mode, size, (200, 10, 10, 255) if mode == "RGBA" else (200, 10, 10)).save(
        buf, format="PNG")
    return buf.getvalue()


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# --------------------------------------------------------------------------- #
# Pasteboard -> items
# --------------------------------------------------------------------------- #


class TestClassify:
    def test_files_and_folders_by_kind_as_one_list(self, tmp_path):
        names = ["foto.PNG", "informe.pdf", "nota.md", "datos.csv", "app.py",
                 "voz.m4a", "cosas.zip", "raro.xyz"]
        for n in names:
            (tmp_path / n).write_bytes(b"x" * 10)
        (tmp_path / "carpeta").mkdir()
        paths = [str(tmp_path / n) for n in names] + [str(tmp_path / "carpeta")]
        items = drops.classify(FakePasteboard(paths, strings={drops.STRING: "ignored"}))
        assert [i.kind for i in items] == ["image", "pdf", "doc", "sheet", "code",
                                           "audio", "archive", "unknown", "folder"]
        assert items[0].size == 10 and items[0].name == "foto.PNG"
        assert items[-1].size is None

    def test_a_browser_link(self):
        items = drops.classify(FakePasteboard(strings={
            drops.URL: "https://example.com/a", drops.STRING: "Example"}))
        assert [(i.kind, i.text) for i in items] == [("url", "https://example.com/a")]

    def test_text_that_is_only_a_link_is_a_link(self):
        items = drops.classify(FakePasteboard(strings={drops.STRING: "  https://x.org/b \n"}))
        assert items[0].kind == "url" and items[0].text == "https://x.org/b"

    def test_plain_text(self):
        items = drops.classify(FakePasteboard(strings={drops.STRING: "hola, mirá esto"}))
        assert [(i.kind, i.text) for i in items] == [("text", "hola, mirá esto")]

    def test_a_picture_without_a_file_is_saved(self, tmp_path):
        items = drops.classify(FakePasteboard(
            strings={drops.URL: "https://img.example/x.png"},
            data={drops.PNG: png_bytes()}), picture_dir=tmp_path)
        (item,) = items
        assert item.kind == "image" and item.source == "https://img.example/x.png"
        assert Image.open(item.path).size == (40, 30)

    def test_tiff_becomes_png(self, tmp_path):
        import io

        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="TIFF")
        (item,) = drops.classify(FakePasteboard(data={drops.TIFF: buf.getvalue()}),
                                 picture_dir=tmp_path)
        assert item.path.endswith(".png") and Image.open(item.path).format == "PNG"

    def test_nothing_she_understands(self):
        assert drops.classify(FakePasteboard(strings={drops.STRING: "   "})) == []
        assert drops.classify(FakePasteboard()) == []

    def test_file_urls_to_paths(self):
        assert drops.paths_from_file_urls(["file:///a%20b/c.txt", "https://x"]) == ["/a b/c.txt"]


class TestLines:
    @pytest.mark.parametrize("kind,line", [
        ("image", "¡Una foto! ¿Qué hago?"), ("pdf", "Uh, un papel largo. ¿Qué hago?"),
        ("doc", "Uh, un papel largo. ¿Qué hago?"), ("url", "¿Voy a ver?"),
        ("text", "Lo tengo."), ("audio", "¿Lo paso a texto?"), ("sheet", "Muchos números…"),
        ("code", "Huele a código."), ("folder", "¡Cuántas cosas!"), ("archive", "Huele raro…"),
        ("unknown", "Huele raro…"),
    ])
    def test_one_line_per_kind(self, kind, line):
        assert drops.catch_line([drops.DropItem(kind=kind)]) == line

    def test_several(self):
        two = [drops.DropItem(kind="image"), drops.DropItem(kind="image")]
        assert drops.catch_line(two) == "¡Fotos! ¿Qué hago?"
        mixed = [drops.DropItem(kind="image"), drops.DropItem(kind="pdf")]
        assert drops.catch_line(mixed) == drops.LINES["many"]

    def test_the_log_line_has_no_contents(self):
        items = [drops.DropItem(kind="text", text="mi clave secreta"),
                 drops.DropItem(kind="pdf", name="x.pdf", path="/x.pdf", size=2048)]
        line = drops.summary(items)
        assert "secreta" not in line and "x.pdf" not in line
        assert "2 item(s)" in line and "2 KB" in line and "16 chars" in line


# --------------------------------------------------------------------------- #
# Context for the brain
# --------------------------------------------------------------------------- #


class TestContext:
    def test_an_image_is_attached_not_inlined(self, tmp_path):
        path = tmp_path / "foto.png"
        path.write_bytes(png_bytes((2000, 1000)))
        prefix, images = drops.Drop([drops.item_for_path(str(path))]).context()
        assert prefix.startswith(f"[Te pasé: imagen {path} (foto.png, ")
        assert "la imagen va adjunta" in prefix
        (img,) = images
        assert isinstance(img, brain.ImageAttachment) and img.media_type == "image/jpeg"
        assert img.note == "[imagen que te pasó: foto.png, 1280x640]"  # resized like a screenshot
        assert img.data not in prefix

    def test_long_text_is_cut(self):
        text = "palabra " * 3000  # ~24 KB
        prefix, images = drops.Drop([drops.DropItem(kind="text", text=text)]).context()
        assert images == ()
        assert "(recortado)" in prefix
        assert len(prefix.encode()) < 8 * 1024 + 300

    def test_a_small_file_is_excerpted_and_a_big_one_is_not(self, tmp_path):
        small = tmp_path / "app.py"
        small.write_text("def hola():\n    return 'chau'\n")
        big = tmp_path / "big.csv"
        big.write_text("a,b\n" * 60000)  # > 200 KB
        prefix, _ = drops.Drop([drops.item_for_path(str(small)),
                                drops.item_for_path(str(big))]).context()
        assert prefix.startswith("[Te pasé 2 cosas:\n- código ")
        assert "def hola():" in prefix
        assert "a,b" not in prefix  # listed by name only
        assert f"- planilla {big}" in prefix

    def test_audio_is_transcribed_locally(self, tmp_path):
        import soundfile as sf

        path = tmp_path / "voz.wav"
        sf.write(path, np.zeros(8000, dtype=np.float32), 8000)  # 1 s at 8 kHz
        heard = []

        def stt(audio):
            heard.append(audio)
            return "comprar pan"

        drop = drops.Drop([drops.item_for_path(str(path))])
        prefix, _ = drop.context(stt)
        assert "Transcripción de voz.wav" in prefix and "comprar pan" in prefix
        assert heard[0].dtype == np.float32 and heard[0].size == 16000  # resampled
        drop.context(stt)
        assert len(heard) == 1  # read once, kept for follow-ups

    def test_an_unreadable_pdf_says_only_its_name(self, tmp_path):
        path = tmp_path / "roto.pdf"
        path.write_bytes(b"not a pdf")
        prefix, _ = drops.Drop([drops.item_for_path(str(path))]).context()
        assert "solo ves su nombre" in prefix

    def test_a_link_goes_inline(self):
        prefix, _ = drops.Drop([drops.DropItem(kind="url", text="https://x.org")]).context()
        assert prefix == "[Te pasé: link https://x.org]\n"

    def test_settings_from_config(self):
        s = drops.DropSettings.from_config({"window_s": 20, "hold_s": "x", "max_text_kb": 4})
        assert (s.window_s, s.hold_s, s.max_text) == (20.0, drops.HOLD_S, 4096)


class TestBrainImages:
    def test_one_shot_turn_sends_image_blocks_and_reports_the_note(self, monkeypatch):
        seen = {}

        async def fake(prompt, options):  # noqa: ARG001
            msgs = [m async for m in prompt]
            seen["msgs"] = msgs
            yield _assistant("Es un perro.")
            yield _result()

        monkeypatch.setattr(brain, "_sdk_query", fake)
        img = brain.ImageAttachment(data="QUJD", note="[imagen que te pasó: x.png, 1x1]")
        res = brain.ask_sync("[Te pasé: imagen /x.png] ¿qué es?", model="m", images=(img,))
        (msg,) = seen["msgs"]
        content = msg["message"]["content"]
        assert content[0]["type"] == "image" and content[0]["source"]["data"] == "QUJD"
        assert content[1] == {"type": "text", "text": "[Te pasé: imagen /x.png] ¿qué es?"}
        assert res.screen_notes == ("[imagen que te pasó: x.png, 1x1]",)

    def test_the_live_engine_rotates_after_an_image(self, fake_client):  # noqa: F811
        engine = brain.Engine()
        try:
            img = brain.ImageAttachment(data="QUJD", note="[imagen que te pasó: x.png, 1x1]")
            res = _engine_ask(engine, "¿qué es?", images=(img,))
            assert res.screen_notes == (img.note,)
            assert engine._stale  # the image dies with this client
        finally:
            engine.close()
        prompt = fake_client.instances[0].prompts[0]
        assert not isinstance(prompt, str)  # content blocks, not a string


# --------------------------------------------------------------------------- #
# The orchestrator: window, microphone, order, hold, notes, refusal
# --------------------------------------------------------------------------- #


class DropListener:
    def __init__(self, require_phrase=False):
        self.require_phrase = require_phrase
        self.wake = []
        self.follow_ups = []
        self.is_capturing = False
        self.follow_up_open = False
        self.echo = None

    def set_wake_enabled(self, on):
        self.wake.append(bool(on))

    def arm_follow_up(self, seconds):
        self.follow_ups.append(seconds)

    def set_echo_text(self, text):
        self.echo = text

    def set_speech_floor(self, level):
        pass

    def pause(self):
        pass

    def resume(self):
        pass


class DropPanel(Recorder):
    def __init__(self):
        super().__init__()
        self.bubbles = []
        self.caught = []
        self._board = None

    def drop_bubble(self, text):
        self.bubbles.append(text)

    def drop_caught(self, line):
        self.caught.append(line)

    def banner(self, text):
        pass


def make_drop_orch(monkeypatch, *, window=0.3, hold=60.0, muted=False, require_phrase=False):
    from tests.test_orchestrator import StreamCfg

    class Cfg(StreamCfg):
        tts_streaming = False
        drops = {"window_s": window, "hold_s": hold}

    o = make_turn_orch(Cfg())
    o.state.set_state(State.IDLE)
    o._turn_lock.release()
    o.state.subscribe(o._on_state_change)
    o.listener = DropListener(require_phrase)
    o.panel = DropPanel()
    o.printer = DropPanel()
    o._listening = not muted
    o._drop = None
    o._drop_unmuted = False
    o._drop_lock = threading.Lock()
    o._notes = None
    o.inflight = orch_mod.Inflight(None)
    monkeypatch.setattr(orch_mod, "_DROP_CATCH_PAUSE_S", 0.0)
    monkeypatch.setattr(orch_mod.Orchestrator, "_clear_echo_after_tail", lambda self: None)
    return o


@pytest.fixture
def brain_calls(monkeypatch):
    calls = []

    def fake(user_text, **kw):
        calls.append((user_text, kw.get("images", ())))
        return brain.BrainResult(text="Listo.", tool_calls=(), session_id="s1")

    monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
    return calls


def _order(o, text):
    """The owner says *text* in the window: a turn, as the listener makes it."""
    o.stt = type("T", (), {"transcribe": staticmethod(lambda *a, **kw: text)})()
    assert o._turn_lock.acquire(timeout=2.0)
    o.state.set_state(State.THINKING)
    o._do_turn(AUDIO)


class TestWindow:
    def test_she_catches_says_it_and_opens_a_long_window(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch, window=0.2)
        o._handle_drop([drops.DropItem(kind="url", text="https://x.org")])
        assert o.panel.caught == ["¿Voy a ver?"]
        assert o.voice.said == ["¿Voy a ver?"]
        assert o.listener.follow_ups == [0.2]  # the drop window, not the usual one
        # Expired with nobody talking: dropped quietly, no brain call.
        assert o.panel.bubbles == [drops.LEFT_TEXT]
        assert brain_calls == []
        assert o._drop is not None and not o._drop.awaiting  # still held a while

    def test_wake_phrase_mode_needs_no_wake_word(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch, window=0.1, require_phrase=True)
        o._handle_drop([drops.DropItem(kind="text", text="hola")])
        # The follow-up window is what never needs the phrase; the mode
        # itself is left alone.
        assert o.listener.follow_ups == [0.1]
        assert o.listener.require_phrase is True

    def test_a_muted_mic_is_on_for_the_window_and_back_off_after(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch, window=0.1, muted=True)
        o._handle_drop([drops.DropItem(kind="text", text="hola")])
        assert o.listener.wake == [True, False]
        assert o._listening is False and o._drop_unmuted is False
        assert brain_calls == []

    def test_a_muted_mic_comes_back_off_after_the_order(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch, window=5.0, muted=True)
        t = threading.Thread(target=o._handle_drop,
                             args=([drops.DropItem(kind="text", text="hola")],))
        t.start()
        assert wait_for(lambda: o.listener.follow_ups)
        assert o.listener.wake == [True]
        _order(o, "traducilo")
        t.join(2.0)
        assert not t.is_alive()  # the window ended with the order
        assert o.listener.wake == [True, False]
        assert len(brain_calls) == 1

    def test_the_owner_switching_the_mic_wins(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch, window=0.1, muted=True)
        o._listening = False
        o._drop_unmuted = True
        o._on_listening_toggled(True)
        o._restore_drop_mic()
        assert o.listener.wake == [True]  # nothing put back over the owner's click

    def test_in_a_call_she_only_shows_it(self, monkeypatch, brain_calls):
        from wawi import reflexes

        monkeypatch.setattr(reflexes, "is_quiet", lambda: True)
        o = make_drop_orch(monkeypatch, window=0.1)
        o._handle_drop([drops.DropItem(kind="text", text="hola")])
        assert o.voice.said == []
        assert o.panel.bubbles[0] == "Lo tengo." and o.listener.follow_ups == [0.1]


class TestOrder:
    def test_the_order_carries_the_item_in_the_user_turn(self, monkeypatch, brain_calls,
                                                         tmp_path):
        path = tmp_path / "foto.png"
        path.write_bytes(png_bytes())
        o = make_drop_orch(monkeypatch, window=5.0)
        t = threading.Thread(target=o._handle_drop, args=([drops.item_for_path(str(path))],))
        t.start()
        assert wait_for(lambda: o.listener.follow_ups)
        _order(o, "describila")
        t.join(2.0)
        (text, images), = brain_calls
        assert text.startswith(f"[Te pasé: imagen {path} (foto.png, ")
        assert text.endswith("]\ndescribila")
        assert len(images) == 1 and images[0].note.startswith("[imagen que te pasó: foto.png")

    def test_follow_ups_keep_it_until_she_buries_it(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch, window=0.05, hold=0.5)
        o._handle_drop([drops.DropItem(kind="text", text="good morning")])
        _order(o, "traducilo")
        _order(o, "y ahora en francés")
        assert all("good morning" in text for text, _ in brain_calls)
        time.sleep(0.6)
        _order(o, "¿qué hora es?")
        assert brain_calls[-1][0] == "¿qué hora es?"
        assert o._drop is None

    def test_nothing_is_sent_without_an_order(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch, window=0.05)
        o._handle_drop([drops.DropItem(kind="text", text="secreto")])
        time.sleep(0.1)
        assert brain_calls == []

    def test_a_later_drop_replaces_the_first(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch, window=0.05)
        o._handle_drop([drops.DropItem(kind="text", text="uno")])
        o._handle_drop([drops.DropItem(kind="text", text="dos")])
        _order(o, "leelo")
        assert "dos" in brain_calls[0][0] and "uno" not in brain_calls[0][0]


class TestBusy:
    def test_mid_turn_she_refuses(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch)
        o._turn_lock.acquire()
        o.state.set_state(State.THINKING)
        o._handle_drop([drops.DropItem(kind="text", text="hola")])
        assert o.panel.bubbles == [drops.BUSY_TEXT]
        assert o._drop is None and o.voice.said == [] and o.panel.caught == []

    def test_while_taking_notes_it_goes_into_the_note(self, monkeypatch, brain_calls, tmp_path):
        o = make_drop_orch(monkeypatch)
        note = NoteFile(tmp_path)
        o._notes = NoteSession(note)
        o._handle_drop([drops.DropItem(kind="pdf", name="acta.pdf", path="/a/acta.pdf"),
                        drops.DropItem(kind="text", text="punto tres")])
        content = note.read()
        assert "[archivo: acta.pdf]" in content and "[texto] punto tres" in content
        assert o.panel.bubbles == [drops.NOTED_TEXT]
        assert o._drop is None and o.voice.said == [] and brain_calls == []


# --------------------------------------------------------------------------- #
# The panel half and the abilities list
# --------------------------------------------------------------------------- #


class TestPanel:
    def _panel(self):
        from wawi.panel import PanelObserver

        p = PanelObserver(enabled=False)
        p._mascot_animator.set_available(mascot.FRAME_NAMES)
        return p

    def test_a_drop_reaches_the_callback(self, tmp_path):
        p = self._panel()
        got = []
        p.on_dropped(got.append)
        (tmp_path / "a.txt").write_text("x")
        assert p._drag_dropped(FakePasteboard([str(tmp_path / "a.txt")])) is True
        assert [i.kind for i in got[0]] == ["doc"]
        assert p._drag_dropped(FakePasteboard()) is False  # nothing she understands

    def test_ready_pose_while_hovering_then_the_catch(self):
        p = self._panel()
        anim = p._mascot_animator
        p._drag_hover(True)
        p._tick_catch()
        assert anim.act_name == "catch_ready"
        p._drag_hover(False)
        p._tick_catch()
        assert anim.act_name is None
        p.drop_caught("Lo tengo.")
        p._tick_catch()
        assert anim.act_name == "catch"
        assert p._drop_bubble() == "Lo tengo."

    def test_the_catch_frames_have_fallbacks(self):
        for name in mascot.CATCH_FRAMES:
            assert name in mascot.FRAME_NAMES and name in mascot.FRAME_FALLBACK


def test_abilities_list_it(tmp_path):
    from wawi.board import Board

    old = tools._BOARD
    try:
        tools.set_board(Board(directory=tmp_path))
        rows = [row for _, items in tools.ability_groups() for row in items]
    finally:
        tools.set_board(old)
    assert ("drop_item", "Tirale un archivo, foto o link encima y decile qué hacer",
            "Wawi, resumime esto (después de tirárselo encima)") in rows
