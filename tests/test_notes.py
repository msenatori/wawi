"""Taking notes (wawi/notes.py): the phrases, the file, and the routing.

The orchestrator is built without ``__init__`` (no Whisper, no mic, no
Kokoro), like tests/test_orchestrator.py: what is pinned down here is that a
note session never reaches the brain, never opens a follow-up window, never
lets her talk over it, and gives the listener back exactly as it found it.
"""

from __future__ import annotations

import random
import threading
import time
from datetime import datetime

import numpy as np
import pytest

from wawi import mascot, notes, notes_ui
from wawi import orchestrator as orch_mod
from wawi.notes import NoteFile, NoteSession, parse_start, parse_stop
from wawi.orchestrator import Orchestrator
from wawi.state import State, StateMachine

SPEECH = (np.sin(np.linspace(0, 400, 16000)) * 0.3).astype(np.float32)


class TestPhrases:
    @pytest.mark.parametrize("text", [
        "Tomá nota.", "tomá notas", "Tomar nota", "Anotá esto.", "anotá todo",
        "Modo nota", "modo notas", "Grabá la charla.", "grabá esto",
        "Empezá a grabar.", "Dale, tomá nota por favor.", "¿Podés tomar nota?",
        "tomá nota de la reunión",
    ])
    def test_start(self, text):
        assert parse_start(text)

    @pytest.mark.parametrize("text", [
        "", "¿cómo tomo nota en Notion?", "anotá esto: comprar pan y leche",
        "tomá nota de que mañana llamo al dentista a las cinco",
        "qué hora es", "grabá un audio para mamá",
    ])
    def test_not_start(self, text):
        assert not parse_start(text)

    @pytest.mark.parametrize("text", [
        "Pará.", "pará de anotar", "Listo.", "Terminá.", "Basta.", "Basta, gracias.",
        "dejá de grabar", "terminamos", "stop",
    ])
    def test_stop_alone(self, text):
        assert parse_stop(text)
        assert parse_stop(text, named=True)

    def test_with_the_name_a_clear_command_may_run_on(self):
        assert parse_stop("listo por hoy", named=True)
        assert parse_stop("pará de anotar ya mismo", named=True)
        # ...but without it, a sentence that starts with "listo" is dictation.
        assert not parse_stop("listo por hoy")

    @pytest.mark.parametrize("text", [
        "para mañana tengo que llamar a Juan", "listo el informe, falta la tabla",
        "", "pará que te cuento algo largo de verdad",
        # Said in passing mid-dictation, it ended a real note at 15 s.
        "Ya está.",
    ])
    def test_dictation_is_not_a_stop(self, text):
        assert not parse_stop(text)

    def test_the_ambiguous_para_never_stops_a_sentence_even_named(self):
        assert not parse_stop("para mañana tengo que llamar", named=True)


class TestNoteFile:
    def test_title_lines_and_close(self, tmp_path):
        note = NoteFile(tmp_path / "Notas", now=datetime(2026, 10, 16, 14, 5))
        assert note.path.name == "2026-10-16 14-05.md"
        assert note.append("Primera idea.", when=datetime(2026, 10, 16, 14, 5, 9))
        # Written as it goes: readable before the note is closed.
        assert "- **14:05:09** Primera idea." in note.path.read_text()
        assert not note.append("   ")
        note.append("  dos   espacios ", when=datetime(2026, 10, 16, 14, 6, 0))
        note.close(when=datetime(2026, 10, 16, 14, 7, 0))
        text = note.read()
        assert text.startswith("# Nota del 16/10/2026 a las 14:05\n")
        assert "- **14:06:00** dos espacios\n" in text
        assert "Terminada a las 14:07:00" in text
        assert note.lines == 2 and note.closed
        assert not note.append("tarde")
        note.close()  # twice is harmless

    def test_same_minute_never_overwrites(self, tmp_path):
        when = datetime(2026, 10, 16, 9, 0)
        first = NoteFile(tmp_path, now=when)
        second = NoteFile(tmp_path, now=when)
        third = NoteFile(tmp_path, now=when)
        assert [p.path.name for p in (first, second, third)] == [
            "2026-10-16 09-00.md", "2026-10-16 09-00 (2).md", "2026-10-16 09-00 (3).md"]

    def test_an_empty_note_says_so(self, tmp_path):
        note = NoteFile(tmp_path)
        note.close()
        assert "No se escuchó nada" in note.read()


class TestStopCard:
    def test_it_hides_even_when_the_click_ended_the_note_before_a_redraw(self):
        class Window:
            visible = True

            def isVisible(self):
                return self.visible

            def orderOut_(self, _sender):
                self.visible = False

        button = notes_ui.StopButton()
        button.window = Window()
        button._clicked()  # resets _drawn to repaint pressed
        button.tick(None)
        assert button.window.visible is False

    def test_label(self):
        assert notes_ui.label(192, 5) == "● 03:12 · 5 líneas"
        assert notes_ui.label(3725, 1) == "● 1:02:05 · 1 línea"

    def test_place_beside_her_and_on_screen(self):
        screen = (0.0, 0.0, 1440.0, 900.0)
        x, y = notes_ui.place((100.0, 100.0, 200.0, 200.0), screen)
        assert x == 100 + 200 + notes_ui.GAP
        # At the right edge it goes to her left instead.
        x, _ = notes_ui.place((1300.0, 100.0, 130.0, 200.0), screen)
        assert x + notes_ui.WIDTH <= 1300
        # No anchor: bottom-right of the screen, inside it.
        x, y = notes_ui.place(None, screen)
        assert 0 <= x <= 1440 - notes_ui.WIDTH and 0 <= y


# ---------------------------------------------------------------------------
# Orchestrator routing
# ---------------------------------------------------------------------------

class Cfg:
    conversation_mode = True
    wake_follow_up_s = 20.0
    wake_follow_up_idle_s = 0.0
    wake_phrases = ("wawi",)
    barge_in = False
    sample_rate = 16000
    assistant_name = "Wawi"
    stt_filter_junk = False

    def __init__(self, directory, idle_min=30):
        self.notes = {"dir": str(directory), "idle_stop_min": idle_min}


class Listener:
    def __init__(self, require_phrase=True):
        self.require_phrase = require_phrase
        self.modes = []
        self.echo = ""

    def set_require_phrase(self, on):
        self.modes.append(on)
        self.require_phrase = on
        return True

    def set_dictation(self, on):
        self.dictation = on

    def set_wake_enabled(self, on):
        pass

    def set_echo_text(self, text):
        self.echo = text

    def set_speech_floor(self, level):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def arm_follow_up(self, seconds):
        raise AssertionError("no follow-up window around a note")


class Board:
    def __init__(self):
        self.shown = []

    def show(self, title, content, format="markdown"):
        self.shown.append((title, content, format))


class Panel:
    def __init__(self):
        self.channel_open = False
        self.session = None
        self.sessions = []
        self.user_texts = []
        self._board = Board()
        self.title_text = "Wawi"
        self.listening = True

    def set_notes_session(self, session):
        self.session = session
        self.sessions.append(session)

    def set_texts(self, **kw):
        if "user_text" in kw:
            self.user_texts.append(kw["user_text"])

    def set_listening(self, on):
        self.listening = on

    def __call__(self, old, new):
        pass


class Printer:
    def banner(self, text):
        pass

    def question(self, text):
        pass

    def answer(self, text):
        pass

    def note(self, text):
        pass

    def set_texts(self, **kw):
        pass


class Voice:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class Stt:
    """Hands back the scripted transcripts, one per utterance."""

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio, **kw):  # noqa: ARG002
        return self.texts.pop(0) if self.texts else ""


def make(tmp_path, texts=(), idle_min=30, require_phrase=True):
    o = object.__new__(Orchestrator)
    o._cfg = Cfg(tmp_path / "Notas", idle_min)
    o._language = "es"
    o._listening = True
    o._last_response = ""
    o._barge_in_armed = False
    o._reaction_speech = False
    o._turn_lock = threading.Lock()
    o._stop_refresh = threading.Event()
    o._pending_utterance = None
    o._pending_lock = threading.Lock()
    o._notes_lock = threading.Lock()
    o.listener = Listener(require_phrase)
    o.panel = Panel()
    o.tui = Printer()
    o.printer = Printer()
    o.voice = Voice()
    o.stt = Stt(texts)
    o._stt_prompt = lambda: ""
    o.state = StateMachine()
    o.state.subscribe(o._on_state_change)
    o.transitions = []
    o.state.subscribe(lambda old, new: o.transitions.append(new))
    return o


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


@pytest.fixture
def no_brain(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("a note line reached the brain")

    monkeypatch.setattr(orch_mod.brain, "ask_sync", boom)
    # Echo-tail clearing spawns a timer thread; not under test here.
    monkeypatch.setattr(Orchestrator, "_clear_echo_after_tail", lambda self: None)


class TestRouting:
    def test_the_phrase_starts_a_note_without_the_model(self, tmp_path, no_brain):
        o = make(tmp_path)
        text, reply = o._local_reply("Wawi, tomá nota.", None)
        assert (text, reply) == ("", notes.START_TEXT)
        session = o._notes
        assert session is not None and o.panel.session is session
        assert o.listener.require_phrase is False  # no wake word while recording
        assert session.note.path.parent == tmp_path / "Notas"
        assert o._start_notes() == notes.BUSY_TEXT  # one session at a time
        o._stop_notes("test", speak=False)
        assert wait_for(lambda: session.note.closed)

    def test_utterances_become_lines_and_never_reach_the_brain(self, tmp_path, no_brain):
        o = make(tmp_path, texts=["Primera idea.", "Wawi.", "Wawi, la segunda.",
                                  "Te escucho, anoto todo.", "Wawi, pará de anotar."])
        o._start_notes()
        session = o._notes
        for _ in range(5):
            o._dispatch_turn(SPEECH, already_locked=False)
        assert wait_for(lambda: session.note.closed)
        text = session.note.read()
        assert "Primera idea." in text and "la segunda." in text
        assert "Wawi, la segunda" not in text  # her name is not part of the line
        assert "anoto todo" not in text  # her own voice is not dictation
        assert "pará" not in text  # nor is the stop command
        assert session.lines == 2
        # Never a turn: no THINKING, and the lock was never taken.
        assert State.THINKING not in o.transitions
        assert not o._turn_lock.locked()

    def test_stop_restores_the_listener_and_shows_the_note(self, tmp_path, no_brain):
        o = make(tmp_path, texts=["Algo para guardar.", "Listo."], require_phrase=True)
        o._start_notes()
        session = o._notes
        o._dispatch_turn(SPEECH, already_locked=False)
        o._dispatch_turn(SPEECH, already_locked=False)
        assert wait_for(lambda: notes.STOP_TEXT in o.voice.said)
        assert o._notes is None and o.panel.session is None
        assert o.listener.modes == [False, True]  # keyword mode stays keyword mode
        (title, content, fmt), = o.panel._board.shown
        assert fmt == "markdown" and "Algo para guardar." in content
        assert str(session.note.path) not in " ".join(o.voice.said)  # never a path out loud

    def test_a_note_is_dictation_while_it_lasts(self, tmp_path, no_brain):
        # Without it the end-of-turn model cut every mid-sentence pause at
        # 300 ms, and the 1 s scraps came back as junk or invented text.
        o = make(tmp_path)
        o._start_notes()
        assert o.listener.dictation is True
        o._stop_notes("test", speak=False)
        assert o.listener.dictation is False

    def test_always_on_stays_always_on(self, tmp_path, no_brain):
        o = make(tmp_path, require_phrase=False)
        o._start_notes()
        o._stop_notes("test", speak=False)
        assert o.listener.require_phrase is False

    def test_the_menu_starts_and_stops(self, tmp_path, no_brain):
        o = make(tmp_path)
        o._on_notes_toggled(True)
        assert wait_for(lambda: notes.START_TEXT in o.voice.said)
        session = o._notes
        assert session is not None
        o._on_notes_toggled(False)
        assert wait_for(lambda: notes.EMPTY_TEXT in o.voice.said)
        assert session.note.closed and o._notes is None

    def test_the_menu_unmutes_to_take_a_note(self, tmp_path, no_brain):
        o = make(tmp_path)
        o._listening = False
        o.panel.listening = False
        o._on_notes_toggled(True)
        assert wait_for(lambda: o._notes is not None)
        assert o._listening and o.panel.listening
        o._stop_notes("test", speak=False)

    def test_muting_ends_the_note(self, tmp_path, no_brain):
        o = make(tmp_path)
        o._start_notes()
        session = o._notes
        o._on_listening_toggled(False)
        assert wait_for(lambda: session.note.closed)
        assert o._notes is None

    def test_silence_ends_the_note(self, tmp_path, no_brain):
        o = make(tmp_path, idle_min=0.5 / 60)  # half a second
        o._start_notes()
        session = o._notes
        assert wait_for(lambda: session.note.closed, timeout=5.0)
        assert o._notes is None

    def test_no_follow_up_and_no_remarks_while_taking_notes(self, tmp_path, no_brain):
        o = make(tmp_path)
        o._start_notes()
        assert o._follow_up_seconds() == 0.0  # even in conversation mode
        assert o._speak_reaction("¡guau!") is False
        assert o._run_routine("algo") is False
        assert o.voice.said == []
        # A capture does not paint LISTENING (no earcon in a meeting).
        o._on_capture_start()
        assert o.state.state is State.IDLE
        o._stop_notes("test", speak=False)

    def test_the_start_confirmation_opens_no_window(self, tmp_path, no_brain):
        # Spoken by the turn, with the session already open: the SPEAKING ->
        # IDLE edge must not arm a follow-up (Listener raises if it does).
        o = make(tmp_path)
        _, reply = o._local_reply("tomá nota", None)
        o._speak(reply)
        o.state.set_state(State.IDLE)
        assert o.panel.channel_open is False
        o._stop_notes("test", speak=False)


class TestPose:
    def anim(self):
        return mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(4))

    def test_she_holds_the_recorder_and_the_light_blinks(self):
        anim = self.anim()
        anim.taking_notes = True
        seen = set()
        for _ in range(int(8 / 0.033)):
            anim.update(0.033, State.IDLE)
            seen.add(anim.frame)
            assert anim.behaviour is None and anim.locomotion is None
        assert {"notes_a", "notes_b"} <= seen
        assert seen <= set(mascot.NOTES_FRAMES)

    def test_a_walk_stops_for_the_note_and_life_resumes_after(self):
        anim = self.anim()
        anim.set_locomotion("walk")
        assert anim.locomotion == "walk"
        anim.taking_notes = True
        anim.update(0.033, State.IDLE)
        assert anim.locomotion is None and anim.frame in mascot.NOTES_FRAMES
        anim.taking_notes = False
        for _ in range(10):
            anim.update(0.033, State.IDLE)
        assert anim.frame not in mascot.NOTES_FRAMES

    def test_she_talks_with_her_mouth_not_the_recorder(self):
        anim = self.anim()
        anim.taking_notes = True
        anim.update(0.033, State.SPEAKING)
        assert anim.frame not in mascot.NOTES_FRAMES

    def test_without_the_art_she_falls_back_to_listening(self):
        frames = [n for n in mascot.FRAME_NAMES if n not in mascot.NOTES_FRAMES]
        anim = mascot.MascotAnimator(available=frames, rng=random.Random(4))
        anim.taking_notes = True
        anim.update(0.033, State.IDLE)
        assert anim.frame == "listening"


class TestMenuItem:
    def test_the_item_names_the_action_and_reports_the_click(self):
        from wawi.panel import PanelObserver

        p = PanelObserver(enabled=False)
        got = []
        p.on_notes_toggled(got.append)
        assert p._notes_menu_title() == "Tomar nota"
        p._handle_notes_click()
        p.set_notes_session(object())
        assert p._notes_menu_title() == "Terminar nota"
        p._handle_notes_click()
        p._stop_notes_clicked()  # the giant STOP
        assert got == [True, False, False]
        p.set_notes_session(None)
        p._stop_notes_clicked()  # nothing to stop
        assert got == [True, False, False]

    def test_hidden_without_a_callback(self):
        from wawi.panel import PanelObserver

        p = PanelObserver(enabled=False)
        p._handle_notes_click()  # no callback: nothing happens, nothing raises


class TestArt:
    def test_the_note_frames_ship_in_the_shared_box(self):
        PIL = pytest.importorskip("PIL.Image")
        sizes = {PIL.open(mascot.frame_path(n)).size for n in mascot.NOTES_FRAMES + ("listening",)}
        assert len(sizes) == 1

    def test_fit_transform_keeps_baseline_centre_and_area(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "make_wawi_assets.py"
        spec = importlib.util.spec_from_file_location("make_wawi_assets", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        scale, x, y = mod.fit_transform((0, 0, 400, 400), (10, 100, 310, 280), (320, 286))
        w = h = round(400 * scale)
        assert w * h == pytest.approx(300 * 180, rel=0.02)
        assert y + h == 280 and x + w / 2 == pytest.approx(160, abs=1)
        # Never larger than the frame.
        scale, x, y = mod.fit_transform((0, 0, 100, 1000), (0, 0, 320, 286), (320, 286))
        assert round(1000 * scale) <= 286 and x >= 0 and y >= 0
