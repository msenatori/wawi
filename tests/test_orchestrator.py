"""Tests for orchestrator turn policy.

Built without running ``Orchestrator.__init__``: that loads Whisper, Kokoro and
the microphone, none of which the decisions under test depend on. The policy is
pure — response text in, seconds out — and that is exactly what should be
pinned down, because getting it wrong is invisible until someone is talking to
a machine that stopped listening.
"""

from __future__ import annotations

import pytest

from wawi.orchestrator import Orchestrator


class FakeCfg:
    conversation_mode = False
    wake_follow_up_s = 20.0
    wake_follow_up_idle_s = 0.0
    barge_in = False
    wake_engine = "whisper"
    wake_phrases = ("wawi", "viernes")


def make_orch(response: str = "", pending=None, cfg=None) -> Orchestrator:
    orch = object.__new__(Orchestrator)
    orch._cfg = cfg or FakeCfg()
    orch._last_response = response
    orch._pending_confirmation = pending
    orch.listener = None
    return orch


class TestFollowUpDecision:
    """Walkie-talkie: hold the channel open only when a reply is expected."""

    @pytest.mark.parametrize(
        "response",
        [
            "¿En cuál proyecto?",
            "Proyecto 6 es tienda. ¿Qué pantalla querés levantar?",
            "No encontré ese proyecto, ¿cuál era?",
        ],
    )
    def test_a_question_keeps_the_channel_open(self, response):
        assert make_orch(response)._follow_up_seconds() == 20.0

    @pytest.mark.parametrize(
        "response",
        [
            "El proyecto 3 terminó.",
            "Listo, anotado.",
            "Hay tres agentes trabajando.",
        ],
    )
    def test_a_statement_closes_the_channel(self, response):
        """Otherwise any nearby speech starts a turn nobody asked for."""
        assert make_orch(response)._follow_up_seconds() == 0.0

    def test_a_held_confirmation_always_waits(self):
        """The action is suspended pending a yes; closing the channel strands it."""
        orch = make_orch("Voy a escribir en la terminal viva de recetas. ¿Lo mando?",
                         pending={"tool": "send_to_agent"})
        assert orch._follow_up_seconds() == 20.0

    def test_an_empty_answer_closes_the_channel(self):
        assert make_orch("")._follow_up_seconds() == 0.0
        assert make_orch("   ")._follow_up_seconds() == 0.0

    def test_an_opening_question_mark_counts(self):
        """Spanish opens questions, and TTS text keeps both marks."""
        assert make_orch("¿Querés que lo corra")._follow_up_seconds() == 20.0

    def test_the_idle_window_is_configurable(self):
        cfg = FakeCfg()
        cfg.wake_follow_up_idle_s = 4.0
        assert make_orch("Terminó.", cfg=cfg)._follow_up_seconds() == 4.0

    def test_a_question_far_from_the_end_does_not_count(self):
        """A question early in a long answer is not an invitation to reply."""
        response = (
            "¿Sabés qué? " + "El agente terminó sin errores y dejó todo listo. " * 4
        )
        assert make_orch(response)._follow_up_seconds() == 0.0


class TestWakePhraseModeWaitsToBeCalled:
    """Keyword mode: "vos tenés que esperar que yo diga Wawi"."""

    class _Listener:
        def __init__(self, required):
            self.require_phrase = required

    def _orch(self, response, required):
        cfg = FakeCfg()
        cfg.conversation_mode = True
        orch = make_orch(response, cfg=cfg)
        orch.listener = self._Listener(required)
        return orch

    def test_conversation_mode_keeps_the_chat_going_in_phrase_mode(self):
        """«Wawi» opens the conversation; (silencio) is what closes it."""
        assert self._orch("Sigo acá.", required=True)._follow_up_seconds() == 20.0

    def test_without_conversation_mode_a_statement_closes(self):
        orch = self._orch("Sigo acá.", required=True)
        orch._cfg.conversation_mode = False
        assert orch._follow_up_seconds() == 0.0

    def test_a_question_still_waits_for_the_reply(self):
        assert self._orch("¿La mando?", required=True)._follow_up_seconds() == 20.0

    def test_always_on_keeps_conversation_mode(self):
        assert self._orch("Sigo acá.", required=False)._follow_up_seconds() == 20.0


class TestSilentReply:
    @pytest.mark.parametrize("text", ["(silencio)", " (silencio). ", "Silencio", "(SILENCIO)"])
    def test_marker_is_silent(self, text):
        from wawi.prompts import is_silent_reply

        assert is_silent_reply(text)

    @pytest.mark.parametrize("text", ["", None, "Hay silencio en la sala.", "Listo."])
    def test_real_answers_are_not(self, text):
        from wawi.prompts import is_silent_reply

        assert not is_silent_reply(text)

    def test_the_streaming_speaker_never_says_it(self):
        from wawi.orchestrator import _StreamingSpeaker

        speaker = _StreamingSpeaker(object(), enabled=True)
        speaker.submit("(silencio)")
        assert not speaker.spoke

    def test_both_prompts_teach_it(self):
        from wawi.prompts import SILENT_REPLY, system_prompt

        assert SILENT_REPLY in system_prompt(language="es")
        assert SILENT_REPLY in system_prompt(language="en")


class TestBargeInSafety:
    """Barge-in must never let the assistant wake on its own voice."""

    def test_disabled_when_barge_in_is_off(self):
        orch = make_orch()
        orch.listener = object()
        assert orch._barge_in_safe("Hay tres agentes") is False

    def test_disabled_without_a_listener(self):
        cfg = FakeCfg()
        cfg.barge_in = True
        orch = make_orch(cfg=cfg)
        assert orch._barge_in_safe("Hay tres agentes") is False

    def test_disabled_for_openwakeword(self):
        """No transcript to check the detection against."""
        cfg = FakeCfg()
        cfg.barge_in = True
        cfg.wake_engine = "openwakeword"
        orch = make_orch(cfg=cfg)
        orch.listener = object()
        assert orch._barge_in_safe("Hay tres agentes") is False

    def test_allowed_for_an_ordinary_answer(self):
        cfg = FakeCfg()
        cfg.barge_in = True
        orch = make_orch(cfg=cfg)
        orch.listener = object()
        assert orch._barge_in_safe("Hay tres agentes trabajando") is True

    @pytest.mark.parametrize(
        "response",
        ["Soy Wawi, y hay tres agentes", "Me llamo wawi", "Hoy es viernes"],
    )
    def test_blocked_when_the_answer_says_the_wake_phrase(self, response):
        """Without an echo canceller this would wake it on its own voice."""
        cfg = FakeCfg()
        cfg.barge_in = True
        orch = make_orch(cfg=cfg)
        orch.listener = object()
        assert orch._barge_in_safe(response) is False


class TestConversationMode:
    """Continuous conversation: the wake phrase restarts, it does not toll."""

    def _conv(self, response, pending=None):
        cfg = FakeCfg()
        cfg.conversation_mode = True
        return make_orch(response, pending, cfg=cfg)

    @pytest.mark.parametrize(
        "response",
        ["El proyecto 3 terminó.", "Listo, anotado.", "¿En cuál proyecto?"],
    )
    def test_the_channel_stays_open_after_every_answer(self, response):
        assert self._conv(response)._follow_up_seconds() == 20.0

    def test_the_mode_wins_over_every_other_rule(self):
        """Conversation mode is a decision, not a hint: nothing overrides it."""
        cfg = FakeCfg()
        cfg.conversation_mode = True
        cfg.wake_follow_up_idle_s = 0.0
        assert make_orch("Terminó.", cfg=cfg)._follow_up_seconds() == 20.0

    def test_turning_it_off_restores_question_only_waiting(self):
        cfg = FakeCfg()
        cfg.conversation_mode = False
        assert make_orch("El proyecto 3 terminó.", cfg=cfg)._follow_up_seconds() == 0.0
        assert make_orch("¿Cuál proyecto?", cfg=cfg)._follow_up_seconds() == 20.0


class FakeVoice:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class FakeTui:
    def set_texts(self, **kw):
        pass


def make_announcer(state_value="IDLE"):
    """An orchestrator with just enough wired for _on_watch_batch."""
    import threading

    from wawi.conversation import Conversation
    from wawi.reporter import Reporter
    from wawi.state import State, StateMachine

    orch = object.__new__(Orchestrator)
    orch._cfg = FakeCfg()
    orch.reporter = Reporter()
    orch.voice = FakeVoice()
    orch.tui = FakeTui()
    orch.listener = None
    orch.conversation = Conversation()
    orch.state = StateMachine()
    orch._turn_lock = threading.Lock()
    orch._stop_refresh = threading.Event()
    orch._last_response = ""
    orch._barge_in_armed = False
    orch.state.set_state(getattr(State, state_value))
    return orch


def finished_event(name="agenda"):
    from wawi.watcher import EventKind, WorktreeEvent

    return WorktreeEvent(
        worktree_id=f"id-{name}",
        display_name=name,
        kind=EventKind.FINISHED,
        at=0,
        preview="",
        last_assistant_message="",
        agent_phase="done",
    )




class TestSessionContinuity:
    """The session id is what makes a series of questions one conversation.

    It is also the thing whose failure is invisible: a dropped session still
    answers, just without context and at full price.
    """

    def _orch(self):
        from wawi.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch._session_id = None
        return orch

    def test_it_starts_with_no_session(self):
        # The first turn of a run has nothing to resume.
        assert self._orch()._session_id is None

    def test_the_first_answer_starts_a_session(self):
        orch = self._orch()
        orch._remember_session("sess-abc123")
        assert orch._session_id == "sess-abc123"

    def test_a_missing_session_id_does_not_erase_the_one_we_have(self):
        # A backend that reports no session (Ollama) must not silently end a
        # conversation the Claude backend was keeping.
        orch = self._orch()
        orch._remember_session("sess-abc123")
        orch._remember_session(None)
        assert orch._session_id == "sess-abc123"

    def test_a_failed_turn_drops_the_session(self):
        """Keeping a rejected id turns one bad turn into a dead assistant.

        If the server refused the session — expired, unknown — resuming it
        fails identically on every later turn. Starting fresh loses the
        context; holding a broken id loses everything.
        """
        orch = self._orch()
        orch._remember_session("sess-dead")

        orch._forget_session()

        assert orch._session_id is None

    def test_forgetting_twice_is_harmless(self):
        orch = self._orch()
        orch._forget_session()
        orch._forget_session()
        assert orch._session_id is None


class TestHeldUtterance:
    """Speech that arrives while Wawi is still thinking.

    The old behaviour discarded it and filed a ticket saying so, which is the
    worst of both worlds: from outside it looks like the wake word broke, and
    the only record lands in a file nobody reads mid-conversation. A person
    interrupted mid-thought answers late; they do not pretend they heard
    nothing.
    """

    def _orch(self):
        import threading

        from wawi.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch._pending_utterance = None
        orch._pending_lock = threading.Lock()
        orch.panel = type("P", (), {"set_texts": lambda self, **kw: None})()
        return orch

    def test_a_held_utterance_comes_back(self):
        import numpy as np

        orch = self._orch()
        buf = np.ones(4, dtype=np.float32)
        orch._hold_utterance(buf)
        assert orch._take_held_utterance() is buf

    def test_taking_it_clears_the_slot(self):
        # Otherwise the same sentence is answered once per finished turn,
        # forever.
        import numpy as np

        orch = self._orch()
        orch._hold_utterance(np.ones(4, dtype=np.float32))
        orch._take_held_utterance()
        assert orch._take_held_utterance() is None

    def test_nothing_held_returns_nothing(self):
        assert self._orch()._take_held_utterance() is None

    def test_the_latest_wins(self):
        # Three sentences said while Wawi thinks are usually one sentence
        # rephrased; answering all three in a row is worse than answering the
        # last.
        import numpy as np

        orch = self._orch()
        first = np.ones(4, dtype=np.float32)
        last = np.zeros(4, dtype=np.float32)
        orch._hold_utterance(first)
        orch._hold_utterance(last)
        assert orch._take_held_utterance() is last

    def test_a_stale_utterance_is_dropped(self):
        # Answering something said a minute ago -- after the user gave up and
        # said it again -- starts a second conversation instead of finishing
        # the first.
        import time

        import numpy as np

        from wawi import orchestrator as orch_mod

        orch = self._orch()
        orch._hold_utterance(np.ones(4, dtype=np.float32))
        stale = time.monotonic() - (orch_mod._HELD_UTTERANCE_MAX_AGE_S + 1)
        orch._pending_utterance = (orch._pending_utterance[0], stale)
        assert orch._take_held_utterance() is None

    def test_a_stale_utterance_still_clears_the_slot(self):
        import time

        import numpy as np

        from wawi import orchestrator as orch_mod

        orch = self._orch()
        orch._hold_utterance(np.ones(4, dtype=np.float32))
        stale = time.monotonic() - (orch_mod._HELD_UTTERANCE_MAX_AGE_S + 1)
        orch._pending_utterance = (orch._pending_utterance[0], stale)
        orch._take_held_utterance()
        assert orch._pending_utterance is None


# --------------------------------------------------------------------------- #
# Streaming speech — start talking before the answer is finished
# --------------------------------------------------------------------------- #

import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from wawi import brain  # noqa: E402
from wawi import lang  # noqa: E402
from wawi import orchestrator as orch_mod  # noqa: E402
from wawi.orchestrator import TurnMetrics, _StreamingSpeaker  # noqa: E402
from wawi.recall import Recall  # noqa: E402
from wawi.state import State, StateMachine  # noqa: E402


class RecordingVoice:
    """Records the pieces handed to TTS and can pretend to be interrupted."""

    def __init__(self, abort_after: int | None = None) -> None:
        self.parts: list[tuple[str, bool]] = []
        self.said: list[str] = []
        self._abort_after = abort_after
        self.aborted = False

    def say_part(self, text: str, *, first: bool) -> bool:
        if self.aborted:
            return False
        self.parts.append((text, first))
        if self._abort_after is not None and len(self.parts) >= self._abort_after:
            self.aborted = True
            return False
        return True

    def say(self, text: str) -> None:
        self.said.append(text)

    def interrupt(self) -> None:
        self.aborted = True


class Recorder:
    """Stands in for the panel, the TUI and the printer at once."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.answers: list[str] = []
        self.channel_open = False
        self.title_text = ""

    def set_texts(self, **kw) -> None:
        if "response" in kw:
            self.texts.append(kw["response"])

    def set_metrics(self, *a, **kw) -> None:
        pass

    def answer(self, text: str) -> None:
        self.answers.append(text)

    def question(self, text: str) -> None:
        pass

    def note(self, text: str) -> None:
        pass


class StreamCfg(FakeCfg):
    tts_streaming = True
    backend = "claude"
    sample_rate = 16000
    model_on_demand = "m"
    ollama_model = "o"
    ollama_url = "u"
    assistant_name = "Wawi"
    dangerous_skip_permissions = False
    monitor = 1
    max_edge = 1280
    jpeg_quality = 65
    stt_initial_prompt = ""


class FakeConversation:
    def __init__(self) -> None:
        self.added: list[tuple] = []

    def as_prompt_context(self, name: str) -> str:  # noqa: ARG002
        return ""

    def add(self, user_text, response, tools) -> None:
        self.added.append((user_text, response, tools))

    def __len__(self) -> int:
        return len(self.added)


class FakeIssues:
    def __init__(self) -> None:
        self.records: list[tuple] = []

    def prompt_hint(self) -> str:
        return ""

    def record(self, kind, title, **kw) -> None:
        self.records.append((kind, title, kw))


def make_turn_orch(cfg=None):
    """An Orchestrator with only what _do_turn touches, and no I/O anywhere."""
    o = object.__new__(Orchestrator)
    o._cfg = cfg or StreamCfg()
    o._language = "es"
    o._session_id = None
    o._screen_notes = ()
    o._engine = None
    o._connectors = None
    o._last_response = ""
    o._barge_in_armed = False
    o._pending_utterance = None
    o._pending_lock = threading.Lock()
    o._turn_lock = threading.Lock()
    o._turn_lock.acquire()
    o._stop_refresh = threading.Event()
    o.listener = None
    o.tasks = None
    o.voice = RecordingVoice()
    o.panel = Recorder()
    o.tui = Recorder()
    o.printer = Recorder()
    o.conversation = FakeConversation()
    o.issues = FakeIssues()
    o.preferences = type("P", (), {"as_prompt_block": staticmethod(lambda: "")})()
    o.recall = Recall(None)
    o._session = type("S", (), {"accumulate": staticmethod(lambda m: None)})()
    o.stt = type("T", (), {"transcribe": staticmethod(lambda *a, **kw: "cómo va")})()

    o.state = StateMachine()
    o.transitions: list[tuple[State, State]] = []
    o.state.subscribe(lambda old, new: o.transitions.append((old, new)))
    o.state.set_state(State.THINKING)
    return o


def streaming_brain(sentences, text=None, error=None):
    """A fake ask_sync that streams *sentences* and then returns or raises."""

    def fake(user_text, **kw):  # noqa: ARG001
        hook = kw.get("on_sentence")
        for sentence in sentences:
            if hook is not None:
                hook(sentence)
        if error is not None:
            raise error
        return brain.BrainResult(
            text=text if text is not None else " ".join(sentences),
            cost_usd=0.01,
            tool_calls=(),
            session_id="sess-1",
        )

    return fake


AUDIO = np.full(1600, 0.5, dtype=np.float32)


class TestStreamingSpeaker:
    """The pieces reach TTS in order, and only the first one opens SPEAKING."""

    def test_it_speaks_each_sentence_and_only_moves_state_once(self):
        o = make_turn_orch()
        speaker = _StreamingSpeaker(o, enabled=True)
        speaker.submit("Uno.")
        speaker.submit("Dos.")
        speaker.close()

        assert o.voice.parts == [("Uno.", True), ("Dos.", False)]
        assert o.transitions == [
            (State.IDLE, State.THINKING),
            (State.THINKING, State.SPEAKING),
        ]

    def test_disabled_means_nothing_is_spoken_and_no_thread_runs(self):
        o = make_turn_orch()
        speaker = _StreamingSpeaker(o, enabled=False)
        speaker.submit("Uno.")
        speaker.close()

        assert o.voice.parts == []
        assert speaker.spoke is False

    def test_a_barge_in_stops_the_rest_and_leaves_SPEAKING(self):
        """The voice stopped but the brain has not: THINKING is what is true,
        and it is the state where late speech is held instead of dropped."""
        o = make_turn_orch()
        o.voice = RecordingVoice(abort_after=1)
        speaker = _StreamingSpeaker(o, enabled=True)
        speaker.submit("Uno.")
        speaker.submit("Dos.")
        speaker.submit("Tres.")
        speaker.close()

        assert [p[0] for p in o.voice.parts] == ["Uno."]
        assert o.state.state is State.THINKING

    def test_close_waits_for_the_audio_to_finish(self):
        """Returning to IDLE mid-sentence reopens the mic onto our own voice."""
        started = threading.Event()
        release = threading.Event()
        o = make_turn_orch()

        class SlowVoice(RecordingVoice):
            def say_part(self, text, *, first):
                started.set()
                release.wait(2.0)
                return super().say_part(text, first=first)

        o.voice = SlowVoice()
        speaker = _StreamingSpeaker(o, enabled=True)
        speaker.submit("Uno.")
        assert started.wait(2.0)

        closed = threading.Event()
        threading.Thread(
            target=lambda: (speaker.close(), closed.set()), daemon=True
        ).start()
        assert not closed.wait(0.2)
        release.set()
        assert closed.wait(2.0)
        assert [p[0] for p in o.voice.parts] == ["Uno."]

    def test_a_thrown_sentence_does_not_stop_the_next_one(self):
        o = make_turn_orch()

        class HalfBrokenVoice(RecordingVoice):
            def say_part(self, text, *, first):
                if text == "Uno.":
                    raise RuntimeError("the speaker exploded")
                return super().say_part(text, first=first)

        o.voice = HalfBrokenVoice()
        speaker = _StreamingSpeaker(o, enabled=True)
        speaker.submit("Uno.")
        speaker.submit("Dos.")
        speaker.close()

        assert [p[0] for p in o.voice.parts] == ["Dos."]


class TestDurableMemoryInTheTurn:
    """The rolling cache reaches the model, and the turn feeds it back."""

    def test_what_previous_runs_stored_reaches_the_brain_call(
        self, monkeypatch, tmp_path
    ):
        path = tmp_path / "recall.txt"
        Recall(path).add("cómo se llama el proyecto", "wawi")

        o = make_turn_orch()
        o.recall = Recall(path)
        seen: dict = {}

        def fake(user_text, **kw):  # noqa: ARG001
            seen.update(kw)
            return brain.BrainResult(text="listo", tool_calls=())

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o._do_turn(AUDIO)

        assert "cómo se llama el proyecto" in seen["recall"]

    def test_the_turn_is_written_to_the_store_for_the_next_run(
        self, monkeypatch, tmp_path
    ):
        path = tmp_path / "recall.txt"
        o = make_turn_orch()
        o.recall = Recall(path)
        monkeypatch.setattr(orch_mod.brain, "ask_sync", streaming_brain(["Listo."]))

        o._do_turn(AUDIO)

        assert [e.user for e in Recall(path).entries()] == ["cómo va"]

    def test_a_disabled_store_sends_nothing_and_still_answers(
        self, monkeypatch, tmp_path
    ):
        o = make_turn_orch()
        o.recall = Recall(None)
        seen: dict = {}

        def fake(user_text, **kw):  # noqa: ARG001
            seen.update(kw)
            return brain.BrainResult(text="listo", tool_calls=())

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o._do_turn(AUDIO)

        assert seen["recall"] == ""
        assert o._last_response == "listo"
        assert list(tmp_path.iterdir()) == []


class TestStreamedTurn:
    """The turn as a whole: spoken early, recorded whole."""

    def test_the_answer_is_spoken_in_pieces_and_never_repeated(self, monkeypatch):
        o = make_turn_orch()
        monkeypatch.setattr(
            orch_mod.brain, "ask_sync", streaming_brain(["Uno.", "Dos."])
        )

        o._do_turn(AUDIO)

        assert [p[0] for p in o.voice.parts] == ["Uno.", "Dos."]
        assert o.voice.said == []  # the whole thing is never said again

    def test_memory_and_the_transcript_still_get_the_whole_answer(
        self, monkeypatch
    ):
        """Streaming changes WHEN text is spoken, not WHAT is recorded."""
        o = make_turn_orch()
        monkeypatch.setattr(
            orch_mod.brain, "ask_sync", streaming_brain(["Uno.", "Dos."])
        )

        o._do_turn(AUDIO)

        assert o.conversation.added == [("cómo va", "Uno. Dos.", ())]
        assert o.printer.answers == ["Uno. Dos."]
        assert o._last_response == "Uno. Dos."

    def test_the_state_sequence_is_the_one_the_microphone_expects(
        self, monkeypatch
    ):
        o = make_turn_orch()
        monkeypatch.setattr(orch_mod.brain, "ask_sync", streaming_brain(["Uno."]))

        o._do_turn(AUDIO)

        assert o.transitions == [
            (State.IDLE, State.THINKING),
            (State.THINKING, State.SPEAKING),
            (State.SPEAKING, State.IDLE),
        ]

    def test_speaking_starts_before_the_brain_returns(self, monkeypatch):
        """The whole point: the user hears sentence one while two is written."""
        spoke_at_state: list[State] = []
        o = make_turn_orch()

        def fake(user_text, **kw):  # noqa: ARG001
            kw["on_sentence"]("Uno.")
            for _ in range(200):
                if o.state.state is State.SPEAKING:
                    break
                time.sleep(0.01)
            spoke_at_state.append(o.state.state)
            return brain.BrainResult(text="Uno. Dos.", tool_calls=())

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o._do_turn(AUDIO)

        assert spoke_at_state == [State.SPEAKING]

    def test_a_brain_failure_after_speaking_still_says_what_went_wrong(
        self, monkeypatch
    ):
        o = make_turn_orch()
        monkeypatch.setattr(
            orch_mod.brain,
            "ask_sync",
            streaming_brain(["Empecé a mirar."], error=brain.BrainError("boom")),
        )

        o._do_turn(AUDIO)

        assert [p[0] for p in o.voice.parts] == ["Empecé a mirar."]
        assert o.voice.said == ["no pude conectarme"]

    def test_a_spent_turn_budget_does_not_blame_the_network(self, monkeypatch):
        o = make_turn_orch()
        monkeypatch.setattr(
            orch_mod.brain,
            "ask_sync",
            streaming_brain([], error=brain.BrainBudgetError("out of turns")),
        )

        o._do_turn(AUDIO)

        assert o.voice.said == [lang.BUDGET_TEXT["es"]]

    def test_an_abandoned_stream_is_replaced_by_the_whole_answer(
        self, monkeypatch
    ):
        """The retry wrote a different answer; the fragment already spoken is
        not half of it."""
        o = make_turn_orch()

        def fake(user_text, **kw):  # noqa: ARG001
            kw["on_sentence"]("Voy a mirar.")
            kw["on_restart"]()
            return brain.BrainResult(text="Eran tres.", tool_calls=())

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o._do_turn(AUDIO)

        assert o.voice.said == ["Eran tres."]

    def test_streaming_off_keeps_the_old_wait_then_speak_path(self, monkeypatch):
        cfg = StreamCfg()
        cfg.tts_streaming = False
        o = make_turn_orch(cfg)
        monkeypatch.setattr(
            orch_mod.brain, "ask_sync", streaming_brain(["Uno.", "Dos."])
        )

        o._do_turn(AUDIO)

        assert o.voice.parts == []
        assert o.voice.said == ["Uno. Dos."]

    def test_streaming_off_does_not_ask_the_sdk_for_partial_messages(
        self, monkeypatch
    ):
        """The hook is also what turns partial messages on; a turn nobody
        listens to should not pay to stream them."""
        cfg = StreamCfg()
        cfg.tts_streaming = False
        o = make_turn_orch(cfg)
        seen: dict = {}

        def fake(user_text, **kw):  # noqa: ARG001
            seen["hook"] = kw.get("on_sentence")
            return brain.BrainResult(text="Uno.", tool_calls=())

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o._do_turn(AUDIO)

        assert seen["hook"] is None

    def test_the_held_utterance_still_comes_back_after_a_streamed_turn(
        self, monkeypatch
    ):
        """The mid-turn hold is what keeps speech during THINKING from being
        dropped; streaming must not cost it."""
        o = make_turn_orch()
        dispatched: list = []
        o._dispatch_turn = lambda buf, already_locked: dispatched.append(buf)
        o._pending_utterance = (AUDIO, time.monotonic())
        monkeypatch.setattr(orch_mod.brain, "ask_sync", streaming_brain(["Uno."]))

        o._do_turn(AUDIO)

        assert len(dispatched) == 1
        assert o._pending_utterance is None


class TestKeyBargeInDuringAStreamedAnswer:
    """The key must still open a capture when the brain is still writing."""

    def test_a_cut_off_stream_does_not_swallow_the_press(self):
        o = make_turn_orch()  # the turn lock is held, as during a live turn
        o.state.set_state(State.SPEAKING)
        started: list[bool] = []
        o.listener = type("L", (), {"force_start": lambda self: started.append(True)})()

        class CuttingVoice(RecordingVoice):
            def interrupt(inner) -> None:  # noqa: N805
                super().interrupt()
                # What _streamed_speech_aborted does once say_part gives up.
                o.state.set_state(State.THINKING)

        o.voice = CuttingVoice()
        o._on_press()

        assert started == [True]


class TestScreenshotRotation:
    """A session that holds a screenshot is not resumed.

    Resuming it re-sends the image on every later turn until the CLI compacts.
    The next turn starts fresh with the conversation as text plus one line per
    capture, and from then on resumes as usual.
    """

    NOTE = "[captura 14:50:08, 1280x800, monitor 1]"

    def test_a_turn_that_looked_is_not_resumed_and_leaves_a_note(self, monkeypatch):
        o = make_turn_orch()
        o._session_id = "sess-old"
        calls: list[dict] = []
        results = [
            brain.BrainResult(
                text="Veo el editor.",
                tool_calls=(brain.SCREEN_TOOL,),
                session_id="sess-with-image",
                screen_notes=(self.NOTE,),
            ),
            brain.BrainResult(text="Es el archivo 3.", session_id="sess-clean"),
            brain.BrainResult(text="De nada.", session_id="sess-clean"),
        ]

        def fake(user_text, **kw):  # noqa: ARG001
            calls.append(kw)
            return results[len(calls) - 1]

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)

        o._do_turn(AUDIO)
        assert o._session_id is None
        assert o._screen_notes == (self.NOTE,)

        o._turn_lock.acquire()
        o._do_turn(AUDIO)
        assert calls[1]["resume"] is None
        assert self.NOTE in calls[1]["history"]
        assert o._session_id == "sess-clean"
        assert o._screen_notes == ()

        o._turn_lock.acquire()
        o._do_turn(AUDIO)
        assert calls[2]["resume"] == "sess-clean"
        assert self.NOTE not in calls[2]["history"]


class TestJunkTranscripts:
    """Whisper's hallucinations on noise must never become a paid turn."""

    @staticmethod
    def _stt(text, metrics):
        return type("T", (), {
            "transcribe_detailed": staticmethod(lambda *a, **kw: (text, metrics)),
        })()

    @staticmethod
    def _audio(voiced_ms):
        from wawi.listener import CapturedAudio
        from wawi.speechgate import SpeechEvidence

        buf = AUDIO.copy().view(CapturedAudio)
        buf.speech = SpeechEvidence(voiced_ms=voiced_ms, active_ms=voiced_ms + 100)
        return buf

    def _turn(self, monkeypatch, text, metrics, voiced_ms):
        o = make_turn_orch()
        o.stt = self._stt(text, metrics)
        asked: list[str] = []

        def fake(user_text, **kw):  # noqa: ARG001
            asked.append(user_text)
            return brain.BrainResult(text="listo", tool_calls=())

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o._do_turn(self._audio(voiced_ms))
        return o, asked

    def test_the_prompt_echo_from_noise_is_dropped(self, monkeypatch, caplog):
        from wawi.speechgate import SttMetrics

        caplog.set_level("INFO", logger="wawi.orchestrator")
        noisy = SttMetrics(avg_logprob=-0.55, no_speech_prob=0.0, compression_ratio=0.6, segments=1)
        o, asked = self._turn(monkeypatch, "¿Está bien?", noisy, voiced_ms=300)
        assert asked == []
        assert o.conversation.added == []
        assert "Dropped transcript as junk" in caplog.text
        assert o.state.state is State.IDLE

    def test_a_short_genuine_reply_still_reaches_the_brain(self, monkeypatch):
        from wawi.speechgate import SttMetrics

        noisy = SttMetrics(avg_logprob=-0.55, no_speech_prob=0.0, compression_ratio=0.6, segments=1)
        _, asked = self._turn(monkeypatch, "Sí, dale.", noisy, voiced_ms=300)
        assert asked == ["Sí, dale."]

    def test_the_bare_name_heard_by_the_wake_detector_is_not_an_echo(self, monkeypatch, caplog):
        # The STT prompt carries the name, so a short "Wawi." looks like its
        # echo; the wake detector already heard it in this audio, so it must
        # reach the bare-summon path instead of being dropped.
        from wawi.speechgate import SttMetrics

        caplog.set_level("INFO", logger="wawi.orchestrator")
        o = make_turn_orch()
        o.stt = self._stt("Wawi.", SttMetrics(
            avg_logprob=-0.42, no_speech_prob=0.0, compression_ratio=0.38, segments=1))
        monkeypatch.setattr(o, "_stt_prompt", lambda: "Wawi, Wawi.", raising=False)
        asked: list[str] = []
        monkeypatch.setattr(
            orch_mod.brain, "ask_sync",
            lambda t, **kw: asked.append(t) or brain.BrainResult(text="ok", tool_calls=()),
        )
        buf = self._audio(360)
        buf.wake = "wawi"
        assert o._transcribe_filtered(buf) == "Wawi."
        assert "Dropped transcript as junk" not in caplog.text
        # In the reply window nothing checks for the name: a voiced one is
        # kept (she was ignoring being called), a breath is not.
        assert o._transcribe_filtered(self._audio(360)) == "Wawi."
        assert o._transcribe_filtered(self._audio(150)) == ""

    def test_the_filter_can_be_switched_off(self, monkeypatch):
        from wawi.speechgate import SttMetrics

        class Cfg(StreamCfg):
            stt_filter_junk = False

        o = make_turn_orch(Cfg())
        noisy = SttMetrics(avg_logprob=-0.55, no_speech_prob=0.0, compression_ratio=0.6, segments=1)
        o.stt = self._stt("Gracias.", noisy)
        asked: list[str] = []
        monkeypatch.setattr(
            orch_mod.brain, "ask_sync",
            lambda t, **kw: asked.append(t) or brain.BrainResult(text="ok", tool_calls=()),
        )
        o._do_turn(self._audio(200))
        assert asked == ["Gracias."]


class TestDroppedCaptureClearsListening:
    def test_listening_goes_back_to_idle(self):
        o = object.__new__(Orchestrator)
        o.state = StateMachine()
        o.state.set_state(State.LISTENING)
        o._on_capture_dropped()
        assert o.state.state is State.IDLE

    def test_a_turn_in_flight_is_left_alone(self):
        o = object.__new__(Orchestrator)
        o.state = StateMachine()
        o.state.set_state(State.THINKING)
        o._on_capture_dropped()
        assert o.state.state is State.THINKING


class TestLoneNameInTheReplyWindow:
    """Observed: "Wawi." said in the reply window was dropped as a prompt echo."""

    class _Evidence:
        def __init__(self, voiced_ms):
            self.voiced_ms = voiced_ms

    def _orch(self):
        orch = make_orch("Listo.")
        orch.listener = None
        return orch

    def test_a_voiced_name_alone_is_kept(self):
        assert self._orch()._lone_name("Wawi.", self._Evidence(360))

    def test_a_breath_is_not_a_name(self):
        assert not self._orch()._lone_name("Wawi.", self._Evidence(120))

    def test_more_than_the_name_is_not_a_lone_name(self):
        assert not self._orch()._lone_name("Wawi, mirá esto.", self._Evidence(900))
