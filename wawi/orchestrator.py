"""On-demand voice pipeline orchestrator.

Two ways in, one pipeline. The wake word is the primary path and push-to-talk
is the fallback:

  "Wawi, …" → Listener captures until you stop talking ─┐
  key-press/release → Listener captures while you hold it ────┴→ utterance
  utterance  → THINKING (spawn worker)
  worker     → STT → screenshot → brain → SPEAKING → voice.say → IDLE

With ``tts.streaming`` on, SPEAKING starts at the brain's FIRST finished
sentence rather than after its last one, and the sentences are spoken by
``_StreamingSpeaker`` as they arrive. The states, the turn lock and the
microphone bookkeeping are unchanged — only the timing moves.

Microphone ownership
--------------------
When the wake word is available, ``listener.Listener`` owns the one and only
``sd.InputStream`` and push-to-talk becomes a forced capture on that same
stream (``force_start``/``force_stop``). Two input streams on one CoreAudio
device is not reliably supported — see the listener module docstring.
``audio_input.Recorder`` is instantiated ONLY as the fallback for when the
listener cannot run at all (openwakeword missing or its weights not
downloaded); in that mode push-to-talk is the only way in.

Two locks, one ordering (design §Concurrency):
  _turn_lock  (threading.Lock) — non-blocking acquire gates whole turns;
                                 acquired when an utterance is dispatched,
                                 released in the worker's finally
  voice.VOICE_LOCK             — leaf lock, acquired only inside voice.say()
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from wawi import brain
from wawi.audio_input import Recorder
from wawi.config import Config
from wawi.connectors import Connectors
from wawi.console import TurnPrinter
from wawi.conversation import Conversation
from wawi import lang
from wawi import paths as _paths
from wawi.feedback import EarconObserver
from wawi import inflight as inflight_module
from wawi.inflight import Inflight
from wawi import drops as drops_module
from wawi import inbox as inbox_module
from wawi import notes as notes_module
from wawi.listener import Listener, ListenerConfig, ListenerError
from wawi.panel import PanelObserver, run_event_loop
from wawi import selfimprove
from wawi import speechgate
from wawi.preferences import Preferences
from wawi.recall import Recall
from wawi.state import State, StateMachine
from wawi.stt import STT
from wawi.ui import TuiObserver
from wawi.voice import Voice

log = logging.getLogger(__name__)

# These three never reach the model, so nothing else would translate them:
# they are what the assistant says when the model was not consulted at all.
# See wawi/lang.py.


@dataclass
class TurnMetrics:
    """Latency and cost metrics for a single completed turn."""

    stt_s: float = 0.0
    brain_s: float = 0.0
    total_s: float = 0.0
    cost_usd: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass
class SessionMetrics:
    """Accumulated metrics across all turns in the current session."""

    turn_count: int = 0
    total_cost_usd: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    def accumulate(self, turn: TurnMetrics) -> None:
        self.turn_count += 1
        if turn.cost_usd is not None:
            self.total_cost_usd += turn.cost_usd
        if turn.tokens_in is not None:
            self.total_tokens_in += turn.tokens_in
        if turn.tokens_out is not None:
            self.total_tokens_out += turn.tokens_out

# Peak amplitude below this = effectively silence. Skip the turn entirely so
# Whisper's silence hallucination ("gracias por ver el video") never reaches
# the brain. Real speech peaks well above this; a denied mic gives exactly 0.0.
_SILENCE_PEAK = 0.01

# Where the HUD's position and size are remembered between runs. Beside the
# config rather than in it: this is something the user set by DRAGGING, and
# rewriting their hand-written config file to record a drag would be rude.
def _panel_state() -> Path:
    return _paths.private("panel_state.json")

# How long the assistant's own words stay "recent enough" to be recognised as
# speaker bleed after playback stops. Covers room decay and the follow-up
# window opening on the tail of its own sentence.
_ECHO_TAIL_S = 1.5

# How long the executor's voice waits for the turn lock before giving up on
# saying something. Long enough to sit through an ordinary answer, short enough
# that the executor's thread is not parked behind a wedged turn.
_TASK_SPEECH_WAIT_S = 25.0

#: How long an utterance heard mid-turn is still worth answering. Past this you
#: have already given up and said it again, so a reply is not a late answer --
#: it is Wawi starting a second conversation with itself.
_HELD_UTTERANCE_MAX_AGE_S = 45.0

#: After catching something dropped on her, she gets this long to show it
#: (the catch clip) before she says her line and opens the microphone.
_DROP_CATCH_PAUSE_S = 0.9
#: How often the drop window is checked, and the most it may outlast its
#: length while the owner is still talking (a capture already open).
_DROP_POLL_S = 0.1
_DROP_OVERRUN_S = 120.0

#: How much of the executor's closing report is read out loud. Kokoro speaks at
#: roughly 15 characters a second, so this is about 15 seconds -- already long
#: for something you cannot interrupt. The full text stays in the HUD and the
#: log; this is only what the ear gets.
_SPOKEN_RESULT_CHARS = 220

#: How long after it is asked a "¿lo retomo?" still takes a yes. Past this, a
#: "sí" is about something else, and resuming on it would be a surprise.
_RESUME_OFFER_S = 90.0

#: How much of the interrupted goal is read back in the question. Enough to
#: recognise it, short enough that the question stays one breath.
_RESUME_WHAT_CHARS = 70


class _StreamingSpeaker:
    """Say an answer sentence by sentence while the brain is still writing it.

    Exists as a thread of its own for one reason: the brain hands sentences
    over from the SDK's event loop, and synthesis plus playback takes seconds.
    Speaking inline there would block the very stream that produces the rest of
    the answer, which turns streaming into the slow path with extra steps.

    The turn's own state sequence does not change — THINKING → SPEAKING →
    IDLE — only the moment SPEAKING starts moves earlier, to the first finished
    sentence instead of the last. That is strictly safer for the microphone:
    :meth:`Orchestrator._on_state_change` pauses the listener on SPEAKING, so
    it is now paused for MORE of the turn than before, never less.
    """

    def __init__(self, orch: "Orchestrator", *, enabled: bool) -> None:
        self._orch = orch
        self.enabled = enabled
        #: perf_counter of the first sentence the brain finished (latency log).
        self.first_sentence_at: float | None = None
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._said: list[str] = []
        self._submitted = 0
        self._aborted = False
        #: Set when the answer being streamed was abandoned and rewritten. What
        #: was already spoken belongs to an answer that no longer exists, so
        #: the caller must deliver the real one in full.
        self.voided = False

    @property
    def spoke(self) -> bool:
        """True once at least one sentence has been handed over to be spoken.

        Counted at submission, not at playback: the caller reads this the
        moment the brain returns, while the speaker thread may still be on the
        first sentence, and "not spoken YET" would make it say the whole answer
        a second time.
        """
        with self._lock:
            return self._submitted > 0

    def submit(self, sentence: str) -> None:
        """Queue one finished sentence. Called from the SDK's event loop.

        Never blocks and never raises: the brain must not lose an answer
        because the speaker is busy or broken.
        """
        if not self.enabled or not sentence:
            return
        from wawi.prompts import is_silent_reply

        if is_silent_reply(sentence):
            return  # she chose to stay quiet: nothing to say, nothing to arm
        with self._lock:
            self._submitted += 1
            if self.first_sentence_at is None:
                self.first_sentence_at = time.perf_counter()
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="wawi-stream-voice", daemon=True
            )
            self._thread.start()
        self._queue.put(sentence)

    def void(self) -> None:
        """Mark what was spoken as belonging to an abandoned answer."""
        self.voided = True

    def close(self) -> None:
        """Wait for everything queued to be spoken. Safe to call twice.

        Blocking on purpose: the worker owns the turn lock and returns to IDLE
        right after, and doing either while a sentence is still playing would
        reopen the microphone onto the assistant's own voice.
        """
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while True:
            sentence = self._queue.get()
            if sentence is None:
                return
            if self._aborted:
                # Drained rather than spoken: after a barge-in the rest of this
                # answer is no longer wanted, but submit() must keep working or
                # the brain blocks on a queue nobody reads.
                continue
            with self._lock:
                first = not self._said
                self._said.append(sentence)
                so_far = " ".join(self._said)
            try:
                self._orch._begin_streamed_speech(so_far, first=first)
                if not self._orch.voice.say_part(sentence, first=first):
                    self._aborted = True
                    self._orch._streamed_speech_aborted()
            except Exception:
                # One failed sentence must not silence the ones behind it.
                log.warning("Streamed sentence failed", exc_info=True)


class Orchestrator:
    """Wire the full voice pipeline.

    Args:
        cfg: Loaded and validated Config instance.
    """

    # Class-level defaults so the recovery path costs nothing where it was
    # never set up: a disabled store keeps nothing, and no offer is pending.
    inflight: Inflight = Inflight(None)
    _resume_offer: "inflight_module.Record | None" = None
    _resume_offer_until: float = 0.0
    # Taking notes (wawi/notes.py): the session, and the queue of utterances
    # its worker transcribes. None while no note is being taken.
    _notes: "notes_module.NoteSession | None" = None
    _notes_queue: "queue.Queue | None" = None
    _notes_lock = threading.Lock()
    # True while saying something that expects no reply (a note started or
    # ended from the menu): no follow-up window opens after it.
    _aside_speech: bool = False
    # Something dragged onto her (wawi/drops.py): what she holds, the window
    # length her catch line opens instead of the usual one, and whether that
    # window switched a muted microphone on (put back when it is over).
    _drop: "drops_module.Drop | None" = None
    _drop_follow_up: float = 0.0
    _drop_unmuted: bool = False
    _drop_lock = threading.Lock()

    def __init__(self, cfg: Config, debug: bool = False) -> None:
        self._cfg = cfg
        self._debug = debug

        # State machine — observers attach before any transition fires
        self.state = StateMachine()

        # Pipeline components. The listener is built here but its models are
        # not loaded and the mic is not opened until run() calls start().
        self.listener = self._build_listener(cfg)
        # Fallback recorder — only used when the listener is unavailable. In
        # listener mode it is never started, so it never opens a stream.
        self.recorder = Recorder(cfg.sample_rate)
        self.stt = STT(cfg.stt_model)
        self.voice = Voice(
            cfg.tts_voices_dir,
            cfg.tts_voice,
            speed=cfg.tts_speed,
            pitch_semitones=getattr(cfg, "tts_pitch_semitones", 0.0),
            brightness=getattr(cfg, "tts_brightness", 0.0),
            home_language=cfg.language,
        )

        # Observer modules
        self.tui = TuiObserver(cfg.show_tui)
        # The terminal transcript. Without the Rich TUI the console printed
        # nothing at all, so a spoken answer left no readable trace anywhere.
        # They are mutually exclusive: the TUI owns stdout when it runs.
        self.printer = TurnPrinter(enabled=not cfg.show_tui)
        self.earcons = EarconObserver(cfg.sound_cues)
        self.panel = PanelObserver(cfg.show_panel)
        from wawi.board import Board
        self.panel._board = Board(allow_network=bool(cfg.board.get("allow_network", False)))
        from wawi import tools as _board_tools
        _board_tools.set_board(self.panel._board)
        self.panel.title_text = cfg.assistant_name
        self.panel.show_mascot = cfg.show_mascot
        # Her body (needs, desktop life, reactions): config section `pet:`.
        self.panel.configure_pet(getattr(cfg, "pet", {}) or {})
        # What she learns from you (wawi/learn): tiny numpy models, no LLM.
        from wawi import learn as _learn

        _learn.install(_learn.build(cfg))
        # Her instincts (wawi/instinct): your day's rhythm, local only.
        from wawi import instinct as _instinct

        _instinct.install(_instinct.build(cfg))

        # Games against her (wawi/games): config section `games:`.
        from wawi import games as _games

        _games.install(_games.build(cfg))
        # Coding tasks sent to Orca by voice (wawi/orca): config `orca:`.
        # Building spawns nothing; the watcher resumes in run().
        from wawi import orca as _orca

        _orca.install(_orca.build(cfg))
        from wawi import shortcuts as _shortcuts

        _shortcuts.install(_shortcuts.build(cfg))
        # Her own little computer (wawi/webvm): config `webvm:`. Building
        # starts nothing; the container starts on her first web step.
        from wawi import webvm as _webvm

        _webvm.install(_webvm.build(cfg))
        # Agregados (wawi/plugins.py): the owner's services, from
        # plugins_local.json. Building reads nothing; the first turn does.
        from wawi import plugins as _plugins

        _plugins.install(_plugins.build())
        if getattr(cfg, "mascot_only", False):
            # Config, not memory: with ui.mascot_only the panel starts hidden
            # at EVERY launch, whatever the last run left behind, and Wawi
            # stands on her own. The menu-bar item can still bring it up.
            self.panel.panel_visible = False
            self.panel.show_mascot = True
        # The language is LIVE: the HUD can flip it between turns, so config is
        # only its starting value. Everything that depends on it reads this
        # attribute, never cfg.language, or half the pipeline would keep
        # answering in the old one.
        self._language = cfg.language
        # Speech heard while a turn was already thinking, waiting for its turn.
        self._pending_utterance: tuple[np.ndarray, float] | None = None
        self._pending_lock = threading.Lock()
        self.panel.set_language(self._language)

        # Register observers — order: earcons before TUI (earcons are non-blocking)
        self.state.subscribe(self.earcons)
        self.state.subscribe(self.tui)
        self.state.subscribe(self.panel)
        # Deafen the listener while Wawi speaks, so it cannot wake on its own
        # voice coming back through the speakers.
        self.state.subscribe(self._on_state_change)
        # Curiosidades (wawi/curiosities.py): a tiny fact in her bubble now
        # and then while idle; any turn keeps them quiet for a while.
        from wawi import curiosities as _curiosities

        _curiosities.install(_curiosities.build(cfg, board=self.panel._board))
        if _curiosities.get() is not None:
            self.state.subscribe(_curiosities.get().observe_state)
        # Bailar con música (wawi/music.py): the microphone half listens to
        # the blocks the listener already gets; it never gates anything.
        from wawi import music as _music

        try:
            _music.install(_music.build(cfg))
        except Exception:
            log.warning("Music detection unavailable", exc_info=True)
            _music.install(None)
        if _music.get() is not None and self.listener is not None:
            self.listener.add_block_observer(_music.get().feed)
        # Iniciativa (wawi/initiative): she acts on her own. Started in run().
        self._install_initiative(cfg)

        # Session-level accumulated metrics
        self._session = SessionMetrics()

        # Rolling conversation memory — without this every turn starts blind.
        self.conversation = Conversation(
            max_turns=cfg.memory_turns,
            log_path=cfg.conversation_log,
            assistant_name=cfg.assistant_name,
        )
        # Durable rolling memory. The in-memory window above dies with the
        # process; this is the handful of exchanges that walk back in with it,
        # so a restart does not put the user in front of a machine that has
        # never met them. It rides in the system prompt, frozen for the run —
        # see wawi/recall.py for why re-rendering it per turn would tear down
        # the live SDK connection every turn.
        self.recall = Recall(cfg.recall_file, max_entries=cfg.recall_turns)
        if len(self.recall):
            log.info("%d exchange(s) recalled from previous runs", len(self.recall))
        self._stop_refresh = threading.Event()

        # Self-improvement: records problems as issues. It CANNOT change code —
        # see wawi/selfimprove.py. Asking the assistant to put an agent to
        # work on a repo is a different, unrestricted capability (the executor).
        self.issues = selfimprove.IssueJournal(
            directory=cfg.issues_dir,
            assistant_name=cfg.assistant_name,
            github=cfg.issues_to_github,
        )
        from wawi import tools as _tools

        _tools.set_journal(self.issues)
        # Autoarreglo (wawi/selffix/): she fixes the issues above on her own,
        # behind gates and a watchdog she cannot edit. Idle until run().
        from wawi import selffix as _selffix

        try:
            _selffix.install(_selffix.build(cfg, board=getattr(self.panel, "_board", None)))
        except Exception:
            log.warning("Selffix unavailable", exc_info=True)
            _selffix.install(None)
        if _selffix.get() is not None:
            self.state.subscribe(_selffix.get().observe_state)

        # Standing instructions the user teaches by voice. Behaviour improves
        # without a line of code changing — see wawi/preferences.py.
        self.preferences = Preferences(cfg.preferences_file)
        _tools.set_preferences(self.preferences)
        _tools.set_screen_options(cfg.screenshots_dir, cfg.monitor)

        if len(self.preferences):
            log.info("%d preferences in effect", len(self.preferences))

        # Wire the UI to whichever component owns the microphone.
        mic = self.listener if self.listener is not None else self.recorder
        self.tui.set_level_source(mic.current_level)
        # The panel's spectrum needs raw samples, not the scalar level.
        self.panel.set_samples_source(mic.current_block)
        # Full-height bars while recording, damped while merely hearing the
        # room — so you can see it take your words as you say them.
        if self.listener is not None:
            self.panel.set_capture_source(lambda: self.listener.is_capturing)
        # Wawi's talking bob follows what she is actually saying.
        output_level = getattr(self.voice, "output_level", None)
        if callable(output_level):
            self.panel.set_voice_level_source(output_level)

        # Turn lock: non-blocking acquire = ignore mid-turn presses (REQ-F1-10)
        self._turn_lock = threading.Lock()
        self._turn_acquired = False  # fallback path only: did this press take the lock?
        # False while the user has muted Wawi from the panel switch. Kept
        # here so the end of a turn does not silently re-open the microphone
        # behind their back.
        self._listening = True
        # The SDK session every turn continues. None means the next turn opens
        # a fresh conversation — which is the correct state after a failure,
        # since resuming a session the server rejected fails the same way twice.
        self._session_id: str | None = None
        # Notes for the screenshots the last turn looked at. That turn's
        # session was dropped so its image is not re-sent forever; these lines
        # are what the next turn gets instead, once, as text.
        self._screen_notes: tuple[str, ...] = ()
        # A live SDK connection, so a question does not pay the ~1.3s the
        # `claude` binary takes to start. Built lazily on the first turn: it
        # spawns a process, and doing that during startup would delay the
        # microphone for something no one has asked for yet.
        self._engine = brain.Engine()
        # MCP servers borrowed from Claude Code / Desktop, gated. Nothing is
        # read from disk unless config.yaml enables at least one of them.
        self._connectors = Connectors.from_config(getattr(cfg, "connectors", None))
        from wawi import tools as _connector_tools
        _connector_tools.set_connectors(self._connectors)
        # True while the user may cut into the answer being spoken.
        self._barge_in_armed = False
        # The answer just spoken, for the follow-up decision.
        self._last_response = ""
        # The delegated executor (wawi/task.py), or None when tasks are off.
        # Wawi drives it through wawi.tools; it never drives Wawi.
        # What a crash or restart cut off (wawi/inflight.py). Read, and
        # deleted, right away: whatever the last run left is offered once, by
        # voice, as soon as the microphone is up — see _offer_resume.
        self.inflight = Inflight(cfg.inflight_file)
        self._resume_offer = self.inflight.take(cfg.resume_max_age_h * 3600)
        self._session_id = self.inflight.conversation_session()
        self.tasks = self._build_task_runner(cfg)
        _tools.set_task_runner(self.tasks)
        self.mind = self._build_mind(cfg)
        # Her reactions may speak (Reaction.speak), outside any turn and
        # through the same echo guard as an answer. See _speak_reaction.
        self._reaction_speech = False
        self._build_pet_voice(cfg)
        # Explicit owner reminders bypass the incidental-reaction quiet/rate
        # budget, but reuse the same locked speech path and echo guard.
        from wawi import reminders as _reminders

        reminder_cfg = getattr(cfg, "reminders", {}) or {}
        self.reminders = None
        if reminder_cfg.get("enabled", True):
            self.reminders = _reminders.ReminderService(
                _paths.resolve(reminder_cfg.get("local_file") or "reminders_local.json"),
                announce=self._announce_reminder,
                run_routine=self._run_routine,
                poll_s=float(reminder_cfg.get("poll_s", 1.0)),
            )
        _reminders.install(self.reminders)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_listener(self, cfg: Config) -> Listener | None:
        """Return a configured Listener, or None to fall back to push-to-talk.

        Never raises and never downloads: a missing openwakeword install or
        missing weights degrades to push-to-talk with a warning that names the
        fix, exactly as ``wawi --check`` reports it.
        """
        if not cfg.wake_enabled:
            log.info("Wake word disabled in config.yaml — push-to-talk only")
            return None
        try:
            detector = None
            engine = cfg.wake_engine
            if not cfg.wake_require_phrase:
                log.info(
                    "Always-on listening (wake.require_phrase: false) — any "
                    "speech starts a turn, no wake phrase needed"
                )
            if engine == "openwakeword" and cfg.wake_model:
                from wawi.listener import custom_wake_model_missing

                missing = custom_wake_model_missing(cfg.wake_model)
                if missing:
                    # A trained model that is not there yet must not cost the
                    # owner their wake word: the phrase mode works everywhere.
                    log.warning(
                        "Wake model %s not found — using the whisper phrase mode "
                        "instead (see scripts/wakeword/README.md to train it)", missing,
                    )
                    engine = "whisper"
            # The transcriber costs a whole extra Whisper model, so it is
            # LOADED only when something needs it: matching a trigger phrase,
            # or recognising the assistant's own voice while barge-in keeps
            # the microphone open during playback. It is still BUILT (cheap,
            # nothing loaded) so the live switch to phrase mode has one.
            needs_transcriber = cfg.wake_require_phrase or cfg.barge_in
            if engine == "whisper" and cfg.wake_phrases:
                from wawi.wake import WhisperWakeDetector

                detector = WhisperWakeDetector(
                    cfg.wake_phrases,
                    model=cfg.wake_whisper_model,
                    language=cfg.language,
                    sample_rate=cfg.sample_rate,
                    aliases=cfg.wake_aliases,
                    prompt=cfg.wake_whisper_prompt,
                )
            return Listener(
                on_utterance=self._on_utterance,
                on_capture_start=self._on_capture_start,
                on_capture_dropped=self._on_capture_dropped,
                wake_detector=detector,
                verifier=self._build_verifier(cfg) if detector is not None else None,
                cfg=ListenerConfig(
                    engine=engine,
                    require_phrase=cfg.wake_require_phrase,
                    wake_window_s=cfg.wake_window_s,
                    wake_model=cfg.wake_model if engine == "openwakeword" else "",
                    preload_transcriber=needs_transcriber,
                    wake_patience=cfg.wake_patience,
                    sample_rate=cfg.sample_rate,
                    pre_roll_s=cfg.wake_pre_roll_s,
                    silence_ms=cfg.wake_silence_ms,
                    silence_long_ms=cfg.wake_silence_long_ms,
                    long_after_s=cfg.wake_long_after_s,
                    smart_turn=cfg.wake_smart_turn,
                    smart_turn_threshold=cfg.wake_smart_turn_threshold,
                    smart_turn_pause_ms=cfg.wake_smart_turn_pause_ms,
                    smart_turn_sure=cfg.wake_smart_turn_sure,
                    smart_turn_recheck_ms=cfg.wake_smart_turn_recheck_ms,
                    max_utterance_s=cfg.wake_max_utterance_s,
                    follow_up_s=cfg.wake_follow_up_s,
                    vad_threshold=cfg.wake_vad_threshold,
                    min_speech_level=cfg.wake_min_speech_level,
                    wake_threshold=cfg.wake_threshold,
                    vad_onset_threshold=cfg.wake_vad_onset_threshold,
                    min_onset_ms=cfg.wake_min_onset_ms,
                    onset_window_ms=cfg.wake_onset_window_ms,
                    min_voiced_ms=cfg.wake_min_voiced_ms,
                    min_voiced_ratio=cfg.wake_min_voiced_ratio,
                ),
            )
        except Exception as exc:
            # Covers ListenerError and WakeError (bad phrases, missing
            # mlx-whisper). Either way push-to-talk must still come up.
            log.warning("Wake word unavailable (%s) — falling back to push-to-talk", exc)
            return None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the TUI, warm up STT, start the key listener, and block.

        Blocks the main thread in the Rich Live / pynput listener loop until
        the user sends SIGINT (Ctrl+C).
        """
        from pynput import keyboard  # type: ignore[import]

        log.info("Warming up STT model (first transcription is slow)…")
        try:
            self.stt.transcribe(np.zeros(1, dtype=np.float32))
        except Exception:
            log.warning("STT warm-up failed; first turn may be slow", exc_info=True)

        self._restore_panel_geometry()
        self.tui.start()
        # Must run on the MAIN thread: AppKit requires it. Non-blocking — it
        # creates NSApp, the window and the render timer, then returns.
        self.panel.start()

        self._start_listener()
        self._prewarm_brain()

        # Wired regardless of the input path: the language pill is about how
        # Wawi ANSWERS, which has nothing to do with how you talk to it.
        self.panel.on_language_changed(self._on_language_toggled)
        # "Tomar nota" in the menu, and the giant STOP while one is taken.
        self.panel.on_notes_toggled(self._on_notes_toggled)
        # Files, links, text and pictures dragged onto her (wawi/drops.py).
        on_dropped = getattr(self.panel, "on_dropped", None)
        if callable(on_dropped):
            on_dropped(self._on_dropped)
        # The buzón "Wawi investiga" (wawi/inbox.py): drop there, type the order.
        configure_inbox = getattr(self.panel, "configure_inbox", None)
        if callable(configure_inbox):
            configure_inbox(
                enabled_default=inbox_module.enabled_from_config(getattr(self._cfg, "inbox", None)),
                on_drop=self._on_inbox_dropped,
                on_submit=self.submit_typed,
            )

        if self.listener is not None:
            # The panel toggle means "push-to-talk only". Wake-word mode is the
            # default, so it starts OFF (see panel.PanelObserver).
            self.panel.set_listening(True)
            self.panel.on_listening_changed(self._on_listening_toggled)
            if self._phrase_mode_available():
                self.panel.set_wake_mode(self.listener.require_phrase)
                self.panel.on_wake_mode_changed(self._on_wake_mode_toggled)
            if not self._cfg.wake_require_phrase:
                log.info(
                    "Wawi ready. Just talk; the panel switch mutes it. "
                    "Ctrl+C to quit."
                )
            elif self._cfg.wake_engine == "whisper":
                spoken = " / ".join(f'"{p}"' for p in self._cfg.wake_phrases)
                log.info(
                    "Wawi ready. Say %s, or hold %r; Ctrl+C to quit.",
                    spoken,
                    self._cfg.push_to_talk_key,
                )
            else:
                log.info(
                    'Wawi ready. Say "hey jarvis" (openWakeWord\'s pretrained '
                    'phrase), or hold %r; Ctrl+C to quit.',
                    self._cfg.push_to_talk_key,
                )
        else:
            log.info(
                "Wawi ready. Hold %r to speak; Ctrl+C to quit.",
                self._cfg.push_to_talk_key,
            )

        self._print_banner()
        self._offer_resume()
        self._start_sensors()
        if self.reminders is not None:
            self.reminders.start()
        mind = getattr(self, "mind", None)
        if mind is not None:
            # Started only now: it subscribes to the reaction bus and runs the
            # bubble ticker, and neither has anyone to show anything to before
            # the panel and the mascot exist.
            mind.start()
        from wawi import instinct as _instinct

        if _instinct.get() is not None:
            _instinct.get().start()  # same reason: nobody to show anything to before
        from wawi import curiosities as _curiosities

        if _curiosities.get() is not None:
            _curiosities.get().start()  # tops up its cache in the background
        from wawi import music as _music

        if _music.get() is not None:
            _music.get().start()
        from wawi import initiative as _initiative

        if _initiative.get() is not None:
            _initiative.get().start()  # after the panel: her offers need a bubble
        from wawi import selffix as _selffix

        if _selffix.get() is not None:
            _selffix.get().start()  # folds the watchdog's verdict, then waits for idle
            from wawi import initiative as _initiative_gate

            # The initiative layer sees Meet tabs and Zoom calls the reflexes
            # miss: no self-fix (and its restart) in the middle of a call.
            _selffix.add_work_gate(lambda: (not _initiative_gate.in_meeting(), "reunion"))
        from wawi import orca as _orca

        if _orca.get() is not None:
            try:
                _orca.get().resume()  # tasks left in flight by the last run
            except Exception:
                log.warning("Orca watcher failed to resume", exc_info=True)

        target_key = _parse_key(self._cfg.push_to_talk_key)

        def _on_press(key) -> None:  # type: ignore[no-untyped-def]
            if not _key_matches(key, target_key):
                return
            self._on_press()

        def _on_release(key) -> None:  # type: ignore[no-untyped-def]
            if not _key_matches(key, target_key):
                return
            self._on_release()

        try:
            with keyboard.Listener(on_press=_on_press, on_release=_on_release):
                # Park the main thread inside the listener context. pynput runs
                # its own CFRunLoop on its own daemon thread, so the main thread
                # is free for AppKit. run_event_loop() services the panel and
                # falls back to threading.Event().wait() when it is disabled or
                # AppKit is unavailable, so the panel-off path is unchanged.
                # KeyboardInterrupt propagates out of the `with` block cleanly.
                run_event_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._save_panel_geometry()
            self._close_note_on_quit()
            self._stop_sensors()
            if self.reminders is not None:
                self.reminders.stop()
            self._engine.close()
            if self.tasks is not None:
                self.tasks.close()
            if getattr(self, "mind", None) is not None:
                self.mind.stop()
            from wawi import instinct as _instinct

            if _instinct.get() is not None:
                _instinct.get().stop()
            from wawi import orca as _orca

            if _orca.get() is not None:
                _orca.get().stop()
            from wawi import webvm as _webvm

            _webvm.shutdown()  # her computer stops with her (bounded wait)
            if getattr(self, "_pet_voice", None) is not None:
                self._pet_voice.stop()
            self._stop_refresh.set()
            from wawi import music as _music

            if _music.get() is not None:
                _music.get().stop()
            from wawi import initiative as _initiative

            if _initiative.get() is not None:
                _initiative.get().stop()
            from wawi import selffix as _selffix

            if _selffix.get() is not None:
                _selffix.get().stop()
            if self.listener is not None:
                self.listener.stop()
            self.panel.stop()
            self.tui.stop()
            self.printer.banner(f"{self._cfg.assistant_name}: cerrado.")
            log.info("Wawi stopped.")

    def _install_initiative(self, cfg: Config) -> None:
        """Build wawi/initiative with what it borrows from here: whether a
        turn, a note or a game is on, how to start a note, the board, the
        reaction voice, the issues. Never fatal."""
        from wawi import initiative as _initiative
        from wawi.initiative.core import Hooks

        def busy() -> str | None:
            if self.state.state is not State.IDLE:
                return "turn"
            if self._notes is not None:
                return "notes"
            return "game" if _game_on() else None

        def speak(text: str) -> None:
            voice = getattr(self, "_pet_voice", None)
            if voice is not None:
                voice(text)

        hooks = Hooks(
            busy=busy,
            start_note=lambda: self._on_notes_toggled(True),
            notes_active=lambda: self._notes is not None,
            board=self.panel._board,
            speak=speak,
            issues=lambda: [i.title for i in self.issues.issues()],
            notes_dir=self._notes_settings()[0],
        )
        _initiative.install(_initiative.build(cfg, hooks))

    def _start_sensors(self) -> None:
        """Start the sensor thread + reflexes (wawi/sensors), if any are enabled.

        Never fatal: a broken sensor config costs the sensors, not the app.
        """
        from wawi import tools as _tools

        try:
            from wawi import sensors

            running = sensors.start(self._cfg) is not None
            if running:
                # A health check on an agregado is one more sensor, so a dead
                # service gets the same firefighter as a dead website.
                from wawi import plugins as _plugins

                try:
                    _plugins.attach_health()
                except Exception:
                    log.warning("Agregados: health checks not started", exc_info=True)
        except Exception:
            log.warning("Sensors failed to start", exc_info=True)
            running = False
        _tools.set_sensors_active(running)

    def _stop_sensors(self) -> None:
        from wawi import tools as _tools

        try:
            from wawi import sensors

            sensors.stop()
        except Exception:
            log.debug("Sensors failed to stop cleanly", exc_info=True)
        _tools.set_sensors_active(False)

    def _remember_session(self, session_id: str | None) -> None:
        """Carry the SDK session forward so the next turn continues it.

        From here on the model sees the real conversation instead of our
        rendered summary of it, and re-reads it from cache instead of paying
        for it again.
        """
        if not session_id:
            return
        if session_id != self._session_id:
            log.info(
                "conversation session %s (%s)",
                "started" if self._session_id is None else "changed",
                session_id[:8],
            )
        self._session_id = session_id
        self.inflight.remember_session(session_id)

    def _end_turn_session(self, result: "brain.BrainResult") -> None:
        """Decide what the next turn continues, given how this one went.

        A turn that looked at the screen left an image in its session, and
        resuming that session re-sends the image on every later turn until the
        CLI compacts. So it is not resumed: the next turn starts fresh with the
        conversation as text plus one line per capture, and from then on
        resumes normally. The live engine rotates its own client for the same
        reason (see :class:`wawi.brain.Engine`).
        """
        if result.screen_notes:
            if self._session_id is not None or result.session_id:
                log.info(
                    "conversation session not carried over: it holds %d "
                    "screenshot(s)", len(result.screen_notes)
                )
            self._session_id = None
            self.inflight.remember_session(None)
            self._screen_notes = tuple(result.screen_notes)
            return
        self._screen_notes = ()
        self._remember_session(result.session_id)

    def _forget_session(self) -> None:
        """Start the next turn on a fresh conversation.

        Called when a turn fails. If the server rejected the session — expired,
        unknown — resuming it would fail identically on every later turn,
        turning one bad turn into a dead assistant. A fresh conversation loses
        the context; holding a broken id loses everything.
        """
        if self._session_id is not None:
            log.info("Dropping the conversation session after a failed turn")
        self._session_id = None
        self.inflight.remember_session(None)

    # ------------------------------------------------------------------
    # Picking up what a restart cut off (wawi/inflight.py)
    # ------------------------------------------------------------------

    def _offer_resume(self) -> None:
        """Ask, once, whether to pick up the work the last run died in.

        Spoken on a thread under the turn lock, like the executor's voice, so
        it never talks over a turn the user started first. The answer is NOT
        collected here: it arrives as an ordinary utterance, and the question
        ends in "?", so the follow-up channel is open for it with no wake
        phrase. See :meth:`_answer_resume_offer`.
        """
        offer = self._resume_offer
        if offer is None:
            return
        if offer.kind == inflight_module.KIND_TASK and self.tasks is None:
            log.info("Interrupted task dropped: delegated tasks are off now")
            self._resume_offer = None
            return
        from wawi.task import _shorten

        table = (
            lang.RESUME_ASK_TASK
            if offer.kind == inflight_module.KIND_TASK
            else lang.RESUME_ASK_TURN
        )
        sentence = lang.get(table, self._language).format(
            what=_shorten(offer.text, _RESUME_WHAT_CHARS)
        )
        log.info("Offering to resume an interrupted %s: %s", offer.kind, offer.text[:120])

        def _run() -> None:
            got = self._turn_lock.acquire(timeout=_TASK_SPEECH_WAIT_S)
            if not got:
                log.warning("Resume offer dropped: a turn held the lock")
                self._resume_offer = None
                return
            try:
                if self._resume_offer is not offer:
                    return
                self._resume_offer_until = time.monotonic() + _RESUME_OFFER_S
                self._speak(sentence)
            except Exception:
                log.warning("Resume offer could not be spoken", exc_info=True)
            finally:
                self.state.set_state(State.IDLE)
                try:
                    self._turn_lock.release()
                except RuntimeError:
                    pass

        threading.Thread(target=_run, name="wawi-resume-offer", daemon=True).start()

    def _answer_resume_offer(
        self, user_text: str
    ) -> tuple[str, str | None, str | None, bool]:
        """Read *user_text* as the answer to "¿lo retomo?", if one is pending.

        Returns ``(user_text, response, resume, replay)``: the text the turn
        should send to the brain ("" to send nothing), a fixed reply to speak
        instead, the session to resume, and whether that session belongs to a
        replayed turn and so deserves one fresh retry when it fails.

        Only the FIRST utterance after the question counts, and only while it
        is fresh. A no ends it in silence. Anything that is neither yes nor no
        drops the offer and is answered as the ordinary turn it was: the user
        moved on, and making them answer first would be the friction this
        whole feature exists to remove.
        """
        offer = self._resume_offer
        if offer is None or not self._resume_offer_until:
            return user_text, None, self._session_id, False
        self._resume_offer = None
        if time.monotonic() > self._resume_offer_until:
            log.info("Resume offer expired unanswered")
            return user_text, None, self._session_id, False
        verdict = lang.affirmative(user_text)
        if verdict is False:
            log.info("Resume declined")
            return "", None, self._session_id, False
        if verdict is None:
            log.info("Resume offer dropped: the reply was about something else")
            return user_text, None, self._session_id, False
        if offer.kind == inflight_module.KIND_TASK:
            started = None
            if self.tasks is not None:
                started = self.tasks.start(offer.text, resume=offer.session_id)
            if started is None:
                # Only when another task was started in the meantime: that one
                # owns the executor now, and it reports for itself.
                log.warning("Could not resume the interrupted task: one is running")
                return "", None, self._session_id, False
            log.info("Resuming the interrupted task as %s", started.id)
            ok = lang.get(lang.RESUME_TASK_OK, self._language)
            return "", ok, self._session_id, False
        log.info("Replaying the interrupted turn: %r", offer.text[:120])
        self.tui.set_texts(user_text=offer.text)
        self.panel.set_texts(user_text=offer.text)
        return offer.text, None, offer.session_id, bool(offer.session_id)

    def _stt_prompt(self) -> str:
        """Return the decoder seed for the language in force right now.

        Read per turn rather than cached at startup: the language toggle in
        the HUD changes it mid-session, and a prompt written in the language
        you are no longer speaking steers the decoder the wrong way -- which
        is worse than sending none at all.
        """
        override = self._cfg.stt_initial_prompt.strip()
        if override:
            return override
        return lang.stt_prompt(self._language, self._cfg.assistant_name)

    def _on_language_toggled(self, code: str) -> None:
        """Apply a language switch made from the HUD.

        Runs on the AppKit main thread, so it must stay fast: swapping the
        Kokoro voice is two attribute writes (the voice is a per-call argument)
        and everything else is read per turn from ``self._language``. Nothing is
        reloaded, so this is safe to do between turns.

        The wake detector is deliberately NOT rebuilt: it holds the microphone
        and its own Whisper model, and dropping that mid-session to change a
        trigger word the user did not change is the wrong trade.
        """
        self._language = lang.normalise(code)
        try:
            self.voice.set_language(self._language)
        except Exception:
            log.warning("Could not switch the TTS voice", exc_info=True)
        log.info("Language switched to %s (voice %s)", self._language,
                 lang.voice_for(self._language))
        self.printer.banner(f"idioma: {self._language}")

    def _restore_panel_geometry(self) -> None:
        """Put the HUD back where and how the user last left it.

        Position and size are a preference, not state: re-imposing the default
        on every restart is the same as not letting them move it at all. A
        missing or corrupt file is not an error — it just means "first run".
        """
        try:
            data = json.loads(_panel_state().read_text())
        except FileNotFoundError:
            return
        except Exception:
            log.debug("Ignoring unreadable %s", _panel_state(), exc_info=True)
            return
        try:
            if "size" in data:
                self.panel.set_size(*data["size"])
            if "origin" in data:
                self.panel.set_origin(*data["origin"])
            if data.get("panel_hidden"):
                # Someone who hid the panel meant it. Re-showing it on every
                # start is the same as not letting them hide it.
                self.panel.panel_visible = False
            origin = data.get("mascot_origin")
            if origin:
                self.panel.set_mascot_origin(*origin)
        except Exception:
            log.debug("Could not restore the panel geometry", exc_info=True)

    def _save_panel_geometry(self) -> None:
        """Persist where and how big the HUD is. Never raises."""
        try:
            data = {}
            origin = self.panel.get_origin()
            size = self.panel.get_size()
            if origin:
                data["origin"] = list(origin)
            if size:
                data["size"] = list(size)
            # Under ui.mascot_only the hiding is config's, not a choice to
            # remember: turning the key off later must bring the panel back.
            if not self.panel.panel_visible and not getattr(self._cfg, "mascot_only", False):
                data["panel_hidden"] = True
            mascot_origin = self.panel.get_mascot_origin()
            if mascot_origin:
                data["mascot_origin"] = list(mascot_origin)
            if data:
                _panel_state().parent.mkdir(parents=True, exist_ok=True)
                _panel_state().write_text(json.dumps(data))
        except Exception:
            log.debug("Could not save the panel geometry", exc_info=True)

    def _print_banner(self) -> None:
        """Say, in the terminal, how to talk to it and where the answers go.

        The log already says all of this, but the log is a file nobody is
        looking at. A process that prints nothing on startup is
        indistinguishable from one that failed to start.
        """
        name = self._cfg.assistant_name
        if self.listener is not None and self._cfg.wake_require_phrase:
            spoken = " / ".join(f'"{p}"' for p in self._cfg.wake_phrases) or f'"{name}"'
            how = f"deci {spoken}"
        elif self.listener is not None:
            how = 'hablá; el interruptor "escuchando" del panel lo silencia'
        else:
            how = f"mantené {self._cfg.push_to_talk_key!r}"
        self.printer.banner(
            f"{name} escuchando — {how}. Ctrl+C para salir."
        )

    def _start_listener(self) -> None:
        """Open the mic and start wake-word detection, or degrade gracefully.

        A listener that cannot start (no mic, weights vanished between the
        constructor and here) must not take Wawi down: it is dropped and the
        Recorder-backed push-to-talk path takes over, with the UI rewired to
        the Recorder so the spectrum keeps working.
        """
        if self.listener is None:
            return
        try:
            self.listener.start()
        except ListenerError as exc:
            log.warning(
                "Could not start the wake-word listener (%s) — falling back to "
                "push-to-talk", exc,
            )
            self.listener = None
            self.tui.set_level_source(self.recorder.current_level)
            self.panel.set_samples_source(self.recorder.current_block)

    def _prewarm_brain(self) -> None:
        """Hide the first SDK connection behind the time before a voice turn."""
        if self._cfg.backend != "claude":
            return
        from wawi import plugins
        from wawi.prompts import system_prompt

        plugins.sync(self._connectors)
        self._engine.prewarm(
            model=self._cfg.model_on_demand,
            system=system_prompt(
                self._cfg.assistant_name,
                self.preferences.as_prompt_block(),
                self.issues.prompt_hint(),
                language=self._language,
                recall=self.recall.as_prompt_block(self._cfg.assistant_name),
            ),
            max_iterations=brain.MAX_TOOL_ITERATIONS,
            screen_options=brain.ScreenOptions(
                monitor=self._cfg.monitor,
                max_edge=self._cfg.max_edge,
                jpeg_quality=self._cfg.jpeg_quality,
            ),
            bypass_permissions=self._cfg.dangerous_skip_permissions,
            streaming=self._cfg.tts_streaming,
            connectors=self._connectors,
        )

    # ------------------------------------------------------------------
    # Key-event callbacks (run on pynput's daemon thread — must be fast)
    # ------------------------------------------------------------------

    def _on_press(self) -> None:
        """Handle push-to-talk key-press.

        In listener mode the key opens a forced capture on the microphone the
        listener already owns; the turn lock is not taken here, because every
        utterance — wake word or key — competes for it in one place,
        :meth:`_on_utterance`.

        Barge-in: a press while Wawi is SPEAKING interrupts the TTS and waits
        briefly for the turn to unwind, so the listener is resumed before the
        capture opens (while SPEAKING it is paused and would drop the press).
        """
        if self.state.state is State.SPEAKING:
            log.debug("Key press during SPEAKING — barge-in: interrupting TTS")
            self.voice.interrupt()
            # The worker's finally sets IDLE and then releases the lock, so
            # holding it for an instant proves the turn is fully unwound.
            # Bounded: this runs on pynput's daemon thread and must stay fast.
            if self._turn_lock.acquire(timeout=1.0):
                self._turn_lock.release()
            elif self.state.state is not State.SPEAKING:
                # A STREAMED answer was cut off: the voice stopped but the
                # brain behind it is still writing and still holds the turn
                # lock, so waiting for it would drop this press for as long as
                # the model feels like taking. Leaving SPEAKING already resumed
                # the listener, and speech during THINKING is HELD and answered
                # when the turn unwinds — see _dispatch_turn.
                log.info("Barge-in: answer cut off; capturing while it unwinds")
            else:
                log.warning("Barge-in: turn did not unwind within 1s; ignoring press")
                return

        if self.listener is not None:
            self.listener.force_start()
            return

        # Fallback path: no listener, so the Recorder opens its own stream and
        # the press itself gates the turn.
        if not self._turn_lock.acquire(blocking=False):
            log.debug("Key press ignored — turn already in flight (%s)", self.state.state.value)
            return
        self._turn_acquired = True
        self.recorder.start()
        self.state.set_state(State.LISTENING)

    def _on_release(self) -> None:
        """Handle push-to-talk key-release: close the capture and dispatch it."""
        if self.listener is not None:
            self.listener.force_stop()
            return

        if not self._turn_acquired:
            return

        self._turn_acquired = False
        audio_buf = self.recorder.stop()
        self._dispatch_turn(audio_buf, already_locked=True)

    # ------------------------------------------------------------------
    # Listener callbacks (run on the listener's worker thread — must be fast)
    # ------------------------------------------------------------------

    def _on_capture_start(self) -> None:
        """The listener opened an utterance: show LISTENING if Wawi is idle.

        Guarded on IDLE so a capture opened while an earlier turn is still
        THINKING does not paint over that turn's state. The utterance itself
        will be dropped by :meth:`_on_utterance` for the same reason.
        """
        if self._notes is not None:
            # Taking notes: every capture is a line, not a turn, and she shows
            # it with her recorder -- no LISTENING, so no earcon either.
            return
        warm = getattr(self.voice, "warm", None)
        if callable(warm):
            warm()  # an answer is coming: have the voice ready for it
        if self.state.state is State.IDLE:
            self.state.set_state(State.LISTENING)

    def _on_capture_dropped(self) -> None:
        """The listener threw away a capture it had announced (noise, echo).

        Undo the LISTENING painted by :meth:`_on_capture_start`, or the HUD
        and Wawi keep saying "escuchando" to a door slam until the next turn.
        """
        if self.state.state is State.LISTENING:
            self.state.set_state(State.IDLE)

    def _transcribe_filtered(self, audio_buf: np.ndarray) -> str:
        """Transcribe, then drop Whisper's hallucinations on noise.

        Returns "" for a dropped transcript, which the turn already treats as
        "nothing was said". The reason is logged at INFO so the thresholds in
        config.yaml (stt.*) can be tuned from wawi.log.
        """
        prompt = self._stt_prompt()
        detailed = getattr(self.stt, "transcribe_detailed", None)
        if callable(detailed):
            text, metrics = detailed(
                audio_buf, language=self._language, initial_prompt=prompt
            )
        else:
            text, metrics = self.stt.transcribe(
                audio_buf, language=self._language, initial_prompt=prompt
            ), None
        cfg = self._cfg
        reason = speechgate.judge(
            text,
            metrics=metrics,
            evidence=getattr(audio_buf, "speech", None),
            prompt=prompt,
            cfg=speechgate.FilterConfig(
                enabled=getattr(cfg, "stt_filter_junk", True),
                min_avg_logprob=getattr(cfg, "stt_min_avg_logprob", -1.0),
                max_compression_ratio=getattr(cfg, "stt_max_compression_ratio", 2.4),
                max_no_speech_prob=getattr(cfg, "stt_max_no_speech_prob", 0.6),
                junk_min_voiced_ms=getattr(cfg, "stt_junk_min_voiced_ms", 700),
                junk_min_logprob=getattr(cfg, "stt_junk_min_logprob", -0.45),
            ),
        )
        if metrics is not None:
            log.info(
                "STT confidence: avg_logprob=%.2f no_speech=%.2f compression=%.2f",
                metrics.avg_logprob, metrics.no_speech_prob, metrics.compression_ratio,
            )
        if reason is not None and reason.startswith("echo of the STT prompt") and getattr(
            audio_buf, "wake", None
        ):
            # The prompt carries the name, so "Wawi." alone looks like its
            # echo -- but the wake detector already heard the name in this very
            # audio. Dropping it would swallow the bare summon ("¿Sí?").
            log.info("Kept %r: looks like the prompt, but the wake word was heard", text)
            reason = None
        elif reason is not None and reason.startswith("echo of the STT prompt") \
                and self._lone_name(text, getattr(audio_buf, "speech", None)):
            # In the reply window nothing checks for the name, so "Wawi." said
            # there looked like Whisper echoing its prompt and was dropped:
            # she ignored being called. A real spoken name has a voiced core.
            log.info("Kept %r: the name alone, said in the reply window", text)
            reason = None
        if reason is not None:
            log.info("Dropped transcript as junk: %s: %r", reason, text)
            return ""
        return text

    def _hold_utterance(self, audio_buf: np.ndarray) -> None:
        """Keep the latest utterance heard mid-turn, to answer when we can.

        Latest wins rather than a queue: if you said three things while Wawi
        was thinking, answering all three in a row is worse than answering the
        last one -- by then the first two are usually rephrasings of it.
        """
        with self._pending_lock:
            replaced = self._pending_utterance is not None
            self._pending_utterance = (audio_buf, time.monotonic())
        log.info(
            "Heard you mid-turn; holding it for when this turn ends%s",
            " (replacing an earlier one)" if replaced else "",
        )
        self.panel.set_texts(user_text="…te escuché, dame un segundo")

    def _take_held_utterance(self) -> "np.ndarray | None":
        """Return the held utterance if it is still worth answering.

        Stale ones are thrown away: an answer to something you said a minute
        ago, after you already gave up and said it again, is not a late answer.
        It is a second conversation starting on its own.
        """
        with self._pending_lock:
            held, self._pending_utterance = self._pending_utterance, None
        if held is None:
            return None
        audio_buf, heard_at = held
        age = time.monotonic() - heard_at
        if age > _HELD_UTTERANCE_MAX_AGE_S:
            log.info("Held utterance is %.0fs old — too stale to answer", age)
            return None
        log.info("Answering the utterance held %.1fs ago", age)
        return audio_buf

    def _on_utterance(self, audio_buf: np.ndarray) -> None:
        """The listener closed an utterance. Runs on its worker thread.

        Must not block: while this runs the listener processes no audio. It
        only takes the turn lock and hands off to a worker thread.
        """
        self._dispatch_turn(audio_buf, already_locked=False)

    def _dispatch_turn(self, audio_buf: np.ndarray, already_locked: bool) -> None:
        """Take the turn lock (unless held) and spawn the pipeline worker."""
        if self._notes is not None and self._queue_note(audio_buf):
            # Taking notes: a line for the note, never a turn for the brain.
            if already_locked:
                self.state.set_state(State.IDLE)
                self._turn_lock.release()
            return
        if not already_locked and not self._turn_lock.acquire(blocking=False):
            # Barge-in: the wake phrase arrived while Wawi was still talking.
            # Cut the answer off and take the turn, the way interrupting a
            # person works. Only reachable when _barge_in_safe allowed it.
            if self._barge_in_armed and self.state.state is State.SPEAKING:
                log.info("Barge-in: wake phrase heard mid-answer — cutting TTS off")
                self.voice.interrupt()
                # The worker's finally sets IDLE then releases; a bounded wait
                # keeps the listener thread from stalling if that goes wrong.
                if not self._turn_lock.acquire(timeout=1.5):
                    log.warning("Barge-in: turn did not unwind in 1.5s; utterance dropped")
                    return
            elif self.state.state is State.THINKING:
                # Held, not dropped. THINKING means Wawi is NOT talking, so
                # this is really you and not the microphone hearing our own
                # playback -- which is the whole reason the SPEAKING case below
                # still discards. A person interrupted mid-thought answers late;
                # they do not pretend they heard nothing.
                self._hold_utterance(audio_buf)
                if self.state.state is State.LISTENING:
                    self.state.set_state(State.IDLE)
                return
            else:
                log.warning(
                    "IGNORED YOU: speech arrived but a turn was already in "
                    "flight (%s). From outside this is indistinguishable from a "
                    "broken wake word.",
                    self.state.state.value,
                )
                self.issues.record(
                    selfimprove.KIND_UTTERANCE_DROPPED,
                    "Te ignoró: hablaste mientras había un turno en curso",
                    detail=(
                        "El usuario habló pero el turno anterior seguía corriendo "
                        "y no era seguro encolarlo, así que se descartó. Desde "
                        "afuera se ve como que no contestó."
                    ),
                    subject=f"dropped during {self.state.state.value}",
                    context={"state": self.state.state.value},
                )
                # Undo the LISTENING we optimistically painted on capture start.
                if self.state.state is State.LISTENING:
                    self.state.set_state(State.IDLE)
                return

        self.state.set_state(State.THINKING)
        worker = threading.Thread(
            target=self._do_turn,
            args=(audio_buf,),
            daemon=True,
            name="wawi-worker",
        )
        worker.start()

    def _on_state_change(self, old: State, new: State) -> None:
        """Manage the microphone around Wawi's own voice.

        Two modes, and the difference is whether the user can cut in:

        - barge-in OFF: the listener is paused outright while speaking, which
          is airtight against self-triggering but means the only way to stop a
          long answer is the keyboard.
        - barge-in ON: the listener keeps running and the wake phrase cuts the
          answer off mid-sentence, like interrupting a person. The risk is
          Wawi waking on its OWN voice through the speakers, so it is only
          allowed for responses that do not themselves contain the phrase —
          see :meth:`_barge_in_safe`.
        """
        if self.listener is None:
            return
        if new is State.SPEAKING:
            if self._barge_in_armed:
                # Keep listening, but only for audio clearly louder than the
                # assistant's own playback — otherwise it transcribes itself
                # continuously and falls behind realtime.
                self.listener.set_speech_floor(self._cfg.barge_in_level)
                log.debug(
                    "Barge-in armed (floor %.2f) — say the wake phrase to cut in",
                    self._cfg.barge_in_level,
                )
            else:
                self.listener.pause()
        elif old is State.SPEAKING:
            self.listener.set_speech_floor(None)
            self.listener.resume()
            self._barge_in_armed = False
            # Hold the channel open only when the answer actually expects a
            # reply. See _follow_up_seconds.
            seconds = self._follow_up_seconds()
            if seconds > 0:
                self.listener.arm_follow_up(seconds)
                self._open_channel(seconds)
                log.info("Channel open %.0fs for your reply — no wake phrase needed", seconds)
            else:
                self.panel.channel_open = False
                log.info("Answer needed no reply — say the wake phrase to continue")
        elif new is State.THINKING:
            # A new turn started: the previous invitation is over.
            self.panel.channel_open = False

    def _open_channel(self, seconds: float) -> None:
        """Show "your turn" for the length of the follow-up window."""
        self.panel.channel_open = True

        def _close() -> None:
            if self._stop_refresh.wait(seconds):
                return
            self.panel.channel_open = False

        threading.Thread(target=_close, name="wawi-channel", daemon=True).start()

    def _follow_up_seconds(self) -> float:
        """How long to keep listening after an answer, or 0 to close the channel.

        A fixed timer gets this wrong in both directions: it hangs open after a
        statement that needed no reply (so any nearby speech starts a turn), and
        it can expire while the user is still thinking about a question that was
        just put to them.

        So the ANSWER decides, which is the only thing that knows:

        - It ended in a question -> a reply is expected, keep the channel open.
        - It stated something -> ``wake_follow_up_idle_s``.

        Conversation mode makes the last case the same as the others: the
        channel stays open after EVERY answer. That is the right default for
        talking about one subject over several turns, where insisting on the
        wake phrase between exchanges breaks the thread. The cost is real and
        is why it is a setting: an always-open channel means speech near the
        machine can start a turn you did not intend.
        """
        if self._drop_follow_up > 0:
            # She just caught something and asked what to do with it: the
            # window for the order, whatever the mode.
            return self._drop_follow_up
        if getattr(self, "_reaction_speech", False) or self._aside_speech:
            # She spoke up on her own ("¡se cayó la web!"): nobody asked
            # anything, so nothing said nearby should start a turn.
            return 0.0
        if self._notes is not None:
            # Taking notes: what is said next is dictation, not a reply.
            return 0.0
        if _game_on():
            # Mid-game every reply expects the next move: insisting on the
            # wake phrase before each "piedra" would kill the game.
            return self._cfg.wake_follow_up_s
        # Also in wake-phrase mode: «Wawi» starts a conversation, not every
        # sentence of it. What ends a chat with someone else in the room is
        # the model answering (silencio), which opens no window at all.
        if self._cfg.conversation_mode:
            return self._cfg.wake_follow_up_s

        response = (self._last_response or "").strip()
        if not response:
            return 0.0
        # Spanish opens questions too, and TTS text keeps both marks.
        if response.endswith("?") or response.endswith("¿") or "¿" in response[-80:]:
            return self._cfg.wake_follow_up_s
        return self._cfg.wake_follow_up_idle_s

    def _speak(self, response: str) -> None:
        """Say *response* with every piece of bookkeeping that hangs off it.

        Extracted from :meth:`_do_turn` because the delegated executor has
        things to say too (see :mod:`wawi.task`), and a second copy of this
        would drift. When it drifts, the microphone stays open while the
        assistant talks and it starts answering itself — which is not a bug
        anyone finds by reading the code.

        Does NOT return to IDLE: the caller owns that, because it is the same
        caller that owns the turn lock.
        """
        # Decide barge-in BEFORE the state change, since the observer reads
        # this flag as it fires.
        self._barge_in_armed = self._barge_in_safe(response)
        self._present(response)
        self.state.set_state(State.SPEAKING)
        self.voice.say(response)
        self._clear_echo_after_tail()

    def _present(self, response: str) -> None:
        """Show an answer and hand it to everything that must know its text.

        Split out of :meth:`_speak` because a streamed answer is shown and
        remembered at the END, once the whole text exists, while it was spoken
        from the first sentence onwards. Both paths must run exactly this, or
        the split becomes two sets of bookkeeping that drift.
        """
        self.tui.set_texts(response=response)
        # Printed BEFORE the TTS call on purpose: synthesis takes real time and
        # can fail, and the text is the part the user asked to keep. It must be
        # readable even when nothing is ever spoken.
        self.printer.answer(response)
        self.panel.set_texts(response=response)
        # Tell the listener its own script, so anything it hears that repeats
        # it is discarded as speaker bleed rather than answered. Remembered so
        # the follow-up decision can read it once SPEAKING unwinds.
        self._last_response = response
        if self.listener is not None:
            self.listener.set_echo_text(response)

    def _begin_streamed_speech(self, so_far: str, *, first: bool) -> None:
        """Bookkeeping before each sentence of an answer still being written.

        Runs on the streaming speaker's thread. Only the first sentence moves
        the state machine, so the listener is paused once and stays paused for
        the whole answer.

        Wake-phrase barge-in is refused outright here. :meth:`_barge_in_safe`
        answers a question about the WHOLE answer — does it say the wake phrase
        out loud — and mid-stream that answer does not exist yet. Arming on the
        first sentence and discovering the assistant's own name in the fourth
        is how it ends up waking itself, so a streamed answer keeps the paused
        microphone. Interrupting with the push-to-talk key still works.
        """
        self.panel.set_texts(response=so_far)
        self.tui.set_texts(response=so_far)
        if self.listener is not None:
            self.listener.set_echo_text(so_far)
        if first:
            self._barge_in_armed = False
            self.state.set_state(State.SPEAKING)

    def _streamed_speech_aborted(self) -> None:
        """A streamed answer was cut off mid-way; stop calling this SPEAKING.

        Back to THINKING rather than straight to IDLE, because that is what is
        actually happening: the voice stopped but the brain is still writing,
        and the turn lock is still held. It also puts the microphone back where
        the rest of the pipeline expects it during a turn — the listener
        resumes, and anything said now is HELD and answered when this turn
        unwinds, instead of being dropped as speech during SPEAKING.
        """
        if self.state.state is State.SPEAKING:
            self.state.set_state(State.THINKING)

    # ------------------------------------------------------------------
    # Delegated work — the orchestrator half of the two-agent split
    # ------------------------------------------------------------------

    def _build_task_runner(self, cfg: Config):
        """Build the executor, or None when tasks are switched off.

        Off is the default. The executor is the only thing in Wawi that can
        run a command, so it exists only because someone wrote it into their
        config, and until then its four tools are not even sent to the model.
        """
        if not getattr(cfg, "tasks_enabled", False):
            return None
        from wawi import task as task_module

        runner = task_module.TaskRunner(
            task_module.TaskOptions(
                model=cfg.model_on_demand,
                cwd=cfg.tasks_cwd or str(Path.cwd()),
                tools=tuple(cfg.tasks_tools),
                ask_timeout_s=cfg.tasks_ask_timeout_s,
                approval_timeout_s=cfg.tasks_approval_timeout_s,
                grant_minutes=cfg.tasks_grant_minutes,
                yolo_at_start=cfg.dangerous_skip_permissions,
            ),
            inflight=self.inflight,
        )
        runner.mailbox.set_observer(self._on_task_event)
        log.info(
            "Delegated tasks ENABLED — the executor may run %s in %s",
            ", ".join(cfg.tasks_tools),
            cfg.tasks_cwd or str(Path.cwd()),
        )
        return runner

    def _build_mind(self, cfg: Config):
        """Build Wawi's mind (wawi/mind/) and hand its tools to the brain.

        Building is cheap -- no model loads and no SDK starts until something
        needs them -- and a failure here must never cost the assistant, so it
        degrades to "no mind" with a warning.
        """
        from wawi import mind as mind_module
        from wawi import tools as _tools

        try:
            mind = mind_module.build(cfg)
        except Exception:
            log.warning("mind: could not be built; running without it", exc_info=True)
            mind = None
        mind_module.install(mind)
        _tools.set_mind(mind)
        return mind

    def _on_task_event(self, event: str) -> None:
        """React to the executor. Runs on the EXECUTOR's thread — stay fast.

        This is the whole orchestration seam. Progress is shown, never spoken:
        an assistant narrating its own work out loud is unusable next to
        someone trying to think. Only two things get the speaker — a question
        that blocks the work, and the outcome.
        """
        from wawi import task as task_module

        runner = self.tasks
        if runner is None:
            return
        try:
            self.panel.set_task(runner.is_running, runner.mailbox.latest())
        except Exception:
            log.debug("Could not update the panel's task line", exc_info=True)

        if event == task_module.EVENT_REQUEST:
            request = runner.mailbox.pending()
            if request is not None and runner.mailbox.mark_spoken(request.id):
                self._speak_for_task(_request_sentence(request, self._language))
        elif event == task_module.EVENT_FINISHED:
            current = runner.current
            if current is not None and current.status is task_module.TaskStatus.DONE:
                # A job well done gets the big celebration (with a jump); she
                # does it once she is idle again, after saying so.
                try:
                    self.panel.celebrate(big=True)
                except Exception:
                    log.debug("Could not make Wawi celebrate", exc_info=True)
            self._speak_for_task(_outcome_sentence(current, self._language))

    def _speak_for_task(self, text: str) -> None:
        """Say something on the executor's behalf, on a thread of our own.

        Spawned rather than spoken inline for two reasons that both end in a
        wedged executor: this runs on the executor's thread, and speaking takes
        the turn lock, which a live conversational turn is holding.

        The turn lock is what keeps the two agents from talking over each
        other. Waiting for it bounded — rather than forever, or not at all —
        means a question asked while the user is mid-conversation is spoken
        right after that answer instead of on top of it.
        """
        if not text:
            return
        if self._notes is not None:
            # Taking notes: shown on the HUD, not said into the recording.
            log.info("Task speech not spoken while taking notes: %s", text[:120])
            return

        def _run() -> None:
            got = self._turn_lock.acquire(timeout=_TASK_SPEECH_WAIT_S)
            if not got:
                # The question is still pending and still visible on the HUD,
                # so it is not lost — only unspoken. Saying it over a live turn
                # would cost the user the answer they were listening to.
                log.warning("Task speech dropped: a turn held the lock for %.0fs",
                            _TASK_SPEECH_WAIT_S)
                return
            try:
                self._speak(text)
            except Exception:
                log.warning("Task speech failed", exc_info=True)
            finally:
                self.state.set_state(State.IDLE)
                try:
                    self._turn_lock.release()
                except RuntimeError:
                    pass

        threading.Thread(target=_run, name="wawi-task-voice", daemon=True).start()

    def _speak_reaction(self, text: str) -> bool:
        """Say a reaction's line (``Reaction.speak``) if, and only if, idle.

        Called by :class:`wawi.petvoice.PetVoice`, which owns WHEN (queue,
        staleness, quiet mode, rate). This owns HOW, and it is the same how as
        an answer: barge-in decided by :meth:`_barge_in_safe`, the listener
        paused through the SPEAKING state change, its echo text set, the echo
        tail cleared. A second, simpler path would be the one that lets her
        hear herself.

        Returns False without saying anything when a turn (or the executor's
        voice) holds the pipeline, so the caller retries later. Never blocks
        for the lock: waiting here would only queue her behind the user.
        """
        if not text or self.state.state is not State.IDLE:
            return False
        from wawi.initiative.meeting import in_meeting

        if in_meeting():
            return False  # never over a call; PetVoice / the reminder retry later
        if self._notes is not None:
            # Taking notes: she would be talking into her own recording.
            # PetVoice retries later; a reminder waits for the note to end.
            return False
        if getattr(self.panel, "channel_open", False):
            # The owner's turn to talk: a remark now would close the window
            # they are about to answer in. PetVoice retries later.
            return False
        if not self._turn_lock.acquire(blocking=False):
            return False
        try:
            # Re-checked under the lock: a capture may have opened (LISTENING)
            # between the first look and the acquire.
            if self.state.state is not State.IDLE:
                return False
            self._reaction_speech = True
            self._barge_in_armed = self._barge_in_safe(text)
            if self.listener is not None:
                self.listener.set_echo_text(text)
            self.state.set_state(State.SPEAKING)
            try:
                self.voice.say(text)
            finally:
                self._clear_echo_after_tail()
                self.state.set_state(State.IDLE)
            return True
        finally:
            self._reaction_speech = False
            self._turn_lock.release()

    def _build_pet_voice(self, cfg: Config) -> None:
        """Connect the body's ``speak`` reactions to the voice. Never fatal."""
        try:
            from wawi import petvoice

            self._pet_voice = petvoice.build(cfg, self._speak_reaction)
            self.panel.set_pet_speaker(self._pet_voice)
        except Exception:
            log.warning("Spoken reactions unavailable", exc_info=True)
            self._pet_voice = None

    def _announce_reminder(self, text: str) -> bool:
        """Retry while a turn owns the voice; explicit reminders are not news."""
        if not self._speak_reaction(text):
            return False
        from wawi.reactions import BUS, Reaction

        BUS.push(Reaction(behaviour="bark", sound="bark", bubble=text,
                          speak=False, priority=3))
        return True

    def _run_routine(self, instruction: str) -> bool:
        """Run an autonomous brain turn only when no owner turn is active."""
        if self._notes is not None:
            return False  # retried after the note: no brain, no voice now
        if self.state.state is not State.IDLE or not self._turn_lock.acquire(blocking=False):
            return False
        try:
            if self.state.state is not State.IDLE:
                return False
            self.state.set_state(State.THINKING)
            try:
                result = brain.ask_sync(
                    instruction,
                    model=self._cfg.model_on_demand,
                    backend=self._cfg.backend,
                    ollama_model=self._cfg.ollama_model,
                    ollama_url=self._cfg.ollama_url,
                    assistant_name=self._cfg.assistant_name,
                    history=self.conversation.as_prompt_context(self._cfg.assistant_name),
                    preferences=self.preferences.as_prompt_block(),
                    issues_hint=self.issues.prompt_hint(),
                    recall=self.recall.as_prompt_block(self._cfg.assistant_name),
                    bypass_permissions=self._cfg.dangerous_skip_permissions,
                    language=self._language,
                    resume=self._session_id,
                    engine=self._engine,
                    connectors=self._connectors,
                    screen_options=brain.ScreenOptions(
                        monitor=self._cfg.monitor,
                        max_edge=self._cfg.max_edge,
                        jpeg_quality=self._cfg.jpeg_quality,
                    ),
                )
                self._end_turn_session(result)
                self.conversation.add(instruction, result.text, result.tool_calls)
                self.recall.add(instruction, result.text)
                answer = result.text or "La rutina terminó sin un resultado para contarte."
            except Exception:
                log.warning("Scheduled routine failed", exc_info=True)
                self._forget_session()
                answer = "No pude completar la rutina programada."
            from wawi.reactions import BUS, Reaction

            BUS.push(Reaction(behaviour="bark", sound="bark", bubble=answer,
                              speak=False, priority=3))
            self._speak(answer)
            return True
        finally:
            self.state.set_state(State.IDLE)
            self._turn_lock.release()

    def _clear_echo_after_tail(self) -> None:
        """Forget the spoken text, but not immediately.

        Room reverberation and the speaker's own decay outlive the last sample
        of playback. Clearing on the state change let the follow-up window open
        on the tail of the assistant's own sentence — which is exactly how it
        ended up answering itself.
        """
        if self.listener is None:
            return

        def _clear() -> None:
            if self._stop_refresh.wait(_ECHO_TAIL_S):
                return
            if self.listener is not None:
                self.listener.set_echo_text("")

        threading.Thread(target=_clear, name="wawi-echo-tail", daemon=True).start()

    def _barge_in_safe(self, response: str) -> bool:
        """Can the user interrupt THIS response without Wawi waking itself?

        Only when the response does not contain a wake phrase. Without an
        acoustic echo canceller the microphone hears the speakers, so an answer
        that says its own name out loud would trigger a turn about itself.
        Rare, but a loop that answers itself is bad enough to be worth the
        cheap check: those responses fall back to the paused behaviour.
        """
        if not self._cfg.barge_in or self.listener is None:
            return False
        if self._cfg.wake_engine != "whisper":
            # openwakeword scores continuously and would re-fire on its own
            # phrase with no transcript to check it against.
            return False
        from wawi.wake import normalize

        spoken = normalize(response)
        for phrase in self._cfg.wake_phrases:
            if normalize(phrase) in spoken:
                log.info(
                    "Barge-in disabled for this answer: it says %r out loud", phrase
                )
                return False
        return True

    @staticmethod
    def _build_verifier(cfg: Config):
        """The acoustic second opinion for the phrase mode, or None."""
        name = getattr(cfg, "wake_verify_model", "")
        if not name:
            return None
        from wawi.listener import custom_wake_model_missing, resolve_wake_model

        if custom_wake_model_missing(name):
            log.info("No acoustic verifier (%s not found): Whisper alone decides", name)
            return None
        from wawi.wake import AcousticVerifier

        return AcousticVerifier(resolve_wake_model(name)[1], cfg.wake_verify_threshold)

    # ------------------------------------------------------------------
    # Always-on vs wake phrase (wawi/wakemode.py)
    # ------------------------------------------------------------------

    def _phrase_mode_available(self) -> bool:
        listener = self.listener
        return (listener is not None and listener.config.engine == "whisper"
                and getattr(listener, "_wake_detector", None) is not None)

    def _set_wake_mode(self, require_phrase: bool) -> bool:
        """Switch the listening mode live and remember it. Any thread."""
        if not self._phrase_mode_available():
            return False
        if not self.listener.set_require_phrase(require_phrase):
            return False
        from wawi import wakemode

        wakemode.write_local(self._cfg.wake_local_file, require_phrase)
        self.panel.set_wake_mode(require_phrase)
        self.printer.banner(
            "palabra clave: " + (" / ".join(self._cfg.wake_phrases) if require_phrase
                                 else "no, escucha siempre")
        )
        return True

    def _on_wake_mode_toggled(self, require_phrase: bool) -> None:
        """Menu item. Main thread: the switch itself never blocks."""
        self._set_wake_mode(require_phrase)

    def _local_reply(self, user_text: str, audio_buf: np.ndarray) -> tuple[str, str | None]:
        """Handle what needs no model: taking notes, the mode command and the
        bare summon.

        Returns ``(text_for_the_brain, fixed_reply)``. The wake phrase is
        taken off the front of the request ("Wawi, ¿qué hora es?" is asked as
        "¿qué hora es?"); the name alone ("Wawi", "che Wawi") gets a short
        "¿Sí?" that opens the channel for the actual request instead of a
        paid model turn -- which is also what a false wake ("Huawei") costs
        now: one "¿sí?", not an answer to a question nobody asked.
        """
        from wawi import wakemode

        matched, rest = self._split_wake(user_text)

        if notes_module.parse_start(rest):
            log.info("Note command %r — taking notes, no model", rest)
            return "", self._start_notes()

        mode = wakemode.parse_command(rest)
        if mode is not None and self._phrase_mode_available():
            log.info("Mode command %r -> %s", rest, "wake phrase" if mode else "always-on")
            self._set_wake_mode(mode)
            return "", wakemode.confirmation(mode, self._cfg.assistant_name)

        if matched is not None and not rest:
            log.info("Bare summon (%r) — answering locally and listening", user_text)
            return "", lang.get(lang.SUMMON_TEXT, self._language)
        if matched is not None:
            log.info("Wake phrase stripped: %r -> %r", user_text, rest)
        return rest, None

    #: Voiced speech a lone "Wawi" needs to be taken as said, not as
    #: Whisper echoing its prompt over a breath or a click.
    _LONE_NAME_MIN_VOICED_MS = 300

    def _lone_name(self, text: str, evidence: object) -> bool:
        """*text* is just the wake phrase, with a voice behind it."""
        if evidence is None or getattr(evidence, "voiced_ms", 0) < self._LONE_NAME_MIN_VOICED_MS:
            return False
        matched, rest = self._split_wake(text)
        return matched is not None and not rest.strip(" .,!¡?¿")

    def _split_wake(self, user_text: str) -> tuple[str | None, str]:
        """``(matched_phrase, rest)``: the wake phrase taken off the front,
        or ``(None, user_text)`` when it is not there."""
        from wawi.wake import split_request

        detector = getattr(self.listener, "_wake_detector", None) if self.listener else None
        if detector is not None and hasattr(detector, "split"):
            matched, rest = detector.split(user_text)
        else:
            matched, rest = split_request(user_text, self._cfg.wake_phrases or ("wawi",))
        if matched is None:
            rest = user_text
        return matched, rest

    def _on_listening_toggled(self, listening: bool) -> None:
        """Panel switch: ON means Wawi hears you, OFF means it does not.

        The stream itself stays open either way. That is deliberate: closing it
        would drop the pre-roll ring, so the first syllable after switching back
        on would be lost, and reopening a CoreAudio input device is exactly the
        operation that is unreliable enough to lose the microphone entirely.
        Gating the speech instead is instant and cannot fail.
        """
        self._listening = bool(listening)
        # The owner's own click: a drop window that had switched the
        # microphone on has nothing left to put back.
        self._drop_unmuted = False
        if self.listener is not None:
            self.listener.set_wake_enabled(self._listening)
        self.printer.banner("escuchando" if self._listening else "micrófono en silencio")
        if not self._listening and self._notes is not None:
            # A deaf recorder records nothing: the note ends with the mic.
            threading.Thread(target=self._stop_notes, args=("mic off",),
                             name="wawi-notes-stop", daemon=True).start()

    # ------------------------------------------------------------------
    # Taking notes (wawi/notes.py): dictation to a file, no model
    # ------------------------------------------------------------------

    def _notes_settings(self) -> tuple[Path, float]:
        """(where notes go, seconds of silence that end a session)."""
        section = getattr(self._cfg, "notes", None) or {}
        directory = str(section.get("dir") or notes_module.DEFAULT_DIR)
        try:
            idle_s = float(section.get("idle_stop_min", 30)) * 60.0
        except (TypeError, ValueError):
            idle_s = notes_module.IDLE_STOP_S
        return Path(directory).expanduser(), idle_s if idle_s > 0 else notes_module.IDLE_STOP_S

    def _start_notes(self) -> str:
        """Open a note and start taking it. Returns what to say. Any thread.

        The listener goes always-on for the session (every utterance is a
        line, with or without "Wawi") and is put back as it was at the end;
        the switch is not saved to wake_local.json, it is not the owner's
        choice of mode.
        """
        with self._notes_lock:
            if self._notes is not None:
                return notes_module.BUSY_TEXT
            directory, idle_s = self._notes_settings()
            try:
                note = notes_module.NoteFile(directory)
            except OSError:
                log.warning("Could not open a note in %s", directory, exc_info=True)
                return notes_module.FAILED_TEXT
            restore = None
            if self.listener is not None:
                restore = self.listener.require_phrase
                self.listener.set_require_phrase(False)
                # Dictation pauses mid-sentence: end lines on silence alone.
                self.listener.set_dictation(True)
            session = notes_module.NoteSession(note, restore=restore)
            lines: queue.Queue = queue.Queue()
            self._notes, self._notes_queue = session, lines
        log.info("Taking notes in %s (ends after %.0f min of silence)", note.path, idle_s / 60)
        self.panel.channel_open = False
        self.panel.set_notes_session(session)
        self.printer.banner("tomando nota — decí «pará» o tocá STOP para terminar")
        threading.Thread(target=self._notes_worker, args=(session, lines),
                         name="wawi-notes", daemon=True).start()
        threading.Thread(target=self._notes_watchdog, args=(session, idle_s),
                         name="wawi-notes-idle", daemon=True).start()
        return notes_module.START_TEXT

    def _stop_notes(self, reason: str, *, only: "notes_module.NoteSession | None" = None,
                    speak: bool = True) -> bool:
        """End the note session (*only* that one, when given). Any thread.

        The microphone mode and the HUD come back at once; the worker then
        writes whatever was still waiting to be transcribed, closes the file,
        puts the note on the board and says so. False when there was none.
        """
        with self._notes_lock:
            session = self._notes
            if session is None or (only is not None and session is not only):
                return False
            lines, self._notes, self._notes_queue = self._notes_queue, None, None
        session.speak_on_end = speak
        session.ended.set()
        if self.listener is not None:
            self.listener.set_dictation(False)
            if session.restore is not None:
                self.listener.set_require_phrase(bool(session.restore))
        self.panel.set_notes_session(None)
        log.info("Note ended (%s)", reason)
        if lines is not None:
            lines.put(None)  # after every line already queued
        else:
            self._finish_note(session)
        return True

    def _close_note_on_quit(self) -> None:
        """Quitting mid-note: write what is queued and close the file, briefly
        and silently. Every line already written is on disk regardless."""
        session = self._notes
        if session is None or not self._stop_notes("quit", speak=False):
            return
        deadline = time.monotonic() + 5.0
        while not session.note.closed and time.monotonic() < deadline:
            time.sleep(0.05)

    def _on_notes_toggled(self, start: bool) -> None:
        """Menu item and the giant STOP. Never blocks the caller (AppKit)."""

        def _run() -> None:
            if not start:
                self._stop_notes("menu")
                return
            if not self._listening:
                # "Tomar nota" with the mic off: the click says listen.
                self.panel.set_listening(True)
                self._on_listening_toggled(True)
            self._say_aside(self._start_notes())

        threading.Thread(target=_run, name="wawi-notes-menu", daemon=True).start()

    def _queue_note(self, audio_buf: np.ndarray) -> bool:
        """Hand an utterance to the note worker. False when no note is open."""
        with self._notes_lock:
            lines = self._notes_queue
            if lines is None:
                return False
            lines.put(audio_buf)
        return True

    def _notes_worker(self, session: "notes_module.NoteSession", lines: "queue.Queue") -> None:
        """Transcribe and write each utterance, in order, until the end."""
        try:
            while True:
                audio_buf = lines.get()
                if audio_buf is None:
                    break
                try:
                    self._note_utterance(session, audio_buf)
                except Exception:
                    log.warning("A note line was lost", exc_info=True)
        finally:
            self._finish_note(session)

    def _note_utterance(self, session: "notes_module.NoteSession", audio_buf: np.ndarray) -> None:
        """One utterance of a note: a line, a stop command, or nothing."""
        n = len(audio_buf)
        peak = float(np.abs(audio_buf).max()) if n else 0.0
        if peak < _SILENCE_PEAK:
            return
        text = self._transcribe_filtered(audio_buf)
        if not text:
            return
        if notes_module.is_own_line(text):
            log.info("Note: dropped her own voice %r", text)
            return
        matched, rest = self._split_wake(text)
        if matched is not None and (not rest or notes_module.parse_start(rest)):
            return  # only her name, or "tomá nota" again: nothing to write
        if not session.ended.is_set() and notes_module.parse_stop(
                rest, named=matched is not None):
            log.info("Note: stop command %r", text)
            self._stop_notes("voice", only=session)
            return
        line = rest if matched is not None else text
        if session.note.append(line):
            session.heard()
            log.info("Note line %d: %r", session.lines, line[:120])
            self.printer.question(line)
            self.panel.set_texts(user_text=line)

    def _notes_watchdog(self, session: "notes_module.NoteSession", idle_s: float) -> None:
        """End a note nobody has spoken into for *idle_s* seconds."""
        step = min(30.0, max(0.5, idle_s / 4))
        while not session.ended.wait(step):
            if session.idle_for() >= idle_s:
                log.info("Note silent for %.0f min — ending it", idle_s / 60)
                self._stop_notes("silence", only=session)
                return

    def _finish_note(self, session: "notes_module.NoteSession") -> None:
        """Close the file, show the note on the board, say it is saved.
        Runs once per session, on the worker's thread. Never raises."""
        note = session.note
        if note.closed:
            return
        try:
            note.close()
        except OSError:
            log.warning("Could not close the note %s", note.path, exc_info=True)
        log.info("Note saved: %d line(s) in %s", note.lines, note.path)
        self.printer.banner(f"nota guardada ({note.lines} línea{'s' if note.lines != 1 else ''})")
        board = getattr(self.panel, "_board", None)
        content = note.read()
        if board is not None and content.strip():
            try:
                board.show("Nota", content, "markdown")
            except Exception:
                log.warning("Could not show the note on the board", exc_info=True)
        if session.speak_on_end:
            self._say_aside(notes_module.STOP_TEXT if note.lines else notes_module.EMPTY_TEXT)

    def _say_aside(self, text: str) -> None:
        """Say *text* under the turn lock, outside any turn, expecting no
        reply. Blocks the calling thread (never the main one) until said."""
        if not text:
            return
        if not self._turn_lock.acquire(timeout=_TASK_SPEECH_WAIT_S):
            log.warning("Not said, a turn held the lock: %s", text)
            return
        try:
            self._aside_speech = True
            self._speak(text)
        except Exception:
            log.warning("Could not say %r", text, exc_info=True)
        finally:
            self.state.set_state(State.IDLE)
            self._aside_speech = False
            self._turn_lock.release()

    # ------------------------------------------------------------------
    # Things dragged onto her (wawi/drops.py)
    # ------------------------------------------------------------------

    def _drop_settings(self) -> "drops_module.DropSettings":
        return drops_module.DropSettings.from_config(getattr(self._cfg, "drops", None))

    def _on_dropped(self, items: "list[drops_module.DropItem]") -> None:
        """The panel's drop callback (main thread): never blocks it."""
        threading.Thread(target=self._handle_drop, args=(list(items),),
                         name="wawi-drop", daemon=True).start()

    def _handle_drop(self, items: "list[drops_module.DropItem]") -> None:
        """Catch *items*, say so, and open the microphone for the order.

        Nothing reaches the model here: the items are only HELD
        (:attr:`_drop`), and the next turn the owner speaks carries them
        (see :meth:`_take_drop`). During a note they become note lines; during
        a turn she does not catch them at all.
        """
        if not items:
            return
        log.info("Drop: %s", drops_module.summary(items))
        if self._notes is not None:
            self._note_drop(items)
            return
        if self.state.state is not State.IDLE or self._turn_lock.locked():
            log.info("Drop refused: a turn is in flight (%s)", self.state.state.value)
            self._drop_bubble(drops_module.BUSY_TEXT)
            return
        settings = self._drop_settings()
        drop = drops_module.Drop(items, settings=settings)
        with self._drop_lock:
            self._drop = drop
        line = drops_module.catch_line(items)
        caught = getattr(self.panel, "drop_caught", None)
        if callable(caught):
            caught(line)
        threading.Thread(target=self._bury_drop, args=(drop,),
                         name="wawi-drop-hold", daemon=True).start()
        # Read the files (and transcribe an audio) now, while she waits for
        # the order, so the turn does not wait for it. Local only.
        threading.Thread(target=self._prepare_drop, args=(drop,),
                         name="wawi-drop-read", daemon=True).start()
        if self._stop_refresh.wait(_DROP_CATCH_PAUSE_S):
            return
        self._open_drop_window(drop, line)

    def _prepare_drop(self, drop: "drops_module.Drop") -> None:
        try:
            drop.context(self._transcribe_dropped)
        except Exception:
            log.warning("Could not read what was dropped", exc_info=True)

    def _transcribe_dropped(self, audio: np.ndarray) -> str:
        """A dropped audio file, through the same local Whisper as a turn."""
        return self.stt.transcribe(audio, language=self._language)

    def _drop_bubble(self, text: str) -> None:
        bubble = getattr(self.panel, "drop_bubble", None)
        if callable(bubble):
            bubble(text)

    def _note_drop(self, items: "list[drops_module.DropItem]") -> None:
        """Taking notes: what was dropped is written into the note."""
        session = self._notes
        if session is None:
            return
        written = 0
        for item in items:
            if session.note.append(drops_module.note_line(item)):
                written += 1
        if written:
            session.heard()
        log.info("Drop: %d line(s) added to the note", written)
        self._drop_bubble(drops_module.NOTED_TEXT)

    def _open_drop_window(self, drop: "drops_module.Drop", line: str) -> None:
        """Say *line* and hold the microphone open for the order.

        Like the window after a question, longer, and with no wake word in
        any mode (the follow-up window never needs one). A muted microphone
        is switched on for it and put back afterwards, said or not; the
        switch itself (and wake_local.json) is never touched. In a call
        (the quiet heuristic, wawi/reflexes.py) she only shows the line.
        """
        window = drop.settings.window_s
        if self.listener is None:
            # Push-to-talk only: there is no window to open; the next press
            # carries the drop all the same.
            self._drop_bubble(line)
            return
        if not self._turn_lock.acquire(timeout=2.0):
            log.info("Drop: a turn took the pipeline first; not asking")
            return
        try:
            if self._drop is not drop:
                return
            if not self._listening:
                self._drop_unmuted = True
                self.listener.set_wake_enabled(True)
                log.info("Drop: microphone on for %.0fs (it was muted)", window)
            from wawi import reflexes

            quiet = reflexes.is_quiet() is True
            self._drop_follow_up = window
            try:
                if quiet:
                    self._drop_bubble(line)
                    self.listener.arm_follow_up(window)
                    self._open_channel(window)
                else:
                    self._speak(line)
            finally:
                self.state.set_state(State.IDLE)
                self._drop_follow_up = 0.0
        except Exception:
            log.warning("Drop: could not ask what to do", exc_info=True)
        finally:
            self._turn_lock.release()
        log.info("Drop: waiting %.0fs for the order — no wake phrase needed", window)
        self._watch_drop_window(drop, window)

    def _watch_drop_window(self, drop: "drops_module.Drop", window: float) -> None:
        """Wait out the order window. It ends when a turn takes the drop, a
        new drop replaces it, or the time runs out with nobody talking."""
        listener = self.listener
        deadline = time.monotonic() + window
        while not self._stop_refresh.wait(_DROP_POLL_S):
            if self._drop is not drop or not drop.awaiting:
                break  # taken by a turn (it restores the mic), or replaced
            now = time.monotonic()
            busy = (self.state.state is not State.IDLE
                    or bool(getattr(listener, "is_capturing", False))
                    or bool(getattr(listener, "follow_up_open", False)))
            if now >= deadline and (not busy or now >= deadline + _DROP_OVERRUN_S):
                drop.awaiting = False
                log.info("Drop window expired with no order; holding it %.0fs more",
                         max(0.0, drop.settings.hold_s - (now - drop.at)))
                self.panel.channel_open = False
                self._drop_bubble(drops_module.LEFT_TEXT)
                self._restore_drop_mic()
                return

    def _restore_drop_mic(self, only_if_done: bool = False) -> None:
        """Put a microphone the drop window switched on back off. With
        *only_if_done*, not while that window is still waiting."""
        with self._drop_lock:
            if not self._drop_unmuted:
                return
            drop = self._drop
            if only_if_done and drop is not None and drop.awaiting:
                return
            self._drop_unmuted = False
        if not self._listening and self.listener is not None:
            self.listener.set_wake_enabled(False)
            log.info("Drop: microphone back to muted")

    def _take_drop(self) -> "drops_module.Drop | None":
        """The drop this turn should carry, if she still holds one. The first
        turn to take it is the order, and closes the window."""
        with self._drop_lock:
            drop = self._drop
            if drop is None:
                return None
            if drop.expired():
                self._drop = None
                log.info("Drop: held %.0fs; buried and forgotten", drop.settings.hold_s)
                return None
            first = drop.awaiting
            drop.awaiting = False
            drop.turns += 1
        if first:
            log.info("Drop: order heard; the turn carries %d item(s)", len(drop.items))
        else:
            log.info("Drop: still holding %d item(s); this turn carries them too",
                     len(drop.items))
        return drop

    def _bury_drop(self, drop: "drops_module.Drop") -> None:
        """After hold_s she buries what she was holding and forgets it."""
        if self._stop_refresh.wait(drop.settings.hold_s):
            return
        with self._drop_lock:
            if self._drop is not drop:
                return
            self._drop = None
        log.info("Drop: held %.0fs; buried and forgotten", drop.settings.hold_s)
        if self.state.state is State.IDLE and self._notes is None:
            self._drop_bubble(drops_module.BURIED_TEXT)

    # ------------------------------------------------------------------
    # The buzón "Wawi investiga" (wawi/inbox.py): dropped there, typed
    # ------------------------------------------------------------------

    def _on_inbox_dropped(self, items: "list[drops_module.DropItem]") -> bool:
        """Something was dropped in the buzón (main thread; quick).

        Held exactly like a drop on her (:attr:`_drop`, so a SPOKEN order
        carries it too), but the order is typed: no catch line and no
        microphone window. True when held (the buzón opens its field);
        False when it went into the note being taken instead.
        """
        if not items:
            return False
        log.info("Buzón drop: %s", drops_module.summary(items))
        if self._notes is not None:
            threading.Thread(target=self._note_drop, args=(list(items),),
                             name="wawi-inbox-note", daemon=True).start()
            return False
        drop = drops_module.Drop(list(items), settings=self._drop_settings())
        with self._drop_lock:
            self._drop = drop
        threading.Thread(target=self._bury_drop, args=(drop,),
                         name="wawi-drop-hold", daemon=True).start()
        # Read the files now, while the owner types.
        threading.Thread(target=self._prepare_drop, args=(drop,),
                         name="wawi-drop-read", daemon=True).start()
        return True

    def submit_typed(self, text: str) -> bool:
        """A typed order (the buzón). Any thread; never blocks.

        The same turn as a spoken one, minus the microphone: see
        :meth:`_run_turn`. True when taken; False when a turn is already in
        flight (her bubble says so and the buzón keeps the text) or there is
        nothing to say. While she takes notes it is a line of the note.
        """
        text = (text or "").strip()
        if not text:
            return False
        session = self._notes
        if session is not None:
            if session.note.append(text):
                session.heard()
                self.printer.question(text)
                self.panel.set_texts(user_text=text)
            self._drop_bubble(drops_module.NOTED_TEXT)
            return True
        if not self._turn_lock.acquire(blocking=False):
            log.info("Typed order refused: a turn is in flight (%s)", self.state.state.value)
            self._drop_bubble(inbox_module.BUSY_TEXT)
            return False
        log.info("Typed order (%d chars)", len(text))
        self.state.set_state(State.THINKING)
        threading.Thread(target=self._do_typed_turn, args=(text,),
                         daemon=True, name="wawi-worker").start()
        return True

    # ------------------------------------------------------------------
    # Worker thread — runs the full STT → brain → TTS pipeline
    # ------------------------------------------------------------------

    def _log_latency(
        self,
        turn_start: float,
        stt_s: float | None,
        brain_start: float | None,
        brain_s: float | None,
        speaker: "_StreamingSpeaker",
    ) -> None:
        """One line per turn: where the wait between "you stopped talking" and
        "Wawi starts talking" went. Times are seconds from the turn start (the
        utterance handed over), plus the silence waited before it. Never raises.
        """
        try:
            def fmt(v: float | None) -> str:
                return "-" if v is None else f"{v:.2f}s"

            silence_ms = getattr(self.listener, "last_trailing_silence_ms", None)
            first_play = getattr(self.voice, "first_play_since", None)
            played = first_play(turn_start) if callable(first_play) else None
            voice_s = None if played is None else played - turn_start
            first_sentence = (
                None if speaker.first_sentence_at is None or brain_start is None
                else speaker.first_sentence_at - brain_start
            )
            felt = (
                None if voice_s is None
                else voice_s + (silence_ms or 0) / 1000.0
            )
            log.info(
                "Latency: silence=%s stt=%s brain_first_sentence=%s brain_total=%s "
                "first_voice=%s felt=%s",
                "-" if silence_ms is None else f"{silence_ms}ms",
                fmt(stt_s), fmt(first_sentence), fmt(brain_s), fmt(voice_s), fmt(felt),
            )
        except Exception:
            log.debug("latency log failed", exc_info=True)

    def _do_turn(self, audio_buf: np.ndarray) -> None:
        """Execute one full voice turn on the worker thread (see
        :meth:`_run_turn`)."""
        self._run_turn(audio_buf=audio_buf)

    def _do_typed_turn(self, text: str) -> None:
        """One typed turn (the buzón, wawi/inbox.py) on the worker thread:
        the same pipeline as a spoken one, from the text on (see
        :meth:`submit_typed`)."""
        self._run_turn(typed=text)

    def _hear_turn(self, audio_buf: np.ndarray) -> "tuple[str | None, str | None, float | None]":
        """``(user_text, response, stt_seconds)`` for a spoken turn: the
        transcript, or None and the "no te escuché" answer for silent audio."""
        # Log the audio level first so we can tell "mic captured silence"
        # (peak/rms ≈ 0 → Whisper hallucinates 'gracias por ver el video')
        # from "real speech captured".
        n = len(audio_buf)
        peak = float(np.abs(audio_buf).max()) if n else 0.0
        rms = float(np.sqrt(np.mean(audio_buf**2))) if n else 0.0
        log.info(
            "audio: %d samples (%.2fs) peak=%.4f rms=%.4f%s",
            n, n / self._cfg.sample_rate, peak, rms,
            "  <-- SILENT (mic not capturing)" if peak < _SILENCE_PEAK else "",
        )

        # Silence gate: don't transcribe/send near-silent audio. Instead,
        # inform the user verbally so they know Wawi heard nothing.
        if peak < _SILENCE_PEAK:
            log.info("Silent audio (peak=%.4f) — announcing to user", peak)
            self.printer.note(lang.get(lang.NO_AUDIO_NOTE, self._language))
            self.issues.record(
                selfimprove.KIND_SILENT_AUDIO,
                "Se abrió un turno con audio en silencio",
                detail=(
                    "El trigger disparó pero el buffer llegó por debajo del "
                    "umbral de silencio, así que no había nada que transcribir. "
                    "Suele ser el umbral de amplitud o el micrófono."
                ),
                subject="silent audio buffer",
                context={"peak": round(peak, 5), "samples": n},
            )
            return None, lang.get(lang.NO_AUDIO_TEXT, self._language), None
        stt_start = time.perf_counter()
        user_text = self._transcribe_filtered(audio_buf)
        stt_elapsed = time.perf_counter() - stt_start
        log.info("STT result: %r  (%.2fs)", user_text, stt_elapsed)
        return user_text, None, stt_elapsed

    def _run_turn(self, audio_buf: np.ndarray | None = None, typed: str | None = None) -> None:
        """Execute one full turn on the worker thread: spoken (*audio_buf*)
        or typed in the buzón (*typed*, which skips the silence gate and STT).

        Transitions: THINKING → SPEAKING → IDLE (happy path)
                     THINKING → (error) → SPEAKING ("no pude conectarme") → IDLE
                     THINKING → IDLE (empty transcription — nothing to say)

        Streaming does not add a state. It moves the SPEAKING transition to the
        first finished sentence instead of the last, so the listener is paused
        for MORE of the turn, and the answer is written to the screen and to
        memory here at the end exactly as it always was.

        IDLE and the turn_lock release ALWAYS happen in the finally block, so a
        dropped turn never leaves the UI stuck or the lock held.
        """
        response: str | None = None
        turn_start = time.perf_counter()
        turn_metrics: TurnMetrics | None = None
        # Speak-as-you-go. Ollama is excluded because it is called with
        # stream: False and has nothing to hand over before it is finished.
        speaker = _StreamingSpeaker(
            self,
            enabled=self._cfg.tts_streaming and self._cfg.backend != "ollama",
        )
        streamed_the_answer = False
        stt_elapsed: float | None = None
        brain_start: float | None = None
        brain_elapsed: float | None = None

        try:
            try:
                # 1. What was said: typed in the buzón, or transcribed.
                if typed is not None:
                    user_text = typed.strip()
                    stt_elapsed = 0.0
                    log.info("Typed: %r", user_text)
                else:
                    user_text, response, stt_elapsed = self._hear_turn(audio_buf)
                if user_text is not None:
                    self.tui.set_texts(user_text=user_text)
                    self.printer.question(user_text)
                    self.panel.set_texts(user_text=user_text)

                    resume = self._session_id
                    replay = False
                    if user_text:
                        user_text, response = self._local_reply(user_text, audio_buf)
                    if user_text:
                        user_text, response, resume, replay = (
                            self._answer_resume_offer(user_text)
                        )

                    if user_text:
                        # The owner's answer to «¿la mando a Orca?». Read
                        # here, from the transcript, so a paid agent starts
                        # only on the owner's own yes, never the model's.
                        from wawi import orca as _orca

                        confirmed = _orca.hear(user_text)
                        if confirmed is not None:
                            log.info("Orca answer: %r -> %r", user_text, confirmed)
                            user_text, response = "", confirmed

                    if user_text:
                        # A configured shortcut only runs after the owner's
                        # next spoken yes, before another model turn.
                        from wawi import shortcuts as _shortcuts

                        confirmed = _shortcuts.hear(user_text)
                        if confirmed is not None:
                            log.info("Shortcuts answer: %r -> %r", user_text, confirmed)
                            user_text, response = "", confirmed

                    if user_text:
                        # «¿Lo hago?» from her computer (log in, buy, pay,
                        # send, post): only the owner's own yes runs it.
                        from wawi import webvm as _webvm

                        confirmed = _webvm.hear(user_text)
                        if confirmed is not None:
                            log.info("Web VM answer: %r -> %r", user_text, confirmed)
                            user_text, response = "", confirmed

                    if user_text:
                        # Recurring instructions need the owner's NEXT spoken
                        # yes, never an approval invented by the model.
                        from wawi import reminders as _reminders

                        confirmed = _reminders.hear(user_text)
                        if confirmed is not None:
                            log.info("Reminders answer: %r -> %r", user_text, confirmed)
                            user_text, response = "", confirmed

                    if user_text:
                        # A move in a game ("piedra", "más alto", "dale" to
                        # her proposal) is answered right here: no brain
                        # round-trip, no cost, no latency.
                        from wawi import games as _games

                        played = _games.hear(user_text)
                        if played is not None:
                            log.info("Game move: %r -> %r", user_text, played)
                            user_text, response = "", played

                    if not user_text:
                        if response is None:
                            log.info("Empty transcription — skipping turn")
                    else:
                        # 2. Brain call. The screenshot is not a mandatory
                        #    pipeline step — it is an optional tool the model
                        #    invokes only when the question is genuinely visual,
                        #    because an image costs ~1,400 tokens at 1280px.
                        log.debug(
                            "Brain: backend=%s model=%s",
                            self._cfg.backend,
                            self._cfg.ollama_model if self._cfg.backend == "ollama"
                            else self._cfg.model_on_demand,
                        )
                        brain_start = time.perf_counter()
                        # The curiosity just shown, if any, as one line in
                        # THIS user turn -- never in the system prompt, whose
                        # change would reconnect and lose the prompt cache.
                        from wawi import curiosities as _curiosities

                        brain_text = _curiosities.with_note(user_text)
                        # Something dropped on her: the order's context, in
                        # THIS user turn too (pictures attached to it).
                        drop_images: tuple = ()
                        drop = self._take_drop()
                        if drop is not None:
                            prefix, drop_images = drop.context(self._transcribe_dropped)
                            brain_text = prefix + brain_text

                        def _ask(resume: str | None) -> brain.BrainResult:
                            return brain.ask_sync(
                                brain_text,
                                model=self._cfg.model_on_demand,
                                backend=self._cfg.backend,
                                ollama_model=self._cfg.ollama_model,
                                ollama_url=self._cfg.ollama_url,
                                assistant_name=self._cfg.assistant_name,
                                history=brain.screen_notes_preamble(self._screen_notes)
                                + self.conversation.as_prompt_context(
                                    self._cfg.assistant_name
                                ),
                                preferences=self.preferences.as_prompt_block(),
                                issues_hint=self.issues.prompt_hint(),
                                recall=self.recall.as_prompt_block(
                                    self._cfg.assistant_name
                                ),
                                bypass_permissions=self._cfg.dangerous_skip_permissions,
                                language=self._language,
                                resume=resume,
                                engine=self._engine,
                                connectors=self._connectors,
                                # None, not a no-op callback, when streaming is
                                # off: the hook is also what asks the SDK for
                                # partial messages, and a turn nobody listens to
                                # should not be paying to stream them.
                                on_sentence=speaker.submit if speaker.enabled else None,
                                on_restart=speaker.void,
                                screen_options=brain.ScreenOptions(
                                    monitor=self._cfg.monitor,
                                    max_edge=self._cfg.max_edge,
                                    jpeg_quality=self._cfg.jpeg_quality,
                                ),
                                images=drop_images,
                            )

                        # Written down until the turn ends, so a crash between
                        # here and the answer can offer to ask it again.
                        self.inflight.mark_turn(user_text, resume)
                        try:
                            brain_result = _ask(resume)
                        except Exception:
                            if not replay:
                                raise
                            # The cut conversation could not be resumed —
                            # usually expired. The question still stands.
                            log.warning("Could not resume the interrupted "
                                        "conversation; asking it fresh",
                                        exc_info=True)
                            speaker.void()
                            brain_result = _ask(None)
                        self._end_turn_session(brain_result)
                        # A voided stream spoke half of an answer that was then
                        # thrown away and rewritten, so the real one still owes
                        # the user every word of itself.
                        streamed_the_answer = speaker.spoke and not speaker.voided
                        from wawi.prompts import is_silent_reply

                        if is_silent_reply(brain_result.text):
                            # Speech meant for someone else (the owner on the
                            # phone, talking to a visitor): no answer, no
                            # memory of it, and no follow-up window either.
                            log.info("Not meant for me (%r) — staying quiet", user_text[:80])
                            return

                        # Remember the exchange BEFORE speaking, so a TTS
                        # failure cannot cost the turn its memory.
                        self.conversation.add(
                            user_text, brain_result.text, brain_result.tool_calls
                        )
                        # Persisted separately from the log above: that file is
                        # only ever read by a human, this one is read back by
                        # the next run. Same failure rule — a store that cannot
                        # be written costs memory, never the turn.
                        self.recall.add(user_text, brain_result.text)
                        log.info(
                            "conversation: %d turns in memory", len(self.conversation)
                        )
                        if brain_result.tool_calls:
                            log.info("Tools used: %s", brain_result.tool_calls)
                        brain_elapsed = time.perf_counter() - brain_start
                        response = brain_result.text
                        log.info(
                            "Brain response: %r  (%.2fs, cost=%.5f)",
                            response[:120] if response else "",
                            brain_elapsed,
                            brain_result.cost_usd or 0.0,
                        )

                        # Build per-turn metrics and push to TUI
                        total_elapsed = time.perf_counter() - turn_start
                        turn_metrics = TurnMetrics(
                            stt_s=round(stt_elapsed, 2),
                            brain_s=round(brain_elapsed, 2),
                            total_s=round(total_elapsed, 2),
                            cost_usd=brain_result.cost_usd,
                            tokens_in=brain_result.input_tokens,
                            tokens_out=brain_result.output_tokens,
                        )
                        self._session.accumulate(turn_metrics)
                        self.tui.set_metrics(turn_metrics, self._session)

                        # A turn that overlapped the executor is not evidence
                        # of a slow turn: the executor holds the voice lock and
                        # competes for the machine, so filing an issue for it
                        # buries the real ones under noise nobody can act on.
                        task_running = self.tasks is not None and self.tasks.is_running
                        if total_elapsed >= selfimprove.SLOW_TURN_S and not task_running:
                            self.issues.record(
                                selfimprove.KIND_SLOW_TURN,
                                f"Turno lento: {total_elapsed:.0f}s",
                                detail=(
                                    "El usuario esperó en silencio todo ese tiempo, "
                                    "porque la respuesta no se transmite mientras se "
                                    "genera. Candidatos: streaming del TTS, menos "
                                    "llamadas a herramientas."
                                ),
                                subject="slow turn",
                                context={
                                    "total_s": round(total_elapsed, 1),
                                    "stt_s": round(stt_elapsed, 2),
                                    "brain_s": round(brain_elapsed, 2),
                                    "tools": list(brain_result.tool_calls),
                                },
                            )

            except Exception as exc:
                log.warning("Turn failed (brain/network/STT error)", exc_info=True)
                self._forget_session()
                # A turn that spent its budget without answering did connect;
                # saying so sends the user to debug a network that is fine.
                response = lang.get(
                    lang.BUDGET_TEXT
                    if isinstance(exc, brain.BrainBudgetError)
                    else lang.ERROR_TEXT,
                    self._language,
                )
                self.issues.record(
                    selfimprove.KIND_TURN_FAILED,
                    f"Turno fallado: {type(exc).__name__}",
                    detail=str(exc)[:1000],
                    # Group by exception type, not by message: the message
                    # usually carries a timestamp or an id and would defeat dedup.
                    subject=type(exc).__name__,
                    context={"user_text": user_text[:300] if "user_text" in dir() else ""},
                )
            finally:
                # Answered or failed, this turn is no longer in flight. A
                # process killed before here never gets this far, which is
                # exactly what leaves the record behind.
                self.inflight.clear_turn()
                # Nothing below may run while a streamed sentence is still
                # playing: both branches touch the state machine, and IDLE with
                # audio still coming out reopens the microphone onto it.
                speaker.close()

            # 4. Speak — only when there is something to say. Push the response
            #    text to the TUI FIRST so it's readable even if TTS fails/hangs.
            #    Voice.say() catches its own errors and never propagates.
            if response and streamed_the_answer:
                # Already said, one sentence at a time, while it was written.
                # Only the writing and the remembering are left — and they are
                # done here, with the whole text, exactly as the spoken path
                # does them.
                self._present(response)
                self._clear_echo_after_tail()
            elif response:
                self._speak(response)
            if typed is None:
                self._log_latency(turn_start, stt_elapsed, brain_start, brain_elapsed, speaker)
        finally:
            # 5. Always return to IDLE and release the turn lock, on every path.
            self.state.set_state(State.IDLE)
            # The order for a drop was heard and answered: a microphone the
            # drop window switched on goes back off (after IDLE, which may
            # have just armed the follow-up window it would otherwise leave).
            self._restore_drop_mic(only_if_done=True)
            try:
                self._turn_lock.release()
            except RuntimeError:
                pass  # already released (shouldn't happen, but guard it)
            # 6. Only now, with the lock genuinely free, answer anything said
            #    while this turn was thinking. After the release, never before:
            #    dispatching under our own lock would drop it a second time.
            held = self._take_held_utterance()
            if held is not None:
                self._dispatch_turn(held, already_locked=False)


def _game_on() -> bool:
    """A game against Wawi is running (wawi/games). Never raises."""
    try:
        from wawi import games

        m = games.get()
        return m is not None and m.playing is not None
    except Exception:
        return False


def _request_sentence(request, language: str) -> str:
    """Turn what the executor is waiting on into something worth hearing.

    Prefixed rather than said bare so the user can tell it apart from Wawi
    asking on its own account. The answer goes somewhere else — to the
    executor, via the answer_task tool — and if that difference is not audible
    it does not exist.
    """
    from wawi import task as task_module

    if request.kind is task_module.RequestKind.APPROVAL:
        if request.prompt:
            # A risky call brings its own short question ("¿Lo mando?").
            return lang.get(request.prompt, language)
        return lang.get(lang.APPROVAL_ASK, language).format(action=request.text)
    return lang.get(lang.TASK_ASK, language).format(question=request.text)


def _outcome_sentence(task, language: str) -> str:
    """What is said when the executor stops. The only report the work gets."""
    from wawi.task import TaskStatus

    if task is None:
        return ""
    table = {
        TaskStatus.DONE: lang.TASK_DONE,
        TaskStatus.FAILED: lang.TASK_FAILED,
        TaskStatus.STOPPED: lang.TASK_STOPPED,
    }.get(task.status)
    if table is None:
        return ""
    sentence = lang.get(table, language).format(id=task.id)
    # The executor's last message is written to be spoken (see its system
    # prompt), so it is appended rather than summarised again -- but clipped,
    # because "written to be spoken" is an instruction the model can ignore
    # and once did: a 511-character result took 34.7 seconds to read out, with
    # no way to cut it short. The prompt asks for brevity; this enforces it.
    if task.status is TaskStatus.DONE and task.result:
        from wawi.task import _shorten

        sentence = f"{sentence} {_shorten(task.result, _SPOKEN_RESULT_CHARS)}"
    return sentence


# ---------------------------------------------------------------------------
# Key-parsing helpers (shared with audio_input; duplicated here to keep
# the orchestrator self-contained and avoid a circular import)
# ---------------------------------------------------------------------------

def _parse_key(name: str):  # type: ignore[no-untyped-def]
    from pynput import keyboard  # type: ignore[import]

    key_attr = name.lower().replace(" ", "_")
    key_obj = getattr(keyboard.Key, key_attr, None)
    if key_obj is not None:
        return key_obj
    return keyboard.KeyCode.from_char(name)


def _key_matches(pressed, target) -> bool:
    return pressed == target
