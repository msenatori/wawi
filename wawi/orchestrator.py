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
from wawi.feedback import EarconObserver
from wawi import inflight as inflight_module
from wawi.inflight import Inflight
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
_PANEL_STATE = Path("panel_state.json")

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
        with self._lock:
            self._submitted += 1
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
        self.voice = Voice(cfg.tts_voices_dir, cfg.tts_voice, speed=cfg.tts_speed)

        # Observer modules
        self.tui = TuiObserver(cfg.show_tui)
        # The terminal transcript. Without the Rich TUI the console printed
        # nothing at all, so a spoken answer left no readable trace anywhere.
        # They are mutually exclusive: the TUI owns stdout when it runs.
        self.printer = TurnPrinter(enabled=not cfg.show_tui)
        self.earcons = EarconObserver(cfg.sound_cues)
        self.panel = PanelObserver(cfg.show_panel)
        self.panel.title_text = cfg.assistant_name
        self.panel.show_mascot = cfg.show_mascot
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
        self.tasks = self._build_task_runner(cfg)
        _tools.set_task_runner(self.tasks)

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
            if not cfg.wake_require_phrase:
                log.info(
                    "Always-on listening (wake.require_phrase: false) — any "
                    "speech starts a turn, no wake phrase needed"
                )
            # The transcriber costs a whole extra Whisper model at startup, so
            # it is built only when something still needs it: matching a
            # trigger phrase, or recognising the assistant's own voice while
            # barge-in keeps the microphone open during playback. With neither,
            # the mic is closed while Wawi speaks and there is no echo to
            # filter — loading it would be paying for a check nobody reads.
            needs_transcriber = cfg.wake_require_phrase or cfg.barge_in
            if cfg.wake_engine == "whisper" and cfg.wake_phrases and needs_transcriber:
                from wawi.wake import WhisperWakeDetector

                detector = WhisperWakeDetector(
                    cfg.wake_phrases,
                    model=cfg.wake_whisper_model,
                    language=cfg.language,
                    sample_rate=cfg.sample_rate,
                )
            return Listener(
                on_utterance=self._on_utterance,
                on_capture_start=self._on_capture_start,
                on_capture_dropped=self._on_capture_dropped,
                wake_detector=detector,
                cfg=ListenerConfig(
                    engine=cfg.wake_engine,
                    require_phrase=cfg.wake_require_phrase,
                    sample_rate=cfg.sample_rate,
                    pre_roll_s=cfg.wake_pre_roll_s,
                    silence_ms=cfg.wake_silence_ms,
                    silence_long_ms=cfg.wake_silence_long_ms,
                    long_after_s=cfg.wake_long_after_s,
                    smart_turn=cfg.wake_smart_turn,
                    smart_turn_threshold=cfg.wake_smart_turn_threshold,
                    smart_turn_pause_ms=cfg.wake_smart_turn_pause_ms,
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

        # Wired regardless of the input path: the language pill is about how
        # Wawi ANSWERS, which has nothing to do with how you talk to it.
        self.panel.on_language_changed(self._on_language_toggled)

        if self.listener is not None:
            # The panel toggle means "push-to-talk only". Wake-word mode is the
            # default, so it starts OFF (see panel.PanelObserver).
            self.panel.set_listening(True)
            self.panel.on_listening_changed(self._on_listening_toggled)
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
            self._engine.close()
            if self.tasks is not None:
                self.tasks.close()
            self._stop_refresh.set()
            if self.listener is not None:
                self.listener.stop()
            self.panel.stop()
            self.tui.stop()
            self.printer.banner(f"{self._cfg.assistant_name}: cerrado.")
            log.info("Wawi stopped.")

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
            data = json.loads(_PANEL_STATE.read_text())
        except FileNotFoundError:
            return
        except Exception:
            log.debug("Ignoring unreadable %s", _PANEL_STATE, exc_info=True)
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
                _PANEL_STATE.write_text(json.dumps(data))
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

    def _on_listening_toggled(self, listening: bool) -> None:
        """Panel switch: ON means Wawi hears you, OFF means it does not.

        The stream itself stays open either way. That is deliberate: closing it
        would drop the pre-roll ring, so the first syllable after switching back
        on would be lost, and reopening a CoreAudio input device is exactly the
        operation that is unreliable enough to lose the microphone entirely.
        Gating the speech instead is instant and cannot fail.
        """
        self._listening = bool(listening)
        if self.listener is not None:
            self.listener.set_wake_enabled(self._listening)
        self.printer.banner("escuchando" if self._listening else "micrófono en silencio")

    # ------------------------------------------------------------------
    # Worker thread — runs the full STT → brain → TTS pipeline
    # ------------------------------------------------------------------

    def _do_turn(self, audio_buf: np.ndarray) -> None:
        """Execute one full voice turn on the worker thread.

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

        try:
            try:
                # 1. Transcribe — first log the audio level so we can tell
                #    "mic captured silence" (peak/rms ≈ 0 → Whisper hallucinates
                #    'gracias por ver el video') from "real speech captured".
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
                    response = lang.get(lang.NO_AUDIO_TEXT, self._language)
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
                else:
                    stt_start = time.perf_counter()
                    user_text = self._transcribe_filtered(audio_buf)
                    stt_elapsed = time.perf_counter() - stt_start
                    log.info("STT result: %r  (%.2fs)", user_text, stt_elapsed)
                    self.tui.set_texts(user_text=user_text)
                    self.printer.question(user_text)
                    self.panel.set_texts(user_text=user_text)

                    resume = self._session_id
                    replay = False
                    if user_text:
                        user_text, response, resume, replay = (
                            self._answer_resume_offer(user_text)
                        )

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

                        def _ask(resume: str | None) -> brain.BrainResult:
                            return brain.ask_sync(
                                user_text,
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
        finally:
            # 5. Always return to IDLE and release the turn lock, on every path.
            self.state.set_state(State.IDLE)
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
