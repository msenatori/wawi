"""Always-on wake-phrase listener: continuous mic -> VAD -> trigger -> utterance.

This module replaces push-to-talk with a hands-free pipeline. The microphone
runs continuously and local models decide what is worth handing downstream:

    1. Silero VAD -- is anyone speaking?  (0.09 ms per 30 ms frame)
    2. the trigger -- was the wake phrase spoken?
    3. downstream -- Whisper + the model, driven by the orchestrator

Two trigger engines, chosen with ``ListenerConfig.engine``:

    "openwakeword" (default) scores every frame with a pretrained model. Cheap
        (1.4 ms per 80 ms chunk) but limited to its six English phrases, and
        those collapse under a non-English accent -- ``hey_jarvis`` measures
        0.9989 from an English voice and 0.0014 from a Spanish one.

    "whisper" transcribes each VAD speech segment and matches it against
        configured phrases (see :mod:`wawi.wake`). The trigger is then any
        word in any language, with no training, at ~40 ms per segment SPOKEN --
        nothing at all while the room is quiet.

Nothing leaves the machine, and Whisper never runs on ambient audio: it only
sees buffers that cleared both stages. Both models are local ONNX files; the
only network access this module can ever cause is the one-time model download
(see "Model files" below), and that never happens implicitly.

Both stages come from the ``openwakeword`` package -- it bundles Silero VAD
alongside the wake-word models, so there is no separate VAD dependency.

Why the VAD does NOT gate the wake model
----------------------------------------
The obvious design is a cost cascade: run the cheap VAD first and only invoke
the wake model when someone is speaking. That is wrong here, for two measured
reasons.

First, openWakeWord is a **streaming** model. ``AudioFeatures`` keeps a rolling
10-second raw-audio buffer and incremental melspectrogram/embedding buffers,
and each prediction reads context from them. Feeding it only during detected
speech splices discontinuous audio into that buffer, so the first second of
every utterance is scored against a corrupted context -- exactly the moment
the wake phrase lands.

Second, there is nothing to save. Measured on this machine (Apple Silicon,
onnxruntime CPU), the wake model costs 1.4 ms per 80 ms chunk: **1.8% of one
core** to run permanently. Gating it buys back under two percent of a core and
pays for it in missed wakes.

So the wake model is fed every frame while idle, and the VAD serves two
different purposes: it is the endpointing clock, and it **confirms**
detections -- a wake score that clears the threshold while the VAD saw no
speech in the last 120 ms is discarded as a false positive. The VAD is a
precision filter, not a cost gate. (While an utterance is being captured the
wake model is not fed at all, since its buffer is reset when the capture
closes.)

Wake phrase
-----------
With ``engine="openwakeword"`` and ``require_phrase=True`` the trigger is
openWakeWord's pretrained ``hey_jarvis`` model -- it only recognises that
English phrase, and its weights are licensed CC BY-NC-SA 4.0 (non-commercial).
The default is the whisper engine, or no trigger at all
(``require_phrase=False``), and then this model is never loaded.

Pre-roll
--------
People say "Wawi, what error did the blog throw?" in a single breath. By
the time the wake model fires, the command is already half spoken. So the mic
feeds a ring buffer continuously, and on detection we rewind: the utterance
handed to ``on_utterance`` starts ``pre_roll_s`` seconds BEFORE the trigger
fired. The wake phrase itself is included -- Whisper transcribing "Wawi,
what error..." is harmless and far safer than clipping the first syllables of
the command.

Endpointing
-----------
There is no key release any more, so the end of the utterance is inferred:
once the VAD reports ``silence_ms`` of continuous non-speech, the utterance is
closed and dispatched. The tradeoff is direct -- too short truncates a speaker
who pauses mid-sentence to think; too long makes Wawi feel sluggish because
it sits on a finished question. 800 ms is the default: comfortably longer than
an intra-sentence breath (~200-400 ms) and short enough to feel responsive.
A hard ``max_utterance_s`` cap (30 s default) guarantees a noisy room or a
stuck VAD can never capture forever.

With ``smart_turn`` on, the silence timer is no longer the only judge. Once the
pause reaches ``smart_turn_pause_ms`` (300 ms), the last 8 s of the utterance go
to an end-of-turn model (:mod:`wawi.turn`, on its own thread) that hears
whether the sentence SOUNDS finished. "Complete" closes the capture there;
"not finished" keeps listening with ``silence_ms`` as the ceiling, exactly as
before. Speech resuming mid-inference makes the pending verdict stale and it is
ignored. Every capture logs one ``Smart turn: decision=...`` line.

Threading
---------
- ``sounddevice`` calls the audio callback on its own realtime thread. That
  callback does no DSP: it copies the block and enqueues it.
- A single worker thread drains the queue and runs the whole cascade.
- **``on_utterance`` is invoked on that worker thread**, not the caller's.
  Do not block it: while it runs, no audio is being processed (blocks queue
  up and the next utterance is delayed). Hand off to the orchestrator and
  return. Exceptions raised by the callback are caught and logged.

Feedback loop -- pause()/resume()
---------------------------------
Wawi must not hear itself. When TTS plays through speakers, the mic picks it
up and the wake model will happily fire on Wawi saying its own name. Guarding
this is the **caller's** responsibility: call :meth:`pause` when entering
``State.SPEAKING`` and :meth:`resume` on the way back to ``State.IDLE``.
While paused, audio is dropped on the floor -- not buffered -- so no
self-recorded speech can leak into a later pre-roll.

Model files -- NOT bundled with the package
-------------------------------------------
The ``openwakeword`` wheel ships code only. The ONNX weights (the wake model,
the melspectrogram and embedding front-ends, and Silero VAD) are fetched on
first use into the package's own ``resources/models`` directory. This module
never downloads implicitly: :func:`models_ready` reports whether the files are
present without touching the network, :func:`missing_model_files` says which
are absent, and :func:`download_models` is an explicit opt-in. :meth:`start`
refuses to run with a :class:`ListenerError` naming the fix rather than
blocking on a network call or failing somewhere inside onnxruntime.

``models_ready()`` and ``download_models()`` exist for the ``--check`` doctor
in ``__main__.py``; both are safe to call when ``openwakeword`` is not even
installed (``models_ready()`` simply returns False).

Inference framework
-------------------
``onnx`` by default. openWakeWord prefers ``tflite``, but ``tflite-runtime``
publishes Linux wheels only -- on macOS it is not installable, so tflite is not
an option here. ONNX runs on the ``onnxruntime`` that is already a Wawi
dependency, and Silero VAD is ONNX-only regardless.

Microphone ownership -- THE LISTENER OWNS THE MIC
--------------------------------------------------
``audio_input.Recorder`` opens its own ``sd.InputStream`` on key press. If this
module also ran, there would be two input streams on one device: wasteful and,
on macOS CoreAudio, not reliably supported -- depending on the device and
driver the second ``start()`` can fail outright, or silently deliver empty
buffers.

So there is exactly one owner, and it is this class. Push-to-talk is a *forced
capture on the same stream*, not a second stream: :meth:`force_start` opens an
utterance immediately (skipping the VAD/wake-word gates but keeping the
pre-roll) and :meth:`force_stop` closes and dispatches it. That keeps one
``sd.InputStream`` process-wide, keeps ``current_block()`` as the single source
for the panel spectrum, and makes push-to-talk strictly more reliable than a
separate recorder -- it gains pre-roll for free.

``Recorder`` remains a standalone utility for scripts and tests. The
orchestrator instantiates it only as the fallback for when this module cannot
run at all (``openwakeword`` not installed, or its weights not downloaded).

Dependencies
------------
``openwakeword`` is imported lazily, so this module imports (and its tests run)
on a machine where it is not installed. A missing package surfaces as
:class:`ListenerError` with a pip hint, never an ``ImportError`` from deep
inside a C extension.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - environment dependent
    import sounddevice as sd
except Exception:  # pragma: no cover - sounddevice is optional at import time
    sd = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# openWakeWord's pretrained "hey jarvis" model (CC BY-NC-SA 4.0), used only by
# the openwakeword engine with require_phrase on. Shipped as a name, not a file:
# the package resolves it to a path under its own resources/models directory.
WAKE_MODEL_NAME = "hey_jarvis"

# Where a custom openWakeWord model named in ``wake.model`` is looked up:
# ``wake.model: wawi`` means wawi/assets/wakeword/wawi.onnx (see
# scripts/wakeword/README.md for how it is trained). A path works too.
WAKEWORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "wakeword")


def resolve_wake_model(name: str, inference_framework: str = "onnx") -> tuple[str, str]:
    """``(score_key, model)`` for a ``wake.model`` setting.

    "" means openWakeWord's pretrained "hey jarvis" (passed by name, resolved
    by the package). Anything else is a custom model FILE: a path, or a bare
    name looked up in :data:`WAKEWORD_DIR`. openWakeWord reports a custom
    model's score under its file stem, so that is the key to read.
    """
    name = (name or "").strip()
    if not name or name == WAKE_MODEL_NAME:
        return WAKE_MODEL_NAME, WAKE_MODEL_NAME
    suffix = ".onnx" if inference_framework == "onnx" else ".tflite"
    if os.sep in name or name.endswith((".onnx", ".tflite")):
        path = os.path.expanduser(name)
    else:
        path = os.path.join(WAKEWORD_DIR, name + suffix)
    return os.path.splitext(os.path.basename(path))[0], path


def custom_wake_model_missing(name: str, inference_framework: str = "onnx") -> str | None:
    """The path of a configured custom model that is not on disk, else None."""
    key, model = resolve_wake_model(name, inference_framework)
    if model == WAKE_MODEL_NAME:
        return None
    return None if os.path.isfile(model) else model

# Frames per InputStream callback (~64 ms at 16 kHz). Matches
# audio_input.Recorder and deliberately matches NEITHER engine's frame size --
# the reframer exists precisely so the device block size is free to be anything.
_CHUNK_FRAMES = 1024

# openWakeWord's native chunk: 1280 samples = 80 ms at 16 kHz. Shorter inputs
# are accepted but add up to 80 ms of detection delay while the library
# accumulates samples internally, so we hand it exactly what it wants.
_WAKE_FRAME_SAMPLES = 1280

# Silero VAD's recommended frame: 480 samples = 30 ms at 16 kHz. This is also
# the resolution of the endpointing clock.
_VAD_FRAME_MS = 30

# Sample rates the pipeline supports. Both models are 16 kHz-only.
_SUPPORTED_SAMPLE_RATES = frozenset({16000})

# A wake detection is only accepted if the VAD saw speech within this many
# trailing frames (120 ms at 30 ms/frame). Wide enough that a detection landing
# just after a brief unvoiced stretch still counts.
_SPEECH_GATE_FRAMES = 4

# How long stop() waits for the worker thread before giving up and logging.
_JOIN_TIMEOUT_S = 2.0

# Blocks between listener stat summaries (~64 ms each, so roughly every 60 s).
# A heartbeat matters here: the failure being diagnosed is silence, and silence
# produces no log line of its own.
_STATS_EVERY_BLOCKS = 940

# A capture open this long without endpointing is a symptom, not a long
# sentence. It means the VAD never reports silence -- a noisy room, a fan, a
# television -- so the utterance is never closed, never transcribed, and never
# answered. Nothing else in the pipeline says a word about it: from outside it
# is simply "no me contestó".
_LONG_CAPTURE_S = 8.0

# Fraction of a transcript that must also appear in what the assistant is
# currently saying for it to count as the microphone hearing the speakers.
# Well below 1.0 because the transcription of an echo is lossy -- observed,
# the assistant's own "¿Qué pantalla querés levantar?" came back as "¿Qué
# pantalla querés levantar o necesitas que vea...", mangled but unmistakable.
_ECHO_OVERLAP = 0.5

# While the assistant is speaking, a candidate capture is cut short instead of
# running to max_utterance_s. Observed live: a capture opened on the assistant's
# own voice held for the full 30 s cap, and a real barge-in had nowhere to land.
_SPEAKING_CAPTURE_S = 5.0

# Audio blocks the worker may fall behind by (~4 s at 64 ms per block).
#
# This queue MUST be bounded. The worker does real work per block -- with the
# whisper engine it transcribes every speech segment, normally ~40 ms but up to
# ~850 ms when the model hallucinates -- so it can transiently fall behind the
# realtime callback. With an unbounded queue that backlog is permanent: the
# listener drifts further behind wall-clock forever, and the wake phrase you
# just said is processed seconds later, or not usefully at all. Old audio is
# worthless to a live listener, so the oldest block is dropped instead.
_MAX_QUEUED_BLOCKS = 64


class ListenerError(RuntimeError):
    """Raised for missing dependencies, missing models, or bad configuration.

    Exists so callers never have to catch an ImportError or an opaque
    exception thrown from inside onnxruntime.
    """


@dataclass(frozen=True)
class ListenerConfig:
    """Tuning for the wake-word pipeline.

    Attributes:
        engine: Which trigger to use. ``"openwakeword"`` (default) scores every
            frame with a pretrained model — cheap, but limited to its six
            English phrases. ``"whisper"`` transcribes each VAD speech segment
            and matches it against ``wake.phrases``, so the trigger can be any
            word in any language at the cost of ~40 ms per segment spoken. See
            :mod:`wawi.wake`.
        sample_rate: Capture rate in Hz. Both models are 16 kHz-only.
        pre_roll_s: Seconds of audio kept before the wake word fires and
            prepended to the utterance. Must cover the wake phrase itself plus
            the speaker's head start; 1.5 s is comfortable for "Wawi, ...".
        silence_ms: Continuous non-speech required to close a SHORT utterance.
            Too short truncates a speaker pausing mid-sentence; too long makes
            Wawi feel sluggish. Resolved to whole 30 ms VAD frames.
        silence_long_ms: The same, once the utterance has been going for
            ``long_after_s``. 0 disables the escalation.

            One threshold cannot serve both jobs, and choosing between them is
            what makes a voice assistant feel broken. A command ("andá al 3")
            wants a snappy cut. An explanation wants patience: people pause for
            several seconds mid-argument to find the next word, and cutting
            them there splits one thought into two half-transcripts, neither of
            which means anything.

            So the threshold grows with the utterance. By the time someone has
            been talking for ``long_after_s`` they are clearly explaining
            something, not issuing a command, and the extra wait costs nothing
            because they were not expecting an instant answer anyway.
        long_after_s: Seconds of utterance after which ``silence_long_ms``
            applies.
        max_utterance_s: Hard cap on a single capture, so a noisy room cannot
            record forever. The utterance is dispatched, not discarded.
        follow_up_s: Seconds after Wawi finishes speaking during which plain
            speech reopens a capture WITHOUT the wake phrase, so a conversation
            can continue naturally. The window is armed by the caller (see
            :meth:`arm_follow_up`) and closes on the first capture it opens.
            0 disables it: every turn then needs "Wawi" again.
        vad_threshold: Silero speech probability above which a frame counts as
            speech, 0.0..1.0. Higher is stricter (fewer false wakes, more risk
            of clipping a soft talker).
        min_speech_level: Peak amplitude a frame must ALSO reach to count as
            speech, 0.0..1.0. 0 disables the check.

            This exists because ``vad_threshold`` alone cannot save a noisy
            room. Silero scores *speech-likeness*, not loudness, so a fan, a
            distant conversation or music scores high on near-silent audio --
            measured in one real room, ambient noise had a median score of
            0.715, and even at a threshold of 0.95 a third of ambient frames
            still counted as speech. When nothing ever counts as silence, an
            utterance never endpoints and NOTHING is ever dispatched.

            An amplitude floor cuts that cleanly, because the problem noise is
            quiet even when it is speech-like. Set it just above the room's
            ambient peak: ``scripts/diag_wake.py --measure-noise`` reports the
            floor and recommends a value.
        require_phrase: True (default) means a capture only becomes a turn when
            the trigger is present -- an openWakeWord detection, or the phrase
            found in the whisper transcript. False is ALWAYS-ON: every stretch
            of speech the VAD confirms is dispatched as a turn, with no wake
            phrase and no trigger model. The echo check still runs, so the
            assistant's own voice cannot start a turn, and the push-to-talk
            toggle still mutes detection outright.

            Always-on means anything said near the microphone becomes a paid
            turn -- a conversation with someone else in the room included.
        wake_threshold: openWakeWord score above which the wake phrase counts as
            detected, 0.0..1.0. Higher means fewer false positives and more
            missed wakes. NOTE the polarity is the opposite of Porcupine's
            "sensitivity": raising this makes the listener *less* eager.
        inference_framework: "onnx" (default, the only option on macOS) or
            "tflite" (Linux, needs the tflite-runtime package).
        smart_turn: Ask an end-of-turn model (:mod:`wawi.turn`) whether the
            speaker has FINISHED once the pause reaches ``smart_turn_pause_ms``,
            instead of always waiting the full ``silence_ms``. A confident
            "complete" closes the utterance there; anything else keeps
            listening, and ``silence_ms`` / ``silence_long_ms`` stay as the
            ceiling. A missing or failing model degrades to exactly the old
            behaviour. Never applies to push-to-talk captures.
        smart_turn_threshold: P(complete) at or above which the turn closes
            early, 0.0..1.0. Higher = fewer cut-offs, fewer early closes.
        smart_turn_sure: P(complete) that closes at the FIRST check. Between
            ``smart_turn_threshold`` and this the model is only half sure, and
            half sure at 300 ms is what cut the owner off mid-sentence
            ("¿Qué es el defecto…" at P=0.67): the pause is asked about again
            at ``smart_turn_recheck_ms`` and closes only if it still says so.
        smart_turn_recheck_ms: Continuous silence for that second look.
        vad_onset_threshold: Silero score a frame needs to count towards
            OPENING a segment; 0 means "same as vad_threshold". Higher than
            vad_threshold on purpose -- hysteresis: hard to start, easy to
            keep going. Measured: a burst of keyboard clicks scored >= 0.5 on
            50 frames out of 66, but >= 0.6 on only 3, while real speech
            barely notices the difference (14 -> 13 frames for "sí").
        min_onset_ms: Onset-grade speech required inside ``onset_window_ms``
            before a segment opens. 0 = a single frame is enough (the old
            behaviour). A clap, a door or a key press is one or two frames;
            the shortest real word is several.
        onset_window_ms: The sliding window ``min_onset_ms`` is counted in.
        min_voiced_ms: Voiced speech a VAD-opened segment must contain to be
            dispatched at all. Below it the segment is dropped as noise
            before Whisper ever sees it. 0 disables the check. Never applies
            to push-to-talk.
        min_voiced_ratio: Voiced frames over the span from the first to the
            last voiced frame. Speech is dense (0.5-0.9 even with pauses to
            think); sporadic noise that strings a few speechy frames together
            across seconds is not. 0 disables the check.
        smart_turn_pause_ms: Continuous silence before the model is asked.
            Asked ONCE per pause: measured, the model's P drifts towards
            "complete" for every clip as trailing silence grows (0.3 s: 7/10
            mid-sentence cuts called incomplete; 1 s: 0/10), so a second look
            later in the same pause would only be a shorter, noisier timer.
            A new pause after resumed speech gets a fresh look.
    """

    engine: str = "openwakeword"
    require_phrase: bool = True
    sample_rate: int = 16000
    pre_roll_s: float = 1.5
    silence_ms: int = 800
    silence_long_ms: int = 0
    long_after_s: float = 8.0
    max_utterance_s: float = 30.0
    follow_up_s: float = 0.0
    vad_threshold: float = 0.5
    min_speech_level: float = 0.0
    wake_threshold: float = 0.5
    inference_framework: str = "onnx"
    smart_turn: bool = False
    smart_turn_threshold: float = 0.5
    smart_turn_pause_ms: int = 300
    smart_turn_sure: float = 0.85
    smart_turn_recheck_ms: int = 900
    vad_onset_threshold: float = 0.0
    min_onset_ms: int = 0
    onset_window_ms: int = 400
    min_voiced_ms: int = 0
    min_voiced_ratio: float = 0.0
    # Whisper phrase mode: seconds of speech (after the pre-roll) after which
    # the open candidate is transcribed ONCE to look for the trigger. A match
    # confirms the capture early (the UI lights up while you are still
    # talking); no match drops it at once and ignores the rest of that
    # stretch of talk, so long chatter is never transcribed whole. 0 checks
    # only when the segment closes (the old behaviour).
    wake_window_s: float = 1.5
    # openwakeword engine: "" = pretrained "hey jarvis"; a name ("wawi") or a
    # path selects a custom model. See resolve_wake_model.
    wake_model: str = ""
    # Load the transcriber at start() even when nothing needs it yet. False
    # lets the owner's always-on setup skip a Whisper model it never uses,
    # while still holding a transcriber for the live switch to phrase mode
    # (which then loads it in the background).
    preload_transcriber: bool = True
    # openwakeword engine: consecutive 80 ms frames that must clear
    # wake_threshold. Measured on the "wawi" model: 2 frames at 0.8 halve the
    # false wakes of 1 frame at 0.9 for ~2 points of recall.
    wake_patience: int = 1


class CapturedAudio(np.ndarray):
    """An utterance buffer that remembers what the VAD measured in it.

    A plain float32 array everywhere it goes -- Whisper, the turn lock, the
    held-utterance slot -- with one extra attribute, ``speech``
    (:class:`wawi.speechgate.SpeechEvidence` or None). Carried on the buffer
    itself so the evidence cannot drift apart from the audio it describes
    when a turn is held and answered later.
    """

    speech = None
    #: The wake phrase that opened this utterance (whisper phrase mode), or
    #: None. The orchestrator takes it off the front of the transcript, so
    #: "Wawi, ¿qué hora es?" in one breath is asked as "¿qué hora es?".
    wake = None

    def __array_finalize__(self, obj) -> None:  # type: ignore[no-untyped-def]
        self.speech = getattr(obj, "speech", None)
        self.wake = getattr(obj, "wake", None)


# Put on the audio queue by the smart-turn thread when a verdict is ready, so
# the worker applies it at once instead of on the next ~64 ms block.
_TURN_READY = object()


# ---------------------------------------------------------------------------
# Lazy dependency loading and model-file management
# ---------------------------------------------------------------------------

def _load_openwakeword():  # type: ignore[no-untyped-def]
    """Import and return the openwakeword package, or raise ListenerError.

    Importing the package does not touch the network and does not load any
    model weights -- it only makes the class and path tables available.
    """
    try:
        import openwakeword
        import openwakeword.utils  # noqa: F401  (registers the submodule)
    except ImportError as exc:
        raise ListenerError(
            "openwakeword is not installed; the wake word cannot run. "
            "Install it with:  pip install openwakeword"
        ) from exc
    return openwakeword


def _required_model_paths(inference_framework: str = "onnx", wake_model: str = "") -> list[str]:
    """Return every model file the cascade needs, for *inference_framework*.

    Raises:
        ListenerError: openwakeword is not installed.
    """
    oww = _load_openwakeword()
    suffix = ".onnx" if inference_framework == "onnx" else ".tflite"

    _key, model = resolve_wake_model(wake_model, inference_framework)
    paths = [oww.MODELS[WAKE_MODEL_NAME]["model_path"]] if model == WAKE_MODEL_NAME else []
    paths += [m["model_path"] for m in oww.FEATURE_MODELS.values()]
    paths = [p.replace(".tflite", suffix) for p in paths]
    if model != WAKE_MODEL_NAME:
        paths.append(model)
    # Silero VAD is distributed as ONNX only, whichever framework is in use.
    paths += [m["model_path"] for m in oww.VAD_MODELS.values()]
    return paths


def missing_model_files(inference_framework: str = "onnx", wake_model: str = "") -> list[str]:
    """Return the model files that are absent. Empty list means ready.

    Purely a filesystem check -- no network access, no downloads, no model
    loading.

    Raises:
        ListenerError: openwakeword is not installed.
    """
    return [p for p in _required_model_paths(inference_framework, wake_model)
            if not os.path.exists(p)]


def models_ready(inference_framework: str = "onnx") -> bool:
    """Return True if every required model file is already on disk.

    Never downloads and never raises: returns False when ``openwakeword``
    is not installed. Intended for the ``--check`` doctor.
    """
    try:
        return not missing_model_files(inference_framework)
    except ListenerError:
        return False
    except Exception:  # pragma: no cover - defensive; the doctor must not crash
        log.warning("Could not determine openwakeword model status", exc_info=True)
        return False


def download_models() -> None:
    """Download the wake-word, feature, and VAD models. Explicit opt-in.

    Fetches roughly 8 MB into openwakeword's own ``resources/models``
    directory. Already-present files are skipped, so this is safe to re-run.

    Raises:
        ListenerError: openwakeword is not installed, or the download failed.
    """
    oww = _load_openwakeword()
    log.info("Downloading openwakeword models (%s)…", WAKE_MODEL_NAME)
    try:
        oww.utils.download_models([WAKE_MODEL_NAME])
    except Exception as exc:
        raise ListenerError(
            f"Downloading the openwakeword models failed: {exc}. "
            "Check network access, then retry."
        ) from exc

    still_missing = missing_model_files()
    if still_missing:
        raise ListenerError(
            "The openwakeword download completed but these files are still "
            f"missing: {still_missing}"
        )
    log.info("openwakeword models are ready")


def _models_missing_error(missing: list[str]) -> ListenerError:
    """Build the actionable error shown when weights have not been fetched."""
    names = ", ".join(os.path.basename(p) for p in missing)
    return ListenerError(
        f"The openwakeword model files are missing ({names}). They are not "
        "bundled with the package and must be downloaded once (~8 MB). Run:  "
        "python -c 'from wawi.listener import download_models; download_models()'"
        "   — or  wawi --check  which reports the same thing."
    )


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

class _Reframer:
    """Accumulate arbitrary-length blocks and emit fixed-size frames.

    openWakeWord wants 1280-sample chunks and Silero VAD wants 480-sample
    frames, while sounddevice delivers whatever block size the device settles
    on. This bridges the two: push whatever arrives, receive back zero or more
    frames of exactly ``frame_length`` samples. Leftover samples are retained
    for the next push, so no sample is ever dropped or duplicated -- which
    matters, because a dropped sample shifts every later frame boundary and
    quietly ruins detection.

    Not thread-safe; each instance belongs to a single consumer thread.
    """

    def __init__(self, frame_length: int) -> None:
        if frame_length <= 0:
            raise ValueError(f"frame_length must be positive, got {frame_length}")
        self._frame_length = frame_length
        self._buf: list[np.ndarray] = []
        self._pending = 0

    @property
    def frame_length(self) -> int:
        return self._frame_length

    @property
    def pending(self) -> int:
        """Number of buffered samples not yet emitted as a full frame."""
        return self._pending

    def reset(self) -> None:
        """Drop buffered samples (used when audio continuity is broken)."""
        self._buf = []
        self._pending = 0

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        """Buffer *block* and return every complete frame now available.

        Args:
            block: 1-D array of samples. May be any length, including 0 or a
                length that is not a multiple of ``frame_length``.

        Returns:
            List of 1-D arrays, each exactly ``frame_length`` samples, in
            chronological order. Empty when not enough samples have arrived.
        """
        if block.size:
            self._buf.append(block)
            self._pending += int(block.size)

        if self._pending < self._frame_length:
            return []

        joined = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        n_frames = joined.size // self._frame_length
        cut = n_frames * self._frame_length

        frames = [
            joined[i * self._frame_length : (i + 1) * self._frame_length]
            for i in range(n_frames)
        ]

        remainder = joined[cut:]
        if remainder.size:
            self._buf = [remainder]
            self._pending = int(remainder.size)
        else:
            self._buf = []
            self._pending = 0

        return frames


class _RingBuffer:
    """Fixed-capacity FIFO of float32 samples, used for pre-roll.

    Holds at most ``capacity`` samples; older audio falls off the front.
    Not thread-safe; the owner serialises access.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(0, int(capacity))
        self._chunks: deque[np.ndarray] = deque()
        self._total = 0

    @property
    def size(self) -> int:
        """Buffered sample count (may briefly exceed capacity by a partial chunk)."""
        return self._total

    def append(self, chunk: np.ndarray) -> None:
        if self._capacity == 0 or chunk.size == 0:
            return
        self._chunks.append(chunk)
        self._total += int(chunk.size)
        # Drop whole chunks while the remainder still covers the capacity.
        while self._chunks and self._total - self._chunks[0].size >= self._capacity:
            self._total -= int(self._chunks.popleft().size)

    def clear(self) -> None:
        self._chunks.clear()
        self._total = 0

    def keep_last(self, n: int) -> None:
        """Drop everything but the newest *n* samples."""
        if n <= 0:
            self.clear()
            return
        if self._total <= n:
            return
        tail = self.read()[-n:]
        self.clear()
        self._chunks.append(tail.copy())
        self._total = int(tail.size)

    def read(self) -> np.ndarray:
        """Return the most recent samples, at most ``capacity`` of them."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        joined = np.concatenate(list(self._chunks)).astype(np.float32, copy=False)
        if joined.size > self._capacity:
            joined = joined[-self._capacity :]
        return joined


def _float_to_int16(block: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] audio to int16, clipping out-of-range samples.

    Both openWakeWord and Silero VAD consume 16-bit PCM; sounddevice gives us
    float32. Clipping (rather than scaling) keeps the conversion stateless.
    """
    clipped = np.clip(block, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------

class Listener:
    """Always-on "Wawi" listener.

    Usage:
        listener = Listener(on_utterance=handle)
        listener.start()
        ...
        listener.pause()   # while Wawi is SPEAKING
        listener.resume()
        ...
        listener.stop()

    ``on_utterance`` receives a float32 mono numpy array at
    ``cfg.sample_rate``, pre-roll included, ready for Whisper. It is called on
    the worker thread -- see the module docstring.
    """

    #: Silence (ms) the last dispatched turn waited for before closing; None
    #: for a forced capture (push-to-talk) or before the first turn.
    last_trailing_silence_ms: int | None = None


    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        cfg: ListenerConfig | None = None,
        on_capture_start: Callable[[], None] | None = None,
        wake_detector=None,  # wawi.wake.WhisperWakeDetector when engine="whisper"
        turn_detector=None,  # wawi.turn.SmartTurn-like; built in start() if None
        on_capture_dropped: Callable[[], None] | None = None,
        verifier=None,  # wawi.wake.AcousticVerifier: second opinion on a whisper match
    ) -> None:
        self._cfg = cfg or ListenerConfig()
        _validate_config(self._cfg)
        self._on_utterance = on_utterance
        self._on_capture_start = on_capture_start
        # Called when a capture already announced through on_capture_start is
        # thrown away (noise, echo), so the UI can stop saying "escuchando".
        self._on_capture_dropped = on_capture_dropped
        if self._cfg.engine == "whisper" and self._cfg.require_phrase and wake_detector is None:
            raise ListenerError(
                "engine='whisper' needs a wake_detector; construct one with "
                "wawi.wake.WhisperWakeDetector(phrases)"
            )
        self._wake_detector = wake_detector
        self._verifier = verifier
        # openwakeword engine: consecutive 80 ms frames over the threshold.
        self._wake_streak = 0
        # The key openWakeWord reports this model's score under.
        self._wake_key = resolve_wake_model(self._cfg.wake_model, self._cfg.inference_framework)[0]
        # Early trigger look (wake_window_s), per capture: pre-roll length,
        # whether it ran, and the phrase it confirmed.
        self._preroll_samples = 0
        self._window_checked = False
        self._trigger: str | None = None
        # A candidate rejected early: ignore speech until the talk stops for
        # silence_ms, so the rest of the same chatter is not re-checked as a
        # fresh candidate every window_s.
        self._dismissed = False
        self._dismiss_silent = 0

        # Cross-thread plumbing
        self._lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=_MAX_QUEUED_BLOCKS
        )
        self._dropped_blocks = 0
        # Why the listener did or did not act. Every one of these used to be
        # invisible, which made "a veces no contesta" impossible to diagnose:
        # from outside, ten different causes look like silence.
        self._stats: dict[str, int] = {}
        self._blocks_seen = 0
        self._stop_evt = threading.Event()
        self._paused = threading.Event()
        # Wake detection can be switched off at runtime (the panel's
        # push-to-talk toggle) without giving up the microphone: the stream,
        # the pre-roll ring and forced captures all keep working.
        self._wake_enabled = threading.Event()
        self._wake_enabled.set()
        self._thread: threading.Thread | None = None
        self._stream = None  # sd.InputStream | None
        self._current_block: np.ndarray | None = None
        # What the assistant is saying out loud, for text-level echo rejection.
        self._echo_text = ""
        # Push-to-talk commands from another thread (True=open, False=close).
        # A FIFO rather than a flag so a fast tap — press and release inside a
        # single 64 ms block — cannot be collapsed into "nothing happened".
        self._force_cmds: deque[bool] = deque()

        # Engines (created in start())
        self._wake_model = None
        self._vad = None

        # Worker-thread-only state
        self._vad_frame_len = int(self._cfg.sample_rate * _VAD_FRAME_MS / 1000)
        self._vad_reframer = _Reframer(self._vad_frame_len)
        self._wake_reframer = _Reframer(_WAKE_FRAME_SAMPLES)
        self._ring = _RingBuffer(int(self._cfg.pre_roll_s * self._cfg.sample_rate))
        self._recent_speech: deque[bool] = deque(maxlen=_SPEECH_GATE_FRAMES)
        # Onset gate (see ListenerConfig.min_onset_ms): the last
        # onset_window_ms of frames, as (counts towards onset, is speech).
        self._onset_threshold = self._cfg.vad_onset_threshold or self._cfg.vad_threshold
        self._min_onset_frames = math.ceil(self._cfg.min_onset_ms / _VAD_FRAME_MS)
        self._onset_window: deque[tuple[bool, bool]] = deque(
            maxlen=max(1, math.ceil(self._cfg.onset_window_ms / _VAD_FRAME_MS),
                       self._min_onset_frames)
        )
        self._min_voiced_frames = math.ceil(self._cfg.min_voiced_ms / _VAD_FRAME_MS)
        # Per-capture voiced-speech accounting, in VAD frames.
        self._cap_frames = 0
        self._cap_voiced = 0
        self._cap_first_voiced: int | None = None
        self._cap_last_voiced = 0
        # Whether on_capture_start has fired for the open capture. Deferred
        # until the capture has min_voiced_ms of speech, so noise that is
        # about to be dropped never paints "escuchando" in the first place.
        self._announced = False
        # Follow-up window a capture consumed; given back if it was noise.
        self._follow_up_restore = 0
        self._capturing = False
        # True while the current capture was opened by push-to-talk. A forced
        # capture is closed by the key release, never by VAD endpointing.
        self._forced = False
        # True while the open capture still has to prove it contains the wake
        # phrase (whisper engine only).
        self._needs_trigger = False
        # Worker-side mirror of _wake_enabled, for edge detection.
        self._wake_was_on = True
        # One warning per capture, not one per block.
        self._warned_long_capture = False
        # Samples of follow-up window still open. Counted in samples rather
        # than wall-clock so the behaviour is deterministic under test and
        # cannot drift when the worker falls behind.
        self._follow_up_left = 0
        self._utterance: list[np.ndarray] = []
        self._utterance_samples = 0
        self._silent_frames = 0
        # Endpointing threshold in whole VAD frames. ceil() so the configured
        # silence is a floor, never rounded away to zero.
        self._silence_frames_needed = max(
            1, math.ceil(self._cfg.silence_ms / _VAD_FRAME_MS)
        )
        self._silence_frames_long = (
            max(1, math.ceil(self._cfg.silence_long_ms / _VAD_FRAME_MS))
            if self._cfg.silence_long_ms > 0
            else self._silence_frames_needed
        )
        self._long_after_samples = int(self._cfg.long_after_s * self._cfg.sample_rate)
        self._max_utterance_samples = int(
            self._cfg.max_utterance_s * self._cfg.sample_rate
        )
        self._follow_up_samples = int(self._cfg.follow_up_s * self._cfg.sample_rate)
        # Amplitude floor in int16 units, since VAD frames arrive as 16-bit PCM.
        self._min_speech_amplitude = int(self._cfg.min_speech_level * 32767)
        # Cross-thread request to open the follow-up window (set by the
        # orchestrator when Wawi stops speaking, consumed by the worker).
        self._arm_follow_up = threading.Event()

        # Smart end-of-turn (see ListenerConfig.smart_turn). The model runs on
        # its own thread so a slow inference can never stall the VAD clock;
        # the worker hands it a snapshot and reads the verdict back.
        self._turn_detector = turn_detector
        self._turn = None  # the LOADED detector; None = plain silence timer
        self._turn_broken = False  # set on the first inference error
        self._turn_inline = False  # tests: run inference on the worker itself
        self._turn_thread: threading.Thread | None = None
        self._turn_requests: queue.Queue = queue.Queue(maxsize=1)
        self._turn_pause_frames = max(
            1, math.ceil(self._cfg.smart_turn_pause_ms / _VAD_FRAME_MS)
        )
        self._turn_recheck_frames = max(
            self._turn_pause_frames + 1,
            math.ceil(self._cfg.smart_turn_recheck_ms / _VAD_FRAME_MS),
        )
        # Bumped whenever a verdict in flight stops meaning anything (speech
        # resumed, the capture closed): results carry the generation they
        # were asked under, and a stale one is ignored.
        self._turn_gen = 0
        self._turn_pending = False
        self._turn_result: tuple[int, float | None, float] | None = None
        self._reset_turn_stats()
        # Passive observers of every block the cascade sees (the music
        # detector, wawi/music.py). They never change what the listener does.
        self._block_observers: list[Callable[[np.ndarray, int, int], None]] = []

    def add_block_observer(self, fn: Callable[[np.ndarray, int, int], None]) -> None:
        """Call ``fn(block, voiced, vad_frames)`` on the worker thread for every
        block processed while not paused: the float32 block and how many of
        the VAD frames it completed were speech. *fn* must be quick; anything
        it raises is swallowed. Gating is untouched: an observer only looks."""
        self._block_observers.append(fn)

    # -- lifecycle ---------------------------------------------------------

    @property
    def config(self) -> ListenerConfig:
        return self._cfg

    @property
    def silence_frames_needed(self) -> int:
        """Consecutive silent VAD frames required to close an utterance."""
        return self._silence_frames_needed

    @property
    def wake_frame_samples(self) -> int:
        """Samples per chunk handed to openWakeWord (1280 = 80 ms)."""
        return _WAKE_FRAME_SAMPLES

    def start(self) -> None:
        """Load the models, open the microphone, and start the worker.

        Idempotent: a second call while running is a no-op. Never downloads
        anything; missing weights are reported, not fetched.

        Raises:
            ListenerError: openwakeword missing, model files not downloaded,
                a model failed to load, or the mic cannot be opened.
        """
        with self._lock:
            if self._thread is not None:
                return  # already running

        whisper_engine = self._cfg.engine == "whisper"
        # Always-on runs no trigger stage, so neither engine's weights are
        # needed -- only Silero VAD, which decides what counts as speech.
        vad_only = whisper_engine or not self._cfg.require_phrase

        # In whisper mode the pretrained wake model and its feature front-ends
        # are never used; only Silero VAD is, and it ships in the same package.
        missing = [
            p
            for p in missing_model_files(self._cfg.inference_framework, self._cfg.wake_model)
            if not vad_only or "silero" in os.path.basename(p).lower()
        ]
        if missing:
            raise _models_missing_error(missing)

        if not self._cfg.require_phrase and self._wake_detector is None:
            # Always-on without a transcriber: nothing to load, and nothing to
            # compare against either -- see _is_own_echo.
            log.warning(
                "Always-on with no transcriber: own-voice echo can only be "
                "avoided by pausing while speaking (barge-in is unsafe here)"
            )
        elif (whisper_engine and self._cfg.require_phrase) or (
            self._cfg.preload_transcriber and (whisper_engine or not self._cfg.require_phrase)
        ):
            try:
                self._wake_detector.load()
            except Exception as exc:
                # Surface as ListenerError so callers catch one exception type.
                raise ListenerError(str(exc)) from exc
        else:
            self._wake_model = self._create_wake_model()
        self._vad = self._create_vad()
        self._turn = self._load_turn_detector() if self._cfg.smart_turn else None
        self._turn_broken = False

        self._reset_capture_state()
        self._stop_evt.clear()

        thread = threading.Thread(target=self._run, name="wawi-listener", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()
        if self._turn is not None and not self._turn_inline:
            self._turn_thread = threading.Thread(
                target=self._turn_loop, name="wawi-smart-turn", daemon=True
            )
            self._turn_thread.start()

        try:
            self._open_stream()
        except Exception as exc:
            # Unwind so a failed mic does not leave a live thread + models.
            self.stop()
            raise ListenerError(f"Could not open the microphone: {exc}") from exc

        if not self._cfg.require_phrase:
            trigger = "always-on, no wake phrase"
        elif whisper_engine:
            trigger = f"whisper: {self._wake_detector.phrases}"
        else:
            trigger = f"openwakeword: {self._wake_key!r}"
        log.info(
            "Listener started (%s): %d Hz, pre-roll %.2fs, "
            "silence %dms (%d frames), max %.0fs",
            trigger,
            self._cfg.sample_rate,
            self._cfg.pre_roll_s,
            self._cfg.silence_ms,
            self._silence_frames_needed,
            self._cfg.max_utterance_s,
        )
        if self._min_onset_frames or self._min_voiced_frames or self._cfg.min_voiced_ratio:
            log.info(
                "Noise gates: onset %dms >= %.2f within %dms; dispatch needs %dms "
                "voiced at ratio >= %.2f",
                self._cfg.min_onset_ms, self._onset_threshold, self._cfg.onset_window_ms,
                self._cfg.min_voiced_ms, self._cfg.min_voiced_ratio,
            )
        if self._turn is not None:
            log.info(
                "Smart turn on: asks at %dms of silence, closes if P >= %.2f "
                "(P >= %.2f re-asked at %dms); otherwise waits up to %dms "
                "(%dms once past %.0fs)",
                self._turn_pause_frames * _VAD_FRAME_MS,
                self._cfg.smart_turn_sure,
                self._cfg.smart_turn_threshold,
                self._turn_recheck_frames * _VAD_FRAME_MS,
                self._cfg.silence_ms,
                self._silence_frames_long * _VAD_FRAME_MS,
                self._cfg.long_after_s,
            )

    def stop(self) -> None:
        """Close the mic, drain the worker, and release the models.

        Safe to call when not running, and safe to call twice. Never blocks
        indefinitely: the worker join is bounded by ``_JOIN_TIMEOUT_S``.
        """
        self._stop_evt.set()

        with self._lock:
            stream = self._stream
            self._stream = None
            thread = self._thread
            self._thread = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.warning("Error closing the listener audio stream", exc_info=True)

        # Sentinel wakes the worker out of its blocking get().
        self._queue.put(None)

        if thread is not None and thread.is_alive():
            thread.join(timeout=_JOIN_TIMEOUT_S)
            if thread.is_alive():
                log.warning(
                    "Listener worker did not exit within %.1fs; abandoning it",
                    _JOIN_TIMEOUT_S,
                )

        turn_thread, self._turn_thread = self._turn_thread, None
        if turn_thread is not None and turn_thread.is_alive():
            self._submit_turn_job(None)  # sentinel
            turn_thread.join(timeout=_JOIN_TIMEOUT_S)

        self.log_stats()
        self._release_models()
        self._drain_queue()
        with self._lock:
            self._current_block = None
        log.debug("Listener stopped")

    def pause(self) -> None:
        """Ignore incoming audio until :meth:`resume`.

        Call this while Wawi is SPEAKING so it cannot wake on its own voice
        coming back through the speakers. Any in-flight capture is discarded
        and the pre-roll ring is cleared, so nothing Wawi said can survive
        into a later utterance.
        """
        self._paused.set()
        log.debug("Listener paused")

    def resume(self) -> None:
        """Resume processing audio after :meth:`pause`."""
        self._paused.clear()
        log.debug("Listener resumed")

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def arm_follow_up(self, seconds: float | None = None) -> None:
        """Open the follow-up window: speech alone reopens a capture.

        Call this when Wawi finishes answering. For ``follow_up_s`` seconds
        the wake phrase is not required — simply talking starts the next turn,
        so a back-and-forth does not need "Wawi" every single time. The
        window closes as soon as it opens a capture, or when it runs out.

        Args:
            seconds: Override the configured window for this one turn. The
                caller knows whether the answer expects a reply; the listener
                does not. 0 or negative closes the channel outright.

        Thread-safe and non-blocking.
        """
        window = self._cfg.follow_up_s if seconds is None else seconds
        if window <= 0:
            return
        with self._lock:
            self._follow_up_override = int(window * self._cfg.sample_rate)
        self._arm_follow_up.set()

    @property
    def follow_up_open(self) -> bool:
        """True while plain speech would start a turn without the wake word."""
        return self._follow_up_left > 0

    def set_echo_text(self, text: str | None) -> None:
        """Tell the listener what the assistant is saying out loud right now.

        Anything transcribed that mostly repeats this text is the microphone
        hearing the speakers, not a person, and is discarded. This replaces
        guessing with knowing: the assistant has its own script.
        """
        with self._lock:
            self._echo_text = text or ""

    def set_speech_floor(self, level: float | None) -> None:
        """Temporarily raise the amplitude a frame needs to count as speech.

        Used while the assistant is talking with barge-in armed. Without it the
        microphone hears the speakers, every sentence the assistant says opens
        a candidate segment, and the whisper engine transcribes ALL of them --
        the worker falls behind realtime, drops blocks, and from outside it
        looks like the assistant hung and stopped listening.

        Raising the floor exploits the one reliable asymmetry available without
        an echo canceller: the person talking is close to the microphone and
        the speakers are not. Pass None to restore the configured value.
        """
        if level is None:
            self._min_speech_amplitude = int(self._cfg.min_speech_level * 32767)
        else:
            self._min_speech_amplitude = int(max(0.0, min(1.0, level)) * 32767)
        log.debug("Speech floor now %d (int16)", self._min_speech_amplitude)

    def set_wake_enabled(self, enabled: bool) -> None:
        """Open or close the microphone gate, without releasing the device.

        This is what the panel's "escuchando" switch drives, and while it is
        closed NOTHING gets through — not the VAD, not the follow-up window,
        not push-to-talk. One switch, one meaning: off is deaf. A key that
        still worked would make "am I being heard right now" depend on which
        of two controls you last touched.

        The audio device itself stays open, which is a different question from
        whether anything is heard. Closing it would drop the pre-roll ring, so
        the first syllable after switching back on would be lost — and
        reopening a CoreAudio input is exactly the operation that fails and
        leaves you with no microphone at all.
        """
        if enabled:
            self._wake_enabled.set()
        else:
            self._wake_enabled.clear()
            # Muting has to take effect on the word you are saying, not on the
            # next one. An open follow-up window and a capture already running
            # both outlive the click otherwise, and the user is left talking to
            # something they just told to stop listening.
            self._follow_up_left = 0
            self._arm_follow_up.clear()
            if self._capturing and not self._forced:
                log.info("Muted mid-capture — dropping the utterance")
                self._reset_capture_state(clear_ring=True)
        log.info("Listening %s", "enabled" if enabled else "disabled")

    @property
    def is_wake_enabled(self) -> bool:
        return self._wake_enabled.is_set()

    def set_require_phrase(self, required: bool) -> bool:
        """Switch between always-on and wake-phrase mode, live.

        Returns False (and changes nothing) when phrase mode cannot run here:
        it needs the whisper engine and a transcriber. The transcriber's model
        is loaded in the background on the way in, so the switch itself never
        blocks the caller (the panel's main thread, or a turn).

        The swap is one attribute assignment, read by the worker once per
        block; a capture already open finishes under the mode it began in,
        except that a candidate still waiting for its trigger simply keeps
        waiting for it.
        """
        from dataclasses import replace

        required = bool(required)
        if required == self._cfg.require_phrase:
            return True
        if required and (self._cfg.engine != "whisper" or self._wake_detector is None):
            log.warning("Wake-phrase mode needs the whisper engine and its transcriber")
            return False
        if required:
            load_async = getattr(self._wake_detector, "load_async", None)
            if callable(load_async):
                load_async()
        self._cfg = replace(self._cfg, require_phrase=required)
        log.info("Listening mode: %s", "wake phrase" if required else "always-on")
        return True

    @property
    def require_phrase(self) -> bool:
        return self._cfg.require_phrase

    # -- push-to-talk on the same stream -----------------------------------

    def force_start(self) -> None:
        """Open an utterance now, without waiting for speech to be detected.

        This is push-to-talk. It reuses the one microphone stream this class
        already owns rather than opening a second one, which macOS CoreAudio
        does not reliably support (see the module docstring). The VAD and the
        wake model are bypassed, but the pre-roll is kept: the capture starts
        ``pre_roll_s`` before the key was pressed, so the first syllables
        spoken on the way to the key are not lost.

        A forced capture is closed by :meth:`force_stop`, never by VAD
        endpointing — the speaker may pause as long as they hold the key. The
        ``max_utterance_s`` cap still applies.

        Thread-safe and non-blocking: the command is queued and applied by the
        worker on the next audio block (~64 ms). Safe to call while a capture
        is already open; it simply takes ownership of that capture.

        No-op while paused, and no-op while the microphone gate is closed: the
        panel switch means deaf, and a key that still got through would make
        "am I being heard" depend on which control you touched last.
        """
        if not self._wake_enabled.is_set():
            log.info("force_start ignored — listening is switched off")
            return
        with self._lock:
            self._force_cmds.append(True)

    def force_stop(self) -> None:
        """Close a capture opened by :meth:`force_start` and dispatch it.

        Thread-safe and non-blocking; the utterance is dispatched from the
        worker thread as usual. A no-op when no forced capture is open.
        """
        with self._lock:
            self._force_cmds.append(False)

    def stats(self) -> dict[str, int]:
        """Counters for every outcome the listener can reach. Thread-safe."""
        with self._lock:
            return dict(self._stats)

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + n

    def log_stats(self) -> None:
        """Emit one line summarising what the listener has been doing."""
        stats = self.stats()
        if not stats:
            log.info("listener: nothing heard yet (%d blocks)", self._blocks_seen)
            return
        body = "  ".join(f"{k}={v}" for k, v in sorted(stats.items()))
        log.info("listener stats: %s", body)

    @property
    def is_capturing(self) -> bool:
        """True while audio is being recorded into an utterance.

        Read by the panel so the spectrum can say "I am recording THIS" while
        it happens. With the whisper engine that is the only honest live
        signal available: whether the segment contains the wake phrase is not
        known until it closes and is transcribed, so waiting for that would
        light the panel up only once the turn had already started.

        A VAD-opened capture that has not yet reached ``min_voiced_ms`` is
        not reported: it is still a noise candidate, and lighting the
        spectrum for a door slam that is about to be dropped is the flicker
        this gate exists to avoid.
        """
        return self._capturing and (self._announced or self._needs_trigger or self._forced)

    def current_level(self) -> float:
        """Return the peak amplitude of the newest block, 0.0 when idle.

        Mirrors ``audio_input.Recorder.current_level`` so the TUI meter can
        read from the Listener without knowing which of the two owns the mic.
        """
        block = self.current_block()
        if block is None or block.size == 0:
            return 0.0
        return float(np.abs(block).max())

    def current_block(self) -> np.ndarray | None:
        """Return the newest captured chunk, or None.

        Feeds the panel's spectrum visualiser. Thread-safe. Returns None
        before the first block arrives, after :meth:`stop`, and it keeps
        returning the last block while paused (the panel should read the
        state machine, not this, to know Wawi is deaf).
        """
        with self._lock:
            return self._current_block

    # -- engine construction ----------------------------------------------

    def _create_wake_model(self):  # type: ignore[no-untyped-def]
        oww = _load_openwakeword()
        _key, model = resolve_wake_model(self._cfg.wake_model, self._cfg.inference_framework)
        try:
            return oww.Model(
                wakeword_models=[model],
                inference_framework=self._cfg.inference_framework,
            )
        except Exception as exc:
            raise ListenerError(
                f"Could not load the openwakeword '{model}' model "
                f"using the '{self._cfg.inference_framework}' framework: {exc}"
            ) from exc

    def _create_vad(self):  # type: ignore[no-untyped-def]
        oww = _load_openwakeword()
        try:
            return oww.VAD()
        except Exception as exc:
            raise ListenerError(
                f"Could not load the Silero VAD model bundled with "
                f"openwakeword: {exc}"
            ) from exc

    def _release_models(self) -> None:
        model = self._wake_model
        self._wake_model = None
        self._vad = None
        if model is not None:
            # openWakeWord holds onnxruntime sessions; clearing its rolling
            # buffers keeps a restarted listener from seeing stale audio.
            try:
                model.reset()
            except Exception:
                log.debug("Error resetting the wake model", exc_info=True)

    def _open_stream(self) -> None:
        if sd is None:
            raise ListenerError(
                "sounddevice is not installed; the microphone cannot be opened. "
                "Install it with:  pip install sounddevice"
            )

        def _callback(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
            # Realtime thread: copy and enqueue only. No DSP here.
            if status:
                log.debug("listener sounddevice status: %s", status)
            block = indata[:, 0].copy()
            with self._lock:
                self._current_block = block
            self._enqueue(block)

        stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=_CHUNK_FRAMES,
            callback=_callback,
        )
        stream.start()
        with self._lock:
            self._stream = stream

    def _enqueue(self, block: np.ndarray) -> None:
        """Hand a block to the worker, dropping stale audio if it is behind.

        Called from the realtime audio thread, so it must never block. When the
        queue is full the OLDEST block goes, not this one: falling behind has
        to cost stale audio, never the words being spoken right now.
        """
        try:
            self._queue.put_nowait(block)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
            self._queue.put_nowait(block)
        except (queue.Empty, queue.Full):  # pragma: no cover - racy with worker
            pass

        self._dropped_blocks += 1
        if self._dropped_blocks % 32 == 1:
            log.warning(
                "Listener is behind realtime; dropped %d stale audio blocks. "
                "The trigger model may be too slow for this machine.",
                self._dropped_blocks,
            )

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        """Worker loop: drain the queue and run the cascade on every block."""
        while not self._stop_evt.is_set():
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if block is None:  # shutdown sentinel
                break
            try:
                if block is _TURN_READY:
                    self._poll_turn()
                    continue
                self._pump(block)
            except Exception:
                # One bad block must never kill the listener.
                log.warning("Listener failed to process an audio block", exc_info=True)
        log.debug("Listener worker exiting")

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _reset_capture_state(self, clear_ring: bool = True) -> None:
        """Return to waiting.

        Args:
            clear_ring: Also discard the pre-roll history. True when the audio
                itself is untrustworthy (paused: the assistant's own voice must
                not survive into a later utterance). FALSE after a normal
                capture, because throwing the ring away there is what CLIPS THE
                NEXT WAKE WORD: the ring refills from empty, so the next
                utterance opens with a fraction of the pre-roll it should have
                and the first syllables of "Wawi" are simply missing.

                It is safe to keep: a capture only closes after silence_ms of
                continuous silence, so what the ring holds at that moment is
                silence, not the tail of what was just said.
        """

        self._capturing = False
        self._forced = False
        self._needs_trigger = False
        self._warned_long_capture = False
        self._utterance = []
        self._utterance_samples = 0
        self._silent_frames = 0
        self._cap_frames = 0
        self._cap_voiced = 0
        self._cap_first_voiced = None
        self._cap_last_voiced = 0
        self._announced = False
        self._follow_up_restore = 0
        # Whatever the model is still chewing on belonged to this capture.
        self._turn_gen += 1
        self._turn_pending = False
        self._reset_turn_stats()
        self._recent_speech.clear()
        self._onset_window.clear()
        if clear_ring:
            self._ring.clear()
        self._vad_reframer.reset()
        self._wake_reframer.reset()
        if self._wake_model is not None:
            # Drop openWakeWord's rolling feature buffer so audio from before
            # the reset cannot contribute to the next detection.
            try:
                self._wake_model.reset()
            except Exception:
                log.debug("Error resetting the wake model", exc_info=True)

    def _pump(self, block: np.ndarray) -> None:
        """Run one audio block through the full cascade. Worker thread only."""
        self._blocks_seen += 1
        # A heartbeat, so a session that goes quiet still says what it saw.
        if self._blocks_seen % _STATS_EVERY_BLOCKS == 0:
            self.log_stats()

        if self._paused.is_set():
            self._bump("blocks_while_paused")
            # Dropped, not buffered: Wawi's own voice must not reach a
            # later pre-roll. Discard any capture that was in flight, and any
            # queued key command with it — a press made while Wawi was
            # speaking must not open a capture once it stops.
            with self._lock:
                self._force_cmds.clear()
            # A follow-up window belongs to the turn that just ended; it is
            # armed again after Wawi stops speaking, never carried across.
            self._follow_up_left = 0
            if self._capturing or self._ring.size:
                self._reset_capture_state()
            return

        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return

        self._ring.append(block)
        if self._capturing:
            self._utterance.append(block)
            self._utterance_samples += int(block.size)

        # Push-to-talk commands, applied after this block joined the ring so a
        # forced capture's pre-roll includes it, and after the append above so
        # it cannot be counted twice.
        self._apply_force_commands()

        pcm = _float_to_int16(block)

        # Stage 1 -- VAD on every 30 ms frame.
        vad_frames = voiced = 0
        for frame in self._vad_reframer.push(pcm):
            self._on_vad_frame(frame)
            vad_frames += 1
            voiced += bool(self._recent_speech and self._recent_speech[-1])
        for observe in self._block_observers:
            try:
                observe(block, voiced, vad_frames)
            except Exception:
                log.debug("Listener block observer failed", exc_info=True)

        # Stage 2a -- follow-up window. Wawi just answered, so for a few
        # seconds plain speech continues the conversation and the wake phrase
        # is not required. Runs BEFORE the trigger stage: this speech is
        # already authorised, and the whisper engine would otherwise claim it
        # as a candidate and demand the wake phrase all over again.
        if self._arm_follow_up.is_set():
            self._arm_follow_up.clear()
            with self._lock:
                override = self._follow_up_override
                self._follow_up_override = 0
            self._follow_up_left = override or self._follow_up_samples
            log.info(
                "Channel open %.0fs to START talking — once you do, it records "
                "until %dms of silence, however long you take",
                self._follow_up_left / self._cfg.sample_rate,
                self._cfg.silence_ms,
            )

        # Read the gate BEFORE the follow-up branch, not after. The follow-up
        # window used to open a capture without consulting it at all, and with
        # conversation_mode the window opens after EVERY answer — so muting the
        # microphone did nothing until the window happened to expire. From the
        # outside that is indistinguishable from a switch that is not wired.
        wake_on = self._wake_enabled.is_set()

        if self._follow_up_left > 0 and not self._capturing:
            self._follow_up_left = max(0, self._follow_up_left - int(block.size))
            if self._speech_onset():
                remaining, self._follow_up_left = self._follow_up_left, 0
                if wake_on:
                    self._begin_capture(follow_up=True)
                    self._follow_up_restore = remaining
                else:
                    self._bump("speech_while_wake_disabled")

        # Stage 2b -- wake word. Fed EVERY frame while waiting (see below);
        # the VAD confirms the detection rather than gating the inference.
        if wake_on is not self._wake_was_on:
            # The toggle moved. Drop the reframer and the model's rolling
            # buffer so audio from across the gap cannot fabricate a detection.
            self._wake_was_on = wake_on
            self._wake_reframer.reset()
            if wake_on and self._wake_model is not None:
                try:
                    self._wake_model.reset()
                except Exception:
                    log.debug("Error resetting the wake model", exc_info=True)

        if not self._cfg.require_phrase:
            # Always-on: no trigger stage at all. Speech the VAD confirms IS
            # the turn. Nothing is rejected later except the assistant's own
            # echo, so this path skips both engines entirely.
            self._dismissed = False
            if not self._capturing and wake_on and self._speech_onset():
                self._begin_capture()
            elif not self._capturing and not wake_on and self._speech_onset():
                self._bump("speech_while_wake_disabled")
        elif self._cfg.engine == "whisper":
            # There is no per-frame score to read. Every stretch of speech is a
            # CANDIDATE: capture it, and decide at the end — once there is a
            # whole segment to transcribe — whether it began with a trigger.
            if self._capturing and self._needs_trigger:
                self._maybe_check_window()
            elif self._dismissed:
                pass  # still the chatter that was already turned down
            elif not self._capturing and wake_on and self._speech_onset():
                self._begin_capture(needs_trigger=True)
            elif not self._capturing and not wake_on and self._speech_onset():
                # Speech while detection is OFF (panel toggle). Silent until
                # now, and indistinguishable from a broken wake word.
                self._bump("speech_while_wake_disabled")
        elif not self._capturing and wake_on:
            for frame in self._wake_reframer.push(pcm):
                detected = self._detect_wake(frame)
                if detected and not any(self._recent_speech):
                    log.debug("wake score cleared the bar but the VAD saw no speech")
                    continue
                if detected:
                    self._begin_capture()
                    break

        # An open capture that will not end is the quietest failure here.
        if self._capturing:
            seconds = self._utterance_samples / self._cfg.sample_rate
            if seconds >= _LONG_CAPTURE_S and not self._warned_long_capture:
                self._warned_long_capture = True
                self._bump("captures_stuck_open")
                needed_ms = self._silence_needed_now() * _VAD_FRAME_MS
                # Name the knob that is ACTUALLY in effect. Once the utterance
                # passes long_after_s the threshold is silence_long_ms, and
                # telling the user to lower silence_ms sends them to a setting
                # that changes nothing here.
                knob = (
                    "wake.silence_long_ms"
                    if self._utterance_samples >= self._long_after_samples
                    else "wake.silence_ms"
                )
                log.warning(
                    "Capture open %.0fs without endpointing: the VAD has not "
                    "reported %dms of continuous silence (%s). Nothing is "
                    "transcribed until it closes. Lower %s, or raise "
                    "wake.min_speech_level if the room is noisy.",
                    seconds, needed_ms, knob, knob,
                )

        # Stage 3 -- hard cap, checked after the block was appended. While the
        # assistant is speaking the cap is much shorter, so a capture opened on
        # its own voice recycles quickly and a real barge-in can land.
        with self._lock:
            speaking = bool(self._echo_text)
        # In always-on there is no needs_trigger flag to key on, but the
        # failure it guards against is the same one: a capture opened on the
        # assistant's own voice must recycle fast enough for a real barge-in.
        unproven = self._needs_trigger or (
            not self._cfg.require_phrase and not self._forced
        )
        cap = (
            int(_SPEAKING_CAPTURE_S * self._cfg.sample_rate)
            if speaking and unproven
            else self._max_utterance_samples
        )
        if self._capturing and self._utterance_samples >= cap:
            log.info(
                "Listener hit the %.0fs utterance cap; dispatching what it has",
                self._cfg.max_utterance_s,
            )
            self._bump("hit_utterance_cap")
            self._finish_capture()

    def _apply_force_commands(self) -> None:
        """Drain queued push-to-talk commands, in order. Worker thread only."""
        with self._lock:
            if not self._force_cmds:
                return
            cmds = list(self._force_cmds)
            self._force_cmds.clear()

        for open_capture in cmds:
            if open_capture:
                if self._capturing:
                    # A wake-word capture is already open; the key just takes
                    # it over, so VAD silence can no longer close it.
                    self._forced = True
                else:
                    self._begin_capture(forced=True)
            elif self._capturing and self._forced:
                self._finish_capture()

    def _on_vad_frame(self, frame: np.ndarray) -> None:
        score = self._speech_score(frame)
        speech = score >= self._cfg.vad_threshold
        self._recent_speech.append(speech)
        self._onset_window.append((score >= self._onset_threshold, speech))
        if self._dismissed and not self._capturing:
            self._dismiss_silent = 0 if speech else self._dismiss_silent + 1
            if self._dismiss_silent >= self._silence_frames_needed:
                self._dismissed = False
                self._dismiss_silent = 0
        if self._capturing and not self._forced:
            self._cap_frames += 1
            if speech:
                self._cap_voiced += 1
                self._cap_last_voiced = self._cap_frames
                if self._cap_first_voiced is None:
                    self._cap_first_voiced = self._cap_frames
                if (
                    not self._announced
                    and not self._needs_trigger
                    and self._cap_voiced >= self._min_voiced_frames
                ):
                    self._announce_capture()
        if not self._capturing or self._forced:
            # A forced capture ends on the key release, not on silence: the
            # speaker is allowed to hold the key and think.
            return
        if speech:
            if self._silent_frames:
                self._on_pause_ended()
            self._silent_frames = 0
            return
        self._silent_frames += 1
        if self._silent_frames >= self._silence_needed_now():
            self._finish_capture()
            return
        if self._turn_usable() and (
            self._silent_frames == self._turn_pause_frames
            or (self._silent_frames == self._turn_recheck_frames
                and self._turn_verdict == "unsure")
        ):
            self._ask_turn()
        self._poll_turn()

    # -- smart end-of-turn -------------------------------------------------

    def _load_turn_detector(self):  # type: ignore[no-untyped-def]
        """Load the end-of-turn model, or return None and say why ONCE."""
        detector = self._turn_detector
        try:
            if detector is None:
                from wawi.turn import SmartTurn

                detector = SmartTurn()
            load = getattr(detector, "load", None)
            if load is not None:
                load()
        except Exception as exc:
            log.warning(
                "Smart turn unavailable (%s) — endpointing falls back to "
                "wake.silence_ms (%dms)",
                exc, self._cfg.silence_ms,
            )
            return None
        return detector

    def _turn_usable(self) -> bool:
        return self._turn is not None and not self._turn_broken

    def _reset_turn_stats(self) -> None:
        """Per-utterance bookkeeping for the one ``Smart turn:`` log line."""
        self._turn_checks = 0
        self._turn_resumed = 0
        self._turn_last_p: float | None = None
        self._turn_last_ms: float | None = None
        self._turn_verdict: str | None = None  # "complete" | "continue"

    def _on_pause_ended(self) -> None:
        """Speech is back: any verdict about the pause that just ended is moot."""
        if self._turn_pending or self._turn_verdict in ("continue", "unsure"):
            # The model said (or was about to say) "not finished" and the
            # speaker proved it. Counted, because it is the evidence the
            # model is earning its keep.
            self._turn_resumed += 1
        self._turn_gen += 1
        self._turn_pending = False
        self._turn_verdict = None

    def _ask_turn(self) -> None:
        """Snapshot the utterance and ask the model about it. Worker thread."""
        audio = self._utterance_tail(self._cfg.sample_rate * 8)
        self._turn_pending = True
        self._turn_checks += 1
        gen = self._turn_gen
        if self._turn_inline:
            self._run_turn_job(gen, audio)
        else:
            self._submit_turn_job((gen, audio))

    def _submit_turn_job(self, job) -> None:  # type: ignore[no-untyped-def]
        """Hand *job* to the model thread, replacing any job not yet started."""
        while True:
            try:
                self._turn_requests.put_nowait(job)
                return
            except queue.Full:
                try:
                    self._turn_requests.get_nowait()
                except queue.Empty:
                    pass

    def _clear_name(self, text: str) -> bool:
        """The transcript holds a configured wake phrase, spelled exactly.

        Anywhere, not only first: "Contame, Wawi" was vetoed by the acoustic
        model (trained on synthetic voices, it scores the owner 0.00) and she
        ignored being called by name.
        """
        from wawi.wake import normalize

        words = normalize(text or "").split()
        # With a song on, a lyric misheard as "…, Wawi" is not her being
        # called: then only a sentence that OPENS with the name counts.
        anywhere = not self._music_playing()
        for phrase in getattr(self._wake_detector, "phrases", None) or ("wawi",):
            target = normalize(phrase).split()
            n = len(target)
            starts = range(len(words) - n + 1) if anywhere else range(min(1, len(words) - n + 1))
            if target and any(words[i:i + n] == target for i in starts):
                return True
        return False

    @staticmethod
    def _music_playing() -> bool:
        try:
            from wawi import music

            service = music.get()
            return service is not None and service.cue() is not None
        except Exception:
            return False

    def _utterance_tail(self, n: int) -> np.ndarray:
        """The newest *n* samples of the open utterance, as one fresh array."""
        parts: list[np.ndarray] = []
        total = 0
        for chunk in reversed(self._utterance):
            parts.append(chunk)
            total += int(chunk.size)
            if total >= n:
                break
        if not parts:
            return np.zeros(0, dtype=np.float32)
        joined = np.concatenate(parts[::-1]).astype(np.float32)
        return joined[-n:]

    def _turn_loop(self) -> None:
        """Model thread: run one job at a time, then nudge the worker."""
        while not self._stop_evt.is_set():
            try:
                job = self._turn_requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            gen, audio = job
            self._run_turn_job(gen, audio)
            # Leave headroom: a nudge must never push real audio out of a
            # queue that is already close to full.
            if self._queue.qsize() < _MAX_QUEUED_BLOCKS - 8:
                try:
                    self._queue.put_nowait(_TURN_READY)
                except queue.Full:  # pragma: no cover - racy with the callback
                    pass
        log.debug("Smart turn thread exiting")

    def _run_turn_job(self, gen: int, audio: np.ndarray) -> None:
        detector = self._turn
        if detector is None:
            return
        t0 = time.perf_counter()
        try:
            p: float | None = float(detector.predict(audio))
        except Exception:
            p = None
            if not self._turn_broken:
                self._turn_broken = True
                log.warning(
                    "Smart turn model failed; endpointing falls back to "
                    "wake.silence_ms for the rest of the session",
                    exc_info=True,
                )
        ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._turn_result = (gen, p, ms)

    def _poll_turn(self) -> None:
        """Apply a finished verdict if it still describes the current pause."""
        if not self._turn_pending or not self._capturing or self._forced:
            return
        with self._lock:
            result = self._turn_result
        if result is None or result[0] != self._turn_gen:
            return
        _, p, ms = result
        self._turn_pending = False
        self._turn_last_ms = ms
        if p is None:
            return  # model error: the silence ceiling decides, as before
        self._turn_last_p = p
        rechecked = self._silent_frames >= self._turn_recheck_frames
        if p >= self._cfg.smart_turn_sure or (
            rechecked and p >= self._cfg.smart_turn_threshold
        ):
            self._turn_verdict = "complete"
            self._finish_capture(early=True)
        elif p >= self._cfg.smart_turn_threshold:
            # Half sure this early: look again once the pause is longer.
            self._turn_verdict = "unsure"
        else:
            self._turn_verdict = "continue"

    def _turn_log_line(self) -> str | None:
        """One line per utterance, for scripts/latency parsing. None = n/a."""
        if self._turn is None or self._forced:
            return None
        if self._turn_verdict in ("complete", "continue", "unsure"):
            decision = self._turn_verdict
        else:
            decision = "fallback"
        p = "-" if self._turn_last_p is None else f"{self._turn_last_p:.3f}"
        infer = "-" if self._turn_last_ms is None else f"{self._turn_last_ms:.1f}"
        return (
            f"Smart turn: decision={decision} pause_ms={self._silent_frames * _VAD_FRAME_MS} "
            f"p={p} infer_ms={infer} checks={self._turn_checks} "
            f"resumed={self._turn_resumed}"
        )

    def _silence_needed_now(self) -> int:
        """Silent frames required to close the CURRENT utterance.

        Grows once the utterance is long enough to be an explanation rather
        than a command. See ``silence_long_ms``.
        """
        if self._utterance_samples >= self._long_after_samples:
            return self._silence_frames_long
        return self._silence_frames_needed

    def _is_speech(self, frame: np.ndarray) -> bool:
        return self._speech_score(frame) >= self._cfg.vad_threshold

    def _speech_score(self, frame: np.ndarray) -> float:
        """Silero's speech probability for *frame*; -1.0 when gated out."""
        vad = self._vad
        if vad is None:
            return -1.0
        # Amplitude gate first: it is far cheaper than the model, and in a
        # noisy room it is what actually decides. See min_speech_level.
        if self._min_speech_amplitude > 0:
            if int(np.abs(frame).max()) < self._min_speech_amplitude:
                return -1.0
        try:
            return float(vad.predict(frame, frame_size=self._vad_frame_len))
        except Exception:
            log.debug("Silero VAD rejected a frame; treating it as silence", exc_info=True)
            return -1.0

    def _speech_onset(self) -> bool:
        """Has enough speech arrived to open a segment? See min_onset_ms.

        With the gate off this is the old rule -- any speech in the last few
        frames -- so a single 30 ms frame is enough. With it on, a click or a
        clap (one or two frames) can no longer start a turn by itself.
        """
        if self._min_onset_frames <= 1 and self._onset_threshold == self._cfg.vad_threshold:
            return any(self._recent_speech)
        onset = sum(1 for strong, _ in self._onset_window if strong)
        if onset < max(1, self._min_onset_frames):
            return False
        # The newest frames must still be speech: a burst that already ended
        # would open a capture on its own trailing silence.
        return any(self._recent_speech)

    def _detect_wake(self, frame: np.ndarray) -> bool:
        model = self._wake_model
        if model is None:
            return False
        try:
            scores = model.predict(frame)
        except Exception:
            log.warning("openwakeword failed on a frame; ignoring it", exc_info=True)
            return False
        try:
            score = float(scores.get(self._wake_key, 0.0))
        except AttributeError:
            log.warning("openwakeword returned %r, expected a dict", type(scores))
            return False
        if score >= self._cfg.wake_threshold:
            self._wake_streak += 1
            log.debug("wake score %.3f >= %.3f (%d/%d)", score, self._cfg.wake_threshold,
                      self._wake_streak, self._cfg.wake_patience)
            if self._wake_streak >= self._cfg.wake_patience:
                self._wake_streak = 0
                return True
            return False
        self._wake_streak = 0
        return False

    def _is_own_echo(self, buf: np.ndarray) -> bool:
        """True when *buf* is mostly the assistant's own voice coming back."""
        with self._lock:
            echo_text = self._echo_text
        if not echo_text or self._wake_detector is None:
            return False
        if not getattr(self._wake_detector, "loaded", True):
            # Not loaded (always-on, never switched): no words to compare, and
            # "no words" must not read as "wordless noise" below.
            return False
        result = self._wake_detector.check(buf)
        if not result.text:
            # The assistant is talking and this segment has no words in it:
            # a cough, a door, the speakers' hum. Letting it through would
            # cut her off mid-sentence for nothing.
            log.info("listener: dropped wordless segment while speaking")
            return True
        from wawi.speechgate import judge
        from wawi.wake import echo_overlap

        junk = judge(result.text)
        if junk is not None:
            log.info("listener: dropped segment while speaking: %s: %r", junk, result.text[:80])
            return True

        overlap = echo_overlap(result.text, echo_text)
        if overlap >= _ECHO_OVERLAP:
            log.info(
                "listener: dropped own-voice echo (%.0f%% overlap): %r",
                overlap * 100, result.text[:80],
            )
            return True
        return False

    def _maybe_check_window(self) -> None:
        """The early look: once ``wake_window_s`` of speech is in, check it.

        The phrase must open the utterance (wawi.wake: first three words), so
        the first second and a half is all the evidence there will ever be.
        Waiting for the segment to close instead meant transcribing a whole
        minute of someone else's phone call to learn it did not start with
        "Wawi" -- and keeping the UI dark while the owner was mid-request.
        """
        window = self._cfg.wake_window_s
        if window <= 0 or self._window_checked:
            return
        spoken = self._utterance_samples - self._preroll_samples
        if spoken < int(window * self._cfg.sample_rate):
            return
        self._window_checked = True
        buf = np.concatenate(self._utterance).astype(np.float32, copy=False)
        phrase = self._trigger_present(buf, quick=True)
        if phrase is not None:
            self._needs_trigger = False
            self._trigger = phrase
            # _trigger_present already told the UI; this capture is now an
            # ordinary one that ends on silence.
            self._announced = True
            self._bump("triggers_early")
            return
        log.info("No wake phrase in the first %.1fs — ignoring this stretch of talk", window)
        self._bump("segments_rejected_early")
        self._reset_capture_state(clear_ring=False)
        self._dismissed = True
        self._dismiss_silent = 0

    def _trigger_present(self, buf: np.ndarray, quick: bool = False) -> str | None:
        """Whisper engine: the wake phrase this segment starts with, or None.

        Runs on the worker thread and blocks it for the transcription (~20 ms
        with the default tiny model). Everything that does not match is dropped
        here, silently — this is the common case, since it fires on every
        sentence spoken near the machine.
        """
        if self._wake_detector is None:
            return None
        if not getattr(self._wake_detector, "loaded", True):
            # Switched to phrase mode live: the model loads in the background
            # and is normally ready long before anyone speaks. If not, this
            # one candidate waits for it rather than being lost.
            try:
                self._wake_detector.load()
            except Exception:
                log.warning("Wake detector failed to load", exc_info=True)
                return None
        result = (self._wake_detector.check(buf, quick=True) if quick
                  else self._wake_detector.check(buf))

        with self._lock:
            echo_text = self._echo_text
        if echo_text and result.text:
            from wawi.wake import echo_overlap

            overlap = echo_overlap(result.text, echo_text)
            if overlap >= _ECHO_OVERLAP:
                log.info(
                    "wake: discarded as own-voice echo (%.0f%% overlap): %r",
                    overlap * 100, result.text[:80],
                )
                self._bump("reason:own-voice-echo")
                return None

        self._bump(f"reason:{result.reason or 'unknown'}")
        if not result.ok:
            return None
        if self._verifier is not None:
            ok, score = self._verifier.accepts(buf)
            if not ok and self._clear_name(result.text):
                # Trained on synthetic voices, it scored the owner's own
                # "Wawi, decime…" 0.00 and ignored them. Whisper spelling the
                # name exactly, first thing said, is the strong evidence; the
                # acoustic model only settles the doubtful matches (an alias,
                # a near spelling, the name mid-sentence).
                log.info("wake: %r accepted over the acoustic model (%.2f < %.2f): "
                         "the name itself opens it", result.text[:80], score,
                         self._verifier.threshold)
                self._bump("reason:acoustic-overruled")
            elif not ok:
                log.info("wake: %r vetoed by the acoustic model (%.2f < %.2f)",
                         result.text[:80], score, self._verifier.threshold)
                self._bump("reason:acoustic-veto")
                return None

        log.info(
            "Wake phrase %r heard in %r (%.0fms)",
            result.matched,
            result.text,
            result.elapsed_s * 1000,
        )
        if self._on_capture_start is not None:
            try:
                self._on_capture_start()
            except Exception:
                log.warning("on_capture_start raised an exception; ignoring", exc_info=True)
        return result.matched

    def _begin_capture(
        self, forced: bool = False, follow_up: bool = False, needs_trigger: bool = False
    ) -> None:
        """Open an utterance, seeded with pre-roll audio from before the trigger."""
        pre_roll = self._ring.read()
        self._capturing = True
        self._forced = forced
        # Only a whisper-engine candidate must prove it contains the trigger.
        # A key press and a follow-up reply are already authorised.
        self._needs_trigger = needs_trigger
        self._utterance = [pre_roll] if pre_roll.size else []
        self._utterance_samples = int(pre_roll.size)
        self._preroll_samples = int(pre_roll.size)
        self._window_checked = False
        self._trigger = None
        self._silent_frames = 0
        # The speech that opened the capture already happened, inside the
        # onset window: count it, or a short "sí" would be judged on the
        # frames AFTER its own onset and look like noise.
        window = list(self._onset_window)
        first = next((i for i, (_, sp) in enumerate(window) if sp), None)
        self._cap_frames = 0 if first is None else len(window) - first
        self._cap_voiced = sum(1 for _, sp in window if sp)
        self._cap_first_voiced = None if first is None else 1
        self._cap_last_voiced = self._cap_frames
        self._announced = False
        # The wake phrase is consumed; its trailing samples must not seed the
        # next detection once this utterance closes.
        self._wake_reframer.reset()
        if forced:
            trigger = "Push-to-talk pressed"
        elif follow_up:
            trigger = "Follow-up speech (no wake word needed)"
        elif needs_trigger:
            trigger = "Speech detected; will check it for the wake phrase"
        elif not self._cfg.require_phrase:
            trigger = "Speech detected (always-on, no wake phrase)"
        else:
            trigger = "Wake word detected"
        log.info(
            "%s; capturing with %.2fs of pre-roll",
            trigger,
            pre_roll.size / self._cfg.sample_rate,
        )
        self._bump("segments_opened")
        if needs_trigger:
            # Nothing is confirmed yet, so the UI must not claim it is
            # listening to a request; that happens only if the trigger matches.
            return
        if forced or self._cap_voiced >= self._min_voiced_frames:
            self._announce_capture()

    def _announce_capture(self) -> None:
        """Tell the UI a capture is under way -- once per capture."""
        self._announced = True
        if self._on_capture_start is not None:
            try:
                self._on_capture_start()
            except Exception:
                log.warning("on_capture_start raised an exception; ignoring", exc_info=True)

    def _notify_dropped(self, announced: bool) -> None:
        if announced and self._on_capture_dropped is not None:
            try:
                self._on_capture_dropped()
            except Exception:
                log.warning("on_capture_dropped raised an exception; ignoring", exc_info=True)

    def _noise_reason(self, voiced: int, active: int) -> str | None:
        """Why a VAD-opened capture is noise rather than speech, or None."""
        if self._min_voiced_frames and voiced < self._min_voiced_frames:
            return (
                f"too little speech (voiced {voiced * _VAD_FRAME_MS}ms < "
                f"wake.min_voiced_ms {self._cfg.min_voiced_ms}ms)"
            )
        if self._cfg.min_voiced_ratio > 0 and active > 0:
            ratio = voiced / active
            if ratio < self._cfg.min_voiced_ratio:
                return (
                    f"sparse speech (voiced ratio {ratio:.2f} < "
                    f"wake.min_voiced_ratio {self._cfg.min_voiced_ratio:.2f})"
                )
        return None

    def _finish_capture(self, early: bool = False) -> None:
        """Close the utterance, dispatch it, and go back to waiting.

        Args:
            early: Closed by the smart-turn model, after a short pause rather
                than the full ``silence_ms``.
        """
        buf = (
            np.concatenate(self._utterance).astype(np.float32, copy=False)
            if self._utterance
            else np.zeros(0, dtype=np.float32)
        )
        needs_trigger = self._needs_trigger
        trigger = self._trigger
        forced = self._forced
        announced = self._announced
        follow_up_restore = self._follow_up_restore
        voiced = self._cap_voiced
        active = (
            self._cap_last_voiced - self._cap_first_voiced + 1
            if self._cap_first_voiced is not None
            else 0
        )
        turn_line = self._turn_log_line()
        trailing_silence = self._silent_frames * self._vad_frame_len
        # How long the user had already been quiet when the turn closed: the
        # first slice of the wait they feel, before any of the turn runs.
        self.last_trailing_silence_ms = (
            None if self._forced else self._silent_frames * _VAD_FRAME_MS
        )
        # Keep the pre-roll: see _reset_capture_state. This is what stops the
        # next wake word from being clipped.
        self._reset_capture_state(clear_ring=False)
        if early:
            # _reset_capture_state keeps the ring on the promise that it holds
            # only silence. After a short pause it also holds the end of what
            # was just said, which would be transcribed AGAIN as the pre-roll
            # of the next capture. Keep only the pause itself.
            self._ring.keep_last(trailing_silence)

        if not forced:
            reason = self._noise_reason(voiced, active)
            if reason is not None:
                # Dropped BEFORE Whisper: no transcription, no turn, no
                # barge-in. Logged at INFO with the numbers, so the knobs can
                # be tuned from wawi.log alone.
                log.info(
                    "Dropped segment as noise: %s; %.2fs captured, %d/%d voiced frames",
                    reason, buf.size / self._cfg.sample_rate, voiced, active,
                )
                self._bump("segments_dropped_noise")
                if follow_up_restore:
                    # The window was for the user's reply; a cough must not
                    # spend it.
                    self._follow_up_left = max(0, follow_up_restore - int(buf.size))
                self._notify_dropped(announced)
                return

        buf = buf.view(CapturedAudio)
        from wawi.speechgate import SpeechEvidence

        buf.speech = SpeechEvidence(
            voiced_ms=voiced * _VAD_FRAME_MS,
            active_ms=active * _VAD_FRAME_MS,
            forced=forced,
        )

        buf.wake = trigger
        if needs_trigger:
            buf.wake = self._trigger_present(buf)
            if buf.wake is None:
                self._bump("segments_rejected")
                if turn_line is not None:
                    # Rejected whisper-engine candidates are most of the
                    # speech near the mic; at INFO they would bury the turns.
                    log.debug("%s", turn_line)
                return
        elif self._is_own_echo(buf):
            # A follow-up or forced capture skips the TRIGGER check, but it
            # must not skip the ECHO check: observed live, the follow-up window
            # opened on the tail of the assistant's own sentence, captured
            # near-silence, and Whisper hallucinated "¡Suscríbete al canal!"
            # into a turn the user never asked for.
            self._bump("segments_rejected_echo")
            self._notify_dropped(announced)
            return
        if not announced and not needs_trigger:
            # Gates off, or a capture that crossed the bar on its very last
            # frame: the UI still has to hear about it before the turn starts.
            self._announce_capture()
        self._bump("utterances_dispatched")

        if turn_line is not None:
            log.info("%s", turn_line)
        log.info(
            "Utterance captured: %d samples (%.2fs)",
            buf.size,
            buf.size / self._cfg.sample_rate,
        )
        try:
            self._on_utterance(buf)
        except Exception:
            # A broken consumer must not take the listener down with it.
            log.warning("on_utterance raised an exception; ignoring", exc_info=True)


def _validate_config(cfg: ListenerConfig) -> None:
    """Reject configurations the cascade cannot honour."""
    if cfg.sample_rate not in _SUPPORTED_SAMPLE_RATES:
        raise ListenerError(
            f"sample_rate={cfg.sample_rate} is not supported; openwakeword and "
            f"Silero VAD are 16000 Hz only."
        )
    if cfg.pre_roll_s < 0:
        raise ListenerError(f"pre_roll_s must be >= 0, got {cfg.pre_roll_s}")
    if cfg.silence_ms <= 0:
        raise ListenerError(f"silence_ms must be positive, got {cfg.silence_ms}")
    if cfg.max_utterance_s <= 0:
        raise ListenerError(f"max_utterance_s must be positive, got {cfg.max_utterance_s}")
    if not 0.0 <= cfg.min_speech_level <= 1.0:
        raise ListenerError(
            f"min_speech_level must be 0.0..1.0, got {cfg.min_speech_level}"
        )
    if not 0.0 <= cfg.vad_threshold <= 1.0:
        raise ListenerError(f"vad_threshold must be 0.0..1.0, got {cfg.vad_threshold}")
    if not 0.0 <= cfg.wake_threshold <= 1.0:
        raise ListenerError(f"wake_threshold must be 0.0..1.0, got {cfg.wake_threshold}")
    if not 0.0 <= cfg.smart_turn_threshold <= 1.0:
        raise ListenerError(
            f"smart_turn_threshold must be 0.0..1.0, got {cfg.smart_turn_threshold}"
        )
    for name in ("vad_onset_threshold", "min_voiced_ratio"):
        value = getattr(cfg, name)
        if not 0.0 <= value <= 1.0:
            raise ListenerError(f"{name} must be 0.0..1.0, got {value}")
    for name in ("min_onset_ms", "min_voiced_ms"):
        if getattr(cfg, name) < 0:
            raise ListenerError(f"{name} must be >= 0, got {getattr(cfg, name)}")
    if cfg.onset_window_ms <= 0:
        raise ListenerError(f"onset_window_ms must be positive, got {cfg.onset_window_ms}")
    if not cfg.smart_turn_threshold <= cfg.smart_turn_sure <= 1.0:
        raise ValueError(
            f"smart_turn_sure must be between smart_turn_threshold and 1.0, got {cfg.smart_turn_sure}"
        )
    if cfg.smart_turn_pause_ms <= 0:
        raise ListenerError(
            f"smart_turn_pause_ms must be positive, got {cfg.smart_turn_pause_ms}"
        )
    if cfg.inference_framework not in ("onnx", "tflite"):
        raise ListenerError(
            f"inference_framework must be 'onnx' or 'tflite', got "
            f"{cfg.inference_framework!r}"
        )
    if cfg.wake_patience < 1:
        raise ListenerError(f"wake_patience must be >= 1, got {cfg.wake_patience}")
    if cfg.wake_window_s < 0:
        raise ListenerError(f"wake_window_s must be >= 0, got {cfg.wake_window_s}")
    if cfg.engine not in ("openwakeword", "whisper"):
        raise ListenerError(
            f"engine must be 'openwakeword' or 'whisper', got {cfg.engine!r}"
        )
