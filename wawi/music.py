"""Bailar con música: when music is playing, Wawi dances.

The owner's idea: "cuando wawi escucha música, debería reaccionar tipo
bailar". Two ways she knows music is on, combined:

* **On this Mac** (the reliable one): every few seconds, if Spotify or
  Music.app is RUNNING (checked without AppleScript, so neither is ever
  launched), one ``osascript`` asks it whether it is playing, and what.
  Browsers (YouTube) are not asked: macOS only exposes that through the
  private MediaRemote framework. Music from a browser still reaches her
  through the microphone like any other music in the room.
* **In the room**, through the microphone: the listener hands every audio
  block it already receives to :meth:`MicMusicDetector.feed` (no second
  stream). The analysis runs here, on this module's thread, once a second
  over the last :data:`WINDOW_S` seconds: sustained energy, a steady beat
  (autocorrelation of the spectral-flux onset envelope, 70–180 BPM), stable
  tones in the long-term spectrum, and the VAD saying it is mostly not
  speech. Conservative on purpose: talk, typing, a fan or TV dialogue must
  not make her dance. The beat's period gives her tempo.

Either source is *evidence*; :class:`Hysteresis` turns it into dancing only
after :data:`DEFAULTS` ``start_after_s`` of it, and stops after
``stop_after_s`` without. Nothing here touches AppKit: the panel reads
:meth:`MusicService.cue` from its render tick and decides whether she may
dance right now (never in a turn, a note, a game, a drag, or asleep at night;
see ``Panel._tick_music``). Silent: no barks, at most a "♪" bubble.

``WAWI_DANCE=1`` forces the music on (debug, like ``WAWI_CURSOR_REACTION``);
``WAWI_DANCE=96`` also fixes the tempo.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # Seconds of confident music before she starts, and of none before she stops.
    "start_after_s": 5.0,
    "stop_after_s": 6.0,
    # How often Spotify / Music.app are asked (only while one is running).
    "poll_s": 4.0,
    # The two sources; either can be switched off on its own.
    "mac": True,
    "mic": True,
    # Mic detector thresholds (see MicMusicDetector.analyze).
    "min_level": 0.004,
    "max_speech": 0.5,
    "min_beat": 0.3,
    "min_tonal": 0.3,
    # A "♪" bubble when she starts, and again now and then.
    "bubble_every_s": 120.0,
    "local_file": "music_local.json",
}

#: Seconds of microphone audio each analysis looks at.
WINDOW_S = 8.0
#: The tempo range a beat is looked for in (BPM).
BPM_MIN = 70.0
BPM_MAX = 180.0
#: Analysis frame and hop at 16 kHz (32 ms / 16 ms).
_N_FFT = 512
_HOP = 256
#: Below this spread the onset envelope has no attacks in it at all (a held
#: tone or a steady hum, whose tiny wobbles would still autocorrelate).
_MIN_PULSE = 0.01
#: FFT bins (31.25 Hz each) where the onset bands start: roughly octaves
#: from 60 Hz to 8 kHz.
_BAND_EDGES = np.array([2, 4, 8, 16, 32, 64, 128, 256])
#: A gap this long in the fed audio (the listener was paused while she
#: spoke) restarts the window: audio from across it is not one piece of music.
_GAP_S = 0.6

#: The apps asked, by bundle id: (bundle id, AppleScript application name).
PLAYERS = (("com.spotify.client", "Spotify"), ("com.apple.Music", "Music"))


@dataclass(frozen=True)
class MusicSettings:
    enabled: bool = True
    start_after_s: float = 5.0
    stop_after_s: float = 6.0
    poll_s: float = 4.0
    mac: bool = True
    mic: bool = True
    min_level: float = 0.004
    max_speech: float = 0.5
    min_beat: float = 0.3
    min_tonal: float = 0.3
    bubble_every_s: float = 120.0
    local_file: str = "music_local.json"

    @classmethod
    def from_dict(cls, raw: Any) -> "MusicSettings":
        d = {**DEFAULTS, **(raw if isinstance(raw, dict) else {})}

        def num(key: str, lo: float, hi: float) -> float:
            try:
                return min(hi, max(lo, float(d[key])))
            except (TypeError, ValueError):
                return float(DEFAULTS[key])

        return cls(
            enabled=bool(d.get("enabled", True)),
            start_after_s=num("start_after_s", 0.0, 60.0),
            stop_after_s=num("stop_after_s", 0.0, 120.0),
            poll_s=num("poll_s", 1.0, 60.0),
            mac=bool(d.get("mac", True)),
            mic=bool(d.get("mic", True)),
            min_level=num("min_level", 0.0, 1.0),
            max_speech=num("max_speech", 0.0, 1.0),
            min_beat=num("min_beat", 0.0, 1.0),
            min_tonal=num("min_tonal", 0.0, 1.0),
            bubble_every_s=num("bubble_every_s", 10.0, 3600.0),
            local_file=str(d.get("local_file") or ""),
        )


# --------------------------------------------------------------------------- #
# Music in the room: the microphone detector
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Reading:
    """One analysis of the last :data:`WINDOW_S` seconds of microphone audio."""

    music: bool
    bpm: float | None = None
    level: float = 0.0
    sustained: float = 0.0
    speech: float = 0.0
    beat: float = 0.0
    tonal: float = 0.0
    why: str = ""


def _onset_envelope(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(onset, frame_rms)``: the spectral-flux onset envelope (one value
    per hop) and each frame's RMS. Gain-independent: the band energies are
    scaled by the window's power before the log."""
    frames = np.lib.stride_tricks.sliding_window_view(audio, _N_FFT)[::_HOP]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    power = np.abs(np.fft.rfft(frames * np.hanning(_N_FFT), axis=1)) ** 2
    # Log energy in a few wide bands rather than per bin: hiss in a room
    # makes every single bin flicker, while a kick or a strum moves a whole
    # band; summing first averages the flicker away.
    bands = np.add.reduceat(power, _BAND_EDGES, axis=1)[:, :-1]
    level = float(np.mean(audio * audio)) * _N_FFT or 1e-12
    loge = np.log10(bands / level + 1e-4)
    flux = np.maximum(0.0, np.diff(loge, axis=0)).sum(axis=1)
    # Take away the slow trend (a crescendo, a fan spinning up): only the
    # attacks are left, which is what a beat is made of.
    k = 31
    trend = np.convolve(flux, np.ones(k) / k, mode="same")
    return np.maximum(0.0, flux - trend), rms


def _tonal(audio: np.ndarray, sample_rate: int) -> float:
    """How peaky the long-term spectrum is: the mean dB by which it rises
    above its own running median (±7 bins, ~55 Hz at 16 kHz), 80 Hz–4 kHz.
    Held notes and chords make narrow peaks that 8 s of averaging keeps;
    hiss, a fan, typing and clicks average out smooth (~0.05–0.25 dB, music
    ~0.4 and up). A finer FFT than the onsets': tones need the resolution."""
    n = 2048
    frames = np.lib.stride_tricks.sliding_window_view(audio, n)[:: n // 2]
    ltas = np.mean(np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2, axis=0)
    db = 10.0 * np.log10(ltas + 1e-20)
    med = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(db, 7, mode="edge"), 15),
                    axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    band = (freqs >= 80.0) & (freqs <= 4000.0)
    return float(np.mean(np.maximum(0.0, (db - med)[band])))


def analyze(audio: np.ndarray, sample_rate: int, speech: float, *,
            min_level: float = DEFAULTS["min_level"],
            max_speech: float = DEFAULTS["max_speech"],
            min_beat: float = DEFAULTS["min_beat"],
            min_tonal: float = DEFAULTS["min_tonal"]) -> Reading:
    """Is *audio* (mono float32, a few seconds) music? Pure; numpy only.

    *speech* is the share of the window the VAD called speech. Music needs
    ALL of: a level above *min_level*, sustained (no long silent stretches,
    unlike talk or typing), mostly not speech, attacks (:data:`_MIN_PULSE`)
    with a steady beat in 70–180 BPM (the onset envelope's autocorrelation
    at the beat AND twice the beat, normalised, at least *min_beat*), and
    stable tones (:func:`_tonal`, at least *min_tonal* dB).
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < sample_rate * 3:
        return Reading(False, why="too short")
    level = float(np.sqrt(np.mean(audio * audio)))
    if level < min_level:
        return Reading(False, level=level, why="quiet")
    onset, rms = _onset_envelope(audio)
    fps = sample_rate / _HOP
    # Sustained: quarter-second stretches that are not near-silent.
    q = max(1, int(round(0.25 * fps)))
    n = rms.size // q
    quarters = np.sqrt(np.mean((rms[: n * q] ** 2).reshape(n, q), axis=1)) if n else rms
    sustained = float(np.mean(quarters >= 0.25 * level)) if quarters.size else 0.0
    tonal = _tonal(audio, sample_rate)
    # Attacks at all: a held tone or a steady hum has a flat envelope whose
    # tiny wobbles would still autocorrelate "perfectly".
    pulse = float(onset.std())
    beat, bpm = _beat(onset, fps) if pulse >= _MIN_PULSE else (0.0, None)
    why = []
    if sustained < 0.85:
        why.append(f"gaps {sustained:.2f}")
    if speech > max_speech:
        why.append(f"speech {speech:.2f}")
    if pulse < _MIN_PULSE:
        why.append(f"no attacks {pulse:.1f}")
    elif beat < min_beat:
        why.append(f"beat {beat:.2f}")
    if tonal < min_tonal:
        why.append(f"tonal {tonal:.2f}")
    if why:
        return Reading(False, bpm, level, sustained, float(speech), beat, tonal, ", ".join(why))
    return Reading(True, bpm, level, sustained, float(speech), beat, tonal, "music")


def _beat(onset: np.ndarray, fps: float) -> tuple[float, float | None]:
    """``(strength, bpm)`` of the steadiest beat in the onset envelope.

    Strength is the normalised autocorrelation at the beat period, but only
    as much as the envelope ALSO repeats at twice that period: one lucky
    peak (two keystrokes that happened to line up) is not a beat; a beat
    that goes on for the whole window is. A preference for ~120 BPM
    settles most of the half/double (and 2:3) doubt; :func:`fold_bpm` the
    rest."""
    x = onset - onset.mean()
    if x.size < 16 or not np.any(x):
        return 0.0, None
    size = 1 << int(np.ceil(np.log2(2 * x.size)))
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conj(f), size)[: x.size]
    if acf[0] <= 0.0:
        return 0.0, None
    # Unbiased: a long lag has fewer products in its sum.
    acf = acf / acf[0] * (x.size / (x.size - np.arange(x.size)))
    lo = int(np.floor(60.0 * fps / BPM_MAX))
    hi = int(np.ceil(60.0 * fps / BPM_MIN))
    if 2 * hi + 2 >= acf.size:
        return 0.0, None
    lags = np.arange(lo, hi + 1)
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.5) ** 2)
    # The beat must repeat at one AND two periods (± a frame of jitter).
    one = np.maximum.reduce([acf[lags - 1], acf[lags], acf[lags + 1]])
    two = np.maximum.reduce([acf[2 * lags - 1], acf[2 * lags], acf[2 * lags + 1]])
    strength = np.minimum(one, two)
    i = int(np.argmax(strength * prior))
    # The true peak near the winner (strength took the best of ±1 frame),
    # then a parabola through it for a tempo finer than a frame.
    j = int(lags[i] - 1 + np.argmax(acf[lags[i] - 1: lags[i] + 2]))
    a, b, c = acf[j - 1], acf[j], acf[j + 1]
    denom = a - 2 * b + c
    lag = float(j)
    if denom < 0:
        lag += float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return float(max(0.0, strength[i])), fold_bpm(60.0 * fps / lag)


def fold_bpm(bpm: float) -> float:
    """*bpm* halved or doubled into the 75–150 a dog can dance to: a beat
    heard at double or half time (the usual doubt) dances the same."""
    while bpm >= 150.0:
        bpm /= 2.0
    while bpm < 75.0:
        bpm *= 2.0
    return bpm


class MicMusicDetector:
    """The microphone half: a ring of the last :data:`WINDOW_S` seconds.

    :meth:`feed` is called by the listener's worker for every block (it only
    copies into the ring: a few microseconds). :meth:`read` is called from
    the music thread about once a second and does the actual work.
    """

    def __init__(self, sample_rate: int = 16000, settings: MusicSettings | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.sample_rate = int(sample_rate)
        self.settings = settings or MusicSettings()
        self._clock = clock
        self._lock = threading.Lock()
        self._ring = np.zeros(int(WINDOW_S * self.sample_rate), dtype=np.float32)
        self._filled = 0
        self._pos = 0
        self._last_feed = -1e9
        self._voiced = 0
        self._vad = 0
        self._speech: list[tuple[int, int]] = []
        self.enabled = True

    def feed(self, block: np.ndarray, voiced: int = 0, vad_frames: int = 0) -> None:
        """One block of microphone audio and how many of its VAD frames were
        speech. Listener worker thread; never raises, never blocks for long."""
        if not self.enabled:
            return
        try:
            block = np.asarray(block, dtype=np.float32).reshape(-1)
            now = self._clock()
            with self._lock:
                if now - self._last_feed > _GAP_S:
                    self._filled = self._pos = 0
                    self._speech.clear()
                self._last_feed = now
                n = min(block.size, self._ring.size)
                block = block[-n:]
                end = self._pos + n
                if end <= self._ring.size:
                    self._ring[self._pos:end] = block
                else:
                    cut = self._ring.size - self._pos
                    self._ring[self._pos:] = block[:cut]
                    self._ring[: n - cut] = block[cut:]
                self._pos = end % self._ring.size
                self._filled = min(self._ring.size, self._filled + n)
                self._speech.append((int(voiced), int(vad_frames)))
                # Enough blocks for the window (1024-sample blocks: ~125).
                keep = int(WINDOW_S * self.sample_rate / max(1, n)) + 1
                if len(self._speech) > keep:
                    del self._speech[: len(self._speech) - keep]
        except Exception:
            log.debug("Music detector feed failed", exc_info=True)

    def snapshot(self) -> tuple[np.ndarray, float] | None:
        """The full window in order, and its speech share; None while the
        window is not full of fresh audio."""
        with self._lock:
            if self._filled < self._ring.size or self._clock() - self._last_feed > _GAP_S:
                return None
            audio = np.concatenate([self._ring[self._pos:], self._ring[: self._pos]])
            voiced = sum(v for v, _ in self._speech)
            total = sum(t for _, t in self._speech)
        return audio, (voiced / total if total else 0.0)

    def read(self) -> Reading | None:
        """Analyse the window now; None when there is no full window."""
        snap = self.snapshot()
        if snap is None:
            return None
        audio, speech = snap
        s = self.settings
        return analyze(audio, self.sample_rate, speech, min_level=s.min_level,
                       max_speech=s.max_speech, min_beat=s.min_beat, min_tonal=s.min_tonal)


# --------------------------------------------------------------------------- #
# Music on this Mac: Spotify and Music.app
# --------------------------------------------------------------------------- #


def app_running(bundle_id: str) -> bool:
    """Whether the app is running, asked of the system (never AppleScript,
    which would LAUNCH it to answer)."""
    try:
        from AppKit import NSRunningApplication

        return len(NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id)) > 0
    except Exception:
        return False


def run_osascript(script: str, timeout: float = 3.0) -> str | None:
    """Run one AppleScript; its output, or None on any failure."""
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                             timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def player_script(app: str) -> str:
    """Ask *app* what it is playing: ``"artist\\ttrack"`` while it plays,
    ``""`` otherwise. ``is running`` is checked again inside the script, so
    an app that quit since :func:`app_running` is not relaunched."""
    return (
        f'if application "{app}" is running then\n'
        f'  tell application "{app}"\n'
        '    if player state is playing then\n'
        '      return (artist of current track) & tab & (name of current track)\n'
        '    end if\n'
        '  end tell\n'
        'end if\n'
        'return ""'
    )


@dataclass(frozen=True)
class NowPlaying:
    app: str
    artist: str = ""
    track: str = ""

    @property
    def label(self) -> str:
        """For the log and the bubble: the artist, else the track."""
        return self.artist or self.track


class NowPlayingPoller:
    """Asks the running players, in order, whether one is playing."""

    def __init__(self, running: Callable[[str], bool] = app_running,
                 osascript: Callable[[str], str | None] = run_osascript,
                 players: tuple[tuple[str, str], ...] = PLAYERS) -> None:
        self._running = running
        self._osascript = osascript
        self._players = players

    def poll(self) -> NowPlaying | None:
        for bundle_id, app in self._players:
            if not self._running(bundle_id):
                continue
            out = self._osascript(player_script(app))
            if out is None or not out.strip():
                continue
            artist, _, track = out.strip().partition("\t")
            return NowPlaying(app, artist.strip(), track.strip())
        return None


# --------------------------------------------------------------------------- #
# Evidence → dancing
# --------------------------------------------------------------------------- #


class Hysteresis:
    """On after *start_after* seconds of continuous evidence, off after
    *stop_after* seconds without. Pure; the caller supplies the clock."""

    def __init__(self, start_after: float, stop_after: float) -> None:
        self.start_after = start_after
        self.stop_after = stop_after
        self.on = False
        self._since: float | None = None
        self._last_yes: float | None = None

    def update(self, now: float, evidence: bool) -> bool:
        if evidence:
            if self._since is None:
                self._since = now
            self._last_yes = now
            if not self.on and now - self._since >= self.start_after:
                self.on = True
        else:
            self._since = None
            if self.on and (self._last_yes is None or now - self._last_yes >= self.stop_after):
                self.on = False
        return self.on

    def reset(self) -> None:
        self.on = False
        self._since = self._last_yes = None


@dataclass(frozen=True)
class Cue:
    """She should dance: where the music was noticed, its tempo (None =
    unknown) and, from the Mac, who is playing."""

    source: str
    bpm: float | None = None
    label: str = ""
    #: Increases every time dancing starts, so the panel can tell a new song
    #: session from the same one.
    serial: int = 0


class Store:
    """``music_local.json``: only the menu's switch."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.enabled: bool | None = None
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            log.warning("Ignoring unreadable %s", self.path)
            return
        if isinstance(data, dict) and isinstance(data.get("enabled"), bool):
            self.enabled = data["enabled"]

    def save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps({"enabled": self.enabled}) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Could not save the dance switch to %s: %s", self.path, exc)


def forced_bpm(value: str | None) -> float | None:
    """``WAWI_DANCE``: "" / "0" = off; "1" = on (tempo from detection, else
    the default); a number in 40..240 = on at that tempo. None = off."""
    v = (value or "").strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return None
    try:
        bpm = float(v)
    except ValueError:
        return 0.0
    return bpm if 40.0 <= bpm <= 240.0 else 0.0


class MusicService:
    """Owns the detector, the poller, the hysteresis and the switch.

    :meth:`tick` runs the whole decision once (the background thread calls
    it about once a second; tests call it with their own clock). The panel
    reads :meth:`cue` from the main thread: a lock-free attribute read.
    """

    def __init__(self, settings: MusicSettings, *, state_path: str | Path | None = None,
                 sample_rate: int = 16000, poller: NowPlayingPoller | None = None,
                 detector: MicMusicDetector | None = None,
                 force: float | None = None) -> None:
        self.settings = settings
        self.store = Store(state_path)
        self.poller = poller if poller is not None else NowPlayingPoller()
        self.detector = detector if detector is not None else MicMusicDetector(
            sample_rate, settings)
        self.force = force
        self._hyst = Hysteresis(settings.start_after_s, settings.stop_after_s)
        self._next_poll = 0.0
        self._playing: NowPlaying | None = None
        self._bpms: list[float] = []
        self._mic_on = False
        self._started_at = 0.0
        self._serial = 0
        self._cue: Cue | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.detector.enabled = self.enabled and settings.mic

    # -- switch ---------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.settings.enabled if self.store.enabled is None else self.store.enabled

    def set_enabled(self, on: bool) -> None:
        """The menu's checkmark. Persisted; switching off stops her at once."""
        self.store.enabled = bool(on)
        self.store.save()
        self.detector.enabled = bool(on) and self.settings.mic
        if not on:
            self._end(time.monotonic())
            self._hyst.reset()
        log.info("Dancing to music %s", "on" if on else "off")

    # -- inputs ---------------------------------------------------------------
    def feed(self, block: np.ndarray, voiced: int = 0, vad_frames: int = 0) -> None:
        """Listener block observer (see ``Listener.add_block_observer``)."""
        self.detector.feed(block, voiced, vad_frames)

    # -- the decision -----------------------------------------------------------
    def cue(self) -> Cue | None:
        return self._cue

    def tick(self, now: float) -> Cue | None:
        """Look and listen once; update and return the cue."""
        if not self.enabled:
            return None
        s = self.settings
        if s.mac and now >= self._next_poll:
            self._next_poll = now + s.poll_s
            try:
                self._playing = self.poller.poll()
            except Exception:
                log.debug("Now-playing poll failed", exc_info=True)
                self._playing = None
        if not s.mac:
            self._playing = None
        reading = self.detector.read() if s.mic else None
        mic = reading is not None and reading.music
        if mic and reading is not None and reading.bpm:
            self._bpms = (self._bpms + [reading.bpm])[-9:]
        elif not mic:
            self._bpms = self._bpms[-3:] if self._cue is not None else []
        if mic != self._mic_on:
            self._mic_on = mic
            if reading is not None:
                log.debug("Mic music %s (bpm %s, beat %.2f, tonal %.2f, gaps %.2f, "
                          "speech %.2f, level %.4f: %s)", "yes" if mic else "no",
                          f"{reading.bpm:.0f}" if reading.bpm else "?", reading.beat,
                          reading.tonal, reading.sustained, reading.speech,
                          reading.level, reading.why)
        forced = self.force is not None
        on = self._hyst.update(now, forced or mic or self._playing is not None)
        if on:
            source = "forced" if forced else "mac" if self._playing is not None else "mic"
            label = self._playing.label if self._playing is not None else ""
            if self.force:
                bpm: float | None = self.force
            else:
                bpm = float(np.median(self._bpms)) if self._bpms else None
            if self._cue is None:
                self._serial += 1
                self._started_at = now
                log.info("Music: dancing (source %s, %s BPM%s)", source,
                         f"{bpm:.0f}" if bpm else "unknown",
                         f", {label}" if label else "")
            self._cue = Cue(source, bpm, label, self._serial)
        else:
            self._end(now)
        return self._cue

    def _end(self, now: float) -> None:
        if self._cue is not None:
            log.info("Music: stopped dancing after %.0fs", max(0.0, now - self._started_at))
        self._cue = None
        self._bpms = []

    # -- the thread -------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="wawi-music", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self.tick(time.monotonic())
            except Exception:
                log.debug("Music tick failed", exc_info=True)


def dance_blocked(*, state: Any, key: str, notes: bool, game: bool, dragging: bool,
                  hidden: bool, presenting: bool, working: bool) -> str | None:
    """Why she may not dance right now, or None. Pure, for the panel.

    Never in a turn (listening, thinking, speaking: the turn's frames win at
    once), taking notes (wawi/notes.py), in a game, being dragged, hidden
    (hide-and-seek), presenting on the board, or showing a tool's work."""
    if getattr(state, "value", state) != "IDLE" or key != "idle":
        return "turn"
    for why, on in (("notes", notes), ("game", game), ("dragging", dragging),
                    ("hidden", hidden), ("board", presenting), ("working", working)):
        if on:
            return why
    return None


# --------------------------------------------------------------------------- #
# Process-wide instance (like wawi.curiosities)
# --------------------------------------------------------------------------- #

_SERVICE: MusicService | None = None


def install(service: MusicService | None) -> None:
    global _SERVICE
    _SERVICE = service


def get() -> MusicService | None:
    return _SERVICE


def build(cfg: Any) -> MusicService:
    """The service for *cfg* (config ``music:``)."""
    settings = MusicSettings.from_dict(getattr(cfg, "music", {}) or {})
    return MusicService(settings, state_path=settings.local_file or None,
                        sample_rate=int(getattr(cfg, "sample_rate", 16000) or 16000),
                        force=forced_bpm(os.environ.get("WAWI_DANCE")))
