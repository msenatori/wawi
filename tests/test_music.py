"""Bailar con música (wawi/music.py): detection, hysteresis, the poller, the
switch, and her dancing (wawi/mascot.py) and when she may (the panel)."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from wawi import mascot as _mascot
from wawi import music
from wawi.music import (
    Cue,
    Hysteresis,
    MicMusicDetector,
    MusicService,
    MusicSettings,
    NowPlayingPoller,
    analyze,
    dance_blocked,
    fold_bpm,
    forced_bpm,
)
from wawi.state import State

SR = 16000
T = np.arange(int(8 * SR)) / SR


# --------------------------------------------------------------------------- #
# Synthetic signals
# --------------------------------------------------------------------------- #


def click_track(bpm: float, rng: np.random.Generator) -> np.ndarray:
    """A metronome (short decaying 1.5 kHz ticks) over a soft held chord: the
    simplest thing that is music, with an exact tempo."""
    out = np.zeros_like(T)
    n = int(0.06 * SR)
    tt = np.arange(n) / SR
    tick = np.sin(2 * np.pi * 1500 * tt) * np.exp(-tt / 0.015)
    k = 0.0
    while k < 8.0:
        s = int(k * SR)
        if s + n < T.size:
            out[s:s + n] += tick
        k += 60.0 / bpm
    chord = sum(np.sin(2 * np.pi * f * T) for f in (220.0, 277.2, 329.6)) / 3
    return (0.1 * out + 0.03 * chord + 0.002 * rng.standard_normal(T.size)).astype(np.float32)


def song(bpm: float, rng: np.random.Generator, hiss: float = 0.0) -> np.ndarray:
    """Kick on every beat, a bass line, hi-hat on the off-beats, a pad."""
    beat = 60.0 / bpm
    out = np.zeros_like(T)
    notes = (110.0, 146.8, 164.8, 98.0)
    for i in range(int(8 / beat) + 1):
        s = int(i * beat * SR)
        n = min(int(beat * SR), T.size - s)
        if n <= 0:
            break
        tt = np.arange(n) / SR
        out[s:s + n] += 0.5 * np.sin(2 * np.pi * notes[(i // 2) % 4] * tt) * np.exp(-tt / 0.3)
        out[s:s + n] += 0.8 * np.sin(2 * np.pi * (60 + 80 * np.exp(-tt / 0.02)) * tt) \
            * np.exp(-tt / 0.08)
        h = int(beat / 2 * SR)
        if h < n:
            m = min(800, n - h)
            out[s + h:s + h + m] += 0.15 * rng.standard_normal(m) * np.exp(-np.arange(m) / 200)
    pad = sum(np.sin(2 * np.pi * f * T) for f in (220.0, 261.6, 329.6)) / 3
    return (0.1 * (out + 0.3 * pad) + hiss * rng.standard_normal(T.size)).astype(np.float32)


def speech_like(rng: np.random.Generator) -> np.ndarray:
    """Band-limited noise plus a wandering voiced buzz, syllable-modulated at
    ~4 Hz with pauses: talk, as far as rhythm and energy go."""
    noise = rng.standard_normal(T.size)
    f = np.fft.rfftfreq(T.size, 1 / SR)
    noise = np.fft.irfft(np.fft.rfft(noise) * ((f > 200) & (f < 3000)), T.size)
    env = np.clip(np.sin(2 * np.pi * 4.3 * T + 2 * np.sin(2 * np.pi * 0.7 * T)), 0, None) ** 2
    env *= np.sin(2 * np.pi * 0.23 * T) > -0.6
    voiced = sum(np.sin(2 * np.pi * k * (130 + 20 * np.sin(2 * np.pi * 0.5 * T)) * T) / k
                 for k in range(1, 8))
    return (0.05 * (0.5 * noise / noise.std() + voiced) * env).astype(np.float32)


def typing(rng: np.random.Generator) -> np.ndarray:
    """Sparse irregular clicks over a quiet room."""
    out = np.zeros_like(T)
    k = 0.2
    while k < 7.9:
        s = int(k * SR)
        out[s:s + 300] += rng.standard_normal(300) * np.exp(-np.arange(300) / 40)
        k += rng.exponential(0.22) + 0.05
    return (0.1 * out + 0.0001 * rng.standard_normal(T.size)).astype(np.float32)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #


class TestAnalyze:
    def test_click_track_at_120_is_music_at_120(self, rng):
        r = analyze(click_track(120, rng), SR, speech=0.0)
        assert r.music, r
        assert r.bpm == pytest.approx(120, abs=3)

    @pytest.mark.parametrize("bpm", [84, 100, 140])
    def test_a_song_is_music_at_its_tempo(self, rng, bpm):
        r = analyze(song(bpm, rng), SR, speech=0.1)
        assert r.music, r
        assert r.bpm == pytest.approx(bpm, rel=0.04)

    def test_a_song_through_room_hiss_still_is(self, rng):
        r = analyze(song(100, rng, hiss=0.01), SR, speech=0.1)
        assert r.music, r
        assert r.bpm == pytest.approx(100, rel=0.04)

    def test_speech_like_noise_is_not_music(self, rng):
        a = speech_like(rng)
        assert not analyze(a, SR, speech=0.7).music
        # Not even if the VAD missed it: no steady beat.
        r = analyze(a, SR, speech=0.0)
        assert not r.music and "beat" in r.why

    def test_white_noise_is_not_music(self, rng):
        r = analyze((0.05 * rng.standard_normal(T.size)).astype(np.float32), SR, speech=0.0)
        assert not r.music

    def test_a_fan_with_a_hum_is_not_music(self, rng):
        a = 0.02 * rng.standard_normal(T.size) + 0.01 * np.sin(2 * np.pi * 120 * T)
        assert not analyze(a.astype(np.float32), SR, speech=0.0).music

    def test_a_held_tone_is_not_music(self):
        r = analyze((0.05 * np.sin(2 * np.pi * 440 * T)).astype(np.float32), SR, speech=0.0)
        assert not r.music and "attacks" in r.why

    def test_silence_is_not_music(self):
        r = analyze(np.zeros(T.size, np.float32), SR, speech=0.0)
        assert not r.music and r.why == "quiet"

    def test_typing_is_not_music(self, rng):
        assert not analyze(typing(rng), SR, speech=0.0).music

    def test_a_metronome_alone_is_not_music(self, rng):
        """Ticks in silence (a clock): a beat, but no sustained sound."""
        out = click_track(120, rng) - 0.03 * (
            sum(np.sin(2 * np.pi * f * T) for f in (220.0, 277.2, 329.6)) / 3)
        assert not analyze(out.astype(np.float32), SR, speech=0.0).music

    def test_music_the_vad_calls_speech_is_not_trusted(self, rng):
        r = analyze(song(100, rng), SR, speech=0.9)
        assert not r.music and "speech" in r.why

    def test_is_cheap(self, rng):
        import time

        a = song(100, rng)
        t0 = time.perf_counter()
        for _ in range(5):
            analyze(a, SR, speech=0.0)
        assert (time.perf_counter() - t0) / 5 < 0.05


def test_fold_bpm():
    assert fold_bpm(120) == 120
    assert fold_bpm(168) == 84
    assert fold_bpm(60) == 120
    assert fold_bpm(150) == 75


class TestMicDetector:
    def feed_all(self, det, audio, voiced=0, clock=None):
        for i in range(0, audio.size, 1024):
            if clock is not None:
                clock[0] += 1024 / SR
            det.feed(audio[i:i + 1024], voiced, 2)

    def test_reads_nothing_until_the_window_is_full(self, rng):
        clock = [0.0]
        det = MicMusicDetector(SR, clock=lambda: clock[0])
        self.feed_all(det, song(100, rng)[: SR * 4], clock=clock)
        assert det.read() is None

    def test_a_full_window_of_music_reads_as_music(self, rng):
        clock = [0.0]
        det = MicMusicDetector(SR, clock=lambda: clock[0])
        self.feed_all(det, song(100, rng), clock=clock)
        r = det.read()
        assert r is not None and r.music and r.bpm == pytest.approx(100, rel=0.04)

    def test_the_vad_share_comes_from_the_blocks(self, rng):
        clock = [0.0]
        det = MicMusicDetector(SR, clock=lambda: clock[0])
        self.feed_all(det, song(100, rng), voiced=2, clock=clock)
        r = det.read()
        assert r is not None and r.speech == 1.0 and not r.music

    def test_a_gap_restarts_the_window(self, rng):
        """The listener paused while she spoke: no stitching across it."""
        clock = [0.0]
        det = MicMusicDetector(SR, clock=lambda: clock[0])
        self.feed_all(det, song(100, rng), clock=clock)
        clock[0] += 2.0
        assert det.read() is None  # stale
        self.feed_all(det, song(100, rng)[:SR], clock=clock)
        assert det.read() is None  # only one fresh second

    def test_disabled_ignores_blocks(self, rng):
        clock = [0.0]
        det = MicMusicDetector(SR, clock=lambda: clock[0])
        det.enabled = False
        self.feed_all(det, song(100, rng), clock=clock)
        assert det.read() is None

    def test_feed_never_raises(self):
        det = MicMusicDetector(SR)
        det.feed(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Hysteresis
# --------------------------------------------------------------------------- #


class TestHysteresis:
    def test_starts_only_after_sustained_evidence(self):
        h = Hysteresis(5.0, 6.0)
        assert not h.update(0.0, True)
        assert not h.update(4.0, True)
        assert h.update(5.0, True)

    def test_a_break_in_the_evidence_restarts_the_count(self):
        h = Hysteresis(5.0, 6.0)
        h.update(0.0, True)
        h.update(3.0, False)
        assert not h.update(7.0, True)
        assert h.update(12.0, True)

    def test_stops_only_after_a_while_without(self):
        h = Hysteresis(5.0, 6.0)
        h.update(0.0, True)
        h.update(5.0, True)
        assert h.update(8.0, False)
        assert h.update(10.9, False)
        assert not h.update(11.0, False)

    def test_a_short_dip_does_not_stop_her(self):
        h = Hysteresis(5.0, 6.0)
        h.update(0.0, True)
        h.update(5.0, True)
        h.update(7.0, False)
        assert h.update(9.0, True)
        assert h.update(14.0, False)


# --------------------------------------------------------------------------- #
# Now playing on this Mac
# --------------------------------------------------------------------------- #


class TestNowPlaying:
    def test_a_player_not_running_is_never_asked(self):
        asked = []
        p = NowPlayingPoller(running=lambda b: False,
                             osascript=lambda s: asked.append(s) or "x\ty")
        assert p.poll() is None
        assert asked == []

    def test_spotify_playing(self):
        p = NowPlayingPoller(running=lambda b: b == "com.spotify.client",
                             osascript=lambda s: "Soda Stereo\tDe música ligera\n")
        now = p.poll()
        assert now is not None
        assert (now.app, now.artist, now.track) == ("Spotify", "Soda Stereo", "De música ligera")
        assert now.label == "Soda Stereo"

    def test_paused_or_failing_is_not_playing(self):
        for out in ("", None):
            p = NowPlayingPoller(running=lambda b: True, osascript=lambda s, o=out: o)
            assert p.poll() is None

    def test_music_app_after_spotify(self):
        scripts = []

        def osa(script):
            scripts.append(script)
            return "" if '"Spotify"' in script else "Charly García\tYendo de la cama al living"

        now = NowPlayingPoller(running=lambda b: True, osascript=osa).poll()
        assert now is not None and now.app == "Music" and len(scripts) == 2

    def test_the_script_cannot_launch_the_app(self):
        script = music.player_script("Spotify")
        assert script.startswith('if application "Spotify" is running then')
        assert "player state is playing" in script


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


class FakePoller:
    def __init__(self, playing=None):
        self.playing = playing
        self.polls = 0

    def poll(self):
        self.polls += 1
        return self.playing


class FakeDetector:
    def __init__(self, reading=None):
        self.reading = reading
        self.enabled = True
        self.fed = 0

    def feed(self, *a):
        self.fed += 1

    def read(self):
        return self.reading


def service(tmp_path=None, playing=None, reading=None, force=None, **kw):
    settings = MusicSettings.from_dict(kw)
    path = tmp_path / "music_local.json" if tmp_path is not None else None
    return MusicService(settings, state_path=path, poller=FakePoller(playing),
                        detector=FakeDetector(reading), force=force)


class TestService:
    def test_spotify_playing_starts_her_after_a_while(self, caplog):
        s = service(playing=music.NowPlaying("Spotify", "Soda Stereo", "Persiana americana"))
        caplog.set_level(logging.INFO, logger="wawi.music")
        assert s.tick(0.0) is None
        assert s.tick(4.0) is None
        cue = s.tick(5.0)
        assert cue is not None and cue.source == "mac" and cue.label == "Soda Stereo"
        assert cue.bpm is None
        assert any("dancing (source mac" in r.message and "Soda Stereo" in r.message
                   for r in caplog.records)

    def test_stops_after_the_music_does(self, caplog):
        s = service(playing=music.NowPlaying("Spotify", "A", "B"), poll_s=1.0)
        s.tick(0.0)
        s.tick(5.0)
        s.poller.playing = None
        caplog.set_level(logging.INFO, logger="wawi.music")
        assert s.tick(6.0) is not None
        assert s.tick(12.0) is None
        assert any("stopped dancing" in r.message for r in caplog.records)

    def test_mic_music_with_its_tempo(self):
        s = service(reading=music.Reading(True, bpm=118.0))
        for t in range(6):
            cue = s.tick(float(t))
        assert cue is not None and cue.source == "mic" and cue.bpm == pytest.approx(118)

    def test_not_music_never_starts(self):
        s = service(reading=music.Reading(False, bpm=118.0))
        assert all(s.tick(float(t)) is None for t in range(30))

    def test_mac_is_polled_at_its_own_pace(self):
        s = service(poll_s=4.0)
        for t in range(10):
            s.tick(float(t))
        assert s.poller.polls == 3  # at 0, 4 and 8

    def test_sources_can_be_switched_off(self):
        s = service(playing=music.NowPlaying("Spotify", "A", "B"),
                    reading=music.Reading(True, bpm=100.0), mac=False, mic=False)
        assert all(s.tick(float(t)) is None for t in range(10))
        assert s.poller.polls == 0

    def test_a_new_session_gets_a_new_serial(self):
        s = service(playing=music.NowPlaying("Music", "A", "B"), poll_s=1.0)
        s.tick(0.0)
        first = s.tick(5.0)
        s.poller.playing = None
        s.tick(6.0)
        s.tick(12.0)
        s.poller.playing = music.NowPlaying("Music", "A", "B")
        s.tick(13.0)
        second = s.tick(18.0)
        assert first is not None and second is not None and second.serial == first.serial + 1

    def test_forced(self):
        s = service(force=96.0)
        s.tick(0.0)
        cue = s.tick(5.0)
        assert cue is not None and cue.source == "forced" and cue.bpm == 96.0

    def test_feed_reaches_the_detector(self):
        s = service()
        s.feed(np.zeros(1024, np.float32), 0, 2)
        assert s.detector.fed == 1


class TestSwitch:
    def test_on_by_default(self, tmp_path):
        assert service(tmp_path).enabled

    def test_off_is_persisted_and_stops_her(self, tmp_path):
        s = service(tmp_path, playing=music.NowPlaying("Spotify", "A", "B"))
        s.tick(0.0)
        assert s.tick(5.0) is not None
        s.set_enabled(False)
        assert s.cue() is None and s.tick(6.0) is None
        assert not s.detector.enabled
        assert json.loads((tmp_path / "music_local.json").read_text()) == {"enabled": False}
        assert not service(tmp_path).enabled

    def test_back_on(self, tmp_path):
        service(tmp_path).set_enabled(False)
        s = service(tmp_path)
        s.set_enabled(True)
        assert service(tmp_path).enabled

    def test_unreadable_file_is_ignored(self, tmp_path):
        (tmp_path / "music_local.json").write_text("{nope")
        assert service(tmp_path).enabled


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("0", None), ("off", None),
    ("1", 0.0), ("yes", 0.0), ("96", 96.0), ("999", 0.0),
])
def test_forced_bpm(value, expected):
    assert forced_bpm(value) == expected


def test_settings_are_clamped():
    s = MusicSettings.from_dict({"start_after_s": -3, "poll_s": "x", "mic": False})
    assert s.start_after_s == 0.0 and s.poll_s == 4.0 and not s.mic


def test_config_has_the_section(tmp_path):
    from wawi.config import load as load_config

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("")
    cfg = load_config(cfg_file)
    assert cfg.music["enabled"] is True
    assert cfg.music["local_file"] == str(tmp_path / "music_local.json")


# --------------------------------------------------------------------------- #
# When she may dance
# --------------------------------------------------------------------------- #


FREE = dict(state=State.IDLE, key="idle", notes=False, game=False, dragging=False,
            hidden=False, presenting=False, working=False)


class TestDanceBlocked:
    def test_free(self):
        assert dance_blocked(**FREE) is None

    @pytest.mark.parametrize("state", [State.LISTENING, State.THINKING, State.SPEAKING])
    def test_never_in_a_turn(self, state):
        assert dance_blocked(**{**FREE, "state": state}) == "turn"

    def test_the_animator_still_in_a_turn(self):
        assert dance_blocked(**{**FREE, "key": "speaking"}) == "turn"

    @pytest.mark.parametrize("what", ["notes", "game", "dragging", "hidden", "presenting",
                                      "working"])
    def test_never_while_busy(self, what):
        assert dance_blocked(**{**FREE, what: True}) is not None


# --------------------------------------------------------------------------- #
# Her dance (wawi/mascot.py)
# --------------------------------------------------------------------------- #


ALL = _mascot.FRAME_NAMES


def dancer(**kw):
    import random

    return _mascot.MascotAnimator(ALL, rng=random.Random(3), **kw)


def run(anim, seconds, state=State.IDLE, **kw):
    frames = []
    for _ in range(int(seconds * 30)):
        anim.update(1 / 30, state, **kw)
        frames.append(anim.frame)
    return frames


class TestDance:
    def test_dances_through_her_frames(self):
        anim = dancer()
        anim.dance_bpm = 120.0
        frames = run(anim, 4.5)
        assert anim.dancing
        assert set(_mascot.DANCE_FRAMES) <= set(frames)

    def test_frames_change_on_the_beat(self):
        """At 120 BPM the first change is half a second in; at 60, a second."""
        for bpm, at in ((120.0, 0.5), (60.0 * 2 * 0.75, 0.5 / 0.75)):
            anim = dancer()
            anim.dance_bpm = fold_bpm(bpm)
            frames = run(anim, 1.5)
            first = next(i for i, f in enumerate(frames) if f != frames[0])
            assert first / 30 == pytest.approx(at, abs=0.05)

    def test_unknown_tempo_still_dances(self):
        anim = dancer()
        anim.dance_bpm = 0.0
        frames = run(anim, 2.0)
        assert "dance_a" in frames and "dance_b" in frames

    def test_a_turn_stops_her_at_once(self):
        anim = dancer()
        anim.dance_bpm = 120.0
        run(anim, 1.0)
        anim.update(1 / 30, State.LISTENING)
        assert not anim.dancing
        assert anim.frame not in _mascot.DANCE_FRAMES

    def test_never_falls_asleep_dancing(self):
        anim = dancer(sleep_after=2.0)
        anim.dance_bpm = 120.0
        run(anim, 10.0)
        assert not anim.asleep and anim.dancing

    def test_wakes_up_to_dance(self):
        anim = dancer(sleep_after=1.0)
        run(anim, 30.0)
        assert anim.asleep
        anim.dance_bpm = 120.0
        run(anim, 0.5)
        assert not anim.asleep and anim.frame in _mascot.DANCE_FRAMES

    def test_an_act_wins_and_she_picks_up_after(self):
        anim = dancer()
        anim.dance_bpm = 120.0
        run(anim, 0.5)
        assert anim.act("wag")
        assert not anim.dancing
        run(anim, 5.0)
        assert anim.dancing

    def test_tool_work_wins(self):
        anim = dancer()
        anim.dance_bpm = 120.0
        anim.working = "laptop"
        run(anim, 0.5)
        assert not anim.dancing

    def test_the_spin_mirrors_her_half_way(self):
        anim = dancer()
        anim.dance_bpm = 120.0
        mirrored = []
        for _ in range(150):
            anim.update(1 / 30, State.IDLE)
            mirrored.append(anim.mirrored)
        assert any(mirrored) and not all(mirrored)

    def test_reduce_motion_is_a_gentle_bob(self):
        anim = dancer()
        anim.dance_bpm = 120.0
        poses, frames = [], []
        for _ in range(90):
            poses.append(anim.update(1 / 30, State.IDLE, reduce_motion=True))
            frames.append(anim.frame)
        assert set(frames) == {"dance_a"}
        assert max(p.dy for p in poses) <= 1.01 and max(p.dy for p in poses) > 0.5
        assert all(p.rot == 0 and p.sx == 1 for p in poses)

    def test_no_cursor_barks_while_dancing(self):
        anim = dancer()
        anim.dance_bpm = 120.0
        run(anim, 0.5)
        assert not anim.react_to_cursor("bark")
        assert not anim.react_to_cursor("wag")

    def test_without_the_art_she_borrows_her_joy_frames(self):
        import random

        have = [f for f in ALL if f not in _mascot.DANCE_FRAMES]
        anim = _mascot.MascotAnimator(have, rng=random.Random(3))
        anim.dance_bpm = 120.0
        frames = set(run(anim, 4.5))
        assert {"happy", "happy_b"} <= frames

    def test_the_art_is_shipped(self):
        assert set(_mascot.DANCE_FRAMES) <= _mascot.available_frames()


# --------------------------------------------------------------------------- #
# The panel's side
# --------------------------------------------------------------------------- #


@pytest.fixture
def stage(monkeypatch):
    from wawi.panel import PanelObserver

    p = PanelObserver(enabled=False)
    p._mascot_animator = dancer()
    s = service()
    music.install(s)
    yield p, s
    music.install(None)


def set_cue(s, cue):
    s._cue = cue


class TestPanel:
    def test_dances_when_idle(self, stage, monkeypatch):
        p, s = stage
        monkeypatch.setattr("time.localtime", lambda *a: __import__("time").struct_time(
            (2026, 10, 17, 15, 0, 0, 5, 290, 0)))
        set_cue(s, Cue("mac", 104.0, "Soda Stereo", 1))
        p._tick_music()
        assert p._mascot_animator.dance_bpm == 104.0
        run(p._mascot_animator, 0.2)
        assert p._music_bubble() == "♪ Soda Stereo"

    def test_not_in_a_turn(self, stage):
        p, s = stage
        set_cue(s, Cue("mic", 100.0, "", 1))
        p._current_state = State.SPEAKING
        p._tick_music()
        assert p._mascot_animator.dance_bpm is None

    def test_not_taking_notes(self, stage):
        p, s = stage
        set_cue(s, Cue("mic", 100.0, "", 1))
        p.notes_session = object()
        p._tick_music()
        assert p._mascot_animator.dance_bpm is None

    def test_not_when_switched_off(self, stage):
        p, s = stage
        set_cue(s, Cue("mic", 100.0, "", 1))
        s.store.enabled = False
        p._tick_music()
        assert p._mascot_animator.dance_bpm is None

    def test_at_night_only_an_ear_twitch(self, stage, monkeypatch):
        p, s = stage
        import time as _time

        monkeypatch.setattr(_time, "localtime", lambda *a: _time.struct_time(
            (2026, 10, 17, 23, 30, 0, 5, 290, 0)))
        anim = p._mascot_animator
        anim.sleep_after = 1.0
        run(anim, 30.0)
        assert anim.asleep
        set_cue(s, Cue("mac", 100.0, "", 1))
        p._tick_music()
        assert anim.dance_bpm is None
        assert anim.behaviour == "cursor_ear_twitch"
        anim.update(1 / 30, State.IDLE)
        p._tick_music()  # same song: no second twitch
        assert anim.dance_bpm is None
