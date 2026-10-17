"""Voice character: pitch shift, voice blends, brightness and the preset.

Kokoro is never loaded: the transforms are pure functions over numpy arrays,
and the Voice plumbing is exercised with a stand-in Kokoro that returns a
synthetic tone whose length follows the requested speed, like the real one.
"""

from __future__ import annotations

import textwrap
import threading

import numpy as np
import pytest

from wawi import config as config_mod
from wawi.voice import (
    TTS_PRESETS,
    Voice,
    brighten,
    parse_voice_spec,
    primary_voice,
    retune,
)

SR = 24000


def tone(freq: float = 200.0, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def f0(samples: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.argmax(spec) * sr / len(samples))


class FakeKokoro:
    """Returns 1 s of a 200 Hz tone at speed 1.0, shorter/longer with speed."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self.voices = {n: rng.normal(size=(4, 1, 8)).astype(np.float32)
                       for n in ("ef_dora", "af_heart", "af_bella", "jf_alpha")}
        self.calls: list[tuple[object, float]] = []

    def get_voice_style(self, name):
        return self.voices[name]

    def create(self, text, voice, speed, lang):
        self.calls.append((voice, speed))
        return tone(200.0, 1.0 / speed), SR


def make_voice(spec="ef_dora", speed=1.0, pitch=0.0, brightness=0.0, device_sr=None) -> Voice:
    v = Voice.__new__(Voice)
    v._kokoro = FakeKokoro()
    v._configured_voice = spec
    v._voice = v._style_for(spec)
    v._speed = speed
    v._pitch = pitch
    v._brightness = brightness
    v._kokoro_lang = "es"
    v._device_sr = device_sr
    v._abort = threading.Event()
    return v


class TestRetune:
    @pytest.mark.parametrize("semitones", [3, 4, 5, 6])
    def test_raises_f0_by_the_interval(self, semitones):
        out = retune(tone(200.0), SR, semitones)
        expected = 200.0 * 2 ** (semitones / 12)
        assert f0(out, SR) == pytest.approx(expected, rel=0.02)

    @pytest.mark.parametrize("semitones", [3, 4, 5, 6])
    def test_duration_is_kept_when_synthesized_slower(self, semitones):
        # What Voice does: synthesize at speed/ratio, then retune.
        v = make_voice(pitch=semitones)
        samples, sr = v._synth_sentence("Hola.")
        plain, _ = make_voice()._synth_sentence("Hola.")
        assert len(samples) / sr == pytest.approx(len(plain) / SR, rel=0.05)
        assert f0(samples[: int(sr * 0.9)], sr) > 200.0 * 1.15

    def test_also_converts_to_the_device_rate_in_the_same_pass(self):
        out = retune(tone(200.0), SR, 4, 48000)
        assert len(out) / 48000 == pytest.approx(1.0 / 2 ** (4 / 12), rel=0.01)
        assert f0(out, 48000) == pytest.approx(200.0 * 2 ** (4 / 12), rel=0.02)

    def test_zero_shift_at_the_same_rate_is_the_input(self):
        x = tone()
        assert np.array_equal(retune(x, SR, 0), x)

    def test_numpy_fallback_agrees_with_scipy(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_scipy(name, *a, **k):
            if name.startswith("scipy"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_scipy)
        out = retune(tone(200.0), SR, 5)
        assert f0(out, SR) == pytest.approx(200.0 * 2 ** (5 / 12), rel=0.02)


class TestBrighten:
    def test_zero_is_a_no_op(self):
        x = tone()
        assert np.array_equal(brighten(x, 0.0), x)

    def test_lifts_treble_more_than_bass_and_never_clips(self):
        low, high = tone(200.0), tone(5000.0)
        gain_low = np.abs(brighten(low * 0.1, 0.5)).max() / np.abs(low * 0.1).max()
        gain_high = np.abs(brighten(high * 0.1, 0.5)).max() / np.abs(high * 0.1).max()
        assert gain_high > gain_low * 1.3
        assert np.abs(brighten(high * 1.9, 1.0)).max() <= 1.9 + 1e-6


class TestBlendSpec:
    def test_single_name(self):
        assert parse_voice_spec("ef_dora") == [("ef_dora", 1.0)]

    def test_weights_are_normalised(self):
        parts = parse_voice_spec("ef_dora:7, af_heart:3")
        assert [n for n, _ in parts] == ["ef_dora", "af_heart"]
        assert [w for _, w in parts] == pytest.approx([0.7, 0.3])

    @pytest.mark.parametrize(
        "bad",
        ["", "ef_dora:", "ef_dora:abc", "ef_dora:0", "ef_dora:-1", "ef_dora,,af_heart",
         "ef_dora:0.5,ef_dora:0.5", "Dora", "ef_dora:inf"],
    )
    def test_malformed_specs_are_refused(self, bad):
        with pytest.raises(ValueError):
            parse_voice_spec(bad)

    def test_primary_is_the_heaviest(self):
        assert primary_voice("af_heart:0.3,ef_dora:0.7") == "ef_dora"

    def test_a_single_voice_reaches_kokoro_as_its_name(self):
        assert make_voice("ef_dora")._voice == "ef_dora"

    def test_a_blend_is_the_weighted_sum_of_styles(self):
        v = make_voice("ef_dora:0.7,af_heart:0.3")
        k = v._kokoro
        expected = 0.7 * k.voices["ef_dora"] + 0.3 * k.voices["af_heart"]
        assert np.allclose(v._voice, expected)

    def test_unknown_voices_are_left_out_of_the_blend(self):
        v = make_voice("ef_dora:0.5,xx_nobody:0.5")
        assert np.allclose(v._voice, v._kokoro.voices["ef_dora"])

    def test_a_language_round_trip_comes_back_to_the_blend(self):
        v = make_voice("ef_dora:0.7,af_heart:0.3")
        v.set_language("en")
        assert v._voice == "af_heart"
        v.set_language("es")
        assert isinstance(v._voice, np.ndarray)

    def test_a_voice_from_another_family_survives_the_round_trip(self):
        # jf_alpha speaking Spanish shares no letter with "es": only the home
        # language brings it back, instead of the stock ef_dora.
        v = make_voice("jf_alpha")
        v._home_language = "es"
        v.set_language("en")
        assert v._voice == "af_heart"
        v.set_language("es")
        assert v._voice == "jf_alpha"

    def test_without_a_home_language_it_falls_back_to_the_stock_voice(self):
        v = make_voice("jf_alpha")
        v.set_language("es")
        assert v._voice == "ef_dora"


class TestDefaultsUnchanged:
    def test_no_character_means_kokoro_output_untouched(self):
        v = make_voice(speed=1.05)
        samples, sr = v._synth_sentence("Hola.")
        assert v._kokoro.calls == [("ef_dora", 1.05)]
        raw = tone(200.0, 1.0 / 1.05)
        assert sr == SR
        assert np.array_equal(samples[: len(raw)], raw)

    def test_pitch_synthesizes_slower_by_the_ratio(self):
        v = make_voice(speed=1.05, pitch=4)
        v._synth_sentence("Hola.")
        assert v._kokoro.calls[0][1] == pytest.approx(1.05 / 2 ** (4 / 12))


def load(tmp_path, body: str):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body))
    return config_mod.load(path)


class TestConfig:
    def test_defaults_are_todays_voice(self, tmp_path):
        cfg = load(tmp_path, "language: es\n")
        assert cfg.tts_voice == "ef_dora"
        assert cfg.tts_pitch_semitones == 0.0
        assert cfg.tts_brightness == 0.0

    def test_auto_still_follows_the_language(self, tmp_path):
        cfg = load(tmp_path, "language: en\ntts:\n  voice: auto\n")
        assert cfg.tts_voice == "af_heart"
        assert cfg.tts_pitch_semitones == 0.0

    def test_preset_fills_in_everything(self, tmp_path):
        cfg = load(tmp_path, "tts:\n  voice: auto\n  preset: cachorra\n")
        p = TTS_PRESETS["cachorra"]
        assert cfg.tts_voice == p["voice"]
        assert cfg.tts_pitch_semitones == p["pitch_semitones"]
        assert cfg.tts_brightness == p["brightness"]

    def test_explicit_keys_beat_the_preset(self, tmp_path):
        cfg = load(tmp_path, """
            tts:
              preset: cachorra
              voice: em_alex
              pitch_semitones: 2
            """)
        assert cfg.tts_voice == "em_alex"
        assert cfg.tts_pitch_semitones == 2.0
        assert cfg.tts_brightness == TTS_PRESETS["cachorra"]["brightness"]

    def test_a_blend_is_accepted(self, tmp_path):
        cfg = load(tmp_path, "tts:\n  voice: 'ef_dora:0.7,af_heart:0.3'\n")
        assert cfg.tts_voice == "ef_dora:0.7,af_heart:0.3"

    @pytest.mark.parametrize(
        "body",
        [
            "tts:\n  preset: lobo\n",
            "tts:\n  voice: 'ef_dora:0'\n",
            "tts:\n  pitch_semitones: 20\n",
            "tts:\n  pitch_semitones: mucho\n",
            "tts:\n  brightness: 2\n",
        ],
    )
    def test_bad_values_are_config_errors(self, tmp_path, body):
        with pytest.raises(config_mod.ConfigError):
            load(tmp_path, body)
