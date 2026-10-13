"""Smart Turn preprocessing and model-file handling. No model, no network.

The numerical match against transformers.WhisperFeatureExtractor is checked by
scripts/bench_smart_turn.py --parity, which needs transformers installed; these
tests pin the shape contract and the padding rule, which are what a silent
regression would break first.
"""

from __future__ import annotations

import numpy as np
import pytest

from wawi import turn

SR = 16000


def test_short_audio_is_padded_at_the_front():
    audio = np.ones(SR, dtype=np.float32)
    out = turn.fit_window(audio)
    assert out.size == 8 * SR
    assert np.all(out[: 7 * SR] == 0.0) and np.all(out[7 * SR :] == 1.0)


def test_long_audio_keeps_the_last_eight_seconds():
    audio = np.arange(10 * SR, dtype=np.float32)
    out = turn.fit_window(audio)
    assert out.size == 8 * SR
    assert out[-1] == audio[-1] and out[0] == audio[2 * SR]


def test_features_have_the_shape_the_model_expects():
    rng = np.random.default_rng(0)
    feats = turn.whisper_features((rng.standard_normal(3 * SR) * 0.1).astype(np.float32))
    assert feats.shape == (1, 80, 800)
    assert feats.dtype == np.float32
    # Whisper's dynamic-range clamp: nothing more than 8 log10 units below
    # the peak, after the (x + 4) / 4 rescale that is 2.0.
    assert feats.max() - feats.min() <= 2.0 + 1e-6


def test_mel_filters_match_whisper_geometry():
    assert turn._MEL_FILTERS.shape == (201, 80)
    assert np.all(turn._MEL_FILTERS >= 0)
    # Every filter covers at least one FFT bin -- none is empty.
    assert np.all(turn._MEL_FILTERS.sum(axis=0) > 0)


def test_model_ready_never_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("WAWI_SMART_TURN_MODEL", str(tmp_path / "absent.onnx"))
    assert turn.model_ready() is False
    present = tmp_path / "m.onnx"
    present.write_bytes(b"x")
    monkeypatch.setenv("WAWI_SMART_TURN_MODEL", str(present))
    assert turn.model_path() == str(present)


def test_loading_without_the_file_names_the_fix(monkeypatch):
    monkeypatch.setattr(turn, "model_path", lambda: None)
    with pytest.raises(turn.TurnModelError, match="download_model"):
        turn.SmartTurn().load()


def test_predict_before_load_is_an_error():
    with pytest.raises(turn.TurnModelError):
        turn.SmartTurn().predict(np.zeros(SR, dtype=np.float32))
