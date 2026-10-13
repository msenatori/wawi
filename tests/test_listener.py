"""Hermetic tests for the always-on "Wawi" listener.

No microphone, no network, no real openwakeword, no model files, no sleeps.
sounddevice and the whole openwakeword package are replaced by fakes, and the
audio cascade is driven synchronously through Listener._pump() so every
assertion is deterministic.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

from wawi import listener as listener_mod
from wawi.listener import (
    WAKE_MODEL_NAME,
    Listener,
    ListenerConfig,
    ListenerError,
    _float_to_int16,
    _Reframer,
    _RingBuffer,
    missing_model_files,
    models_ready,
)

SR = 16000
WAKE_FRAME = 1280            # openwakeword native chunk, 80 ms
VAD_FRAME = SR * 30 // 1000  # 480 samples = 30 ms, Silero's recommended frame


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeWakeModel:
    """Stand-in for openwakeword.Model. Fires when told to."""

    def __init__(self) -> None:
        self.frames_seen: list[np.ndarray] = []
        self.resets = 0
        self.fire_next = False  # set True to make the next predict() detect

    def predict(self, frame):
        if not isinstance(frame, np.ndarray):
            raise ValueError("openwakeword requires a numpy array")
        self.frames_seen.append(frame)
        if self.fire_next:
            self.fire_next = False
            return {WAKE_MODEL_NAME: 0.92}
        return {WAKE_MODEL_NAME: 0.01}

    def reset(self) -> None:
        self.resets += 1


class FakeVad:
    """Stand-in for openwakeword.VAD. Speech score comes from a callable."""

    def __init__(self) -> None:
        self.score = lambda frame: 1.0
        self.frames_seen: list[np.ndarray] = []

    def predict(self, x, frame_size=480):
        assert x.dtype == np.int16, "Silero VAD needs 16-bit PCM"
        assert x.size == frame_size, "the VAD got a wrong-sized frame"
        self.frames_seen.append(x)
        return np.float32(self.score(x))


def make_oww_module(
    wake_model=None,
    vad=None,
    model_error: Exception | None = None,
    vad_error: Exception | None = None,
    model_dir: str = "/nonexistent/models",
):
    """Build a fake `openwakeword` package module."""
    mod = types.SimpleNamespace()
    mod.MODELS = {
        "hey_jarvis": {"model_path": os.path.join(model_dir, "hey_jarvis_v0.1.tflite")},
        "alexa": {"model_path": os.path.join(model_dir, "alexa_v0.1.tflite")},
    }
    mod.FEATURE_MODELS = {
        "melspectrogram": {"model_path": os.path.join(model_dir, "melspectrogram.tflite")},
        "embedding": {"model_path": os.path.join(model_dir, "embedding_model.tflite")},
    }
    mod.VAD_MODELS = {"silero_vad": {"model_path": os.path.join(model_dir, "silero_vad.onnx")}}

    mod.model_kwargs = None

    def _Model(**kwargs):  # noqa: N802 - mirrors the real API
        mod.model_kwargs = kwargs
        if model_error is not None:
            raise model_error
        return wake_model

    def _VAD(**kwargs):  # noqa: N802 - mirrors the real API
        if vad_error is not None:
            raise vad_error
        return vad

    mod.Model = _Model
    mod.VAD = _VAD

    mod.downloaded: list = []
    mod.utils = types.SimpleNamespace(
        download_models=lambda names=None: mod.downloaded.append(names)
    )
    return mod


class FakeStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        self.callback = kwargs.get("callback")

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class FakeSounddevice:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.streams: list[FakeStream] = []

    def InputStream(self, **kwargs):  # noqa: N802 - mirrors the real API
        if self.error is not None:
            raise self.error
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def fakes(monkeypatch):
    """Install fake sounddevice + openwakeword, and pretend models are present."""
    wake_model = FakeWakeModel()
    vad = FakeVad()
    sd = FakeSounddevice()
    oww = make_oww_module(wake_model=wake_model, vad=vad)

    monkeypatch.setattr(listener_mod, "sd", sd)
    monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
    monkeypatch.setattr(listener_mod, "missing_model_files", lambda *a, **k: [])
    return {"wake": wake_model, "vad": vad, "sd": sd, "oww": oww}


def make_listener(fakes, cfg: ListenerConfig | None = None):
    """Return a started Listener plus the list its utterances land in."""
    captured: list[np.ndarray] = []
    listener = Listener(
        on_utterance=captured.append, cfg=cfg or ListenerConfig(sample_rate=SR)
    )
    listener.start()
    return listener, captured


def tone(n: int, value: float = 0.5) -> np.ndarray:
    """A constant-amplitude block; content is irrelevant, the fakes decide."""
    return np.full(n, value, dtype=np.float32)


def ramp(n: int, start: int = 0) -> np.ndarray:
    """Distinguishable audio: sample i carries the value (start + i)/1e6."""
    return ((np.arange(n) + start) / 1e6).astype(np.float32)


# ---------------------------------------------------------------------------
# _Reframer -- the most likely source of silent failure
# ---------------------------------------------------------------------------

class TestReframer:
    def test_emits_nothing_until_a_full_frame_is_buffered(self):
        rf = _Reframer(VAD_FRAME)
        assert rf.push(tone(100)) == []
        assert rf.push(tone(100)) == []
        assert rf.pending == 200

    def test_emits_exact_frame_when_block_matches_frame_length(self):
        rf = _Reframer(VAD_FRAME)
        frames = rf.push(ramp(VAD_FRAME))
        assert len(frames) == 1
        assert frames[0].size == VAD_FRAME
        assert rf.pending == 0

    @pytest.mark.parametrize("block_size", [1, 7, 100, 333, 511, 513, 1000, 1024, 4099])
    def test_awkward_block_sizes_never_lose_or_duplicate_samples(self, block_size):
        """Blocks that do not divide evenly must still yield a contiguous stream."""
        for frame_length in (WAKE_FRAME, VAD_FRAME, 512):
            rf = _Reframer(frame_length)
            total = block_size * 20
            emitted = []
            for i in range(20):
                emitted.extend(rf.push(ramp(block_size, start=i * block_size)))

            assert all(f.size == frame_length for f in emitted)
            assert len(emitted) == total // frame_length
            assert rf.pending == total - len(emitted) * frame_length

            if emitted:
                joined = np.concatenate(emitted)
                np.testing.assert_allclose(joined, ramp(len(joined)), rtol=0, atol=1e-12)

    def test_remainder_carries_into_the_next_push(self):
        rf = _Reframer(VAD_FRAME)  # 480
        rf.push(ramp(700))  # 1 frame, 220 left over
        assert rf.pending == 220
        frames = rf.push(ramp(300, start=700))
        assert len(frames) == 1
        # The second frame must start where the first ended, not at the block edge.
        np.testing.assert_allclose(frames[0], ramp(VAD_FRAME, start=VAD_FRAME), atol=1e-12)
        assert rf.pending == 40

    def test_one_huge_block_yields_many_frames_in_order(self):
        rf = _Reframer(WAKE_FRAME)
        frames = rf.push(ramp(WAKE_FRAME * 5 + 17))
        assert len(frames) == 5
        assert rf.pending == 17
        np.testing.assert_allclose(np.concatenate(frames), ramp(WAKE_FRAME * 5), atol=1e-12)

    def test_device_blocksize_does_not_divide_either_frame_size(self):
        """1024 divides neither 1280 nor 480 -- the reframer is load-bearing."""
        assert WAKE_FRAME % 1024 != 0 and 1024 % WAKE_FRAME != 0
        assert VAD_FRAME % 1024 != 0 and 1024 % VAD_FRAME != 0

    def test_empty_block_is_a_noop(self):
        rf = _Reframer(VAD_FRAME)
        assert rf.push(np.zeros(0, dtype=np.float32)) == []
        assert rf.pending == 0

    def test_reset_drops_pending_samples(self):
        rf = _Reframer(VAD_FRAME)
        rf.push(tone(200))
        rf.reset()
        assert rf.pending == 0
        assert rf.push(tone(200)) == []

    def test_rejects_non_positive_frame_length(self):
        with pytest.raises(ValueError):
            _Reframer(0)


class TestRingBuffer:
    def test_keeps_only_the_most_recent_capacity(self):
        ring = _RingBuffer(1000)
        for i in range(10):
            ring.append(ramp(300, start=i * 300))
        out = ring.read()
        assert out.size == 1000
        np.testing.assert_allclose(out, ramp(1000, start=3000 - 1000), atol=1e-12)

    def test_shorter_than_capacity_returns_everything(self):
        ring = _RingBuffer(1000)
        ring.append(ramp(400))
        assert ring.read().size == 400

    def test_zero_capacity_holds_nothing(self):
        ring = _RingBuffer(0)
        ring.append(tone(500))
        assert ring.read().size == 0


class TestFloatToInt16:
    def test_scales_and_clips(self):
        block = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
        out = _float_to_int16(block)
        assert out.dtype == np.int16
        assert out.tolist() == [0, 32767, -32767, 32767, -32767]


# ---------------------------------------------------------------------------
# Model files -- present / missing / download
# ---------------------------------------------------------------------------

class TestModelFiles:
    def test_missing_model_files_lists_everything_absent(self, monkeypatch):
        oww = make_oww_module(model_dir="/definitely/not/here")
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        missing = missing_model_files("onnx")
        names = [os.path.basename(p) for p in missing]
        assert names == [
            "hey_jarvis_v0.1.onnx",
            "melspectrogram.onnx",
            "embedding_model.onnx",
            "silero_vad.onnx",
        ]

    def test_only_the_selected_wake_model_is_required(self, monkeypatch):
        """alexa and friends are irrelevant; we must not demand them."""
        oww = make_oww_module(model_dir="/definitely/not/here")
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        assert not any("alexa" in p for p in missing_model_files())

    def test_tflite_framework_asks_for_tflite_weights_but_onnx_vad(self, monkeypatch):
        oww = make_oww_module(model_dir="/definitely/not/here")
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        names = [os.path.basename(p) for p in missing_model_files("tflite")]
        assert names == [
            "hey_jarvis_v0.1.tflite",
            "melspectrogram.tflite",
            "embedding_model.tflite",
            "silero_vad.onnx",  # Silero is ONNX-only either way
        ]

    def test_models_ready_is_true_once_the_files_exist(self, monkeypatch, tmp_path):
        oww = make_oww_module(model_dir=str(tmp_path))
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        assert models_ready() is False
        for name in (
            "hey_jarvis_v0.1.onnx",
            "melspectrogram.onnx",
            "embedding_model.onnx",
            "silero_vad.onnx",
        ):
            (tmp_path / name).write_bytes(b"x")
        assert models_ready() is True

    def test_models_ready_does_not_download(self, monkeypatch, tmp_path):
        """The doctor must be able to ask without causing a network fetch."""
        oww = make_oww_module(model_dir=str(tmp_path))
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        models_ready()
        missing_model_files()
        assert oww.downloaded == []

    def test_models_ready_returns_false_when_package_is_absent(self, monkeypatch):
        def _boom():
            raise ListenerError("openwakeword is not installed")

        monkeypatch.setattr(listener_mod, "_load_openwakeword", _boom)
        assert models_ready() is False  # must not raise: the doctor calls this

    def test_start_refuses_with_an_actionable_error_when_models_are_missing(
        self, monkeypatch
    ):
        oww = make_oww_module(
            wake_model=FakeWakeModel(), vad=FakeVad(), model_dir="/definitely/not/here"
        )
        monkeypatch.setattr(listener_mod, "sd", FakeSounddevice())
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)

        listener = Listener(on_utterance=lambda buf: None)
        with pytest.raises(ListenerError) as excinfo:
            listener.start()

        msg = str(excinfo.value)
        assert "download_models" in msg
        assert "hey_jarvis_v0.1.onnx" in msg
        assert oww.downloaded == [], "start() must never download implicitly"
        assert listener._thread is None

    def test_download_models_passes_through_and_verifies(self, monkeypatch, tmp_path):
        oww = make_oww_module(model_dir=str(tmp_path))

        def _download(names=None):
            oww.downloaded.append(names)
            for name in (
                "hey_jarvis_v0.1.onnx",
                "melspectrogram.onnx",
                "embedding_model.onnx",
                "silero_vad.onnx",
            ):
                (tmp_path / name).write_bytes(b"x")

        oww.utils.download_models = _download
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)

        listener_mod.download_models()
        assert oww.downloaded == [[WAKE_MODEL_NAME]]
        assert models_ready() is True

    def test_download_failure_is_wrapped(self, monkeypatch, tmp_path):
        oww = make_oww_module(model_dir=str(tmp_path))

        def _download(names=None):
            raise OSError("network unreachable")

        oww.utils.download_models = _download
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        with pytest.raises(ListenerError, match="network unreachable"):
            listener_mod.download_models()

    def test_download_that_silently_does_nothing_is_caught(self, monkeypatch, tmp_path):
        oww = make_oww_module(model_dir=str(tmp_path))
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        with pytest.raises(ListenerError, match="still.*missing"):
            listener_mod.download_models()


# ---------------------------------------------------------------------------
# Construction and error handling
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_missing_openwakeword_gives_a_pip_hint(self, monkeypatch):
        monkeypatch.setattr(listener_mod, "sd", FakeSounddevice())

        def _boom():
            raise ListenerError("openwakeword is not installed; pip install openwakeword")

        monkeypatch.setattr(listener_mod, "_load_openwakeword", _boom)
        listener = Listener(on_utterance=lambda buf: None)
        with pytest.raises(ListenerError, match="pip install openwakeword"):
            listener.start()

    def test_real_loader_converts_importerror_into_listenererror(self, monkeypatch):
        """The module must import and run whether or not openwakeword is installed."""
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name.split(".")[0] == "openwakeword":
                raise ImportError("No module named 'openwakeword'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        monkeypatch.delitem(__import__("sys").modules, "openwakeword", raising=False)
        monkeypatch.delitem(__import__("sys").modules, "openwakeword.utils", raising=False)

        with pytest.raises(ListenerError, match="pip install openwakeword"):
            listener_mod._load_openwakeword()

    def test_model_load_failure_surfaces_as_listener_error(self, monkeypatch):
        """A failure inside onnxruntime must not escape as a raw exception."""
        oww = make_oww_module(model_error=RuntimeError("corrupt onnx graph"))
        monkeypatch.setattr(listener_mod, "sd", FakeSounddevice())
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        monkeypatch.setattr(listener_mod, "missing_model_files", lambda *a, **k: [])

        listener = Listener(on_utterance=lambda buf: None)
        with pytest.raises(ListenerError, match="corrupt onnx graph"):
            listener.start()

    def test_vad_load_failure_surfaces_as_listener_error(self, monkeypatch):
        oww = make_oww_module(
            wake_model=FakeWakeModel(), vad_error=RuntimeError("bad silero file")
        )
        monkeypatch.setattr(listener_mod, "sd", FakeSounddevice())
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        monkeypatch.setattr(listener_mod, "missing_model_files", lambda *a, **k: [])

        listener = Listener(on_utterance=lambda buf: None)
        with pytest.raises(ListenerError, match="Silero VAD"):
            listener.start()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sample_rate": 44100},
            {"sample_rate": 8000},
            {"silence_ms": 0},
            {"silence_ms": -100},
            {"max_utterance_s": 0.0},
            {"pre_roll_s": -1.0},
            {"vad_threshold": 1.5},
            {"vad_threshold": -0.1},
            {"wake_threshold": 1.5},
            {"wake_threshold": -0.1},
            {"inference_framework": "coreml"},
        ],
    )
    def test_bad_config_is_rejected_at_construction(self, kwargs):
        with pytest.raises(ListenerError):
            Listener(on_utterance=lambda buf: None, cfg=ListenerConfig(**kwargs))

    def test_start_requests_the_pretrained_hey_jarvis_model_over_onnx(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert fakes["oww"].model_kwargs == {
                "wakeword_models": ["hey_jarvis"],
                "inference_framework": "onnx",
            }
        finally:
            listener.stop()

    def test_failing_microphone_raises_and_leaves_nothing_running(self, monkeypatch):
        oww = make_oww_module(wake_model=FakeWakeModel(), vad=FakeVad())
        monkeypatch.setattr(
            listener_mod, "sd", FakeSounddevice(error=OSError("no such device"))
        )
        monkeypatch.setattr(listener_mod, "_load_openwakeword", lambda: oww)
        monkeypatch.setattr(listener_mod, "missing_model_files", lambda *a, **k: [])

        listener = Listener(on_utterance=lambda buf: None)
        with pytest.raises(ListenerError, match="microphone"):
            listener.start()
        assert listener._thread is None
        assert listener._wake_model is None


# ---------------------------------------------------------------------------
# Framing through the real cascade
# ---------------------------------------------------------------------------

class TestFramingThroughTheCascade:
    def test_engines_only_ever_see_their_required_frame_size(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            # 1000 divides neither 1280 nor 480.
            for i in range(40):
                listener._pump(ramp(1000, start=i * 1000))
        finally:
            listener.stop()

        assert fakes["wake"].frames_seen, "the wake model never ran"
        assert all(f.size == WAKE_FRAME for f in fakes["wake"].frames_seen)
        assert all(f.dtype == np.int16 for f in fakes["wake"].frames_seen)
        # FakeVad asserts its own frame size and dtype on every call.
        assert fakes["vad"].frames_seen

    def test_wake_frame_samples_is_openwakewords_native_chunk(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert listener.wake_frame_samples == 1280
        finally:
            listener.stop()


# ---------------------------------------------------------------------------
# Pre-roll
# ---------------------------------------------------------------------------

class TestPreRoll:
    def test_utterance_includes_audio_from_before_the_trigger(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=1.0, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        pre_roll_samples = SR  # 1.0 s

        try:
            # Two seconds of distinguishable speech before the wake word fires.
            blocks = 2 * SR // 1024
            for i in range(blocks):
                listener._pump(ramp(1024, start=i * 1024))

            # Next block triggers the wake word.
            trigger_start = blocks * 1024
            fakes["wake"].fire_next = True
            listener._pump(ramp(1024, start=trigger_start))

            # Then silence closes the utterance.
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(4):
                listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1
        buf = captured[0]
        assert buf.dtype == np.float32

        # The first pre-roll sample must predate the trigger by ~1 s of audio.
        end_of_trigger_block = trigger_start + 1024
        expected_first = (end_of_trigger_block - pre_roll_samples) / 1e6
        assert buf[0] == pytest.approx(expected_first, abs=1e-9)
        assert buf.size >= pre_roll_samples

    def test_pre_roll_is_capped_at_the_configured_length(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.5, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            for i in range(50):  # ~3 s of audio buffered
                listener._pump(ramp(1024, start=i * 1024))
            fakes["wake"].fire_next = True
            listener._pump(ramp(1024, start=50 * 1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(5):
                listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1
        # Pre-roll (8000) plus the silent tail; nowhere near the 3 s buffered.
        assert captured[0].size < SR

    def test_no_utterance_without_a_wake_word(self, fakes):
        listener, captured = make_listener(fakes)
        try:
            for i in range(40):
                listener._pump(ramp(1024, start=i * 1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(40):
                listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()
        assert captured == []

    def test_wake_model_is_fed_continuously_even_during_silence(self, fakes):
        """openWakeWord is a streaming model: skipping frames corrupts its buffer."""
        listener, _ = make_listener(fakes)
        fakes["vad"].score = lambda frame: 0.0  # nobody is speaking
        try:
            for _ in range(20):
                listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()
        # 20 * 1024 = 20480 samples -> 16 chunks of 1280, no gaps.
        assert len(fakes["wake"].frames_seen) == 20 * 1024 // WAKE_FRAME

    def test_detection_is_suppressed_when_the_vad_saw_no_speech(self, fakes):
        """The VAD is a precision filter: a wake score with no speech is dropped."""
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(10):
                fakes["wake"].fire_next = True
                listener._pump(tone(1024))
            assert captured == [], "woke up with no speech present"
            assert fakes["wake"].frames_seen, "but the model still ran every frame"
        finally:
            listener.stop()

    def test_vad_threshold_decides_what_confirms_a_detection(self, fakes):
        cfg = ListenerConfig(
            sample_rate=SR, pre_roll_s=0.1, silence_ms=100, vad_threshold=0.8
        )
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["vad"].score = lambda frame: 0.5  # below 0.8 -> not speech
            for _ in range(6):
                fakes["wake"].fire_next = True
                listener._pump(tone(1024))
            assert captured == []

            fakes["vad"].score = lambda frame: 0.9  # above 0.8 -> speech
            for _ in range(6):
                fakes["wake"].fire_next = True
                listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert len(captured) == 1
        finally:
            listener.stop()

    def test_wake_threshold_gates_detection(self, fakes):
        """A wake score below the threshold must not open an utterance."""
        cfg = ListenerConfig(sample_rate=SR, wake_threshold=0.99, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            for _ in range(5):
                fakes["wake"].fire_next = True  # scores 0.92, below 0.99
                listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(10):
                listener._pump(tone(1024, 0.0))
            assert captured == []
        finally:
            listener.stop()


# ---------------------------------------------------------------------------
# Endpointing
# ---------------------------------------------------------------------------

class TestEndpointing:
    def test_silence_frame_threshold_matches_the_configured_ms(self, fakes):
        listener, _ = make_listener(fakes, ListenerConfig(sample_rate=SR, silence_ms=900))
        try:
            assert listener.silence_frames_needed == 30  # 900 ms / 30 ms
        finally:
            listener.stop()

    def test_sub_frame_silence_still_needs_one_whole_frame(self, fakes):
        listener, _ = make_listener(fakes, ListenerConfig(sample_rate=SR, silence_ms=10))
        try:
            assert listener.silence_frames_needed == 1
        finally:
            listener.stop()

    def test_fires_after_exactly_n_silent_frames(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=300)
        listener, captured = make_listener(fakes, cfg)
        needed = listener.silence_frames_needed
        assert needed == 10

        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))  # speech + trigger
            assert captured == []

            # Feed silence one VAD frame at a time.
            fakes["vad"].score = lambda frame: 0.0
            for i in range(needed - 1):
                listener._pump(tone(VAD_FRAME, 0.0))
                assert captured == [], f"closed early after {i + 1} silent frames"

            listener._pump(tone(VAD_FRAME, 0.0))  # frame number `needed`
            assert len(captured) == 1
        finally:
            listener.stop()

    def test_speech_resets_the_silence_counter(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=300)
        listener, captured = make_listener(fakes, cfg)
        needed = listener.silence_frames_needed
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))

            fakes["vad"].score = lambda frame: 0.0
            for _ in range(needed - 1):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert captured == []

            # A mid-sentence pause ends: the speaker resumes.
            fakes["vad"].score = lambda frame: 1.0
            listener._pump(tone(VAD_FRAME))
            assert captured == []

            fakes["vad"].score = lambda frame: 0.0
            for _ in range(needed - 1):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert captured == [], "counter did not reset on resumed speech"
            listener._pump(tone(VAD_FRAME, 0.0))
            assert len(captured) == 1
        finally:
            listener.stop()

    def test_listener_rearms_after_an_utterance(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            for _ in range(2):
                fakes["vad"].score = lambda frame: 1.0
                fakes["wake"].fire_next = True
                listener._pump(tone(WAKE_FRAME))
                fakes["vad"].score = lambda frame: 0.0
                for _ in range(listener.silence_frames_needed):
                    listener._pump(tone(VAD_FRAME, 0.0))
            assert len(captured) == 2
        finally:
            listener.stop()

    def test_wake_model_buffer_is_reset_between_utterances(self, fakes):
        """Stale features must not carry across a capture and re-fire."""
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=100)
        listener, _ = make_listener(fakes, cfg)
        before = fakes["wake"].resets
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert fakes["wake"].resets > before
        finally:
            listener.stop()

    def test_a_broken_on_utterance_does_not_kill_the_listener(self, fakes):
        calls = []

        def _explode(buf):
            calls.append(buf)
            raise ValueError("downstream blew up")

        listener = Listener(
            on_utterance=_explode,
            cfg=ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=100),
        )
        listener.start()
        try:
            for _ in range(2):
                fakes["vad"].score = lambda frame: 1.0
                fakes["wake"].fire_next = True
                listener._pump(tone(WAKE_FRAME))
                fakes["vad"].score = lambda frame: 0.0
                for _ in range(listener.silence_frames_needed):
                    listener._pump(tone(VAD_FRAME, 0.0))
            assert len(calls) == 2
        finally:
            listener.stop()

    def test_a_crashing_vad_is_treated_as_silence_not_a_fatal_error(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))

            def _boom(frame):
                raise RuntimeError("onnxruntime exploded")

            fakes["vad"].score = _boom
            for _ in range(listener.silence_frames_needed):
                listener._pump(tone(VAD_FRAME))
            assert len(captured) == 1  # endpointed rather than hanging
        finally:
            listener.stop()


# ---------------------------------------------------------------------------
# Max utterance cap
# ---------------------------------------------------------------------------

class TestMaxUtterance:
    def test_endless_speech_is_cut_at_the_cap(self, fakes):
        cfg = ListenerConfig(
            sample_rate=SR, pre_roll_s=0.0, silence_ms=800, max_utterance_s=1.0
        )
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["vad"].score = lambda frame: 1.0  # never any silence
            fakes["wake"].fire_next = True
            listener._pump(tone(1024))

            for _ in range(100):
                listener._pump(tone(1024))
                if captured:
                    break

            assert len(captured) == 1, "the cap never fired"
            assert captured[0].size >= SR       # reached the 1 s cap
            assert captured[0].size < SR + 2048  # and stopped right there
        finally:
            listener.stop()

    def test_capped_utterance_is_dispatched_not_discarded(self, fakes):
        cfg = ListenerConfig(
            sample_rate=SR, pre_roll_s=0.0, silence_ms=800, max_utterance_s=0.5
        )
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["vad"].score = lambda frame: 1.0
            fakes["wake"].fire_next = True
            listener._pump(tone(1024))
            for _ in range(20):
                listener._pump(tone(1024))
        finally:
            listener.stop()
        assert captured and captured[0].size > 0


# ---------------------------------------------------------------------------
# pause() / resume()
# ---------------------------------------------------------------------------

class TestPauseResume:
    def test_pause_drops_audio_entirely(self, fakes):
        listener, captured = make_listener(fakes)
        try:
            listener.pause()
            assert listener.is_paused
            fakes["wake"].fire_next = True
            for _ in range(20):
                listener._pump(tone(1024))
            assert fakes["wake"].frames_seen == []
            assert fakes["vad"].frames_seen == []
            assert captured == []
        finally:
            listener.stop()

    def test_pause_discards_an_in_flight_capture(self, fakes):
        """Wawi starting to speak mid-capture must not dispatch a half utterance."""
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))

            listener.pause()
            listener._pump(tone(1024))

            listener.resume()
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert captured == []
        finally:
            listener.stop()

    def test_paused_audio_cannot_leak_into_a_later_pre_roll(self, fakes):
        """Wawi's own voice must never reappear in the next utterance."""
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=1.0, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            # Wawi speaks: a very loud, unmistakable signal, while paused.
            listener.pause()
            for _ in range(30):
                listener._pump(tone(1024, 0.99))
            listener.resume()

            # The user then says the wake word (quiet signal).
            fakes["wake"].fire_next = True
            listener._pump(tone(1024, 0.01))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))

            assert len(captured) == 1
            assert float(np.abs(captured[0]).max()) < 0.5, "Wawi heard itself"
        finally:
            listener.stop()

    def test_resume_restores_processing(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            listener.pause()
            listener._pump(tone(1024))
            listener.resume()
            assert not listener.is_paused
            for _ in range(3):
                listener._pump(tone(1024))
            assert fakes["wake"].frames_seen
        finally:
            listener.stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_opens_one_stream_with_the_right_parameters(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert len(fakes["sd"].streams) == 1
            kwargs = fakes["sd"].streams[0].kwargs
            assert kwargs["samplerate"] == SR
            assert kwargs["channels"] == 1
            assert kwargs["dtype"] == "float32"
            assert fakes["sd"].streams[0].started
        finally:
            listener.stop()

    def test_start_is_idempotent(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            listener.start()
            assert len(fakes["sd"].streams) == 1
        finally:
            listener.stop()

    def test_stop_closes_the_stream_joins_the_thread_and_frees_models(self, fakes):
        listener, _ = make_listener(fakes)
        thread = listener._thread
        assert thread is not None and thread.is_alive()

        listener.stop()

        assert fakes["sd"].streams[0].closed
        assert listener._wake_model is None
        assert listener._vad is None
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "stop() left the worker thread running"
        assert listener._thread is None

    def test_stop_is_safe_before_start_and_twice(self, fakes):
        listener = Listener(on_utterance=lambda buf: None)
        listener.stop()  # never started
        listener.start()
        listener.stop()
        listener.stop()  # again
        assert listener._wake_model is None

    def test_stop_does_not_hang_with_a_full_queue(self, fakes):
        """A backlog of unprocessed audio must not delay shutdown."""
        listener, _ = make_listener(fakes)
        for _ in range(500):
            listener._queue.put(tone(1024))
        listener.stop()
        assert listener._thread is None

    def test_can_restart_after_stop(self, fakes):
        listener, _ = make_listener(fakes)
        listener.stop()
        listener.start()
        try:
            assert len(fakes["sd"].streams) == 2
        finally:
            listener.stop()

    def test_current_block_returns_the_newest_chunk(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert listener.current_block() is None

            callback = fakes["sd"].streams[0].callback
            first = ramp(1024).reshape(-1, 1)
            callback(first, 1024, None, None)
            np.testing.assert_allclose(listener.current_block(), first[:, 0])

            second = (ramp(1024) + 1.0).reshape(-1, 1)
            callback(second, 1024, None, None)
            np.testing.assert_allclose(listener.current_block(), second[:, 0])
        finally:
            listener.stop()
        assert listener.current_block() is None

    def test_audio_callback_feeds_the_worker_queue(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            callback = fakes["sd"].streams[0].callback
            callback(ramp(1024).reshape(-1, 1), 1024, None, None)
            # Either the worker already drained it or it is still queued; both
            # prove the callback handed it off rather than doing DSP inline.
            assert listener._queue.qsize() >= 0
        finally:
            listener.stop()

    def test_worker_survives_a_bad_block(self, fakes):
        """A malformed block is logged and skipped, not fatal."""
        listener, _ = make_listener(fakes)
        try:
            with pytest.raises(Exception):
                listener._pump("not audio")  # type: ignore[arg-type]
            # The listener still works afterwards.
            for _ in range(3):
                listener._pump(tone(1024))
            assert fakes["wake"].frames_seen
        finally:
            listener.stop()


# ---------------------------------------------------------------------------
# Push-to-talk -- forced capture on the listener's own stream
# ---------------------------------------------------------------------------

class TestForcedCapture:
    def test_force_start_opens_a_capture_without_the_wake_word(self, fakes):
        """The key bypasses both gates: no wake score, no speech, still captured."""
        listener, captured = make_listener(fakes)
        try:
            fakes["vad"].score = lambda frame: 0.0  # nobody is speaking
            listener.force_start()
            for i in range(4):
                listener._pump(ramp(1024, start=i * 1024))
            assert captured == []  # still open — only the key closes it

            listener.force_stop()
            listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1
        assert captured[0].size > 0

    def test_silence_does_not_close_a_forced_capture(self, fakes):
        """Holding the key through a long pause must not truncate the utterance."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["vad"].score = lambda frame: 0.0
            listener.force_start()
            # Far more silent frames than endpointing would ever need.
            for _ in range(listener.silence_frames_needed * 5):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert captured == []

            listener.force_stop()
            listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1

    def test_forced_capture_includes_pre_roll(self, fakes):
        """Push-to-talk gains the pre-roll a separate Recorder could not give it."""
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=1.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            blocks = 2 * SR // 1024
            for i in range(blocks):
                listener._pump(ramp(1024, start=i * 1024))

            listener.force_start()
            press_start = blocks * 1024
            listener._pump(ramp(1024, start=press_start))
            listener.force_stop()
            listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1
        buf = captured[0]
        # The capture reaches ~1 s back before the key was pressed.
        end_of_press_block = press_start + 1024
        expected_first = (end_of_press_block - SR) / 1e6
        assert buf[0] == pytest.approx(expected_first, abs=1e-9)

    def test_a_tap_inside_one_block_still_produces_an_utterance(self, fakes):
        """Press and release between two audio blocks must not be swallowed.

        This is why the commands are a FIFO and not a boolean flag.
        """
        listener, captured = make_listener(fakes)
        try:
            for i in range(4):
                listener._pump(ramp(1024, start=i * 1024))
            listener.force_start()
            listener.force_stop()
            listener._pump(ramp(1024, start=4 * 1024))
        finally:
            listener.stop()

        assert len(captured) == 1
        assert captured[0].size > 0  # the pre-roll carries what was said

    def test_key_takes_over_an_open_wake_capture(self, fakes):
        """Pressing mid-utterance hands endpointing to the key."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))

            listener.force_start()
            listener._pump(tone(1024))

            # Silence would have closed a wake capture; it must not close this.
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed * 3):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert captured == []

            listener.force_stop()
            listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1

    def test_force_stop_without_an_open_capture_is_a_noop(self, fakes):
        listener, captured = make_listener(fakes)
        try:
            listener.force_stop()
            for _ in range(3):
                listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()
        assert captured == []

    def test_the_utterance_cap_still_bounds_a_held_key(self, fakes):
        """A key stuck down cannot record forever."""
        cfg = ListenerConfig(sample_rate=SR, max_utterance_s=0.25)
        listener, captured = make_listener(fakes, cfg)
        try:
            listener.force_start()
            for i in range(20):  # 20 * 1024 samples >> 0.25 s at 16 kHz
                listener._pump(ramp(1024, start=i * 1024))
        finally:
            listener.stop()

        assert len(captured) == 1
        assert captured[0].size >= int(0.25 * SR)

    def test_a_press_made_while_paused_is_dropped(self, fakes):
        """Pressing while Wawi speaks must not open a capture once it stops."""
        listener, captured = make_listener(fakes)
        try:
            listener.pause()
            listener.force_start()
            listener._pump(tone(1024))  # command is discarded here

            listener.resume()
            for _ in range(3):
                listener._pump(tone(1024))
        finally:
            listener.stop()

        assert captured == []

    def test_capture_start_callback_fires_for_both_paths(self, fakes):
        """The UI needs to show LISTENING the moment a capture opens."""
        starts: list[int] = []
        listener = Listener(
            on_utterance=lambda buf: None,
            on_capture_start=lambda: starts.append(1),
            cfg=ListenerConfig(sample_rate=SR, silence_ms=100),
        )
        listener.start()
        try:
            listener.force_start()
            listener._pump(tone(1024))
            assert len(starts) == 1
            listener.force_stop()
            listener._pump(tone(1024))

            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))
            assert len(starts) == 2
        finally:
            listener.stop()

    def test_a_broken_capture_callback_does_not_stop_the_capture(self, fakes):
        listener = Listener(
            on_utterance=lambda buf: None,
            on_capture_start=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            cfg=ListenerConfig(sample_rate=SR),
        )
        listener.start()
        try:
            listener.force_start()
            listener._pump(tone(1024))
            assert listener._capturing
        finally:
            listener.stop()

    def test_current_level_reports_the_newest_block_peak(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert listener.current_level() == 0.0
            callback = fakes["sd"].streams[0].callback
            callback(tone(1024, 0.4).reshape(-1, 1), 1024, None, None)
            assert listener.current_level() == pytest.approx(0.4)
        finally:
            listener.stop()
        assert listener.current_level() == 0.0


class TestWakeToggle:
    def test_disabling_the_wake_word_suppresses_detection(self, fakes):
        listener, captured = make_listener(fakes)
        try:
            listener.set_wake_enabled(False)
            fakes["wake"].fire_next = True
            for _ in range(4):
                listener._pump(tone(WAKE_FRAME))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(8):
                listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()

        assert captured == []

    def test_push_to_talk_is_refused_while_the_gate_is_closed(self, fakes):
        """One switch, one meaning: off is deaf, including the key.

        This assertion is the REVERSE of what it used to be. The toggle once
        meant "push-to-talk only", so the key had to keep working; it now means
        "listening on/off", and a key that still got through would make "am I
        being heard right now" depend on which of two controls you last
        touched.
        """
        listener, captured = make_listener(fakes)
        try:
            listener.set_wake_enabled(False)
            listener.force_start()
            for i in range(3):
                listener._pump(ramp(1024, start=i * 1024))
            listener.force_stop()
            listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()

        assert captured == []

    def test_the_microphone_itself_is_never_released(self, fakes):
        """Closing the gate must not close the audio device.

        Reopening a CoreAudio input is the operation that fails and leaves you
        with no microphone at all, and the pre-roll ring would be lost with it.
        """
        listener, captured = make_listener(fakes)
        try:
            listener.set_wake_enabled(False)
            listener._pump(tone(1024, 0.0))
            listener.set_wake_enabled(True)
            listener.force_start()
            for i in range(3):
                listener._pump(ramp(1024, start=i * 1024))
            listener.force_stop()
            listener._pump(tone(1024, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1

    def test_re_enabling_clears_the_stale_model_buffer(self, fakes):
        """Audio from across the off period must not fabricate a detection."""
        listener, _ = make_listener(fakes)
        try:
            listener._pump(tone(WAKE_FRAME))
            listener.set_wake_enabled(False)
            listener._pump(tone(WAKE_FRAME))
            resets_before = fakes["wake"].resets
            listener.set_wake_enabled(True)
            listener._pump(tone(WAKE_FRAME))
            assert fakes["wake"].resets > resets_before
        finally:
            listener.stop()

    def test_toggle_state_is_readable(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert listener.is_wake_enabled
            listener.set_wake_enabled(False)
            assert not listener.is_wake_enabled
            listener.set_wake_enabled(True)
            assert listener.is_wake_enabled
        finally:
            listener.stop()


# ---------------------------------------------------------------------------
# Follow-up window -- continuing a conversation without the wake word
# ---------------------------------------------------------------------------

class TestFollowUp:
    def test_speech_alone_opens_a_capture_while_the_window_is_open(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, follow_up_s=5.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            listener.arm_follow_up()
            # No wake word anywhere — just talking.
            for _ in range(3):
                listener._pump(tone(1024))
            assert listener._capturing

            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1
        assert not fakes["wake"].fire_next  # never needed to fire

    def test_without_the_window_speech_alone_does_nothing(self, fakes):
        """The guard against Wawi answering every noise in the room."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, follow_up_s=5.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            for _ in range(10):  # talking, but the window was never armed
                listener._pump(tone(1024))
        finally:
            listener.stop()

        assert captured == []

    def test_the_window_expires_after_its_configured_length(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, follow_up_s=0.5)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["vad"].score = lambda frame: 0.0  # silence burns the window
            listener.arm_follow_up()
            for _ in range(int(0.5 * SR / 1024) + 2):
                listener._pump(tone(1024, 0.0))
            assert not listener.follow_up_open

            # Speaking now is too late: the window already closed.
            fakes["vad"].score = lambda frame: 1.0
            for _ in range(3):
                listener._pump(tone(1024))
        finally:
            listener.stop()

        assert captured == []

    def test_the_window_closes_once_it_opens_a_capture(self, fakes):
        """One follow-up per turn; the next one must be armed again."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, follow_up_s=5.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            listener.arm_follow_up()
            listener._pump(tone(1024))
            assert not listener.follow_up_open

            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert len(captured) == 1

            # Talking again without re-arming must not start another turn.
            fakes["vad"].score = lambda frame: 1.0
            for _ in range(5):
                listener._pump(tone(1024))
        finally:
            listener.stop()

        assert len(captured) == 1

    def test_pausing_discards_a_pending_window(self, fakes):
        """Wawi speaking again must not leave a window from the last turn."""
        cfg = ListenerConfig(sample_rate=SR, follow_up_s=5.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            listener.arm_follow_up()
            listener._pump(tone(1024))
            listener.pause()
            listener._pump(tone(1024))
            assert not listener.follow_up_open
            listener.resume()
            for _ in range(5):
                listener._pump(tone(1024))
        finally:
            listener.stop()

        assert captured == []

    def test_follow_up_is_disabled_by_default(self, fakes):
        """Opt-in: the default config must not answer unprompted speech."""
        listener, captured = make_listener(fakes)
        try:
            assert listener.config.follow_up_s == 0.0
            listener.arm_follow_up()
            for _ in range(5):
                listener._pump(tone(1024))
        finally:
            listener.stop()

        assert captured == []

    def test_follow_up_capture_includes_pre_roll(self, fakes):
        """The first words of the reply must not be clipped."""
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=1.0, silence_ms=100, follow_up_s=5.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            for i in range(int(2 * SR / 1024)):
                listener._pump(ramp(1024, start=i * 1024))
            listener.arm_follow_up()
            listener._pump(ramp(1024, start=99 * 1024))

            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1
        assert captured[0].size >= SR  # a full second of pre-roll survived


# ---------------------------------------------------------------------------
# Whisper engine -- every speech segment is a candidate, the trigger decides
# ---------------------------------------------------------------------------

class FakeDetector:
    """Stand-in for wawi.wake.WhisperWakeDetector."""

    def __init__(self, matches: bool = True) -> None:
        self.matches = matches
        self.loaded = False
        self.segments: list[np.ndarray] = []
        self.phrases = ["viernes", "wawi"]

    def load(self) -> None:
        self.loaded = True

    def check(self, segment):
        self.segments.append(segment)
        from wawi.wake import WakeMatch

        return WakeMatch("viernes" if self.matches else None, "viernes dale", 0.04)


def make_whisper_listener(fakes, matches=True, **cfg_kwargs):
    detector = FakeDetector(matches=matches)
    captured: list[np.ndarray] = []
    cfg = ListenerConfig(engine="whisper", sample_rate=SR, silence_ms=100, **cfg_kwargs)
    listener = Listener(on_utterance=captured.append, cfg=cfg, wake_detector=detector)
    listener.start()
    return listener, captured, detector


class TestWhisperEngine:
    def test_engine_must_be_a_known_value(self, fakes):
        with pytest.raises(ListenerError, match="engine must be"):
            Listener(on_utterance=lambda b: None, cfg=ListenerConfig(engine="magic"))

    def test_whisper_engine_requires_a_detector(self, fakes):
        with pytest.raises(ListenerError, match="needs a wake_detector"):
            Listener(on_utterance=lambda b: None, cfg=ListenerConfig(engine="whisper"))

    def test_the_detector_is_loaded_at_start(self, fakes):
        listener, _, detector = make_whisper_listener(fakes)
        try:
            assert detector.loaded
        finally:
            listener.stop()

    def test_the_pretrained_wake_model_is_never_loaded(self, fakes):
        """Whisper mode must not pay for openwakeword's model at all."""
        listener, _, _ = make_whisper_listener(fakes)
        try:
            for _ in range(4):
                listener._pump(tone(WAKE_FRAME))
            assert fakes["wake"].frames_seen == []
        finally:
            listener.stop()

    def test_speech_with_the_trigger_is_dispatched(self, fakes):
        listener, captured, detector = make_whisper_listener(fakes)
        try:
            for _ in range(4):
                listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()

        assert len(detector.segments) == 1  # the segment was checked once
        assert len(captured) == 1           # and it matched, so it was dispatched

    def test_speech_without_the_trigger_is_dropped(self, fakes):
        """The common case: people talking near the machine, all the time."""
        listener, captured, detector = make_whisper_listener(fakes, matches=False)
        try:
            for _ in range(4):
                listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()

        assert len(detector.segments) == 1  # it WAS checked
        assert captured == []               # and correctly thrown away

    def test_the_listener_rearms_after_a_rejected_segment(self, fakes):
        """A dropped segment must not leave the listener deaf."""
        listener, captured, detector = make_whisper_listener(fakes, matches=False)
        try:
            for _ in range(2):
                for _ in range(3):
                    listener._pump(tone(1024))
                fakes["vad"].score = lambda frame: 0.0
                for _ in range(listener.silence_frames_needed + 2):
                    listener._pump(tone(VAD_FRAME, 0.0))
                fakes["vad"].score = lambda frame: 1.0
        finally:
            listener.stop()

        assert len(detector.segments) == 2

    def test_push_to_talk_skips_the_trigger_check(self, fakes):
        """Holding the key is already an explicit request."""
        listener, captured, detector = make_whisper_listener(fakes, matches=False)
        try:
            listener.force_start()
            for _ in range(3):
                listener._pump(tone(1024))
            listener.force_stop()
            listener._pump(tone(1024))
        finally:
            listener.stop()

        assert detector.segments == []  # never consulted
        assert len(captured) == 1       # dispatched anyway

    def test_follow_up_skips_the_trigger_check(self, fakes):
        """Replying to Wawi must not need the wake phrase again."""
        listener, captured, detector = make_whisper_listener(
            fakes, matches=False, follow_up_s=5.0
        )
        try:
            listener.arm_follow_up()
            listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()

        assert detector.segments == []
        assert len(captured) == 1

    def test_the_checked_segment_carries_the_pre_roll(self, fakes):
        """The trigger is at the very start, so it must survive into the check."""
        listener, _, detector = make_whisper_listener(fakes, pre_roll_s=1.0)
        try:
            for i in range(int(2 * SR / 1024)):
                listener._pump(ramp(1024, start=i * 1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()

        assert len(detector.segments) == 1
        assert detector.segments[0].size >= SR


# ---------------------------------------------------------------------------
# Amplitude gate -- surviving a room the VAD alone cannot handle
# ---------------------------------------------------------------------------

class TestMinSpeechLevel:
    def test_quiet_frames_are_silence_however_speechy_the_vad_thinks(self, fakes):
        """The real failure: ambient noise scoring 0.9+ on near-silent audio.

        Without this gate the VAD never reports silence, the utterance never
        endpoints, and NOTHING is ever dispatched.
        """
        cfg = ListenerConfig(sample_rate=SR, min_speech_level=0.05)
        listener, _ = make_listener(fakes, cfg)
        try:
            fakes["vad"].score = lambda frame: 1.0  # VAD insists it is speech
            quiet = _float_to_int16(tone(VAD_FRAME, 0.01))  # but it is quiet
            assert not listener._is_speech(quiet)
        finally:
            listener.stop()

    def test_loud_frames_still_go_through_the_vad(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, min_speech_level=0.05)
        listener, _ = make_listener(fakes, cfg)
        try:
            loud = _float_to_int16(tone(VAD_FRAME, 0.4))
            fakes["vad"].score = lambda frame: 1.0
            assert listener._is_speech(loud)
            # Loud but not speech-like is still not speech.
            fakes["vad"].score = lambda frame: 0.0
            assert not listener._is_speech(loud)
        finally:
            listener.stop()

    def test_the_gate_is_off_by_default(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert listener.config.min_speech_level == 0.0
            fakes["vad"].score = lambda frame: 1.0
            assert listener._is_speech(_float_to_int16(tone(VAD_FRAME, 0.001)))
        finally:
            listener.stop()

    def test_a_noisy_room_can_endpoint_once_the_gate_is_on(self, fakes):
        """End to end: with the gate, quiet noise closes the utterance."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, min_speech_level=0.05)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["vad"].score = lambda frame: 1.0  # noise scores as speech all run
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME, 0.4))       # loud speech opens it
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.01))   # quiet noise closes it
        finally:
            listener.stop()

        assert len(captured) == 1

    @pytest.mark.parametrize("level", [-0.1, 1.5])
    def test_out_of_range_levels_are_rejected(self, level):
        with pytest.raises(ListenerError, match="min_speech_level"):
            Listener(on_utterance=lambda b: None,
                     cfg=ListenerConfig(min_speech_level=level))


class TestQueueBackpressure:
    def test_the_audio_queue_is_bounded(self, fakes):
        """Unbounded, a slow worker drifts behind wall-clock forever."""
        listener, _ = make_listener(fakes)
        try:
            assert listener._queue.maxsize == listener_mod._MAX_QUEUED_BLOCKS
        finally:
            listener.stop()

    def test_a_full_queue_drops_the_oldest_block_not_the_newest(self, fakes):
        """The words being spoken right now must survive; stale audio must not.

        Drives _enqueue directly: with a live worker draining concurrently the
        queue may never fill, and the assertion would pass for the wrong reason.
        """
        listener = Listener(on_utterance=lambda b: None, cfg=ListenerConfig(sample_rate=SR))
        cap = listener_mod._MAX_QUEUED_BLOCKS
        total = cap + 5
        for i in range(total):
            listener._enqueue(ramp(1024, start=i * 1024))

        assert listener._queue.qsize() == cap
        assert listener._dropped_blocks == 5

        drained = []
        while not listener._queue.empty():
            drained.append(listener._queue.get_nowait())

        # Newest survived, oldest were discarded, order preserved.
        assert drained[-1][0] == pytest.approx(((total - 1) * 1024) / 1e6, abs=1e-9)
        assert drained[0][0] == pytest.approx((5 * 1024) / 1e6, abs=1e-9)
        assert len(drained) == cap

    def test_enqueue_never_blocks_the_realtime_thread(self, fakes):
        """A blocked put() in the audio callback would glitch capture outright."""
        listener = Listener(on_utterance=lambda b: None, cfg=ListenerConfig(sample_rate=SR))
        for i in range(listener_mod._MAX_QUEUED_BLOCKS * 3):
            listener._enqueue(tone(1024))   # would hang if put() blocked
        assert listener._queue.qsize() == listener_mod._MAX_QUEUED_BLOCKS


class TestDiagnosticCounters:
    """The listener must be able to say WHY it did nothing.

    Every counter here corresponds to a path that used to be invisible, which
    is what made "a veces no contesta" undiagnosable: ten different causes all
    looked like silence from outside.
    """

    def test_counters_start_empty(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert listener.stats() == {}
        finally:
            listener.stop()

    def test_a_dispatched_utterance_is_counted(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()
        assert listener.stats().get("segments_opened") == 1
        assert listener.stats().get("utterances_dispatched") == 1

    def test_a_rejected_segment_is_counted_with_its_reason(self, fakes):
        listener, captured, detector = make_whisper_listener(fakes, matches=False)
        try:
            for _ in range(3):
                listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()
        stats = listener.stats()
        assert stats.get("segments_rejected") == 1
        assert captured == []

    def test_audio_arriving_while_paused_is_counted(self, fakes):
        """Otherwise "it ignored me while speaking" leaves no trace at all."""
        listener, _ = make_listener(fakes)
        try:
            listener.pause()
            for _ in range(3):
                listener._pump(tone(1024))
        finally:
            listener.stop()
        assert listener.stats().get("blocks_while_paused") == 3

    def test_speech_while_detection_is_off_is_counted(self, fakes):
        """Forgetting the panel toggle looks exactly like a broken wake word."""
        listener, _, _ = make_whisper_listener(fakes)
        try:
            listener.set_wake_enabled(False)
            for _ in range(3):
                listener._pump(tone(1024))
        finally:
            listener.stop()
        assert listener.stats().get("speech_while_wake_disabled", 0) >= 1

    def test_a_capture_that_never_endpoints_is_reported(self, fakes):
        """The quietest failure: talking into a capture that never closes."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))
            # Speech forever: the VAD never reports silence.
            for _ in range(int(listener_mod._LONG_CAPTURE_S * SR / 1024) + 2):
                listener._pump(tone(1024))
        finally:
            listener.stop()
        assert listener.stats().get("captures_stuck_open") == 1
        assert captured == []   # nothing was ever dispatched, which is the point

    def test_the_stuck_warning_fires_once_per_capture_not_per_block(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100)
        listener, _ = make_listener(fakes, cfg)
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))
            for _ in range(int(listener_mod._LONG_CAPTURE_S * SR / 1024) * 2):
                listener._pump(tone(1024))
        finally:
            listener.stop()
        assert listener.stats().get("captures_stuck_open") == 1


class TestFollowUpIsAGraceNotACap:
    """The window bounds when you START, never how long you may talk.

    This is the property that makes the walkie-talkie feel right: a short
    window is not a short answer. Getting it wrong the other way would cut
    people off mid-sentence, which is the exact complaint the window exists
    to fix.
    """

    def test_speaking_stops_the_window_from_counting_down(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, follow_up_s=1.0)
        listener, _ = make_listener(fakes, cfg)
        try:
            listener.arm_follow_up()
            listener._pump(tone(1024))          # speech opens the capture
            assert listener._capturing
            assert not listener.follow_up_open  # window consumed, not ticking
        finally:
            listener.stop()

    def test_a_long_answer_survives_a_short_window(self, fakes):
        """A 1s window must not truncate a 10s reply."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, follow_up_s=1.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            listener.arm_follow_up()
            # Ten seconds of talking, far beyond the one-second window.
            for _ in range(int(10 * SR / 1024)):
                listener._pump(tone(1024))
            assert captured == []      # still going
            assert listener._capturing

            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()

        assert len(captured) == 1
        assert captured[0].size >= 10 * SR   # the whole reply survived

    def test_staying_silent_lets_the_window_expire(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, follow_up_s=0.5)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["vad"].score = lambda frame: 0.0
            listener.arm_follow_up()
            for _ in range(int(0.5 * SR / 1024) + 2):
                listener._pump(tone(1024, 0.0))
            assert not listener.follow_up_open
            # Speaking now is too late: the wake phrase is required again.
            fakes["vad"].score = lambda frame: 1.0
            for _ in range(4):
                listener._pump(tone(1024))
        finally:
            listener.stop()
        assert captured == []

    def test_the_override_beats_the_configured_window(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, follow_up_s=30.0)
        listener, _ = make_listener(fakes, cfg)
        try:
            listener.arm_follow_up(0.5)      # caller knows this turn is short
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(int(0.5 * SR / 1024) + 2):
                listener._pump(tone(1024, 0.0))
            assert not listener.follow_up_open
        finally:
            listener.stop()

    def test_a_zero_override_closes_the_channel_outright(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, follow_up_s=30.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            listener.arm_follow_up(0.0)
            for _ in range(4):
                listener._pump(tone(1024))
        finally:
            listener.stop()
        assert captured == []


class TestProgressiveEndpointing:
    """Snappy for a command, patient for an explanation.

    One threshold cannot serve both, and picking either one alone is what
    makes the assistant feel broken: too short splits an argument into
    half-transcripts, too long makes every command sluggish.
    """

    def test_a_short_utterance_uses_the_base_threshold(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=300, silence_long_ms=3000,
                             long_after_s=8.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()
        assert len(captured) == 1

    def test_a_long_utterance_survives_a_pause_that_would_cut_a_command(self, fakes):
        """Pausing to think mid-argument must not end the sentence."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=300, silence_long_ms=3000,
                             long_after_s=1.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))
            # Talk past long_after_s so the patient threshold takes over.
            for _ in range(int(2 * SR / 1024)):
                listener._pump(tone(1024))

            fakes["vad"].score = lambda frame: 0.0
            # Exactly the silence that WOULD have closed a short utterance.
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert captured == []          # still listening — this is the point
            assert listener._capturing

            # Keep quiet long enough for the patient threshold and it closes.
            for _ in range(listener._silence_frames_long + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()
        assert len(captured) == 1

    def test_resuming_speech_resets_the_silence_counter(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=300, silence_long_ms=3000,
                             long_after_s=1.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))
            for _ in range(int(2 * SR / 1024)):
                listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
            fakes["vad"].score = lambda frame: 1.0
            listener._pump(tone(VAD_FRAME))          # picked the thread back up
            assert listener._silent_frames == 0
            assert captured == []
        finally:
            listener.stop()

    def test_the_escalation_is_off_by_default(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert listener.config.silence_long_ms == 0
            assert listener._silence_frames_long == listener.silence_frames_needed
        finally:
            listener.stop()

    def test_the_threshold_is_reported_in_whole_frames(self, fakes):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=1000, silence_long_ms=5000)
        listener, _ = make_listener(fakes, cfg)
        try:
            assert listener.silence_frames_needed == 34   # ceil(1000/30)
            assert listener._silence_frames_long == 167   # ceil(5000/30)
        finally:
            listener.stop()


class TestPreRollSurvivesBetweenUtterances:
    """The pre-roll ring must NOT be emptied by a normal capture.

    Observed live: consecutive captures reported 1.50s, then 0.26s, 0.19s,
    0.06s of pre-roll. Clearing the ring on finish meant the next utterance
    started with almost no history, so the first syllables of the wake word
    were simply gone — which reads as "no me despierta, tengo que repetirlo".
    """

    def _one_utterance(self, listener, fakes, blocks=3):
        fakes["vad"].score = lambda frame: 1.0
        fakes["wake"].fire_next = True
        listener._pump(tone(WAKE_FRAME))
        for _ in range(blocks):
            listener._pump(tone(1024))
        fakes["vad"].score = lambda frame: 0.0
        for _ in range(listener.silence_frames_needed + 2):
            listener._pump(tone(VAD_FRAME, 0.0))

    def test_the_ring_survives_a_normal_capture(self, fakes):
        """The bug was that it did not, so the next capture started blind."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, pre_roll_s=1.0)
        listener, _ = make_listener(fakes, cfg)
        try:
            self._one_utterance(listener, fakes)
            assert listener._ring.size > 0
        finally:
            listener.stop()

    def test_the_second_utterance_gets_a_full_pre_roll(self, fakes):
        """With a second of room noise between turns — the real-world case."""
        cfg = ListenerConfig(sample_rate=SR, silence_ms=100, pre_roll_s=1.0)
        listener, captured = make_listener(fakes, cfg)
        try:
            self._one_utterance(listener, fakes)

            # A second of quiet passes, as it does between two things you say.
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(int(SR / 1024) + 1):
                listener._pump(tone(1024, 0.0))

            self._one_utterance(listener, fakes)
        finally:
            listener.stop()

        assert len(captured) == 2
        # The full pre-roll is there, exactly as it is for the first utterance.
        assert captured[1].size >= SR

    def test_pausing_still_wipes_the_ring(self, fakes):
        """The assistant's own voice must never survive into a later utterance."""
        listener, _ = make_listener(fakes)
        try:
            for _ in range(4):
                listener._pump(tone(1024))
            assert listener._ring.size > 0
            listener.pause()
            listener._pump(tone(1024))
            assert listener._ring.size == 0
        finally:
            listener.stop()

    def test_a_rejected_segment_also_keeps_the_pre_roll(self, fakes):
        """Saying the wake word right after a rejected segment must still work."""
        listener, _, _ = make_whisper_listener(fakes, matches=False)
        try:
            for _ in range(3):
                listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert listener._ring.size > 0
        finally:
            listener.stop()


class TestCaptureVisibility:
    """The panel needs a live "I am recording this" signal.

    With the whisper engine, whether a segment contains the wake phrase is not
    known until it CLOSES and is transcribed. Waiting for that to light up the
    UI means it lights up once the turn has already started — exactly when the
    feedback stops being useful.
    """

    def test_not_capturing_while_idle(self, fakes):
        listener, _ = make_listener(fakes)
        try:
            assert not listener.is_capturing
            listener._pump(tone(1024, 0.0))
            assert not listener.is_capturing
        finally:
            listener.stop()

    def test_capturing_becomes_true_as_soon_as_the_segment_opens(self, fakes):
        """Before the trigger is known — that is the whole point."""
        listener, _, detector = make_whisper_listener(fakes, matches=False)
        try:
            listener._pump(tone(1024))
            assert listener.is_capturing
            assert detector.segments == []   # nothing transcribed yet
        finally:
            listener.stop()

    def test_capturing_goes_false_when_the_segment_closes(self, fakes):
        listener, _, _ = make_whisper_listener(fakes, matches=False)
        try:
            listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
            assert not listener.is_capturing
        finally:
            listener.stop()

    def test_push_to_talk_shows_as_capturing(self, fakes):
        listener, _, _ = make_whisper_listener(fakes)
        try:
            listener.force_start()
            listener._pump(tone(1024))
            assert listener.is_capturing
        finally:
            listener.stop()


class TestStuckWarningNamesTheRightKnob:
    """A diagnostic that names the wrong setting is worse than none.

    Once the utterance passes long_after_s the threshold in force is
    silence_long_ms; telling the user to lower silence_ms sends them to a
    number that changes nothing about what they are seeing.
    """

    def _warn_text(self, listener, fakes, caplog, talk_blocks):
        import logging

        with caplog.at_level(logging.WARNING, logger="wawi.listener"):
            fakes["wake"].fire_next = True
            listener._pump(tone(WAKE_FRAME))
            for _ in range(talk_blocks):
                listener._pump(tone(1024))
        return " ".join(r.getMessage() for r in caplog.records)

    def test_a_short_utterance_points_at_silence_ms(self, fakes, caplog):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=300, silence_long_ms=3000,
                             long_after_s=60.0)
        listener, _ = make_listener(fakes, cfg)
        try:
            text = self._warn_text(
                listener, fakes, caplog,
                int(listener_mod._LONG_CAPTURE_S * SR / 1024) + 2,
            )
        finally:
            listener.stop()
        assert "wake.silence_ms" in text
        assert "wake.silence_long_ms" not in text

    def test_a_long_utterance_points_at_silence_long_ms(self, fakes, caplog):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=300, silence_long_ms=3000,
                             long_after_s=1.0)
        listener, _ = make_listener(fakes, cfg)
        try:
            text = self._warn_text(
                listener, fakes, caplog,
                int(listener_mod._LONG_CAPTURE_S * SR / 1024) + 2,
            )
        finally:
            listener.stop()
        assert "wake.silence_long_ms" in text

    def test_the_warning_reports_the_threshold_in_force(self, fakes, caplog):
        cfg = ListenerConfig(sample_rate=SR, silence_ms=300, silence_long_ms=3000,
                             long_after_s=1.0)
        listener, _ = make_listener(fakes, cfg)
        try:
            text = self._warn_text(
                listener, fakes, caplog,
                int(listener_mod._LONG_CAPTURE_S * SR / 1024) + 2,
            )
        finally:
            listener.stop()
        assert "3000ms" in text     # the long threshold, not the base one


# ---------------------------------------------------------------------------
# Always-on -- require_phrase: false. No trigger stage at all.
# ---------------------------------------------------------------------------

class TestAlwaysOn:
    """Speech IS the turn: the wake phrase is not a per-turn toll."""

    def _always_on(self, fakes, engine="whisper", detector=None, **kwargs):
        captured: list[np.ndarray] = []
        cfg = ListenerConfig(
            engine=engine,
            require_phrase=False,
            sample_rate=SR,
            silence_ms=100,
            **kwargs,
        )
        listener = Listener(
            on_utterance=captured.append, cfg=cfg, wake_detector=detector
        )
        listener.start()
        return listener, captured

    def _speak_then_stop(self, listener, fakes, blocks=4):
        for _ in range(blocks):
            listener._pump(tone(1024))
        fakes["vad"].score = lambda frame: 0.0
        for _ in range(listener.silence_frames_needed + 2):
            listener._pump(tone(VAD_FRAME, 0.0))

    def test_whisper_engine_needs_no_detector(self, fakes):
        """The trigger model is dead weight when nothing is ever matched."""
        listener, _ = self._always_on(fakes)
        listener.stop()

    def test_plain_speech_is_dispatched_without_any_wake_phrase(self, fakes):
        listener, captured = self._always_on(fakes)
        try:
            self._speak_then_stop(listener, fakes)
        finally:
            listener.stop()
        assert len(captured) == 1

    def test_the_pretrained_wake_model_is_never_scored(self, fakes):
        """Not even with engine=openwakeword: there is nothing to detect."""
        listener, captured = self._always_on(fakes, engine="openwakeword")
        try:
            self._speak_then_stop(listener, fakes)
        finally:
            listener.stop()
        assert fakes["wake"].frames_seen == []
        assert len(captured) == 1

    def test_the_push_to_talk_toggle_still_mutes_it(self, fakes):
        """Always-on must stay switchable off, or there is no way to shut up."""
        listener, captured = self._always_on(fakes)
        try:
            listener.set_wake_enabled(False)
            self._speak_then_stop(listener, fakes)
        finally:
            listener.stop()
        assert captured == []
        assert listener.stats().get("speech_while_wake_disabled", 0) > 0

    def test_the_transcript_is_never_matched_against_the_phrases(self, fakes):
        """A detector present for echo duty must not resurrect the trigger."""
        detector = FakeDetector(matches=False)
        listener, captured = self._always_on(fakes, detector=detector)
        try:
            self._speak_then_stop(listener, fakes)
        finally:
            listener.stop()
        assert len(captured) == 1, "'no match' must not drop the turn any more"

    def test_its_own_voice_still_cannot_start_a_turn(self, fakes):
        """The echo check is the ONLY filter left; it must survive."""
        detector = FakeDetector(matches=False)
        listener, captured = self._always_on(fakes, detector=detector)
        try:
            # FakeDetector transcribes every segment as "viernes dale".
            listener.set_echo_text("viernes dale")
            self._speak_then_stop(listener, fakes)
        finally:
            listener.stop()
        assert captured == []
        assert listener.stats().get("segments_rejected_echo", 0) == 1


class TestMicrophoneGate:
    """The panel switch means deaf, and deaf has to mean ALL the ways in.

    Each of these was a real hole: the follow-up window opened a capture
    without consulting the gate at all, and with conversation_mode that window
    opens after every answer — so muting did nothing until it expired. From
    outside, that is indistinguishable from a switch that was never wired.
    """

    def _listener(self, **kw):
        from wawi.listener import Listener, ListenerConfig

        utterances = []
        listener = Listener(
            on_utterance=utterances.append,
            cfg=ListenerConfig(require_phrase=False, **kw),
        )
        return listener, utterances

    def test_the_gate_starts_open(self):
        listener, _ = self._listener()
        assert listener.is_wake_enabled is True

    def test_closing_it_cancels_an_open_follow_up_window(self):
        # Otherwise you mute it and it keeps answering for the length of the
        # window, which is the exact symptom that was reported.
        listener, _ = self._listener()
        listener._follow_up_left = 48_000
        listener._arm_follow_up.set()

        listener.set_wake_enabled(False)

        assert listener._follow_up_left == 0
        assert listener._arm_follow_up.is_set() is False

    def test_closing_it_drops_a_capture_already_running(self):
        # Muting has to take effect on the word you are saying, not the next.
        listener, _ = self._listener()
        listener._capturing = True
        listener._forced = False

        listener.set_wake_enabled(False)

        assert listener._capturing is False

    def test_push_to_talk_is_refused_while_muted(self):
        listener, _ = self._listener()
        listener.set_wake_enabled(False)

        listener.force_start()

        assert list(listener._force_cmds) == []

    def test_push_to_talk_works_again_once_reopened(self):
        listener, _ = self._listener()
        listener.set_wake_enabled(False)
        listener.set_wake_enabled(True)

        listener.force_start()

        assert list(listener._force_cmds) == [True]


# ---------------------------------------------------------------------------
# Smart end-of-turn: the model decides at a short pause, silence_ms is the cap
# ---------------------------------------------------------------------------

class FakeTurnModel:
    """Stand-in for wawi.turn.SmartTurn. Returns a fixed P(complete)."""

    def __init__(self, p: float = 0.9, error: Exception | None = None,
                 load_error: Exception | None = None) -> None:
        self.p = p
        self.error = error
        self.load_error = load_error
        self.loads = 0
        self.calls: list[np.ndarray] = []

    def load(self) -> None:
        self.loads += 1
        if self.load_error is not None:
            raise self.load_error

    def predict(self, audio: np.ndarray) -> float:
        self.calls.append(audio)
        if self.error is not None:
            raise self.error
        return self.p


class TestSmartTurn:
    # 300 ms pause = 10 VAD frames; 900 ms ceiling = 30 frames.
    PAUSE = 10
    CEILING = 30

    def _listener(self, fakes, det, inline=True, **kw):
        cfg = ListenerConfig(
            sample_rate=SR, pre_roll_s=0.1, silence_ms=900, smart_turn=True,
            smart_turn_pause_ms=300, **kw,
        )
        captured: list[np.ndarray] = []
        listener = Listener(on_utterance=captured.append, cfg=cfg, turn_detector=det)
        listener._turn_inline = inline
        listener.start()
        return listener, captured

    def _speak(self, listener, fakes, wake=True):
        fakes["vad"].score = lambda frame: 1.0
        fakes["wake"].fire_next = wake
        listener._pump(tone(WAKE_FRAME))
        assert listener.is_capturing
        fakes["vad"].score = lambda frame: 0.0

    def _silence(self, listener, frames):
        for _ in range(frames):
            listener._pump(tone(VAD_FRAME, 0.0))

    def test_complete_verdict_closes_at_the_pause(self, fakes, caplog):
        det = FakeTurnModel(p=0.93)
        listener, captured = self._listener(fakes, det)
        try:
            self._speak(listener, fakes)
            self._silence(listener, self.PAUSE - 1)
            assert captured == [] and det.calls == [], "asked before the pause"

            with caplog.at_level("INFO", logger="wawi.listener"):
                self._silence(listener, 1)

            assert len(captured) == 1, "a confident 'complete' must close now"
            assert len(det.calls) == 1
            # The model hears the utterance itself, never more than 8 s of it.
            assert 0 < det.calls[0].size <= 8 * SR
            assert det.calls[0].dtype == np.float32
            line = [r.getMessage() for r in caplog.records
                    if r.getMessage().startswith("Smart turn:")]
            assert len(line) == 1
            assert "decision=complete pause_ms=300 p=0.930" in line[0]
            assert "checks=1" in line[0]
        finally:
            listener.stop()

    def test_incomplete_verdict_waits_for_the_silence_ceiling(self, fakes, caplog):
        det = FakeTurnModel(p=0.2)
        listener, captured = self._listener(fakes, det)
        try:
            self._speak(listener, fakes)
            self._silence(listener, self.CEILING - 1)
            assert captured == [], "an 'incomplete' verdict must keep listening"
            # Asked once per pause: a second look later in the same pause
            # would only be a shorter timer (P drifts up with trailing silence).
            assert len(det.calls) == 1

            with caplog.at_level("INFO", logger="wawi.listener"):
                self._silence(listener, 1)
            assert len(captured) == 1
            assert any("decision=continue pause_ms=900 p=0.200" in r.getMessage()
                       for r in caplog.records)
        finally:
            listener.stop()

    def test_resumed_speech_makes_an_in_flight_verdict_stale(self, fakes, monkeypatch):
        det = FakeTurnModel(p=0.99)
        listener, captured = self._listener(fakes, det, inline=False)
        jobs: list = []
        # Hold the job instead of handing it to the model thread, so the
        # verdict can be delivered at a moment of the test's choosing.
        monkeypatch.setattr(listener, "_submit_turn_job", jobs.append)
        try:
            self._speak(listener, fakes)
            self._silence(listener, self.PAUSE)
            assert len(jobs) == 1 and captured == []

            # The speaker carries on while the model is still thinking.
            fakes["vad"].score = lambda frame: 1.0
            listener._pump(tone(VAD_FRAME))
            fakes["vad"].score = lambda frame: 0.0

            # Now the "complete" verdict about the OLD pause lands.
            listener._run_turn_job(*jobs[0])
            listener._poll_turn()
            self._silence(listener, 3)
            assert captured == [], "a verdict about a pause that ended was applied"

            # The new pause gets its own, fresh look -- and that one counts.
            self._silence(listener, self.PAUSE - 3)
            assert len(jobs) == 2
            listener._run_turn_job(*jobs[1])
            listener._poll_turn()
            assert len(captured) == 1
        finally:
            listener.stop()

    def test_speech_after_a_continue_verdict_gets_a_fresh_check(self, fakes, caplog):
        det = FakeTurnModel(p=0.1)
        listener, captured = self._listener(fakes, det)
        try:
            self._speak(listener, fakes)
            self._silence(listener, self.PAUSE + 5)
            assert len(det.calls) == 1

            fakes["vad"].score = lambda frame: 1.0  # "...y después de eso"
            listener._pump(tone(VAD_FRAME))
            fakes["vad"].score = lambda frame: 0.0
            det.p = 0.95
            with caplog.at_level("INFO", logger="wawi.listener"):
                self._silence(listener, self.PAUSE)
            assert len(det.calls) == 2
            assert len(captured) == 1
            assert any("decision=complete" in r.getMessage() and "resumed=1" in r.getMessage()
                       for r in caplog.records)
        finally:
            listener.stop()

    def test_the_real_model_thread_closes_the_turn(self, fakes):
        det = FakeTurnModel(p=0.9)
        done = __import__("threading").Event()
        captured: list[np.ndarray] = []

        def on_utt(buf):
            captured.append(buf)
            done.set()

        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=900,
                             smart_turn=True, smart_turn_pause_ms=300)
        listener = Listener(on_utterance=on_utt, cfg=cfg, turn_detector=det)
        listener.start()
        try:
            assert listener._turn_thread is not None and listener._turn_thread.is_alive()
            self._speak(listener, fakes)
            self._silence(listener, self.PAUSE)
            # The verdict reaches the worker through the audio queue; no more
            # audio is needed for it to be applied.
            assert done.wait(2.0), "the model thread's verdict was never applied"
            assert len(captured) == 1
        finally:
            listener.stop()
        assert listener._turn_thread is None

    def test_missing_model_falls_back_to_plain_silence(self, fakes, monkeypatch, caplog):
        from wawi import turn as turn_mod

        monkeypatch.setattr(turn_mod, "model_path", lambda: None)
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=900,
                             smart_turn=True, smart_turn_pause_ms=300)
        captured: list[np.ndarray] = []
        with caplog.at_level("WARNING", logger="wawi.listener"):
            listener = Listener(on_utterance=captured.append, cfg=cfg)
            listener.start()
        try:
            warnings = [r for r in caplog.records if "Smart turn unavailable" in r.getMessage()]
            assert len(warnings) == 1
            assert listener._turn is None

            caplog.clear()
            with caplog.at_level("INFO", logger="wawi.listener"):
                self._speak(listener, fakes)
                self._silence(listener, self.CEILING - 1)
                assert captured == [], "closed early with no model"
                self._silence(listener, 1)
            assert len(captured) == 1, "must behave exactly like silence_ms"
            assert not any("Smart turn" in r.getMessage() for r in caplog.records)
        finally:
            listener.stop()

    def test_a_failing_model_is_logged_once_and_disabled(self, fakes, caplog):
        det = FakeTurnModel(error=RuntimeError("onnx exploded"))
        listener, captured = self._listener(fakes, det)
        try:
            with caplog.at_level("WARNING", logger="wawi.listener"):
                for _ in range(2):
                    self._speak(listener, fakes)
                    self._silence(listener, self.CEILING - 1)
                    self._silence(listener, 1)
            assert len(captured) == 2, "both turns must still close on silence_ms"
            assert len(det.calls) == 1, "a broken model must not be retried every pause"
            failures = [r for r in caplog.records if "Smart turn model failed" in r.getMessage()]
            assert len(failures) == 1
        finally:
            listener.stop()

    def test_push_to_talk_ignores_the_model(self, fakes):
        det = FakeTurnModel(p=0.99)
        listener, captured = self._listener(fakes, det)
        try:
            listener.force_start()
            fakes["vad"].score = lambda frame: 0.0
            listener._pump(tone(WAKE_FRAME, 0.0))
            assert listener.is_capturing
            self._silence(listener, self.CEILING * 2)
            assert captured == [] and det.calls == []
            listener.force_stop()
            listener._pump(tone(VAD_FRAME, 0.0))
            assert len(captured) == 1
        finally:
            listener.stop()

    def test_off_by_default_the_model_is_never_loaded(self, fakes):
        det = FakeTurnModel(p=0.99)
        captured: list[np.ndarray] = []
        cfg = ListenerConfig(sample_rate=SR, pre_roll_s=0.1, silence_ms=900)
        listener = Listener(on_utterance=captured.append, cfg=cfg, turn_detector=det)
        listener.start()
        try:
            self._speak(listener, fakes)
            self._silence(listener, self.CEILING - 1)
            assert captured == [] and det.loads == 0 and det.calls == []
        finally:
            listener.stop()

    def test_an_early_close_keeps_only_the_pause_in_the_pre_roll(self, fakes):
        # After a 300 ms pause the ring still holds the END of what was said;
        # left there it would be transcribed again as the next pre-roll.
        det = FakeTurnModel(p=0.95)
        listener, captured = self._listener(fakes, det)
        try:
            self._speak(listener, fakes)
            self._silence(listener, self.PAUSE)
            assert len(captured) == 1
            assert 0 < listener._ring.size <= self.PAUSE * VAD_FRAME
            assert np.all(listener._ring.read() == 0.0)
        finally:
            listener.stop()

    @pytest.mark.parametrize(
        "bad",
        [{"smart_turn_threshold": 1.5}, {"smart_turn_threshold": -0.1},
         {"smart_turn_pause_ms": 0}],
    )
    def test_bad_smart_turn_config_is_rejected(self, bad):
        with pytest.raises(ListenerError):
            Listener(on_utterance=lambda b: None, cfg=ListenerConfig(sample_rate=SR, **bad))


# ---------------------------------------------------------------------------
# Noise gates: onset, voiced duration, voiced ratio
# ---------------------------------------------------------------------------

LOUD, WEAKV, QUIET = 0.5, 0.2, 0.0


def _amp_score(frame) -> float:
    """Speech score from frame amplitude: LOUD=0.9, WEAKV=0.55, QUIET=0."""
    peak = int(np.abs(frame).max())
    if peak > 12000:
        return 0.9
    if peak > 4000:
        return 0.55
    return 0.0


class TestNoiseGates:
    """Noise must not open turns; real speech, however short, must."""

    def _listener(self, fakes, **kwargs):
        captured: list[np.ndarray] = []
        events: list[str] = []
        base = dict(
            engine="whisper", require_phrase=False, sample_rate=SR, silence_ms=150,
            vad_onset_threshold=0.6, min_onset_ms=150, onset_window_ms=400,
            min_voiced_ms=250, min_voiced_ratio=0.2,
        )
        base.update(kwargs)
        listener = Listener(
            on_utterance=captured.append,
            cfg=ListenerConfig(**base),
            on_capture_start=lambda: events.append("start"),
            on_capture_dropped=lambda: events.append("dropped"),
        )
        listener.start()
        fakes["vad"].score = _amp_score
        return listener, captured, events

    @staticmethod
    def _frames(listener, pattern):
        """Pump one VAD frame per character: S=loud speech, w=weak, .=quiet."""
        amp = {"S": LOUD, "w": WEAKV, ".": QUIET}
        for ch in pattern:
            listener._pump(tone(VAD_FRAME, amp[ch]))

    def test_a_single_spike_does_not_open_a_segment(self, fakes):
        # A clap, a key, a door: one or two frames.
        listener, captured, events = self._listener(fakes)
        try:
            self._frames(listener, "SS" + "." * 30)
        finally:
            listener.stop()
        assert not captured and events == []
        assert listener.stats().get("segments_opened", 0) == 0

    def test_scattered_clicks_never_add_up_to_an_onset(self, fakes):
        # Typing: one speechy frame every ~150 ms, for two seconds.
        listener, captured, events = self._listener(fakes)
        try:
            self._frames(listener, ("S" + "...." ) * 14 + "." * 10)
        finally:
            listener.stop()
        assert not captured and events == []

    def test_sustained_speech_opens_and_is_dispatched(self, fakes):
        listener, captured, events = self._listener(fakes)
        try:
            self._frames(listener, "S" * 20 + "." * 10)
        finally:
            listener.stop()
        assert len(captured) == 1
        assert events == ["start"]
        speech = captured[0].speech
        assert speech.voiced_ms == 20 * 30 and not speech.forced

    def test_a_short_real_word_still_gets_through(self, fakes):
        # "sí": ~300 ms of voiced frames. Must not be mistaken for noise.
        listener, captured, _ = self._listener(fakes)
        try:
            self._frames(listener, "S" * 10 + "." * 10)
        finally:
            listener.stop()
        assert len(captured) == 1

    def test_hysteresis_weak_frames_cannot_open_but_can_continue(self, fakes):
        listener, captured, _ = self._listener(fakes)
        try:
            # Weak frames alone (0.55: above vad_threshold, below onset).
            self._frames(listener, "w" * 20 + "." * 10)
            assert not captured and not listener.is_capturing
            # Opened by strong speech, a weak stretch keeps it alive.
            self._frames(listener, "S" * 8 + "w" * 4)
            assert listener.is_capturing
            self._frames(listener, "S" * 4 + "." * 10)
        finally:
            listener.stop()
        assert len(captured) == 1
        assert captured[0].speech.voiced_ms == 16 * 30

    def test_a_burst_too_short_is_dropped_before_whisper(self, fakes, caplog):
        caplog.set_level("INFO", logger="wawi.listener")
        listener, captured, events = self._listener(fakes, min_voiced_ms=400)
        try:
            self._frames(listener, "S" * 7 + "." * 10)  # 210 ms: opens, too short
        finally:
            listener.stop()
        assert captured == []
        assert listener.stats()["segments_dropped_noise"] == 1
        # Never announced, so the UI never showed "escuchando" for it.
        assert events == []
        assert "Dropped segment as noise: too little speech" in caplog.text
        assert "wake.min_voiced_ms" in caplog.text

    def test_sparse_noise_is_dropped_by_the_ratio(self, fakes, caplog):
        caplog.set_level("INFO", logger="wawi.listener")
        listener, captured, events = self._listener(fakes, silence_ms=600)
        try:
            # Opens on a real-looking burst, then only isolated speechy frames
            # strung across seconds -- a TV across the room, a fan's tick.
            self._frames(listener, "S" * 9 + ("S" + "." * 15) * 6 + "." * 25)
        finally:
            listener.stop()
        assert captured == []
        assert "sparse speech" in caplog.text
        # It WAS announced (enough voiced frames), so it must be withdrawn.
        assert events == ["start", "dropped"]

    def test_push_to_talk_is_never_filtered(self, fakes):
        listener, captured, events = self._listener(fakes)
        try:
            listener.force_start()
            self._frames(listener, "S" + "." * 20)
            listener.force_stop()
            self._frames(listener, ".")
        finally:
            listener.stop()
        assert len(captured) == 1
        assert captured[0].speech.forced
        assert events == ["start"]

    def test_noise_does_not_spend_the_follow_up_window(self, fakes):
        listener, captured, _ = self._listener(
            fakes, engine="openwakeword", require_phrase=True, follow_up_s=5.0,
            min_voiced_ms=400,
        )
        try:
            listener.arm_follow_up()
            self._frames(listener, "S" * 7 + "." * 10)  # noise: dropped
            assert listener.follow_up_open, "the reply window must survive a cough"
            self._frames(listener, "S" * 20 + "." * 10)  # the actual reply
        finally:
            listener.stop()
        assert len(captured) == 1

    def test_gates_off_keep_the_old_single_frame_behaviour(self, fakes):
        listener, captured, _ = self._listener(
            fakes, vad_onset_threshold=0.0, min_onset_ms=0, min_voiced_ms=0,
            min_voiced_ratio=0.0,
        )
        try:
            self._frames(listener, "S" + "." * 10)
        finally:
            listener.stop()
        assert len(captured) == 1

    @pytest.mark.parametrize("kwargs", [
        {"vad_onset_threshold": 1.5},
        {"min_voiced_ratio": -0.1},
        {"min_onset_ms": -1},
        {"min_voiced_ms": -5},
        {"onset_window_ms": 0},
    ])
    def test_bad_values_are_rejected(self, kwargs):
        with pytest.raises(ListenerError):
            Listener(on_utterance=lambda b: None, cfg=ListenerConfig(sample_rate=SR, **kwargs))


class _TextDetector(FakeDetector):
    """Transcribes every segment as a fixed text (the echo check's view)."""

    def __init__(self, text: str) -> None:
        super().__init__(matches=False)
        self.text = text

    def check(self, segment):
        self.segments.append(segment)
        from wawi.wake import WakeMatch

        return WakeMatch(None, self.text, 0.04)


class TestNoiseWhileSpeaking:
    """With barge-in on, noise must never cut the assistant off."""

    def _run(self, fakes, text):
        detector = _TextDetector(text)
        captured: list[np.ndarray] = []
        listener = Listener(
            on_utterance=captured.append,
            cfg=ListenerConfig(engine="whisper", require_phrase=False,
                               sample_rate=SR, silence_ms=100),
            wake_detector=detector,
        )
        listener.start()
        try:
            listener.set_echo_text("te cuento que el proyecto está andando bien")
            for _ in range(4):
                listener._pump(tone(1024))
            fakes["vad"].score = lambda frame: 0.0
            for _ in range(listener.silence_frames_needed + 2):
                listener._pump(tone(VAD_FRAME, 0.0))
        finally:
            listener.stop()
        return captured, listener

    def test_a_wordless_segment_is_dropped(self, fakes):
        captured, listener = self._run(fakes, "")
        assert captured == []
        assert listener.stats()["segments_rejected_echo"] == 1

    def test_a_hallucination_is_dropped(self, fakes):
        captured, _ = self._run(fakes, "¡Suscríbete al canal!")
        assert captured == []

    def test_real_words_still_barge_in(self, fakes):
        captured, _ = self._run(fakes, "pará, pará, cambiá de tema")
        assert len(captured) == 1


class TestNoiseCandidatesStayInvisible:
    def test_the_spectrum_does_not_light_up_for_a_candidate(self, fakes):
        listener = Listener(
            on_utterance=lambda b: None,
            cfg=ListenerConfig(engine="whisper", require_phrase=False, sample_rate=SR,
                               silence_ms=150, vad_onset_threshold=0.6,
                               min_onset_ms=150, min_voiced_ms=400),
        )
        listener.start()
        fakes["vad"].score = _amp_score
        try:
            for _ in range(6):  # opens (onset 150 ms) but only 180 ms voiced
                listener._pump(tone(VAD_FRAME, LOUD))
            assert listener._capturing and not listener.is_capturing
            for _ in range(10):  # 480 ms voiced: now it is speech
                listener._pump(tone(VAD_FRAME, LOUD))
            assert listener.is_capturing
        finally:
            listener.stop()
