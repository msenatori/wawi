"""Smart end-of-turn detection: has the speaker FINISHED, or only paused?

The listener closes an utterance after ``wake.silence_ms`` of continuous VAD
silence. That one number has to serve two opposite cases -- "¿Qué hora es en
Tokio?" wants an instant answer, "Quiero que me busques el..." wants patience
-- and measured on this project it is the single largest stage of a turn
(2.6 s fixed, every turn; see issues/*latencia-del-fin-del-habla*).

This module answers the question the silence timer cannot: given the audio of
the turn so far, does it SOUND finished? It wraps Pipecat's Smart Turn v3, a
Whisper-tiny-encoder classifier that reads prosody and grammar straight from
the waveform (no transcript) and returns P(turn complete). The listener asks it
once the pause reaches ``smart_turn_pause_ms``; a confident "complete" closes
the utterance right there, anything else keeps listening and the old
``silence_ms`` stays as the ceiling.

Model
-----
``pipecat-ai/smart-turn-v3`` on Hugging Face, file ``smart-turn-v3.2-gpu.onnx``
(fp32, ~32 MB), BSD 2-Clause (Copyright (c) 2024-2025, Daily). The revision is
pinned and the file's SHA-256 is checked after download, so a silent upstream
change cannot alter behaviour. Upstream reports 89.6% accuracy on Spanish for
the int8 build (FPR 6.95%: a finished-sounding verdict on an unfinished turn)
-- good, not perfect, which is why the silence ceiling stays.

Why the fp32 file and not the 8 MB int8 "cpu" one, measured on Apple Silicon:

- int8 is SLOWER here: 35.7 ms median vs 22.5 ms single-threaded (onnxruntime's
  dynamic-quantisation kernels are not what the M-series is fast at).
- int8 is numerically unstable near the threshold. Jittering the input
  features by 1e-5 -- less than the gap between transformers' own numpy and
  torch front-ends -- moved one clip's P from 0.26 to 0.79. fp32 did not move
  in the fourth decimal.
- On Spanish clips cut mid-sentence, fp32 called 7/10 "incomplete", int8 2/10.

The name says "gpu"; it is plain fp32 ONNX and runs on the CPU provider.

Like every other Wawi model, the weights are NOT bundled and never download
implicitly: :func:`model_ready` is a filesystem check for the ``--check``
doctor and :func:`download_model` is the explicit opt-in. The file lives in the
user's Hugging Face cache, next to the Whisper models.

Preprocessing -- reproduced, not imported
-----------------------------------------
Upstream feeds the model through ``transformers.WhisperFeatureExtractor``
(chunk_length=8, do_normalize=True). Pulling ``transformers`` in for one
spectrogram is a large dependency for ~40 lines of numpy, so
:func:`whisper_features` reimplements it exactly: last 8 s of audio, zero-padded
at the FRONT, zero-mean/unit-variance normalised over the whole 8 s window,
then Whisper's log-mel (n_fft 400, hop 160, periodic Hann, reflect-padded
centre framing, 80 Slaney mel bins 0-8 kHz, log10 floored at 1e-10, clamped to
max-8, scaled (x+4)/4, last frame dropped) -> float32 ``(1, 80, 800)``.
Verified numerically against the transformers implementation; see
``scripts/bench_smart_turn.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time

import numpy as np

log = logging.getLogger(__name__)

MODEL_REPO = "pipecat-ai/smart-turn-v3"
MODEL_FILE = "smart-turn-v3.2-gpu.onnx"
# Pinned: the repo's main branch has already moved v3.0 -> v3.1 -> v3.2 once.
MODEL_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
MODEL_SHA256 = "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be"
MODEL_LICENSE = "BSD-2-Clause (Copyright (c) 2024-2025, Daily)"

SAMPLE_RATE = 16000
WINDOW_S = 8
_N_SAMPLES = WINDOW_S * SAMPLE_RATE
_N_FFT = 400
_HOP = 160
_N_MELS = 80


class TurnModelError(RuntimeError):
    """The Smart Turn model is missing, failed to load, or failed to run."""


# ---------------------------------------------------------------------------
# Model-file management (no implicit network access)
# ---------------------------------------------------------------------------

def model_path() -> str | None:
    """Return the cached model file, or None when it has not been downloaded.

    Purely a cache lookup: never touches the network.
    """
    # JARVIS_SMART_TURN_MODEL is the name this variable had before the rename.
    override = os.environ.get("WAWI_SMART_TURN_MODEL") or os.environ.get(
        "JARVIS_SMART_TURN_MODEL"
    )
    if override:
        return override if os.path.isfile(override) else None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        found = try_to_load_from_cache(
            MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION
        )
    except Exception:  # pragma: no cover - defensive; the doctor must not crash
        log.debug("Smart Turn cache lookup failed", exc_info=True)
        return None
    return found if isinstance(found, str) and os.path.isfile(found) else None


def model_ready() -> bool:
    """True when the model file is on disk. Never downloads, never raises."""
    return model_path() is not None


def download_model() -> str:
    """Fetch the pinned model into the Hugging Face cache. Explicit opt-in.

    ~32 MB. Already-cached files are not fetched again. The SHA-256 is
    verified against the pinned value, so a corrupted or swapped file is
    refused rather than loaded.

    Returns:
        The local path of the model file.

    Raises:
        TurnModelError: huggingface_hub missing, download failed, or the
            checksum does not match.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise TurnModelError(
            "huggingface_hub is not installed (it normally comes with "
            "mlx-whisper):  pip install huggingface_hub"
        ) from exc
    log.info("Downloading %s/%s @ %s…", MODEL_REPO, MODEL_FILE, MODEL_REVISION[:8])
    try:
        path = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION)
    except Exception as exc:
        raise TurnModelError(f"Downloading the Smart Turn model failed: {exc}") from exc
    digest = _sha256(path)
    if digest != MODEL_SHA256:
        raise TurnModelError(
            f"Smart Turn model checksum mismatch: got {digest}, expected "
            f"{MODEL_SHA256}. Delete {path} and retry."
        )
    log.info("Smart Turn model ready: %s", path)
    return path


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Preprocessing: WhisperFeatureExtractor(chunk_length=8), in numpy
# ---------------------------------------------------------------------------

def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    freq = np.asarray(freq, dtype=np.float64)
    mels = 3.0 * freq / 200.0
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
    log_region = freq >= min_log_hz
    mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    freq = 200.0 * mels / 3.0
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def _mel_filters() -> np.ndarray:
    """(201, 80) Slaney-normalised triangular filters, as transformers builds them."""
    n_bins = 1 + _N_FFT // 2
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, n_bins)
    mel_freqs = np.linspace(_hz_to_mel(np.array([0.0]))[0],
                            _hz_to_mel(np.array([8000.0]))[0], _N_MELS + 2)
    filter_freqs = _mel_to_hz(mel_freqs)
    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / filter_diff[:-1]
    up = slopes[:, 2:] / filter_diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2 : _N_MELS + 2] - filter_freqs[:_N_MELS])
    return filters * enorm[None, :]


_MEL_FILTERS = _mel_filters()
# Periodic Hann, i.e. scipy.signal.get_window("hann", 400) / torch.hann_window.
_WINDOW = np.hanning(_N_FFT + 1)[:-1]


def fit_window(audio: np.ndarray) -> np.ndarray:
    """Keep the LAST 8 s, or zero-pad at the front up to 8 s (upstream rule)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= _N_SAMPLES:
        return audio[-_N_SAMPLES:]
    return np.pad(audio, (_N_SAMPLES - audio.size, 0))


def whisper_features(audio: np.ndarray) -> np.ndarray:
    """16 kHz mono float audio -> the model's ``input_features``, (1, 80, 800)."""
    x = fit_window(audio).astype(np.float64)
    # do_normalize=True: over the whole padded window (attention mask is all
    # ones because the input is already exactly max_length long).
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    x = np.pad(x, _N_FFT // 2, mode="reflect")
    n_frames = 1 + (x.size - _N_FFT) // _HOP
    idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n_frames)[:, None]
    power = np.abs(np.fft.rfft(x[idx] * _WINDOW, n=_N_FFT, axis=1)) ** 2
    mel = power @ _MEL_FILTERS                      # (frames, 80)
    log_spec = np.log10(np.maximum(mel, 1e-10)).T  # (80, frames)
    log_spec = log_spec[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[None, :, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class SmartTurn:
    """P(turn complete) for the audio of the current utterance.

    ``predict`` is NOT thread-safe against itself; the listener calls it from
    one dedicated thread. Two intra-op threads: 15 ms median on an 8 s window
    on this Mac, vs 22 ms with one and 11 ms with four -- two leaves the other
    cores to Silero and the audio callback.
    """

    def __init__(self, path: str | None = None, threads: int = 2) -> None:
        self._path = path
        self._threads = max(1, int(threads))
        self._session = None
        self._input_name = "input_features"

    @property
    def path(self) -> str | None:
        return self._path

    def load(self) -> None:
        """Open the ONNX session and run one warm-up inference.

        Raises:
            TurnModelError: model file missing, onnxruntime missing, or the
                session failed to build.
        """
        path = self._path or model_path()
        if not path:
            raise TurnModelError(
                f"Smart Turn model not downloaded ({MODEL_FILE}, ~32 MB). Run:  "
                "python -c 'from wawi.turn import download_model; download_model()'"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise TurnModelError("onnxruntime is not installed") from exc
        try:
            so = ort.SessionOptions()
            # Same session shape as upstream's inference.py.
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = self._threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                path, sess_options=so, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._path = path
            # The first run builds kernels and is several times slower; pay it
            # here, not on the user's first pause.
            self.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))
        except TurnModelError:
            raise
        except Exception as exc:
            self._session = None
            raise TurnModelError(f"Could not load the Smart Turn model {path}: {exc}") from exc

    def predict(self, audio: np.ndarray) -> float:
        """Return P(complete) in 0..1 for 16 kHz mono float32 *audio*."""
        if self._session is None:
            raise TurnModelError("Smart Turn model is not loaded")
        feats = whisper_features(audio)
        out = self._session.run(None, {self._input_name: feats})
        # The ONNX graph ends in a sigmoid: this is already a probability.
        return float(np.asarray(out[0]).reshape(-1)[0])

    def timed_predict(self, audio: np.ndarray) -> tuple[float, float]:
        """``predict`` plus its wall time in milliseconds."""
        t0 = time.perf_counter()
        p = self.predict(audio)
        return p, (time.perf_counter() - t0) * 1000.0
