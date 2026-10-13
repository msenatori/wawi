"""Speech-to-text transcription using mlx-whisper.

mlx-whisper runs on Apple's MLX framework, whose GPU streams and lazy arrays are
THREAD-LOCAL (mlx >= 0.31.2). Calling the model from a different thread than the
one that created its stream/weights raises:

    RuntimeError: There is no Stream(gpu, N) in current thread

The orchestrator runs every turn on a fresh worker thread, so we must NOT let
mlx-whisper touch those threads directly. Instead, STT owns a single long-lived
worker thread (a 1-thread executor): the model is loaded there on first use and
every transcription runs there too, so all MLX work stays on ONE consistent
thread. Callers (main-thread warm-up, per-turn worker threads) just submit work
and block for the result.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from wawi.speechgate import SttMetrics

log = logging.getLogger(__name__)


class STT:
    """Thin wrapper around mlx_whisper for single-language transcription.

    All mlx-whisper calls are funneled onto a dedicated single worker thread to
    satisfy MLX's thread-local stream/array constraints.

    Args:
        model: Path or HuggingFace repo ID passed to mlx_whisper.transcribe().
               Default: "mlx-community/whisper-large-v3-turbo".
    """

    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo") -> None:
        self._model = model
        # Single dedicated MLX thread: model load + every transcription run here
        # so MLX's thread-local GPU stream and lazy weights never cross threads.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-stt")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "es",
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe *audio* (float32, 16 kHz, mono) and return the text.

        Safe to call from any thread: the work is dispatched to STT's dedicated
        MLX thread and this call blocks until the result is ready.

        Args:
            audio: Float32 numpy array at 16 000 Hz, mono.
            language: BCP-47 language code. Defaults to "es" (Spanish).
            initial_prompt: Text the decoder is seeded with. Whisper continues
                the STYLE of this text, so it is how orthography and unusual
                vocabulary are steered -- see :data:`wawi.lang.STT_PROMPT`.
                None leaves the decoder unseeded, i.e. the previous behaviour.

        Returns:
            Stripped transcription string. Empty string if nothing was spoken.
        """
        return self.transcribe_detailed(audio, language, initial_prompt)[0]

    def transcribe_detailed(
        self,
        audio: np.ndarray,
        language: str = "es",
        initial_prompt: str | None = None,
    ) -> tuple[str, SttMetrics | None]:
        """Like :meth:`transcribe`, plus Whisper's own confidence.

        The metrics are what :mod:`wawi.speechgate` uses to tell words from
        a hallucination. None when the result carried no usable segments.
        """
        return self._executor.submit(
            self._transcribe, audio, language, initial_prompt
        ).result()

    def _transcribe(
        self, audio: np.ndarray, language: str, initial_prompt: str | None
    ) -> tuple[str, SttMetrics | None]:
        """Run the actual mlx-whisper transcription on the dedicated MLX thread."""
        import mlx_whisper  # type: ignore[import]

        # Passed only when set: an empty prompt is not the same as no prompt,
        # and older mlx-whisper builds are happier never seeing the argument.
        extra = {"initial_prompt": initial_prompt} if initial_prompt else {}
        result = mlx_whisper.transcribe(
            # Plain ndarray: the listener hands over a subclass carrying its
            # speech measurements, and MLX has no use for those.
            np.asarray(audio, dtype=np.float32),
            path_or_hf_repo=self._model,
            language=language,
            **extra,
        )
        if not isinstance(result, dict):
            return "", None
        text: str = result.get("text", "") or ""
        return text.strip(), SttMetrics.from_segments(result.get("segments") or [])
