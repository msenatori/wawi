"""Push-to-talk audio capture using sounddevice.

The canonical interface is the Recorder class. The convenience wrapper
record_while_held() is provided for simple scripts and tests.

Recording produces a float32 mono numpy array at the configured sample rate.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_CHUNK_FRAMES = 1024  # frames per InputStream callback (≈64 ms at 16 kHz)


class Recorder:
    """Push-to-talk audio recorder.

    Usage:
        recorder = Recorder(sample_rate=16000)
        recorder.start()   # called on key-press
        ...
        buf = recorder.stop()  # called on key-release; returns float32 mono array
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Open the microphone stream and begin buffering audio frames.

        Called on key-press. Idempotent if called when already recording.
        """
        with self._lock:
            if self._stream is not None:
                return  # already recording

            self._frames = []

            def _callback(
                indata: np.ndarray,
                frames: int,  # noqa: ARG001
                time,  # noqa: ARG001
                status: sd.CallbackFlags,
            ) -> None:
                if status:
                    log.debug("sounddevice callback status: %s", status)
                # indata shape: (frames, channels) — copy to avoid buffer reuse
                with self._lock:
                    self._frames.append(indata[:, 0].copy())

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=_CHUNK_FRAMES,
                callback=_callback,
            )
            self._stream.start()
            log.debug("Recorder: stream started at %d Hz", self._sample_rate)

    def current_level(self) -> float:
        """Return the recent peak amplitude of the most recently buffered chunk.

        Thread-safe. Returns 0.0 when not recording or no frames have been
        captured yet. The returned value is in the range 0.0..~1.0 (float32
        audio can theoretically clip beyond 1.0, but in practice stays below).
        """
        with self._lock:
            if not self._frames:
                return 0.0
            last_chunk = self._frames[-1]
        # Compute outside the lock — we already have a copy
        return float(np.abs(last_chunk).max())

    def current_block(self) -> np.ndarray | None:
        """Return the most recently buffered audio chunk, or None.

        Feeds the panel's spectrum visualiser, which needs raw samples rather
        than the single scalar :meth:`current_level` provides. Thread-safe;
        returns None when not recording or before the first chunk arrives.
        """
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def stop(self) -> np.ndarray:
        """Stop buffering, close the stream, and return the captured audio.

        Called on key-release. Returns a float32 mono array (possibly empty if
        stop() is called before start() or if no audio was captured).

        Returns:
            Float32 numpy array, shape (N,), at self._sample_rate Hz.
        """
        with self._lock:
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.warning("Error closing audio stream", exc_info=True)

        with self._lock:
            frames = list(self._frames)
            self._frames = []

        if frames:
            buf = np.concatenate(frames).astype(np.float32)
        else:
            buf = np.zeros(0, dtype=np.float32)

        log.debug("Recorder: captured %d samples (%.2f s)", len(buf), len(buf) / self._sample_rate)
        return buf


def record_while_held(
    key: str,
    sample_rate: int = 16000,
    on_press: Callable[[], None] | None = None,
    on_release: Callable[[], None] | None = None,
) -> np.ndarray:
    """Convenience wrapper: block until *key* is released; return the recording.

    The pynput listener is created internally. This is useful for simple scripts
    and tests. Production code uses Recorder directly via orchestrator callbacks.

    Args:
        key: Key name recognised by pynput (e.g. "ctrl_r", "f9").
        sample_rate: Recording sample rate in Hz.
        on_press: Optional callback fired synchronously when the key is pressed.
        on_release: Optional callback fired synchronously when the key is released.

    Returns:
        Float32 mono numpy array at *sample_rate* Hz.
    """
    from pynput import keyboard  # type: ignore[import]

    recorder = Recorder(sample_rate=sample_rate)
    done = threading.Event()

    target_key = _parse_key(key)

    def _press(k) -> None:  # type: ignore[no-untyped-def]
        if _key_matches(k, target_key):
            recorder.start()
            if on_press is not None:
                on_press()

    def _release(k) -> None:  # type: ignore[no-untyped-def]
        if _key_matches(k, target_key):
            if on_release is not None:
                on_release()
            done.set()
            return False  # stop listener

    with keyboard.Listener(on_press=_press, on_release=_release):
        done.wait()

    return recorder.stop()


# ---------------------------------------------------------------------------
# Key parsing helpers
# ---------------------------------------------------------------------------

def _parse_key(name: str):  # type: ignore[no-untyped-def]  # returns pynput Key or KeyCode
    """Parse a string key name into a pynput Key or KeyCode object."""
    from pynput import keyboard  # type: ignore[import]

    # Try special key names (e.g. "ctrl_r", "f9", "space")
    key_attr = name.lower().replace(" ", "_")
    key_obj = getattr(keyboard.Key, key_attr, None)
    if key_obj is not None:
        return key_obj
    # Fallback: single character KeyCode
    return keyboard.KeyCode.from_char(name)


def _key_matches(pressed, target) -> bool:
    """Return True if *pressed* matches *target*."""
    return pressed == target
