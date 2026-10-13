#!/usr/bin/env python3
"""Standalone visual spike for the Wawi floating HUD panel.

Run it with::

    .venv/bin/python scripts/spike_panel.py

What it demonstrates:
  * A frameless, rounded, translucent always-on-top panel.
  * A live FFT spectrum driven by the real microphone (sounddevice).
  * State-driven colours: LISTENING = cool blue, SPEAKING = warm amber,
    THINKING = violet pulse, IDLE = flat and dim. A background thread cycles
    the states, which also proves the observer callback is safe to call from a
    worker thread (exactly how ``StateMachine.set_state`` invokes it).
  * The debug strip listing watched projects.
  * The push-to-talk toggle (bottom-right). Click it: the new value is printed
    along with a focus check taken from inside the click handler itself, so you
    can confirm the click did not make the panel key. ON = push-to-talk,
    OFF = always-listening (wake word).
  * That the panel NEVER becomes the key window — the focus check is printed
    on start and re-printed every few seconds while you drag the window
    around. Keep typing in your editor: the caret must never move here.

Drag the panel anywhere (including onto a second monitor); the saved origin is
printed on exit so you can see what a caller would persist.

Press Ctrl+C in this terminal to quit.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np

# Make the repo root importable when run directly from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wawi.panel import PanelObserver, appkit_available, run_event_loop  # noqa: E402
from wawi.state import State  # noqa: E402

SAMPLE_RATE = 16000
BLOCK_FRAMES = 1024  # ~64 ms at 16 kHz — one FFT block per render frame


class LiveMic:
    """Continuously capture the microphone, keeping only the latest block.

    Deliberately standalone (it does not use ``wawi.audio_input.Recorder``)
    so this spike runs without the rest of Wawi. A real integration would add
    an equivalent ``current_block()`` accessor to ``Recorder``.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._latest: np.ndarray | None = None
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ANN001, ARG001
            if status:
                logging.debug("sounddevice status: %s", status)
            # Copy: sounddevice reuses the underlying buffer.
            self._latest = indata[:, 0].copy()

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_FRAMES,
            callback=_callback,
        )
        self._stream.start()
        print(f"[spike] microphone open at {self._sample_rate} Hz")

    def latest(self) -> "np.ndarray | None":
        """Return the most recent audio block, or None if nothing captured."""
        return self._latest

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logging.warning("Error closing the mic stream", exc_info=True)
            self._stream = None


# Demo state cycle: (state, seconds to hold it)
_CYCLE: list[tuple[State, float]] = [
    (State.LISTENING, 5.0),   # cool blue  — "user speaking"
    (State.THINKING, 2.5),    # violet pulse
    (State.SPEAKING, 5.0),    # warm amber — "Wawi speaking"
    (State.IDLE, 2.5),        # flat / dim
]


def _cycle_states(panel: PanelObserver, stop_event: threading.Event) -> None:
    """Drive the panel through the state cycle from a worker thread."""
    previous = State.IDLE
    while not stop_event.is_set():
        for state, hold in _CYCLE:
            if stop_event.is_set():
                return
            # Same call shape as StateMachine.set_state -> observer(old, new).
            panel(previous, state)
            previous = state
            print(f"[spike] state -> {state.value:<9}  key_window={panel.is_key_window()}")
            stop_event.wait(hold)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not appkit_available():
        print("[spike] AppKit/PyObjC is unavailable — this spike requires macOS.")
        return 1

    panel = PanelObserver(enabled=True)
    mic = LiveMic()

    try:
        mic.start()
    except Exception as exc:
        print(f"[spike] could not open the microphone ({exc}); bars will idle-animate.")
        print("[spike] grant Microphone permission to your terminal in System Settings.")

    panel.set_samples_source(mic.latest)
    panel.set_sample_rate(SAMPLE_RATE)
    panel.set_watched([("blog", "▶"), ("tienda", "✓")])

    # Push-to-talk toggle. The callback runs on the AppKit main thread, so it
    # must stay fast — here it only prints. A real caller hops to a worker.
    def _on_ptt(on: bool) -> None:
        mode = "PUSH-TO-TALK (hold the key, wake word ignored)" if on else \
               "ALWAYS-LISTENING (wake word drives turns)"
        print(f"[spike] toggle clicked -> push_to_talk={on}  |  {mode}")
        print(f"[spike]   key_window={panel.is_key_window()} (must stay False)")

    panel.on_push_to_talk_changed(_on_ptt)
    panel.set_push_to_talk(True)

    # start() is non-blocking and must run on the main thread.
    panel.start()

    if not panel.enabled:
        print("[spike] the panel failed to start; see the log above.")
        mic.stop()
        return 1

    print(f"[spike] panel visible. is_key_window={panel.is_key_window()} (must be False)")
    print(f"[spike] origin={panel.get_origin()}")
    print(f"[spike] push_to_talk={panel.push_to_talk()}")
    print("[spike] CLICK the 'push to talk' switch (bottom-right) to flip it.")
    print("[spike] drag it anywhere, keep typing in your editor, Ctrl+C here to quit.")

    stop_event = threading.Event()
    cycler = threading.Thread(
        target=_cycle_states,
        args=(panel, stop_event),
        daemon=True,
        name="spike-state-cycler",
    )
    cycler.start()

    def _focus_watchdog() -> None:
        while not stop_event.wait(5.0):
            if panel.is_key_window():
                print("[spike] FAIL: the panel became the key window!")

    threading.Thread(target=_focus_watchdog, daemon=True, name="spike-focus").start()

    try:
        # Parks the main thread in NSApp.run(); Ctrl+C unwinds it.
        run_event_loop()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        origin = panel.get_origin()
        panel.stop()
        mic.stop()
        print(f"\n[spike] final origin (a caller would persist this): {origin}")
        print("[spike] stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
