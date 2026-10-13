"""Always-on-top floating HUD panel for Wawi (macOS / PyObjC / AppKit).

The panel is a small frameless, translucent, draggable window whose primary
element is a real-time audio spectrum visualiser (FFT bars). Below the spectrum
sits a one-line debug strip listing the projects Wawi is currently watching.

Public interface (mirrors ``wawi.ui.TuiObserver`` — when ``enabled=False``
every method is a no-op)::

    PanelObserver(enabled=True)
    .start()
    .stop()
    .__call__(old, new)             — state observer
    .set_samples_source(source)     — Callable[[], np.ndarray | None]
    .set_capture_source(source)     — Callable[[], bool], True while recording
    .get_origin()                   — tuple[float, float] | None
    .set_origin(x, y)
    .listening()                    — bool, is the microphone gate open
    .set_listening(enabled)         — set it programmatically
    .on_listening_changed(cb)       — callback fired on user clicks

===========================================================================
RUN-LOOP INTEGRATION — findings
===========================================================================

The problem as originally posed was: "AppKit needs an NSApplication run loop on
the main thread, but the orchestrator parks the main thread on
``threading.Event().wait()`` AND pynput also needs the macOS main run loop."

Investigation result: **the pynput half of that conflict does not exist.**
There is only one real constraint, and it is easy to satisfy.

--- Finding 1: pynput does NOT need the main thread or the main run loop -----

``pynput.keyboard.Listener`` derives from ``pynput._util.AbstractListener``,
which *is itself a* ``threading.Thread`` (``_util/__init__.py:110``) created
with ``daemon = True`` (``_util/__init__.py:167``). Calling ``.start()`` — or
entering it as a context manager — spawns a **new thread**.

On macOS, that thread runs ``pynput._util.darwin.ListenerMixin._run``
(``_util/darwin.py:215``), which does::

    tap         = self._create_event_tap()
    loop_source = CFMachPortCreateRunLoopSource(None, tap, 0)
    self._loop  = CFRunLoopGetCurrent()          # <-- line 231
    CFRunLoopAddSource(self._loop, loop_source, kCFRunLoopDefaultMode)
    CGEventTapEnable(tap, True)
    while self.running:
        result = CFRunLoopRunInMode(kCFRunLoopDefaultMode, 1, False)

``CFRunLoopGetCurrent()`` returns the run loop of **the calling thread**, and
every thread gets its own CFRunLoop on first request. So pynput creates and
pumps a *private* CFRunLoop on its *own* listener thread. It never touches
``NSApp``, never calls ``CFRunLoopGetMain()``, and never requires the main
thread's run loop to be running.

Consequence: the main thread is completely free. pynput and AppKit do not
compete for it. (The 1-second timeout in ``CFRunLoopRunInMode`` is just how
pynput polls its own ``running`` flag so ``stop()`` is responsive.)

--- Finding 2: AppKit genuinely does need the main thread -------------------

This constraint is real and non-negotiable. NSApplication/NSWindow must be
created and driven from the process's first thread, and ``NSApp.run()`` must
own it. There is no supported way to host AppKit on a worker thread.

--- Finding 3: therefore the fix is a one-line swap in the orchestrator -----

``wawi/orchestrator.py:152`` currently parks the main thread on
``threading.Event().wait()`` purely to keep the process alive while the pynput
daemon thread does the real work. That park is *dead weight* — it can simply
be replaced with the AppKit run loop, which parks the main thread just as
effectively while also servicing the panel.

    WHAT THE CALLER (orchestrator) MUST CHANGE
    ------------------------------------------
    In ``Orchestrator.run()``:

    1. Build the panel and subscribe it BEFORE any state transition:

           self.panel = PanelObserver(cfg.show_panel)
           self.state.subscribe(self.panel)
           self.panel.set_samples_source(self.recorder.current_block)
           # optional, restores the saved two-monitor position
           if saved_origin: self.panel.set_origin(*saved_origin)

    2. Call ``self.panel.start()`` on the MAIN thread. It is non-blocking:
       it creates NSApp + the window + the 30 fps render timer and returns.

    3. Replace the park at orchestrator.py:152::

           -    threading.Event().wait()
           +    panel.run_event_loop()      # from wawi.panel

       ``run_event_loop()`` is this module's blocking main-thread parker. It
       calls ``NSApp.run()`` when the panel is enabled and falls back to
       ``threading.Event().wait()`` when it is disabled or when AppKit is
       unavailable — so the non-macOS / panel-off path behaves exactly as it
       does today.

    4. On shutdown call ``self.panel.stop()`` next to ``self.tui.stop()``.

    Nothing else moves. The pynput listener still runs on its own daemon
    thread, and ``_do_turn`` worker threads are untouched.

--- Finding 4: two gotchas that the naive version gets wrong ----------------

(a) **Ctrl+C stops working.** ``NSApp.run()`` is a blocking C call, and CPython
    only runs signal handlers between bytecodes on the main thread. With the
    main thread inside ``NSApp.run()``, SIGINT is recorded but not delivered
    until Python code next executes. This module fixes it two ways: the 30 fps
    render timer re-enters Python every ~33 ms (giving the handler a chance to
    run), and ``run_event_loop()`` installs a SIGINT handler that calls
    ``NSApp.stop_()`` so Ctrl+C unwinds the loop cleanly.

(b) **The spectrum freezes while the window is dragged.** A plain
    ``scheduledTimer`` is registered only in ``NSDefaultRunLoopMode``; during a
    window drag or menu tracking the run loop switches to
    ``NSEventTrackingRunLoopMode`` and the timer stops firing. The timer is
    therefore explicitly added to ``NSRunLoopCommonModes``.

--- Alternatives considered and rejected -----------------------------------

* *Panel in a separate process over a pipe/socket.* Unnecessary: findings 1-3
  show there is no actual conflict to work around. It would add IPC, audio
  block serialisation at 30 fps, and a second process to supervise.
* *Moving the orchestrator's park to a worker thread.* Nothing to move — the
  park does no work; it is replaced, not relocated.
* *Pumping AppKit manually with ``nextEventMatchingMask``.* Only needed if some
  other library owned the main loop. Nothing does, so plain ``NSApp.run()`` is
  simpler and better behaved.

===========================================================================
FOCUS BEHAVIOUR (hard requirement — the panel must never steal key focus)
===========================================================================

Three independent mechanisms, all required:

1. ``NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)`` —
   the process becomes a background/agent app: no Dock icon, no menu bar, and
   it is never made the active application.
2. ``NSPanel`` with ``NSWindowStyleMaskNonactivatingPanel`` — clicking the
   panel (including click-and-drag) does not activate the owning application.
3. ``orderFrontRegardless()`` instead of ``makeKeyAndOrderFront_`` — shows the
   window without ever asking for key status. ``canBecomeKeyWindow`` and
   ``canBecomeMainWindow`` are additionally overridden to return ``False``, so
   even a programmatic request cannot make it key.

``PanelObserver.is_key_window()`` exposes the live answer for verification.

--- The push-to-talk toggle and focus ---------------------------------------

Adding a clickable control is the one change that could plausibly break the
guarantee above, so the switch is hand-drawn in ``WawiToggleView`` rather
than being an ``NSButton``/``NSSwitch``:

* A real AppKit control asks to become first responder when clicked, and
  renders in its inactive style inside a window that is never key.
* ``WawiToggleView.acceptsFirstResponder`` returns False, so focus is never
  requested; the click is handled purely as a mouse event.
* ``acceptsFirstMouse_`` returns True. This is REQUIRED, not optional: for a
  non-key window AppKit normally consumes the first click just to activate the
  window instead of delivering it. Since this window can never become key,
  every click is a first-mouse click — without this the switch would be dead.
* The view's frame is exactly its hit area, so ``hitTest_`` claims only the
  switch. Clicks anywhere else fall through to the background and
  ``setMovableByWindowBackground_`` dragging is untouched. ``mouseDown_`` /
  ``mouseUp_`` / ``mouseDragged_`` never chain to super, so dragging *from* the
  switch does not move the window either.

Net effect: clicking the toggle changes no window state at all — not key, not
main, not active.
"""

from __future__ import annotations

import logging
import math
import signal
import threading
import time
from typing import Any, Callable

import numpy as np

from wawi import mascot as _mascot
from wawi.state import State

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometry / rendering constants
# ---------------------------------------------------------------------------

PANEL_WIDTH = 460.0

# The spectrum is a FIXED strip, not an elastic one. It used to absorb every
# point the user added by dragging, so growing the panel grew the audio meter
# and left the text exactly as cramped as before — which is the opposite of
# why anyone resizes it. Height added now goes to the transcript, all of it.
_SPECTRUM_H = 52.0
CORNER_RADIUS = 16.0

_TITLE_H = 26.0          # top row: window buttons + name + state
_STRIP_H = 20.0          # bottom row: language pill and the listening switch

#: Collapsed height: title + spectrum + strip, and nothing else. Everything
#: above this is transcript, which makes "how tall is the text" a subtraction
#: rather than a second layout rule that can disagree with the first.
PANEL_HEIGHT = _TITLE_H + _SPECTRUM_H + _STRIP_H
# The transcript block: one dim row for what Wawi HEARD, two for what it
# answered. The heard line earns its row — a wrong answer is usually a
# mistranscription, and without it the HUD shows a confident reply to a
# question you never asked.
# The transcript band grows and shrinks with the ANSWER. A fixed six-row band
# reserved the worst case on every turn, so a two-line answer left half the
# panel as empty grey — the HUD looked broken rather than idle.
# The cap only applies while the panel sizes ITSELF. Once the user drags it to
# a size, that size wins and the row budget comes from the space they gave it.
_TRANSCRIPT_MAX_ROWS = 8
_TRANSCRIPT_ROW_H = 17.0
_TRANSCRIPT_PAD = 8.0
_Q_FONT = 10.5    # what was heard: present, secondary
_A_FONT = 13.0    # the answer: the reason the panel exists

# Minimum the user may drag the window down to. Below this the title, the
# spectrum and the strip start overlapping and the HUD stops being readable.
_MIN_WIDTH = 320.0
_MIN_HEIGHT = PANEL_HEIGHT

# The drag handle, bottom-right corner.
_GRIP = 16.0

# Wawi's status dot (see WawiMascotView._paint_badge): diameter, and the
# same green the spectrum uses for "recording".
_MASCOT_BADGE_D = 7.0
_MASCOT_BADGE_LISTENING = (0.30, 0.85, 0.45)

# How many past exchanges are kept. Bounded because they are only ever drawn
# into a window of finite height — keeping more would cost memory to render
# nothing.
_HISTORY_MAX = 40

# Traffic lights, top-left, in the macOS order and the macOS colours. They are
# hand-drawn for the same reason every other control here is: this window can
# never become key, so the real ones would sit permanently in their inactive
# style and try to take first responder on click.
_DOT_R = 6.0        # diameter
_DOT_GAP = 8.0
_DOT_CLOSE = (1.00, 0.37, 0.34)
_DOT_MIN = (1.00, 0.74, 0.20)
_DOT_MAX = (0.31, 0.78, 0.25)
_BUTTONS_W = 3 * _DOT_R + 2 * _DOT_GAP + 6.0

# Frames of the delegated-agent spinner in the bottom strip. Braille dots
# rather than a spinning slash: they are one glyph wide in every fallback face,
# so the progress line beside them never shifts by a pixel per frame.
_TASK_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# What "zoom" grows to, clamped to the screen at click time.
_ZOOM_W = 900.0
_ZOOM_H = 700.0


def _transcript_height(rows: int) -> float:
    """Band height for *rows* drawn lines, or 0 when there is nothing to draw."""
    if rows <= 0:
        return 0.0
    return rows * _TRANSCRIPT_ROW_H + _TRANSCRIPT_PAD

# The disclosure control, top-right. Collapsing does not merely hide the text:
# the WINDOW shrinks back to PANEL_HEIGHT, because a HUD that keeps the space
# it is not using is just a bigger HUD.
_CHEVRON_BOX = 22.0

# The language pill, bottom-LEFT of the strip (push-to-talk owns the right).
# Two letters rather than a flag: a flag names a country, and the choice here
# is a language.
_LANG_BOX_W = 34.0
_LANG_BOX_H = 16.0

# "Copy the answer", next to the language pill. An answer worth keeping that
# you can only re-read on a HUD is information trapped in a window.
_COPY_BOX_W = 52.0

#: How long the pill says "copiado" before going back. Long enough to read,
#: short enough that it is never ambiguous about WHICH copy it is confirming.
_COPIED_FEEDBACK_S = 1.4

_SIDE_PAD = 14.0
_BAR_COUNT = 20          # number of spectrum bars
_BAR_GAP = 3.0           # horizontal gap between bars, in points
_BAR_MIN_H = 2.0         # bars never fully vanish, so the HUD reads as "alive"

# Spectrum colours while IDLE. Height always reports the true level, so the
# distinction between "monitoring the room" and "recording your words" lives
# here instead — a level meter must never lie about the level.
_MONITOR_COLOR = (0.42, 0.47, 0.58)   # muted slate: awake, listening, not recording
_CAPTURE_COLOR = (0.36, 0.78, 0.98)   # live cyan: these words are being kept

# Render ticks without audio before the microphone is treated as dead. At 30
# fps this is ~0.7 s: long enough to ride out a gap between blocks, short
# enough that a denied or closed mic shows up immediately.
_MIC_DEAD_TICKS = 20

_FPS = 30.0
#: Wawi's round face in the title row, in points.
_FACE_PT = 15.0
_FRAME_INTERVAL = 1.0 / _FPS

# Push-to-talk toggle, laid out at the right end of the bottom strip row.
# The whole control lives in its own subview so that its hit area — and ONLY
# its hit area — swallows clicks; everywhere else keeps window-background drag.
# The switch is the microphone gate now: ON listens, OFF is deaf. It used to
# mean "push-to-talk only", which was a mode SELECTOR — and a selector whose
# label names one of the modes cannot say which one is active.
_TOGGLE_LABEL = "escuchando"
_TOGGLE_BOX_W = 100.0    # reserved width: label + gap + switch
_TOGGLE_BOX_H = 16.0
_SWITCH_W = 26.0
_SWITCH_H = 13.0
_SWITCH_KNOB = 9.0
_SWITCH_GAP = 6.0        # gap between the label and the switch
_TOGGLE_ON_RGB = (0.29, 0.68, 1.00)   # lit track when push-to-talk is ON

# Spectrum band edges (Hz). 16 kHz capture => 8 kHz Nyquist.
_FMIN = 60.0
_FMAX = 7000.0

# Smoothing: bars rise fast and fall slowly, so the display does not flicker.
_ATTACK = 0.55           # weight of the new value when it is louder
_DECAY = 0.14            # weight of the new value when it is quieter

# Magnitude normalisation. Raw FFT magnitudes are tiny; this maps a typical
# speaking level onto roughly the full bar height.
_DB_FLOOR = -62.0
_DB_CEIL = -14.0

# Per-state bar colours as (r, g, b) in 0..1.
_COLORS: dict[State, tuple[float, float, float]] = {
    State.IDLE: (0.45, 0.49, 0.58),        # flat / dim slate
    State.LISTENING: (0.29, 0.68, 1.00),   # cool blue  — the user is speaking
    State.THINKING: (0.72, 0.55, 0.98),    # violet     — idle pulse animation
    State.SPEAKING: (1.00, 0.68, 0.24),    # warm amber — Wawi is speaking
}

_LABELS: dict[State, str] = {
    State.IDLE: "idle",
    State.LISTENING: "listening",
    State.THINKING: "thinking",
    State.SPEAKING: "speaking",
}


# ---------------------------------------------------------------------------
# Spectrum analysis (pure numpy — no AppKit, independently testable)
# ---------------------------------------------------------------------------


def _band_edges(n_bars: int, sample_rate: int, n_bins: int) -> list[tuple[int, int]]:
    """Return ``n_bars`` (start_bin, stop_bin) pairs on a log frequency scale.

    Each returned band is guaranteed to be non-empty (``stop > start``) and to
    stay inside ``n_bins``, so callers can slice without extra guards.
    """
    if n_bins <= 1:
        return [(0, 1)] * n_bars

    nyquist = sample_rate / 2.0
    fmax = min(_FMAX, nyquist * 0.98)
    fmin = min(_FMIN, fmax / 2.0)

    hz_per_bin = nyquist / float(n_bins - 1) if n_bins > 1 else 1.0
    log_min, log_max = math.log10(fmin), math.log10(fmax)

    edges: list[tuple[int, int]] = []
    prev_stop = 1  # skip bin 0 (DC)
    for i in range(n_bars):
        hi_hz = 10.0 ** (log_min + (log_max - log_min) * (i + 1) / n_bars)
        stop = int(hi_hz / hz_per_bin) + 1
        stop = max(prev_stop + 1, min(stop, n_bins))
        start = min(prev_stop, stop - 1)
        edges.append((start, stop))
        prev_stop = stop
        if prev_stop >= n_bins:
            prev_stop = n_bins - 1
    return edges


def compute_bars(
    samples: np.ndarray,
    n_bars: int = _BAR_COUNT,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Compute ``n_bars`` normalised (0..1) log-spaced spectrum magnitudes.

    Args:
        samples: Mono float32 audio block. May be any length; empty or
            all-silent input yields an all-zero result.
        n_bars: Number of output bands.
        sample_rate: Sample rate of *samples* in Hz.

    Returns:
        Float32 array of shape ``(n_bars,)`` with values clamped to 0..1.
    """
    out = np.zeros(n_bars, dtype=np.float32)
    if samples is None:
        return out

    buf = np.asarray(samples, dtype=np.float32).ravel()
    if buf.size < 32 or not np.isfinite(buf).all():
        return out

    # Hann window suppresses spectral leakage between neighbouring bands.
    window = np.hanning(buf.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(buf * window))
    if spectrum.size <= 1:
        return out

    # Normalise by window energy so bar height is independent of block length.
    spectrum /= max(float(window.sum()), 1e-9) / 2.0

    for i, (start, stop) in enumerate(_band_edges(n_bars, sample_rate, spectrum.size)):
        band = spectrum[start:stop]
        if band.size == 0:
            continue
        magnitude = float(band.max())
        db = 20.0 * math.log10(magnitude + 1e-9)
        out[i] = (db - _DB_FLOOR) / (_DB_CEIL - _DB_FLOOR)

    np.clip(out, 0.0, 1.0, out=out)
    return out


def _idle_bars(n_bars: int, phase: float, amplitude: float) -> np.ndarray:
    """Return a smooth travelling-wave bar pattern for non-audio states."""
    idx = np.arange(n_bars, dtype=np.float32)
    wave = np.sin(idx * 0.55 - phase * 2.4) * 0.5 + 0.5
    # Taper the edges so the pulse reads as centred rather than clipped.
    taper = np.sin(np.pi * (idx + 0.5) / n_bars).astype(np.float32)
    return (wave * taper * amplitude).astype(np.float32)


def _text_attrs(size: float, rgb, alpha: float, bold: bool = False):
    """Monospaced attributes. AppKit-only; callers must have imported it."""
    font = NSFont.monospacedSystemFontOfSize_weight_(size, 0.4 if bold else 0.0)
    r, g, b = rgb
    return {
        NSFontAttributeName: font,
        NSForegroundColorAttributeName: (
            NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, alpha)
        ),
    }


def wrap_lines(text: str, attrs, max_w: float, max_lines: int) -> list[str]:
    """Greedy word wrap, measured in real glyphs rather than characters.

    Module level on purpose: the VIEW uses it to draw and the OBSERVER uses it
    to decide how tall the window must be. Two implementations of this would
    drift, and the symptom would be a panel whose last line is cut off by its
    own frame.

    Character counting is not good enough even in a monospaced font — the
    answers carry accents and «» that a fallback face may render at another
    width, and an overflowing line is drawn straight over the panel edge with
    no clipping to save it.
    """
    words = (text or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        probe = NSAttributedString.alloc().initWithString_attributes_(candidate, attrs)
        if probe.size().width <= max_w or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)

    # Say so when text was dropped: a silently truncated answer reads as a
    # complete one, which is worse than showing less.
    if len(lines) == max_lines and len(" ".join(lines).split()) < len(words):
        lines[-1] = lines[-1] + " …"
    return lines


def _transcript_lines(question: str, answer: str, max_w: float, budget: int | None = None):
    """Wrap one exchange into (question_lines, answer_lines).

    The question gets exactly one line — it is context, not content — and the
    answer gets whatever is left of *budget* rows (the self-sizing maximum when
    none is given).
    """
    if budget is None:
        budget = _TRANSCRIPT_MAX_ROWS
    q_lines: list[str] = []
    a_lines: list[str] = []
    if question:
        q_lines = wrap_lines(
            f"› {question}", _text_attrs(_Q_FONT, (1, 1, 1), 1.0), max_w, 1
        )
    if answer:
        a_lines = wrap_lines(
            answer,
            _text_attrs(_A_FONT, (1, 1, 1), 1.0),
            max_w,
            max(1, budget - len(q_lines)),
        )
    return q_lines, a_lines


# ---------------------------------------------------------------------------
# AppKit layer
# ---------------------------------------------------------------------------

_APPKIT_ERROR: str | None = None

try:  # pragma: no cover - import guard exercised only on non-macOS
    import objc  # type: ignore[import]
    from AppKit import (  # type: ignore[import]
        NSApplication,
        NSPasteboard,
        NSPasteboardTypeString,
        NSApplicationActivationPolicyAccessory,
        NSAttributedString,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSColor,
        NSEventModifierFlagControl,
        NSFloatingWindowLevel,
        NSNormalWindowLevel,
        NSAffineTransform,
        NSCompositingOperationSourceOver,
        NSFont,
        NSFontAttributeName,
        NSGraphicsContext,
        NSImage,
        NSImageLeft,
        NSForegroundColorAttributeName,
        NSMakePoint,
        NSMakeRect,
        NSMenu,
        NSMenuItem,
        NSPanel,
        NSScreen,
        NSStatusBar,
        NSVariableStatusItemLength,
        NSView,
        NSAppearance,
        NSAppearanceNameDarkAqua,
        NSVisualEffectBlendingModeBehindWindow,
        NSVisualEffectMaterialHUDWindow,
        NSVisualEffectStateActive,
        NSVisualEffectView,
        NSViewMaxYMargin,
        NSViewMinXMargin,
        NSViewMinYMargin,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
        NSWorkspace,
    )
    from Foundation import (  # type: ignore[import]
        NSObject,
        NSPointInRect,
        NSRunLoop,
        NSRunLoopCommonModes,
        NSTimer,
    )

    _HAS_APPKIT = True
except Exception as exc:  # pragma: no cover
    _HAS_APPKIT = False
    _APPKIT_ERROR = str(exc)


def appkit_available() -> bool:
    """Return True when PyObjC/AppKit could be imported in this process."""
    return _HAS_APPKIT


if _HAS_APPKIT:

    class WawiSpectrumView(NSView):  # type: ignore[misc]
        """Custom view drawing the title row, spectrum bars and debug strip.

        All drawing happens on the main thread inside ``drawRect_``. The view
        reads plain attributes off the owning :class:`PanelObserver`; the
        worker threads only ever *write* those attributes, which is safe under
        the GIL for the single assignments used here (same contract as
        ``TuiObserver``).
        """

        def initWithFrame_owner_(self, frame, owner):  # noqa: N802
            self = objc.super(WawiSpectrumView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._owner = owner
            return self

        # The panel must never intercept the caret: it is draggable via
        # setMovableByWindowBackground_, which needs no mouse handling here.
        def isFlipped(self):  # noqa: N802
            return False

        # -- drawing ----------------------------------------------------

        def drawRect_(self, rect):  # noqa: N802
            """Draw one frame, and NEVER let an exception escape.
        
            AppKit calls +[NSApplication _crashOnException:] when anything is
            raised out of a drawRect_, which kills the whole process — for a
            background assistant that means it vanishes with nothing in the
            log. A drawing bug should cost one frame, not the microphone, the
            session and the conversation. The traceback is logged instead, so
            the next occurrence is diagnosable rather than fatal.
            """
            try:
                self._paint(rect)
            except Exception:
                log.exception('Panel drawing failed; skipping this frame')

        @objc.python_method
        def _paint(self, rect):  # noqa: N802, ARG002
            owner = self._owner
            bounds = self.bounds()
            width = bounds.size.width
            height = bounds.size.height

            state = owner._current_state
            color = _COLORS.get(state, _COLORS[State.IDLE])
            if state is State.IDLE:
                # Waiting is not idling: the mic is attentive either way. The
                # colour says which — muted while merely monitoring, live once
                # the words are actually being recorded.
                color = _CAPTURE_COLOR if owner._is_capturing() else _MONITOR_COLOR
            bars = owner._bars

            # The projects row and the transcript compete for the same band.
            # Only one is drawn, and the spectrum and the separator both have
            # to agree with that choice or the layout tears.
            block_h = owner.bottom_block_height()

            self._draw_title(width, height, state, color)
            self._draw_spectrum(width, height, bars, color, block_h)
            if owner.show_transcript and owner.transcript_expanded:
                self._draw_transcript(width, owner._question, owner._answer)
            self._draw_strip(width, block_h)

        # NOTE: PyObjC turns every plain method on an NSObject subclass into an
        # Objective-C selector, and the derived selector arity must match the
        # Python signature. Internal helpers therefore MUST be marked with
        # @objc.python_method or class creation fails with BadPrototypeError.

        @objc.python_method
        def _attrs(self, size: float, rgb, alpha: float, bold: bool = False):
            weight = 0.4 if bold else 0.0
            font = NSFont.monospacedSystemFontOfSize_weight_(size, weight)
            r, g, b = rgb
            return {
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: (
                    NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, alpha)
                ),
            }

        @objc.python_method
        def _draw_title(self, width, height, state, color) -> None:  # noqa: ARG002
            y = height - _TITLE_H + 4.0

            # Hollow and grey when no audio is arriving: the one thing the HUD
            # must never do is look alive while the microphone is dead.
            mic_live = owner_live = self._owner._silent_ticks < _MIC_DEAD_TICKS
            glyph_char = "◉" if mic_live else "○"
            glyph_rgb = color if mic_live else (0.48, 0.51, 0.58)
            glyph = NSAttributedString.alloc().initWithString_attributes_(
                glyph_char, self._attrs(11.0, glyph_rgb, 1.0 if mic_live else 0.75)
            )
            glyph.drawAtPoint_(NSMakePoint(_SIDE_PAD + _BUTTONS_W, y))

            title_x = _SIDE_PAD + _BUTTONS_W + 16.0
            face = self._owner._face_image
            if face is not None:
                # Wawi's round face before her name. Already circular (and
                # sized to 16 pt) in the asset, so this is a plain draw.
                face.drawInRect_fromRect_operation_fraction_(
                    NSMakeRect(title_x, y - 0.5, _FACE_PT, _FACE_PT),
                    NSMakeRect(0.0, 0.0, 0.0, 0.0),
                    NSCompositingOperationSourceOver,
                    1.0,
                )
                title_x += _FACE_PT + 5.0

            title = NSAttributedString.alloc().initWithString_attributes_(
                (self._owner.title_text or "WAWI").upper(),
                self._attrs(11.0, (0.92, 0.94, 0.98), 0.92, bold=True),
            )
            title.drawAtPoint_(NSMakePoint(title_x, y))

            # The walkie-talkie beat: while the channel is held open the user
            # may just talk. Saying so is the difference between a pause that
            # feels intentional and one that feels broken.
            label = "tu turno" if self._owner.channel_open else _LABELS.get(state, "")
            if label:
                text = NSAttributedString.alloc().initWithString_attributes_(
                    label, self._attrs(9.5, color, 0.85)
                )
                size = text.size()
                # Shifted left by the chevron's box so the two never overlap.
                text.drawAtPoint_(
                    NSMakePoint(width - _SIDE_PAD - _CHEVRON_BOX - size.width, y + 0.5)
                )

        @objc.python_method
        def _draw_spectrum(self, width, height, bars, color, block_h) -> None:  # noqa: ARG002
            # A FIXED strip under the title. Every point the user adds by
            # dragging belongs to the text, not to the audio meter.
            top = height - _TITLE_H
            bottom = top - _SPECTRUM_H
            area_h = max(8.0, _SPECTRUM_H - 6.0)

            n = len(bars)
            if n == 0:
                return
            usable = width - 2 * _SIDE_PAD
            bar_w = max(1.0, (usable - _BAR_GAP * (n - 1)) / n)

            r, g, b = color
            for i in range(n):
                value = float(bars[i])
                bar_h = max(_BAR_MIN_H, value * area_h)
                x = _SIDE_PAD + i * (bar_w + _BAR_GAP)
                # Louder bars are drawn more opaque, which reads as a glow.
                alpha = 0.30 + 0.70 * min(1.0, value * 1.15)
                NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, alpha).set()
                radius = min(bar_w / 2.0, 2.0)
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, bottom, bar_w, bar_h), radius, radius
                )
                path.fill()

        @objc.python_method
        def _draw_transcript(self, width, question, answer) -> None:
            """The conversation, newest at the BOTTOM, older scrolling off the top.

            Bottom-anchored on purpose: the newest answer is the one being read,
            so it must never move when an older one is pushed out. A chat that
            re-lays-out from the top makes you re-find the line you were on
            after every turn.
            """
            max_w = width - 2 * _SIDE_PAD
            band = self._owner.bottom_block_height()
            budget = max(1, int((band - _TRANSCRIPT_PAD) // _TRANSCRIPT_ROW_H))

            if not (question or answer or self._owner._history):
                hint = NSAttributedString.alloc().initWithString_attributes_(
                    "hablá y la respuesta aparece acá",
                    _text_attrs(_Q_FONT, (0.86, 0.89, 0.94), 0.55),
                )
                hint.drawAtPoint_(NSMakePoint(_SIDE_PAD, _STRIP_H + 6.0))
                return

            q_attrs = _text_attrs(_Q_FONT, (0.82, 0.86, 0.92), 0.85)
            a_attrs = _text_attrs(_A_FONT, (0.94, 0.96, 1.0), 0.97)

            # Walk backwards from the newest exchange, taking rows until the
            # band is full. Older ones simply never get measured.
            rows: list[tuple[str, object]] = []
            pairs = list(self._owner._history) + [(question, answer)]
            for q, a in reversed(pairs):
                if not (q or a):
                    continue
                block: list[tuple[str, object]] = []
                if q:
                    for line in wrap_lines(f"› {q}", q_attrs, max_w, 1):
                        block.append((line, q_attrs))
                if a:
                    for line in wrap_lines(a, a_attrs, max_w, budget):
                        block.append((line, a_attrs))
                if len(rows) + len(block) > budget:
                    # Partial block: keep its TAIL, so the newest words survive.
                    keep = budget - len(rows)
                    if keep > 0:
                        rows = block[-keep:] + rows
                    break
                rows = block + rows

            # Drawn from the bottom up: row 0 of `rows` is the topmost line.
            bottom = _STRIP_H + 4.0
            for i, (line, attrs) in enumerate(reversed(rows)):
                NSAttributedString.alloc().initWithString_attributes_(
                    line, attrs
                ).drawAtPoint_(NSMakePoint(_SIDE_PAD, bottom + i * _TRANSCRIPT_ROW_H))

        @objc.python_method
        def _draw_strip(self, width, block_h) -> None:
            """Hairline separator between the transcript and the bottom row."""
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.13).set()
            NSBezierPath.bezierPathWithRect_(
                NSMakeRect(0.0, _STRIP_H + block_h, width, 1.0)
            ).fill()
            self._draw_task(width)

        @objc.python_method
        def _draw_task(self, width) -> None:
            """The delegated agent's heartbeat, in the strip's spare middle.

            This is the primary progress channel on purpose: the executor's
            work is reported here rather than spoken, so the assistant is not
            narrating itself out loud while you are trying to work. Voice is
            kept for the two things that actually need you — a question and the
            outcome.

            Drawn as TEXT ONLY, into space that already exists between the copy
            pill and the push-to-talk switch. Nothing here changes the window's
            geometry: resizing an NSWindow off the main thread kills the
            process, and the executor publishes from its own thread.
            """
            if not self._owner.task_active:
                return
            left = _SIDE_PAD + _LANG_BOX_W + 8.0 + _COPY_BOX_W + 10.0
            max_w = width - _SIDE_PAD - _TOGGLE_BOX_W - 8.0 - left
            if max_w < 60.0:
                # Narrower than this and the line collides with the switch. A
                # HUD that overlaps its own controls is worse than one that
                # shows less.
                return

            # A glyph that keeps moving is the whole claim being made: that
            # something is still running. It is driven by the render phase, so
            # it stops the instant the panel does.
            glyph = _TASK_SPINNER[int(self._owner._phase * 6.0) % len(_TASK_SPINNER)]
            attrs = self._attrs(9.5, (0.55, 0.80, 1.0), 0.95)
            mark = NSAttributedString.alloc().initWithString_attributes_(glyph, attrs)
            mark.drawAtPoint_(NSMakePoint(left, (_STRIP_H - 12.0) / 2.0))

            line = self._owner.task_line or "trabajando"
            text_attrs = self._attrs(9.5, (0.86, 0.90, 0.96), 0.80)
            offset = mark.size().width + 5.0
            wrapped = wrap_lines(line, text_attrs, max_w - offset, 1)
            if not wrapped:
                return
            NSAttributedString.alloc().initWithString_attributes_(
                wrapped[0], text_attrs
            ).drawAtPoint_(NSMakePoint(left + offset, (_STRIP_H - 12.0) / 2.0))

    class WawiChevronView(NSView):  # type: ignore[misc]
        """Disclosure control that shows or hides the transcript.

        Every note in :class:`WawiToggleView` about focus, dragging and
        ``acceptsFirstMouse_`` applies here for the same reasons and is not
        repeated; this class exists separately only because its hit area and
        its glyph differ.
        """

        def initWithFrame_owner_(self, frame, owner):  # noqa: N802
            self = objc.super(WawiChevronView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._owner = owner
            self._pressed = False
            return self

        def isFlipped(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def acceptsFirstResponder(self):  # noqa: N802
            return False

        def mouseDown_(self, _event):  # noqa: N802
            self._pressed = True
            self.setNeedsDisplay_(True)

        def mouseDragged_(self, _event):  # noqa: N802
            pass

        def mouseUp_(self, event):  # noqa: N802
            was_pressed = self._pressed
            self._pressed = False
            self.setNeedsDisplay_(True)
            if not was_pressed:
                return
            point = self.convertPoint_fromView_(event.locationInWindow(), None)
            if NSPointInRect(point, self.bounds()):
                self._owner._handle_chevron_click()

        def drawRect_(self, _rect):  # noqa: N802
            """Draw one frame, and NEVER let an exception escape.
        
            AppKit calls +[NSApplication _crashOnException:] when anything is
            raised out of a drawRect_, which kills the whole process — for a
            background assistant that means it vanishes with nothing in the
            log. A drawing bug should cost one frame, not the microphone, the
            session and the conversation. The traceback is logged instead, so
            the next occurrence is diagnosable rather than fatal.
            """
            try:
                self._paint(_rect)
            except Exception:
                log.exception('Panel drawing failed; skipping this frame')

        @objc.python_method
        def _paint(self, _rect):  # noqa: N802
            bounds = self.bounds()

            # A bare glyph on a translucent panel does not read as a control —
            # it reads as a stray accent. The pill is what makes it look
            # clickable, which is the whole job of a disclosure control.
            fill = 0.26 if self._pressed else 0.16
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, fill).set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bounds, bounds.size.height / 2.0, bounds.size.height / 2.0
            ).fill()

            # "▲" = click to collapse, "▼" = click to expand: the glyph names
            # the ACTION, like every disclosure triangle on the system.
            glyph = "▲" if self._owner.transcript_expanded else "▼"
            font = NSFont.monospacedSystemFontOfSize_weight_(8.0, 0.3)
            attrs = {
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: (
                    NSColor.colorWithSRGBRed_green_blue_alpha_(0.92, 0.95, 1.0, 0.95)
                ),
            }
            text = NSAttributedString.alloc().initWithString_attributes_(glyph, attrs)
            size = text.size()
            text.drawAtPoint_(
                NSMakePoint(
                    (bounds.size.width - size.width) / 2.0,
                    (bounds.size.height - size.height) / 2.0,
                )
            )

    class WawiMenuTarget(NSObject):  # type: ignore[misc]
        """Receives the status-item menu clicks.

        A separate NSObject because AppKit menu items need an Objective-C
        target/selector pair, and the observer is a plain Python object with no
        selectors of its own. Every method here runs on the main thread — menu
        clicks always do — so they call the same handlers a click on the panel
        would.
        """

        def initWithOwner_(self, owner):  # noqa: N802
            self = objc.super(WawiMenuTarget, self).init()
            if self is None:
                return None
            self._owner = owner
            return self

        def showPanel_(self, _sender):  # noqa: N802
            self._owner.toggle_panel()

        def copyAnswer_(self, _sender):  # noqa: N802
            self._owner.copy_answer()

        def toggleListening_(self, _sender):  # noqa: N802
            self._owner._handle_toggle_click()

        def toggleLanguage_(self, _sender):  # noqa: N802
            self._owner._handle_lang_click()

        def toggleMascot_(self, _sender):  # noqa: N802
            self._owner.toggle_mascot()

        def quitWawi_(self, _sender):  # noqa: N802
            self._owner._handle_window_button(0)

    class WawiButtonsView(NSView):  # type: ignore[misc]
        """Close / minimise / zoom, in the macOS idiom.

        What each one MEANS here is not what a document window does, because
        this is not a document window:

        * **close** quits Wawi. It is the app's only window, and a red button
          that merely hid it would leave a microphone open with no way back to
          the UI (the app has no Dock icon by design).
        * **minimise** collapses the transcript — the same thing the chevron
          does. Minimising to the Dock is meaningless without a Dock icon.
        * **zoom** toggles between the size you set and a large one.
        """

        def initWithFrame_owner_(self, frame, owner):  # noqa: N802
            self = objc.super(WawiButtonsView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._owner = owner
            self._hover = -1
            return self

        def isFlipped(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def acceptsFirstResponder(self):  # noqa: N802
            return False

        @objc.python_method
        def _dot_rects(self):
            y = (self.bounds().size.height - _DOT_R) / 2.0
            return [
                NSMakeRect(i * (_DOT_R + _DOT_GAP), y, _DOT_R, _DOT_R)
                for i in range(3)
            ]

        def mouseDown_(self, _event):  # noqa: N802
            pass  # consume it so the window does not start dragging

        def mouseDragged_(self, _event):  # noqa: N802
            pass

        def mouseUp_(self, event):  # noqa: N802
            point = self.convertPoint_fromView_(event.locationInWindow(), None)
            for index, rect in enumerate(self._dot_rects()):
                # Generous hit area: a 6pt dot is smaller than the mouse can
                # reliably be placed, so each one owns the space around it.
                hit = NSMakeRect(
                    rect.origin.x - 3.0, rect.origin.y - 5.0,
                    _DOT_R + 6.0, _DOT_R + 10.0,
                )
                if NSPointInRect(point, hit):
                    self._owner._handle_window_button(index)
                    return

        def drawRect_(self, _rect):  # noqa: N802
            """Draw one frame, and NEVER let an exception escape.
        
            AppKit calls +[NSApplication _crashOnException:] when anything is
            raised out of a drawRect_, which kills the whole process — for a
            background assistant that means it vanishes with nothing in the
            log. A drawing bug should cost one frame, not the microphone, the
            session and the conversation. The traceback is logged instead, so
            the next occurrence is diagnosable rather than fatal.
            """
            try:
                self._paint(_rect)
            except Exception:
                log.exception('Panel drawing failed; skipping this frame')

        @objc.python_method
        def _paint(self, _rect):  # noqa: N802
            for rect, rgb in zip(
                self._dot_rects(), (_DOT_CLOSE, _DOT_MIN, _DOT_MAX)
            ):
                r, g, b = rgb
                NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 0.92).set()
                NSBezierPath.bezierPathWithOvalInRect_(rect).fill()

    class WawiGripView(NSView):  # type: ignore[misc]
        """Bottom-right drag handle that resizes the window.

        A borderless window has no frame to grab, so the grip has to be drawn
        and driven by hand. It anchors the TOP-LEFT corner: the panel lives
        against the top-right of a screen, and resizing from the stored
        bottom-left origin would walk the whole window across the desktop as
        you drag.

        Same focus / first-mouse reasoning as :class:`WawiToggleView`.
        """

        def initWithFrame_owner_(self, frame, owner):  # noqa: N802
            self = objc.super(WawiGripView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._owner = owner
            self._origin_frame = None
            self._start = None
            return self

        def isFlipped(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def acceptsFirstResponder(self):  # noqa: N802
            return False

        def mouseDown_(self, event):  # noqa: N802
            window = self.window()
            if window is None:
                return
            self._origin_frame = window.frame()
            # Screen coordinates: the window is about to move under the mouse,
            # so anything relative to it would drift as we drag.
            self._start = window.convertPointToScreen_(event.locationInWindow())

        def mouseDragged_(self, event):  # noqa: N802
            window = self.window()
            if window is None or self._start is None:
                return
            now = window.convertPointToScreen_(event.locationInWindow())
            start_frame = self._origin_frame
            width = max(_MIN_WIDTH, start_frame.size.width + (now.x - self._start.x))
            height = max(_MIN_HEIGHT, start_frame.size.height - (now.y - self._start.y))
            top = start_frame.origin.y + start_frame.size.height
            window.setFrame_display_(
                NSMakeRect(start_frame.origin.x, top - height, width, height), True
            )

        def mouseUp_(self, _event):  # noqa: N802
            self._start = None
            window = self.window()
            if window is not None:
                frame = window.frame()
                # From here on the user's size wins over the automatic one.
                self._owner._adopt_manual_size(frame.size.width, frame.size.height)

        def drawRect_(self, _rect):  # noqa: N802
            """Draw one frame, and NEVER let an exception escape.
        
            AppKit calls +[NSApplication _crashOnException:] when anything is
            raised out of a drawRect_, which kills the whole process — for a
            background assistant that means it vanishes with nothing in the
            log. A drawing bug should cost one frame, not the microphone, the
            session and the conversation. The traceback is logged instead, so
            the next occurrence is diagnosable rather than fatal.
            """
            try:
                self._paint(_rect)
            except Exception:
                log.exception('Panel drawing failed; skipping this frame')

        @objc.python_method
        def _paint(self, _rect):  # noqa: N802
            bounds = self.bounds()
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.34).set()
            # Three short diagonals: the ordinary macOS grip idiom.
            for offset in (3.0, 7.0, 11.0):
                path = NSBezierPath.bezierPath()
                path.moveToPoint_(NSMakePoint(bounds.size.width - offset, 2.0))
                path.lineToPoint_(NSMakePoint(bounds.size.width - 2.0, offset))
                path.setLineWidth_(1.0)
                path.stroke()

    class WawiCopyView(NSView):  # type: ignore[misc]
        """Copies the last answer to the clipboard.

        Same focus / first-mouse reasoning as :class:`WawiToggleView`, not
        repeated. It confirms in place — a copy button that looks identical
        before and after leaves you pressing it twice and still unsure.
        """

        def initWithFrame_owner_(self, frame, owner):  # noqa: N802
            self = objc.super(WawiCopyView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._owner = owner
            self._pressed = False
            return self

        def isFlipped(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def acceptsFirstResponder(self):  # noqa: N802
            return False

        def mouseDown_(self, _event):  # noqa: N802
            self._pressed = True
            self.setNeedsDisplay_(True)

        def mouseDragged_(self, _event):  # noqa: N802
            pass

        def mouseUp_(self, event):  # noqa: N802
            was_pressed = self._pressed
            self._pressed = False
            self.setNeedsDisplay_(True)
            if not was_pressed:
                return
            point = self.convertPoint_fromView_(event.locationInWindow(), None)
            if NSPointInRect(point, self.bounds()):
                self._owner.copy_answer()

        def drawRect_(self, _rect):  # noqa: N802
            try:
                self._paint(_rect)
            except Exception:
                log.exception("Panel drawing failed; skipping this frame")

        @objc.python_method
        def _paint(self, _rect):
            bounds = self.bounds()
            has_answer = bool(self._owner._answer)
            copied = self._owner.copy_confirmed()

            fill = 0.26 if self._pressed else (0.20 if copied else 0.16)
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, fill).set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bounds, bounds.size.height / 2.0, bounds.size.height / 2.0
            ).fill()

            label = "copiado" if copied else "copiar"
            # Dimmed with nothing to copy: a control that looks live and does
            # nothing is worse than one that says it is not available.
            alpha = 0.95 if (has_answer or copied) else 0.35
            text = NSAttributedString.alloc().initWithString_attributes_(
                label, _text_attrs(8.5, (0.92, 0.95, 1.0), alpha)
            )
            size = text.size()
            text.drawAtPoint_(
                NSMakePoint(
                    (bounds.size.width - size.width) / 2.0,
                    (bounds.size.height - size.height) / 2.0 + 0.5,
                )
            )

    class WawiLangView(NSView):  # type: ignore[misc]
        """ES / EN pill: switches the assistant's language mid-session.

        Same focus, dragging and ``acceptsFirstMouse_`` reasoning as
        :class:`WawiToggleView`, not repeated here.
        """

        def initWithFrame_owner_(self, frame, owner):  # noqa: N802
            self = objc.super(WawiLangView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._owner = owner
            self._pressed = False
            return self

        def isFlipped(self):  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def acceptsFirstResponder(self):  # noqa: N802
            return False

        def mouseDown_(self, _event):  # noqa: N802
            self._pressed = True
            self.setNeedsDisplay_(True)

        def mouseDragged_(self, _event):  # noqa: N802
            pass

        def mouseUp_(self, event):  # noqa: N802
            was_pressed = self._pressed
            self._pressed = False
            self.setNeedsDisplay_(True)
            if not was_pressed:
                return
            point = self.convertPoint_fromView_(event.locationInWindow(), None)
            if NSPointInRect(point, self.bounds()):
                self._owner._handle_lang_click()

        def drawRect_(self, _rect):  # noqa: N802
            """Draw one frame, and NEVER let an exception escape.
        
            AppKit calls +[NSApplication _crashOnException:] when anything is
            raised out of a drawRect_, which kills the whole process — for a
            background assistant that means it vanishes with nothing in the
            log. A drawing bug should cost one frame, not the microphone, the
            session and the conversation. The traceback is logged instead, so
            the next occurrence is diagnosable rather than fatal.
            """
            try:
                self._paint(_rect)
            except Exception:
                log.exception('Panel drawing failed; skipping this frame')

        @objc.python_method
        def _paint(self, _rect):  # noqa: N802
            bounds = self.bounds()
            fill = 0.26 if self._pressed else 0.16
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, fill).set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bounds, bounds.size.height / 2.0, bounds.size.height / 2.0
            ).fill()

            # The CURRENT language, not the one a click would select. A control
            # that shows the other state reads as a label of what you have, and
            # the user cannot tell which reading is intended — showing what is
            # in effect is the one that is never ambiguous.
            text = NSAttributedString.alloc().initWithString_attributes_(
                (self._owner.language or "es").upper(),
                _text_attrs(9.5, (0.92, 0.95, 1.0), 0.95, bold=True),
            )
            size = text.size()
            text.drawAtPoint_(
                NSMakePoint(
                    (bounds.size.width - size.width) / 2.0,
                    (bounds.size.height - size.height) / 2.0 + 0.5,
                )
            )

    class WawiToggleView(NSView):  # type: ignore[misc]
        """The push-to-talk switch: a label plus a small pill switch.

        Deliberately hand-drawn rather than an ``NSButton``/``NSSwitch``:

        * **Focus.** An AppKit control assumes it lives in a window that can
          become key. This panel's window returns ``False`` from
          ``canBecomeKeyWindow``, so a real control would sit in a permanently
          inactive window — it would draw in its disabled/inactive style and
          would try to take first responder on click. Drawing the switch
          ourselves means nothing ever requests key status or first responder.
        * **Dragging.** This view's frame is exactly the control's hit area.
          ``hitTest_`` outside it returns nil (inherited default), so every
          click elsewhere still falls through to the window background and
          ``setMovableByWindowBackground_`` keeps working. Inside it we consume
          the event and never chain to super, so dragging the switch does not
          drag the window.

        ``acceptsFirstMouse_`` is the crucial piece: for a window that is not
        key, AppKit normally swallows the first click just to activate the
        window and does NOT deliver it to the view. Since this window can never
        become key, *every* click is a "first mouse" click — without returning
        True here the switch would simply never respond.
        """

        def initWithFrame_owner_(self, frame, owner):  # noqa: N802
            self = objc.super(WawiToggleView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._owner = owner
            self._pressed = False
            return self

        def isFlipped(self):  # noqa: N802
            return False

        # Deliver clicks even though the window is never key/active.
        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        # The panel must never take first responder away from the user's editor.
        def acceptsFirstResponder(self):  # noqa: N802
            return False

        # -- mouse handling (never calls super => never starts a window drag) --

        def mouseDown_(self, _event):  # noqa: N802
            self._pressed = True
            self.setNeedsDisplay_(True)

        def mouseDragged_(self, _event):  # noqa: N802
            # Swallow drags that begin on the switch so the window stays put.
            pass

        def mouseUp_(self, event):  # noqa: N802
            was_pressed = self._pressed
            self._pressed = False
            self.setNeedsDisplay_(True)
            if not was_pressed:
                return
            # Commit only when released inside the control, like a real button.
            point = self.convertPoint_fromView_(event.locationInWindow(), None)
            if NSPointInRect(point, self.bounds()):
                self._owner._handle_toggle_click()

        # -- drawing ----------------------------------------------------

        @objc.python_method
        def _switch_rect(self):
            bounds = self.bounds()
            x = bounds.size.width - _SWITCH_W
            y = (bounds.size.height - _SWITCH_H) / 2.0
            return NSMakeRect(x, y, _SWITCH_W, _SWITCH_H)

        def drawRect_(self, _rect):  # noqa: N802
            """Draw one frame, and NEVER let an exception escape.
        
            AppKit calls +[NSApplication _crashOnException:] when anything is
            raised out of a drawRect_, which kills the whole process — for a
            background assistant that means it vanishes with nothing in the
            log. A drawing bug should cost one frame, not the microphone, the
            session and the conversation. The traceback is logged instead, so
            the next occurrence is diagnosable rather than fatal.
            """
            try:
                self._paint(_rect)
            except Exception:
                log.exception('Panel drawing failed; skipping this frame')

        @objc.python_method
        def _paint(self, _rect):  # noqa: N802
            on = bool(self._owner._listening)
            bounds = self.bounds()

            # Label — brighter when the mode is engaged.
            label_alpha = 0.88 if on else 0.55
            font = NSFont.monospacedSystemFontOfSize_weight_(8.5, 0.0)
            attrs = {
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: (
                    NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.86, 0.89, 0.94, label_alpha
                    )
                ),
            }
            text = NSAttributedString.alloc().initWithString_attributes_(
                _TOGGLE_LABEL, attrs
            )
            size = text.size()
            track = self._switch_rect()
            text.drawAtPoint_(
                NSMakePoint(
                    max(0.0, track.origin.x - _SWITCH_GAP - size.width),
                    (bounds.size.height - size.height) / 2.0,
                )
            )

            # Track.
            radius = _SWITCH_H / 2.0
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                track, radius, radius
            )
            if on:
                r, g, b = _TOGGLE_ON_RGB
                alpha = 0.75 if self._pressed else 0.92
                NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, alpha).set()
            else:
                alpha = 0.26 if self._pressed else 0.17
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, alpha).set()
            path.fill()

            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.22).set()
            path.setLineWidth_(0.5)
            path.stroke()

            # Knob — right when ON, left when OFF.
            inset = (_SWITCH_H - _SWITCH_KNOB) / 2.0
            knob_x = (
                track.origin.x + _SWITCH_W - _SWITCH_KNOB - inset
                if on
                else track.origin.x + inset
            )
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                1.0, 1.0, 1.0, 0.97 if on else 0.72
            ).set()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(knob_x, track.origin.y + inset, _SWITCH_KNOB, _SWITCH_KNOB)
            ).fill()

    class WawiPanelWindow(NSPanel):  # type: ignore[misc]
        """Non-activating floating panel that can never become the key window."""

        def canBecomeKeyWindow(self):  # noqa: N802
            return False

        def canBecomeMainWindow(self):  # noqa: N802
            return False

    class _TimerTarget(NSView):  # type: ignore[misc]
        """Minimal Objective-C object used as the repeating timer's target."""

        def initWithOwner_(self, owner):  # noqa: N802
            self = objc.super(_TimerTarget, self).initWithFrame_(
                NSMakeRect(0, 0, 1, 1)
            )
            if self is None:
                return None
            self._owner = owner
            return self

        def tick_(self, _timer):  # noqa: N802
            self._owner._tick()

    class WawiMascotView(NSView):  # type: ignore[misc]
        """Draws Wawi: the animator's frame, in the pose it computed.

        The frame (and the one crossfading away) is moved, scaled, tilted and
        faded as a whole. Dragging her beside the panel moves her relative to
        it, so she keeps sitting by it wherever it goes; on her own (panel
        hidden) dragging moves HER, and the spot is remembered. A plain click
        (no drag) makes her react to WHERE she was touched (head, nose, belly,
        tail, ear, paw: see ``mascot.touch_zone``); a double-click makes her
        hop. Stroking her head with the cursor (no button) is petting: a
        tracking area that is active even though the window never becomes key
        delivers the moves, so this never takes focus. A right-click or a
        control-click opens the menu-bar item's menu: with no panel, she is
        the thing on screen you reach for.
        """

        #: A press that moves less than this (points) is a click, not a drag.
        _CLICK_SLOP = 3.0

        def initWithFrame_owner_(self, frame, owner):  # noqa: N802
            self = objc.super(WawiMascotView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._owner = owner
            self._drag_from = None
            self._press_from = None
            self._press_point = None
            self._dragged = False
            try:
                # Hover without focus: ActiveAlways delivers mouseMoved even
                # though her window can never become key; InVisibleRect keeps
                # the area matching the view without any bookkeeping.
                from AppKit import (  # type: ignore[import]
                    NSTrackingActiveAlways,
                    NSTrackingArea,
                    NSTrackingInVisibleRect,
                    NSTrackingMouseEnteredAndExited,
                    NSTrackingMouseMoved,
                )

                area = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
                    frame,
                    NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited
                    | NSTrackingActiveAlways | NSTrackingInVisibleRect,
                    self,
                    None,
                )
                self.addTrackingArea_(area)
            except Exception:
                log.debug("No hover tracking for Wawi; petting is off", exc_info=True)
            return self

        def isOpaque(self):  # noqa: N802
            return False

        @objc.python_method
        def _local(self, event):
            """The event's position in this view (points, y up), or None."""
            try:
                p = self.convertPoint_fromView_(event.locationInWindow(), None)
                return (float(p.x), float(p.y))
            except Exception:
                return None

        def mouseMoved_(self, event):  # noqa: N802
            try:
                self._owner._hover_mascot(self._local(event))
            except Exception:
                log.debug("Mascot hover failed", exc_info=True)

        def mouseExited_(self, _event):  # noqa: N802
            try:
                self._owner._hover_mascot(None)
            except Exception:
                log.debug("Mascot hover end failed", exc_info=True)

        # A click on her must work without first "activating" anything: the
        # window can never become key, so the first click is the only click.
        def acceptsFirstMouse_(self, _event):  # noqa: N802
            return True

        def mouseDown_(self, event):  # noqa: N802
            try:
                if event.modifierFlags() & NSEventModifierFlagControl:
                    self._drag_from = None
                    self._owner._mascot_menu(event, self)
                    return
            except Exception:
                log.debug("Mascot control-click failed", exc_info=True)
            self._dragged = False
            try:
                self._drag_from = NSEvent_mouse_location()
            except Exception:
                self._drag_from = None
            self._press_from = self._drag_from
            self._press_point = self._local(event)

        def rightMouseDown_(self, event):  # noqa: N802
            self._drag_from = None
            self._press_from = None
            self._owner._mascot_menu(event, self)

        def mouseDragged_(self, _event):  # noqa: N802
            start = self._drag_from
            if start is None:
                return
            try:
                now = NSEvent_mouse_location()
                press = self._press_from or start
                if not self._dragged:
                    # A hand is never perfectly still: until the pointer has
                    # really moved, this may still be a click, and she must
                    # not budge.
                    if (abs(now[0] - press[0]) < self._CLICK_SLOP
                            and abs(now[1] - press[1]) < self._CLICK_SLOP):
                        return
                    self._dragged = True
                    start = press
                    self._owner._begin_drag_mascot()
                self._drag_from = now
                self._owner._drag_mascot(now[0] - start[0], now[1] - start[1])
            except Exception:
                log.debug("Mascot drag failed", exc_info=True)

        def mouseUp_(self, event):  # noqa: N802
            pressed = self._drag_from is not None
            dragged = self._dragged
            point = self._press_point
            self._drag_from = None
            self._press_from = None
            self._press_point = None
            self._dragged = False
            if not pressed:
                return
            try:
                if dragged:
                    self._owner._end_drag_mascot()
                else:
                    try:
                        clicks = int(event.clickCount())
                    except Exception:
                        clicks = 1
                    self._owner._touch_mascot(point, clicks)
            except Exception:
                log.debug("Mascot click/drag end failed", exc_info=True)

        def drawRect_(self, rect):  # noqa: N802
            # Same contract as the panel: an exception out of drawRect_ kills
            # the process, and a mascot frame is not worth that.
            try:
                self._paint()
            except Exception:
                log.debug("Mascot drawing failed; skipping this frame", exc_info=True)

        @objc.python_method
        def _paint(self) -> None:
            owner = self._owner
            anim = owner._mascot_animator
            frames = owner._mascot_frames
            image = frames.get(anim.frame) if anim.frame else None
            previous = None
            if image is None:
                image = owner._mascot_image  # the single drawing
            elif anim.prev_frame and anim.frame_mix < 1.0:
                previous = frames.get(anim.prev_frame)
            if image is None:
                return
            pose = owner._mascot_pose
            alpha = max(0.0, min(1.0, pose.alpha))
            if previous is not None:
                prev_alpha, alpha = _mascot.crossfade_alphas(anim.frame_mix, alpha)
            bounds = self.bounds()
            w, h = owner._mascot_sprite_size
            NSGraphicsContext.saveGraphicsState()
            try:
                xf = NSAffineTransform.transform()
                xf.translateXBy_yBy_(
                    bounds.size.width / 2.0 + pose.dx, _mascot.PAD_BOTTOM + pose.dy
                )
                xf.rotateByDegrees_(pose.rot)
                # Mirrored (running back home): flipped about her centre line.
                flip = -1.0 if anim.mirrored else 1.0
                xf.scaleXBy_yBy_(pose.sx * flip, pose.sy)
                xf.concat()
                rect = NSMakeRect(-w / 2.0, 0.0, w, h)
                whole = NSMakeRect(0.0, 0.0, 0.0, 0.0)  # zero rect = whole image
                if previous is not None and prev_alpha > 0.0:
                    previous.drawInRect_fromRect_operation_fraction_(
                        rect, whole, NSCompositingOperationSourceOver, prev_alpha
                    )
                if alpha > 0.0:
                    image.drawInRect_fromRect_operation_fraction_(
                        rect, whole, NSCompositingOperationSourceOver, alpha
                    )
            finally:
                NSGraphicsContext.restoreGraphicsState()
            self._paint_badge(bounds, pose)

        @objc.python_method
        def _paint_badge(self, bounds, pose) -> None:
            """The tiny status dot over her head: recording, or mic off.

            Outside the transform on purpose: it rides up and down with her
            hop but never tilts or squashes, so it stays a crisp dot.
            """
            kind = _mascot.badge(self._owner._mascot_animator.key)
            if kind is None:
                return
            d = _MASCOT_BADGE_D
            w, h = self._owner._mascot_sprite_size
            x = bounds.size.width / 2.0 + w * 0.22 + pose.dx
            y = min(
                bounds.size.height - d - 2.0,
                _mascot.PAD_BOTTOM + pose.dy + h * pose.sy + 4.0,
            )
            dot = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x, y, d, d))
            if kind == "listening":
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    *_MASCOT_BADGE_LISTENING, 0.95
                ).setFill()
                dot.fill()
            else:
                # Hollow and grey: "off", readable without relying on colour.
                NSColor.colorWithCalibratedWhite_alpha_(0.55, 0.9).setStroke()
                dot.setLineWidth_(1.5)
                dot.stroke()

    def NSEvent_mouse_location() -> tuple[float, float]:  # noqa: N802
        """Mouse position in screen points (deltas survive the window moving)."""
        from AppKit import NSEvent  # type: ignore[import]

        loc = NSEvent.mouseLocation()
        return (float(loc.x), float(loc.y))


# ---------------------------------------------------------------------------
# PanelObserver — public class
# ---------------------------------------------------------------------------


class PanelObserver:
    """Floating always-on-top spectrum HUD, usable as a Wawi state observer.

    Mirrors the :class:`wawi.ui.TuiObserver` interface. When ``enabled`` is
    False — or when PyObjC/AppKit is unavailable — every method is a no-op.

    Args:
        enabled: If False, all methods are no-ops.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled) and _HAS_APPKIT
        if enabled and not _HAS_APPKIT:
            log.warning("Panel disabled: AppKit unavailable (%s)", _APPKIT_ERROR)

        # Written by worker threads, read by the main-thread render timer.
        self._current_state: State = State.IDLE
        # Shown in the title row; set from config.assistant_name.
        self.title_text = "WAWI"
        # The last exchange, drawn in the band the projects row is not using.
        # This is the answer ON SCREEN: the terminal transcript is only visible
        # to whoever is looking at the terminal, and the whole point of a
        # floating HUD is that you are not.
        self.show_transcript = True
        # Collapsed hides the text AND shrinks the window back to PANEL_HEIGHT:
        # a HUD that keeps space it is not using is just a bigger HUD.
        self.transcript_expanded = True
        self._question = ""
        self._answer = ""
        # Past exchanges, oldest first. The panel shows as many as fit, so a
        # taller window means MORE CONVERSATION — without this, dragging it
        # bigger just added grey, because one exchange cannot fill a big panel
        # no matter how the rows are laid out.
        self._history: list[tuple[str, str]] = []
        self._chevron_view: Any = None
        # Set from any thread; consumed by _tick on the AppKit main thread.
        self._pending_resize = False
        # Current language shown on the ES/EN pill. The orchestrator owns the
        # real switch; the panel only reports the click.
        # Set once the user drags the grip. From then on THEIR size wins: the
        # automatic height is a good default, not a preference to keep
        # re-imposing on someone who already said what they wanted.
        self._manual_size: tuple[float, float] | None = None
        # Size to go back to when zoom is clicked a second time.
        self._zoom_restore: tuple[float, float] | None = None
        self._grip_view: Any = None
        self._status_item: Any = None
        self._status_menu: Any = None
        self._status_target: Any = None
        self._status_drawn: tuple | None = None
        self._status_glyph_drawn: str | None = None
        self._spin = 0
        #: Whether the HUD window is on screen. False lives entirely in the
        #: menu bar — see :meth:`toggle_panel`.
        self.panel_visible = True
        self._pending_visibility = False
        self._copy_view: Any = None
        #: When the answer was last copied, for the pill's confirmation.
        self._copied_at: float | None = None
        self._copy_drawn = False
        self._status_ticks = 0
        self._buttons_view: Any = None
        # Set when the user asks to quit from the red button. The orchestrator
        # polls it; the panel itself must not decide to end the process.
        self.quit_requested = False
        self.language = "es"
        self._lang_view: Any = None
        self._lang_cb: Callable[[str], None] | None = None
        # Microphone liveness, for the title indicator. Starts "dead" so the
        # panel never claims to be hearing before a single block has arrived.
        self._heard_audio = False
        self._silent_ticks = _MIC_DEAD_TICKS
        # True while a reply is expected without the wake phrase.
        self.channel_open = False
        # The delegated agent's state. Written by the executor's thread, read
        # by the main-thread render timer — plain attributes, exactly like
        # channel_open above, because that is the one cross-thread pattern in
        # this module that cannot resize a window from the wrong thread.
        self.task_active = False
        self.task_line = ""
        self._samples_source: Callable[[], "np.ndarray | None"] | None = None
        # Reports whether an utterance is being recorded right now.
        self._capture_source: Callable[[], bool] | None = None

        # The microphone gate. ON = Wawi hears you, OFF = it does not.
        # Defaults to ON: a HUD that starts deaf and says nothing about it is
        # indistinguishable from a broken microphone.
        # _listening_drawn lets the main-thread timer notice a change made from
        # a worker thread and schedule the redraw on the main thread.
        self._listening = True
        self._listening_drawn = True
        self._listening_callback: Callable[[bool], None] | None = None

        # Main-thread-only render state.
        self._bars = np.zeros(_BAR_COUNT, dtype=np.float32)
        self._phase = 0.0
        self._sample_rate = 16000

        # AppKit objects, created in start().
        self._app: Any = None
        self._window: Any = None
        self._view: Any = None
        self._toggle_view: Any = None
        self._timer: Any = None
        self._timer_target: Any = None

        # Origin requested before the window exists (restore-before-start).
        self._pending_origin: tuple[float, float] | None = None

        # Wawi, the mascot (config ui.mascot). A separate little window beside
        # the panel; everything about her is main-thread only, driven by _tick.
        self.show_mascot = True
        self._mascot_window: Any = None
        self._mascot_view: Any = None
        self._mascot_image: Any = None
        #: Animation frames by name (NSImage), loaded once; empty = the
        #: single drawing in _mascot_image is used instead.
        self._mascot_frames: dict[str, Any] = {}
        #: Set from any thread (state observer, orchestrator); applied by
        #: the next render tick: None, "cheer" or "party".
        self._mascot_celebrate: str | None = None
        self._mascot_sprite_size: tuple[float, float] = (0.0, 0.0)
        self._mascot_animator = _mascot.MascotAnimator()
        #: Strokes of the cursor over her head (fed by the view's hover).
        self._mascot_pet = _mascot.PetDetector()
        #: How far her window is currently shifted by a dash (points, x).
        self._mascot_exc = 0.0
        self._mascot_pose = _mascot.REST
        self._mascot_drawn: tuple | None = None
        #: Where the user dragged her, relative to her anchor by the panel.
        self._mascot_offset: tuple[float, float] = (0.0, 0.0)
        #: Her own spot when the panel is hidden; None = the default corner.
        #: Persisted by the orchestrator (get/set_mascot_origin).
        self._mascot_alone_origin: tuple[float, float] | None = None
        #: Whether she is currently sitting by the panel (a child window) or
        #: standing on her own. None until the first tick decides.
        self._mascot_attached: bool | None = None
        self._mascot_dragging = False
        #: True once she has been put somewhere on her own this run.
        self._mascot_alone_placed = False
        self._mascot_shown = False
        #: Level of the speech being played, for her talking bob. Set by the
        #: orchestrator to Voice.output_level; None = synthetic rhythm.
        self._voice_level_source: Callable[[], float | None] | None = None
        self._reduce_motion = False
        self._face_image: Any = None
        self._status_image_set = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create the panel and start the 30 fps render timer.

        MUST be called from the main thread. Non-blocking: it returns as soon
        as the window is on screen. The caller is responsible for then parking
        the main thread in the AppKit event loop via :func:`run_event_loop`.
        No-op when disabled.
        """
        if not self.enabled or self._window is not None:
            return

        if threading.current_thread() is not threading.main_thread():
            log.error("PanelObserver.start() must run on the main thread; skipping")
            self.enabled = False
            return

        try:
            self._build_window()
        except Exception:
            log.warning("Failed to build the Wawi panel; disabling it", exc_info=True)
            self.enabled = False
            self._window = None

    def _build_window(self) -> None:
        self._app = NSApplication.sharedApplication()
        # Accessory policy: background app — no Dock icon, never activated.
        self._app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        width = self._manual_size[0] if self._manual_size is not None else PANEL_WIDTH
        frame = NSMakeRect(0.0, 0.0, width, self._window_height())
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel

        window = WawiPanelWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False
        )
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setHasShadow_(True)
        # An ORDINARY window level. It used to be NSFloatingWindowLevel, which
        # keeps the HUD above every other window forever — right for a status
        # indicator you glance at, wrong for something you hold a conversation
        # with and then want out of the way. It still joins all Spaces and
        # survives full-screen, so it is where you left it when you come back.
        window.setLevel_(NSNormalWindowLevel)
        window.setMovableByWindowBackground_(True)
        window.setIgnoresMouseEvents_(False)
        window.setHidesOnDeactivate_(False)
        window.setReleasedWhenClosed_(False)
        window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Translucent "glass" backdrop.
        effect = NSVisualEffectView.alloc().initWithFrame_(frame)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateActive)
        # Pin the panel to a dark appearance regardless of the system theme.
        #
        # Every colour in this file is a light value chosen for a dark backdrop.
        # NSVisualEffectView blends with whatever is BEHIND the window, so in
        # light mode the glass turns pale and the HUD becomes light-on-light —
        # unreadable, which is what the user hit. Adapting the whole palette to
        # two themes is the alternative; pinning it is one line and keeps the
        # floating-HUD identity the design already has. A dark HUD over a light
        # desktop is an ordinary macOS look (Spotlight, Now Playing), not a
        # theme violation.
        effect.setMaterial_(NSVisualEffectMaterialHUDWindow)
        try:
            dark = NSAppearance.appearanceNamed_(NSAppearanceNameDarkAqua)
            if dark is not None:
                effect.setAppearance_(dark)
                window.setAppearance_(dark)
        except Exception:
            # An OS without DarkAqua still gets the HUD material above.
            log.debug("Could not pin the panel to a dark appearance", exc_info=True)
        effect.setWantsLayer_(True)
        effect.layer().setCornerRadius_(CORNER_RADIUS)
        effect.layer().setMasksToBounds_(True)
        window.setContentView_(effect)

        view = WawiSpectrumView.alloc().initWithFrame_owner_(frame, self)
        view.setAutoresizingMask_(18)  # width | height sizable
        effect.addSubview_(view)
        self._view = view

        # The toggle is added AFTER the spectrum view, so it sits above it in
        # z-order and wins hitTest_. Its frame is exactly its hit area: clicks
        # anywhere else fall through to the background and still drag the
        # window. Anchored bottom-right (autoresize: min-X + max-Y flexible).
        toggle_frame = NSMakeRect(
            width - _SIDE_PAD - _TOGGLE_BOX_W,
            (_STRIP_H - _TOGGLE_BOX_H) / 2.0,
            _TOGGLE_BOX_W,
            _TOGGLE_BOX_H,
        )
        toggle = WawiToggleView.alloc().initWithFrame_owner_(toggle_frame, self)
        # Pinned to the BOTTOM-right: the left margin and the margin ABOVE it
        # absorb any resize. The old literal 8 was NSViewMinYMargin, which pins
        # a view to the TOP — invisible while the panel never resized, and the
        # moment the transcript made it grow the switch floated into the middle
        # of the answer.
        toggle.setAutoresizingMask_(NSViewMinXMargin | NSViewMaxYMargin)
        effect.addSubview_(toggle)
        self._toggle_view = toggle

        # Top-right, on the title row. Added after the spectrum view for the
        # same z-order reason as the toggle.
        chevron_frame = NSMakeRect(
            width - _SIDE_PAD - _CHEVRON_BOX,
            self._window_height() - _TITLE_H + 1.0,
            _CHEVRON_BOX,
            _CHEVRON_BOX,
        )
        chevron = WawiChevronView.alloc().initWithFrame_owner_(chevron_frame, self)
        # Pinned to the TOP-right: left margin and the margin BELOW it flex.
        chevron.setAutoresizingMask_(NSViewMinXMargin | NSViewMinYMargin)
        effect.addSubview_(chevron)
        self._chevron_view = chevron

        # Bottom-LEFT of the strip, opposite push-to-talk. Same bottom anchor.
        lang_frame = NSMakeRect(
            _SIDE_PAD,
            (_STRIP_H - _LANG_BOX_H) / 2.0,
            _LANG_BOX_W,
            _LANG_BOX_H,
        )
        lang_view = WawiLangView.alloc().initWithFrame_owner_(lang_frame, self)
        lang_view.setAutoresizingMask_(NSViewMaxYMargin)
        effect.addSubview_(lang_view)
        self._lang_view = lang_view

        copy_frame = NSMakeRect(
            _SIDE_PAD + _LANG_BOX_W + 8.0,
            (_STRIP_H - _LANG_BOX_H) / 2.0,
            _COPY_BOX_W,
            _LANG_BOX_H,
        )
        copy_view = WawiCopyView.alloc().initWithFrame_owner_(copy_frame, self)
        copy_view.setAutoresizingMask_(NSViewMaxYMargin)
        effect.addSubview_(copy_view)
        self._copy_view = copy_view

        # Bottom-right CORNER, past the switch. Anchored to that corner so it
        # stays under the mouse while the drag it is driving resizes the window.
        grip_frame = NSMakeRect(width - _GRIP, 0.0, _GRIP, _GRIP)
        buttons_frame = NSMakeRect(
            _SIDE_PAD - 2.0,
            self._window_height() - _TITLE_H + 4.0,
            _BUTTONS_W,
            _TITLE_H - 8.0,
        )
        buttons = WawiButtonsView.alloc().initWithFrame_owner_(buttons_frame, self)
        buttons.setAutoresizingMask_(NSViewMinYMargin)
        effect.addSubview_(buttons)
        self._buttons_view = buttons

        grip = WawiGripView.alloc().initWithFrame_owner_(grip_frame, self)
        grip.setAutoresizingMask_(NSViewMinXMargin | NSViewMaxYMargin)
        effect.addSubview_(grip)
        self._grip_view = grip

        self._set_app_icon()
        self._face_image = self._load_image(_mascot.FACE_ICON, _FACE_PT)
        self._build_status_item()

        self._window = window
        # A visibility restored from disk was set before there was a window to
        # apply it to.
        if not self.panel_visible:
            self._apply_panel_visibility()

        # Position: restore a saved origin, else default to the top-right of
        # the main screen (works on a multi-monitor setup once restored).
        if self._pending_origin is not None:
            x, y = self._pending_origin
            self._pending_origin = None
            window.setFrameOrigin_(NSMakePoint(x, y))
        else:
            screen = NSScreen.mainScreen()
            if screen is not None:
                visible = screen.visibleFrame()
                window.setFrameOrigin_(
                    NSMakePoint(
                        visible.origin.x + visible.size.width - width - 24.0,
                        visible.origin.y
                        + visible.size.height
                        - self._window_height()
                        - 24.0,
                    )
                )

        # Show WITHOUT taking key focus. Never use makeKeyAndOrderFront_ here.
        # Only when visible: a panel hidden from config or from the last run
        # must not flash up at every start (it was ordered out above).
        if self.panel_visible:
            window.orderFrontRegardless()

        self._build_mascot()
        self._start_timer()
        log.info("Wawi panel started (is_key_window=%s)", self.is_key_window())

    def _start_timer(self) -> None:
        self._timer_target = _TimerTarget.alloc().initWithOwner_(self)
        timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            _FRAME_INTERVAL, self._timer_target, "tick:", None, True
        )
        # Common modes keeps the spectrum animating while the user drags the
        # window (default mode alone stops firing during event tracking).
        NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
        self._timer = timer

    def stop(self) -> None:
        """Tear down the timer and close the panel. Safe if never started."""
        if self._timer is not None:
            try:
                self._timer.invalidate()
            except Exception:
                log.warning("Error invalidating panel timer", exc_info=True)
            self._timer = None
        self._timer_target = None

        # Menu bar items outlive their creator unless removed by hand: without
        # this, quitting leaves a dead glyph up there until you log out.
        if self._status_item is not None:
            try:
                NSStatusBar.systemStatusBar().removeStatusItem_(self._status_item)
            except Exception:
                log.debug("Could not remove the menu bar icon", exc_info=True)
            self._status_item = None
            self._status_menu = None
            self._status_target = None

        self._close_mascot()

        if self._window is not None:
            try:
                self._window.orderOut_(None)
                self._window.close()
            except Exception:
                log.warning("Error closing the Wawi panel", exc_info=True)
            self._window = None
            self._view = None
            self._toggle_view = None

    # ------------------------------------------------------------------
    # Render tick (main thread, ~30 fps)
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Recompute smoothed bars and request a redraw. Never raises."""
        try:
            self._phase += _FRAME_INTERVAL
            state = self._current_state
            target = self._compute_target(state)

            # Track microphone liveness for the title indicator. Counting ticks
            # rather than timestamping keeps this on the render clock, which is
            # the only clock the drawing code has.
            if self._heard_audio:
                self._silent_ticks = 0
            else:
                self._silent_ticks = min(self._silent_ticks + 1, _MIC_DEAD_TICKS)

            # Asymmetric smoothing: fast attack, slow decay — no flicker.
            current = self._bars
            rising = target > current
            weights = np.where(rising, _ATTACK, _DECAY).astype(np.float32)
            self._bars = (current + (target - current) * weights).astype(np.float32)

            if self._view is not None:
                self._view.setNeedsDisplay_(True)

            # Pick up a push-to-talk change made from a worker thread and
            # repaint the switch here, on the main thread, where it is safe.
            if self._toggle_view is not None and self._listening_drawn != self._listening:
                self._listening_drawn = self._listening
                self._toggle_view.setNeedsDisplay_(True)

            # Same reason: a new answer arrives on a worker thread and the
            # window has to grow or shrink to fit it. Doing that here is the
            # only place it is safe.
            if self._pending_resize:
                self._pending_resize = False
                self._resize_to_content()

            if self._pending_visibility:
                self._pending_visibility = False
                self._apply_panel_visibility()

            self._tick_mascot()

            # The "copiado" label expires on a clock, not on a click, so the
            # pill has to be repainted when nothing else happened.
            confirmed = self.copy_confirmed()
            if confirmed != self._copy_drawn:
                self._copy_drawn = confirmed
                if self._copy_view is not None:
                    self._copy_view.setNeedsDisplay_(True)

            # The menu-bar glyph is the only status visible when the panel is
            # hidden, so it animates on its own clock rather than only when a
            # click happens to change something. Throttled well below the
            # panel's frame rate: the status bar relays out the whole item on
            # every title change.
            self._status_ticks += 1
            every = max(1, int(_FPS / self._STATUS_FPS))
            if self._status_ticks % every == 0:
                self._refresh_status_item()

            # The red button asked to quit. Raise SIGINT rather than calling
            # NSApp.stop_(): stop_ does not unwind app.run() until the next
            # event arrives, so the app just hangs there. SIGINT reuses the
            # handler run_event_loop already installs — closing the window is
            # then EXACTLY Ctrl+C, down to the same shutdown path, instead of a
            # second way out that has to be kept working separately.
            if self.quit_requested:
                self.quit_requested = False
                signal.raise_signal(signal.SIGINT)
        except Exception:
            log.debug("Panel tick failed", exc_info=True)

    # ------------------------------------------------------------------
    # Listening toggle (the microphone gate)
    # ------------------------------------------------------------------

    def _handle_toggle_click(self) -> None:
        """Flip listening from a user click. Runs on the AppKit main thread."""
        self._listening = not self._listening
        self._listening_drawn = self._listening
        if self._toggle_view is not None:
            self._toggle_view.setNeedsDisplay_(True)
        log.info("Listening toggled -> %s", "ON" if self._listening else "OFF")
        self._fire_listening_callback()

    def _fire_listening_callback(self) -> None:
        """Invoke the registered callback, never letting it break the UI."""
        callback = self._listening_callback
        if callback is None:
            return
        try:
            callback(self._listening)
        except Exception:
            log.warning(
                "push-to-talk callback %r raised; ignoring", callback, exc_info=True
            )

    def _compute_target(self, state: State) -> np.ndarray:
        """Return the un-smoothed target bar heights for *state*.

        IDLE drives the bars from the LIVE microphone, quieter than during a
        capture. That is deliberate and it is the point of the panel: a flat
        line used to be drawn whether the microphone was healthy or dead, so
        the one question the HUD exists to answer — "is it hearing me?" — was
        the one thing it could not tell you. Now movement means the microphone
        is delivering audio, and a flat line means it is not.
        """
        if state is State.IDLE:
            samples = self._live_samples()
            if samples is None:
                # Nothing arriving: the mic is closed, paused, or denied. Flat
                # on purpose, and now it MEANS something.
                return np.full(_BAR_COUNT, 0.03, dtype=np.float32)
            # ALWAYS full scale. Height is a level meter and nothing else:
            # damping it while waiting for the wake phrase understated the
            # level and read as "distracted", when in fact the microphone is
            # attentive the entire time — it is listening for the wake phrase.
            # Whether it is RECORDING is a separate fact, so it gets a separate
            # channel: colour. See _draw_spectrum.
            return compute_bars(samples, _BAR_COUNT, self._sample_rate)

        if state is State.THINKING:
            # Subtle breathing pulse; no audio is flowing in either direction.
            breath = 0.20 + 0.16 * (math.sin(self._phase * 2.0) * 0.5 + 0.5)
            return _idle_bars(_BAR_COUNT, self._phase, breath)

        # LISTENING / SPEAKING — drive the bars from live audio when available.
        samples = self._live_samples()
        if samples is None:
            # No audio yet: keep a gentle animation instead of a dead panel.
            return _idle_bars(_BAR_COUNT, self._phase, 0.14)

        return compute_bars(samples, _BAR_COUNT, self._sample_rate)

    def set_capture_source(self, source) -> None:
        """Callable returning True while an utterance is being recorded."""
        self._capture_source = source

    def _is_capturing(self) -> bool:
        source = self._capture_source
        if source is None:
            return False
        try:
            return bool(source())
        except Exception:
            log.debug("Capture source raised", exc_info=True)
            return False

    def _live_samples(self):
        """Newest audio block, or None when nothing is arriving. Never raises."""
        source = self._samples_source
        if source is None:
            return None
        try:
            samples = source()
        except Exception:
            log.debug("Samples source raised", exc_info=True)
            return None
        if samples is None or len(samples) == 0:
            self._heard_audio = False
            return None
        self._heard_audio = True
        return samples

    # ------------------------------------------------------------------
    # Observer callback (state transitions from worker threads)
    # ------------------------------------------------------------------

    def __call__(self, old: State, new: State) -> None:
        """State observer: record the new state for the next render frame."""
        if not self.enabled:
            return
        self._current_state = new
        if old is State.SPEAKING and new is State.IDLE:
            # An answer was just given: a little wag of satisfaction. A
            # plain attribute write, picked up by the render tick.
            if self._mascot_celebrate is None:
                self._mascot_celebrate = "cheer"

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_samples_source(self, source: Callable[[], "np.ndarray | None"]) -> None:
        """Set the callable returning the most recent audio block to visualise.

        Args:
            source: Zero-argument callable returning a float32 mono numpy
                array, or None when no audio is currently available.
        """
        if not self.enabled:
            return
        self._samples_source = source

    def set_sample_rate(self, sample_rate: int) -> None:
        """Set the sample rate used to map FFT bins onto frequency bands."""
        if not self.enabled:
            return
        self._sample_rate = int(sample_rate)

    def _transcript_band(self) -> float:
        """Height the current exchange actually needs.

        Measured with the same wrapper the drawing uses, so the window can
        never be a line shorter than what is about to be drawn into it. A
        placeholder still gets one row, or the collapsed panel would have a
        band of zero height and no way to show the hint.
        """
        if not (self._question or self._answer):
            return _transcript_height(1)
        try:
            q, a = _transcript_lines(
                self._question, self._answer, self._current_width() - 2 * _SIDE_PAD
            )
            return _transcript_height(len(q) + len(a))
        except Exception:
            # No AppKit (tests on another platform): fall back to the maximum,
            # which is wrong-but-safe — too tall never clips text.
            return _transcript_height(_TRANSCRIPT_MAX_ROWS)

    def _adopt_manual_size(self, width: float, height: float) -> None:
        """Remember a size the user chose by dragging the grip."""
        self._manual_size = (float(width), float(height))
        log.info("Panel resized by hand to %.0fx%.0f", width, height)
        if self._view is not None:
            self._view.setNeedsDisplay_(True)

    def _current_width(self) -> float:
        """Live window width, or the default before there is a window."""
        if self._window is not None:
            try:
                return float(self._window.frame().size.width)
            except Exception:
                pass
        if self._manual_size is not None:
            return self._manual_size[0]
        return PANEL_WIDTH

    def get_size(self) -> tuple[float, float] | None:
        """Current window size in points, or None before :meth:`start`."""
        if self._window is None:
            return None
        try:
            size = self._window.frame().size
            return (float(size.width), float(size.height))
        except Exception:
            log.debug("Could not read panel size", exc_info=True)
            return None

    def set_size(self, width: float, height: float) -> None:
        """Restore a size saved from a previous session.

        Treated exactly like a drag: a restored size is still the user's
        choice, so the panel must stop resizing itself afterwards.
        """
        self._manual_size = (
            max(_MIN_WIDTH, float(width)),
            max(_MIN_HEIGHT, float(height)),
        )
        # Applied by the render tick, on the main thread. Restoring a saved
        # size happens before start() (no window yet) or from a caller with no
        # AppKit thread of its own, and resizing an NSWindow off the main
        # thread kills the process.
        self._pending_resize = True

    def _window_height(self) -> float:
        """Panel height for the current disclosure state.

        PANEL_HEIGHT is the collapsed height; expanding adds exactly the
        transcript band. Deriving it here rather than storing it means the
        window and the drawing can never disagree about how tall the panel is.
        """
        if not (self.show_transcript and self.transcript_expanded):
            # Collapsed: always the compact height, even after a manual resize.
            # Hiding the text and keeping its space would defeat the control.
            return PANEL_HEIGHT
        if self._manual_size is not None:
            return self._manual_size[1]
        return PANEL_HEIGHT + self._transcript_band()

    def _build_status_item(self) -> None:
        """Put an icon in the menu bar. Never fatal if it cannot be created.

        This is the ONLY way back to a panel that has gone behind another
        window. The app runs under the accessory activation policy, so it has
        no Dock icon by design — without this, a HUD you cannot see is a
        process you cannot reach.
        """
        try:
            bar = NSStatusBar.systemStatusBar()
            item = bar.statusItemWithLength_(NSVariableStatusItemLength)
            target = WawiMenuTarget.alloc().initWithOwner_(self)

            menu = NSMenu.alloc().init()
            for title, selector in (
                ("Ocultar el panel", "showPanel:"),
                ("Copiar la respuesta", "copyAnswer:"),
                (None, None),
                ("Escuchando", "toggleListening:"),
                ("Idioma", "toggleLanguage:"),
                (self._mascot_menu_title(), "toggleMascot:"),
                (None, None),
                (f"Cerrar {self.title_text or 'Wawi'}", "quitWawi:"),
            ):
                if title is None:
                    menu.addItem_(NSMenuItem.separatorItem())
                    continue
                entry = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    title, selector, ""
                )
                entry.setTarget_(target)
                menu.addItem_(entry)

            item.setMenu_(menu)
            # Wawi's head beside the live glyph. The glyph keeps doing all the
            # talking (setTitle_ in _refresh_status_item); she is the face of
            # it, so the item is recognisable among the other menu-bar icons.
            icon = self._load_image(_mascot.MENUBAR_ICON, 18.0)
            if icon is not None:
                try:
                    button = item.button()
                    button.setImage_(icon)
                    button.setImagePosition_(NSImageLeft)
                    self._status_image_set = True
                except Exception:
                    log.debug("Could not put Wawi in the menu bar", exc_info=True)
            self._status_item = item
            self._status_menu = menu
            # Held so PyObjC does not collect the target while the menu still
            # points at it — the menu holds only a weak reference.
            self._status_target = target
            self._refresh_status_item()
        except Exception:
            log.warning("Could not create the menu bar icon", exc_info=True)

    #: What the menu-bar glyph says when nothing is moving. The microphone gate
    #: comes first because it is the one thing you cannot infer from anywhere
    #: else on screen: a muted Wawi looks exactly like an idle one.
    _STATUS_GLYPHS = {
        "muted": "◌",
        State.LISTENING: "◉",
        State.THINKING: "◍",
        State.SPEAKING: "◎",
        State.IDLE: "●",
    }

    #: Eight heights driven by the LIVE microphone level. This is not
    #: decoration: it answers "is it hearing me?" from the menu bar, which is
    #: the one question you cannot ask a hidden panel. A frozen bar means a
    #: dead microphone, and that used to be invisible until you spoke into it.
    _LEVEL_BARS = "▁▂▃▄▅▆▇█"

    #: Braille spinner for THINKING — the only state with no level to show,
    #: because the microphone is closed while the model works.
    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧"

    #: Menu-bar refresh rate. Far below the panel's: the status bar redraws the
    #: whole item on every title change, and 30 fps of that is real CPU spent
    #: on something the size of a full stop.
    _STATUS_FPS = 10.0

    def _status_glyph(self) -> str:
        """The character the menu bar should be showing right now.

        Animated where there is something true to animate, static where there
        is not — a spinner that turns while nothing happens teaches you to stop
        looking at it.
        """
        if not self._listening:
            return self._STATUS_GLYPHS["muted"]

        state = self._current_state
        if state is State.THINKING:
            self._spin = (self._spin + 1) % len(self._SPINNER)
            return self._SPINNER[self._spin]

        if state in (State.LISTENING, State.IDLE, State.SPEAKING):
            # The same smoothed bars the panel draws, collapsed to one column.
            try:
                level = float(self._bars.max()) if self._bars.size else 0.0
            except Exception:
                level = 0.0
            if state is State.IDLE and level < 0.06:
                # Below the noise floor, hold a steady dot instead of jittering
                # on the room. A menu bar that twitches forever is noise.
                return self._STATUS_GLYPHS[State.IDLE]
            index = min(len(self._LEVEL_BARS) - 1, int(level * len(self._LEVEL_BARS)))
            return self._LEVEL_BARS[index]

        return self._STATUS_GLYPHS.get(state, "●")

    def _refresh_status_item(self) -> None:
        """Repaint the menu-bar glyph and the menu's live labels."""
        item = self._status_item
        if item is None:
            return
        try:
            glyph = self._status_glyph()
            if glyph != self._status_glyph_drawn:
                self._status_glyph_drawn = glyph
                item.button().setTitle_(glyph)
            menu = self._status_menu
            if menu is not None:
                menu.itemAtIndex_(0).setTitle_(
                    "Ocultar el panel" if self.panel_visible else "Mostrar el panel"
                )
                menu.itemAtIndex_(1).setEnabled_(bool(self._answer))
                menu.itemAtIndex_(3).setTitle_(
                    "Escuchando" if self._listening else "Silenciado"
                )
                menu.itemAtIndex_(4).setTitle_(
                    f"Idioma: {(self.language or 'es').upper()}"
                )
                menu.itemAtIndex_(5).setTitle_(self._mascot_menu_title())
        except Exception:
            log.debug("Could not refresh the menu bar icon", exc_info=True)

    def toggle_panel(self) -> None:
        """Show or hide the panel window, remembering which.

        Hiding it is a real mode, not a convenience: with the glyph animating
        on the live microphone level, the menu bar answers "is it hearing me?"
        on its own, and the panel becomes the thing you open to READ an answer
        rather than something that has to be on screen for Wawi to be usable.
        """
        self.panel_visible = not self.panel_visible
        self._pending_visibility = True
        self._refresh_status_item()

    def _apply_panel_visibility(self) -> None:
        """Order the window in or out. MAIN THREAD ONLY; never raises.

        Same rule as every other window operation here: only the render tick
        touches the window. A menu click already runs on the main thread, but
        having two ways in is precisely how the OTHER one ends up being called
        from a worker — which is how this file already lost a process twice.
        """
        window = self._window
        if window is None:
            return
        try:
            if self.panel_visible:
                # orderFrontRegardless, NOT makeKeyAndOrderFront: this window
                # can never become key, and the whole point of the accessory
                # policy is that showing the HUD never takes the caret out of
                # whatever you were typing in.
                window.orderFrontRegardless()
            else:
                window.orderOut_(None)
        except Exception:
            log.debug("Could not change the panel visibility", exc_info=True)

    def bring_to_front(self) -> None:
        """Make the panel visible, whatever it was doing before."""
        self.panel_visible = True
        self._pending_visibility = True

    # ------------------------------------------------------------------
    # Wawi — the mascot, the app icon and the little faces (main thread)
    # ------------------------------------------------------------------

    #: How often the Reduce Motion setting is re-read, in render ticks.
    _REDUCE_MOTION_EVERY = 60

    def _load_image(self, name: str, height: float | None = None) -> Any:
        """An NSImage for a bundled asset, or None. Never raises.

        Every caller treats None as "no Wawi here" and carries on: a missing
        or broken image costs the decoration, never the panel.
        """
        path = _mascot.asset_path(name)
        if path is None:
            return None
        try:
            image = NSImage.alloc().initWithContentsOfFile_(str(path))
        except Exception:
            image = None
        if image is None:
            if name not in _mascot._missing_logged:
                _mascot._missing_logged.add(name)
                log.warning("Could not load mascot image %s", path)
            return None
        if height is not None:
            size = image.size()
            if size.height > 0:
                image.setSize_((size.width * height / size.height, height))
        return image

    def _set_app_icon(self) -> None:
        """Wawi as the application icon, for alerts and dialogs.

        The accessory policy stays: there is still no Dock icon. This only
        changes the picture AppKit uses wherever it does show the app's icon.
        """
        image = self._load_image(_mascot.APP_ICON)
        if image is None:
            return
        try:
            self._app.setApplicationIconImage_(image)
        except Exception:
            log.debug("Could not set the application icon", exc_info=True)

    def _build_mascot(self) -> None:
        """Create Wawi's window. Never raises; on failure she just is not there."""
        if not self.show_mascot or self._mascot_window is not None:
            return
        if self._mascot_image is None and not self._mascot_frames:
            self._load_mascot_frames()
            image = None
            if not self._mascot_frames:
                image = self._load_image(_mascot.SPRITE)
                if image is None:
                    self.show_mascot = False
                    return
                self._mascot_image = image
            else:
                image = self._mascot_frames[_mascot.BASE_FRAME]
            size = image.size()
            w = _mascot.SPRITE_WIDTH
            h = w * size.height / max(1.0, size.width)
            self._mascot_sprite_size = (w, h)
        try:
            w, h = _mascot.window_size(*self._mascot_sprite_size)
            frame = NSMakeRect(0.0, 0.0, w, h)
            style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
            window = WawiPanelWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                frame, style, NSBackingStoreBuffered, False
            )
            window.setOpaque_(False)
            window.setBackgroundColor_(NSColor.clearColor())
            window.setHasShadow_(False)
            # Same level and Space behaviour as the panel, so the two always
            # travel together; as its child window she also stays ordered
            # just above it and moves with it while it is dragged.
            window.setLevel_(NSNormalWindowLevel)
            window.setIgnoresMouseEvents_(False)
            # Hover (petting) without ever becoming key; see WawiMascotView.
            window.setAcceptsMouseMovedEvents_(True)
            window.setHidesOnDeactivate_(False)
            window.setReleasedWhenClosed_(False)
            window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )
            view = WawiMascotView.alloc().initWithFrame_owner_(frame, self)
            window.setContentView_(view)
            self._mascot_window = window
            self._mascot_view = view
            self._mascot_shown = False
            self._reduce_motion = self._read_reduce_motion()
        except Exception:
            log.warning("Could not create Wawi's window; going on without her",
                        exc_info=True)
            self._mascot_window = None
            self._mascot_view = None
            self.show_mascot = False

    def _load_mascot_frames(self) -> None:
        """Load every animation frame present, once. Never raises.

        A frame that is missing or will not load is just left out; the
        animator falls back to a relative of it. Without the base frame the
        whole set is dropped and the single drawing is used.
        """
        frames: dict[str, Any] = {}
        try:
            for name in _mascot.available_frames():
                image = self._load_image(f"{_mascot.FRAME_DIR}/{name}.png")
                if image is not None:
                    frames[name] = image
        except Exception:
            log.warning("Could not load Wawi's frames; using the single drawing",
                        exc_info=True)
            frames = {}
        if _mascot.BASE_FRAME not in frames:
            frames = {}
        self._mascot_frames = frames
        self._mascot_animator.set_available(frames.keys())
        if frames:
            log.info("Wawi: %d animation frames loaded", len(frames))

    def celebrate(self, big: bool = False) -> None:
        """Make Wawi celebrate (a task got done, …). Any thread.

        Applied by the next render tick, and only once she is idle.
        """
        self._mascot_celebrate = "party" if big else (self._mascot_celebrate or "cheer")

    def _poke_mascot(self) -> None:
        """She was clicked (not dragged), nowhere in particular. Main thread."""
        self._mascot_animator.poke()

    def _mascot_uv(self, point: tuple[float, float]) -> tuple[float, float]:
        view = self._mascot_view
        width = float(view.bounds().size.width) if view is not None else 0.0
        return _mascot.sprite_uv(point, width, self._mascot_sprite_size, self._mascot_pose)

    def _mascot_zone(self, point: tuple[float, float] | None) -> str | None:
        if point is None:
            return None
        anim = self._mascot_animator
        u, v = self._mascot_uv(point)
        return _mascot.touch_zone(u, v, mirrored=anim.mirrored, lying=anim.lying)

    def _touch_mascot(self, point: tuple[float, float] | None, clicks: int = 1) -> None:
        """A click on her at *point* (view points). Main thread; never raises."""
        try:
            anim = self._mascot_animator
            if clicks >= 2:
                anim.double_click()
            else:
                anim.touch(self._mascot_zone(point))
        except Exception:
            log.debug("Mascot touch failed", exc_info=True)

    def _hover_mascot(self, point: tuple[float, float] | None) -> None:
        """The cursor moved over her (no button), or left her (None)."""
        if point is None:
            self._mascot_pet.reset()
            return
        u, _v = self._mascot_uv(point)
        self._mascot_pet.feed(self._mascot_animator.t, u, self._mascot_zone(point))

    def _begin_drag_mascot(self) -> None:
        """A press turned into a drag: a dash in progress stops right there."""
        anim = self._mascot_animator
        if anim.excursion != 0.0:
            anim.interrupt()

    def _mascot_cursor_offset(self) -> tuple[float, float] | None:
        """Where the cursor is from her head, in points (y up). Main thread."""
        window = self._mascot_window
        if window is None or self._mascot_dragging:
            return None
        try:
            mx, my = NSEvent_mouse_location()
            f = window.frame()
        except Exception:
            return None
        w, h = self._mascot_sprite_size
        pose = self._mascot_pose
        hx = f.origin.x + f.size.width / 2.0 + pose.dx + (_mascot.HEAD_UV[0] - 0.5) * w
        hy = f.origin.y + _mascot.PAD_BOTTOM + pose.dy + _mascot.HEAD_UV[1] * h
        return (mx - hx, my - hy)

    def _apply_mascot_excursion(self, attached: bool) -> None:
        """Carry her window along on a dash, and back. Main thread.

        The run moves the WINDOW (her padding is far too small to run in),
        never off her screen, and only a little (mascot.RUN_DISTANCE). If a
        dash is cut short she slides home over a few frames instead of
        jumping.
        """
        window = self._mascot_window
        if window is None:
            return
        want = 0.0 if self._mascot_dragging else self._mascot_animator.excursion
        cur = self._mascot_exc
        new = want if abs(want - cur) <= 3.0 else cur + (want - cur) * 0.3
        if abs(new) < 0.25 and want == 0.0:
            new = 0.0
        f = window.frame()
        base = f.origin.x - cur
        try:
            screen = window.screen() or NSScreen.mainScreen()
            if screen is not None and new != 0.0:
                v = screen.visibleFrame()
                lo = v.origin.x - base
                hi = v.origin.x + v.size.width - f.size.width - base
                new = min(max(new, min(0.0, lo)), max(0.0, hi))
        except Exception:
            pass
        if new == cur:
            return
        self._mascot_exc = new
        if not attached:
            window.setFrameOrigin_(NSMakePoint(base + new, f.origin.y))

    def _close_mascot(self) -> None:
        window = self._mascot_window
        self._mascot_window = None
        self._mascot_view = None
        self._mascot_shown = False
        self._mascot_attached = None
        if window is None:
            return
        try:
            parent = window.parentWindow()
            if parent is not None:
                parent.removeChildWindow_(window)
            window.orderOut_(None)
            window.close()
        except Exception:
            log.debug("Error closing Wawi's window", exc_info=True)

    @staticmethod
    def _read_reduce_motion() -> bool:
        try:
            return bool(
                NSWorkspace.sharedWorkspace().accessibilityDisplayShouldReduceMotion()
            )
        except Exception:
            return False

    def _mascot_level(self) -> float:
        try:
            return float(self._bars.max()) if self._bars.size else 0.0
        except Exception:
            return 0.0

    def _tick_mascot(self) -> None:
        """Show, hide, place and animate Wawi. Main thread; never raises.

        She no longer depends on the panel: with the panel hidden she stands
        on her own (and is then the only thing on screen saying what the
        assistant is doing), and when it comes back she returns beside it.
        """
        if self._window is None:
            return
        try:
            wanted = bool(self.show_mascot)
            if wanted and self._mascot_window is None:
                self._build_mascot()
            window = self._mascot_window
            if window is None:
                return

            if wanted != self._mascot_shown:
                self._mascot_shown = wanted
                if wanted:
                    self._mascot_attached = None  # re-decided just below
                else:
                    self._detach_mascot()
                    window.orderOut_(None)
            if not wanted:
                return

            attached = bool(self.panel_visible)
            if attached != self._mascot_attached:
                self._set_mascot_attached(attached)

            if self._status_ticks % self._REDUCE_MOTION_EVERY == 0:
                self._reduce_motion = self._read_reduce_motion()
                if not attached and not self._mascot_dragging:
                    # A screen can disappear under her (a monitor unplugged);
                    # this is cheap enough to check once every two seconds.
                    self._place_mascot_alone(clamp=True)
            celebrate = self._mascot_celebrate
            if celebrate is not None:
                self._mascot_celebrate = None
                self._mascot_animator.celebrate(big=celebrate == "party")
            anim = self._mascot_animator
            anim.set_petting(self._mascot_pet.stroking(anim.t))
            anim.look_at(self._mascot_cursor_offset())
            self._mascot_pose = self._mascot_animator.update(
                _FRAME_INTERVAL,
                self._current_state,
                muted=not self._listening,
                level=self._mascot_level(),
                reduce_motion=self._reduce_motion,
                voice=self._voice_level(),
            )
            self._apply_mascot_excursion(attached)
            if attached:
                # Placed every frame: the panel collapses, grows and is
                # restored from code paths that know nothing about her.
                self._place_mascot()
            anim = self._mascot_animator
            drawn = (self._mascot_pose, anim.frame, anim.prev_frame, anim.frame_mix,
                     anim.mirrored)
            if drawn != self._mascot_drawn and self._mascot_view:
                self._mascot_drawn = drawn
                self._mascot_view.setNeedsDisplay_(True)
        except Exception:
            log.debug("Mascot tick failed", exc_info=True)

    def _voice_level(self) -> float | None:
        """What she is saying right now, 0..1, or None when unknown/silent."""
        source = self._voice_level_source
        if source is None or self._current_state is not State.SPEAKING:
            return None
        try:
            value = source()
        except Exception:
            return None
        return None if value is None else float(value)

    def set_voice_level_source(self, source: "Callable[[], float | None] | None") -> None:
        """Callable returning the level of the speech playing now (0..1) or None.

        THREADING: called from the main-thread render tick; it must be cheap
        and must not block. ``Voice.output_level`` is exactly that.
        """
        self._voice_level_source = source

    def _set_mascot_attached(self, attached: bool) -> None:
        """Move her between "by the panel" and "on her own". Main thread."""
        window = self._mascot_window
        if window is None:
            return
        self._mascot_attached = attached
        if attached:
            # Next time she is alone she goes back to HER spot, not to
            # wherever the panel happened to put her.
            self._mascot_alone_placed = False
            # Same level as the panel, and its child window: ordered just
            # above it and moved with it while it is dragged.
            window.setLevel_(NSNormalWindowLevel)
            self._place_mascot()
            window.orderFrontRegardless()
            if window.parentWindow() is None:
                self._window.addChildWindow_ordered_(window, 1)  # above
        else:
            # A child window is ordered out with its parent, which is exactly
            # how she used to vanish with the panel. Cut the tie first.
            self._detach_mascot()
            # Floating, unlike the panel: alone she is the whole interface,
            # and a status you cannot see behind an editor is no status. She
            # is small and can be dragged out of the way.
            window.setLevel_(NSFloatingWindowLevel)
            self._place_mascot_alone(clamp=True)
            window.orderFrontRegardless()

    def _detach_mascot(self) -> None:
        window = self._mascot_window
        if window is None:
            return
        parent = window.parentWindow()
        if parent is not None:
            parent.removeChildWindow_(window)

    @staticmethod
    def _screen_rects() -> tuple[list[tuple[float, float, float, float]], Any]:
        """Visible frames of every screen, and the main screen's (or None)."""
        rects = []
        for screen in NSScreen.screens() or []:
            v = screen.visibleFrame()
            rects.append((v.origin.x, v.origin.y, v.size.width, v.size.height))
        main = NSScreen.mainScreen()
        main_rect = None
        if main is not None:
            v = main.visibleFrame()
            main_rect = (v.origin.x, v.origin.y, v.size.width, v.size.height)
        return rects, main_rect

    def _place_mascot_alone(self, clamp: bool) -> None:
        """Put her at her own spot. *clamp* keeps her fully on a screen."""
        window = self._mascot_window
        if window is None:
            return
        mine = window.frame()
        size = (mine.size.width, mine.size.height)
        if clamp:
            try:
                rects, main_rect = self._screen_rects()
            except Exception:
                rects, main_rect = [], None
            base = self._mascot_alone_origin
            if base is None and self._mascot_alone_placed:
                # Already standing in the default corner: stay there. "Main
                # screen" is the one with focus, so recomputing the default
                # every check would make her hop monitors as you switch apps.
                base = (float(mine.origin.x) - self._mascot_exc, float(mine.origin.y))
            spot = _mascot.alone_origin(base, size, rects, main_rect)
            if spot is None:
                return
            self._mascot_alone_placed = True
            if self._mascot_alone_origin is not None:
                self._mascot_alone_origin = spot
        else:
            spot = self._mascot_alone_origin
            if spot is None:
                return
        x, y = spot[0] + self._mascot_exc, spot[1]
        if abs(x - mine.origin.x) > 0.5 or abs(y - mine.origin.y) > 0.5:
            window.setFrameOrigin_(NSMakePoint(x, y))

    def _place_mascot(self) -> None:
        window = self._mascot_window
        if window is None or self._window is None:
            return
        panel = self._window.frame()
        screen_rect = None
        try:
            screen = self._window.screen() or NSScreen.mainScreen()
            if screen is not None:
                v = screen.visibleFrame()
                screen_rect = (v.origin.x, v.origin.y, v.size.width, v.size.height)
        except Exception:
            pass
        mine = window.frame()
        x, y = _mascot.mascot_origin(
            (panel.origin.x, panel.origin.y, panel.size.width, panel.size.height),
            (mine.size.width, mine.size.height),
            screen_rect,
            self._mascot_offset,
        )
        x += self._mascot_exc  # a dash in progress
        if abs(x - mine.origin.x) > 0.5 or abs(y - mine.origin.y) > 0.5:
            window.setFrameOrigin_(NSMakePoint(x, y))

    def _drag_mascot(self, dx: float, dy: float) -> None:
        """Move Wawi by a drag delta. Main thread.

        By the panel the delta is kept relative to it; alone it moves her
        own spot, unclamped while the mouse is down so she can be carried
        across to another screen (clamped on release, in _end_drag_mascot).
        """
        if self._mascot_attached is False:
            window = self._mascot_window
            if window is None:
                return
            self._mascot_dragging = True
            if self._mascot_alone_origin is None:
                o = window.frame().origin
                self._mascot_alone_origin = (float(o.x) - self._mascot_exc, float(o.y))
            x, y = self._mascot_alone_origin
            self._mascot_alone_origin = (x + dx, y + dy)
            self._place_mascot_alone(clamp=False)
            return
        ox, oy = self._mascot_offset
        limit = 800.0
        self._mascot_offset = (
            max(-limit, min(limit, ox + dx)),
            max(-limit, min(limit, oy + dy)),
        )
        self._place_mascot()

    def _end_drag_mascot(self) -> None:
        """Mouse released: keep her where she was dropped, fully on screen."""
        self._mascot_dragging = False
        if self._mascot_attached is False:
            self._place_mascot_alone(clamp=True)

    def _mascot_menu(self, event: Any, view: Any) -> None:
        """Open the menu-bar item's menu at the click. Main thread."""
        menu = self._status_menu
        if menu is None:
            return
        self._refresh_status_item()
        NSMenu.popUpContextMenu_withEvent_forView_(menu, event, view)

    def get_mascot_origin(self) -> tuple[float, float] | None:
        """Her own spot (panel hidden), or None while it is the default.

        The caller persists it, exactly like :meth:`get_origin`.
        """
        if self._mascot_alone_origin is None:
            return None
        x, y = self._mascot_alone_origin
        return (float(x), float(y))

    def set_mascot_origin(self, x: float, y: float) -> None:
        """Restore her own spot. Safe before :meth:`start`; clamped on use."""
        self._mascot_alone_origin = (float(x), float(y))
        if self._mascot_attached is False:
            self._place_mascot_alone(clamp=True)

    def toggle_mascot(self) -> None:
        """Show or hide Wawi (the menu-bar item). Applied by the next tick."""
        self.show_mascot = not self.show_mascot
        self._refresh_status_item()

    def _mascot_menu_title(self) -> str:
        name = (self.title_text or "Wawi").strip() or "Wawi"
        name = name[:1].upper() + name[1:]
        return f"Ocultar a {name}" if self.show_mascot else f"Mostrar a {name}"

    def copy_answer(self) -> bool:
        """Put the last answer on the clipboard. MAIN THREAD; never raises.

        Returns True when something was copied. The answer is copied as it was
        SPOKEN — no timestamp, no "Wawi:" prefix — because what you want is to
        paste the content somewhere, not a transcript of the exchange.
        """
        answer = (self._answer or "").strip()
        if not answer:
            log.info("Nothing to copy yet")
            return False
        try:
            board = NSPasteboard.generalPasteboard()
            # clearContents is required: without it the write is appended to
            # whatever declaration was there and a stale type can win the paste.
            board.clearContents()
            board.setString_forType_(answer, NSPasteboardTypeString)
        except Exception:
            log.warning("Could not write to the clipboard", exc_info=True)
            return False
        self._copied_at = time.monotonic()
        if self._copy_view is not None:
            self._copy_view.setNeedsDisplay_(True)
        log.info("Copied %d chars to the clipboard", len(answer))
        return True

    def copy_confirmed(self) -> bool:
        """True while the pill should still be saying "copiado"."""
        if self._copied_at is None:
            return False
        return (time.monotonic() - self._copied_at) < _COPIED_FEEDBACK_S

    def _handle_lang_click(self) -> None:
        """Flip ES/EN from a click. Runs on the AppKit main thread."""
        self.language = "en" if self.language == "es" else "es"
        if self._lang_view is not None:
            self._lang_view.setNeedsDisplay_(True)
        log.info("Language toggled -> %s", self.language)
        cb = self._lang_cb
        if cb is None:
            return
        try:
            cb(self.language)
        except Exception:
            # A failing callback must not take the render loop down with it.
            log.warning("Language callback failed", exc_info=True)

    def on_language_changed(self, cb: "Callable[[str], None]") -> None:
        """Register the callback fired when the user clicks the ES/EN pill.

        THREADING: invoked on the AppKit **main thread**, like the push-to-talk
        callback. Keep it fast; hop to a worker for anything slow.
        """
        self._lang_cb = cb

    def set_language(self, code: str) -> None:
        """Show *code* on the pill without firing the callback."""
        if not self.enabled:
            return
        self.language = code
        if self._lang_view is not None:
            self._lang_view.setNeedsDisplay_(True)

    def _handle_window_button(self, index: int) -> None:
        """Act on a traffic light. Runs on the AppKit main thread.

        Args:
            index: 0 close, 1 minimise, 2 zoom — the macOS order.
        """
        if index == 0:
            # The panel does NOT kill the process itself: the orchestrator owns
            # the microphone, the model and the log, and it is the only thing
            # that knows how to put them down cleanly.
            log.info("Quit requested from the panel")
            self.quit_requested = True
            return
        if index == 1:
            if self.transcript_expanded:
                self._handle_chevron_click()
            return
        self._handle_zoom()

    def _handle_zoom(self) -> None:
        """Toggle between the user's size and a large one."""
        if self._zoom_restore is not None:
            width, height = self._zoom_restore
            self._zoom_restore = None
        else:
            self._zoom_restore = self._manual_size or self.get_size()
            width, height = _ZOOM_W, _ZOOM_H
            try:
                # Never larger than the screen it is on. The fallback is not an
                # error path: before start() there is no window to ask.
                visible = self._window.screen().visibleFrame()
                width = min(visible.size.width - 48.0, _ZOOM_W)
                height = min(visible.size.height - 48.0, _ZOOM_H)
            except Exception:
                log.debug("Zoom could not read the screen; using the default")
        # Zooming to a big window that shows no text would be absurd.
        self.transcript_expanded = True
        self.set_size(width, height)

    def _handle_chevron_click(self) -> None:
        """Toggle the transcript and resize the window around its TOP edge.

        The top edge is the anchor because the panel lives against the top of
        the screen: growing from the bottom-left origin AppKit actually stores
        would slide the whole HUD upward, off the screen, every time an answer
        arrived. Runs on the AppKit main thread (it comes from a click).
        """
        self.transcript_expanded = not self.transcript_expanded
        # One rule, no exceptions: only the render tick resizes the window.
        # A click already runs on the main thread, but having two paths into
        # setFrame is how the OTHER one ends up being called from a worker.
        self._pending_resize = True
        if self._chevron_view is not None:
            self._chevron_view.setNeedsDisplay_(True)

    def _resize_to_content(self) -> None:
        """Match the window to :meth:`_window_height`, anchored at the top."""
        window = self._window
        if window is None:
            return
        frame = window.frame()
        target_h = self._window_height()
        # Width only ever changes because the user asked for it; _window_height
        # already returns THEIR height once they have, so a manual size makes
        # this a no-op instead of a fight.
        target_w = (
            self._manual_size[0] if self._manual_size is not None
            else frame.size.width
        )
        if (abs(target_h - frame.size.height) < 0.5
                and abs(target_w - frame.size.width) < 0.5):
            return
        top = frame.origin.y + frame.size.height
        window.setFrame_display_animate_(
            NSMakeRect(
                frame.origin.x,
                top - target_h,          # keep the TOP edge where it is
                target_w,
                target_h,
            ),
            True,
            False,
        )
        if self._view is not None:
            self._view.setNeedsDisplay_(True)

    def bottom_block_height(self) -> float:
        """Height of whichever block sits between the spectrum and the strip.

        The spectrum and the separator both derive their geometry from this, so
        it has to be one function rather than two matching constants.
        """
        if self.show_transcript and self.transcript_expanded:
            if self._manual_size is not None:
                return max(_transcript_height(1), self._manual_size[1] - PANEL_HEIGHT)
            return self._transcript_band()
        return 0.0

    def set_texts(self, user_text: str | None = None, response: str | None = None) -> None:
        """Update the transcript shown on the panel.

        Args:
            user_text: What the STT heard. Passing it clears the previous
                answer: leaving the old one under the new question would show
                an exchange that never happened.
            response: What the assistant answered.
        """
        if not self.enabled:
            return
        if user_text is not None:
            # The finished exchange moves into history before the new question
            # replaces it, so the panel reads as a conversation rather than as
            # a field that keeps being overwritten.
            if self._question and self._answer:
                self._history.append((self._question, self._answer))
                del self._history[:-_HISTORY_MAX]
            self._question = user_text
            self._answer = ""
        if response is not None:
            self._answer = response
            # A new answer ends the previous confirmation: leaving "copiado" up
            # would claim THIS answer is on the clipboard when the old one is.
            self._copied_at = None
        # The band is sized to the text, so a new exchange changes the window
        # height — but set_texts is called from the orchestrator's WORKER
        # thread, and resizing an NSWindow off the main thread kills the
        # process. Flag it and let the render tick do it, exactly as the
        # push-to-talk repaint already does.
        self._pending_resize = True

    def set_task(self, active: bool, line: str = "") -> None:
        """Show (or hide) the delegated agent's progress on the strip.

        Safe from any thread and before :meth:`start`: it writes two plain
        attributes that the render timer reads on its next frame. Nothing here
        touches AppKit, which is the whole reason it is safe — the executor
        publishes from its own thread and an NSWindow call from there would
        kill the process.
        """
        if not self.enabled:
            return
        self.task_active = bool(active)
        if line:
            self.task_line = line
        elif not active:
            self.task_line = ""

    def listening(self) -> bool:
        """Return True when the microphone gate is open.

        True  = Wawi hears you.
        False = deaf; speech is ignored until it is switched back on.

        Always False when the panel is disabled.
        """
        if not self.enabled:
            return False
        return self._listening

    def set_listening(self, enabled: bool) -> None:
        """Set the microphone gate programmatically.

        Safe to call from any thread and before :meth:`start`. Deliberately
        does NOT fire the callback registered with
        :meth:`on_listening_changed` — that callback reports *user* clicks, so
        a caller syncing its own state never gets its own change echoed back.
        The switch repaints on the next render frame.
        """
        if not self.enabled:
            return
        self._listening = bool(enabled)

    def on_listening_changed(self, cb: Callable[[bool], None]) -> None:
        """Register the callback fired when the user clicks the switch.

        The callback receives the new value (True = listening).

        THREADING: it is invoked on the AppKit **main thread**, inside event
        handling. Keep it fast and non-blocking — anything slow will stall the
        render loop and the whole UI. Hop to a worker thread for real work::

            def _on_listening(on: bool) -> None:
                threading.Thread(target=apply_gate, args=(on,), daemon=True).start()

            panel.on_listening_changed(_on_listening)

        Exceptions raised by the callback are caught and logged, never
        propagated into AppKit (same contract as ``StateMachine`` observers).
        """
        if not self.enabled:
            return
        self._listening_callback = cb

    # ------------------------------------------------------------------
    # Position (two-monitor persistence — the caller saves/restores these)
    # ------------------------------------------------------------------

    def get_origin(self) -> tuple[float, float] | None:
        """Return the panel's current bottom-left origin in screen points.

        Returns None when the panel is disabled or not started. The caller is
        responsible for persisting this value (this module never writes config).
        """
        if not self.enabled or self._window is None:
            return None
        try:
            origin = self._window.frame().origin
            return (float(origin.x), float(origin.y))
        except Exception:
            log.debug("Could not read panel origin", exc_info=True)
            return None

    def set_origin(self, x: float, y: float) -> None:
        """Move the panel to the given bottom-left screen origin.

        Safe to call before :meth:`start`; the position is applied when the
        window is created.
        """
        if not self.enabled:
            return
        if self._window is None:
            self._pending_origin = (float(x), float(y))
            return
        try:
            self._window.setFrameOrigin_(NSMakePoint(float(x), float(y)))
        except Exception:
            log.debug("Could not set panel origin", exc_info=True)

    # ------------------------------------------------------------------
    # Verification helper
    # ------------------------------------------------------------------

    def is_key_window(self) -> bool:
        """Return True if the panel currently holds keyboard focus.

        Must always be False — the panel is designed never to steal focus.
        """
        if self._window is None:
            return False
        try:
            return bool(self._window.isKeyWindow())
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Main-thread event loop helper
# ---------------------------------------------------------------------------


def run_event_loop() -> None:
    """Park the main thread, servicing AppKit when the panel is available.

    Drop-in replacement for the ``threading.Event().wait()`` park at
    ``wawi/orchestrator.py:152``. Falls back to that exact behaviour when
    AppKit is unavailable, so non-macOS and panel-off paths are unchanged.

    Installs a SIGINT handler so Ctrl+C unwinds ``NSApp.run()`` cleanly
    (see "Finding 4a" in the module docstring).
    """
    if not _HAS_APPKIT:
        threading.Event().wait()
        return

    app = NSApplication.sharedApplication()
    previous = None

    def _on_sigint(_signum, _frame):  # type: ignore[no-untyped-def]
        log.info("SIGINT received — stopping the AppKit run loop")
        try:
            app.stop_(None)
        except Exception:
            log.warning("Could not stop NSApp cleanly", exc_info=True)
        raise KeyboardInterrupt

    try:
        previous = signal.signal(signal.SIGINT, _on_sigint)
    except ValueError:
        # Not on the main thread — signal handlers cannot be installed.
        log.debug("Could not install SIGINT handler (not the main thread)")

    try:
        app.run()
    finally:
        if previous is not None:
            try:
                signal.signal(signal.SIGINT, previous)
            except ValueError:
                pass
