"""Tests for the panel's spectrum semantics.

No window is created: PanelObserver(enabled=False) skips AppKit entirely and
the bar computation under test does not need one. The public setters no-op
while disabled, so the sources are wired directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from wawi.panel import (
    _BAR_COUNT,
    _CAPTURE_COLOR,
    _MONITOR_COLOR,
    PanelObserver,
    compute_bars,
)
from wawi.state import State


@pytest.fixture
def panel():
    p = PanelObserver(enabled=False)
    return p


def audio(amplitude=0.35, n=1024):
    rng = np.random.default_rng(7)
    return (rng.standard_normal(n) * amplitude).astype(np.float32)


class TestSpectrumTellsTheTruth:
    """Height is a level meter and nothing else.

    An earlier version damped the resting spectrum to distinguish "monitoring"
    from "recording". That understated the level and read as inattentive, when
    the microphone is in fact listening for the wake phrase the whole time. A
    meter that lowers the level it displays is not a meter.
    """

    def test_the_resting_spectrum_reports_the_real_level(self, panel):
        block = audio()
        panel._samples_source = lambda: block
        panel._capture_source = lambda: False
        np.testing.assert_allclose(
            panel._compute_target(State.IDLE),
            compute_bars(block, _BAR_COUNT, panel._sample_rate),
        )

    def test_recording_does_not_change_the_height(self, panel):
        block = audio()
        panel._samples_source = lambda: block
        panel._capture_source = lambda: True
        np.testing.assert_allclose(
            panel._compute_target(State.IDLE),
            compute_bars(block, _BAR_COUNT, panel._sample_rate),
        )

    def test_a_louder_room_gives_taller_bars(self, panel):
        panel._capture_source = lambda: False
        panel._samples_source = lambda: audio(0.05)
        quiet = panel._compute_target(State.IDLE).max()
        panel._samples_source = lambda: audio(0.5)
        loud = panel._compute_target(State.IDLE).max()
        assert loud > quiet

    def test_no_audio_is_flat_and_that_is_the_only_flat_case(self, panel):
        """Flat must mean deaf, so it can be trusted to mean deaf."""
        panel._samples_source = lambda: None
        panel._capture_source = lambda: False
        bars = panel._compute_target(State.IDLE)
        assert bars.max() <= 0.03

    def test_an_empty_block_counts_as_no_audio(self, panel):
        panel._samples_source = lambda: np.zeros(0, dtype=np.float32)
        assert panel._compute_target(State.IDLE).max() <= 0.03

    def test_a_broken_samples_source_never_raises(self, panel):
        def boom():
            raise RuntimeError("mic exploded")

        panel._samples_source = boom
        assert panel._compute_target(State.IDLE).max() <= 0.03


class TestCaptureIsShownInColour:
    def test_the_two_states_use_different_colours(self):
        assert _MONITOR_COLOR != _CAPTURE_COLOR

    def test_a_broken_capture_source_reads_as_not_recording(self, panel):
        def boom():
            raise RuntimeError("nope")

        panel._capture_source = boom
        assert panel._is_capturing() is False

    def test_no_capture_source_reads_as_not_recording(self, panel):
        assert panel._is_capturing() is False

    def test_the_capture_source_is_consulted(self, panel):
        panel._capture_source = lambda: True
        assert panel._is_capturing() is True


class TestMicLiveness:
    def test_the_panel_starts_assuming_the_mic_is_dead(self, panel):
        """It must never claim to be hearing before a block has arrived."""
        from wawi.panel import _MIC_DEAD_TICKS

        assert panel._silent_ticks == _MIC_DEAD_TICKS

    def test_receiving_audio_marks_the_mic_live(self, panel):
        panel._samples_source = lambda: audio()
        panel._live_samples()
        assert panel._heard_audio is True

    def test_losing_audio_marks_the_mic_dead(self, panel):
        panel._samples_source = lambda: None
        panel._live_samples()
        assert panel._heard_audio is False


class TestTranscript:
    """The answer ON SCREEN.

    The terminal transcript only helps whoever is looking at the terminal, and
    the entire point of a floating HUD is that you are not. These pin the state
    the drawing reads; the drawing itself needs AppKit and is exercised by
    running the panel.
    """

    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        return obs

    def test_a_new_question_clears_the_previous_answer(self):
        # Otherwise the panel shows the old answer under the new question —
        # an exchange that never happened, and the most misleading thing a
        # status display can do.
        obs = self._observer()
        obs.set_texts(user_text="cuánto es dos más dos", response="Cuatro.")
        obs.set_texts(user_text="y tres más tres")
        assert obs._question == "y tres más tres"
        assert obs._answer == ""

    def test_the_answer_is_kept_without_touching_the_question(self):
        obs = self._observer()
        obs.set_texts(user_text="hola")
        obs.set_texts(response="Hola, ¿en qué andás?")
        assert obs._question == "hola"
        assert obs._answer == "Hola, ¿en qué andás?"

    def test_a_disabled_panel_stores_nothing(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(False)
        obs.set_texts(user_text="hola", response="chau")
        assert obs._question == ""
        assert obs._answer == ""


class TestCollapse:
    """The transcript can be hidden, and hiding it shrinks the window.

    The anchor is the load-bearing part. AppKit stores a window's origin at its
    BOTTOM-left, but this panel lives against the TOP of the screen. Resizing
    naively keeps the bottom edge and slides the whole HUD upward — off the
    screen — every time an answer arrives.
    """

    class _FakeRect:
        def __init__(self, x, y, w, h):
            self.origin = type("O", (), {"x": x, "y": y})()
            self.size = type("S", (), {"width": w, "height": h})()

    class _FakeWindow:
        def __init__(self, rect):
            self._rect = rect
            self.applied = None

        def frame(self):
            return self._rect

        def setFrame_display_animate_(self, rect, _display, _animate):  # noqa: N802
            self.applied = rect

    def _observer(self, height):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        obs._window = self._FakeWindow(self._FakeRect(1516.0, 798.0, 380.0, height))
        return obs

    def test_collapsing_keeps_the_top_edge_exactly(self):
        from wawi import panel

        obs = self._observer(panel.PANEL_HEIGHT)
        obs.set_texts(user_text="cuánto es dos más dos", response="Cuatro.")
        expanded_h = obs._window_height()
        obs._window._rect = self._FakeRect(1516.0, 798.0, 380.0, expanded_h)
        top_before = 798.0 + expanded_h

        obs._handle_chevron_click()
        obs._tick()                       # the tick is what resizes

        applied = obs._window.applied
        assert obs.transcript_expanded is False
        assert applied.size.height == panel.PANEL_HEIGHT
        assert applied.origin.y + applied.size.height == top_before

    def test_expanding_also_keeps_the_top_edge(self):
        from wawi import panel

        obs = self._observer(panel.PANEL_HEIGHT)
        obs.transcript_expanded = False
        obs.set_texts(user_text="cuánto es dos más dos", response="Cuatro.")
        obs._window._rect = self._FakeRect(1516.0, 798.0, 380.0, panel.PANEL_HEIGHT)
        obs._window.applied = None
        top_before = 798.0 + panel.PANEL_HEIGHT

        obs._handle_chevron_click()
        obs._tick()                       # the tick is what resizes

        applied = obs._window.applied
        assert obs.transcript_expanded is True
        assert applied.size.height == panel.PANEL_HEIGHT + obs._transcript_band()
        assert applied.origin.y + applied.size.height == top_before

    def test_a_collapsed_panel_reserves_no_band(self):
        obs = self._observer(150.0)
        obs.transcript_expanded = False
        assert obs.bottom_block_height() == 0.0


class TestBandFollowsTheText:
    """The band is as tall as the answer, and never shorter.

    A fixed band reserved the worst case on every turn, so a two-line answer
    left half the panel as empty grey and a long one was silently clipped.
    Both are failures of the same missing measurement.
    """

    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        return obs

    def test_a_longer_answer_needs_a_taller_band(self):
        short = self._observer()
        short.set_texts(user_text="hola", response="Sí.")

        long = self._observer()
        long.set_texts(
            user_text="hola",
            response=(
                "En C no hay un booleano nativo. Usás 1 para verdadero, o true "
                "si incluís stdbool.h. En C99 en adelante stdbool.h define bool, "
                "true y false como macros sobre int, así que el tipo real sigue "
                "siendo un entero."
            ),
        )

        assert long._transcript_band() > short._transcript_band()

    def test_the_band_never_exceeds_the_row_budget(self):
        from wawi import panel

        obs = self._observer()
        obs.set_texts(user_text="hola", response="palabra " * 400)
        assert obs._transcript_band() <= panel._transcript_height(
            panel._TRANSCRIPT_MAX_ROWS
        )

    def test_an_empty_transcript_still_reserves_its_hint_row(self):
        from wawi import panel

        obs = self._observer()
        assert obs._transcript_band() == panel._transcript_height(1)

    def test_the_window_is_the_base_height_plus_the_band(self):
        from wawi import panel

        obs = self._observer()
        obs.set_texts(user_text="hola", response="Sí.")
        assert obs._window_height() == panel.PANEL_HEIGHT + obs._transcript_band()


class TestThreadSafety:
    """set_texts is called from the orchestrator's WORKER thread.

    Resizing an NSWindow off the AppKit main thread kills the process — the
    panel took Wawi down on the second answer. The fix is the same one the
    push-to-talk repaint already uses: flag it here, act on it in the tick.
    """

    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        return obs

    def test_set_texts_does_not_touch_the_window(self):
        touched = []

        class _Exploding:
            def frame(self):
                touched.append("frame")
                raise AssertionError("resized from the calling thread")

            def setFrame_display_animate_(self, *a):  # noqa: N802
                touched.append("setFrame")
                raise AssertionError("resized from the calling thread")

        obs = self._observer()
        obs._window = _Exploding()
        obs.set_texts(user_text="hola", response="Sí.")

        assert touched == []
        assert obs._pending_resize is True

    def test_the_tick_consumes_the_flag_once(self):
        obs = self._observer()
        calls = []
        obs._resize_to_content = lambda: calls.append(1)  # type: ignore[method-assign]

        obs.set_texts(user_text="hola", response="Sí.")
        obs._tick()
        obs._tick()

        assert calls == [1]
        assert obs._pending_resize is False


class TestListeningGate:
    """The switch is the microphone gate: ON hears you, OFF does not.

    It used to mean "push-to-talk only" — a mode SELECTOR whose label named one
    of the two modes, so it could never say which one was active. The gate has
    no such ambiguity, which is the point of the change.
    """

    def _observer(self):
        from wawi.panel import PanelObserver

        return PanelObserver(True)

    def test_it_starts_listening(self):
        # A HUD that starts deaf and says nothing about it is indistinguishable
        # from a broken microphone.
        assert self._observer().listening() is True

    def test_a_click_flips_it_and_reports_the_new_value(self):
        seen = []
        obs = self._observer()
        obs.on_listening_changed(seen.append)

        obs._handle_toggle_click()
        assert obs.listening() is False
        obs._handle_toggle_click()
        assert obs.listening() is True

        assert seen == [False, True]

    def test_setting_it_programmatically_does_not_echo_the_callback(self):
        # The orchestrator syncs its own state on startup; hearing its own
        # change back would toggle the microphone a second time.
        seen = []
        obs = self._observer()
        obs.on_listening_changed(seen.append)
        obs.set_listening(False)
        assert obs.listening() is False
        assert seen == []

    def test_a_failing_callback_never_reaches_appkit(self):
        obs = self._observer()

        def _boom(_value):
            raise RuntimeError("callback exploded")

        obs.on_listening_changed(_boom)
        obs._handle_toggle_click()          # must not raise
        assert obs.listening() is False


class TestManualSize:
    """Once the user drags the panel, THEIR size wins.

    An automatic size is a good default, not a preference to keep re-imposing
    on someone who already said what they wanted. The one exception is the
    collapse control: hiding the text while keeping its space would defeat it.
    """

    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        return obs

    def test_a_manual_height_survives_a_new_answer(self):
        obs = self._observer()
        obs.set_size(760, 400)
        obs.set_texts(user_text="hola", response="Sí.")
        assert obs._window_height() == 400

    def test_collapsing_still_overrides_it(self):
        from wawi import panel

        obs = self._observer()
        obs.set_size(760, 400)
        obs.transcript_expanded = False
        assert obs._window_height() == panel.PANEL_HEIGHT

    def test_the_extra_space_goes_to_the_text(self):
        from wawi import panel

        obs = self._observer()
        obs.set_size(760, 400)
        # Everything above the base layout is band, i.e. more rows of answer —
        # the only reason anyone drags a HUD taller.
        assert obs.bottom_block_height() == 400 - panel.PANEL_HEIGHT

    def test_a_size_below_the_minimum_is_clamped(self):
        from wawi import panel

        obs = self._observer()
        obs.set_size(10, 10)
        assert obs._manual_size == (panel._MIN_WIDTH, panel._MIN_HEIGHT)

    def test_set_size_never_touches_the_window_itself(self):
        class _Exploding:
            def frame(self):
                raise AssertionError("resized from the calling thread")

        obs = self._observer()
        obs._window = _Exploding()
        obs.set_size(760, 400)            # must not raise
        assert obs._pending_resize is True


class TestHistory:
    """A taller panel must show MORE CONVERSATION, not more grey.

    One exchange cannot fill a large window no matter how the rows are laid
    out, so the size control and the history are the same feature.
    """

    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        return obs

    def test_a_finished_exchange_moves_into_history(self):
        obs = self._observer()
        obs.set_texts(user_text="cómo se escribe sol", response="Sun.")
        obs.set_texts(user_text="y luna")
        assert obs._history == [("cómo se escribe sol", "Sun.")]
        assert obs._question == "y luna"
        assert obs._answer == ""

    def test_an_unanswered_question_is_not_remembered(self):
        # It never happened as an exchange; filing it would show a question
        # the assistant is still working on as though it had been dropped.
        obs = self._observer()
        obs.set_texts(user_text="primera")
        obs.set_texts(user_text="segunda")
        assert obs._history == []

    def test_history_is_bounded(self):
        from wawi import panel

        obs = self._observer()
        for i in range(panel._HISTORY_MAX + 15):
            obs.set_texts(user_text=f"p{i}", response=f"r{i}")
        obs.set_texts(user_text="última")
        assert len(obs._history) == panel._HISTORY_MAX
        # The newest survive; the oldest are the ones dropped.
        assert obs._history[-1][0] == f"p{panel._HISTORY_MAX + 14}"


class TestWindowButtons:
    """Close / minimise / zoom, in the macOS order."""

    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        return obs

    def test_close_asks_to_quit_but_does_not_act(self):
        # The panel owns no microphone, no model and no log. Deciding to end
        # the process is not its call — it raises the flag and the tick turns
        # that into the same SIGINT path as Ctrl+C.
        obs = self._observer()
        obs._handle_window_button(0)
        assert obs.quit_requested is True

    def test_minimise_collapses_and_does_not_toggle_back(self):
        obs = self._observer()
        obs._handle_window_button(1)
        assert obs.transcript_expanded is False
        obs._handle_window_button(1)          # a second click must not reopen
        assert obs.transcript_expanded is False

    def test_zoom_grows_then_restores_the_previous_size(self):
        obs = self._observer()
        obs.set_size(500, 300)
        obs._handle_zoom()
        zoomed = obs._manual_size
        assert zoomed is not None and zoomed[1] > 300

        obs._handle_zoom()
        assert obs._manual_size == (500, 300)

    def test_zoom_reopens_a_collapsed_panel(self):
        # Zooming to a big window that shows no text would be absurd.
        obs = self._observer()
        obs.transcript_expanded = False
        obs._handle_zoom()
        assert obs.transcript_expanded is True


class TestStatusItem:
    """The menu-bar icon is the only way back to a hidden panel.

    The app runs under the accessory activation policy — no Dock icon, by
    design, so clicking the HUD never steals the caret from your editor. That
    choice is what makes this icon load-bearing rather than decorative: without
    it, a panel behind another window is a process you cannot reach.
    """

    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        return obs

    def test_the_glyph_says_muted_before_it_says_anything_else(self):
        # The gate is the one thing you cannot infer from the rest of the
        # screen: a muted Wawi looks exactly like an idle one.
        from wawi import panel

        obs = self._observer()
        obs._listening = False
        assert panel.PanelObserver._STATUS_GLYPHS["muted"] != (
            panel.PanelObserver._STATUS_GLYPHS[panel.State.IDLE]
        )

    def test_every_state_has_its_own_glyph(self):
        from wawi import panel

        glyphs = panel.PanelObserver._STATUS_GLYPHS
        for state in panel.State:
            assert state in glyphs, state
        assert len(set(glyphs.values())) == len(glyphs), "two states look alike"

    def test_refreshing_without_an_item_is_harmless(self):
        # start() may never have run (panel disabled, or non-macOS).
        obs = self._observer()
        obs._status_item = None
        obs._refresh_status_item()

    def test_bringing_a_missing_window_to_front_is_harmless(self):
        obs = self._observer()
        obs._window = None
        obs.bring_to_front()


class TestStatusAnimation:
    """The glyph shows the live microphone, not a decorative loop.

    That distinction is the whole value: a spinner that turns while nothing
    happens teaches you to stop looking at it, while a bar that moves with your
    voice answers "is it hearing me?" from the menu bar — the one question you
    cannot ask a hidden panel.
    """

    def _observer(self, **kw):
        import numpy as np
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        for k, v in kw.items():
            setattr(obs, k, v)
        obs._bars = np.zeros(4, dtype="float32")
        return obs

    def test_muted_wins_over_every_other_state(self):
        from wawi import panel

        obs = self._observer(_listening=False, _current_state=panel.State.LISTENING)
        obs._bars[:] = 1.0
        assert obs._status_glyph() == panel.PanelObserver._STATUS_GLYPHS["muted"]

    def test_the_menu_item_names_the_action(self):
        assert self._observer(_listening=True)._listening_menu_title() == "Dejar de escuchar"
        assert self._observer(_listening=False)._listening_menu_title() == "Escuchar"

    def test_the_wake_mode_item_names_the_mode_and_calls_back(self):
        obs = self._observer()
        assert "Wawi" in obs._wake_mode_menu_title()
        got = []
        obs.on_wake_mode_changed(got.append)
        obs._handle_wake_mode_click()
        assert got == [True]
        obs.set_wake_mode(True)
        # Same title either way: the checkmark carries the state.
        assert obs._wake_mode_menu_title() == "Solo cuando me digan «Wawi»"
        obs._handle_wake_mode_click()
        assert got == [True, False]

    def test_a_louder_voice_gives_a_taller_bar(self):
        from wawi import panel

        quiet = self._observer(_current_state=panel.State.LISTENING)
        quiet._bars[:] = 0.1
        loud = self._observer(_current_state=panel.State.LISTENING)
        loud._bars[:] = 0.95

        bars = panel.PanelObserver._LEVEL_BARS
        assert bars.index(loud._status_glyph()) > bars.index(quiet._status_glyph())

    def test_a_silent_room_holds_still_instead_of_twitching(self):
        # A menu bar that jitters on room noise forever is noise itself.
        from wawi import panel

        obs = self._observer(_current_state=panel.State.IDLE)
        obs._bars[:] = 0.01
        assert obs._status_glyph() == panel.PanelObserver._STATUS_GLYPHS[panel.State.IDLE]

    def test_thinking_spins_because_there_is_no_level_to_show(self):
        # The microphone is closed while the model works, so the only honest
        # thing to animate is "still working".
        from wawi import panel

        obs = self._observer(_current_state=panel.State.THINKING)
        seen = {obs._status_glyph() for _ in range(len(panel.PanelObserver._SPINNER))}
        assert len(seen) == len(panel.PanelObserver._SPINNER)

    def test_the_level_never_indexes_past_the_end(self):
        from wawi import panel

        obs = self._observer(_current_state=panel.State.LISTENING)
        obs._bars[:] = 5.0          # a level above 1.0 must not crash
        assert obs._status_glyph() in panel.PanelObserver._LEVEL_BARS


class TestPanelVisibility:
    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        return obs

    def test_it_starts_visible(self):
        assert self._observer().panel_visible is True

    def test_toggling_hides_then_shows(self):
        obs = self._observer()
        obs.toggle_panel()
        assert obs.panel_visible is False
        obs.toggle_panel()
        assert obs.panel_visible is True

    def test_bring_to_front_shows_a_hidden_panel(self):
        # The menu item has to work no matter which state it was left in.
        obs = self._observer()
        obs.panel_visible = False
        obs.bring_to_front()
        assert obs.panel_visible is True


class TestVisibilityIsMainThreadOnly:
    """Only the tick touches the window — including for show/hide.

    A menu click already runs on the main thread, but having two ways in is
    exactly how the OTHER one ends up being called from a worker. This file has
    already lost the process twice that way (set_texts resizing, and the probe
    that hid the panel from a thread).
    """

    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False

        class _Exploding:
            def orderOut_(self, _):  # noqa: N802
                raise AssertionError("window touched from the calling thread")

            def orderFrontRegardless(self):  # noqa: N802
                raise AssertionError("window touched from the calling thread")

        obs._window = _Exploding()
        return obs

    def test_toggling_only_schedules(self):
        obs = self._observer()
        obs.toggle_panel()                 # must not raise
        assert obs._pending_visibility is True

    def test_bring_to_front_only_schedules(self):
        obs = self._observer()
        obs.panel_visible = False
        obs.bring_to_front()
        assert obs._pending_visibility is True

    def test_the_tick_consumes_it_once(self):
        obs = self._observer()
        applied = []
        obs._apply_panel_visibility = lambda: applied.append(1)  # type: ignore[method-assign]

        obs.toggle_panel()
        obs._tick()
        obs._tick()

        assert applied == [1]
        assert obs._pending_visibility is False


class TestDrawingNeverKillsTheApp:
    """An exception out of drawRect_ calls +[NSApplication _crashOnException:].

    That is not a theory: it killed the running assistant, and the crash report
    named the path — method_stub -> PyObjCErr_ToObjCWithGILState ->
    objc_exception_throw -> _NSViewDrawRect. The process vanished with nothing
    in the log, taking the microphone, the session and the conversation with
    it. A drawing bug has to cost one frame instead.
    """

    def test_every_view_wraps_its_painting(self):
        """Read the source: PyObjC selectors cannot be introspected.

        A structural check on purpose — the next drawRect_ someone adds is the
        one that will not be guarded, and no runtime test can catch a method
        that does not exist yet.
        """
        import re
        from pathlib import Path

        import wawi.panel

        src = Path(wawi.panel.__file__).read_text()
        bodies = re.findall(
            r"def drawRect_\(self, \w+\):(.*?)(?=\n\s*@objc\.python_method)",
            src,
            re.S,
        )
        assert bodies, "no drawRect_ found — the check would pass vacuously"

        for body in bodies:
            assert "self._paint(" in body, "a view still paints inline"
            assert "except Exception" in body, "a view can still crash the app"

    # NOTE: there is deliberately no runtime test that makes a real view raise.
    # Allocating an NSView inside pytest leaves AppKit in a state that hangs the
    # whole suite at exit — the test passed alone and hung in company, which is
    # a worse trade than the structural check above, and the structural check is
    # the one that catches the NEXT unguarded drawRect_ anyway.


class TestCopyAnswer:
    """An answer worth keeping that you can only re-read on a HUD is
    information trapped in a window."""

    def _observer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.show_projects = False
        return obs

    def test_there_is_nothing_to_copy_before_an_answer(self):
        obs = self._observer()
        assert obs.copy_answer() is False
        assert obs.copy_confirmed() is False

    def test_a_whitespace_only_answer_is_not_copied(self):
        # Clearing the clipboard to write nothing into it would destroy
        # whatever the user had copied before.
        obs = self._observer()
        obs.set_texts(response="   \n  ")
        assert obs.copy_answer() is False

    def test_the_answer_reaches_the_clipboard(self):
        from AppKit import NSPasteboard, NSPasteboardTypeString

        obs = self._observer()
        obs.set_texts(user_text="cómo se escribe sol", response="Sun.")

        assert obs.copy_answer() is True
        assert NSPasteboard.generalPasteboard().stringForType_(
            NSPasteboardTypeString
        ) == "Sun."

    def test_it_copies_the_answer_alone_not_the_exchange(self):
        # You want to paste the content, not a transcript of the conversation.
        from AppKit import NSPasteboard, NSPasteboardTypeString

        obs = self._observer()
        obs.set_texts(user_text="cuánto es dos más dos", response="Cuatro.")
        obs.copy_answer()

        pasted = NSPasteboard.generalPasteboard().stringForType_(NSPasteboardTypeString)
        assert pasted == "Cuatro."
        assert "dos más dos" not in pasted

    def test_copying_confirms_and_the_confirmation_expires(self):
        import time

        from wawi import panel

        obs = self._observer()
        obs.set_texts(response="Sun.")
        obs.copy_answer()
        assert obs.copy_confirmed() is True

        obs._copied_at = time.monotonic() - panel._COPIED_FEEDBACK_S - 0.1
        assert obs.copy_confirmed() is False

    def test_a_new_answer_ends_the_previous_confirmation(self):
        """Leaving "copiado" up would claim THIS answer is on the clipboard."""
        obs = self._observer()
        obs.set_texts(response="Sun.")
        obs.copy_answer()
        assert obs.copy_confirmed() is True

        obs.set_texts(user_text="y luna")
        obs.set_texts(response="Moon.")
        assert obs.copy_confirmed() is False


class TestRestartItem:
    """«Reiniciar Wawi»: only when wawictl owns this process."""

    def test_offered_when_the_pid_file_names_us(self, tmp_path, monkeypatch):
        import os

        from wawi import panel as panel_mod

        pid_file = panel_mod.Path(panel_mod.__file__).resolve().parent.parent / "wawi.pid"
        real = panel_mod.Path.read_text

        def fake_read(self, *a, **k):
            if self == pid_file:
                return f"{os.getpid()}\n"
            return real(self, *a, **k)

        monkeypatch.setattr(panel_mod.Path, "read_text", fake_read)
        cmd = panel_mod.restart_command()
        assert cmd is not None and cmd[0].endswith("scripts/wawictl") and cmd[1] == "restart"
        # Another process's pid file: restarting would start a second Wawi.
        assert panel_mod.restart_command(pid=os.getpid() + 1) is None
