"""Tests for the buzón "Wawi investiga" (wawi/inbox.py) and typed turns.

No window is built here: :class:`wawi.inbox.Inbox` is driven through its
plain methods (what the AppKit views call), the pasteboard is the fake from
tests/test_drops.py, and the orchestrator is the same ``__init__``-less one
the drop tests use, with ``brain.ask_sync`` replaced.
"""

from __future__ import annotations

import json
import threading

import pytest

from wawi import drops, inbox
from wawi.notes import NoteFile, NoteSession
from wawi.state import State
from tests.test_drops import FakePasteboard, brain_calls, make_drop_orch, png_bytes, wait_for  # noqa: F401


# --------------------------------------------------------------------------- #
# Pure parts
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_enabled_by_default(self):
        assert inbox.enabled_from_config(None) is True
        assert inbox.enabled_from_config({}) is True
        assert inbox.enabled_from_config({"enabled": False}) is False
        assert inbox.enabled_from_config({"enabled": "nope"}) is True

    def test_the_config_carries_the_section(self):
        from wawi.config import _DEFAULTS, Config

        assert _DEFAULTS["inbox"] == {"enabled": True}
        assert "inbox" in Config.__dataclass_fields__


class TestLine:
    def test_nothing_dropped_is_the_hint(self):
        assert inbox.items_line([]) == inbox.HINT

    def test_one_picture_by_name(self):
        item = drops.DropItem(kind="image", name="Captura.png", path="/tmp/Captura.png")
        assert inbox.items_line([item]) == "Imagen: Captura.png"

    def test_a_text_is_peeked_and_clipped(self):
        line = inbox.items_line([drops.DropItem(kind="text", text="hola " * 40)])
        assert line.startswith("Texto: «hola hola")
        assert len(line) <= inbox.LINE_MAX and line.endswith("…")

    def test_a_link(self):
        line = inbox.items_line([drops.DropItem(kind="url", text="https://x.org/a")])
        assert line == "Link: https://x.org/a"

    def test_several_are_counted_by_kind(self):
        items = [drops.DropItem(kind="image", name="a.png"),
                 drops.DropItem(kind="image", name="b.png"),
                 drops.DropItem(kind="pdf", name="c.pdf")]
        assert inbox.items_line(items) == "3 cosas: 2 imágenes, PDF"


class TestOrder:
    def test_typed_text_wins(self):
        assert inbox.order_text("  traducilo ", []) == "traducilo"

    def test_empty_enter_with_items_investigates(self):
        assert inbox.order_text("", [drops.DropItem(kind="text", text="x")]) == "Investigá esto."

    def test_empty_enter_without_items_sends_nothing(self):
        assert inbox.order_text("   ", []) is None


class TestPaste:
    def test_a_picture_is_an_item(self, tmp_path):
        pb = FakePasteboard(data={drops.PNG: png_bytes()})
        items = inbox.paste_items(pb, picture_dir=tmp_path)
        assert [i.kind for i in items] == ["image"]

    def test_copied_files_are_items(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        items = inbox.paste_items(FakePasteboard([str(tmp_path / "a.pdf")]))
        assert [i.kind for i in items] == ["pdf"]

    def test_text_is_pasted_as_text(self, tmp_path):
        # A rich copy carries a picture of itself too: the text wins.
        pb = FakePasteboard(strings={drops.STRING: "hola"}, data={drops.TIFF: b"x"})
        assert inbox.paste_items(pb, picture_dir=tmp_path) == []
        assert inbox.paste_items(FakePasteboard()) == []

    def test_selector_names(self):
        assert inbox.selector_name(b"insertNewline:") == "insertNewline:"
        assert inbox.selector_name("cancelOperation:") == "cancelOperation:"


class TestGeometry:
    def test_home_is_bottom_right(self):
        x, y = inbox.home((0, 0, 1000, 800))
        assert x == 1000 - inbox.WIDTH - 24 and y == 96

    def test_on_screen(self):
        assert inbox.on_screen((100, 100), [(0, 0, 1000, 800)])
        assert not inbox.on_screen((5000, 100), [(0, 0, 1000, 800)])
        assert not inbox.on_screen((100, 100), [])

    def test_the_field_goes_above_or_below(self):
        visible = (0, 0, 1000, 800)
        x, y = inbox.field_origin((400, 100, inbox.WIDTH, inbox.HEIGHT), visible)
        assert y == 100 + inbox.HEIGHT + inbox.GAP
        assert x == pytest.approx(400 + inbox.WIDTH / 2 - inbox.FIELD_WIDTH / 2)
        _x, y = inbox.field_origin((400, 750 - inbox.HEIGHT, inbox.WIDTH, inbox.HEIGHT), visible)
        assert y == 750 - inbox.HEIGHT - inbox.GAP - inbox.FIELD_HEIGHT
        x, _y = inbox.field_origin((990, 100, inbox.WIDTH, inbox.HEIGHT), visible)
        assert x == 1000 - inbox.FIELD_WIDTH - 4


# --------------------------------------------------------------------------- #
# The Inbox state (what the views call)
# --------------------------------------------------------------------------- #


class Calls:
    def __init__(self, hold=True, take=True):
        self.drops: list[list] = []
        self.orders: list[str] = []
        self.hold = hold
        self.take = take

    def on_drop(self, items):
        self.drops.append(list(items))
        return self.hold

    def on_submit(self, text):
        self.orders.append(text)
        return self.take


def make_inbox(tmp_path, calls=None, **kw):
    calls = calls or Calls()
    box = inbox.Inbox(on_drop=calls.on_drop, on_submit=calls.on_submit,
                      local_file=tmp_path / "inbox_local.json", **kw)
    return box, calls


class TestInbox:
    def test_the_menu_switch_is_kept_and_wins_over_the_config(self, tmp_path):
        box, _ = make_inbox(tmp_path)
        assert box.enabled is True
        box.set_enabled(False)
        again, _ = make_inbox(tmp_path, enabled_default=True)
        assert again.enabled is False
        assert json.loads((tmp_path / "inbox_local.json").read_text())["enabled"] is False

    def test_the_config_default_without_a_choice(self, tmp_path):
        box, _ = make_inbox(tmp_path, enabled_default=False)
        assert box.enabled is False

    def test_where_it_was_left_is_remembered(self, tmp_path):
        box, _ = make_inbox(tmp_path)
        box.moved(120.0, 340.5)
        again, _ = make_inbox(tmp_path)
        assert again.origin == (120.0, 340.5)

    def test_a_broken_local_file_is_ignored(self, tmp_path):
        (tmp_path / "inbox_local.json").write_text("{nope")
        box, _ = make_inbox(tmp_path, enabled_default=True)
        assert box.enabled is True and box.origin is None

    def test_a_drop_is_held_and_opens_the_field(self, tmp_path):
        box, calls = make_inbox(tmp_path)
        (tmp_path / "a.png").write_bytes(png_bytes())
        assert box.drop_pasteboard(FakePasteboard([str(tmp_path / "a.png")])) is True
        assert [i.kind for i in calls.drops[0]] == ["image"]
        assert box.field_open and box.line == "Imagen: a.png"
        assert box.drop_pasteboard(FakePasteboard()) is False  # nothing she understands

    def test_more_drops_join_the_open_field(self, tmp_path):
        box, calls = make_inbox(tmp_path)
        box.dropped([drops.DropItem(kind="text", text="uno")])
        box.dropped([drops.DropItem(kind="url", text="https://x.org")])
        assert [i.kind for i in calls.drops[-1]] == ["text", "url"]
        assert box.line == "2 cosas: texto, link"

    def test_a_drop_she_did_not_hold_opens_nothing(self, tmp_path):
        box, calls = make_inbox(tmp_path, Calls(hold=False))
        assert box.dropped([drops.DropItem(kind="text", text="x")]) is True  # taken (a note)
        assert not box.field_open and box.items == []

    def test_enter_sends_and_closes(self, tmp_path):
        box, calls = make_inbox(tmp_path)
        box.dropped([drops.DropItem(kind="text", text="x")])
        assert box.submit("") is True
        assert calls.orders == ["Investigá esto."]
        assert not box.field_open and box.items == []

    def test_empty_enter_with_nothing_just_closes(self, tmp_path):
        box, calls = make_inbox(tmp_path)
        box.double_clicked()
        assert box.field_open and box.items == [] and box.line == inbox.HINT
        assert box.submit("  ") is True
        assert calls.orders == [] and not box.field_open

    def test_busy_keeps_the_field_and_says_so(self, tmp_path):
        box, calls = make_inbox(tmp_path, Calls(take=False))
        box.double_clicked()
        assert box.submit("¿qué es esto?") is False
        assert box.field_open and box.line == inbox.BUSY_TEXT
        assert calls.orders == ["¿qué es esto?"]

    def test_esc_closes_and_forgets_the_field_items(self, tmp_path):
        box, calls = make_inbox(tmp_path)
        box.dropped([drops.DropItem(kind="text", text="x")])
        box.cancel()
        assert not box.field_open and box.items == []
        assert len(calls.drops) == 1  # nothing un-held: the orchestrator keeps it

    def test_a_double_click_is_a_plain_question(self, tmp_path):
        box, calls = make_inbox(tmp_path)
        box.dropped([drops.DropItem(kind="text", text="x")])
        box.cancel()
        box.double_clicked()
        assert box.items == [] and box.field_open
        box.submit("¿qué hora es?")
        assert calls.orders == ["¿qué hora es?"]

    def test_paste_adds_a_picture_but_not_text(self, tmp_path):
        box, calls = make_inbox(tmp_path)
        box.double_clicked()
        assert box.paste(FakePasteboard(strings={drops.STRING: "hola"})) is False
        assert box.paste(FakePasteboard(data={drops.PNG: png_bytes()})) is True
        assert [i.kind for i in box.items] == ["image"]

    def test_switching_off_closes_the_field(self, tmp_path):
        box, _ = make_inbox(tmp_path)
        box.double_clicked()
        box.set_enabled(False)
        assert not box.field_open


# --------------------------------------------------------------------------- #
# The orchestrator: held drops and typed turns
# --------------------------------------------------------------------------- #


def _typed(o, text):
    """Submit *text* as the buzón does, and wait for the turn to end."""
    assert o.submit_typed(text) is True
    assert wait_for(lambda: o.state.state is State.IDLE and not o._turn_lock.locked())


class TestTyped:
    def test_a_buzon_drop_is_held_without_the_voice_window(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch)
        assert o._on_inbox_dropped([drops.DropItem(kind="text", text="good morning")]) is True
        assert o._drop is not None and o._drop.awaiting
        assert o.voice.said == [] and o.listener.follow_ups == [] and o.panel.caught == []
        assert brain_calls == []

    def test_the_typed_order_carries_the_drop(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch)
        o._on_inbox_dropped([drops.DropItem(kind="text", text="good morning")])
        _typed(o, "traducilo")
        (text, _images), = brain_calls
        assert "good morning" in text and text.endswith("traducilo")
        assert o.voice.said == ["Listo."]
        assert o.conversation.added[0][:2] == ("traducilo", "Listo.")
        assert o._drop is not None and not o._drop.awaiting  # held for follow-ups

    def test_a_spoken_order_carries_a_buzon_drop_too(self, monkeypatch, brain_calls):
        from tests.test_drops import _order

        o = make_drop_orch(monkeypatch)
        o._on_inbox_dropped([drops.DropItem(kind="text", text="good morning")])
        _order(o, "traducilo")
        assert "good morning" in brain_calls[0][0]

    def test_a_plain_typed_question(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch)
        _typed(o, "Wawi, ¿qué hora es?")
        assert brain_calls[0][0] == "¿qué hora es?"  # the wake phrase is taken off

    def test_the_thinking_state_starts_at_once(self, monkeypatch):
        o = make_drop_orch(monkeypatch)
        gate = threading.Event()

        def slow(user_text, **kw):
            gate.wait(5)
            from wawi import brain

            return brain.BrainResult(text="Listo.", tool_calls=(), session_id="s")

        monkeypatch.setattr("wawi.orchestrator.brain.ask_sync", slow)
        assert o.submit_typed("investigá") is True
        assert o.state.state is State.THINKING
        gate.set()
        assert wait_for(lambda: o.state.state is State.IDLE and not o._turn_lock.locked())

    def test_mid_turn_it_is_refused_with_a_bubble(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch)
        o._turn_lock.acquire()
        o.state.set_state(State.THINKING)
        assert o.submit_typed("hola") is False
        assert o.panel.bubbles == [inbox.BUSY_TEXT]
        assert brain_calls == []
        o._turn_lock.release()

    def test_empty_text_is_not_a_turn(self, monkeypatch, brain_calls):
        o = make_drop_orch(monkeypatch)
        assert o.submit_typed("   ") is False
        assert not o._turn_lock.locked()

    def test_a_confirmation_reader_answers_before_the_brain(self, monkeypatch, brain_calls):
        from wawi import orca

        monkeypatch.setattr(orca, "hear", lambda text: "Listo, la mandé." if text == "sí" else None)
        o = make_drop_orch(monkeypatch)
        _typed(o, "sí")
        assert brain_calls == [] and o.voice.said == ["Listo, la mandé."]

    def test_while_taking_notes_it_is_a_line(self, monkeypatch, brain_calls, tmp_path):
        o = make_drop_orch(monkeypatch)
        note = NoteFile(tmp_path)
        o._notes = NoteSession(note)
        assert o.submit_typed("comprar pan") is True
        assert "comprar pan" in note.read()
        assert brain_calls == [] and not o._turn_lock.locked()
        assert o._on_inbox_dropped([drops.DropItem(kind="text", text="punto")]) is False
        assert wait_for(lambda: "[texto] punto" in note.read())


# --------------------------------------------------------------------------- #
# The panel half
# --------------------------------------------------------------------------- #


class TestPanel:
    def test_configure_and_the_menu_switch(self, monkeypatch, tmp_path):
        from wawi.panel import PanelObserver

        monkeypatch.setenv("WAWI_PRIVATE_DIR", str(tmp_path))
        p = PanelObserver(enabled=False)
        assert p._inbox is None
        p.configure_inbox(enabled_default=True, on_drop=lambda items: True,
                          on_submit=lambda text: True)
        assert p._inbox.enabled is True
        p._handle_inbox_click()
        assert p._inbox.enabled is False
        saved = json.loads((tmp_path / inbox.LOCAL_FILE).read_text())
        assert saved["enabled"] is False
        p._inbox.tick()  # no window while off; never raises

    def test_the_tag_is_free(self):
        from wawi.panel import PanelObserver
        from wawi.pet import personality_menu
        from wawi.webvm import menu as webvm_menu

        tags = {PanelObserver._WAKE_MODE_TAG, PanelObserver._ORCA_TAG,
                PanelObserver._NOTES_TAG, PanelObserver._CURIOSITY_TAG,
                PanelObserver._PLUGINS_TAG, PanelObserver._MUSIC_TAG,
                webvm_menu.TAG, personality_menu.TAG}
        assert PanelObserver._INBOX_TAG == inbox.TAG == 4250 and inbox.TAG not in tags
