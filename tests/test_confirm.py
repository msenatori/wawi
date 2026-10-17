"""Confirmation queue tests need no AppKit or display server."""
from __future__ import annotations

import threading

from wawi.confirm import Registry


class FakePresenter:
    def __init__(self):
        self.events = []

    def show(self, item):
        self.events.append(("show", item.title))

    def hide(self):
        self.events.append(("hide", None))


def test_one_visible_then_next_and_buttons():
    now = [0.0]
    view = FakePresenter()
    queue = Registry(clock=lambda: now[0], presenter=view)
    calls = []
    first = queue.add("First", "one", "agent", lambda: calls.append("first"),
                      lambda: calls.append("cancel first"))
    second = queue.add("Second", "two", "shortcut", lambda: calls.append("second"),
                       lambda: calls.append("cancel second"))
    queue.tick()
    assert view.events == [("show", "First")]
    assert queue.resolve(second, True)
    assert calls == ["second"]
    queue.tick()
    assert view.events == [("show", "First")]
    assert queue.resolve(first, False)
    queue.tick()
    assert calls == ["second", "cancel first"]
    assert view.events[-1] == ("hide", None)


def test_expiry_silent_and_advances_queue():
    now = [0.0]
    view = FakePresenter()
    queue = Registry(clock=lambda: now[0], presenter=view)
    calls = []
    first = queue.add("First", "", "orca", lambda: calls.append("wrong"),
                      lambda: calls.append("expired"), timeout=5)
    queue.add("Second", "", "reminders", lambda: None, lambda: None)
    queue.tick()
    now[0] = 6
    queue.tick()
    assert calls == ["expired"]
    assert not queue.resolve(first, True)
    assert view.events[-1] == ("show", "Second")


def test_concurrent_resolution_runs_callback_once():
    queue = Registry()
    calls = []
    identifier = queue.add("One", "", "orca", lambda: calls.append(1), lambda: None)
    threads = [threading.Thread(target=queue.resolve, args=(identifier, True)) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert calls == [1]
