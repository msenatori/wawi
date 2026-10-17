"""The per-turn latency line: where the wait before Wawi speaks went."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from wawi.orchestrator import Orchestrator
from wawi.voice import Voice


def fake_orch(silence_ms, play_at):
    voice = Voice.__new__(Voice)
    for t in play_at:
        voice.__dict__.setdefault("_play_marks", []).append(t)
    return SimpleNamespace(
        listener=SimpleNamespace(last_trailing_silence_ms=silence_ms), voice=voice
    )


def test_line_splits_the_wait(caplog):
    orch = fake_orch(330, [5.0, 103.1, 104.0])
    speaker = SimpleNamespace(first_sentence_at=102.4)
    with caplog.at_level(logging.INFO, logger="wawi.orchestrator"):
        Orchestrator._log_latency(orch, 100.0, 0.55, 101.0, 2.6, speaker)
    line = caplog.records[-1].getMessage()
    assert "silence=330ms" in line and "stt=0.55s" in line
    assert "brain_first_sentence=1.40s" in line and "brain_total=2.60s" in line
    assert "first_voice=3.10s" in line and "felt=3.43s" in line


def test_missing_pieces_show_as_dashes_and_never_raise(caplog):
    orch = SimpleNamespace(listener=None, voice=object())
    speaker = SimpleNamespace(first_sentence_at=None)
    with caplog.at_level(logging.INFO, logger="wawi.orchestrator"):
        Orchestrator._log_latency(orch, 1.0, None, None, None, speaker)
    line = caplog.records[-1].getMessage()
    assert "silence=-" in line and "first_voice=-" in line and "felt=-" in line


def test_voice_finds_the_first_play_after_the_turn():
    v = Voice.__new__(Voice)
    assert v.first_play_since(0.0) is None
    for t in (1.0, 2.0, 3.0):
        v.__dict__.setdefault("_play_marks", []).append(t)
    assert v.first_play_since(1.5) == 2.0
