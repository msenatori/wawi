"""Config loading — only the rules that change how the listener behaves.

The rest of the file is plain key reading; what is worth pinning here is the
always-on switch, because getting it wrong makes Wawi either deaf or
impossible to shut up.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from wawi import config as config_mod
from wawi import paths


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_defaults_are_always_on_with_the_whisper_engine(tmp_path):
    cfg = config_mod.load(write(tmp_path, "language: es\n"))
    assert cfg.wake_require_phrase is False
    assert cfg.wake_engine == "whisper"
    assert cfg.wake_phrases == ("wawi",)


def test_require_phrase_true_is_read(tmp_path):
    cfg = config_mod.load(
        write(tmp_path, "wake:\n  engine: whisper\n  require_phrase: true\n")
    )
    assert cfg.wake_require_phrase is True


def test_require_phrase_false_is_read(tmp_path):
    cfg = config_mod.load(
        write(
            tmp_path,
            """
            wake:
              engine: whisper
              require_phrase: false
            """,
        )
    )
    assert cfg.wake_require_phrase is False


def test_whisper_without_phrases_is_an_error_only_when_the_phrase_is_required(
    tmp_path,
):
    body = """
        wake:
          engine: whisper
          phrases: []
          require_phrase: {required}
    """
    with pytest.raises(config_mod.ConfigError, match="wake.phrases is empty"):
        config_mod.load(write(tmp_path, body.format(required="true")))

    # Always-on never matches a phrase, so an empty list is the point.
    cfg = config_mod.load(write(tmp_path, body.format(required="false")))
    assert cfg.wake_phrases == ()


def test_recall_defaults_to_a_small_enabled_store(tmp_path):
    cfg = config_mod.load(write(tmp_path, "language: es\n"))
    assert cfg.recall_file == paths.private("recall.txt")
    assert cfg.recall_turns == 6


def test_an_empty_recall_file_disables_the_durable_memory(tmp_path):
    """Same escape hatch as conversation_log: off, and no file left behind."""
    cfg = config_mod.load(write(tmp_path, 'recall_file: ""\n'))
    assert cfg.recall_file is None


def test_the_recall_bound_is_read_and_never_negative(tmp_path):
    assert config_mod.load(write(tmp_path, "recall_turns: 2\n")).recall_turns == 2
    assert config_mod.load(write(tmp_path, "recall_turns: -5\n")).recall_turns == 0


def test_smart_turn_is_off_unless_asked_for(tmp_path):
    # Off by default: it needs a model download, and an unconfigured install
    # must endpoint exactly as it always did.
    cfg = config_mod.load(write(tmp_path, "language: es\n"))
    assert cfg.wake_smart_turn is False
    assert cfg.wake_smart_turn_threshold == 0.5
    assert cfg.wake_smart_turn_pause_ms == 300


def test_smart_turn_keys_are_read(tmp_path):
    cfg = config_mod.load(
        write(
            tmp_path,
            """
            wake:
              smart_turn: true
              smart_turn_threshold: 0.7
              smart_turn_pause_ms: 210
            """,
        )
    )
    assert cfg.wake_smart_turn is True
    assert cfg.wake_smart_turn_threshold == 0.7
    assert cfg.wake_smart_turn_pause_ms == 210


def test_noise_gates_are_on_by_default(tmp_path):
    cfg = config_mod.load(write(tmp_path, "language: es\n"))
    assert cfg.wake_vad_onset_threshold == 0.6
    assert cfg.wake_min_onset_ms == 150
    assert cfg.wake_min_voiced_ms == 250
    assert cfg.wake_min_voiced_ratio == 0.2
    assert cfg.stt_filter_junk is True
    assert cfg.stt_junk_min_voiced_ms == 700


def test_noise_gate_keys_are_read(tmp_path):
    cfg = config_mod.load(write(tmp_path, """
        wake:
          vad_onset_threshold: 0.7
          min_onset_ms: 210
          onset_window_ms: 500
          min_voiced_ms: 400
          min_voiced_ratio: 0.3
        stt:
          filter_junk: false
          min_avg_logprob: -0.8
          max_compression_ratio: 2.0
          max_no_speech_prob: 0.5
          junk_min_voiced_ms: 900
          junk_min_logprob: -0.3
    """))
    assert (cfg.wake_vad_onset_threshold, cfg.wake_min_onset_ms, cfg.wake_onset_window_ms,
            cfg.wake_min_voiced_ms, cfg.wake_min_voiced_ratio) == (0.7, 210, 500, 400, 0.3)
    assert cfg.stt_filter_junk is False
    assert (cfg.stt_min_avg_logprob, cfg.stt_max_compression_ratio, cfg.stt_max_no_speech_prob,
            cfg.stt_junk_min_voiced_ms, cfg.stt_junk_min_logprob) == (-0.8, 2.0, 0.5, 900, -0.3)
