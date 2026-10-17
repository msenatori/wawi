"""The "Wawi" wake word: matcher, request splitting, the decoder prompt, the
live always-on / phrase switch and its persistence. No model, no audio.

The matcher cases come from real whisper-tiny transcripts of Kokoro clips
(scripts/wakeword/measure_phrase.py): what the positives came back as, and
what the everyday words that must NOT wake her came back as.
"""

from __future__ import annotations

import textwrap
import threading

import numpy as np
import pytest

from wawi import config as config_mod
from wawi import wakemode
from wawi.orchestrator import Orchestrator
from wawi.wake import (
    WhisperWakeDetector,
    match_trigger,
    split_request,
    wake_prompt,
)

PHRASES = ["wawi"]


class TestMatcher:
    @pytest.mark.parametrize("heard", [
        "Wawi.", "¡Wawi!", "¿Wawi?", "wawi", "Wawi, ¿qué hora es?",
        "Che Wawi.", "¡Che Wawi!", "Chewawi", "Hola, Wawi.", "Ey, Wawi.",
        "Guagui.", "Guaui", "Waui", "Wawy", "Wowie", "Wawi, Wawi.",
        "Cheewawy",  # glued interjection + a seen spelling
    ])
    def test_wakes(self, heard):
        assert match_trigger(heard, PHRASES) == "wawi"

    @pytest.mark.parametrize("heard", [
        "guau", "guau guau", "¡Guau!", "wifi", "el wifi no anda", "bueno",
        "bueno, dale", "agua", "Huawei", "tengo un Huawei", "y vos", "¿y vos?",
        "Wally", "wow", "guay", "Hawái", "huevo", "Guagua", "jaguar", "guapa",
        "Chihuahua", "Wabies", "¿Qué hacemos hoy para comer?",
    ])
    def test_does_not_wake(self, heard):
        assert match_trigger(heard, PHRASES) is None

    def test_the_name_late_in_a_sentence_does_not_wake(self):
        assert match_trigger("el otro día le conté a Wawi lo del perro", PHRASES) is None

    def test_aliases_are_exact_never_fuzzy(self):
        # "guaui" is a known spelling; "guau" is one edit away from it and
        # is her own bark. Fuzzy aliases would make every bark a summon.
        assert match_trigger("guaui", PHRASES) == "wawi"
        assert match_trigger("guau", PHRASES) is None

    def test_configured_aliases_add_to_the_built_in_ones(self):
        det = WhisperWakeDetector(PHRASES, aliases={"wawi": ["guaguí"]})
        det._transcribe = lambda a, **kw: {"text": "Guaguí, ¿estás?"}
        assert det.check(np.full(16000, 0.2, np.float32)).ok
        assert "guagui" in det.aliases["wawi"]


class TestSplitRequest:
    @pytest.mark.parametrize("heard, rest", [
        ("Wawi, ¿qué hora es?", "¿qué hora es?"),
        ("Wawi qué hora es", "qué hora es"),
        ("Che Wawi, poné música.", "poné música."),
        ("Chewawi, ¿me escuchás?", "¿me escuchás?"),
        ("Hola Wawi", ""),
        ("Wawi.", ""),
        ("¿Wawi?", ""),
        ("Wawi, Wawi.", "Wawi."),
    ])
    def test_the_rest_is_the_request(self, heard, rest):
        matched, request = split_request(heard, PHRASES)
        assert matched == "wawi"
        assert request == rest

    def test_without_the_phrase_the_text_is_untouched(self):
        assert split_request("¿qué hora es?", PHRASES) == (None, "¿qué hora es?")


class TestPrompt:
    def test_the_prompt_names_the_phrases(self):
        assert wake_prompt(["wawi"]) == "Wawi, Wawi."
        assert wake_prompt([]) == ""

    def test_the_detector_sends_it_and_caps_the_quick_decode(self):
        seen = []

        def stub(audio, **kw):
            seen.append(kw)
            return {"text": "Wawi."}

        det = WhisperWakeDetector(PHRASES)
        det._transcribe = stub
        audio = np.full(16000, 0.2, np.float32)
        assert det.check(audio).ok
        assert det.check(audio, quick=True).ok
        assert seen[0]["initial_prompt"] == "Wawi, Wawi."
        assert "sample_len" not in seen[0] and seen[1]["sample_len"] > 0

    def test_an_empty_prompt_sends_none(self):
        seen = []
        det = WhisperWakeDetector(PHRASES, prompt="")
        det._transcribe = lambda a, **kw: seen.append(kw) or {"text": "x"}
        det.check(np.full(16000, 0.2, np.float32))
        assert "initial_prompt" not in seen[0]

    def test_the_prompt_recited_over_silence_is_not_a_summon(self):
        det = WhisperWakeDetector(PHRASES)
        det._transcribe = lambda a, **kw: {"text": "Wawi, Wawi.", "no_speech_prob": 0.9}
        result = det.check(np.full(16000, 0.2, np.float32))
        assert not result.ok and result.reason == "prompt-echo"

    def test_saying_the_name_twice_is_a_summon(self):
        det = WhisperWakeDetector(PHRASES)
        det._transcribe = lambda a, **kw: {"text": "Wawi, Wawi.", "no_speech_prob": 0.05}
        assert det.check(np.full(16000, 0.2, np.float32)).ok


class TestModeCommand:
    @pytest.mark.parametrize("text", [
        "usá palabra clave", "Usá la palabra clave.", "activá la palabra de activación",
        "modo palabra clave", "solo cuando te llamo", "esperá que te llame",
    ])
    def test_phrase_mode(self, text):
        assert wakemode.parse_command(text) is True

    @pytest.mark.parametrize("text", [
        "escuchá siempre", "escuchame siempre", "sacá la palabra clave",
        "sin palabra clave", "modo siempre",
    ])
    def test_always_on(self, text):
        assert wakemode.parse_command(text) is False

    @pytest.mark.parametrize("text", [
        "¿qué hora es?", "",
        "che, si uso palabra clave y no me escucha, ¿qué hago con eso?",
    ])
    def test_anything_else_is_not_the_command(self, text):
        assert wakemode.parse_command(text) is None

    def test_the_choice_survives_a_restart(self, tmp_path):
        local = tmp_path / "wake_local.json"
        assert wakemode.read_local(local) == {}
        assert wakemode.write_local(local, True)
        assert wakemode.read_local(local) == {"require_phrase": True}

    def test_a_broken_file_is_ignored(self, tmp_path):
        local = tmp_path / "wake_local.json"
        local.write_text("{nope")
        assert wakemode.read_local(local) == {}


def _write_cfg(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body))
    return path


class TestConfig:
    def test_defaults(self, tmp_path):
        cfg = config_mod.load(_write_cfg(tmp_path, "language: es\n"))
        assert cfg.wake_require_phrase is False
        assert cfg.wake_window_s == 1.5
        assert cfg.wake_whisper_prompt is None  # "auto"
        assert cfg.wake_model == ""
        assert cfg.wake_verify_model == "wawi" and cfg.wake_verify_threshold == 0.5
        assert cfg.wake_patience == 1
        assert cfg.wake_local_file == tmp_path / "wake_local.json"

    def test_the_local_file_overrides_config_yaml(self, tmp_path):
        (tmp_path / "wake_local.json").write_text('{"require_phrase": true}')
        cfg = config_mod.load(_write_cfg(tmp_path, "language: es\n"))
        assert cfg.wake_require_phrase is True

    def test_aliases_as_a_list_belong_to_the_first_phrase(self, tmp_path):
        cfg = config_mod.load(_write_cfg(tmp_path, """
            wake:
              phrases: ["wawi"]
              aliases: ["guaguí", "uawi"]
              whisper_prompt: ""
              window_s: 0
              model: wawi
        """))
        assert cfg.wake_aliases == {"wawi": ("guaguí", "uawi")}
        assert cfg.wake_whisper_prompt == ""
        assert cfg.wake_window_s == 0.0
        assert cfg.wake_model == "wawi"


class _Panel:
    def __init__(self):
        self.mode = None

    def set_wake_mode(self, on):
        self.mode = on


class _Printer:
    def banner(self, *a):
        pass


class _Listener:
    def __init__(self, detector):
        self._wake_detector = detector
        self.require_phrase = False
        self.config = type("C", (), {"engine": "whisper"})()

    def set_require_phrase(self, on):
        self.require_phrase = on
        return True


class _Cfg:
    wake_phrases = ("wawi",)
    assistant_name = "Wawi"

    def __init__(self, local):
        self.wake_local_file = local


def _orch(tmp_path):
    o = object.__new__(Orchestrator)
    o._cfg = _Cfg(tmp_path / "wake_local.json")
    o._language = "es"
    o.listener = _Listener(WhisperWakeDetector(PHRASES))
    o.panel = _Panel()
    o.printer = _Printer()
    return o


class TestLocalReply:
    def test_the_wake_word_is_taken_off_the_request(self, tmp_path):
        o = _orch(tmp_path)
        assert o._local_reply("Wawi, ¿qué hora es?", None) == ("¿qué hora es?", None)

    def test_the_bare_name_gets_a_short_question(self, tmp_path):
        o = _orch(tmp_path)
        text, reply = o._local_reply("Che Wawi.", None)
        assert text == "" and reply == "¿Sí?"

    def test_ordinary_requests_pass_through(self, tmp_path):
        o = _orch(tmp_path)
        assert o._local_reply("poné música", None) == ("poné música", None)

    def test_the_mode_command_switches_and_persists(self, tmp_path):
        o = _orch(tmp_path)
        text, reply = o._local_reply("Wawi, usá palabra clave", None)
        assert text == "" and "Wawi" in reply
        assert o.listener.require_phrase is True
        assert o.panel.mode is True
        assert wakemode.read_local(tmp_path / "wake_local.json") == {"require_phrase": True}
        text, reply = o._local_reply("escuchá siempre", None)
        assert o.listener.require_phrase is False and "siempre" in reply

    def test_without_a_transcriber_the_command_is_just_a_request(self, tmp_path):
        o = _orch(tmp_path)
        o.listener._wake_detector = None
        assert o._local_reply("usá palabra clave", None) == ("usá palabra clave", None)

    def test_the_menu_switch_runs_off_the_audio_thread(self, tmp_path):
        o = _orch(tmp_path)
        done = threading.Event()
        o.listener.set_require_phrase = lambda on: done.set() or True
        o._on_wake_mode_toggled(True)
        assert done.is_set() and o.panel.mode is True


class TestShippedModel:
    """The trained wawi.onnx, through the real openWakeWord. Skipped without it."""

    def _verifier(self):
        pytest.importorskip("openwakeword")
        from wawi.listener import custom_wake_model_missing, resolve_wake_model
        from wawi.wake import AcousticVerifier

        if custom_wake_model_missing("wawi"):
            pytest.skip("wawi.onnx not present")
        return AcousticVerifier(resolve_wake_model("wawi")[1])

    def test_silence_and_noise_do_not_score(self):
        v = self._verifier()
        rng = np.random.default_rng(0)
        assert v.score(np.zeros(32000, np.float32)) < 0.5
        assert v.score(rng.normal(0, 0.05, 32000).astype(np.float32)) < 0.5

    def test_the_orchestrator_builds_it_by_default(self, tmp_path):
        self._verifier()
        cfg = config_mod.load(_write_cfg(tmp_path, "language: es\n"))
        assert Orchestrator._build_verifier(cfg) is not None
        cfg2 = config_mod.load(_write_cfg(tmp_path, "wake:\n  verify_model: ''\n"))
        assert Orchestrator._build_verifier(cfg2) is None
