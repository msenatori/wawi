"""The tool layer: four tools, and nothing that can hurt you.

What is NOT here is the point of the module. Wawi listens all day; a
mistranscription has to be cheap. So the tools write a screenshot, a preference
line, an issue file and the clipboard, and that is the entire blast radius — no commands, no
code edits, no keystrokes into anything live.
"""

from __future__ import annotations

import json

import pytest

from wawi import tools


class TestDispatch:
    def test_every_declared_tool_can_actually_be_called(self):
        # A schema with no implementation is a tool the model will try, fail to
        # use, and apologise for — with the user hearing the apology.
        # Both lists, because both are dispatched through the same table: the
        # task schemas are merely held back until an executor is installed.
        declared = {
            tool["name"]
            for tool in (
                *tools.TOOLS, *tools.TASK_TOOLS,
                *tools.MIND_TOOLS, *tools.BOARD_TOOLS, *tools.PET_TOOLS, *tools.SENSOR_TOOLS,
                *tools.INSTINCT_TOOLS, *tools.INITIATIVE_TOOLS, *tools.GAME_TOOLS, *tools.ORCA_TOOLS,
                *tools.SHORTCUT_TOOLS,
                *tools.REMINDER_TOOLS, *tools.SELFFIX_TOOLS, *tools.WEBVM_TOOLS,
            )
        }
        assert declared == set(tools._DISPATCH)

    def test_an_unknown_tool_speaks_instead_of_raising(self):
        # The agent loop has no way to say a traceback out loud.
        result = tools.dispatch("borrar_todo", {})

        assert result.ok is False
        assert result.data["error"] == "unknown_tool"

    def test_missing_arguments_do_not_raise(self):
        result = tools.dispatch("report_issue", None)

        assert result.ok is False

    def test_a_tool_that_raises_is_reported_not_propagated(self, monkeypatch):
        def _boom(_args):
            raise RuntimeError("boom")

        monkeypatch.setitem(tools._DISPATCH, "report_issue", _boom)
        result = tools.dispatch("report_issue", {"title": "x"})

        assert result.ok is False
        assert result.data["error"] == "tool_failed"


class TestSchemas:
    def test_they_stay_terse(self):
        """The schemas are re-sent on every turn; keep the surface small.

        Raising these numbers is a cost decision, not a formality: every tool
        and every character is paid for on every single question.
        """
        assert len(tools.TOOLS) <= 4
        serialized = json.dumps(tools.TOOLS)
        assert len(serialized) < 1400, f"tool schemas grew to {len(serialized)} chars"

    def test_capture_screen_lists_the_spanish_synonyms(self):
        """The model has to route "pantallazo" and "screenshot" to this tool."""
        spec = next(t for t in tools.TOOLS if t["name"] == "capture_screen")
        described = spec["description"].lower()
        for word in ("captura", "screenshot", "pantallazo"):
            assert word in described

    def test_active_tools_is_a_copy(self):
        # Callers hand this to the SDK; mutating it must not edit the source.
        active = tools.active_tools()
        active.clear()
        assert tools.TOOLS


class TestToolResult:
    def test_overlong_speech_is_clipped_on_construction(self):
        # last_assistant_message can be thousands of characters of markdown;
        # feeding that to TTS talks for minutes.
        result = tools.ToolResult(True, "a" * 5000)

        assert len(result.speak) <= tools.MAX_SPEAK_CHARS


class TestCaptureScreen:
    """A screenshot saved as a FILE, not the image the model looks at."""

    def _fake_screen(self, monkeypatch, fail=False):
        """Patch the real module's attributes.

        NOT sys.modules: ``capture_screen`` does ``from wawi import screen``,
        which reads the attribute already bound on the package, so replacing
        the entry in sys.modules has no effect on it.
        """
        from wawi import screen as real_screen

        calls = {}

        def capture(monitor=1):
            if fail:
                raise RuntimeError("screen recording denied")
            calls["monitor"] = monitor
            return "IMG"

        def save_jpeg(img, directory, **kw):
            from pathlib import Path

            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "pantalla-test.jpg"
            path.write_bytes(b"jpeg")
            calls["directory"] = str(directory)
            return path

        monkeypatch.setattr(real_screen, "capture", capture)
        monkeypatch.setattr(real_screen, "save_jpeg", save_jpeg)
        return calls

    def test_a_screenshot_is_written_and_the_path_reported(self, monkeypatch, tmp_path):
        calls = self._fake_screen(monkeypatch)
        tools.set_screen_options(str(tmp_path / "shots"), 1)

        result = tools.dispatch("capture_screen", {})

        assert result.ok
        assert result.data["path"].endswith(".jpg")
        assert "shots" in calls["directory"]

    def test_the_configured_monitor_is_used(self, monkeypatch, tmp_path):
        calls = self._fake_screen(monkeypatch)
        tools.set_screen_options(str(tmp_path), 2)

        tools.dispatch("capture_screen", {})

        assert calls["monitor"] == 2

    @pytest.mark.parametrize("monitor", [0, 2])
    def test_requested_monitor_overrides_default(self, monkeypatch, tmp_path, monitor):
        calls = self._fake_screen(monkeypatch)
        tools.set_screen_options(str(tmp_path), 1)

        result = tools.dispatch("capture_screen", {"monitor": monitor})

        assert result.ok
        assert calls["monitor"] == monitor
        spec = next(t for t in tools.TOOLS if t["name"] == "capture_screen")
        assert "monitor" in spec["input_schema"]["properties"]

    def test_a_capture_failure_names_the_likely_cause(self, monkeypatch, tmp_path):
        """Screen Recording permission is the reason this fails in practice."""
        self._fake_screen(monkeypatch, fail=True)
        tools.set_screen_options(str(tmp_path), 1)

        result = tools.dispatch("capture_screen", {})

        assert not result.ok
        assert "pantalla" in result.speak.lower()
        assert result.data["error"] == "capture_failed"

    def test_the_note_survives_into_the_result(self, monkeypatch, tmp_path):
        self._fake_screen(monkeypatch)
        tools.set_screen_options(str(tmp_path), 1)

        result = tools.dispatch("capture_screen", {"note": "mirá el panel"})

        assert result.data["note"] == "mirá el panel"


class TestRememberPreference:
    def test_it_refuses_when_there_is_nowhere_to_write(self, monkeypatch):
        # Silently accepting would teach the user a habit that does nothing.
        monkeypatch.setattr(tools, "_PREFERENCES", None)
        result = tools.dispatch("remember_preference", {"text": "sé más breve"})

        assert result.ok is False
        assert result.data["error"] == "preferences_disabled"

    def test_it_stores_the_instruction(self, monkeypatch):
        class _Store:
            enabled = True

            def __init__(self):
                self.added = []

            def add(self, text):
                self.added.append(text)
                return type("P", (), {"text": text})()

        store = _Store()
        monkeypatch.setattr(tools, "_PREFERENCES", store)

        result = tools.dispatch("remember_preference", {"text": "sé más breve"})

        assert result.ok is True
        assert store.added == ["sé más breve"]


class TestReportIssue:
    def test_it_refuses_when_the_journal_is_off(self, monkeypatch):
        monkeypatch.setattr(tools, "_JOURNAL", None)
        result = tools.dispatch("report_issue", {"title": "contestó cualquiera"})

        assert result.ok is False
        assert result.data["error"] == "journal_disabled"

    def test_an_empty_title_is_refused(self, monkeypatch):
        monkeypatch.setattr(tools, "_JOURNAL", type("J", (), {"enabled": True})())
        result = tools.dispatch("report_issue", {"title": "   "})

        assert result.ok is False
        assert result.data["error"] == "empty_title"

    def test_a_repeat_report_says_how_many_times(self, monkeypatch):
        # Hearing "ya van 3 veces" is the difference between a one-off and a
        # pattern, and the user is the only one who can tell you which it is.
        class _Journal:
            enabled = True

            def record(self, *a, **kw):
                return type("I", (), {"count": 3})()

        monkeypatch.setattr(tools, "_JOURNAL", _Journal())
        result = tools.dispatch("report_issue", {"title": "se cortó"})

        assert result.ok is True
        assert "3" in result.speak


class TestCopyToClipboard:
    """"Copiá eso": the voice path to the panel's copy pill."""

    def _fake_run(self, monkeypatch, exc=None):
        import subprocess

        calls = []

        def run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if exc is not None:
                raise exc
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", run)
        return calls

    def test_copies_the_text_verbatim_through_pbcopy(self, monkeypatch):
        calls = self._fake_run(monkeypatch)
        text = "git log --oneline | head -5\n  && echo 'acción'"

        result = tools.dispatch("copy_to_clipboard", {"text": text})

        assert result.ok is True
        assert result.speak == "Listo, copiado."
        assert len(calls) == 1
        cmd, kwargs = calls[0]
        # No shell: the text is stdin, never part of a command line.
        assert cmd == ["pbcopy"]
        assert not kwargs.get("shell")
        assert kwargs["input"] == text.encode("utf-8")
        assert kwargs["env"]["LC_ALL"].endswith("UTF-8")
        assert kwargs["timeout"] > 0

    def test_empty_text_copies_nothing(self, monkeypatch):
        calls = self._fake_run(monkeypatch)

        result = tools.copy_to_clipboard("  \n ")

        assert result.ok is False
        assert result.data["error"] == "empty"
        assert calls == []

    def test_a_failing_pbcopy_is_spoken_not_raised(self, monkeypatch):
        import subprocess

        self._fake_run(monkeypatch, exc=subprocess.TimeoutExpired("pbcopy", 5))

        result = tools.copy_to_clipboard("ls -la")

        assert result.ok is False
        assert result.data["error"] == "copy_failed"
        assert "portapapeles" in result.speak

    def test_a_missing_pbcopy_is_spoken_not_raised(self, monkeypatch):
        self._fake_run(monkeypatch, exc=FileNotFoundError("pbcopy"))

        assert tools.copy_to_clipboard("ls").ok is False

    def test_it_is_always_on_and_tells_the_model_to_copy_verbatim(self):
        spec = next(t for t in tools.active_tools() if t["name"] == "copy_to_clipboard")
        described = spec["description"].lower()
        assert "copiá eso" in described
        assert "exacto" in described
        assert spec["input_schema"]["required"] == ["text"]

    def test_the_brain_pre_approves_it(self):
        """dontAsk + allowed_tools: a tool not listed there is silently denied."""
        from wawi import brain

        pytest.importorskip("claude_agent_sdk")
        bridge = brain._build_bridge(brain._TurnState(), None)
        assert f"mcp__{brain.MCP_SERVER_NAME}__copy_to_clipboard" in bridge.tool_names

    def test_the_system_prompt_mentions_it(self):
        from wawi import prompts

        assert "copy_to_clipboard" in prompts.system_prompt(language="es")
        assert "copy_to_clipboard" in prompts.system_prompt(language="en")
