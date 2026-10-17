"""Shortcuts must never run without a real voice confirmation when configured."""

import json
import subprocess

from wawi import shortcuts, tools


def bridge(tmp_path, *, allow=(), confirm=(), names="Café\nCafe Timer\nShazam shortcut\n"):
    calls = []

    def runner(argv, timeout):
        calls.append((argv, timeout))
        if argv[1] == "list":
            return subprocess.CompletedProcess(argv, 0, names, "")
        return subprocess.CompletedProcess(argv, 0, "z" * 4000, "")

    settings = shortcuts.Settings(allow=tuple(allow), confirm=tuple(confirm),
                                  local_file=tmp_path / "shortcuts_local.json")
    return shortcuts.ShortcutsBridge(settings, runner, lambda: 100), calls


def test_list_is_cached_and_persisted(tmp_path):
    service, calls = bridge(tmp_path)
    assert service.list()[2]["names"] == ["Café", "Cafe Timer", "Shazam shortcut"]
    assert service.list()[0]
    assert len(calls) == 1
    assert json.loads((tmp_path / "shortcuts_local.json").read_text())["expires"] == 400


def test_ambiguity_and_allowlist_never_run(tmp_path):
    service, calls = bridge(tmp_path)
    assert set(service._match("cafe tim", ["Cafe Timer", "Cafe Time"])[1]) == {"Cafe Timer", "Cafe Time"}
    result = service.run("cafe")
    # Exact accent-insensitive match selects Café; an allowed-only name cannot
    # be bypassed with fuzzy matching.
    assert result[2]["name"] == "Café"
    restricted, restricted_calls = bridge(tmp_path / "restricted", allow=("Shazam shortcut",))
    assert restricted.run("Cafe")[2]["error"] == "not_found"
    assert all(argv[1] == "list" for argv, _ in restricted_calls)


def test_confirmation_ignores_unrelated_transcript(tmp_path):
    service, calls = bridge(tmp_path, confirm=("Café",))
    assert service.run("cafe")[2]["pending"]
    assert all(argv[1] == "list" for argv, _ in calls)
    assert service.hear("sí, pero después") is None
    assert service.hear("sí") == "Corrí «Café»."
    assert service.run("cafe")[2]["pending"]
    assert service.hear("no") == "Bueno, no corro el atajo."
    assert service.run("cafe")[2]["pending"]
    assert service.hear("sí") == "Corrí «Café»."
    assert len([argv for argv, _ in calls if argv[1] == "run"]) == 2


def test_input_file_and_truncated_output(tmp_path):
    observed = []

    def runner(argv, timeout):
        if argv[1] == "list":
            return subprocess.CompletedProcess(argv, 0, "Demo\n", "")
        observed.append((argv, open(argv[-1]).read()))
        return subprocess.CompletedProcess(argv, 0, "x" * 4000, "")

    service = shortcuts.ShortcutsBridge(shortcuts.Settings(local_file=tmp_path / "state.json"), runner)
    ok, _, data = service.run("Demo", "hola")
    assert ok and data["truncated"] and len(data["output"]) == shortcuts.MAX_OUTPUT
    assert observed[0][0][3] == "-i" and observed[0][1] == "hola"


def test_tool_schema_and_dispatch(tmp_path):
    service, _ = bridge(tmp_path)
    shortcuts.install(service)
    try:
        assert tools.dispatch("list_shortcuts", {}).ok
        assert tools.dispatch("run_shortcut", {"name": "Shazam shortcut"}).ok
        assert {s["name"] for s in tools.active_tools()} >= {"list_shortcuts", "run_shortcut"}
        assert all("animation" not in s for s in tools.active_tools())
    finally:
        shortcuts.install(None)


def test_cli_failure_and_timeout_are_spoken(tmp_path):
    def failed(argv, timeout):
        return subprocess.CompletedProcess(argv, 1, "", "sin permiso")

    service = shortcuts.ShortcutsBridge(
        shortcuts.Settings(local_file=tmp_path / "state.json"), failed)
    ok, speech, data = service.list()
    assert not ok and "Atajos falló" in speech and data["error"] == "list_failed"

    def delayed(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    service = shortcuts.ShortcutsBridge(
        shortcuts.Settings(local_file=tmp_path / "state2.json"), delayed)
    assert "tardó demasiado" in service.list()[1]
