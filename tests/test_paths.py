"""wawi/paths.py: the one private folder, and the move into it."""

from __future__ import annotations

import subprocess
from pathlib import Path

from wawi import config as config_mod
from wawi import paths

REPO = Path(__file__).resolve().parents[1]


class TestHelper:
    def test_default_is_private_relative(self, monkeypatch):
        monkeypatch.delenv(paths.ENV, raising=False)
        assert paths.private_dir() == Path("private")
        assert paths.private("wawi.log") == Path("private/wawi.log")

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv(paths.ENV, str(tmp_path / "elsewhere"))
        assert paths.private("recall.txt") == tmp_path / "elsewhere" / "recall.txt"

    def test_resolve_bare_name_goes_private(self, tmp_path, monkeypatch):
        monkeypatch.setenv(paths.ENV, str(tmp_path))
        assert paths.resolve("conversation.log") == tmp_path / "conversation.log"

    def test_resolve_keeps_explicit_paths(self, tmp_path):
        assert paths.resolve(tmp_path / "x.log") == tmp_path / "x.log"
        assert paths.resolve("logs/x.log") == Path("logs/x.log")
        assert paths.resolve("~/x.log") == Path("~/x.log").expanduser()

    def test_under_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(paths.ENV, "private")
        assert paths.under(tmp_path) == tmp_path / "private"
        monkeypatch.setenv(paths.ENV, str(tmp_path / "abs"))
        assert paths.under(Path("/nowhere")) == tmp_path / "abs"

    def test_config_path_prefers_private(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.ENV, "private")
        assert paths.config_path() == Path("private/config.yaml")  # neither: the new one
        (tmp_path / "config.yaml").write_text("language: es\n")
        assert paths.config_path() == Path("config.yaml")  # not moved yet
        (tmp_path / "private").mkdir()
        (tmp_path / "private" / "config.yaml").write_text("language: es\n")
        assert paths.config_path() == Path("private/config.yaml")

    def test_root_config_still_loads_with_state_in_private(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.ENV, "private")
        (tmp_path / "config.yaml").write_text("language: es\n")
        cfg = config_mod.load()
        assert cfg.wake_local_file == Path("private/wake_local.json")
        assert Path(cfg.reminders["local_file"]) == Path("private/reminders_local.json")
        assert cfg.conversation_log == Path("private/conversation.log")

    def test_private_config_keeps_local_files_beside_it(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.ENV, "private")
        (tmp_path / "private").mkdir()
        (tmp_path / "private" / "config.yaml").write_text(
            "language: es\nconversation_log: conversation.log\n")
        cfg = config_mod.load()
        assert cfg.conversation_log == Path("private/conversation.log")
        assert Path(cfg.selffix["local_file"]) == Path("private/selffix_local.json")


class TestMigrate:
    def test_moves_old_files_and_dirs(self, tmp_path):
        (tmp_path / "conversation.log").write_text("hola")
        (tmp_path / "orca_local.json").write_text("{}")
        (tmp_path / "issues").mkdir()
        (tmp_path / "issues" / "a.md").write_text("issue")
        (tmp_path / "README.md").write_text("not personal")
        moved = paths.migrate(tmp_path, tmp_path / "private")
        assert sorted(moved) == ["conversation.log", "issues", "orca_local.json"]
        assert (tmp_path / "private" / "conversation.log").read_text() == "hola"
        assert (tmp_path / "private" / "issues" / "a.md").read_text() == "issue"
        assert not (tmp_path / "conversation.log").exists()
        assert not (tmp_path / "issues").exists()
        assert (tmp_path / "README.md").exists()

    def test_never_overwrites(self, tmp_path):
        (tmp_path / "private").mkdir()
        (tmp_path / "private" / "recall.txt").write_text("new")
        (tmp_path / "recall.txt").write_text("old")
        assert paths.migrate(tmp_path, tmp_path / "private") == []
        assert (tmp_path / "private" / "recall.txt").read_text() == "new"
        assert (tmp_path / "recall.txt").read_text() == "old"

    def test_idempotent(self, tmp_path):
        (tmp_path / "pet_state.json").write_text("{}")
        assert paths.migrate(tmp_path, tmp_path / "private") == ["pet_state.json"]
        assert paths.migrate(tmp_path, tmp_path / "private") == []
        assert (tmp_path / "private" / "pet_state.json").exists()

    def test_nothing_to_move_creates_nothing(self, tmp_path):
        assert paths.migrate(tmp_path, tmp_path / "private") == []
        assert not (tmp_path / "private").exists()

    def test_relative_dest_is_under_root(self, tmp_path):
        (tmp_path / "config.yaml").write_text("language: es\n")
        assert paths.migrate(tmp_path, Path("private")) == ["config.yaml"]
        assert (tmp_path / "private" / "config.yaml").exists()

    def test_pid_files_are_left_to_wawictl(self):
        # A running Wawi started by an older wawictl is found through it.
        assert "wawi.pid" not in paths.LEGACY_NAMES
        assert "jarvis.pid" not in paths.LEGACY_NAMES

    def test_every_legacy_name_is_gitignored_at_the_root(self):
        ignored = set((REPO / ".gitignore").read_text().split())
        for name in paths.LEGACY_NAMES:
            assert name in ignored or f"{name}/" in ignored or name.endswith(".log"), name
        assert "private/" in ignored


class TestWawictl:
    def _ctl(self, tmp_path: Path) -> Path:
        (tmp_path / "scripts").mkdir()
        ctl = tmp_path / "scripts" / "wawictl"
        ctl.write_text((REPO / "scripts" / "wawictl").read_text())
        ctl.chmod(0o755)
        return ctl

    def _run(self, ctl: Path, *args: str, env_private: str | None = None):
        import os

        env = {k: v for k, v in os.environ.items() if k != paths.ENV}
        if env_private:
            env[paths.ENV] = env_private
        return subprocess.run([str(ctl), *args], capture_output=True, text=True,
                              env=env, timeout=30)

    def test_status_moves_answers_log_and_finds_nothing(self, tmp_path):
        ctl = self._ctl(tmp_path)
        (tmp_path / "answers.log").write_text("old answers")
        r = self._run(ctl, "status")
        assert r.returncode == 1 and "not running" in r.stdout
        assert (tmp_path / "private" / "answers.log").read_text() == "old answers"
        assert not (tmp_path / "answers.log").exists()

    def test_does_not_overwrite_private_answers(self, tmp_path):
        ctl = self._ctl(tmp_path)
        (tmp_path / "private").mkdir()
        (tmp_path / "private" / "answers.log").write_text("new")
        (tmp_path / "answers.log").write_text("old")
        self._run(ctl, "status")
        assert (tmp_path / "private" / "answers.log").read_text() == "new"
        assert (tmp_path / "answers.log").read_text() == "old"

    def test_finds_a_running_wawi_through_the_old_pid_file(self, tmp_path):
        ctl = self._ctl(tmp_path)
        # Any live process whose command line says "wawi" passes the check.
        import sys

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "wawi"])
        try:
            (tmp_path / "wawi.pid").write_text(f"{proc.pid}\n")
            r = self._run(ctl, "status")
            assert r.returncode == 0 and f"pid {proc.pid}" in r.stdout
            # The old pid file is only read, never moved.
            assert (tmp_path / "wawi.pid").exists()
            assert not (tmp_path / "private" / "wawi.pid").exists()
        finally:
            proc.kill()
            proc.wait()

    def test_honours_the_env_override(self, tmp_path):
        ctl = self._ctl(tmp_path)
        (tmp_path / "answers.log").write_text("x")
        self._run(ctl, "status", env_private=str(tmp_path / "elsewhere"))
        assert (tmp_path / "elsewhere" / "answers.log").exists()
