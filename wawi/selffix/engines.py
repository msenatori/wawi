"""The coding engines (Codex, Claude Code) and where they run (executors).

Engines
-------
Both are driven non-interactively, prompt on stdin, from the attempt's
worktree:

- **codex** (default author, the owner has credit there)::

      codex exec -C <worktree> -s workspace-write --ephemeral
                 --skip-git-repo-check -o <last_message_file> [-m model] -

  ``workspace-write`` lets it edit the worktree and run the tests, nothing
  else. As a reviewer it gets ``-s read-only``.
- **claude** (fallback author, default reviewer of Codex' work)::

      claude -p --permission-mode acceptEdits --max-budget-usd <usd>
             --no-session-persistence --allowedTools <edit + pytest + git diff>

  As a reviewer it only gets Read/Grep/Glob.

Neither commits: the worktree's ``.git`` points back into the main repo, out
of a sandboxed agent's reach, so the service commits what the agent left.

Executors
---------
Every engine run is a :mod:`wawi.selffix.runjob` job: a command, a hard
time budget, a result file. :class:`LocalExecutor` runs it as a child of this
process. :class:`OrcaExecutor` types the same command into a visible Orca
terminal, so the owner can watch her work in his Orca window, and waits for
the result file; when Orca is not running (or refuses) it falls back to the
local executor. Orca's supervised dispatch (worker-start / worker_done) is
NOT used: its safety floor forbids stopping a worker without proof it
exited, and a self-fix needs a hard 30-minute kill.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from wawi.selffix import runjob

log = logging.getLogger(__name__)

RUNJOB_PATH = Path(runjob.__file__).resolve()


@dataclass(frozen=True)
class Job:
    name: str
    argv: list[str]
    cwd: Path
    timeout_s: float
    stdin_text: str = ""
    dir: Path = Path(".")
    visible: bool = False        # worth showing in Orca (the author, not git)
    title: str = ""

    @property
    def out_file(self) -> Path:
        return self.dir / f"{self.name}.out"

    @property
    def result_file(self) -> Path:
        return self.dir / f"{self.name}.result.json"


@dataclass(frozen=True)
class JobResult:
    code: int
    timed_out: bool = False
    elapsed_s: float = 0.0
    output: str = ""
    where: str = "local"

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timed_out


def _job_dict(job: Job) -> dict:
    job.dir.mkdir(parents=True, exist_ok=True)
    stdin_file = ""
    if job.stdin_text:
        path = job.dir / f"{job.name}.stdin"
        path.write_text(job.stdin_text, encoding="utf-8")
        stdin_file = str(path)
    for stale in (job.out_file, job.result_file):
        stale.unlink(missing_ok=True)
    return {"argv": list(job.argv), "cwd": str(job.cwd), "timeout_s": job.timeout_s,
            "stdin_file": stdin_file, "out_file": str(job.out_file),
            "result_file": str(job.result_file)}


def _read_result(job: Job, where: str) -> JobResult:
    try:
        raw = json.loads(job.result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {"code": -1, "timed_out": True}
    try:
        output = job.out_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = ""
    return JobResult(int(raw.get("code", -1)), bool(raw.get("timed_out")),
                     float(raw.get("elapsed_s", 0.0)), output, where)


class LocalExecutor:
    name = "local"

    def run(self, job: Job) -> JobResult:
        runjob.run(_job_dict(job))
        return _read_result(job, "local")


class OrcaExecutor:
    """The same job in a visible Orca terminal; local when Orca is not there."""

    name = "orca"

    def __init__(self, cli: Any, repo: Path, *, fallback: LocalExecutor | None = None,
                 python: str = sys.executable, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, poll_s: float = 5.0) -> None:
        self.cli = cli            # wawi.orca.OrcaCLI (or a fake with .run)
        self.repo = repo
        self.fallback = fallback or LocalExecutor()
        self.python = python
        self.sleep = sleep
        self.clock = clock
        self.poll_s = poll_s

    def available(self) -> bool:
        try:
            result = self.cli.run("status", timeout=15.0)
        except Exception:
            return False
        from wawi.orca import find_key

        return bool(find_key(result.get("runtime") or {}, "reachable"))

    def run(self, job: Job) -> JobResult:
        if not job.visible or not self.available():
            return self.fallback.run(job)
        spec = job.dir / f"{job.name}.job.json"
        spec.write_text(json.dumps(_job_dict(job)), encoding="utf-8")
        command = shlex.join([self.python, str(RUNJOB_PATH), str(spec)])
        try:
            from wawi.orca import find_key

            created = self.cli.run("terminal", "create", "--worktree", f"path:{self.repo}",
                                   "--title", job.title or f"Wawi · Autoarreglo · {job.name}",
                                   "--command", command)
            handle = str(find_key(created, "handle") or "")
        except Exception as exc:
            log.info("selffix: Orca terminal refused (%s); running locally", exc)
            return self.fallback.run(job)
        deadline = self.clock() + job.timeout_s + 120.0
        while self.clock() < deadline and not job.result_file.exists():
            self.sleep(self.poll_s)
        if handle:
            try:
                self.cli.run("terminal", "close", "--terminal", handle)
            except Exception:
                log.debug("selffix: could not close Orca terminal %s", handle, exc_info=True)
        if not job.result_file.exists():
            return JobResult(-1, True, job.timeout_s, "", "orca")
        return _read_result(job, "orca")


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #


def engine_available(engine: str, which: Callable[[str], str | None] = shutil.which) -> bool:
    return which(engine) is not None


def author_argv(engine: str, worktree: Path, last_message: Path, settings: Any,
                python: str) -> list[str]:
    if engine == "codex":
        argv = ["codex", "exec", "-C", str(worktree), "-s", "workspace-write", "--ephemeral",
                "--skip-git-repo-check", "-o", str(last_message)]
        if settings.codex_model:
            argv += ["-m", settings.codex_model]
        return [*argv, "-"]
    argv = ["claude", "-p", "--permission-mode", "acceptEdits",
            "--max-budget-usd", f"{settings.max_budget_usd:.2f}", "--no-session-persistence",
            "--output-format", "text",
            "--allowedTools", "Read", "Edit", "Write", "Glob", "Grep",
            f"Bash({python} -m pytest:*)", "Bash(git diff:*)", "Bash(git status:*)"]
    if settings.claude_model:
        argv += ["--model", settings.claude_model]
    return argv


def reviewer_argv(engine: str, worktree: Path, last_message: Path, settings: Any) -> list[str]:
    if engine == "codex":
        argv = ["codex", "exec", "-C", str(worktree), "-s", "read-only", "--ephemeral",
                "--skip-git-repo-check", "-o", str(last_message)]
        if settings.codex_model:
            argv += ["-m", settings.codex_model]
        return [*argv, "-"]
    argv = ["claude", "-p", "--max-budget-usd", f"{min(settings.max_budget_usd, 1.0):.2f}",
            "--no-session-persistence", "--output-format", "text",
            "--allowedTools", "Read", "Glob", "Grep"]
    if settings.claude_model:
        argv += ["--model", settings.claude_model]
    return argv


def final_message(engine: str, result: JobResult, last_message: Path) -> str:
    """What the agent said last: Codex writes it to a file, Claude to stdout."""
    if engine == "codex":
        try:
            return last_message.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return result.output[-4000:]
    return result.output[-8000:]

