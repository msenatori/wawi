"""The supervisor of a self-applied fix: restart, watch, and revert if bad.

It must not depend on the code it supervises working, so this file is
**standard library only** and imports nothing from ``wawi``. Before a
self-applied restart the service copies it into the attempt's directory and
runs that frozen copy, detached in a session of its own, with one JSON job
file::

    python watchdog.py <attempt_dir>/watch_job.json

What it does, in order:

1. Restarts Wawi with ``wawictl restart`` (it is the restart: Wawi itself
   cannot survive its own stop).
2. **Startup** (``startup_s``): the process must be up, ``Wawi ready`` must be
   logged after the restart, and no uncaught traceback or ERROR line may
   appear before it. Otherwise: revert.
3. **Watch** (``watch_s``): the fixed issue must not recur at the same or a
   higher rate than before (counted from the journal's ``fp=<fingerprint>``
   log lines), and Wawi must not die without saying "Wawi stopped." (a
   crash) nor come up more than once on its own (a crash loop). Otherwise:
   revert. A clean stop by the owner ends the watch, inconclusive, with no
   revert.

Revert rules, the whole reason this file exists:

- Only ever ``git revert`` (never reset, never force), and only of the one
  commit it was given, after checking that commit carries the
  ``Wawi-Selffix:`` trailer, is an ancestor of HEAD and was not already
  reverted. Commits that are not self-fixes are never touched.
- Only on the main branch with no uncommitted changes to tracked files: a
  human working in the tree wins, the revert is skipped and reported.
- After the revert, restart again, and write the verdict to ``result_path``
  for the service to fold into its history on its next run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

TRAILER = "Wawi-Selffix"
REVERT_TRAILER = "Wawi-Selffix-Revert"
READY_MARK = "Wawi ready"
STOPPED_MARK = "Wawi stopped."
_LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ (?P<level>[A-Z]+) ")

#: git(repo, *args) -> (returncode, stdout, stderr)
Git = Callable[..., "tuple[int, str, str]"]


def run_git(repo: str, *args: str, timeout: float = 120.0) -> tuple[int, str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              timeout=timeout, env=env, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


# --------------------------------------------------------------------------- #
# Revert
# --------------------------------------------------------------------------- #


def commit_message(repo: str, sha: str, git: Git = run_git) -> str:
    code, out, _ = git(repo, "log", "-1", "--format=%B", sha)
    return out if code == 0 else ""


def is_selffix(message: str) -> bool:
    return re.search(rf"^{TRAILER}:\s*\S+", message or "", re.M) is not None


def already_reverted(repo: str, sha: str, git: Git = run_git) -> bool:
    code, out, _ = git(repo, "log", "--format=%B", f"--grep={REVERT_TRAILER}: {sha}", "HEAD")
    return code == 0 and f"{REVERT_TRAILER}: {sha}" in out


def tracked_dirty(repo: str, git: Git = run_git) -> bool:
    code, out, _ = git(repo, "status", "--porcelain", "--untracked-files=no")
    return code != 0 or bool(out.strip())


def revert_selffix(repo: str, sha: str, reason: str, *, branch: str = "main",
                   git: Git = run_git) -> tuple[bool, str]:
    """``git revert`` one self-fix commit. (ok, detail). Never resets."""
    code, out, _ = git(repo, "rev-parse", "--verify", f"{sha}^{{commit}}")
    if code != 0:
        return False, "no_such_commit"
    sha = out.strip()
    message = commit_message(repo, sha, git)
    if not is_selffix(message):
        return False, "not_a_selffix_commit"
    code, out, _ = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if code != 0 or out.strip() != branch:
        return False, "not_on_main"
    if git(repo, "merge-base", "--is-ancestor", sha, "HEAD")[0] != 0:
        return False, "not_in_history"
    if already_reverted(repo, sha, git):
        return False, "already_reverted"
    if tracked_dirty(repo, git):
        return False, "dirty_tree"
    code, _, err = git(repo, "revert", "--no-edit", "--no-commit", sha)
    if code != 0:
        git(repo, "revert", "--abort")
        return False, f"revert_failed: {err.strip()[:200]}"
    subject = (message.splitlines() or [""])[0][:60]
    issue = re.search(rf"^{TRAILER}:\s*(\S+)", message, re.M)
    body = (f"Revertir arreglo automático: {subject}\n\n"
            f"Wawi deshizo su propio arreglo: {reason}\n\n"
            f"{REVERT_TRAILER}: {sha}\n"
            f"{TRAILER}-Issue: {issue.group(1) if issue else '?'}\n")
    code, _, err = git(repo, "commit", "--no-verify", "-m", body)
    if code != 0:
        git(repo, "revert", "--abort")
        return False, f"commit_failed: {err.strip()[:200]}"
    code, out, _ = git(repo, "rev-parse", "HEAD")
    return True, out.strip()


# --------------------------------------------------------------------------- #
# Reading the log
# --------------------------------------------------------------------------- #


def read_from(path: str, offset: int) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def startup_problems(text: str) -> list[str]:
    """Uncaught tracebacks and ERROR/CRITICAL lines up to "Wawi ready"."""
    cut = text.find(READY_MARK)
    head = text if cut < 0 else text[:cut]
    problems = []
    previous_level = ""
    for line in head.splitlines():
        m = _LOG_LINE.match(line)
        if m:
            previous_level = m.group("level")
            if previous_level in ("ERROR", "CRITICAL"):
                problems.append(line.strip()[:200])
            continue
        if line.startswith("Traceback") and previous_level not in ("ERROR", "CRITICAL", "WARNING"):
            problems.append("uncaught traceback")
    return problems


def count_issue(text: str, fingerprint: str) -> int:
    if not fingerprint:
        return 0
    return len(re.findall(rf"fp={re.escape(fingerprint)}\b", text))


def unclean_restarts(text: str) -> int:
    """Times Wawi came up again WITHOUT having logged a clean stop first.

    *text* starts at the first "Wawi ready". A restart from the menu logs
    "Wawi stopped." before the next "Wawi ready"; a crash does not."""
    segments = text.split(READY_MARK)
    # segments[0] precedes the first ready; each later boundary is a restart.
    return sum(1 for seg in segments[1:-1] if STOPPED_MARK not in seg)


def pid_alive(pid_file: str) -> bool:
    try:
        pid = int(Path(pid_file).read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# The watch
# --------------------------------------------------------------------------- #


class Watch:
    """One supervised restart. Every side effect is injectable for tests."""

    def __init__(self, job: dict, *, git: Git = run_git,
                 restart: Callable[[], None] | None = None,
                 alive: Callable[[], bool] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.job = job
        self.git = git
        self._restart = restart
        self._alive = alive
        self.clock = clock
        self.sleep = sleep

    # -- effects ------------------------------------------------------------
    def restart(self) -> None:
        if self._restart is not None:
            self._restart()
            return
        ctl = self.job.get("wawictl")
        if ctl:
            subprocess.run([ctl, "restart"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120)

    def alive(self) -> bool:
        if self._alive is not None:
            return self._alive()
        return pid_alive(self.job.get("pid_file", ""))

    # -- the run ------------------------------------------------------------
    def run(self) -> dict:
        job = self.job
        log_path = job["log_path"]
        poll = float(job.get("poll_s", 10.0))
        offset = file_size(log_path)
        if job.get("restart", True):
            self.restart()
        # 1. Startup.
        deadline = self.clock() + float(job.get("startup_s", 90.0))
        ready = False
        while self.clock() < deadline:
            text = read_from(log_path, offset)
            if READY_MARK in text:
                ready = True
                break
            self.sleep(min(poll, 2.0))
        text = read_from(log_path, offset)
        problems = startup_problems(text)
        if not ready:
            return self.rollback("no arrancó: no dijo «Wawi ready»", problems)
        if problems:
            return self.rollback("errores al arrancar", problems)
        if not self.alive():
            return self.rollback("se cayó al arrancar", [])
        # 2. Watch.
        ready_at = text.find(READY_MARK)
        watch_start = self.clock()
        deadline = watch_start + float(job.get("watch_s", 1800.0))
        baseline = float(job.get("baseline_per_hour", 0.0))
        fingerprint = str(job.get("fingerprint", ""))
        dead_checks = 0
        while self.clock() < deadline:
            self.sleep(poll)
            text = read_from(log_path, offset)
            after = text[ready_at:]
            if STOPPED_MARK in after and not self.alive():
                return self.finish("inconclusive", "lo cerraste vos; dejé de mirar", [])
            if unclean_restarts(after) > 0:
                return self.rollback("se cayó y volvió a arrancar (crash loop)",
                                     startup_problems(after))
            if not self.alive():
                dead_checks += 1
                if dead_checks >= 2:
                    return self.rollback("se cayó sin cerrar bien", [])
            else:
                dead_checks = 0
            seen = count_issue(after, fingerprint)
            hours = max(1e-6, (self.clock() - watch_start) / 3600.0)
            window_h = float(job.get("watch_s", 1800.0)) / 3600.0
            # The same or a higher rate than before the fix -- judged once the
            # count is meaningful (two recurrences, or one over a slow baseline).
            if seen >= 2 or (seen >= 1 and baseline * window_h < 1.0):
                if seen / hours >= baseline:
                    return self.rollback(
                        f"el problema volvió ({seen} veces en {hours * 60:.0f} min)", [])
        return self.finish("healthy", "anduvo bien media hora", [])

    def finish(self, outcome: str, reason: str, evidence: list[str],
               revert_sha: str = "") -> dict:
        result = {"outcome": outcome, "reason": reason, "evidence": evidence[:20],
                  "commit": self.job.get("commit", ""), "revert": revert_sha,
                  "issue": self.job.get("issue", ""), "attempt": self.job.get("attempt", ""),
                  "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        path = self.job.get("result_path")
        if path:
            tmp = f"{path}.tmp"
            Path(tmp).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        return result

    def rollback(self, reason: str, evidence: list[str]) -> dict:
        ok, detail = revert_selffix(self.job["repo"], self.job["commit"], reason,
                                    branch=self.job.get("branch", "main"), git=self.git)
        if not ok:
            return self.finish("revert_blocked", reason, [*evidence, f"no pude revertir: {detail}"])
        self.restart()
        return self.finish("rolled_back", reason, evidence, revert_sha=detail)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: watchdog.py <watch_job.json>", file=sys.stderr)
        return 2
    job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    Watch(job).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
