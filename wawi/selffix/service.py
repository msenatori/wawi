"""The self-fix loop: pick an issue, have it fixed in a worktree, gate it,
apply it (or offer it), and hand the restart to the watchdog.

One attempt at a time, at most ``max_attempts_per_day``, only while she is
idle, on AC power and not in a meeting (or in the night window). Every step
that can fail ends the attempt with a reason that is kept, shown in the
day's summary and fed into the next attempt's brief.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Callable

from wawi import paths as _paths
from wawi import working as _working
from wawi.selffix import brief as _brief
from wawi.selffix import classify as _classify
from wawi.selffix import gates as _gates
from wawi.selffix import gitops, issues as _issues, report as _report
from wawi.selffix import watchdog as _watchdog
from wawi.selffix.engines import (
    Job,
    JobResult,
    LocalExecutor,
    author_argv,
    final_message,
    reviewer_argv,
)
from wawi.selffix.settings import SelffixSettings
from wawi.selffix.state import Store

log = logging.getLogger(__name__)

_TIME_FMT = "%Y-%m-%d %H:%M:%S"

#: Outcomes that mean the attempt is over and nothing of it is in main.
FAILED = ("rejected", "error", "rolled_back", "revert_blocked")


def on_ac_power() -> bool | None:
    """True on AC, False on battery, None when unknown (a desktop Mac has no
    battery and ``pmset`` says "AC Power" all the same)."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    if "AC Power" in out:
        return True
    if "Battery Power" in out:
        return False
    return None


_QUIET = None


def in_meeting() -> bool:
    """A meeting app in front or running (the reflexes' detector)."""
    global _QUIET
    try:
        from wawi import reflexes

        state = reflexes.quiet_state()
        if state is None:
            if _QUIET is None:
                _QUIET = reflexes.QuietDetector({})
            state = _QUIET.check()
        return bool(state.meeting)
    except Exception:
        log.debug("selffix: meeting check failed", exc_info=True)
        return False


def keyboard_idle_s() -> float | None:
    try:
        from wawi.sensors.clock import seconds_idle

        return seconds_idle()
    except Exception:
        return None


def notify(text: str) -> None:
    """A bubble, not spoken: she does not interrupt to talk about herself."""
    try:
        from wawi.reactions import BUS, Reaction

        BUS.push(Reaction(bubble=text, priority=1, duration=8.0))
    except Exception:
        log.debug("selffix: bubble failed", exc_info=True)


def spawn_detached(argv: list[str], cwd: Path | None = None) -> None:
    subprocess.Popen(argv, cwd=str(cwd) if cwd else None, start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True)


def _in_window(t: dtime, window: tuple[dtime, dtime] | None) -> bool:
    if window is None:
        return False
    a, b = window
    return a <= t < b if a <= b else (t >= a or t < b)


class SelffixService:
    """See the module docstring. Every side effect is injectable."""

    def __init__(
        self,
        settings: SelffixSettings,
        *,
        repo: Path,
        issues_dir: Path | None,
        log_path: Path | None,
        store: Store,
        python: str | None = None,
        author_executor: Any = None,
        local: Any = None,
        git: Callable[..., tuple[int, str, str]] = _watchdog.run_git,
        which: Callable[[str], str | None] = shutil.which,
        notify: Callable[[str], None] = notify,
        board: Any = None,
        restart_cmd: Callable[[], list[str] | None] = lambda: None,
        spawn: Callable[[list[str]], None] = spawn_detached,
        power: Callable[[], bool | None] = on_ac_power,
        meeting: Callable[[], bool] = in_meeting,
        keyboard_idle: Callable[[], float | None] = keyboard_idle_s,
        pure_python: Callable[[str, str], bool] = _gates.pure_python_on_pypi,
        now: Callable[[], datetime] = datetime.now,
        clock: Callable[[], float] = time.monotonic,
        main_branch: str = "main",
        pid_file: Path | None = None,
    ) -> None:
        self.settings = settings
        self.repo = Path(repo)
        self.issues_dir = issues_dir
        self.log_path = log_path
        self.store = store
        venv = self.repo / ".venv" / "bin" / "python"
        self.python = python or (str(venv) if venv.exists() else sys.executable)
        self.local = local or LocalExecutor()
        self.author_executor = author_executor or self.local
        self.git = git
        self.which = which
        self.notify = notify
        self.board = board
        self.restart_cmd = restart_cmd
        self.spawn = spawn
        self.power = power
        self.meeting = meeting
        self.keyboard_idle = keyboard_idle
        self.pure_python = pure_python
        self.now = now
        self.clock = clock
        self.main = main_branch
        self.pid_file = pid_file or _paths.under(self.repo) / "wawi.pid"
        self._busy = threading.Lock()
        self._last_activity = clock()
        self._work_gates: list[Callable[[], Any]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.running: dict | None = None

    # ------------------------------------------------------------------ switch
    @property
    def enabled(self) -> bool:
        override = self.store["enabled"]
        return self.settings.enabled if override is None else bool(override)

    def set_enabled(self, on: bool) -> None:
        self.store["enabled"] = bool(on)
        self.store.save()
        log.info("Selffix %s", "on" if on else "off")

    # ------------------------------------------------------------ the hooks
    def observe_state(self, _old: Any, new: Any) -> None:
        """StateMachine observer: any turn resets her idle clock."""
        if getattr(new, "name", str(new)) != "IDLE":
            self._last_activity = self.clock()

    def add_work_gate(self, gate: Callable[[], Any]) -> None:
        """Extra condition for working (the initiative layer's, later).

        *gate* returns True/False, or (bool, reason)."""
        self._work_gates.append(gate)

    def watching(self) -> bool:
        """A self-applied change whose watchdog has not given its verdict."""
        return any(a.get("outcome") == "watching" and a.get("watch_result")
                   and not Path(a["watch_result"]).exists() for a in self.store.attempts())

    def attempts_today(self) -> int:
        day = self.now().strftime("%Y-%m-%d")
        return sum(1 for a in self.store.attempts() if str(a.get("started", "")).startswith(day))

    def quota(self) -> tuple[int, int]:
        return self.attempts_today(), self.settings.max_attempts_per_day

    def can_work(self, *, manual: bool = False) -> tuple[bool, str]:
        """(may start an attempt now, why not)."""
        if not self.enabled:
            return False, "apagado"
        if self._busy.locked() or self.running is not None:
            return False, "ocupada"
        used, total = self.quota()
        if used >= total:
            return False, "cupo"
        if self.store["pending"]:
            return False, "pendiente"
        if self.watching():
            return False, "vigilando"
        if not manual:
            night = _in_window(self.now().time(), self.settings.night_window)
            idle_for = self.clock() - self._last_activity
            keys = self.keyboard_idle()
            if not night:
                if idle_for < self.settings.idle_after_s:
                    return False, "no_idle"
                if keys is not None and keys < self.settings.idle_after_s:
                    return False, "no_idle"
            elif idle_for < 60 or (keys is not None and keys < 60):
                return False, "no_idle"
            if self.power() is False:
                return False, "bateria"
            if self.meeting():
                return False, "reunion"
        for gate in self._work_gates:
            try:
                verdict = gate()
            except Exception:
                log.debug("selffix: work gate raised", exc_info=True)
                return False, "gate_error"
            ok, why = (verdict if isinstance(verdict, tuple) else (bool(verdict), "gate"))
            if not ok:
                return False, str(why or "gate")
        return True, ""

    # -------------------------------------------------------- choosing work
    def _issues(self) -> list[_issues.IssueRecord]:
        return _issues.load_issues(self.issues_dir)

    def owner_requests(self) -> list[dict]:
        return list(self.store["owner_requests"])

    def dismiss_request(self, issue_id: str) -> None:
        self.store["owner_requests"] = [r for r in self.store["owner_requests"]
                                        if r.get("id") != issue_id]
        self.store.save()

    def candidates(self) -> list[_issues.IssueRecord]:
        """Eligible issues, most useful first; blocked ones become requests."""
        out = []
        known = {r.get("id") for r in self.store["owner_requests"]}
        histories = self.store.issues()
        for issue in _issues.rank(self._issues(), self.settings, histories, self.now()):
            check = _classify.precheck(issue)
            if check.verdict == _classify.BLOCKED:
                if issue.id not in known:
                    self.store["owner_requests"] = [*self.store["owner_requests"],
                                                    _classify.owner_request(issue, check)]
                    self.store.issue(issue.id)["blocked_on_owner"] = check.need
                    self.store.save()
                continue
            out.append(issue)
        return out

    # ------------------------------------------------------------ scheduling
    def tick(self) -> str:
        """One scheduler step. Returns what it did (for logs and tests)."""
        self.fold_watch_results()
        self._expire_pending()
        pending = self.store["pending"]
        if pending and pending.get("approved"):
            return self._try_apply_pending()
        ok, why = self.can_work()
        if not ok:
            return why
        work = self.candidates()
        if not work:
            return "nada"
        self.start_attempt(work[0])
        return "started"

    def start(self) -> None:
        self.fold_watch_results()
        for row in self.store["announce"]:
            self.notify(str(row.get("text", "")))
        self.store["announce"] = []
        self.store.save()
        if self._thread is not None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(self.settings.poll_s):
                try:
                    self.tick()
                except Exception:
                    log.warning("selffix: tick failed", exc_info=True)

        self._thread = threading.Thread(target=loop, name="wawi-selffix", daemon=True)
        self._thread.start()
        used, cap = self.quota()
        log.info("selffix: %s, autonomy %s, %d/%d attempts today, %d candidate(s)",
                 "on" if self.enabled else "off", self.settings.autonomy, used, cap,
                 len(self.candidates()))

    def stop(self) -> None:
        self._stop.set()

    def fix_now(self, words: str = "") -> tuple[bool, str]:
        """The menu's "Arreglarme ahora" and "Wawi, arreglate lo de…"."""
        ok, why = self.can_work(manual=True)
        if not ok:
            return False, {
                "apagado": "Tengo el autoarreglo apagado.",
                "ocupada": "Ya me estoy arreglando algo; de a una.",
                "cupo": "Ya hice mis intentos de hoy; mañana sigo.",
                "pendiente": "Tengo un arreglo esperando que me digas si lo aplico.",
                "vigilando": "Todavía estoy vigilando mi último arreglo; después sigo.",
            }.get(why, "Ahora no puedo ponerme a arreglarme.")
        if words.strip():
            issue = _issues.match(self._issues(), words)
            if issue is None:
                return False, "No encuentro ese problema en lo que tengo anotado."
            verdict = _issues.eligibility(issue, self.settings, self.store.issues().get(issue.id),
                                          self.now())
            if not verdict.eligible and verdict.reason != "too_rare":
                return False, {
                    "backoff": "Eso ya lo intenté hace poco y falló; espero unos días.",
                    "wontfix": "Eso quedó marcado para no arreglar.",
                    "blocked_on_owner": "Para eso necesito algo tuyo primero.",
                    "already_fixed": "Eso ya me lo arreglé.",
                    "pending_owner": "Ya tengo un arreglo de eso esperando tu ok.",
                }.get(verdict.reason, "Eso no lo puedo encarar ahora.")
            check = _classify.precheck(issue)
            if check.verdict == _classify.BLOCKED:
                return False, check.request
        else:
            work = self.candidates()
            if not work:
                return False, "No tengo nada anotado para arreglarme."
            issue = work[0]
        self.start_attempt(issue)
        return True, f"Dale, me pongo a arreglarme lo de «{issue.title}»."

    def start_attempt(self, issue: _issues.IssueRecord, *, wait: bool = False) -> None:
        thread = threading.Thread(target=self.run_attempt, args=(issue,),
                                  name="wawi-selffix-attempt", daemon=True)
        thread.start()
        if wait:
            thread.join()

    # ------------------------------------------------------------ the attempt
    def _attempt_dir(self, attempt_id: str) -> Path:
        return self.settings.work_dir / attempt_id

    def _engine(self, preferred: str) -> str | None:
        for engine in (preferred, self.settings.fallback_engine):
            if engine and self.which(engine):
                return engine
        return None

    def run_attempt(self, issue: _issues.IssueRecord) -> dict:
        """The whole pipeline for one issue. Never raises."""
        if not self._busy.acquire(blocking=False):
            return {"outcome": "busy"}
        # Her little "…" while she works on herself (wawi/working.py).
        _working.set_musing("selffix", True)
        stamp = self.now()
        attempt = {
            "id": f"{stamp:%Y%m%d-%H%M%S}-{issue.id[:8]}-{uuid.uuid4().hex[:4]}",
            "issue": issue.id, "title": issue.title, "kind": issue.kind,
            "capability": issue.is_capability, "started": stamp.strftime(_TIME_FMT),
            "outcome": "running", "reasons": [], "gates": [],
            "count_at_start": issue.count, "rate_per_hour": round(issue.rate_per_hour(stamp), 4),
        }
        self.running = attempt
        self.store.add_attempt(attempt)
        hist = self.store.issue(issue.id)
        hist["attempts"] = int(hist.get("attempts", 0)) + 1
        hist["last_attempt"] = attempt["started"]
        hist["title"] = issue.title
        self.store.save()
        self.notify(f"Me estoy arreglando lo de «{_short(issue.title)}»…")
        workdir = self._attempt_dir(attempt["id"])
        worktree = workdir / "worktree"
        branch = f"{gitops.BRANCH_PREFIX}{attempt['id']}"
        attempt.update(worktree=str(worktree), branch=branch)
        keep_worktree = False
        try:
            keep_worktree = self._pipeline(issue, attempt, workdir, worktree, branch)
        except Exception as exc:
            log.warning("selffix: attempt %s crashed", attempt["id"], exc_info=True)
            self._fail(attempt, "error", f"{type(exc).__name__}: {str(exc)[:200]}")
        finally:
            if not keep_worktree:
                try:
                    gitops.remove_worktree(self.repo, worktree, branch, self.git)
                except Exception:
                    log.debug("selffix: worktree cleanup failed", exc_info=True)
            self.running = None
            self.store.save()
            _working.set_musing("selffix", False)
            self._busy.release()
        return attempt

    def _pipeline(self, issue: _issues.IssueRecord, attempt: dict, workdir: Path,
                  worktree: Path, branch: str) -> bool:
        """Returns True when the worktree must be kept (pending or applying)."""
        base = gitops._ok(self.git, self.repo, "rev-parse", self.main).strip()
        attempt["base"] = base
        workdir.mkdir(parents=True, exist_ok=True)
        gitops.create_worktree(self.repo, worktree, branch, base, self.git)

        # 1. Author.
        engine = self._engine(self.settings.engine)
        if engine is None:
            self._fail(attempt, "error", "no hay ni codex ni claude instalados")
            return False
        attempt["engine"] = engine
        hist = self.store.issue(issue.id)
        text = _brief.build_brief(
            issue,
            excerpt=_brief.log_excerpt(self.log_path, issue, self.settings.log_excerpt_lines),
            previous_failures=list(hist.get("reasons") or []),
            python=self.python, max_lines=self.settings.max_changed_lines,
            max_files=self.settings.max_files)
        (workdir / "brief.md").write_text(text, encoding="utf-8")
        last = workdir / "author.last.txt"
        result = self.author_executor.run(Job(
            name="author", argv=author_argv(engine, worktree, last, self.settings, self.python),
            cwd=worktree, timeout_s=self.settings.time_budget_s, stdin_text=text, dir=workdir,
            visible=self.settings.use_orca, title=f"Wawi · Autoarreglo · {_short(issue.title, 40)}"))
        if result.code == 127 and engine != self.settings.fallback_engine:
            fallback = self._engine(self.settings.fallback_engine)
            if fallback and fallback != engine:
                engine = attempt["engine"] = fallback
                result = self.author_executor.run(Job(
                    name="author", argv=author_argv(engine, worktree, last, self.settings, self.python),
                    cwd=worktree, timeout_s=self.settings.time_budget_s, stdin_text=text,
                    dir=workdir, visible=self.settings.use_orca,
                    title=f"Wawi · Autoarreglo · {_short(issue.title, 40)}"))
        report = final_message(engine, result, last)
        attempt["author_where"] = result.where
        if result.timed_out:
            self._fail(attempt, "rejected", f"se le acabó el tiempo ({self.settings.time_budget_s / 60:.0f} min)")
            return False
        subject, summary = _brief.parse_report(report)
        attempt["summary"] = summary[:600]
        sha = gitops.commit_all(worktree, subject or f"Autoarreglo: {issue.title}", self.git)
        if sha is None:
            self._fail(attempt, "rejected", "el agente no cambió nada"
                       + ("" if result.ok else f" (salió con código {result.code})"))
            return False

        # 2. Cheap gates.
        diff = gitops.diff(self.repo, base, branch, self.git)
        (workdir / "diff.patch").write_text(diff, encoding="utf-8")
        stats = gitops.numstat(self.repo, base, branch, self.git)
        attempt["files"] = [s.path for s in stats]
        deps = _gates.gate_dependencies(stats, diff, capability=issue.is_capability,
                                        pure_python=self.pure_python)
        for gate in (_gates.gate_forbidden(stats, diff),
                     _gates.gate_size(stats, self.settings.max_changed_lines, self.settings.max_files),
                     deps, _gates.gate_secrets(diff)):
            if not self._gate(attempt, gate):
                return False

        # 3. Tests, in the worktree.
        if not self._gate(attempt, self._run_tests(worktree, workdir)):
            return False

        # 4. Second opinion.
        reviewer = self._engine(self.settings.reviewer(engine)) or engine
        attempt["reviewer"] = reviewer
        prompt = _brief.build_review(issue, diff, report)
        rlast = workdir / "review.last.txt"
        review = self.local.run(Job(
            name="review", argv=reviewer_argv(reviewer, worktree, rlast, self.settings),
            cwd=worktree, timeout_s=self.settings.review_timeout_s, stdin_text=prompt, dir=workdir))
        approved, reasons = _brief.parse_verdict(final_message(reviewer, review, rlast))
        if review.timed_out:
            approved, reasons = False, "el revisor no terminó a tiempo"
        if not self._gate(attempt, _gates.gate_review(approved, reasons, reviewer)):
            return False

        # 5. Does it need him?
        new_deps = bool(deps.detail)
        verdict = _classify.classify_change(
            [s.path for s in stats], diff, agent_report=report, capability=issue.is_capability,
            new_deps=new_deps, autonomy=self.settings.autonomy)
        attempt["classification"] = verdict.verdict
        attempt["ask_reasons"] = verdict.reasons
        if verdict.verdict == _classify.AUTO:
            return self._apply(attempt)
        return self._offer(attempt, diff)

    def _gate(self, attempt: dict, gate: _gates.GateResult) -> bool:
        attempt["gates"].append({"name": gate.name, "ok": gate.ok, "reason": gate.reason,
                                 "detail": gate.detail[:10]})
        if not gate.ok:
            detail = f" ({'; '.join(gate.detail[:3])})" if gate.detail else ""
            self._fail(attempt, "rejected", f"{gate.reason}{detail}")
        return gate.ok

    def _run_tests(self, worktree: Path, workdir: Path) -> _gates.GateResult:
        argv = [self.python, *_brief.TEST_COMMAND.split()]
        result: JobResult = self.local.run(Job(
            name="tests", argv=argv, cwd=worktree, timeout_s=self.settings.test_timeout_s,
            dir=workdir))
        return _gates.gate_tests(result.code, result.timed_out, result.output)

    def _fail(self, attempt: dict, outcome: str, reason: str) -> None:
        attempt["outcome"] = outcome
        attempt["reasons"] = [*attempt.get("reasons", []), reason]
        attempt["finished"] = self.now().strftime(_TIME_FMT)
        hist = self.store.issue(attempt["issue"])
        hist["outcome"] = outcome
        hist["last_failed"] = attempt["finished"]
        hist["reasons"] = [*hist.get("reasons", []), reason][-8:]
        self.store.save()
        log.info("selffix: attempt %s %s: %s", attempt.get("id"), outcome, reason)
        if outcome in ("rejected", "error"):
            self.notify(f"No me salió el arreglo de «{_short(attempt['title'])}»; lo anoto.")

    # ------------------------------------------------------- ask the owner
    def _offer(self, attempt: dict, diff: str) -> bool:
        attempt["outcome"] = "pending_owner"
        attempt["finished"] = self.now().strftime(_TIME_FMT)
        self.store["pending"] = {
            "attempt": attempt["id"], "title": attempt["title"], "issue": attempt["issue"],
            "summary": attempt.get("summary", ""), "ask_reasons": attempt.get("ask_reasons", []),
            "created": attempt["finished"], "approved": False,
        }
        hist = self.store.issue(attempt["issue"])
        hist["pending"] = True
        self.store.save()
        self.notify(f"Me armé un arreglo para «{_short(attempt['title'])}», pero te toca a vos: "
                    "¿lo aplico? Te lo dejo en la pizarra.")
        self.show_pending(diff)
        return True

    def show_pending(self, diff: str | None = None) -> bool:
        pending = self.store["pending"]
        if not pending or self.board is None:
            return False
        if diff is None:
            try:
                diff = (self._attempt_dir(pending["attempt"]) / "diff.patch").read_text(encoding="utf-8")
            except OSError:
                diff = ""
        try:
            self.board.show(f"Arreglo: {_short(pending['title'], 50)}",
                            _report.pending_markdown(pending, diff), "markdown")
        except Exception:
            log.debug("selffix: board failed", exc_info=True)
            return False
        return True

    def decide_pending(self, apply: bool) -> tuple[bool, str]:
        """Aplicar / Descartar, from the menu or by voice."""
        pending = self.store["pending"]
        if not pending:
            return False, "No tengo ningún arreglo esperando."
        attempt = self.store.find_attempt(pending["attempt"]) or {}
        if not apply:
            self._discard_pending(attempt, "discarded")
            return True, "Listo, lo descarté."
        pending["approved"] = True
        self.store.save()
        what = self._try_apply_pending()
        if what == "applied":
            return True, "Dale, lo aplico y me reinicio."
        if what == "waiting_clean":
            return True, "Lo aplico en cuanto tu repo esté limpio y en main."
        if what == "no_restart":
            return True, "Lo aplico cuando me arranques con wawictl, así me puedo reiniciar sola."
        return False, "No pude aplicarlo; quedó anotado por qué."

    def _discard_pending(self, attempt: dict, outcome: str) -> None:
        pending = self.store["pending"] or {}
        if attempt:
            attempt["outcome"] = outcome
            attempt["finished"] = self.now().strftime(_TIME_FMT)
        hist = self.store.issue(pending.get("issue", attempt.get("issue", "")))
        hist.pop("pending", None)
        hist["outcome"] = outcome
        if outcome == "discarded":
            hist["wontfix"] = True    # he said no: do not offer it again
        self.store["pending"] = None
        self.store.save()
        worktree, branch = attempt.get("worktree"), attempt.get("branch")
        if worktree and branch:
            try:
                gitops.remove_worktree(self.repo, Path(worktree), branch, self.git)
            except Exception:
                log.debug("selffix: pending cleanup failed", exc_info=True)

    def _expire_pending(self) -> None:
        pending = self.store["pending"]
        if not pending or pending.get("approved"):
            return
        try:
            created = datetime.strptime(pending["created"], _TIME_FMT)
        except (KeyError, ValueError):
            return
        if (self.now() - created).total_seconds() > self.settings.pending_days * 86400:
            attempt = self.store.find_attempt(pending["attempt"]) or {}
            self._discard_pending(attempt, "expired")

    def _try_apply_pending(self) -> str:
        pending = self.store["pending"]
        attempt = self.store.find_attempt(pending["attempt"]) if pending else None
        if attempt is None:
            self.store["pending"] = None
            self.store.save()
            return "nada"
        if self.watching():
            return "vigilando"     # one self-applied change under watch at a time
        if not self._busy.acquire(blocking=False):
            return "ocupada"
        try:
            kept = self._apply(attempt)
            if attempt.get("outcome") in ("watching", *FAILED):
                self.store.issue(attempt["issue"]).pop("pending", None)
                self.store["pending"] = None
                if not kept:
                    gitops.remove_worktree(self.repo, Path(attempt["worktree"]),
                                           attempt["branch"], self.git)
            self.store.save()
            outcome = str(attempt.get("outcome", ""))
            if outcome == "waiting_clean" and attempt.get("wait_reason") == "no_restart":
                return "no_restart"
            return {"watching": "applied"}.get(outcome, outcome)
        finally:
            self._busy.release()

    # ------------------------------------------------------------- apply
    def _apply(self, attempt: dict) -> bool:
        """Squash into main and hand the restart to the watchdog.

        Returns True when the worktree must be kept (waiting for a clean tree
        or a restartable Wawi), False once it is merged or rejected."""
        why = gitops.can_apply(self.repo, self.main, self.git)
        restart = self.restart_cmd()
        if why or restart is None:
            attempt["outcome"] = "waiting_clean"
            attempt["wait_reason"] = why or "no_restart"
            pending = self.store["pending"] or {
                "attempt": attempt["id"], "title": attempt["title"], "issue": attempt["issue"],
                "summary": attempt.get("summary", ""), "ask_reasons": attempt.get("ask_reasons", []),
                "created": self.now().strftime(_TIME_FMT)}
            pending["approved"] = True
            self.store["pending"] = pending
            self.store.issue(attempt["issue"])["pending"] = True
            self.store.save()
            log.info("selffix: %s ready, waiting (%s)", attempt["id"], why or "no_restart")
            return True
        worktree, branch = Path(attempt["worktree"]), attempt["branch"]
        head = gitops.head(self.repo, self.git)
        if not gitops.is_ancestor(self.repo, head, branch, self.git):
            # main moved since the attempt started: rebase and test again.
            if not gitops.rebase(worktree, head, self.git):
                self._fail(attempt, "rejected", "no aplica sobre el main actual")
                return False
            if not self._gate(attempt, self._run_tests(worktree, self._attempt_dir(attempt["id"]))):
                return False
        self.store["last_good"] = head
        self.store.save()
        message = gitops.commit_message(attempt["title"], attempt["issue"], attempt.get("summary", ""))
        try:
            sha = gitops.squash_apply(self.repo, branch, message, main=self.main, git=self.git)
        except gitops.GitError as exc:
            self._fail(attempt, "rejected", f"no pude mergear: {exc}")
            return False
        attempt.update(outcome="watching", commit=sha, last_good=head,
                       finished=self.now().strftime(_TIME_FMT))
        hist = self.store.issue(attempt["issue"])
        hist.update(outcome="watching", commit=sha, count_at_apply=attempt.get("count_at_start", 0))
        self.store["announce"] = [*self.store["announce"], {
            "text": f"Me arreglé lo de «{_short(attempt['title'])}». Si algo anda raro, "
                    "decime «deshacé el último arreglo»."}]
        self.store.save()
        try:
            gitops.remove_worktree(self.repo, worktree, branch, self.git)
        except Exception:
            log.debug("selffix: worktree cleanup after merge failed", exc_info=True)
        self._launch_watchdog(attempt, restart)
        self.notify(f"Listo, me arreglé lo de «{_short(attempt['title'])}». Me reinicio.")
        return False

    def _launch_watchdog(self, attempt: dict, restart: list[str]) -> None:
        workdir = self._attempt_dir(attempt["id"])
        workdir.mkdir(parents=True, exist_ok=True)
        frozen = workdir / "watchdog.py"
        shutil.copyfile(_watchdog.__file__, frozen)
        job = {
            "repo": str(self.repo), "branch": self.main, "commit": attempt["commit"],
            "issue": attempt["issue"], "attempt": attempt["id"],
            "fingerprint": attempt["issue"] if not attempt.get("capability") else "",
            "baseline_per_hour": attempt.get("rate_per_hour", 0.0),
            "log_path": str(self.log_path or _paths.under(self.repo) / "wawi.log"),
            "pid_file": str(self.pid_file), "wawictl": restart[0],
            "startup_s": self.settings.watchdog_startup_s,
            "watch_s": self.settings.watchdog_watch_s, "poll_s": 15.0,
            "result_path": str(workdir / "watch_result.json"), "restart": True,
        }
        spec = workdir / "watch_job.json"
        spec.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        attempt["watch_result"] = job["result_path"]
        self.store.save()
        self.spawn([sys.executable, str(frozen), str(spec)])

    def _watch_overdue(self, attempt: dict) -> bool:
        try:
            done = datetime.strptime(str(attempt.get("finished", "")), _TIME_FMT)
        except ValueError:
            return False
        limit = self.settings.watchdog_startup_s + self.settings.watchdog_watch_s + 3600
        return (self.now() - done).total_seconds() > limit

    def fold_watch_results(self) -> None:
        """Bring the watchdog's verdicts into the history (and tell him)."""
        changed = False
        for attempt in self.store.attempts():
            if attempt.get("outcome") != "watching" or not attempt.get("watch_result"):
                continue
            try:
                result = json.loads(Path(attempt["watch_result"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                if self._watch_overdue(attempt):
                    # The watchdog never reported (a reboot, a kill): it is in
                    # main and nobody saw it fail; stop blocking new work.
                    attempt["outcome"] = self.store.issue(attempt["issue"])["outcome"] = "applied"
                    attempt["reasons"] = [*attempt.get("reasons", []), "el vigilante no informó"]
                    changed = True
                continue
            outcome = str(result.get("outcome", ""))
            reason = str(result.get("reason", ""))
            hist = self.store.issue(attempt["issue"])
            if outcome in ("healthy", "inconclusive"):
                attempt["outcome"] = hist["outcome"] = "applied"
            elif outcome in ("rolled_back", "revert_blocked"):
                attempt["outcome"] = outcome
                attempt["reasons"] = [*attempt.get("reasons", []), reason,
                                      *result.get("evidence", [])[:3]]
                attempt["revert"] = result.get("revert", "")
                hist["outcome"] = outcome
                hist["last_failed"] = self.now().strftime(_TIME_FMT)
                hist["reasons"] = [*hist.get("reasons", []), f"se deshizo tras aplicarlo: {reason}"][-8:]
                self.notify(f"Deshice mi arreglo de «{_short(attempt['title'])}»: {reason}."
                            if outcome == "rolled_back" else
                            f"Mi arreglo de «{_short(attempt['title'])}» falló y no pude "
                            "deshacerlo sola; miralo cuando puedas.")
            else:
                continue
            attempt["watched"] = result.get("at", "")
            changed = True
        if changed:
            self.store.save()

    # ------------------------------------------------------------- undo
    def undo_last(self) -> tuple[bool, str]:
        """Revert the newest self-fix commit still in main, then restart."""
        try:
            commits = gitops.selffix_commits(self.repo, git=self.git)
        except gitops.GitError:
            return False, "No pude leer el historial del repo."
        if not commits:
            return False, "No tengo ningún arreglo mío para deshacer."
        last = commits[0]
        ok, detail = _watchdog.revert_selffix(str(self.repo), last.sha, "me lo pediste",
                                              branch=self.main, git=self.git)
        if not ok:
            return False, {
                "dirty_tree": "Tenés cambios sin commitear; no toco nada hasta que estén guardados.",
                "not_on_main": "El repo no está en main; no toco nada.",
            }.get(detail, "No pude deshacerlo; quedó todo como estaba.")
        for attempt in self.store.attempts():
            if attempt.get("commit") == last.sha:
                attempt["outcome"] = "undone"
                attempt["revert"] = detail
                hist = self.store.issue(attempt["issue"])
                hist["outcome"] = "undone"
                hist["last_failed"] = self.now().strftime(_TIME_FMT)
                hist["reasons"] = [*hist.get("reasons", []), "el dueño lo deshizo"][-8:]
        self.store.save()
        restart = self.restart_cmd()
        if restart is not None:
            self.spawn(restart)
            return True, "Listo, deshice mi último arreglo y me reinicio."
        return True, "Listo, deshice mi último arreglo; reiniciame para que se note."

    # ------------------------------------------------------------- reporting
    def today_markdown(self) -> str:
        return _report.today_markdown(self.store.attempts(), self.owner_requests(),
                                      self.now().strftime("%Y-%m-%d"))

    def history_markdown(self) -> str:
        return _report.history_markdown(self.store.attempts(), self.store["pending"],
                                        self.owner_requests(), self.enabled, self.quota())

    def show_history(self) -> bool:
        if self.board is None:
            return False
        try:
            self.board.show("Autoarreglo", self.history_markdown(), "markdown")
        except Exception:
            log.debug("selffix: board failed", exc_info=True)
            return False
        return True

    def spoken_report(self) -> str:
        day = self.now().strftime("%Y-%m-%d")
        rows = [a for a in self.store.attempts() if str(a.get("started", "")).startswith(day)]
        if not rows:
            pending = self.store["pending"]
            if pending:
                return f"Hoy no apliqué nada; tengo listo «{_short(pending['title'])}» esperando tu ok."
            return "Hoy no me arreglé nada."
        return " ".join(_report.attempt_line(a) for a in rows[-3:])


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
