"""Wawi sends coding tasks to Orca, and tells you how they went.

While you talk, you and Wawi come up with a feature or a fix. "Mandalo a
Orca" turns it into a task for an Orca coding agent that works in a worktree
of its own, while Wawi keeps an ear on it and tells you, as the pet, when it
finishes, asks something or fails.

Orca (the multi-agent desktop app) is driven only through its public CLI, the
same way a person would from a terminal. Five rules shape this module:

1. **Wawi owns one idle terminal and one Run.** Orchestration commands need a
   sender terminal, so Wawi creates its own ("Wawi"), binds one Run to it and
   keeps both in a gitignored ``private/orca_local.json`` next to ``config.yaml``,
   with the tasks still in flight. A terminal that is gone is created again
   and the same Run is re-bound to it (``run-use``).
2. **Never someone else's mailbox.** Wawi may itself run inside an Orca
   terminal, and an inherited ``ORCA_TERMINAL_HANDLE`` would make every
   command act for THAT terminal. Every ``ORCA_*`` variable is stripped from
   the child's environment and ``--from`` / ``--terminal`` are always explicit.
3. **A paid agent starts only after a spoken yes.** ``send_to_orca`` only
   drafts; the yes is read from the owner's NEXT transcript (:func:`hear`),
   never from anything the model says, the same rule as the connectors' sends.
4. **The orchestration safety floor.** A failed start is reported, never
   relaunched. A timeout or an empty wait is a checkpoint, never a reason to
   stop, retry or release. A worker is released only after its own
   ``worker_done``; a batch is acknowledged only after every message in it
   was handled.
5. **Never fatal.** No CLI, no app running, a broken answer: the tools say
   so, the watcher backs off, and Wawi goes on being Wawi.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from wawi import paths as _paths

log = logging.getLogger(__name__)

#: Where the app ships its CLI, the last place looked.
APP_CLI = "/Applications/Orca.app/Contents/Resources/bin/orca"

#: The repository Wawi lives in: the default target of a task.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    # Off = no tools, no watcher. It also turns itself off when there is no CLI.
    "enabled": True,
    # Path to the orca CLI. "" = PATH, then the app bundle.
    "cli": "",
    # Orca agent id that does the work (claude, codex, opencode…).
    "agent": "claude",
    # Repository a task goes to when the owner names none. "" = Wawi's own.
    "default_repo": "",
    # new-top-level (a fresh worktree per task, so parallel tasks never
    # collide), new-child, current, or an exact selector such as path:/x.
    "worktree": "new-top-level",
    # Wawi's terminal, Run and tasks in flight (gitignored).
    "local_file": "orca_local.json",
    # Each sent spec is also written here as <slug>.md (gitignored).
    "spec_dir": "issues",
    # One `check --wait` lasts at most this long; a timeout is a checkpoint.
    "wait_timeout_s": 300,
    # Voice handoff to a visible, unsupervised terminal.
    "terminal_handoff_enabled": True,
}

#: The run every voice task goes into.
RUN_OBJECTIVE = "Wawi: tareas mandadas por voz"

#: What a draft needs to be a real spec (Orca's task-spec contract).
SPEC_SECTIONS = ("target", "change", "constraints", "ownership", "acceptance")

#: A spoken yes only counts this long after the question.
CONFIRM_TIMEOUT_S = 180.0

#: Longest answer still read as a bare yes/no. "Sí, pero cambiale el título"
#: is an edit, not a yes, and goes to the model instead.
_MAX_CONFIRM_WORDS = 8

#: How long a worker gets to come up (worker-start waits for `ready`).
_START_TIMEOUT_S = 600.0
_CALL_TIMEOUT_S = 60.0
_STATUS_TTL_S = 30.0
_BACKOFF_FIRST_S = 5.0
_BACKOFF_MAX_S = 300.0

#: What Wawi wants to hear about while she waits.
WAIT_TYPES = "worker_done,escalation,question"

#: Repo conventions appended to every spec for Wawi's own repository.
WAWI_CONVENTIONS = (
    "Convenciones del repo (obligatorias):\n"
    "- Commits en español, y cada mensaje termina con la línea "
    "\"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\".\n"
    "- Tests: .venv/bin/python -m pytest -q --deselect "
    "tests/test_listener.py::TestLifecycle::test_stop_does_not_hang_with_a_full_queue\n"
    "- No toques config.yaml, private/ ni ningún *_local.json.\n"
    "- No hagas push.\n"
    "- Commiteá en tu rama y NO hagas merge: el merge lo hace el dueño."
)

#: The same promises, for any other repository.
GENERIC_CONVENTIONS = (
    "Convenciones:\n"
    "- Commits en español, y cada mensaje termina con la línea "
    "\"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\".\n"
    "- Corré los tests del repo antes de terminar.\n"
    "- No hagas push.\n"
    "- Commiteá en tu rama y NO hagas merge: el merge lo hace el dueño."
)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OrcaSettings:
    enabled: bool = True
    cli: str = ""
    agent: str = "claude"
    default_repo: Path = REPO_ROOT
    worktree: str = "new-top-level"
    local_file: Path | None = field(default_factory=lambda: _paths.private("orca_local.json"))
    spec_dir: Path | None = field(default_factory=lambda: _paths.private("issues"))
    wait_timeout_s: float = 300.0
    terminal_handoff_enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "OrcaSettings":
        d = {**DEFAULTS, **(raw or {})}
        repo = str(d.get("default_repo") or "").strip()
        local = str(d.get("local_file") or "").strip()
        spec_dir = str(d.get("spec_dir") or "").strip()
        try:
            wait = float(d.get("wait_timeout_s", 300))
        except (TypeError, ValueError):
            wait = 300.0
        return cls(
            enabled=bool(d.get("enabled", True)),
            cli=str(d.get("cli") or "").strip(),
            agent=str(d.get("agent") or "claude").strip() or "claude",
            default_repo=Path(repo).expanduser() if repo else REPO_ROOT,
            worktree=str(d.get("worktree") or "new-top-level").strip() or "new-top-level",
            local_file=_paths.resolve(local) if local else None,
            spec_dir=_paths.resolve(spec_dir) if spec_dir else None,
            wait_timeout_s=max(5.0, min(wait, 3600.0)),
            terminal_handoff_enabled=bool(d.get("terminal_handoff_enabled", True)),
        )

    @property
    def creates_worktree(self) -> bool:
        return self.worktree in ("new-top-level", "new-child")


def resolve_cli(
    configured: str = "",
    *,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.exists,
) -> str | None:
    """The orca executable: config, then PATH, then the app. None = no Orca."""
    if configured:
        path = os.path.expanduser(configured)
        if exists(path):
            return path
        found = which(configured)
        if found:
            return found
        log.warning("orca: orca.cli=%r does not exist", configured)
        return None
    found = which("orca")
    if found:
        return found
    if exists(APP_CLI):
        return APP_CLI
    return None


def clean_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """*env* without any ``ORCA_*`` variable (rule 2 above)."""
    source = os.environ if env is None else env
    return {k: v for k, v in source.items() if not k.startswith("ORCA_")}


def slugify(text: str, limit: int = 40) -> str:
    """ASCII, lowercase, dashes: «Botón de silencio» -> boton-de-silencio."""
    plain = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
    if len(slug) > limit:
        slug = slug[:limit].rstrip("-")
    return slug or "tarea"


def missing_sections(spec: str) -> list[str]:
    """Which of Target/Change/Constraints/Ownership/Acceptance the spec lacks."""
    low = (spec or "").lower()
    return [name for name in SPEC_SECTIONS if name not in low]


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


class OrcaError(Exception):
    """A CLI call that did not succeed. *payload* is the parsed JSON, if any."""

    def __init__(self, code: str, message: str = "", payload: dict | None = None) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message
        self.payload = payload or {}


#: runner(argv, env, timeout) -> (returncode, stdout, stderr)
Runner = Callable[[list[str], dict[str, str], float], tuple[int, str, str]]


class SubprocessRunner:
    """The real runner. Remembers live children so :meth:`kill_all` can end
    a ten-minute ``check --wait`` when Wawi quits."""

    def __init__(self) -> None:
        self._live: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    def __call__(self, argv: list[str], env: dict[str, str], timeout: float) -> tuple[int, str, str]:
        proc = subprocess.Popen(
            argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        with self._lock:
            self._live.add(proc)
        try:
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                raise OrcaError("timeout", f"{argv[1:3]} took over {timeout:.0f}s")
            return proc.returncode, out or "", err or ""
        finally:
            with self._lock:
                self._live.discard(proc)

    def kill_all(self) -> None:
        with self._lock:
            live = list(self._live)
        for proc in live:
            try:
                proc.kill()
            except Exception:
                pass


def _parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class OrcaCLI:
    """``orca <args> --json``, in a clean environment, as a dict."""

    def __init__(self, cli: str, runner: Runner | None = None) -> None:
        self.cli = cli
        self.runner: Runner = runner or SubprocessRunner()

    def run(self, *args: str, timeout: float = _CALL_TIMEOUT_S) -> dict:
        argv = [self.cli, *args]
        if "--json" not in args:
            argv.append("--json")
        try:
            code, out, err = self.runner(argv, clean_env(), timeout)
        except OrcaError:
            raise
        except Exception as exc:
            raise OrcaError("spawn_failed", str(exc)[:200]) from exc
        payload = _parse_json(out)
        if payload is None:
            raise OrcaError("bad_output", (err or out or "")[:200])
        if code != 0 or payload.get("ok") is False:
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            raise OrcaError(
                str(error.get("code") or f"exit_{code}"),
                str(error.get("message") or "")[:300],
                payload,
            )
        result = payload.get("result")
        return result if isinstance(result, dict) else {}

    def kill_all(self) -> None:
        kill = getattr(self.runner, "kill_all", None)
        if callable(kill):
            kill()


def find_key(value: Any, key: str) -> Any:
    """The first value under *key* anywhere in nested dicts/lists, or None.

    Orca's receipts are versioned and optional fields come and go; looking a
    field up by name wherever it sits is sturdier than a fixed path."""
    if isinstance(value, dict):
        if key in value and value[key] not in (None, ""):
            return value[key]
        for item in value.values():
            found = find_key(item, key)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_key(item, key)
            if found not in (None, ""):
                return found
    return None


# --------------------------------------------------------------------------- #
# Local state (orca_local.json)
# --------------------------------------------------------------------------- #


class LocalState:
    """``{"terminal", "run", "dispatches": {id: {...}}, "questions": [...]}``.

    Written atomically on every change; a missing or broken file is an empty
    state, never an error.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.terminal: str = ""
        self.run: str = ""
        self.dispatches: dict[str, dict[str, Any]] = {}
        self.questions: list[dict[str, Any]] = []
        self.opened_terminals: list[dict[str, str]] = []
        self.load()

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            log.warning("orca: %s is unreadable; starting empty", self.path)
            return
        if not isinstance(data, dict):
            return
        self.terminal = str(data.get("terminal") or "")
        self.run = str(data.get("run") or "")
        raw = data.get("dispatches")
        self.dispatches = {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)} \
            if isinstance(raw, dict) else {}
        qs = data.get("questions")
        self.questions = [dict(q) for q in qs if isinstance(q, dict) and q.get("id")] \
            if isinstance(qs, list) else []
        opened = data.get("opened_terminals")
        self.opened_terminals = [
            {key: str(row.get(key) or "") for key in ("handle", "title", "repo")}
            for row in opened if isinstance(row, dict) and row.get("handle")
        ] if isinstance(opened, list) else []
        _show_busy(bool(self.dispatches))

    def save(self) -> None:
        _show_busy(bool(self.dispatches))
        if self.path is None:
            return
        data = {
            "version": 1,
            "terminal": self.terminal,
            "run": self.run,
            "dispatches": self.dispatches,
            "questions": self.questions,
            "opened_terminals": self.opened_terminals,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            os.replace(tmp, self.path)
        except OSError:
            log.warning("orca: could not write %s", self.path, exc_info=True)


# --------------------------------------------------------------------------- #
# Announcements
# --------------------------------------------------------------------------- #

#: kind -> (behaviour, priority). Clips degrade gracefully (wawi/pet/acting).
_NOTICE_LOOK = {
    "done": ("celebrate", 2),
    "failed": ("sad", 2),
    "question": ("think_bubble", 2),
    "info": (None, 1),
}


def bus_notify(kind: str, text: str) -> None:
    """Say *text* as the pet: bubble + voice through the reaction bus, which
    respects quiet mode, the hourly budget and the echo guard (petvoice)."""
    from wawi.reactions import BUS, Reaction

    behaviour, priority = _NOTICE_LOOK.get(kind, (None, 1))
    BUS.push(Reaction(behaviour=behaviour, bubble=text, speak=True, priority=priority))


def _first_sentence(text: str, limit: int = 140) -> str:
    text = " ".join(str(text or "").split())
    for stop in (". ", "! ", "? "):
        cut = text.find(stop)
        if 0 < cut < limit:
            return text[: cut + 1]
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _clean_branch(branch: Any) -> str:
    return str(branch or "").removeprefix("refs/heads/").strip()


# --------------------------------------------------------------------------- #
# The bridge
# --------------------------------------------------------------------------- #


@dataclass
class Draft:
    title: str
    spec: str
    repo_selector: str
    repo_path: str
    repo_name: str
    agent: str
    default_repo: bool


@dataclass
class _Pending:
    """Something that runs only after the owner's next words are a yes."""

    kind: str  # "send" | "cancel"
    at: float
    action: Callable[[], None]
    yes_reply: str
    no_reply: str
    detail: dict = field(default_factory=dict)
    confirmation_id: str | None = None


class OrcaBridge:
    def __init__(
        self,
        settings: OrcaSettings,
        cli: OrcaCLI,
        *,
        notify: Callable[[str, str], None] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], bool] | None = None,
        background: bool = True,
    ) -> None:
        self.settings = settings
        self.cli = cli
        self.state = LocalState(settings.local_file)
        self._notify = notify or bus_notify
        self._clock = clock
        self._stop = threading.Event()
        self._sleep = sleep or self._stop.wait
        self._background = background
        self._lock = threading.RLock()
        self._pending: _Pending | None = None
        self._watcher: threading.Thread | None = None
        self._status_at = -1e9
        self._status_ok = False
        self._terminal_checked = False

    def _set_pending(self, pending: _Pending, title: str, detail: str) -> None:
        from wawi.confirm import REGISTRY
        with self._lock:
            if self._pending and self._pending.confirmation_id:
                REGISTRY.discard(self._pending.confirmation_id)
            self._pending = pending
            def clear() -> None:
                with self._lock:
                    if self._pending is pending:
                        self._pending = None
            def accepted() -> None:
                clear()
                self._spawn(pending.action, f"wawi-orca-{pending.kind}")
            pending.confirmation_id = REGISTRY.add(title, detail, "orca", accepted, clear)

    # -- reachability ------------------------------------------------------
    def available(self, *, fresh: bool = False) -> bool:
        """True when the Orca runtime answers (``status`` says reachable)."""
        now = time.monotonic()
        if not fresh and now - self._status_at < _STATUS_TTL_S:
            return self._status_ok
        try:
            result = self.cli.run("status", timeout=15.0)
            ok = bool(find_key(result.get("runtime") or {}, "reachable"))
        except OrcaError as exc:
            log.info("orca: not reachable (%s)", exc.code)
            ok = False
        self._status_at, self._status_ok = now, ok
        return ok

    # -- terminal and run ----------------------------------------------------
    def _terminal_alive(self, handle: str) -> bool:
        try:
            result = self.cli.run("terminal", "show", "--terminal", handle)
        except OrcaError as exc:
            log.info("orca: terminal %s is gone (%s)", handle, exc.code)
            return False
        term = result.get("terminal") or {}
        return term.get("connected") is not False and not term.get("exitCause")

    def ensure_terminal(self, *, recheck: bool = False) -> str:
        """Wawi's own idle terminal, created (or re-created) when needed."""
        with self._lock:
            handle = self.state.terminal
            if handle and (self._terminal_checked and not recheck):
                return handle
            if handle and self._terminal_alive(handle):
                self._terminal_checked = True
                return handle
            result = self.cli.run(
                "terminal", "create",
                "--worktree", f"path:{self.settings.default_repo}",
                "--title", "Wawi",
            )
            new = str(find_key(result, "handle") or "")
            if not new:
                raise OrcaError("no_handle", "terminal create returned no handle")
            log.info("orca: created Wawi's terminal %s", new)
            self.state.terminal = new
            self._terminal_checked = True
            if self.state.run:
                # The Run is bound to the terminal that made it: re-bind it so
                # the tasks in flight still report to Wawi.
                try:
                    self.cli.run("orchestration", "run-use", "--id", self.state.run, "--from", new)
                except OrcaError as exc:
                    log.warning("orca: could not re-bind run %s (%s); a new one will be made",
                                self.state.run, exc.code)
                    self.state.run = ""
            self.state.save()
            return new

    def ensure_run(self) -> tuple[str, str]:
        with self._lock:
            handle = self.ensure_terminal()
            if self.state.run:
                return handle, self.state.run
            result = self.cli.run(
                "orchestration", "run-create", "--from", handle, "--objective", RUN_OBJECTIVE,
            )
            run = str((result.get("run") or {}).get("id") or find_key(result, "id") or "")
            if not run:
                raise OrcaError("no_run", "run-create returned no id")
            log.info("orca: created run %s", run)
            self.state.run = run
            self.state.save()
            return handle, run

    # -- repositories ----------------------------------------------------------
    def resolve_repo(self, name: str | None) -> tuple[str, str, str, bool] | None:
        """(selector, path, display name, is_default) for a spoken repo name.

        None when the owner named a repo Orca does not know."""
        default = self.settings.default_repo
        wanted = (name or "").strip().lower()
        if not wanted or wanted in (default.name.lower(), "wawi", "este", "este repo", "acá"):
            return f"path:{default}", str(default), default.name, True
        try:
            repos = self.cli.run("repo", "list").get("repos") or []
        except OrcaError as exc:
            log.info("orca: repo list failed (%s)", exc.code)
            repos = []
        for repo in repos:
            if not isinstance(repo, dict) or not repo.get("path"):
                continue
            names = {str(repo.get("displayName") or "").lower(), Path(str(repo["path"])).name.lower()}
            if wanted in names:
                path = str(repo["path"])
                return f"path:{path}", path, str(repo.get("displayName") or Path(path).name), \
                    Path(path) == default
        return None

    def known_repos(self) -> list[str]:
        try:
            repos = self.cli.run("repo", "list").get("repos") or []
        except OrcaError:
            return []
        return [str(r.get("displayName") or "") for r in repos if isinstance(r, dict)]

    # -- visible terminals (unsupervised handoff, never a Run/Task) ---------
    def open_terminal(self, kind: str, repo: str = "", prompt: str = "") -> tuple[bool, str, dict]:
        kind = (kind or "").strip().lower()
        prompt = (prompt or "").strip()
        if not self.settings.terminal_handoff_enabled:
            return False, "La apertura de terminales está desactivada.", {"error": "disabled"}
        if kind not in ("claude", "codex", "shell"):
            return False, "Solo puedo abrir Claude, Codex o una shell.", {"error": "invalid_kind"}
        if not self.available():
            return False, "Orca no está abierta.", {"error": "orca_down"}
        resolved = self.resolve_repo(repo)
        if resolved is None:
            return False, "No conozco ese repo en Orca.", {"error": "unknown_repo", "repos": self.known_repos()}
        selector, _, name, _ = resolved
        # A prompt is terminal input even when the terminal is a free shell.
        if kind != "shell" or prompt:
            self._set_pending(_Pending(
                "open_terminal", self._clock(),
                lambda: self._open_terminal(kind, selector, name, prompt),
                "Dale, la abro en Orca.", "Bueno, no la abro.",
                {"kind": kind, "repo": name}),
                "Abrir terminal de Orca", f"{kind.title()} en {name}")
            return True, f"¿Abro una terminal {kind} en {name}?", {
                "sent": False, "note": "Esperá el sí del dueño; no llames otra herramienta."}
        return self._create_terminal(kind, selector, name, prompt)

    def _open_terminal(self, kind: str, selector: str, name: str, prompt: str) -> None:
        ok, speak, _ = self._create_terminal(kind, selector, name, prompt)
        self._notify("info" if ok else "failed", speak)

    def _create_terminal(self, kind: str, selector: str, name: str,
                         prompt: str) -> tuple[bool, str, dict]:
        title = f"Wawi · {kind.title()} · {name}"
        args = ["terminal", "create", "--worktree", selector, "--title", title, "--focus"]
        if kind != "shell":
            args += ["--command", kind]
        try:
            result = self.cli.run(*args)
            handle = str(find_key(result, "handle") or "")
            if not handle:
                raise OrcaError("no_handle", "terminal create returned no handle")
            with self._lock:
                self.state.opened_terminals.append({"handle": handle, "title": title, "repo": name})
                self.state.save()
            if prompt:
                if kind != "shell":
                    ready = self.cli.run("terminal", "wait", "--terminal", handle,
                                         "--for", "tui-idle", "--timeout-ms", "60000",
                                         timeout=70.0)
                    if find_key(ready, "satisfied") is not True:
                        return False, "Abrí el agente, pero todavía no está listo para recibir texto.", {
                            "error": "not_ready", "handle": handle}
                # A failed send is not retried: the terminal remains open and named.
                self.cli.run("terminal", "send", "--terminal", handle,
                             "--text", prompt, "--enter")
        except OrcaError as exc:
            return False, "No pude abrir o escribir en la terminal de Orca.", {"error": exc.code}
        return True, f"Abrí {kind} en {name}.", {"handle": handle, "title": title, "repo": name}

    def list_terminals(self) -> tuple[bool, str, dict]:
        with self._lock:
            rows = list(self.state.opened_terminals)
        if not rows:
            return True, "No abrí ninguna terminal en Orca.", {"terminals": []}
        names = "; ".join(f"{row['title']} ({row['handle']})" for row in rows)
        return True, f"Abrí estas terminales: {names}.", {"terminals": rows}

    def _find_opened(self, which: str) -> tuple[dict | None, str | None]:
        wanted = (which or "").strip().lower()
        with self._lock:
            rows = list(self.state.opened_terminals)
        if not wanted and len(rows) == 1:
            return rows[0], None
        matches = [r for r in rows if wanted and (
            r["handle"].lower() == wanted or wanted in r["title"].lower())]
        if len(matches) == 1:
            return matches[0], None
        return None, "ambiguous" if matches or rows else "nothing_opened"

    def terminal_read(self, which: str = "") -> tuple[bool, str, dict]:
        row, error = self._find_opened(which)
        if row is None:
            return False, "¿Cuál terminal? Decime el nombre o identificador.", {"error": error}
        try:
            result = self.cli.run("terminal", "read", "--terminal", row["handle"],
                                  "--screen", "--limit", "30")
        except OrcaError as exc:
            return False, "No pude leer esa terminal.", {"error": exc.code}
        terminal = result.get("terminal") or {}
        tail = terminal.get("tail") or []
        lines = [str(line).strip() for line in tail if str(line).strip()]
        excerpt = " ".join(lines[-4:]) if lines else "no muestra nada todavía"
        return True, f"{row['title']}: {_first_sentence(excerpt, 180)}", {
            "handle": row["handle"], "tail": lines[-30:], "source": terminal.get("source")}

    def terminal_send(self, which: str, text: str) -> tuple[bool, str, dict]:
        row, error = self._find_opened(which)
        if row is None:
            return False, "¿A cuál terminal se lo mando?", {"error": error}
        text = (text or "").strip()
        if not text:
            return False, "No escuché qué querés escribir.", {"error": "empty"}
        self._set_pending(_Pending(
            "terminal_send", self._clock(),
            lambda: self._send_terminal(row["handle"], text),
            "Dale, se lo escribo.", "Bueno, no escribo nada.",
            {"handle": row["handle"]}),
            "Enviar texto a terminal", f"{row['title']}: {text[:55]}")
        return True, f"¿Le mando ese texto a {row['title']}?", {
            "sent": False, "note": "Esperá el sí del dueño; no llames otra herramienta."}

    def _send_terminal(self, handle: str, text: str) -> None:
        try:
            self.cli.run("terminal", "send", "--terminal", handle,
                         "--text", text, "--enter")
        except OrcaError:
            self._notify("failed", "No pude escribir en esa terminal de Orca.")
            return
        self._notify("info", "Ya se lo escribí a la terminal.")

    # -- drafting and the spoken yes ---------------------------------------------
    def draft(self, title: str, spec: str, repo: str | None = None,
              agent: str | None = None) -> tuple[bool, str, dict]:
        """Prepare a task and ask. Nothing reaches Orca until :meth:`hear`."""
        title = " ".join((title or "").split())[:100]
        spec = (spec or "").strip()
        if not title or not spec:
            return False, "Decime qué tarea querés mandar.", {"error": "empty"}
        lacking = missing_sections(spec)
        if lacking:
            return False, "Me falta completar la tarea antes de mandarla.", {
                "error": "incomplete_spec",
                "missing": lacking,
                "hint": "Reescribí el spec con secciones Target:, Change:, "
                        "Constraints:, Ownership: y Acceptance: y volvé a llamar.",
            }
        if not self.available():
            return False, "Orca no está abierta, no puedo mandarle nada.", {"error": "orca_down"}
        resolved = self.resolve_repo(repo)
        if resolved is None:
            names = ", ".join(n for n in self.known_repos() if n) or "ninguno"
            return False, f"No conozco ese repo en Orca. Tengo: {names}.", {"error": "unknown_repo"}
        selector, path, name, is_default = resolved
        draft = Draft(title, spec, selector, path, name,
                      (agent or self.settings.agent).strip() or self.settings.agent, is_default)
        question = f"Tarea «{title}» para {name}. ¿La mando a Orca?"
        self._set_pending(_Pending(
            "send", self._clock(), lambda: self._launch(draft),
            "Dale, se la mando a Orca.", "Bueno, no la mando.",
            {"title": title, "repo": name}),
            "Enviar tarea a Orca", f"{title} · {name}")
        log.info("orca: draft %r for %s waits for a spoken yes", title, name)
        return True, question, {
            "sent": False,
            "title": title,
            "repo": name,
            "agent": draft.agent,
            "note": "NO se mandó todavía. Decí la pregunta tal cual y esperá: "
                    "el sí del dueño lo lee Wawi sola. No llames otra herramienta.",
        }

    def hear(self, text: str) -> str | None:
        """The owner's next words after a draft. The reply to say, or None
        when they were not a plain yes/no (the brain gets them instead)."""
        from wawi.confirm import REGISTRY
        with self._lock:
            pending = self._pending
        if pending is None:
            return None
        if self._clock() - pending.at > CONFIRM_TIMEOUT_S:
            REGISTRY.discard(pending.confirmation_id or "")
            with self._lock:
                if self._pending is pending:
                    self._pending = None
            log.info("orca: %s draft expired unanswered", pending.kind)
            return None
        from wawi import lang
        words = [w for w in (text or "").lower().split() if w]
        answer = lang.affirmative(text)
        if answer is True and len(words) <= _MAX_CONFIRM_WORDS and "pero" not in words:
            if REGISTRY.resolve(pending.confirmation_id or "", True):
                log.info("orca: %s confirmed by voice", pending.kind)
                return pending.yes_reply
        if answer is False and len(words) <= 4:
            if REGISTRY.resolve(pending.confirmation_id or "", False):
                log.info("orca: %s declined by voice", pending.kind)
                return pending.no_reply
        log.info("orca: %s ignored non-confirmation speech", pending.kind)
        return None

    def pending_kind(self) -> str | None:
        with self._lock:
            return self._pending.kind if self._pending else None

    def _spawn(self, fn: Callable[[], None], name: str) -> None:
        if not self._background:
            fn()
            return
        threading.Thread(target=fn, name=name, daemon=True).start()

    # -- launching -----------------------------------------------------------------
    def _unique_slug(self, title: str) -> str:
        base = slugify(title)
        taken = {str(d.get("slug") or "") for d in self.state.dispatches.values()}
        slug, n = base, 2
        while slug in taken:
            slug, n = f"{base}-{n}", n + 1
        return slug

    def full_spec(self, draft: Draft) -> str:
        conventions = WAWI_CONVENTIONS if draft.default_repo else GENERIC_CONVENTIONS
        return f"{draft.spec.rstrip()}\n\n{conventions}\n"

    def _record(self, slug: str, draft: Draft, spec: str, outcome: str) -> None:
        directory = self.settings.spec_dir
        if directory is None:
            return
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{slug}.md"
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(self._clock()))
            path.write_text(
                f"# {draft.title}\n\n"
                f"- Mandada a Orca: {stamp}\n- Repo: {draft.repo_name} ({draft.repo_path})\n"
                f"- Agente: {draft.agent}\n- Resultado: {outcome}\n\n## Spec\n\n{spec}"
            )
        except OSError:
            log.warning("orca: could not write the spec record", exc_info=True)

    def launch(self, draft: Draft) -> tuple[bool, str, dict]:
        """Start the worker. Exits with one attempt: a failure is never retried."""
        try:
            handle, run = self.ensure_run()
        except OrcaError as exc:
            return False, "No pude preparar mi terminal en Orca.", {"error": exc.code}
        with self._lock:
            slug = self._unique_slug(draft.title)
        spec = self.full_spec(draft)
        args = [
            "orchestration", "worker-start", "--from", handle, "--run", run,
            "--spec", spec, "--task-title", draft.title, "--agent", draft.agent,
            "--worktree", self.settings.worktree,
        ]
        if self.settings.creates_worktree:
            args += ["--repo", draft.repo_selector, "--name", f"wawi-{slug}"]
        try:
            receipt = self.cli.run(*args, timeout=_START_TIMEOUT_S)
        except OrcaError as exc:
            stage = find_key(exc.payload, "failedStage") or find_key(exc.payload, "stage") or exc.code
            residual = find_key(exc.payload, "residualResources")
            log.warning("orca: worker-start failed at %s (residual: %s); not relaunching",
                        stage, residual)
            self._record(slug, draft, spec, f"falló al arrancar ({stage})")
            return False, f"Orca no pudo arrancar «{draft.title}»: falló en {stage}. No lo reintento.", {
                "error": "start_failed", "stage": stage, "residual": residual,
            }
        dispatch = str(find_key(receipt, "dispatchId") or "")
        task = str(find_key(receipt, "taskId") or "")
        if not dispatch:
            log.warning("orca: worker-start answered without a dispatch id: %s", receipt)
            self._record(slug, draft, spec, "estado desconocido (sin dispatch)")
            return False, "Orca contestó raro; no sé si arrancó. Fijate en Orca.", {
                "error": "no_dispatch"}
        branch, worktree = self._placement(receipt, dispatch)
        with self._lock:
            self.state.dispatches[dispatch] = {
                "task": task, "title": draft.title, "slug": slug,
                "repo": draft.repo_name, "branch": branch, "worktree": worktree,
                "started": self._clock(),
            }
            self.state.save()
        self._record(slug, draft, spec, f"arrancó (dispatch {dispatch})")
        log.info("orca: dispatched %r as %s on %s", draft.title, dispatch, branch or worktree)
        self.start_watcher()
        return True, f"Listo, un agente de Orca ya está con «{draft.title}».", {
            "dispatch": dispatch, "task": task, "branch": branch, "worktree": worktree}

    def _launch(self, draft: Draft) -> None:
        ok, speak, _ = self.launch(draft)
        self._notify("info" if ok else "failed", speak)

    def _placement(self, receipt: dict, dispatch: str) -> tuple[str, str]:
        branch = _clean_branch(find_key(receipt, "branch"))
        worktree = str(find_key(receipt, "worktreePath") or "")
        if not worktree:
            path = find_key(receipt.get("worktree") or {}, "path")
            worktree = str(path or "")
        if branch and worktree:
            return branch, worktree
        try:
            shown = self.cli.run("orchestration", "worker-show", "--dispatch", dispatch)
            branch = branch or _clean_branch(find_key(shown, "branch"))
            worktree = worktree or str(find_key(shown, "worktreePath") or "")
            if not worktree:
                # Orca's worktree ids read "<repo-id>::<absolute path>".
                wid = str(find_key(shown, "worktreeId") or find_key(receipt, "worktreeId") or "")
                worktree = wid.split("::", 1)[1] if "::" in wid else ""
        except OrcaError:
            pass
        if not branch and worktree:
            try:
                out = subprocess.run(
                    ["git", "-C", worktree, "branch", "--show-current"],
                    capture_output=True, text=True, timeout=5,
                )
                branch = out.stdout.strip()
            except Exception:
                pass
        return branch, worktree

    # -- status, answers, cancel ---------------------------------------------------
    def status_summary(self) -> tuple[str, dict]:
        with self._lock:
            dispatches = dict(self.state.dispatches)
            questions = list(self.state.questions)
        if not dispatches and not questions:
            return "No hay tareas en Orca.", {"tasks": []}
        live = self._liveness()
        now = self._clock()
        bits = []
        for did, d in dispatches.items():
            minutes = max(0, int((now - float(d.get("started") or now)) / 60))
            state = live.get(did, "")
            how = "parece que se cortó" if state == "exited" else f"lleva {minutes} min"
            bits.append(f"«{d.get('title')}» {how}")
        count = len(dispatches)
        head = ("Hay una tarea en Orca: " if count == 1 else f"Hay {count} tareas en Orca: ") \
            if count else ""
        speak = head + "; ".join(bits) + ("." if bits else "")
        if questions:
            q = questions[0]
            speak += f" Hay una pregunta esperando sobre «{q.get('title')}»: {q.get('text')}"
        return speak.strip(), {"tasks": dispatches, "questions": questions, "liveness": live}

    def _liveness(self) -> dict[str, str]:
        if not self.state.run or not self.state.dispatches:
            return {}
        try:
            result = self.cli.run("orchestration", "worker-list", "--run", self.state.run, timeout=20)
        except OrcaError:
            return {}
        out: dict[str, str] = {}
        rows = result.get("workers") or result.get("rows") or result.get("items") or []
        for row in rows if isinstance(rows, list) else []:
            did = str(find_key(row, "dispatchId") or "")
            verdict = find_key(row.get("projection") or {}, "liveness") if isinstance(row, dict) else None
            if isinstance(verdict, dict):
                verdict = verdict.get("verdict") or verdict.get("state")
            if did and isinstance(verdict, str):
                out[did] = verdict
        return out

    def answer(self, text: str) -> tuple[bool, str, dict]:
        text = (text or "").strip()
        if not text:
            return False, "No entendí qué contestarle.", {"error": "empty"}
        with self._lock:
            if not self.state.questions:
                return False, "Orca no está esperando ninguna respuesta.", {"error": "nothing_pending"}
            q = self.state.questions[0]
        try:
            handle, run = self.ensure_run()
            self.cli.run("orchestration", "reply", "--id", str(q["id"]), "--body", text,
                         "--from", handle, "--run", run)
        except OrcaError as exc:
            return False, "No pude pasarle la respuesta a Orca.", {"error": exc.code}
        with self._lock:
            self.state.questions = [x for x in self.state.questions if x.get("id") != q["id"]]
            self.state.save()
        return True, "Se lo paso.", {"answered": q.get("title"), "id": q["id"]}

    def find_dispatch(self, title: str) -> tuple[str, dict] | None:
        with self._lock:
            items = list(self.state.dispatches.items())
        if not items:
            return None
        wanted = slugify(title or "") if (title or "").strip() else ""
        if not wanted:
            return items[0] if len(items) == 1 else None
        for did, d in items:
            if wanted in str(d.get("slug") or "") or str(d.get("slug") or "") in wanted:
                return did, d
        return None

    def ask_cancel(self, title: str = "") -> tuple[bool, str, dict]:
        found = self.find_dispatch(title)
        if found is None:
            with self._lock:
                names = [d.get("title") for d in self.state.dispatches.values()]
            if not names:
                return False, "No hay tareas en Orca para cortar.", {"error": "nothing_running"}
            return False, "¿Cuál corto? " + ", ".join(f"«{n}»" for n in names) + ".", {
                "error": "ambiguous"}
        did, d = found
        self._set_pending(_Pending(
            "cancel", self._clock(), lambda: self._cancel(did),
            "Dale, la corto.", "Bueno, la dejo seguir.", {"dispatch": did}),
            "Cancelar tarea de Orca", str(d.get("title") or "Tarea"))
        return True, f"¿Corto la tarea «{d.get('title')}» en Orca?", {
            "sent": False, "note": "Esperá el sí del dueño; no llames otra herramienta."}

    def cancel(self, dispatch: str) -> tuple[bool, str]:
        """The guide's explicit cancel: worker-stop, then mark the Task failed."""
        with self._lock:
            d = dict(self.state.dispatches.get(dispatch) or {})
        title = d.get("title") or "la tarea"
        try:
            self.cli.run("orchestration", "worker-stop", "--dispatch", dispatch)
        except OrcaError as exc:
            log.warning("orca: worker-stop %s failed (%s)", dispatch, exc.code)
            return False, f"No pude cortar «{title}» en Orca."
        if d.get("task"):
            try:
                self.cli.run("orchestration", "task-update", "--id", str(d["task"]),
                             "--status", "failed", "--result", "cancelled")
            except OrcaError as exc:
                log.info("orca: task-update after stop failed (%s)", exc.code)
        with self._lock:
            self.state.dispatches.pop(dispatch, None)
            self.state.questions = [q for q in self.state.questions if q.get("dispatch") != dispatch]
            self.state.save()
        return True, f"Corté «{title}» en Orca."

    def _cancel(self, dispatch: str) -> None:
        ok, speak = self.cancel(dispatch)
        self._notify("info" if ok else "failed", speak)

    # -- the watcher -------------------------------------------------------------------
    def start_watcher(self) -> bool:
        """Watch while there is something in flight. Idempotent."""
        with self._lock:
            if not self.state.dispatches or self._stop.is_set():
                return False
            if self._watcher is not None and self._watcher.is_alive():
                return False
            if not self._background:
                return False
            self._watcher = threading.Thread(target=self.watch, name="wawi-orca-watch", daemon=True)
            self._watcher.start()
            return True

    def resume(self) -> bool:
        """At startup: pick the watch up again if tasks were left in flight."""
        if self.state.dispatches:
            log.info("orca: %d task(s) in flight from the last run; watching", len(self.state.dispatches))
        return self.start_watcher()

    def stop(self) -> None:
        self._stop.set()
        self.cli.kill_all()

    def watch(self) -> None:
        """``check --wait`` until nothing is left in flight, acking each batch
        only once every message in it was handled. Backs off on errors."""
        backoff = _BACKOFF_FIRST_S
        ack: str | None = None
        ack_failures = 0
        while not self._stop.is_set():
            with self._lock:
                active = bool(self.state.dispatches)
            if not active and ack is None:
                break
            try:
                handle, run = self.ensure_run()
                args = ["orchestration", "check", "--terminal", handle, "--run", run]
                if ack:
                    args += ["--ack", ack]
                if active:
                    ms = int(self.settings.wait_timeout_s * 1000)
                    args += ["--wait", "--types", WAIT_TYPES, "--timeout-ms", str(ms)]
                result = self.cli.run(*args, timeout=self.settings.wait_timeout_s + 60)
            except OrcaError as exc:
                log.warning("orca: check failed (%s); retrying in %.0fs", exc.code, backoff)
                if exc.code in ("consumer_fenced", "terminal_handle_stale", "no_active_sender_terminal"):
                    self._terminal_checked = False

                if ack:
                    ack_failures += 1
                    if ack_failures >= 3:
                        # Replays are handled idempotently; never wedge on an ack.
                        ack, ack_failures = None, 0
                if self._sleep(backoff):
                    break
                backoff = min(backoff * 2, _BACKOFF_MAX_S)
                continue
            backoff, ack, ack_failures = _BACKOFF_FIRST_S, None, 0
            messages = result.get("messages") or []
            delivery = result.get("deliveryId")
            if messages and delivery:
                for message in messages:
                    try:
                        self.handle_message(message)
                    except Exception:
                        log.warning("orca: could not handle a message", exc_info=True)
                ack = str(delivery)
            elif not active:
                break
        log.info("orca: watcher stopped")

    def handle_message(self, message: dict) -> None:
        kind = str(message.get("type") or "")
        payload = message.get("payload")
        if isinstance(payload, str):
            payload = _parse_json(payload) or {}
        if not isinstance(payload, dict):
            payload = {}
        dispatch = str(payload.get("dispatchId") or message.get("dispatchId") or "")
        if kind == "worker_done":
            self._on_done(dispatch, str(payload.get("outcome") or ""), message)
        elif kind in ("question", "escalation"):
            self._on_question(kind, dispatch, message)
        else:
            log.info("orca: ignoring a %r message", kind)

    def _on_done(self, dispatch: str, outcome: str, message: dict) -> None:
        with self._lock:
            entry = self.state.dispatches.get(dispatch)
        if entry is None:
            # Not one of Wawi's, or already handled (a replayed batch).
            log.info("orca: worker_done for unknown dispatch %r; nothing to do", dispatch)
            return
        title = entry.get("title") or "la tarea"
        summary = _first_sentence(message.get("body") or message.get("subject") or "")
        if outcome == "succeeded":
            where = f" Quedó en la rama {entry['branch']}." if entry.get("branch") else ""
            self._notify("done", f"¡Terminé lo de «{title}»!{where}")
        elif outcome == "failed":
            self._notify("failed", f"No salió lo de «{title}». {summary}".strip())
        else:
            # Not an accepted settlement: say it, keep the terminal, no release.
            log.warning("orca: worker_done for %s without a clear outcome (%r)", dispatch, outcome)
            self._notify("info", f"Orca terminó «{title}», pero no sé cómo salió.")
            with self._lock:
                self.state.dispatches.pop(dispatch, None)
                self.state.save()
            return
        try:
            self.cli.run("orchestration", "worker-release", "--dispatch", dispatch)
        except OrcaError as exc:
            log.warning("orca: release of %s uncertain (%s)", dispatch, exc.code)
        with self._lock:
            self.state.dispatches.pop(dispatch, None)
            self.state.questions = [q for q in self.state.questions if q.get("dispatch") != dispatch]
            self.state.save()

    def _on_question(self, kind: str, dispatch: str, message: dict) -> None:
        mid = str(message.get("id") or "")
        if not mid:
            return
        with self._lock:
            if any(q.get("id") == mid for q in self.state.questions):
                return  # replayed batch: already asked
            entry = self.state.dispatches.get(dispatch) or {}
            title = entry.get("title") or str(message.get("subject") or "una tarea")
            text = " ".join(str(message.get("body") or message.get("subject") or "").split())
            self.state.questions.append({
                "id": mid, "dispatch": dispatch, "title": title, "text": text[:400],
                "kind": kind, "at": self._clock(),
            })
            self.state.save()
        lead = "Orca necesita ayuda" if kind == "escalation" else "Orca pregunta"
        self._notify("question", f"{lead} sobre «{title}»: {_first_sentence(text, 160)}")


# --------------------------------------------------------------------------- #
# Module-level door (like wawi.games): tools and the orchestrator use these.
# --------------------------------------------------------------------------- #

_BRIDGE: OrcaBridge | None = None


def get() -> OrcaBridge | None:
    return _BRIDGE


def install(bridge: OrcaBridge | None) -> None:
    global _BRIDGE
    _BRIDGE = bridge


def build(cfg: Any, *, runner: Runner | None = None) -> OrcaBridge | None:
    """A bridge for *cfg*'s ``orca:`` section, or None (off, or no CLI).

    Spawns nothing: whether the app is running is asked on first use."""
    try:
        settings = OrcaSettings.from_dict(getattr(cfg, "orca", None) or {})
        if not settings.enabled:
            return None
        cli = resolve_cli(settings.cli)
        if cli is None:
            log.info("orca: no CLI found; sending tasks to Orca is off")
            return None
        return OrcaBridge(settings, OrcaCLI(cli, runner))
    except Exception:
        log.warning("orca: could not be set up; running without it", exc_info=True)
        return None


def hear(text: str) -> str | None:
    """Offer the transcript to a pending «¿la mando a Orca?». Never raises."""
    bridge = _BRIDGE
    if bridge is None:
        return None
    try:
        return bridge.hear(text)
    except Exception:
        log.warning("orca: hear failed", exc_info=True)
        return None


def pending_questions() -> list[dict]:
    bridge = _BRIDGE
    return list(bridge.state.questions) if bridge is not None else []


def _show_busy(on: bool) -> None:
    """Wawi types on her little laptop while any Orca task is in flight."""
    try:
        from wawi import working

        working.BOARD.set_busy("orca", on)
    except Exception:
        log.debug("orca: could not update the working board", exc_info=True)
