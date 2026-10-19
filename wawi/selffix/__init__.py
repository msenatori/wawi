"""Autoarreglo: Wawi fixes her own recurring problems and builds the
capabilities she was asked for -- on her own, behind gates she cannot edit.

The owner's rule: "todo sola, salvo que sea algo que me involucre o necesite
ayuda de mi parte; su objetivo es serme útil". So:

1. **Pick** (issues.py): a recurring or user-reported issue, or a capability
   request; most useful to him first; never one marked won't-fix or one that
   failed in the last week. Blocked-on-him work becomes a friendly request
   instead (classify.py, :func:`owner_requests`).
2. **Work** (service.py, engines.py): a fresh git worktree outside the repo,
   a coding agent (Codex by default, Claude as fallback; in a visible Orca
   terminal when Orca runs) with a tight, scrubbed brief and 30 minutes.
3. **Gate** (gates.py): forbidden paths, diff size, no new dependencies,
   secret scan, the full test suite, and a second-opinion review by the
   other engine. Any failure discards the attempt and is remembered.
4. **Decide** (classify.py): internal changes apply on their own; anything
   that involves him is offered (bubble + diff on the board).
5. **Apply** (gitops.py): one squash commit on main with a
   ``Wawi-Selffix: <issue>`` trailer, only on a clean main checkout; the
   previous HEAD is kept as ``last_good``.
6. **Watch** (watchdog.py): a frozen, stdlib-only copy restarts her, checks
   the startup and the next half hour, and ``git revert``s the self-fix (never
   anything else, never a reset) if she crashes or the issue comes back.

Nothing is ever pushed. This package is itself a forbidden path for the
agents it runs: she cannot rewrite her own guardrails.

Hooks for the initiative layer (``wawi/initiative``), stable on purpose:
:func:`can_work`, :func:`add_work_gate`, :func:`today_markdown`,
:func:`owner_requests`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from wawi import paths as _paths

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_SERVICE = None


def install(service: Any) -> None:
    global _SERVICE
    _SERVICE = service


def get() -> Any:
    return _SERVICE


def build(cfg: Any, *, board: Any = None, repo: Path | None = None) -> Any:
    """The service for *cfg* (config ``selffix:``). None outside a git checkout."""
    from wawi.selffix.engines import LocalExecutor, OrcaExecutor
    from wawi.selffix.service import SelffixService
    from wawi.selffix.settings import SelffixSettings
    from wawi.selffix.state import Store

    repo = Path(repo or REPO_ROOT)
    if not (repo / ".git").exists():
        log.info("selffix: %s is not a git checkout; off", repo)
        return None
    settings = SelffixSettings.from_dict(getattr(cfg, "selffix", {}) or {})
    local = LocalExecutor()
    author: Any = local
    if settings.use_orca:
        try:
            from wawi import orca

            cli = orca.resolve_cli(str((getattr(cfg, "orca", {}) or {}).get("cli") or ""))
            if cli:
                author = OrcaExecutor(orca.OrcaCLI(cli), repo, fallback=local)
        except Exception:
            log.debug("selffix: no Orca executor", exc_info=True)

    def restart_cmd() -> list[str] | None:
        try:
            from wawi.panel import restart_command

            return restart_command()
        except Exception:
            return None

    issues_dir = getattr(cfg, "issues_dir", None)
    return SelffixService(
        settings, repo=repo, issues_dir=Path(issues_dir) if issues_dir else None,
        log_path=_paths.private("wawi.log").resolve(),
        store=Store(_paths.resolve(settings.local_file) if settings.local_file else None),
        author_executor=author, local=local, board=board, restart_cmd=restart_cmd)


def remote_executor(cfg: Any) -> Any:
    """The optional ssh executor for heavy jobs, or None when it is off."""
    from wawi.selffix.remote import RemoteError, RemoteExecutor
    from wawi.selffix.settings import SelffixSettings

    settings = SelffixSettings.from_dict(getattr(cfg, "selffix", {}) or {})
    try:
        return RemoteExecutor(settings.remote)
    except RemoteError:
        return None


# --------------------------------------------------------------------------- #
# Hooks for the initiative layer
# --------------------------------------------------------------------------- #


def can_work() -> tuple[bool, str]:
    """(may she start a self-fix now, why not)."""
    service = get()
    if service is None:
        return False, "sin_servicio"
    return service.can_work()


def add_work_gate(gate: Callable[[], Any]) -> bool:
    """Add a condition for working (True/False or (bool, reason))."""
    service = get()
    if service is None:
        return False
    service.add_work_gate(gate)
    return True


def today_markdown() -> str:
    """The daily summary's "Lo que me arreglé hoy" section ("" = nothing)."""
    service = get()
    return service.today_markdown() if service is not None else ""


def owner_requests() -> list[dict]:
    """What she needs from the owner to do blocked work:
    ``[{"id", "issue", "need", "text"}]``, text ready to offer."""
    service = get()
    return service.owner_requests() if service is not None else []
