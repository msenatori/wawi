"""An Orca agent finished, failed, or is waiting for the owner.

Every ``orca_every_s`` (30 s), and only while the Orca app is running
(NSWorkspace, no launch), one ``orca worktree ps --json`` (~0.1 s, local)
lists every worktree with its agents' state: ``working``, ``done`` (with
``mainAgent.outcome`` ``failure`` when it failed), ``blocked`` / ``waiting``
(a question or a permission prompt). The watcher keeps the previous state per
agent pane and turns transitions into moments:

* working → done: "Terminó «tarea» ✓" (or "«tarea» falló");
* anything → blocked / waiting: "«tarea» te está esperando".

The first look after Wawi starts is a baseline only: a restart never
announces what was already done. Agents Wawi dispatched herself
(``wawi/orca.py`` keeps their worktree) are skipped: that bridge already
says how they went. An interrupted agent was stopped by the owner, who
knows.

The act: she trots over to where the owner works, and the bubble's link
switches Orca to that agent's terminal (``orca terminal switch``), or, when
the terminal is gone, shows the agent's last message on the board. Spoken
only when the owner was away when it happened, or the kind was promoted to
``say``. Nothing goes to Claude.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable

from wawi.initiative.core import Act, Ctx, Moment, register

log = logging.getLogger(__name__)

ORCA_BUNDLE = "com.stablyai.orca"
RUNNING = frozenset({"working"})
WAITING = frozenset({"blocked", "waiting"})
DONE = frozenset({"done"})
#: After a failed call, wait this long before asking again (seconds).
BACKOFF_S = 300.0
#: Long names are cut in the bubble.
TITLE_MAX = 40


@dataclass(frozen=True)
class Agent:
    pane: str
    state: str
    title: str
    repo: str = ""
    path: str = ""
    failed: bool = False
    interrupted: bool = False
    #: When it entered *state* (ms since epoch, Orca's clock) or 0.
    since_ms: float = 0.0
    last_message: str = ""


def _short(text: str, limit: int = TITLE_MAX) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def parse_ps(result: dict[str, Any]) -> dict[str, Agent]:
    """``orca worktree ps --json``'s result → {pane key: Agent}."""
    out: dict[str, Agent] = {}
    rows = result.get("worktrees") if isinstance(result, dict) else None
    for wt in rows if isinstance(rows, list) else []:
        if not isinstance(wt, dict) or wt.get("isArchived"):
            continue
        name = str(wt.get("displayName") or "")
        repo = str(wt.get("repo") or "")
        where = name if name and name not in ("main", "master") else repo
        for a in wt.get("agents") or []:
            if not isinstance(a, dict) or not a.get("paneKey"):
                continue
            main = a.get("mainAgent") if isinstance(a.get("mainAgent"), dict) else {}
            state = str(main.get("state") or a.get("state") or "")
            title = str(a.get("taskTitle") or a.get("displayName") or where or "un agente")
            try:
                since = float(main.get("stateStartedAt") or a.get("stateStartedAt") or 0)
            except (TypeError, ValueError):
                since = 0.0
            out[str(a["paneKey"])] = Agent(
                pane=str(a["paneKey"]), state=state, title=_short(title), repo=repo,
                path=str(wt.get("path") or ""),
                failed=str(main.get("outcome") or "") in ("failure", "failed"),
                interrupted=bool(a.get("interrupted")), since_ms=since,
                last_message=str(a.get("lastAssistantMessage") or "")[:2000])
    return out


def _orca_cli() -> Any:
    from wawi import orca

    bridge = orca.get()
    if bridge is not None:
        return bridge.cli  # the configured CLI, clean environment and all
    cli = orca.resolve_cli("")
    return None if cli is None else orca.OrcaCLI(cli)


def _orca_running() -> bool:
    from wawi.initiative.meeting import running_bundles

    return ORCA_BUNDLE in running_bundles()


def _own_worktrees() -> set[str]:
    """Worktrees of the tasks Wawi sent herself (wawi/orca.py announces those)."""
    try:
        from wawi import orca

        bridge = orca.get()
        if bridge is None:
            return set()
        return {str(d.get("worktree") or "") for d in bridge.state.dispatches.values()} - {""}
    except Exception:
        return set()


class OrcaWatcher:
    kinds = ("orca_done", "orca_waiting")

    def __init__(
        self,
        engine: Any,
        *,
        cli: Callable[[], Any] = _orca_cli,
        running: Callable[[], bool] = _orca_running,
        own: Callable[[], set[str]] = _own_worktrees,
        every_s: float = 30.0,
    ) -> None:
        self.engine = engine
        self._cli_factory = cli
        self._cli: Any = None
        self._running = running
        self._own = own
        self.every_s = float(every_s)
        self._next_at = 0.0
        self.agents: dict[str, Agent] | None = None

    def snapshot(self) -> dict[str, Agent] | None:
        if self._cli is None:
            self._cli = self._cli_factory()
            if self._cli is None:
                return None
        return parse_ps(self._cli.run("worktree", "ps", timeout=15))

    def poll(self, ctx: Ctx) -> list[Moment]:
        if all(self.engine.level(k) == "off" for k in self.kinds) and \
                self.engine.level("morning") == "off":
            return []
        if ctx.now < self._next_at:
            return []
        self._next_at = ctx.now + self.every_s
        try:
            if not self._running():
                return []
            now = self.snapshot()
        except Exception as exc:
            log.info("initiative: Orca did not answer (%s); retry in %d min",
                     getattr(exc, "code", exc), BACKOFF_S // 60)
            self._next_at = ctx.now + BACKOFF_S
            return []
        if now is None:
            self._next_at = ctx.now + 3600.0  # no CLI: look again much later
            return []
        before, self.agents = self.agents, now
        if before is None:
            return []  # the baseline: a restart announces nothing
        own = self._own()
        out: list[Moment] = []
        for pane, a in now.items():
            if a.path and a.path in own:
                continue
            prev = before.get(pane)
            was = prev.state if prev is not None else ""
            if a.state in DONE and was in RUNNING and not a.interrupted:
                out.append(Moment("orca_done", f"orca:{pane}:{a.since_ms:.0f}:done", {"agent": a}))
            elif a.state in WAITING and was not in WAITING:
                out.append(Moment("orca_waiting", f"orca:{pane}:{a.since_ms:.0f}:wait", {"agent": a}))
        return out

    def act(self, moment: Moment, ctx: Ctx) -> Act | None:
        a: Agent = moment.data["agent"]
        if moment.kind == "orca_done":
            text = f"«{a.title}» falló ✗" if a.failed else f"Terminó «{a.title}» ✓"
            say = f"Che, {a.title} falló." if a.failed else f"Che, terminó {a.title}."
        else:
            text = f"«{a.title}» te está esperando"
            say = f"Che, {a.title} te está esperando."
        return Act(moment.kind, text, link="Ver en Orca…",
                   accept=lambda: self.focus(a), say=say, walk=True,
                   ttl_s=1800.0, show_s=40.0, key=moment.key)

    # -- the click ------------------------------------------------------------------
    def focus(self, a: Agent) -> None:
        """Orca in front, on that agent's terminal; the board when it is gone."""
        handle = None
        try:
            cli = self._cli or self._cli_factory()
            if cli is not None:
                rows = cli.run("terminal", "list", timeout=15).get("terminals") or []
                for t in rows if isinstance(rows, list) else []:
                    if isinstance(t, dict) and f"{t.get('tabId')}:{t.get('leafId')}" == a.pane:
                        handle = str(t.get("handle") or "") or None
                        break
                if handle:
                    cli.run("terminal", "switch", "--terminal", handle, timeout=15)
        except Exception:
            log.info("initiative: could not switch Orca to %s", a.title, exc_info=True)
            handle = None
        if handle:
            try:
                subprocess.run(["open", "-b", ORCA_BUNDLE], timeout=5, check=False)
            except Exception:
                log.debug("initiative: could not bring Orca forward", exc_info=True)
            return
        board = self.engine.hooks.board
        if board is not None:
            body = a.last_message.strip() or "_Sin mensaje._"
            board.show(a.title, f"# {a.title}\n\n{body}")


def overnight(watcher: "OrcaWatcher | None", now: float, hours: float = 14.0) -> dict[str, list[str]]:
    """For the morning summary: {"working": [...], "done": [...], "failed": [...],
    "waiting": [...]} titles from the last snapshot; done/failed only within *hours*."""
    out: dict[str, list[str]] = {"working": [], "done": [], "failed": [], "waiting": []}
    agents = getattr(watcher, "agents", None) or {}
    for a in agents.values():
        if a.state in RUNNING:
            out["working"].append(a.title)
        elif a.state in WAITING:
            out["waiting"].append(a.title)
        elif a.state in DONE and a.since_ms and now - a.since_ms / 1000.0 <= hours * 3600.0:
            out["failed" if a.failed else "done"].append(a.title)
    return out


@register("orca", order=20)
def _orca_watcher(engine: Any) -> OrcaWatcher | None:
    if not engine.settings.orca:
        return None
    return OrcaWatcher(engine, every_s=engine.settings.orca_every_s)
