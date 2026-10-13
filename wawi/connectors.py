"""Connectors: reuse the MCP servers the user already set up in other apps.

Wawi cannot reach Linear, Obsidian or anything else on its own, but Claude
Code and Claude Desktop often already can. This module borrows those servers
instead of wiring each one again by hand.

Four rules shape it:

1. **Read, never write.** The source files (``~/.claude.json``, a project's
   ``.mcp.json``, Claude Desktop's config) are opened read-only and never
   written back, not even to fix them.
2. **Off until named.** An imported server does nothing until ``config.yaml``
   enables it by name (``connectors: {obsidian: {enabled: true}}``). With no
   connector enabled, the brain's options are byte-for-byte what they were.
3. **Read-only fails closed.** A tool counts as a read only when the server
   marks it ``readOnlyHint: true`` AND it is on a curated list. A missing
   annotation, a missing list or a server whose status cannot be read all
   mean "not a read". In ``read_only`` mode everything else is denied.
4. **Secrets stay in memory.** env and header VALUES go to the SDK and
   nowhere else: logs, ``repr`` and every model-facing message carry names
   only.

The ``read_send`` mode lets the other tools through after ONE spoken yes.
That yes is read from the next transcript, not from text the model chose:
the first call is refused with a preview the model reads aloud, and the same
call goes through only if the user's reply opens with a yes. A mistranscribed
"mandale" therefore costs a question, never a message.

The SDK only exposes tool annotations through ``get_mcp_status()`` on a
connected client, so the read-only set is learned at connect time and on
demand from inside the gate (see :meth:`Connectors.set_status_source`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from wawi import lang

log = logging.getLogger(__name__)

MODE_READ_ONLY = "read_only"
"""Spoken as "solo lectura": only verified reads are callable."""

MODE_READ_SEND = "read_send"
"""Spoken as "lectura y envío": reads run, anything else asks once by voice."""

MODES = (MODE_READ_ONLY, MODE_READ_SEND)

#: The brain's own in-process server. A borrowed server with this name would
#: shadow Wawi's tools, so it is never imported.
RESERVED_NAMES = frozenset({"wawi", "wawi_task"})

#: How long a pending "¿lo mando?" stays answerable. Past it a late "sí" is
#: not a yes to anything, and the model has to ask again.
CONFIRM_TIMEOUT_S = 120.0

#: Bound on the on-demand status refresh inside the gate. A server that cannot
#: say what its tools are within this is treated as having no reads.
_STATUS_TIMEOUT_S = 5.0

#: Longest argument value repeated in a spoken preview. The model reads the
#: preview aloud, and a whole document read back is not a confirmation.
_PREVIEW_VALUE_CHARS = 240

#: Example curated read lists for a few popular community MCP servers
#: (mcp-obsidian, Engram, CodeGraph), keyed by the name the server is usually
#: registered under. They are examples, not a dependency: any other server
#: starts with no reads until you list them. Still subject to
#: ``readOnlyHint``: being on this list is necessary, not enough.
#: ``read_tools`` in config.yaml replaces or adds the entry for a server.
DEFAULT_READ_TOOLS: dict[str, frozenset[str]] = {
    "obsidian": frozenset({
        "obsidian_list_files_in_vault",
        "obsidian_list_files_in_dir",
        "obsidian_get_file_contents",
        "obsidian_batch_get_file_contents",
        "obsidian_simple_search",
        "obsidian_complex_search",
        "obsidian_search_by_tag",
        "obsidian_get_frontmatter",
        "obsidian_get_periodic_note",
        "obsidian_get_recent_periodic_notes",
        "obsidian_get_recent_changes",
    }),
    "engram": frozenset({
        "mem_search",
        "mem_context",
        "mem_get_observation",
    }),
    "codegraph": frozenset({"codegraph_explore"}),
}


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


def claude_code_path(home: Path) -> Path:
    return home / ".claude.json"


def desktop_path(home: Path) -> Path:
    return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"


@dataclass(frozen=True)
class ImportedServer:
    """One borrowed server, ready to hand to the SDK.

    ``config`` holds the env and header values, so it is kept out of ``repr``:
    a stray ``log.debug("%r", server)`` must not be how a token leaks.
    """

    name: str
    source: str
    transport: str
    config: dict[str, Any] = field(repr=False, compare=False)

    def describe(self) -> str:
        """Names only: what a log line or the panel may show."""
        secret_names = sorted(
            {*self.config.get("env", {}), *self.config.get("headers", {})}
        )
        names = f" secrets={','.join(secret_names)}" if secret_names else ""
        return f"{self.name} ({self.transport}, {self.source}){names}"


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` the way Claude Code does."""
    if isinstance(value, str):
        return _VAR.sub(
            lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else m.group(0)),
            value,
        )
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _expand(v) for k, v in value.items()}
    return value


def _to_sdk(raw: Any) -> tuple[str, dict[str, Any]] | None:
    """Map one ``mcpServers`` entry to an SDK config. None when unusable."""
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type") or ("stdio" if "command" in raw else "http" if "url" in raw else ""))
    if kind == "stdio":
        command = raw.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        config: dict[str, Any] = {"type": "stdio", "command": _expand(command)}
        args = raw.get("args")
        if isinstance(args, list):
            config["args"] = [str(a) for a in _expand(args)]
        env = raw.get("env")
        if isinstance(env, dict) and env:
            config["env"] = {str(k): str(v) for k, v in _expand(env).items()}
        return "stdio", config
    if kind in ("http", "streamable-http", "sse"):
        url = raw.get("url")
        if not isinstance(url, str) or not url.strip():
            return None
        transport = "sse" if kind == "sse" else "http"
        config = {"type": transport, "url": _expand(url)}
        headers = raw.get("headers")
        if isinstance(headers, dict) and headers:
            config["headers"] = {str(k): str(v) for k, v in _expand(headers).items()}
        return transport, config
    return None


def _read_json(path: Path) -> dict[str, Any]:
    """Parse *path* read-only. Errors log the file and the error class only:
    a JSON error message can quote the line it choked on, token and all."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("connectors: could not read %s (%s)", path, type(exc).__name__)
        return {}
    return data if isinstance(data, dict) else {}


def discover(home: Path | None = None, cwd: Path | None = None) -> dict[str, ImportedServer]:
    """Every server the user's other apps declare, by name. Writes nothing.

    When two sources declare the same name, the first one below wins: Claude
    Code's user scope, then *cwd*'s ``.mcp.json``, then Claude Code's
    per-project entries (sorted by path, so the result is stable), then
    Claude Desktop.
    """
    home = Path(home) if home is not None else Path.home()
    cwd = Path(cwd) if cwd is not None else Path.cwd()

    claude = _read_json(claude_code_path(home))
    sources: list[tuple[str, Any]] = [("claude_code", claude.get("mcpServers"))]
    sources.append(("mcp.json", _read_json(cwd / ".mcp.json").get("mcpServers")))
    projects = claude.get("projects")
    if isinstance(projects, dict):
        for path in sorted(projects):
            entry = projects[path]
            if isinstance(entry, dict):
                sources.append(("claude_code:project", entry.get("mcpServers")))
    sources.append(("claude_desktop", _read_json(desktop_path(home)).get("mcpServers")))

    found: dict[str, ImportedServer] = {}
    for source, servers in sources:
        if not isinstance(servers, dict):
            continue
        for name, raw in servers.items():
            name = str(name)
            if name in found:
                continue
            mapped = _to_sdk(raw)
            if mapped is None:
                log.info("connectors: %s from %s is not importable", name, source)
                continue
            found[name] = ImportedServer(name, source, mapped[0], mapped[1])
    return found


# --------------------------------------------------------------------------- #
# Enabled connectors and the gate
# --------------------------------------------------------------------------- #


def tool_prefix(name: str) -> str:
    """The server part of ``mcp__<server>__<tool>``, as the CLI normalizes it."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


@dataclass(frozen=True)
class _Connector:
    server: ImportedServer
    prefix: str
    mode: str
    read_tools: frozenset[str]


@dataclass
class _Pending:
    """A send the user was asked about and has not answered yet."""

    key: tuple[str, str]
    turn: int
    at: float
    confirmed: bool = False


#: What the gate decides about one call: None for "not mine, no opinion",
#: else (allowed, model-facing reason).
Decision = tuple[bool, str] | None


class Connectors:
    """The enabled subset of the imported servers, plus the gate over them.

    One instance per run: it remembers which tools proved read-only and the
    send waiting on a yes, and both must outlive any single connection.
    """

    def __init__(
        self,
        servers: dict[str, ImportedServer],
        settings: dict[str, dict[str, Any]] | None = None,
        *,
        confirm_timeout_s: float = CONFIRM_TIMEOUT_S,
    ) -> None:
        self._enabled: dict[str, _Connector] = {}
        for name, setting in (settings or {}).items():
            if not setting.get("enabled"):
                continue
            server = servers.get(name)
            if server is None:
                log.warning("connectors: %r is enabled but no app declares it", name)
                continue
            prefix = tool_prefix(name)
            if prefix in RESERVED_NAMES or prefix in self._enabled:
                log.warning("connectors: %r clashes with another server; skipped", name)
                continue
            mode = setting.get("mode") or MODE_READ_ONLY
            read_tools = setting.get("read_tools")
            if read_tools is None:
                curated = DEFAULT_READ_TOOLS.get(name, frozenset())
            else:
                curated = frozenset(str(t) for t in read_tools)
            self._enabled[prefix] = _Connector(server, prefix, mode, curated)
            log.info("connectors: enabled %s mode=%s", server.describe(), mode)
        self._confirm_timeout_s = confirm_timeout_s
        self._verified: dict[str, frozenset[str]] = {}
        self._seen: dict[str, frozenset[str]] = {}
        self._status_source: Callable[[], Awaitable[Any]] | None = None
        self._pending: _Pending | None = None
        self._turn = 0
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        settings: dict[str, dict[str, Any]] | None,
        *,
        home: Path | None = None,
        cwd: Path | None = None,
    ) -> "Connectors":
        """Import only when something is enabled: no setting, no file read."""
        if not any(s.get("enabled") for s in (settings or {}).values()):
            return cls({}, {})
        return cls(discover(home, cwd), settings)

    def __bool__(self) -> bool:
        return bool(self._enabled)

    # -- what the SDK is given ------------------------------------------

    def mcp_servers(self) -> dict[str, dict[str, Any]]:
        return {p: dict(c.server.config) for p, c in self._enabled.items()}

    def allowed_tools(self) -> list[str]:
        """read_only lists the curated reads by name, so even with the gate
        gone nothing else is pre-approved; read_send pre-approves the server
        and leaves every non-read call to the gate."""
        allowed: list[str] = []
        for prefix, conn in self._enabled.items():
            if conn.mode == MODE_READ_SEND:
                allowed.append(f"mcp__{prefix}")
            else:
                allowed.extend(f"mcp__{prefix}__{t}" for t in sorted(conn.read_tools))
        return allowed

    def disallowed_tools(self) -> list[str]:
        """Non-read tools of read_only servers seen so far, to keep their
        schemas out of the model's context from the next connection on."""
        hidden: list[str] = []
        for prefix, conn in self._enabled.items():
            if conn.mode != MODE_READ_ONLY:
                continue
            reads = self._verified.get(prefix, frozenset()) & conn.read_tools
            hidden.extend(f"mcp__{prefix}__{t}" for t in sorted(self._seen.get(prefix, frozenset()) - reads))
        return hidden

    def hooks(self) -> dict[str, list[Any]]:
        """A PreToolUse hook on EVERY tool. A narrower matcher would be a regex
        the CLI interprets, and a matcher that misses is a gate that is open."""
        from claude_agent_sdk import HookMatcher  # type: ignore[import]

        return {"PreToolUse": [HookMatcher(matcher=None, hooks=[self._pre_tool_use])]}

    def signature(self) -> tuple:
        return tuple(sorted((p, c.mode) for p, c in self._enabled.items()))

    # -- read-only evidence ---------------------------------------------

    def set_status_source(self, source: Callable[[], Awaitable[Any]] | None) -> None:
        """Where the gate asks for annotations it has not seen yet, normally
        a connected client's ``get_mcp_status``."""
        self._status_source = source

    def learn(self, status: Any) -> None:
        """Record which tools each enabled server marks read-only.

        Takes a ``get_mcp_status()`` response. Annotations arrive as
        ``readOnly`` there; ``readOnlyHint`` is accepted too.
        """
        servers = status.get("mcpServers") if isinstance(status, dict) else None
        for entry in servers or []:
            if not isinstance(entry, dict):
                continue
            prefix = tool_prefix(str(entry.get("name", "")))
            if prefix not in self._enabled or not entry.get("tools"):
                continue
            seen: set[str] = set()
            reads: set[str] = set()
            for tool in entry["tools"]:
                if not isinstance(tool, dict):
                    continue
                name = str(tool.get("name", "")).removeprefix(f"mcp__{prefix}__")
                notes = tool.get("annotations") or {}
                seen.add(name)
                if notes.get("readOnly") is True or notes.get("readOnlyHint") is True:
                    reads.add(name)
            self._seen[prefix] = frozenset(seen)
            self._verified[prefix] = frozenset(reads)
            log.info(
                "connectors: %s has %d tool(s), %d usable as reads",
                prefix,
                len(seen),
                len(reads & self._enabled[prefix].read_tools),
            )

    async def refresh(self) -> None:
        """Re-read the status. Failure leaves the old evidence: fail closed."""
        source = self._status_source
        if source is None:
            return
        try:
            self.learn(await asyncio.wait_for(source(), _STATUS_TIMEOUT_S))
        except Exception as exc:
            log.warning("connectors: status unavailable (%s)", type(exc).__name__)

    def is_read(self, prefix: str, tool: str) -> bool:
        conn = self._enabled.get(prefix)
        return (
            conn is not None
            and tool in conn.read_tools
            and tool in self._verified.get(prefix, frozenset())
        )

    # -- the gate ---------------------------------------------------------

    def begin_turn(self, user_text: str) -> None:
        """Called with the TRANSCRIPT before each turn: the one place a yes
        can come from. A pending send survives only into the very next turn,
        and only if that turn opens with a yes."""
        with self._lock:
            self._turn += 1
            pending = self._pending
            if pending is None:
                return
            fresh = time.monotonic() - pending.at <= self._confirm_timeout_s
            if pending.turn == self._turn - 1 and fresh and lang.affirmative(user_text) is True:
                pending.confirmed = True
                return
            self._pending = None
        log.info("connectors: %s not confirmed; nothing sent", pending.key[0])

    def _split(self, tool_name: str) -> tuple[str, str] | None:
        for prefix in self._enabled:
            head = f"mcp__{prefix}__"
            if tool_name.startswith(head):
                return prefix, tool_name[len(head):]
        return None

    def decide(self, tool_name: str, tool_input: dict[str, Any]) -> Decision:
        """The gate itself, free of SDK types so it can be tested directly."""
        split = self._split(tool_name)
        if split is None:
            return None
        prefix, tool = split
        if self.is_read(prefix, tool):
            return True, ""
        conn = self._enabled[prefix]
        if conn.mode != MODE_READ_SEND:
            log.info("connectors: denied %s (read-only connector)", tool_name)
            return False, (
                f"Not available: {prefix} is connected read-only, and {tool} is "
                "not a verified read. Tell the user you can only read there."
            )

        key = (tool_name, json.dumps(tool_input, sort_keys=True, default=str))
        with self._lock:
            pending = self._pending
            if pending is not None and pending.key == key and pending.confirmed:
                self._pending = None
                log.info("connectors: %s confirmed by voice", tool_name)
                return True, ""
            self._pending = _Pending(key, self._turn, time.monotonic())
        log.info(
            "connectors: %s waits for a spoken yes (args: %s)",
            tool_name,
            ", ".join(sorted(tool_input)) or "-",
        )
        return False, (
            "NOT SENT YET: this action needs the user's spoken yes first. Do not "
            "call any tool again this turn. Tell the user, in one short Spanish "
            f"sentence, exactly this and ask: «{preview(prefix, tool, tool_input)}» "
            "¿Lo mando? If their next reply is yes, call the same tool again with "
            "exactly the same arguments. Otherwise tell them nothing was sent."
        )

    async def _pre_tool_use(self, input_data: Any, tool_use_id: Any, context: Any) -> dict[str, Any]:
        tool_name = str(input_data.get("tool_name", ""))
        tool_input = dict(input_data.get("tool_input") or {})
        split = self._split(tool_name)
        if split is not None and split[1] in self._enabled[split[0]].read_tools \
                and not self.is_read(*split):
            # A curated read the server has not vouched for YET, usually
            # because it was still starting at connect time. Ask once.
            await self.refresh()
        decision = self.decide(tool_name, tool_input)
        # An allow needs no say from us: every call the gate lets through is
        # already pre-approved by allowed_tools. Only a deny has to be spoken.
        if decision is None or decision[0]:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": decision[1],
            }
        }


def preview(connector: str, tool: str, args: dict[str, Any]) -> str:
    """Spanish, speakable: the connector, the action and EVERY argument."""
    action = tool.replace("_", " ")
    parts = []
    for key in sorted(args):
        value = args[key]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        text = " ".join(text.split())
        if len(text) > _PREVIEW_VALUE_CHARS:
            text = text[: _PREVIEW_VALUE_CHARS - 1].rstrip() + "…"
        parts.append(f"{key}: {text}")
    detail = "; ".join(parts) if parts else "sin datos"
    return f"Voy a usar {action} en {connector} con {detail}."
