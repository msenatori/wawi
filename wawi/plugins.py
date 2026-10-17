"""Agregados: the owner's own services, read through MCP servers or REST APIs.

``wawi/connectors.py`` borrows servers other apps already declare. This module
is the other half: services the owner adds HERE, from the menu ("Agregados…")
or the terminal (``scripts/wawictl plugins …``), persisted in a gitignored
``plugins_local.json`` next to config.yaml.

Two kinds:

* **mcp** — a stdio command (``npx -y some-mcp``) or an http/sse URL. It
  becomes one more MCP server for the brain, gated by the connectors' hook
  (prefix ``plug_<id>``).
* **rest** — a base URL plus named endpoints. It becomes ONE tool,
  ``svc_<id>``, with an ``endpoint`` enum and a ``params`` object: one schema
  per service instead of one per endpoint, because every schema is re-sent
  and re-paid for on every turn.

Rules that shape it:

1. **Read-only by default.** REST plugins expose only GET endpoints; MCP
   plugins only tools that look like reads (annotation or verb, see
   :func:`wawi.connectors.plugin_tool_is_read`). ``allow_writes`` opts a
   plugin in to the rest, and every write still waits for an on-screen
   "Aceptar" (wawi/confirm.py). Nobody to click means no write.
2. **Secrets never touch the JSON.** A token lives in the macOS Keychain
   (service ``wawi-plugin-<id>``) or in an environment variable; the JSON
   keeps only the reference (``{"keychain": "token"}`` / ``{"env": "VAR"}``).
   Values are resolved in memory when a call is made and are never logged,
   printed or returned to the model.
3. **Live.** The file is re-read when it changes (a stat per turn). A REST
   change rebuilds the tools and an MCP change rebuilds the brain's server
   list; either reconnects the brain session ONCE on the next turn (the
   system prompt never changes, so the prompt cache survives everything but
   that one reconnect). No restart needed.
4. **Short answers.** 10 s timeouts, bodies capped, JSON trimmed and
   pretty-printed under :data:`MAX_RESULT_CHARS`, errors as one short line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from wawi.connectors import plugin_tool_is_read
from wawi.reactions import BUS, Event, Reaction
from wawi.sensors.base import Debouncer, Sensor

log = logging.getLogger(__name__)

DEFAULT_FILE = "plugins_local.json"
#: Overrides the registry path (tests, a second checkout).
ENV_FILE = "WAWI_PLUGINS_FILE"

KINDS = ("rest", "mcp")
AUTH_TYPES = ("none", "bearer", "header")
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
TRANSPORTS = ("stdio", "http", "sse")

#: ``svc_<id>``: the one tool a REST plugin becomes.
REST_TOOL_PREFIX = "svc_"
#: ``plug_<id>``: the MCP server name of an MCP plugin (tools are
#: ``mcp__plug_<id>__<tool>``). Distinct from borrowed connectors' names.
MCP_PREFIX = "plug_"
#: The "¿qué servicios tenés?" tool, present while any plugin exists.
SERVICES_TOOL = "my_services"

KEYCHAIN_PREFIX = "wawi-plugin-"
TIMEOUT_S = 10.0
MAX_BODY_BYTES = 512_000
#: What the model gets of a response body, at most (~6 KB).
MAX_RESULT_CHARS = 6000
#: How long a write waits for "Aceptar" before it is dropped.
CONFIRM_TIMEOUT_S = 45.0
#: Longest description that reaches a tool schema (paid per turn).
MAX_TOOL_DESCRIPTION = 600

USER_AGENT = "Wawi/1.0 (agregados)"

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_NAME_REF = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
#: Header and env names that carry credentials: refused as plain values, so a
#: token typed into the wrong field cannot end up in the JSON.
_SECRET_NAME = re.compile(r"(auth|token|secret|passw|key|cookie|session|credential)", re.I)


class PluginError(ValueError):
    """A plugin definition or call that cannot work. Message is Spanish and
    safe to show or say: it never contains a secret."""


# --------------------------------------------------------------------------- #
# Secrets: Keychain or environment, never the JSON
# --------------------------------------------------------------------------- #


def keychain_service(plugin_id: str) -> str:
    return f"{KEYCHAIN_PREFIX}{plugin_id}"


def _security(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run ``/usr/bin/security`` (the test seam). Output is captured and
    never logged: on a read it IS the secret."""
    return subprocess.run(["security", *args], input=stdin, capture_output=True,
                          text=True, timeout=10, check=False)


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def keychain_set(plugin_id: str, account: str, value: str) -> None:
    """Store *value* in the login Keychain. Through ``security -i`` on stdin,
    so the value never appears in a process listing (argv is public)."""
    if not value or "\n" in value or "\r" in value:
        raise PluginError("El token no puede estar vacío ni tener saltos de línea.")
    command = (f"add-generic-password -U -s {_quote(keychain_service(plugin_id))} "
               f"-a {_quote(account)} -w {_quote(value)}\n")
    result = _security(["-i"], stdin=command)
    if result.returncode != 0:
        raise PluginError("No pude guardar el token en el Llavero.")
    log.info("plugins: stored a secret for %s (%s) in the Keychain", plugin_id, account)


def keychain_get(plugin_id: str, account: str) -> str | None:
    try:
        result = _security(["find-generic-password", "-s", keychain_service(plugin_id),
                            "-a", account, "-w"])
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.rstrip("\n")
    return value or None


def keychain_delete(plugin_id: str, account: str) -> None:
    try:
        _security(["delete-generic-password", "-s", keychain_service(plugin_id), "-a", account])
    except (OSError, subprocess.SubprocessError):
        log.debug("plugins: keychain delete failed for %s", plugin_id)


def resolve_secret(plugin_id: str, ref: dict[str, str] | None) -> str | None:
    """The value a reference points at, in memory only. None when missing."""
    if not ref:
        return None
    if "env" in ref:
        return os.environ.get(ref["env"]) or None
    if "keychain" in ref:
        return keychain_get(plugin_id, ref["keychain"])
    return None


def describe_ref(ref: dict[str, str] | None) -> str:
    """Where a secret lives, for the CLI and the window. Names only."""
    if not ref:
        return "sin token"
    if "env" in ref:
        return f"variable ${ref['env']}"
    return f"Llavero ({ref['keychain']})"


def _secret_refs(plugin: dict[str, Any]) -> list[dict[str, str]]:
    refs = []
    secret = (plugin.get("auth") or {}).get("secret")
    if secret:
        refs.append(secret)
    refs.extend(((plugin.get("mcp") or {}).get("secret_env") or {}).values())
    return refs


# --------------------------------------------------------------------------- #
# Validation: one shape for the file, the CLI and the window
# --------------------------------------------------------------------------- #


def slug(name: str) -> str:
    """An id from a display name: «Mi API (prod)» -> ``mi_api_prod``."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (text or "servicio")[:32].rstrip("_") or "servicio"


def _ref(raw: Any, where: str) -> dict[str, str] | None:
    if raw in (None, {}, ""):
        return None
    if not isinstance(raw, dict) or len(raw) != 1:
        raise PluginError(f"{where}: la referencia al token tiene que ser keychain o env.")
    (key, value), = raw.items()
    if key not in ("keychain", "env") or not isinstance(value, str) or not _NAME_REF.match(value):
        raise PluginError(f"{where}: la referencia al token tiene que ser keychain o env.")
    return {key: value}


def _http_url(raw: Any, where: str) -> str:
    url = str(raw or "").strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise PluginError(f"{where}: necesito una URL http(s) completa.")
    if parsed.username or parsed.password:
        raise PluginError(f"{where}: no pongas usuario ni clave en la URL; usá un token.")
    return url


def _plain_map(raw: Any, where: str) -> dict[str, str]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise PluginError(f"{where} tiene que ser un mapa nombre -> valor.")
    out = {}
    for key, value in raw.items():
        key = str(key).strip()
        if not key:
            continue
        if _SECRET_NAME.search(key):
            raise PluginError(
                f"{where}: «{key}» parece una credencial. Guardala como token "
                "(Llavero o variable de entorno), no como texto.")
        out[key] = str(value)
    return out


def normalize(raw: Any) -> dict[str, Any]:
    """Validate one plugin definition and return its canonical form.

    Raises :class:`PluginError` with a Spanish message. Unknown keys are
    dropped, so nothing smuggled into the file reaches a request.
    """
    if not isinstance(raw, dict):
        raise PluginError("Cada agregado tiene que ser un objeto.")
    name = " ".join(str(raw.get("name") or raw.get("id") or "").split())[:60]
    pid = str(raw.get("id") or slug(name)).strip().lower()
    if not _ID.match(pid):
        raise PluginError(f"Id inválido «{pid}»: minúsculas, números, _ o -, hasta 32.")
    kind = str(raw.get("kind") or "").lower()
    if kind not in KINDS:
        raise PluginError(f"{pid}: el tipo tiene que ser rest o mcp.")
    plugin: dict[str, Any] = {
        "id": pid,
        "name": name or pid,
        "kind": kind,
        "description": " ".join(str(raw.get("description") or "").split())[:300],
        "enabled": bool(raw.get("enabled", True)),
        "allow_writes": bool(raw.get("allow_writes", False)),
    }
    auth_raw = raw.get("auth") or {}
    if not isinstance(auth_raw, dict):
        raise PluginError(f"{pid}: auth tiene que ser un objeto.")
    auth_type = str(auth_raw.get("type") or "none").lower()
    if auth_type not in AUTH_TYPES:
        raise PluginError(f"{pid}: auth tiene que ser none, bearer o header.")
    auth: dict[str, Any] = {"type": auth_type}
    if auth_type != "none":
        secret = _ref(auth_raw.get("secret"), pid)
        if secret is None:
            raise PluginError(f"{pid}: con auth {auth_type} hace falta un token (Llavero o variable).")
        auth["secret"] = secret
    if auth_type == "header":
        header = str(auth_raw.get("header") or "").strip()
        if not re.match(r"^[A-Za-z0-9-]{1,64}$", header):
            raise PluginError(f"{pid}: decime el nombre del encabezado del token (p. ej. X-Api-Key).")
        auth["header"] = header
    plugin["auth"] = auth

    if kind == "rest":
        plugin["base_url"] = _http_url(raw.get("base_url"), pid).rstrip("/")
        plugin["headers"] = _plain_map(raw.get("headers"), f"{pid}: headers")
        endpoints = []
        seen: set[str] = set()
        for item in raw.get("endpoints") or []:
            endpoint = _endpoint(item, pid)
            if endpoint["name"] in seen:
                raise PluginError(f"{pid}: el endpoint «{endpoint['name']}» está repetido.")
            seen.add(endpoint["name"])
            endpoints.append(endpoint)
        plugin["endpoints"] = endpoints
    else:
        plugin["mcp"] = _mcp(raw.get("mcp") or {}, pid, auth_type)

    health = raw.get("health")
    if health:
        plugin["health"] = _health(health, plugin)
    return plugin


def _endpoint(raw: Any, pid: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PluginError(f"{pid}: cada endpoint tiene que ser un objeto.")
    name = str(raw.get("name") or "").strip()
    if not re.match(r"^[A-Za-z0-9_-]{1,40}$", name):
        raise PluginError(f"{pid}: nombre de endpoint inválido «{name}» (letras, números, _ o -).")
    method = str(raw.get("method") or "GET").upper()
    if method not in METHODS:
        raise PluginError(f"{pid}.{name}: método {method} no soportado.")
    path = str(raw.get("path") or "").strip()
    if not path.startswith("/"):
        path = "/" + path
    if "://" in path or ".." in path or path.startswith("//"):
        raise PluginError(f"{pid}.{name}: el path tiene que ser relativo a la URL base.")
    query = raw.get("query") or []
    if isinstance(query, str):
        query = [q for q in re.split(r"[,\s]+", query) if q]
    return {
        "name": name,
        "method": method,
        "path": path,
        "query": [str(q) for q in query if re.match(r"^[A-Za-z0-9_.\[\]-]{1,40}$", str(q))],
        "description": " ".join(str(raw.get("description") or "").split())[:160],
    }


def _mcp(raw: Any, pid: str, auth_type: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PluginError(f"{pid}: mcp tiene que ser un objeto.")
    transport = str(raw.get("transport") or ("stdio" if raw.get("command") else "http")).lower()
    if transport not in TRANSPORTS:
        raise PluginError(f"{pid}: transporte MCP {transport} no soportado (stdio, http o sse).")
    out: dict[str, Any] = {"transport": transport}
    if transport == "stdio":
        command = str(raw.get("command") or "").strip()
        if not command:
            raise PluginError(f"{pid}: falta el comando del servidor MCP.")
        out["command"] = command
        out["args"] = [str(a) for a in raw.get("args") or []]
        out["env"] = _plain_map(raw.get("env"), f"{pid}: env")
        secret_env = {}
        for key, ref in (raw.get("secret_env") or {}).items():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)):
                raise PluginError(f"{pid}: nombre de variable inválido «{key}».")
            secret_env[str(key)] = _ref(ref, pid)
        out["secret_env"] = {k: v for k, v in secret_env.items() if v}
        if auth_type != "none":
            raise PluginError(f"{pid}: un MCP por comando lleva sus tokens como variables (--secret-env).")
    else:
        out["url"] = _http_url(raw.get("url"), pid)
        out["headers"] = _plain_map(raw.get("headers"), f"{pid}: headers")
    return out


def _health(raw: Any, plugin: dict[str, Any]) -> dict[str, Any]:
    pid = plugin["id"]
    if isinstance(raw, str):
        raw = {"url": raw} if "://" in raw else {"endpoint": raw}
    if not isinstance(raw, dict):
        raise PluginError(f"{pid}: health tiene que ser un endpoint o una URL.")
    out: dict[str, Any] = {}
    if raw.get("endpoint"):
        name = str(raw["endpoint"])
        endpoint = next((e for e in plugin.get("endpoints", []) if e["name"] == name), None)
        if endpoint is None or endpoint["method"] != "GET" or _PARAM.search(endpoint["path"]):
            raise PluginError(f"{pid}: el chequeo de salud tiene que ser un endpoint GET sin parámetros.")
        out["endpoint"] = name
    else:
        out["url"] = _http_url(raw.get("url"), f"{pid}: health")
    try:
        out["expect"] = int(raw.get("expect", 200))
        out["interval"] = max(30, int(raw.get("interval", 120)))
    except (TypeError, ValueError):
        raise PluginError(f"{pid}: expect e interval tienen que ser números.") from None
    return out


# --------------------------------------------------------------------------- #
# The registry file
# --------------------------------------------------------------------------- #


def default_path() -> Path:
    return Path(os.environ.get(ENV_FILE) or DEFAULT_FILE).expanduser()


class Store:
    """``plugins_local.json``: ``{"version": 1, "plugins": [...]}``.

    Re-read only when its mtime or size changes, so asking every turn is one
    ``stat``. Saved atomically, readable by the owner only.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        self._lock = threading.RLock()
        self._stamp: tuple[int, int] | None = None
        self._plugins: list[dict[str, Any]] = []
        #: Bumped on every reload that changed something.
        self.generation = 0

    def _current_stamp(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def refresh(self) -> bool:
        """Reload if the file changed on disk. True when it did."""
        with self._lock:
            stamp = self._current_stamp()
            if stamp == self._stamp and self.generation:
                return False
            self._stamp = stamp
            plugins = self._read()
            changed = plugins != self._plugins or not self.generation
            self._plugins = plugins
            if changed:
                self.generation += 1
            return changed

    def _read(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            log.warning("plugins: could not read %s (%s)", self.path, type(exc).__name__)
            return []
        out: list[dict[str, Any]] = []
        ids: set[str] = set()
        for raw in (data.get("plugins") if isinstance(data, dict) else None) or []:
            try:
                plugin = normalize(raw)
            except PluginError as exc:
                log.warning("plugins: skipped an entry: %s", exc)
                continue
            if plugin["id"] in ids:
                log.warning("plugins: duplicate id %s skipped", plugin["id"])
                continue
            ids.add(plugin["id"])
            out.append(plugin)
        return out

    def plugins(self) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            return json.loads(json.dumps(self._plugins))

    def get(self, plugin_id: str) -> dict[str, Any] | None:
        return next((p for p in self.plugins() if p["id"] == plugin_id), None)

    def save(self, plugins: list[dict[str, Any]]) -> None:
        clean = [normalize(p) for p in plugins]
        text = json.dumps({"version": 1, "plugins": clean}, ensure_ascii=False, indent=2) + "\n"
        tmp = self.path.with_name(self.path.name + ".tmp")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            # Force the next refresh to notice, even within one mtime tick.
            self._stamp = None

    def add(self, plugin: dict[str, Any]) -> dict[str, Any]:
        plugin = normalize(plugin)
        plugins = self.plugins()
        if any(p["id"] == plugin["id"] for p in plugins):
            raise PluginError(f"Ya hay un agregado con id «{plugin['id']}».")
        self.save([*plugins, plugin])
        return plugin

    def update(self, plugin_id: str, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        plugins = self.plugins()
        for index, plugin in enumerate(plugins):
            if plugin["id"] == plugin_id:
                change(plugin)
                plugins[index] = normalize(plugin)
                self.save(plugins)
                return plugins[index]
        raise PluginError(f"No hay ningún agregado «{plugin_id}».")

    def set_enabled(self, plugin_id: str, enabled: bool) -> dict[str, Any]:
        return self.update(plugin_id, lambda p: p.__setitem__("enabled", bool(enabled)))

    def remove(self, plugin_id: str, *, forget_secrets: bool = True) -> bool:
        plugins = self.plugins()
        gone = [p for p in plugins if p["id"] == plugin_id]
        if not gone:
            return False
        self.save([p for p in plugins if p["id"] != plugin_id])
        if forget_secrets:
            for ref in _secret_refs(gone[0]):
                if "keychain" in ref:
                    keychain_delete(plugin_id, ref["keychain"])
        return True


# --------------------------------------------------------------------------- #
# REST: one tool per service
# --------------------------------------------------------------------------- #


def rest_tool_name(plugin_id: str) -> str:
    return REST_TOOL_PREFIX + re.sub(r"[^A-Za-z0-9_]", "_", plugin_id)


def is_plugin_tool(name: str) -> bool:
    return name == SERVICES_TOOL or name.startswith(REST_TOOL_PREFIX)


def _visible_endpoints(plugin: dict[str, Any]) -> list[dict[str, Any]]:
    """GET-only unless the owner allowed writes: the rest are not even named."""
    return [e for e in plugin.get("endpoints", [])
            if e["method"] == "GET" or plugin.get("allow_writes")]


def _signature_of(endpoint: dict[str, Any]) -> str:
    names = _PARAM.findall(endpoint["path"]) + [f"{q}?" for q in endpoint["query"]]
    head = f"{endpoint['name']}({','.join(names)})" if names else endpoint["name"]
    if endpoint["method"] != "GET":
        head += f" [{endpoint['method']}]"
    return f"{head}: {endpoint['description']}" if endpoint["description"] else head


def rest_spec(plugin: dict[str, Any]) -> dict[str, Any]:
    """The schema for one REST plugin, with Wawi's local fields
    (``ability``/``example``/``animation``) that :func:`wawi.tools.for_model`
    strips before the model sees it."""
    endpoints = _visible_endpoints(plugin)
    about = plugin["description"] or f"el servicio {plugin['name']}"
    parts = [f"Consultá {plugin['name']} ({about})."]
    if endpoints:
        parts.append("Endpoints: " + "; ".join(_signature_of(e) for e in endpoints) + ".")
        properties: dict[str, Any] = {
            "endpoint": {"type": "string", "enum": [e["name"] for e in endpoints]},
            "params": {"type": "object"},
        }
        required = ["endpoint"]
    else:
        parts.append("GET a un path relativo a su URL base.")
        properties = {"path": {"type": "string"}, "params": {"type": "object"}}
        required = ["path"]
    if any(e["method"] != "GET" for e in endpoints):
        parts.append("Lo que no es GET pide confirmación en pantalla.")
    description = " ".join(parts)
    if len(description) > MAX_TOOL_DESCRIPTION:
        description = description[: MAX_TOOL_DESCRIPTION - 1].rstrip() + "…"
    return {
        "name": rest_tool_name(plugin["id"]),
        "animation": "binoculars",
        "ability": f"Consultar {plugin['name']}",
        "example": f"Wawi, ¿cómo está {plugin['name']}?",
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


SERVICES_SPEC: dict[str, Any] = {
    "name": SERVICES_TOOL,
    "animation": "binoculars",
    "ability": "Contarte qué servicios tuyos tengo conectados",
    "example": "Wawi, ¿qué servicios tenés?",
    "description": "¿Qué servicios tenés?: listá los agregados del dueño (qué son, si están prendidos y si andan).",
    "input_schema": {"type": "object", "properties": {}},
}


@dataclass(frozen=True)
class Request:
    method: str
    url: str
    #: Includes the auth header. Never logged or repr'd.
    headers: dict[str, str] = field(repr=False)
    body: bytes | None = field(default=None, repr=False)
    #: What a log line or the confirmation card may show.
    shown: str = ""


def _auth_headers(plugin: dict[str, Any]) -> dict[str, str]:
    auth = plugin.get("auth") or {}
    if auth.get("type", "none") == "none":
        return {}
    token = resolve_secret(plugin["id"], auth.get("secret"))
    if not token:
        raise PluginError(f"No encuentro el token de {plugin['name']} ({describe_ref(auth.get('secret'))}).")
    if auth["type"] == "bearer":
        return {"Authorization": f"Bearer {token}"}
    return {auth["header"]: token}


def build_request(plugin: dict[str, Any], args: dict[str, Any]) -> Request:
    """Turn a tool call into a request. Raises :class:`PluginError`."""
    params = args.get("params") or {}
    if not isinstance(params, dict):
        raise PluginError("params tiene que ser un objeto.")
    params = {str(k): v for k, v in params.items() if v is not None}
    name = str(args.get("endpoint") or "").strip()
    endpoints = plugin.get("endpoints", [])
    if endpoints or name:
        endpoint = next((e for e in endpoints if e["name"] == name), None)
        if endpoint is None:
            raise PluginError(f"{plugin['name']} no tiene el endpoint «{name}».")
        if endpoint["method"] != "GET" and not plugin.get("allow_writes"):
            raise PluginError(f"{plugin['name']} está en solo lectura: no puedo usar {endpoint['method']}.")
        method, path = endpoint["method"], endpoint["path"]
    else:
        method = "GET"
        path = "/" + str(args.get("path") or "").strip().lstrip("/")
        if "://" in path or ".." in path or "\\" in path:
            raise PluginError("El path tiene que ser relativo a la URL del servicio.")

    def _fill(match: re.Match) -> str:
        key = match.group(1)
        if key not in params:
            raise PluginError(f"Falta el parámetro «{key}».")
        return urllib.parse.quote(str(params.pop(key)), safe="")

    path = _PARAM.sub(_fill, path)
    url = plugin["base_url"] + path
    body = None
    rest = {k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False)
            for k, v in params.items()}
    if method == "GET" or method == "DELETE":
        if rest:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(rest)
    elif params:
        body = json.dumps(params, ensure_ascii=False).encode("utf-8")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/plain;q=0.9, */*;q=0.5",
               **plugin.get("headers", {}), **_auth_headers(plugin)}
    if body is not None:
        headers["Content-Type"] = "application/json"
    shown = f"{method} {urllib.parse.urlsplit(url).path or '/'}"
    return Request(method, url, headers, body, shown)


#: urllib's opener, swapped out by the tests: no test ever hits the network.
_urlopen: Callable[..., Any] = urllib.request.urlopen


@dataclass(frozen=True)
class Response:
    status: int | None
    body: bytes = b""
    content_type: str = ""
    error: str = ""
    latency_ms: int = 0


def send(request: Request, *, timeout: float = TIMEOUT_S) -> Response:
    """Never raises; never puts a header in an error message."""
    start = time.monotonic()
    req = urllib.request.Request(request.url, data=request.body, method=request.method,
                                 headers=request.headers)
    try:
        with _urlopen(req, timeout=timeout) as resp:
            body = resp.read(MAX_BODY_BYTES)
            ctype = resp.headers.get("Content-Type", "") if getattr(resp, "headers", None) else ""
            return Response(resp.status, body, ctype, "", _ms(start))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(16_000)
        except Exception:
            body = b""
        return Response(exc.code, body, "", f"HTTP {exc.code}", _ms(start))
    except Exception as exc:  # timeout, DNS, TLS, refused…
        reason = getattr(exc, "reason", None) or exc
        text = " ".join(str(reason).split())[:120] or type(exc).__name__
        return Response(None, b"", "", text, _ms(start))


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def shape(value: Any, depth: int = 0) -> Any:
    """Trim a decoded JSON value so the useful part fits: long lists keep
    their head, long strings are clipped, deep nesting is cut."""
    if depth >= 6:
        return "…"
    if isinstance(value, dict):
        items = list(value.items())
        out = {str(k): shape(v, depth + 1) for k, v in items[:40]}
        if len(items) > 40:
            out["…"] = f"{len(items) - 40} claves más"
        return out
    if isinstance(value, list):
        out_list = [shape(v, depth + 1) for v in value[:20]]
        if len(value) > 20:
            out_list.append(f"… {len(value) - 20} más")
        return out_list
    if isinstance(value, str) and len(value) > 400:
        return value[:399] + "…"
    return value


def render_body(raw: bytes, limit: int = MAX_RESULT_CHARS) -> tuple[str, bool]:
    """(text for the model, was it truncated)."""
    text = raw.decode("utf-8", "replace").strip()
    try:
        out = json.dumps(shape(json.loads(text)), ensure_ascii=False, indent=1)
    except ValueError:
        if "<html" in text[:500].lower():
            text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
            text = re.sub(r"<[^>]+>", " ", text)
        out = " ".join(text.split())
    if len(out) <= limit:
        return out, False
    return out[: limit - 1].rstrip() + "…", True


# --------------------------------------------------------------------------- #
# On-screen confirmation for writes
# --------------------------------------------------------------------------- #


def confirm_on_screen(title: str, detail: str, *, timeout: float = CONFIRM_TIMEOUT_S) -> bool:
    """Show Wawi's Aceptar/Cancelar card and wait for a click. Blocking: call
    it from a worker thread. No presenter (headless, tests) means no."""
    from wawi.confirm import REGISTRY

    if REGISTRY.presenter is None:
        log.info("plugins: no confirmation panel; write refused")
        return False
    done = threading.Event()
    answer: list[bool] = []

    def _yes() -> None:
        answer.append(True)
        done.set()

    def _no() -> None:
        answer.append(False)
        done.set()

    ident = REGISTRY.add(title[:45], detail[:95], "plugins", _yes, _no, timeout=timeout)
    if not done.wait(timeout + 1.0):
        REGISTRY.discard(ident)
        return False
    return bool(answer and answer[0])


def _confirm(title: str, detail: str) -> bool:
    """Indirection the tests replace."""
    return confirm_on_screen(title, detail)


def _clip_args(args: dict[str, Any], limit: int = 60) -> str:
    parts = []
    for key in sorted(args):
        value = args[key]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        text = " ".join(str(text).split())
        parts.append(f"{key}={text[:limit]}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# The manager: what the brain, the tools and the window ask
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PluginServer:
    """An MCP plugin ready for :meth:`wawi.connectors.Connectors.set_plugins`.
    ``config`` holds resolved secrets and is kept out of ``repr``."""

    prefix: str
    name: str
    transport: str
    config: dict[str, Any] = field(repr=False, compare=False)
    allow_writes: bool = False
    description: str = ""


def mcp_prefix(plugin_id: str) -> str:
    return MCP_PREFIX + re.sub(r"[^A-Za-z0-9_]", "_", plugin_id)


def mcp_config(plugin: dict[str, Any]) -> dict[str, Any]:
    """The SDK's server config, secrets resolved. Raises PluginError."""
    spec = plugin["mcp"]
    if spec["transport"] == "stdio":
        env = dict(spec.get("env", {}))
        for key, ref in spec.get("secret_env", {}).items():
            value = resolve_secret(plugin["id"], ref)
            if not value:
                raise PluginError(f"No encuentro {key} de {plugin['name']} ({describe_ref(ref)}).")
            env[key] = value
        config: dict[str, Any] = {"type": "stdio", "command": spec["command"], "args": list(spec.get("args", []))}
        if env:
            config["env"] = env
        return config
    headers = {**spec.get("headers", {}), **_auth_headers(plugin)}
    config = {"type": spec["transport"], "url": spec["url"]}
    if headers:
        config["headers"] = headers
    return config


class Manager:
    """The live view of the registry. One per run; any thread."""

    def __init__(self, store: Store | None = None) -> None:
        self.store = store or Store()
        self._lock = threading.Lock()
        self._servers: list[PluginServer] = []
        self._servers_generation = -1
        #: id -> "up" / "down", written by the health sensor.
        self.health: dict[str, str] = {}

    # -- reading --------------------------------------------------------
    def plugins(self) -> list[dict[str, Any]]:
        return self.store.plugins()

    def enabled(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [p for p in self.plugins() if p["enabled"] and (kind is None or p["kind"] == kind)]

    def find_tool(self, tool_name: str) -> dict[str, Any] | None:
        return next((p for p in self.enabled("rest") if rest_tool_name(p["id"]) == tool_name), None)

    # -- what the model gets ----------------------------------------------
    def tool_specs(self) -> list[dict[str, Any]]:
        """``my_services`` while any plugin exists, plus one tool per enabled
        REST plugin. Stable text: it is part of the cached prompt."""
        plugins = self.plugins()
        if not plugins:
            return []
        return [SERVICES_SPEC, *(rest_spec(p) for p in plugins if p["enabled"] and p["kind"] == "rest")]

    def signature(self) -> tuple:
        """Changes exactly when what the brain is given changes."""
        out = []
        for p in self.plugins():
            if not p["enabled"]:
                out.append((p["id"], "off"))
                continue
            digest = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]
            out.append((p["id"], digest))
        return tuple(out)

    def mcp_servers(self) -> list[PluginServer]:
        """Enabled MCP plugins with their secrets resolved. Cached per
        registry generation, so the Keychain is read once per change."""
        self.store.refresh()
        with self._lock:
            if self._servers_generation == self.store.generation:
                return list(self._servers)
            servers = []
            for plugin in self.enabled("mcp"):
                try:
                    config = mcp_config(plugin)
                except PluginError as exc:
                    log.warning("plugins: %s skipped: %s", plugin["id"], exc)
                    continue
                servers.append(PluginServer(mcp_prefix(plugin["id"]), plugin["name"],
                                            plugin["mcp"]["transport"], config,
                                            plugin["allow_writes"], plugin["description"]))
            self._servers, self._servers_generation = servers, self.store.generation
            return list(servers)

    def ability_entries(self) -> list[tuple[str, str, str]]:
        """show_abilities rows for MCP plugins (REST ones come from their
        tool spec's own ability/example)."""
        return [(mcp_prefix(p["id"]), f"Consultar {p['name']}", f"Wawi, ¿cómo está {p['name']}?")
                for p in self.enabled("mcp")]

    # -- calls ---------------------------------------------------------------
    def dispatch(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, str, dict]:
        """Run ``svc_<id>`` or ``my_services``: (ok, speak, data)."""
        if tool_name == SERVICES_TOOL:
            return self.services()
        plugin = self.find_tool(tool_name)
        if plugin is None:
            return False, "Ese servicio ya no está conectado.", {"error": "unknown_service"}
        return call_rest(plugin, args)

    def services(self) -> tuple[bool, str, dict]:
        plugins = self.plugins()
        if not plugins:
            return True, "No tengo servicios agregados todavía.", {"services": []}
        rows = []
        for p in plugins:
            row = {"name": p["name"], "kind": p["kind"], "enabled": p["enabled"],
                   "description": p["description"], "writes": p["allow_writes"]}
            if p["id"] in self.health:
                row["health"] = self.health[p["id"]]
            if p["kind"] == "rest":
                row["endpoints"] = [e["name"] for e in _visible_endpoints(p)]
            rows.append(row)
        on = [p["name"] for p in plugins if p["enabled"]]
        speak = ("Tengo " + ", ".join(on) + ".") if on else "Tengo agregados, pero están apagados."
        return True, speak, {"services": rows}


def call_rest(plugin: dict[str, Any], args: dict[str, Any]) -> tuple[bool, str, dict]:
    """One REST tool call, end to end. Errors are one short Spanish line."""
    name = plugin["name"]
    try:
        request = build_request(plugin, args)
    except PluginError as exc:
        return False, str(exc), {"error": "bad_request"}
    if request.method != "GET":
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        if not _confirm(f"{name}: {request.method}", f"{request.shown} {_clip_args(params)}".strip()):
            log.info("plugins: %s %s not confirmed; nothing sent", plugin["id"], request.shown)
            return False, "No lo mandé: no lo confirmaste en pantalla.", {"error": "not_confirmed"}
    response = send(request)
    log.info("plugins: %s %s -> %s (%d ms)", plugin["id"], request.shown,
             response.status if response.status is not None else "no answer", response.latency_ms)
    if response.status is None:
        return False, f"No pude conectar con {name}.", {"error": "unreachable", "detail": response.error}
    body, truncated = render_body(response.body)
    data = {"service": name, "status": response.status, "body": body}
    if truncated:
        data["truncated"] = True
    if response.status >= 400:
        data["error"] = f"HTTP {response.status}"
        data["body"] = body[:600]
        return False, f"{name} contestó con error {response.status}.", data
    return True, f"{name} respondió.", data


# --------------------------------------------------------------------------- #
# Health: feeds the sensors, so a dead service gets the firefighter
# --------------------------------------------------------------------------- #


def health_request(plugin: dict[str, Any]) -> Request | None:
    """The check's request, or None when the plugin has none. Auth headers
    only go to the plugin's own base URL."""
    health = plugin.get("health")
    if not health:
        return None
    if health.get("endpoint"):
        return build_request(plugin, {"endpoint": health["endpoint"]})
    url = health["url"]
    headers = {"User-Agent": USER_AGENT}
    base = plugin.get("base_url") or (plugin.get("mcp") or {}).get("url") or ""
    if base and url.startswith(base):
        headers.update(_auth_headers(plugin))
    return Request("GET", url, headers, None, f"GET {urllib.parse.urlsplit(url).path or '/'}")


def check_health(plugin: dict[str, Any]) -> tuple[bool, str, str]:
    """(healthy, short detail, url for the event). Never raises."""
    try:
        request = health_request(plugin)
    except PluginError as exc:
        return False, str(exc), plugin.get("base_url", "")
    if request is None:
        return True, "sin chequeo", ""
    response = send(request)
    url = request.url.split("?", 1)[0]
    expect = int(plugin["health"].get("expect", 200))
    if response.status is None:
        return False, response.error or "sin respuesta", url
    ok = response.status == expect or (expect == 200 and 200 <= response.status < 300)
    return ok, f"HTTP {response.status} en {response.latency_ms} ms", url


class PluginHealthSensor(Sensor):
    """Polls each enabled plugin's ``health`` on its own interval and emits
    the SAME events as the web sensor (``web.down`` / ``web.up``), so the
    existing reactions run: firefighter, bark, and the celebration when it is
    back. Re-reads the registry on every poll, so a check added from the
    terminal starts without anyone touching the app."""

    name = "plugins"
    default_interval = 30.0

    def __init__(self, manager: Manager, config: dict[str, Any] | None = None, *,
                 check: Callable[[dict[str, Any]], tuple[bool, str, str]] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 cancel: Callable[[str], None] | None = None) -> None:
        super().__init__(config)
        self.manager = manager
        self._check = check or check_health
        self._clock = clock
        self._cancel = cancel or (lambda key: BUS.push(Reaction(cancel_key=key, priority=0)))
        self._deb = Debouncer(fail_after=2, recover_after=1, alert_initial=True)
        self._due: dict[str, float] = {}
        self._keys: dict[str, str] = {}

    def poll(self) -> list[Event]:
        now = self._clock()
        events: list[Event] = []
        watched = {p["id"]: p for p in self.manager.enabled() if p.get("health")}
        gone = [pid for pid in self._keys if pid not in watched]
        for pid in gone:
            # Removed or switched off while down: whatever she was doing
            # about it stops with it.
            if self._deb.state(pid) == "down":
                self._cancel(self._keys[pid])
            self._keys.pop(pid, None)
            self._due.pop(pid, None)
            self.manager.health.pop(pid, None)
        if gone:
            self._deb.forget(list(watched))
        for pid, plugin in watched.items():
            if now < self._due.get(pid, 0.0):
                continue
            self._due[pid] = now + float(plugin["health"].get("interval", 120))
            healthy, detail, url = self._check(plugin)
            key = f"web:{url or pid}"
            self._keys[pid] = key
            change = self._deb.update(pid, not healthy)
            state = self._deb.state(pid)
            if state:
                self.manager.health[pid] = state
            if change is None:
                continue
            data = {"url": url or pid, "host": plugin["name"], "plugin": pid, "detail": detail}
            if change == "down":
                events.append(Event("web.down", "urgent", key, f"Se cayó {plugin['name']}", data))
            else:
                events.append(Event("web.up", "notice", key, f"{plugin['name']} volvió", data))
        return events

    def summary(self) -> str | None:
        watched = [p for p in self.manager.enabled() if p.get("health")]
        if not watched:
            return None
        down = [p["name"] for p in watched if self._deb.state(p["id"]) == "down"]
        if not down:
            return "tus agregados andan"
        return f"está caído {down[0]}" if len(down) == 1 else "están caídos " + ", ".join(down)


# --------------------------------------------------------------------------- #
# Module-level wiring (installed by the orchestrator)
# --------------------------------------------------------------------------- #

_MANAGER: Manager | None = None


def build(path: str | Path | None = None) -> Manager:
    return Manager(Store(path))


def install(manager: Manager | None) -> None:
    global _MANAGER
    _MANAGER = manager


def get() -> Manager | None:
    return _MANAGER


def signature() -> tuple:
    manager = _MANAGER
    return manager.signature() if manager is not None else ()


def sync(connectors: Any) -> None:
    """Before a turn: pick up registry changes and hand the MCP plugins to
    the connectors' gate. Never raises — a broken file costs the plugins,
    not the answer."""
    manager = _MANAGER
    if manager is None or connectors is None:
        return
    try:
        connectors.set_plugins(manager.mcp_servers())
    except Exception:
        log.warning("plugins: could not sync the MCP plugins", exc_info=True)


def attach_health() -> bool:
    """Install the health sensor in the running sensors. False when sensors
    are off or no manager is installed."""
    manager = _MANAGER
    if manager is None:
        return False
    from wawi import sensors

    runner = getattr(sensors, "_RUNNER", None)
    if runner is None:
        return False
    runner.replace(PluginHealthSensor(manager))
    return True


# --------------------------------------------------------------------------- #
# Terminal: python -m wawi.plugins … (scripts/wawictl plugins …)
# --------------------------------------------------------------------------- #


def _line(plugin: dict[str, Any]) -> str:
    state = "on " if plugin["enabled"] else "off"
    where = plugin.get("base_url") or (plugin.get("mcp") or {}).get("url") \
        or (plugin.get("mcp") or {}).get("command", "")
    writes = " +escritura" if plugin["allow_writes"] else ""
    return f"[{state}] {plugin['id']:<16} {plugin['kind']:<4} {plugin['name']} — {where}{writes}"


def describe(plugin: dict[str, Any]) -> list[str]:
    """Every field, secrets as references only. For `show` and the window."""
    lines = [_line(plugin)]
    if plugin["description"]:
        lines.append(f"  descripción: {plugin['description']}")
    auth = plugin.get("auth") or {}
    if auth.get("type", "none") != "none":
        header = f" ({auth['header']})" if auth.get("header") else ""
        lines.append(f"  auth: {auth['type']}{header}, token en {describe_ref(auth.get('secret'))}")
    for key, value in (plugin.get("headers") or {}).items():
        lines.append(f"  header {key}: {value}")
    for endpoint in plugin.get("endpoints", []):
        hidden = "" if endpoint["method"] == "GET" or plugin["allow_writes"] else " (oculto: solo lectura)"
        lines.append(f"  endpoint {endpoint['method']} {endpoint['name']} {endpoint['path']}{hidden}"
                     + (f" — {endpoint['description']}" if endpoint["description"] else ""))
    mcp = plugin.get("mcp") or {}
    if mcp.get("args"):
        lines.append("  args: " + " ".join(shlex.quote(a) for a in mcp["args"]))
    for key in (mcp.get("env") or {}):
        lines.append(f"  env {key}=…")
    for key, ref in (mcp.get("secret_env") or {}).items():
        lines.append(f"  env {key} desde {describe_ref(ref)}")
    health = plugin.get("health")
    if health:
        target = health.get("endpoint") or health.get("url")
        lines.append(f"  salud: {target} espera {health['expect']} cada {health['interval']} s")
    return lines


def test_plugin(plugin: dict[str, Any]) -> tuple[bool, str]:
    """`test`: the health check, else the first parameterless GET, else (MCP)
    the tool list. A short line, no secrets."""
    if plugin.get("health"):
        ok, detail, _ = check_health(plugin)
        return ok, ("Anda: " if ok else "No anda: ") + detail
    if plugin["kind"] == "rest":
        endpoint = next((e for e in plugin.get("endpoints", [])
                         if e["method"] == "GET" and not _PARAM.search(e["path"])), None)
        args = {"endpoint": endpoint["name"]} if endpoint else {"path": "/"}
        if not endpoint and plugin.get("endpoints"):
            return False, "No hay un endpoint GET sin parámetros para probar."
        ok, speak, data = call_rest(plugin, args)
        if not ok:
            return False, f"{speak} {data.get('detail', '')}".strip()
        body = str(data.get("body", ""))
        return True, f"HTTP {data['status']}: {' '.join(body.split())[:240]}"
    return list_mcp_tools(plugin)


def list_mcp_tools(plugin: dict[str, Any], *, timeout: float = 25.0) -> tuple[bool, str]:
    """Start the server, list its tools, classify them. Uses the ``mcp``
    package the SDK already depends on."""
    import asyncio

    try:
        config = mcp_config(plugin)
    except PluginError as exc:
        return False, str(exc)

    async def _list() -> list[Any]:
        from mcp import ClientSession  # type: ignore[import]

        async def _session(read, write) -> list[Any]:
            async with ClientSession(read, write) as session:
                await session.initialize()
                return list((await session.list_tools()).tools)

        if config["type"] == "stdio":
            from mcp import StdioServerParameters  # type: ignore[import]
            from mcp.client.stdio import stdio_client  # type: ignore[import]

            env = {**os.environ, **config.get("env", {})}
            params = StdioServerParameters(command=config["command"], args=config.get("args", []), env=env)
            with open(os.devnull, "w") as devnull:
                async with stdio_client(params, errlog=devnull) as (read, write):
                    return await _session(read, write)
        if config["type"] == "sse":
            from mcp.client.sse import sse_client  # type: ignore[import]

            async with sse_client(config["url"], headers=config.get("headers")) as (read, write):
                return await _session(read, write)
        from mcp.client.streamable_http import streamable_http_client  # type: ignore[import]
        from mcp.shared._httpx_utils import create_mcp_http_client  # type: ignore[import]

        async with create_mcp_http_client(headers=config.get("headers")) as client:
            async with streamable_http_client(config["url"], http_client=client) as streams:
                return await _session(streams[0], streams[1])

    try:
        found = asyncio.run(asyncio.wait_for(_list(), timeout))
    except Exception as exc:
        return False, f"No pude hablar con el servidor MCP ({type(exc).__name__})."
    reads, writes = [], []
    for tool in found:
        notes = getattr(tool, "annotations", None)
        notes = notes.model_dump(exclude_none=True) if hasattr(notes, "model_dump") else dict(notes or {})
        (reads if plugin_tool_is_read(tool.name, notes) else writes).append(tool.name)
    line = f"{len(found)} herramienta(s). Lectura: {', '.join(reads) or 'ninguna'}."
    if writes:
        state = "piden confirmación" if plugin["allow_writes"] else "ocultas"
        line += f" Escritura ({state}): {', '.join(writes)}."
    return True, line


def _parse_pairs(items: list[str] | None, what: str) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise PluginError(f"{what}: usá NOMBRE=valor.")
        key, value = item.split("=", 1)
        out[key.strip()] = value
    return out


def _cli_add(store: Store, ns: argparse.Namespace, *, read_token: Callable[[str], str]) -> dict[str, Any]:
    kinds = [x for x in (ns.rest, ns.mcp_cmd, ns.mcp_url) if x]
    if len(kinds) != 1:
        raise PluginError("Elegí uno: --rest URL, --mcp-cmd \"comando\" o --mcp-url URL.")
    pid = (ns.id or slug(ns.name)).lower()
    if ns.token and ns.token_env:
        raise PluginError("Usá --token o --token-env, no los dos.")
    auth_type = ns.auth or ("bearer" if (ns.token or ns.token_env) else "none")
    raw: dict[str, Any] = {"id": pid, "name": ns.name, "description": ns.desc or "",
                           "enabled": not ns.disabled, "allow_writes": ns.allow_writes,
                           "auth": {"type": auth_type, "header": ns.header_name or ""}}
    token_value = None
    if auth_type != "none":
        if ns.token_env:
            raw["auth"]["secret"] = {"env": ns.token_env}
        else:
            raw["auth"]["secret"] = {"keychain": "token"}
            token_value = read_token(f"Token para {ns.name} (no se muestra): ")
    secret_values: dict[str, str] = {}
    if ns.rest:
        raw.update(kind="rest", base_url=ns.rest, headers=_parse_pairs(ns.header, "--header"))
    elif ns.mcp_cmd:
        parts = shlex.split(ns.mcp_cmd)
        if not parts:
            raise PluginError("--mcp-cmd está vacío.")
        secret_env = {}
        for name in ns.secret_env or []:
            secret_env[name] = {"keychain": name}
            secret_values[name] = read_token(f"Valor de {name} (no se muestra): ")
        raw.update(kind="mcp", mcp={"transport": "stdio", "command": parts[0], "args": parts[1:],
                                    "env": _parse_pairs(ns.env, "--env"), "secret_env": secret_env})
    else:
        raw.update(kind="mcp", mcp={"transport": "sse" if ns.sse else "http", "url": ns.mcp_url,
                                    "headers": _parse_pairs(ns.header, "--header")})
    if ns.health:
        if "://" not in ns.health:
            raise PluginError("--health es una URL completa; para usar un endpoint: "
                              "endpoint add <id> … --health.")
        raw["health"] = {"url": ns.health, "expect": ns.expect, "interval": ns.interval}
    plugin = normalize(raw)
    # Secrets go in BEFORE the JSON: a plugin is never saved pointing at a
    # Keychain item that does not exist.
    if token_value is not None:
        keychain_set(pid, "token", token_value)
    for name, value in secret_values.items():
        keychain_set(pid, name, value)
    return store.add(plugin)


def _getpass(prompt: str) -> str:
    import getpass

    value = getpass.getpass(prompt).strip()
    if not value:
        raise PluginError("No ingresaste nada.")
    return value


def main(argv: list[str] | None = None, *, read_token: Callable[[str], str] = _getpass,
         out: Any = None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="wawictl plugins",
                                     description="Agregados: servicios tuyos que Wawi puede leer.")
    parser.add_argument("--file", help=f"registro (por defecto {DEFAULT_FILE} o ${ENV_FILE})")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="listar agregados")
    for name in ("remove", "enable", "disable", "show", "test", "set-token"):
        sub.add_parser(name).add_argument("id")
    add = sub.add_parser("add", help="agregar un servicio")
    add.add_argument("name")
    add.add_argument("--id")
    add.add_argument("--desc", help="qué es el servicio (lo lee el modelo)")
    add.add_argument("--rest", metavar="URL", help="URL base de una API REST")
    add.add_argument("--mcp-cmd", metavar="CMD", help='servidor MCP por comando, p. ej. "npx -y foo-mcp"')
    add.add_argument("--mcp-url", metavar="URL", help="servidor MCP por http(s)")
    add.add_argument("--sse", action="store_true", help="--mcp-url es SSE, no HTTP")
    add.add_argument("--auth", choices=AUTH_TYPES)
    add.add_argument("--header-name", help="encabezado del token con --auth header")
    add.add_argument("--token", action="store_true", help="pedir el token y guardarlo en el Llavero")
    add.add_argument("--token-env", metavar="VAR", help="leer el token de esta variable de entorno")
    add.add_argument("--secret-env", action="append", metavar="NAME",
                     help="variable secreta del MCP; se pide y va al Llavero (repetible)")
    add.add_argument("--env", action="append", metavar="K=V", help="variable no secreta del MCP")
    add.add_argument("--header", action="append", metavar="K=V", help="encabezado no secreto")
    add.add_argument("--allow-writes", action="store_true", help="permitir escrituras (con confirmación)")
    add.add_argument("--disabled", action="store_true")
    add.add_argument("--health", metavar="URL", help="URL de salud")
    add.add_argument("--expect", type=int, default=200)
    add.add_argument("--interval", type=int, default=120)
    endpoint = sub.add_parser("endpoint", help="endpoints de un agregado REST")
    esub = endpoint.add_subparsers(dest="ecmd", required=True)
    eadd = esub.add_parser("add")
    eadd.add_argument("id")
    eadd.add_argument("--name", required=True)
    eadd.add_argument("--path", required=True)
    eadd.add_argument("--method", default="GET")
    eadd.add_argument("--desc", default="")
    eadd.add_argument("--query", default="", help="parámetros de query, separados por coma")
    eadd.add_argument("--health", action="store_true", help="usarlo como chequeo de salud")
    erm = esub.add_parser("remove")
    erm.add_argument("id")
    erm.add_argument("name")
    ns = parser.parse_args(argv)
    store = Store(ns.file)

    def say(text: str) -> None:
        print(text, file=out)

    try:
        if ns.cmd == "list":
            plugins = store.plugins()
            if not plugins:
                say(f"No hay agregados todavía ({store.path}).")
            for plugin in plugins:
                say(_line(plugin))
        elif ns.cmd == "add":
            plugin = _cli_add(store, ns, read_token=read_token)
            say(f"Agregado: {_line(plugin)}")
            say("Wawi lo toma en su próxima pregunta; no hace falta reiniciar.")
        elif ns.cmd == "remove":
            if not store.remove(ns.id):
                raise PluginError(f"No hay ningún agregado «{ns.id}».")
            say(f"Quitado: {ns.id} (y sus tokens del Llavero).")
        elif ns.cmd in ("enable", "disable"):
            plugin = store.set_enabled(ns.id, ns.cmd == "enable")
            say(_line(plugin))
        elif ns.cmd == "show":
            plugin = store.get(ns.id)
            if plugin is None:
                raise PluginError(f"No hay ningún agregado «{ns.id}».")
            for line in describe(plugin):
                say(line)
        elif ns.cmd == "test":
            plugin = store.get(ns.id)
            if plugin is None:
                raise PluginError(f"No hay ningún agregado «{ns.id}».")
            ok, line = test_plugin(plugin)
            say(("OK  " if ok else "MAL ") + line)
            return 0 if ok else 1
        elif ns.cmd == "set-token":
            plugin = store.get(ns.id)
            if plugin is None:
                raise PluginError(f"No hay ningún agregado «{ns.id}».")
            secret = (plugin.get("auth") or {}).get("secret") or {}
            if "keychain" not in secret:
                raise PluginError(f"{ns.id} no guarda su token en el Llavero ({describe_ref(secret or None)}).")
            keychain_set(ns.id, secret["keychain"], read_token(f"Token nuevo para {plugin['name']}: "))
            # Re-save so the running Wawi notices and reconnects with it.
            store.update(ns.id, lambda p: None)
            say("Token actualizado en el Llavero.")
        elif ns.cmd == "endpoint":
            if ns.ecmd == "add":
                new = {"name": ns.name, "path": ns.path, "method": ns.method,
                       "description": ns.desc, "query": ns.query}

                def _add(p: dict[str, Any]) -> None:
                    if p["kind"] != "rest":
                        raise PluginError(f"{p['id']} no es REST.")
                    p["endpoints"] = [e for e in p["endpoints"] if e["name"] != ns.name] + [new]
                    if ns.health:
                        p["health"] = {**(p.get("health") or {}), "endpoint": ns.name}
                        p["health"].pop("url", None)

                plugin = store.update(ns.id, _add)
                say(f"Endpoint {ns.name} listo en {plugin['id']}.")
            else:
                def _remove(p: dict[str, Any]) -> None:
                    before = len(p.get("endpoints", []))
                    p["endpoints"] = [e for e in p.get("endpoints", []) if e["name"] != ns.name]
                    if len(p["endpoints"]) == before:
                        raise PluginError(f"{p['id']} no tiene el endpoint «{ns.name}».")
                    if (p.get("health") or {}).get("endpoint") == ns.name:
                        p.pop("health")

                store.update(ns.id, _remove)
                say(f"Endpoint {ns.name} quitado.")
    except PluginError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
