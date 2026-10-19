"""The ``webvm:`` section of config.yaml: where her little computer runs.

Pure parsing, no I/O. Every key is optional; :data:`DEFAULTS` documents them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Kinds of action she must ask the owner about before doing (wawi/webvm/safety.py).
ASK_KINDS = ("login", "purchase", "payment", "message", "post", "personal_data")

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    # `docker --context <context>` runs every lifecycle command. Yours, in
    # private/config.yaml: there is no sensible default for someone's server.
    "context": "",
    # The LAN address the ports are published on, and that Wawi connects to.
    "host": "",
    # For `wawictl webvm build`: ssh alias (default: the context) and the
    # project dir there.
    "ssh_host": "",
    "remote_dir": "~/work/wawi-webvm",
    "api_port": 6180,
    "view_port": 6181,
    "image": "wawi-webvm:1",
    "container": "wawi-webvm",
    # Stop the container after this many minutes without use.
    "idle_stop_min": 15,
    # Web tool calls allowed in ONE voice turn (on top of the usual budget).
    "max_steps": 20,
    "memory": "2g",
    "cpus": "2",
    "shm_size": "1g",
    "locale": "es-AR",
    "timezone": "America/Argentina/Buenos_Aires",
    # Keep cookies between sessions (a named volume). Off: a fresh profile
    # every start.
    "persist_profile": False,
    # Let her browser open LAN / localhost addresses. Off by default.
    "allow_lan": False,
    "ask_before": list(ASK_KINDS),
    # The small view beside her: show it while she browses.
    "show_view": True,
}


class SettingsError(ValueError):
    """config.yaml has a webvm value that cannot work."""


def _port(value: Any, name: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise SettingsError(f"webvm.{name} tiene que ser un número de puerto") from None
    if not 1 <= port <= 65535:
        raise SettingsError(f"webvm.{name} tiene que estar entre 1 y 65535")
    return port


@dataclass(frozen=True)
class WebVMSettings:
    enabled: bool = False
    context: str = ""
    host: str = ""
    ssh_host: str = ""
    remote_dir: str = "~/work/wawi-webvm"
    api_port: int = 6180
    view_port: int = 6181
    image: str = "wawi-webvm:1"
    container: str = "wawi-webvm"
    idle_stop_min: float = 15.0
    max_steps: int = 20
    memory: str = "2g"
    cpus: str = "2"
    shm_size: str = "1g"
    locale: str = "es-AR"
    timezone: str = "America/Argentina/Buenos_Aires"
    persist_profile: bool = False
    allow_lan: bool = False
    ask_before: tuple[str, ...] = field(default_factory=lambda: ASK_KINDS)
    show_view: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WebVMSettings":
        if raw is not None and not isinstance(raw, dict):
            raise SettingsError("webvm tiene que ser una sección (mapping)")
        merged = {**DEFAULTS, **(raw or {})}
        unknown = set(merged) - set(DEFAULTS)
        if unknown:
            raise SettingsError(f"webvm: claves desconocidas: {', '.join(sorted(unknown))}")
        ask = merged["ask_before"]
        if not isinstance(ask, (list, tuple)):
            raise SettingsError("webvm.ask_before tiene que ser una lista")
        bad = [k for k in ask if k not in ASK_KINDS]
        if bad:
            raise SettingsError(f"webvm.ask_before: no conozco {', '.join(map(str, bad))} "
                                f"(valen: {', '.join(ASK_KINDS)})")
        try:
            idle = float(merged["idle_stop_min"])
            steps = int(merged["max_steps"])
        except (TypeError, ValueError):
            raise SettingsError("webvm.idle_stop_min y webvm.max_steps son números") from None
        if idle <= 0:
            raise SettingsError("webvm.idle_stop_min tiene que ser mayor que cero")
        if not 1 <= steps <= 60:
            raise SettingsError("webvm.max_steps tiene que estar entre 1 y 60")
        required = ("context", "host", "image", "container") if merged["enabled"] \
            else ("image", "container")
        for key in required:
            if not str(merged[key] or "").strip():
                raise SettingsError(f"webvm.{key} no puede estar vacío")
        api_port = _port(merged["api_port"], "api_port")
        view_port = _port(merged["view_port"], "view_port")
        if api_port == view_port:
            raise SettingsError("webvm.api_port y webvm.view_port tienen que ser distintos")
        return cls(
            enabled=bool(merged["enabled"]),
            context=str(merged["context"]).strip(),
            host=str(merged["host"]).strip(),
            ssh_host=str(merged["ssh_host"] or merged["context"]).strip(),
            remote_dir=str(merged["remote_dir"]).strip(),
            api_port=api_port,
            view_port=view_port,
            image=str(merged["image"]).strip(),
            container=str(merged["container"]).strip(),
            idle_stop_min=idle,
            max_steps=steps,
            memory=str(merged["memory"]),
            cpus=str(merged["cpus"]),
            shm_size=str(merged["shm_size"]),
            locale=str(merged["locale"]),
            timezone=str(merged["timezone"]),
            persist_profile=bool(merged["persist_profile"]),
            allow_lan=bool(merged["allow_lan"]),
            ask_before=tuple(str(k) for k in ask),
            show_view=bool(merged["show_view"]),
        )

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.api_port}"

    @property
    def view_url(self) -> str:
        return f"http://{self.host}:{self.view_port}/"
