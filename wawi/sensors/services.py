"""Local services: processes by name, TCP ports, Docker containers.

The first reading of each service is a silent baseline: a dev server that is
not running when Wawi starts was never "up", so it cannot have "gone down".
Only transitions after that are events.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
from typing import Any, Callable

from wawi.reactions import Event
from wawi.sensors.base import Debouncer, Sensor, as_list

_KIND_ES = {"process": "proceso", "port": "puerto", "container": "contenedor"}


def list_processes() -> set[str]:
    """Executable basenames of every running process (no arguments read)."""
    try:
        out = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip().rsplit("/", 1)[-1] for line in out.stdout.splitlines() if line.strip()}


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def docker_ps(docker: str = "docker", timeout: float = 8.0) -> dict[str, str] | None:
    """{container name: state} or None when Docker is missing or not running."""
    exe = shutil.which(docker)
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "ps", "-a", "--format", "{{json .}}"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None  # daemon not running: unknown, not "everything down"
    return parse_docker_ps(out.stdout)


def parse_docker_ps(text: str) -> dict[str, str]:
    containers: dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        name = str(row.get("Names") or row.get("Name") or "").split(",")[0].strip()
        if not name:
            continue
        state = str(row.get("State") or "").lower()
        if not state:
            state = "running" if str(row.get("Status", "")).lower().startswith("up") else "exited"
        containers[name] = state
    return containers


def parse_port(spec: Any) -> tuple[str, int] | None:
    """3000 / "3000" / "localhost:3000" / {"host":..., "port":...}."""
    if isinstance(spec, dict):
        try:
            return str(spec.get("host", "127.0.0.1")), int(spec["port"])
        except (KeyError, TypeError, ValueError):
            return None
    text = str(spec).strip()
    host, _, port = text.rpartition(":")
    try:
        return (host or "127.0.0.1"), int(port)
    except ValueError:
        return None


class ServicesSensor(Sensor):
    name = "services"
    default_interval = 60.0

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        processes: Callable[[], set[str]] = list_processes,
        port_check: Callable[[str, int], bool] = port_open,
        docker: Callable[[], dict[str, str] | None] = docker_ps,
    ) -> None:
        super().__init__(config)
        c = self.config
        self.process_names = [str(p) for p in as_list(c.get("processes"))]
        self.ports = [p for p in (parse_port(s) for s in as_list(c.get("ports"))) if p]
        raw_docker = c.get("docker", False)
        self.docker_all = raw_docker is True
        self.docker_names = [str(n) for n in as_list(raw_docker)] if not isinstance(raw_docker, bool) else []
        self._processes = processes
        self._port_check = port_check
        self._docker = docker
        self._deb = Debouncer(
            fail_after=int(c.get("fail_after", 2)), recover_after=1, alert_initial=False
        )
        self._labels: dict[str, tuple[str, str]] = {}

    def _observe(self, kind: str, name: str, up: bool, events: list[Event], detail: str = "") -> None:
        key = f"service:{kind}:{name}"
        self._labels[key] = (kind, name)
        change = self._deb.update(key, not up)
        if change is None:
            return
        data = {"name": name, "kind": kind, "kind_es": _KIND_ES[kind], "detail": detail}
        if change == "down":
            events.append(Event("service.down", "warning", key, f"Se cayó {name}", data))
        else:
            events.append(Event("service.up", "notice", key, f"{name} volvió", data))

    def poll(self) -> list[Event]:
        events: list[Event] = []
        if self.process_names:
            running = self._processes()
            for name in self.process_names:
                self._observe("process", name, name in running, events)
        for host, port in self.ports:
            label = f"{host}:{port}" if host not in ("127.0.0.1", "localhost") else f"puerto {port}"
            self._observe("port", label, self._port_check(host, port), events)
        if self.docker_all or self.docker_names:
            containers = self._docker()
            if containers is not None:
                if self.docker_all:
                    # Watch whatever has been seen running; a container that
                    # was always stopped is not news.
                    names = set(containers) | {n for k, (kind, n) in self._labels.items() if kind == "container"}
                else:
                    names = set(self.docker_names)
                for name in sorted(names):
                    state = containers.get(name, "missing")
                    self._observe("container", name, state == "running", events, state)
        return events

    def inherit(self, old: Sensor) -> None:
        if not isinstance(old, ServicesSensor):
            return
        keep = {f"service:process:{n}" for n in self.process_names}
        for host, port in self.ports:
            label = f"{host}:{port}" if host not in ("127.0.0.1", "localhost") else f"puerto {port}"
            keep.add(f"service:port:{label}")
        for key, (kind, name) in old._labels.items():
            if kind == "container" and (self.docker_all or name in self.docker_names):
                keep.add(key)
        deb = old._deb.copy()
        deb.forget(keep)
        self._deb = deb
        self._labels = {k: v for k, v in old._labels.items() if k in keep}

    def summary(self) -> str | None:
        states = self._deb.states()
        if not states:
            return None
        down = [self._labels[k][1] for k, st in states.items() if st == "down" and k in self._labels]
        if not down:
            return "todos los servicios andan"
        return f"está caído {down[0]}" if len(down) == 1 else f"están caídos {', '.join(down)}"
