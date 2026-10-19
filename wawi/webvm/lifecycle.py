"""Start, stop and build her little computer on a LAN server, through the docker CLI.

Every command is ``docker --context <context> …`` (an ssh context), run by an
injectable *runner* so the tests never touch Docker. Only ONE container is
ever touched, by its exact name, and it carries the label ``wawi.webvm=1``;
nothing here lists, prunes or removes anything else.

Secrets never appear in argv: ``-e WEBVM_TOKEN`` (no value) makes the docker
CLI copy it from ITS environment, which only the runner call receives.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from wawi.webvm.client import Client, Unreachable, WebVMError
from wawi.webvm.settings import WebVMSettings

log = logging.getLogger(__name__)

LABEL = "wawi.webvm=1"
#: Inside the container (image/server.py and entrypoint.sh).
API_PORT_INSIDE = 8800
VIEW_PORT_INSIDE = 6080
PROFILE_VOLUME = "wawi-webvm-profile"
IMAGE_DIR = Path(__file__).with_name("image")

#: ``runner(argv, env=None, timeout=...) -> CompletedProcess`` (the test seam).
Runner = Callable[..., subprocess.CompletedProcess]


def _run(argv: list[str], env: dict | None = None, timeout: float = 60.0) -> subprocess.CompletedProcess:
    import os

    full_env = None
    if env:
        full_env = {**os.environ, **env}
    return subprocess.run(argv, env=full_env, capture_output=True, text=True,
                          timeout=timeout, check=False)


_UNREACHABLE_HINTS = ("cannot connect", "connection refused", "no route to host",
                      "could not resolve", "connection timed out", "operation timed out",
                      "host key verification", "permission denied", "error during connect",
                      "is the docker daemon running", "ssh:", "connection reset")


def _explain(stderr: str) -> str:
    low = (stderr or "").lower()
    if any(h in low for h in _UNREACHABLE_HINTS):
        return "No llego al servidor de mi compu: ¿está prendido y en la red?"
    line = (stderr or "").strip().splitlines()
    return f"Docker en el servidor de mi compu falló: {(line[-1] if line else 'sin detalle')[:160]}"


@dataclass(frozen=True)
class Status:
    exists: bool
    running: bool
    healthy: bool
    url: str = ""
    title: str = ""
    detail: str = ""

    def describe(self) -> str:
        if self.healthy:
            where = f" en «{self.title or self.url}»" if (self.title or self.url) else ""
            return f"Mi compu está prendida{where}."
        if self.running:
            return "Mi compu está arrancando."
        return self.detail or "Mi compu está apagada."


class Lifecycle:
    def __init__(self, settings: WebVMSettings, credentials: Callable[[], tuple[str, str]],
                 runner: Runner = _run, client_factory: Callable[[str, str], Client] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.settings = settings
        self._credentials = credentials
        self._runner = runner
        self._client_factory = client_factory or (lambda url, token: Client(url, token))
        self._clock = clock
        self._sleep = sleep
        self._client: Client | None = None

    # -- docker ---------------------------------------------------------------
    def _docker(self, *args: str, env: dict | None = None, timeout: float = 60.0) -> subprocess.CompletedProcess:
        argv = ["docker", "--context", self.settings.context, *args]
        try:
            return self._runner(argv, env=env, timeout=timeout)
        except FileNotFoundError:
            raise WebVMError("No encuentro el comando docker en esta Mac.") from None
        except subprocess.TimeoutExpired:
            raise WebVMError("Docker en el servidor de mi compu tardó demasiado en contestar.") from None

    def client(self) -> Client:
        if self._client is None:
            token, _vnc = self._credentials()
            self._client = self._client_factory(self.settings.api_url, token)
        return self._client

    def container_state(self) -> str:
        """``running``, ``exited``, ``created``… or "" when there is none."""
        s = self.settings
        res = self._docker("ps", "-a", "--filter", f"name=^/{s.container}$",
                           "--filter", f"label={LABEL}", "--format", "{{.State}}", timeout=30)
        if res.returncode != 0:
            raise WebVMError(_explain(res.stderr))
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        return lines[0] if lines else ""

    def image_present(self) -> bool:
        res = self._docker("image", "inspect", "--format", "{{.Id}}", self.settings.image, timeout=30)
        if res.returncode == 0:
            return True
        if "no such image" in (res.stderr or "").lower() or "not found" in (res.stderr or "").lower():
            return False
        raise WebVMError(_explain(res.stderr))

    def run_argv(self) -> list[str]:
        """The ``docker run`` arguments (after ``docker --context X``). No secret in them."""
        s = self.settings
        argv = [
            "run", "-d", "--rm",
            "--name", s.container,
            "--label", LABEL,
            "--memory", s.memory, "--cpus", s.cpus, "--shm-size", s.shm_size,
            "--security-opt", "no-new-privileges",
            "-p", f"{s.host}:{s.api_port}:{API_PORT_INSIDE}",
            "-p", f"{s.host}:{s.view_port}:{VIEW_PORT_INSIDE}",
            "-e", "WEBVM_TOKEN", "-e", "WEBVM_VNC_PASSWORD",
            "-e", f"WEBVM_IDLE_EXIT_MIN={s.idle_stop_min + 5:g}",
            "-e", f"WEBVM_LOCALE={s.locale}",
            "-e", f"WEBVM_TIMEZONE={s.timezone}",
        ]
        if s.allow_lan:
            argv += ["-e", "WEBVM_ALLOW_LAN=1"]
        if s.persist_profile:
            argv += ["-v", f"{PROFILE_VOLUME}:/home/pwuser/profile",
                     "-e", "WEBVM_PROFILE_DIR=/home/pwuser/profile"]
        argv.append(s.image)
        return argv

    # -- lifecycle ------------------------------------------------------------
    def healthy(self) -> dict | None:
        try:
            return self.client().health()
        except WebVMError:
            return None

    def ensure_running(self, wait_s: float = 45.0) -> dict:
        """Make sure the container answers; start it if needed. Returns /health."""
        got = self.healthy()
        if got is not None:
            return got
        state = self.container_state()
        if state and state != "running":
            # Ours (exact name + label): a leftover that --rm did not clean.
            self._docker("rm", "-f", self.settings.container, timeout=30)
            state = ""
        if not state:
            if not self.image_present():
                raise WebVMError("Mi compu no está construida todavía: corré "
                                 "«scripts/wawictl webvm build».")
            token, vnc = self._credentials()
            res = self._docker(*self.run_argv(),
                               env={"WEBVM_TOKEN": token, "WEBVM_VNC_PASSWORD": vnc},
                               timeout=90)
            if res.returncode != 0:
                if "port is already allocated" in (res.stderr or "").lower() or \
                        "address already in use" in (res.stderr or "").lower():
                    raise WebVMError("Los puertos de mi compu están ocupados en su servidor.")
                raise WebVMError(_explain(res.stderr))
            log.info("webvm: container started on %s", self.settings.context)
        deadline = self._clock() + wait_s
        while self._clock() < deadline:
            got = self.healthy()
            if got is not None:
                return got
            self._sleep(1.0)
        raise WebVMError("Mi compu arrancó pero no contesta.")

    def stop(self) -> bool:
        """Stop the container (``--rm`` removes it). True when one was running."""
        state = self.container_state()
        if not state:
            return False
        res = self._docker("stop", "-t", "5", self.settings.container, timeout=40)
        if res.returncode != 0 and "no such container" not in (res.stderr or "").lower():
            raise WebVMError(_explain(res.stderr))
        log.info("webvm: container stopped")
        return state == "running"

    def status(self) -> Status:
        got = self.healthy()
        if got is not None:
            return Status(True, True, True, str(got.get("url", "")), str(got.get("title", "")))
        try:
            state = self.container_state()
        except WebVMError as exc:
            return Status(False, False, False, detail=str(exc))
        return Status(bool(state), state == "running", False)

    # -- build (CLI only: slow) -----------------------------------------------
    def build_commands(self) -> list[list[str]]:
        s = self.settings
        remote = s.remote_dir
        return [
            ["ssh", s.ssh_host, f"mkdir -p {remote}"],
            ["rsync", "-a", "--delete", "--exclude", "build.log", "--exclude", "__pycache__",
             f"{IMAGE_DIR}/", f"{s.ssh_host}:{remote.replace('~/', '', 1)}/"],
            ["ssh", s.ssh_host,
             f"cd {remote} && docker build --label {LABEL} -t {s.image} . > build.log 2>&1; "
             f"status=$?; tail -n 5 build.log; exit $status"],
        ]

    def build(self, echo: Callable[[str], None] = print) -> None:
        for argv in self.build_commands():
            echo("$ " + " ".join(argv))
            try:
                res = self._runner(argv, env=None, timeout=1800)
            except FileNotFoundError:
                raise WebVMError(f"No encuentro {argv[0]} en esta Mac.") from None
            if res.stdout.strip():
                echo(res.stdout.strip())
            if res.returncode != 0:
                raise WebVMError(_explain(res.stderr or res.stdout))


__all__ = ["Lifecycle", "Status", "LABEL", "WebVMError", "Unreachable"]
