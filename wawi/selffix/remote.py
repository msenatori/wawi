"""An optional remote executor: heavy or long jobs on a LAN server over ssh.

For work that should not load the Mac -- model training (the wake word's
retraining), long test runs, data processing. Off by default; the host is an
ssh alias from config (``selffix.remote.host``), never a name in the code.

House rules for a shared server, enforced here rather than promised:

- Jobs run **detached** under ``<base_dir>/<project>/.wawi-jobs/<label>/``.
- Everything created is **labelled** (``wawi-<project>-<name>-<stamp>``), and
  cleanup removes only such a directory.
- **Kill by PID**, never by pattern, and only after checking that PID's
  command line carries the job's label.
- **Free resources are checked** before a heavy job: load and free memory
  must leave headroom (``max_load_ratio``, ``min_free_mem_mb``).
- **Secrets** go to a chmod-600 env file written over ssh's stdin, never on
  a command line or in a log.
- **No sudo, no apt, no docker** (production containers live there), and no
  command naming another project's containers or volumes.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Callable

from wawi.selffix.engines import Job, JobResult

log = logging.getLogger(__name__)

#: ssh(argv_after_host, stdin_text, timeout) -> (code, stdout, stderr)
Ssh = Callable[[str, str, float], "tuple[int, str, str]"]

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_DIR = re.compile(r"^~?[A-Za-z0-9/._-]+$")
_REFUSED = re.compile(r"\b(sudo|apt|apt-get|dpkg|docker|podman|systemctl)\b")


class RemoteError(Exception):
    pass


def make_ssh(host: str) -> Ssh:
    def run(command: str, stdin_text: str = "", timeout: float = 60.0) -> tuple[int, str, str]:
        try:
            proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                   host, command], input=stdin_text, capture_output=True,
                                  text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 255, "", str(exc)
        return proc.returncode, proc.stdout, proc.stderr
    return run


@dataclass(frozen=True)
class Resources:
    cpus: int
    load1: float
    mem_free_mb: int


@dataclass(frozen=True)
class RemoteJob:
    label: str
    dir: str
    pid: int


class RemoteExecutor:
    name = "remote"

    def __init__(self, settings, *, ssh: Ssh | None = None, project: str = "wawi-selffix",
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, poll_s: float = 30.0) -> None:
        if not settings.enabled or not settings.host:
            raise RemoteError("remote executor is off")
        if not _SAFE.match(project) or not _SAFE_DIR.match(settings.base_dir):
            raise RemoteError("unsafe project or base_dir")
        self.settings = settings
        self.ssh = ssh or make_ssh(settings.host)
        self.project = project
        self.sleep = sleep
        self.clock = clock
        self.poll_s = poll_s
        self.jobs: dict[str, RemoteJob] = {}

    # -- resources ----------------------------------------------------------
    def resources(self) -> Resources | None:
        code, out, _ = self.ssh("nproc; cut -d' ' -f1 /proc/loadavg; "
                                "awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo",
                                "", 20.0)
        if code != 0:
            return None
        try:
            cpus, load, mem = out.split()[:3]
            return Resources(int(cpus), float(load), int(mem))
        except ValueError:
            return None

    def headroom(self) -> tuple[bool, str]:
        res = self.resources()
        if res is None:
            return False, "no pude leer los recursos del servidor"
        if res.load1 > self.settings.max_load_ratio * res.cpus:
            return False, f"está ocupado (carga {res.load1:.1f} de {res.cpus} CPU)"
        if res.mem_free_mb < self.settings.min_free_mem_mb:
            return False, f"poca memoria libre ({res.mem_free_mb} MB)"
        return True, ""

    # -- lifecycle ----------------------------------------------------------
    def _dir(self, label: str) -> str:
        return f"{self.settings.base_dir.rstrip('/')}/{self.project}/.wawi-jobs/{label}"

    def start(self, name: str, command: str, *, env: dict[str, str] | None = None,
              heavy: bool = True) -> RemoteJob:
        if not _SAFE.match(name):
            raise RemoteError("unsafe job name")
        if _REFUSED.search(command) or any(
                re.search(rf"\b{re.escape(p)}", command) for p in self.settings.protected):
            raise RemoteError("refused: sudo/apt/docker/production names are not allowed")
        if heavy:
            ok, why = self.headroom()
            if not ok:
                raise RemoteError(why)
        label = f"wawi-{self.project}-{name}-{time.strftime('%Y%m%d%H%M%S')}"
        jobdir = self._dir(label)
        script = ("#!/usr/bin/env bash\n"
                  f"# {label}\n"
                  "set -u\n"
                  f"cd {jobdir}\n"
                  "[ -f env ] && set -a && . ./env && set +a\n"
                  f"( cd {self.settings.base_dir}/{self.project} && {command} )\n"
                  "echo $? > exit_code\n")
        code, _, err = self.ssh(f"mkdir -p {jobdir} && umask 077 && cat > {jobdir}/run.sh",
                                script, 30.0)
        if code != 0:
            raise RemoteError(f"could not write the job: {err.strip()[:200]}")
        if env:
            text = "".join(f"{k}={shlex.quote(str(v))}\n" for k, v in env.items()
                           if _SAFE.match(k))
            code, _, _ = self.ssh(f"umask 077 && cat > {jobdir}/env && chmod 600 {jobdir}/env",
                                  text, 30.0)
            if code != 0:
                raise RemoteError("could not write the env file")
        code, out, err = self.ssh(
            f"cd {jobdir} && nohup setsid bash {jobdir}/run.sh > out.log 2>&1 < /dev/null & echo $!",
            "", 30.0)
        try:
            pid = int(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise RemoteError(f"no pid: {err.strip()[:200]}") from None
        job = RemoteJob(label, jobdir, pid)
        self.jobs[label] = job
        log.info("remote: started %s (pid %d)", label, pid)
        return job

    def status(self, job: RemoteJob) -> tuple[str, int | None]:
        """("running" | "done" | "unknown", exit code)."""
        code, out, _ = self.ssh(f"cat {job.dir}/exit_code 2>/dev/null || echo RUNNING", "", 20.0)
        if code != 0:
            return "unknown", None
        text = out.strip()
        if text == "RUNNING":
            return "running", None
        try:
            return "done", int(text)
        except ValueError:
            return "unknown", None

    def output(self, job: RemoteJob, lines: int = 200) -> str:
        code, out, _ = self.ssh(f"tail -n {int(lines)} {job.dir}/out.log", "", 20.0)
        return out if code == 0 else ""

    def stop(self, job: RemoteJob) -> bool:
        """Kill the job's own PID, after checking it is still that job."""
        code, _, _ = self.ssh(
            f"ps -o args= -p {int(job.pid)} | grep -qF {shlex.quote(job.label)} && kill {int(job.pid)}",
            "", 20.0)
        return code == 0

    def cleanup(self, job: RemoteJob) -> bool:
        """Remove the job's directory: only a labelled one this executor made."""
        if job.label not in self.jobs or not job.dir.endswith(f"/.wawi-jobs/{job.label}"):
            return False
        code, _, _ = self.ssh(f"rm -rf -- {job.dir}", "", 30.0)
        if code == 0:
            self.jobs.pop(job.label, None)
        return code == 0

    # -- the executor interface ---------------------------------------------
    def run(self, job: Job) -> JobResult:
        """Run *job*'s argv on the server (in ``<base_dir>/<project>``)."""
        try:
            remote = self.start(job.name.replace("_", "-"), shlex.join(job.argv))
        except RemoteError as exc:
            return JobResult(126, False, 0.0, str(exc), "remote")
        deadline = self.clock() + job.timeout_s
        state, code = "running", None
        while self.clock() < deadline:
            state, code = self.status(remote)
            if state == "done":
                break
            self.sleep(self.poll_s)
        output = self.output(remote)
        if state != "done":
            self.stop(remote)
            return JobResult(-1, True, job.timeout_s, output, "remote")
        self.cleanup(remote)
        return JobResult(int(code if code is not None else -1), False, 0.0, output, "remote")
