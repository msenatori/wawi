"""Run one command with a hard time budget and leave its verdict in a file.

Standard library only, and run by path (``python runjob.py job.json``), so
the same protocol works in a background process, in a visible Orca terminal
and anywhere else a command can be typed. The job file::

    {"argv": [...], "cwd": "...", "timeout_s": 1800,
     "stdin_file": "...", "out_file": "...", "result_file": "...",
     "env": {"KEY": "value"}}

The command runs in a process group of its own; on timeout the whole group
is killed (a coding agent leaves test runners and shells behind). The result
file is written last and atomically, so its existence means "finished".
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def run(job: dict) -> dict:
    started = time.monotonic()
    env = {k: v for k, v in os.environ.items() if not k.startswith("ORCA_")}
    env.update({str(k): str(v) for k, v in (job.get("env") or {}).items()})
    stdin = open(job["stdin_file"], "rb") if job.get("stdin_file") else subprocess.DEVNULL
    out = open(job["out_file"], "ab") if job.get("out_file") else subprocess.DEVNULL
    timed_out = False
    code = -1
    try:
        try:
            proc = subprocess.Popen(job["argv"], cwd=job.get("cwd") or None, env=env,
                                    stdin=stdin, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        except OSError as exc:
            result = {"code": 127, "timed_out": False, "elapsed_s": 0.0, "error": str(exc)}
            _write(job, result)
            return result
        try:
            code = proc.wait(timeout=float(job.get("timeout_s", 1800)))
        except subprocess.TimeoutExpired:
            timed_out = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except OSError:
                    pass
                try:
                    proc.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
            code = proc.returncode if proc.returncode is not None else -9
    finally:
        for fh in (stdin, out):
            if hasattr(fh, "close"):
                fh.close()
    result = {"code": code, "timed_out": timed_out,
              "elapsed_s": round(time.monotonic() - started, 1)}
    _write(job, result)
    return result


def _write(job: dict, result: dict) -> None:
    path = job.get("result_file")
    if not path:
        return
    tmp = f"{path}.tmp"
    Path(tmp).write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: runjob.py <job.json>", file=sys.stderr)
        return 2
    result = run(json.loads(Path(argv[1]).read_text(encoding="utf-8")))
    return 0 if result.get("code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
