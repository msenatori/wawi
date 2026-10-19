"""The gates a self-fix must pass before it may touch main. All of them.

Cheap, deterministic gates first (forbidden paths, size, dependencies,
secrets), then the full test suite in the worktree, then a second-opinion
review by a different engine. The first failure ends the attempt; its reason
goes into the issue's history and into the next attempt's brief.
"""

from __future__ import annotations

import fnmatch
import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from wawi.selffix.brief import FORBIDDEN_PATHS
from wawi.selffix.gitops import FileStat
from wawi.selffix.scrub import scan_diff


@dataclass
class GateResult:
    name: str
    ok: bool
    reason: str = ""
    detail: list[str] = field(default_factory=list)


def _forbidden(path: str) -> bool:
    path = path.lstrip("./")
    name = path.rsplit("/", 1)[-1]
    for pattern in FORBIDDEN_PATHS:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(name, pattern):
            return True
        if pattern.endswith("/*") and path.startswith(pattern[:-1]):
            return True
    return path.startswith("LICENSE") or "/.git/" in f"/{path}"


def gate_forbidden(stats: list[FileStat], diff: str) -> GateResult:
    bad = [s.path for s in stats if _forbidden(s.path)]
    # .gitignore: adding rules is fine, dropping one could publish a secret.
    removed_ignores = [line for line in _file_hunks(diff, ".gitignore")
                       if line.startswith("-") and not line.startswith("---") and line[1:].strip()]
    if removed_ignores:
        bad.append(".gitignore (quita reglas)")
    if bad:
        return GateResult("forbidden", False, "toca archivos prohibidos", bad)
    return GateResult("forbidden", True)


def gate_size(stats: list[FileStat], max_lines: int, max_files: int) -> GateResult:
    lines = sum(s.added + s.removed for s in stats)
    if not stats:
        return GateResult("size", False, "no cambió nada")
    if len(stats) > max_files:
        return GateResult("size", False, f"demasiados archivos ({len(stats)} > {max_files})")
    if lines > max_lines:
        return GateResult("size", False, f"diff demasiado grande ({lines} > {max_lines} líneas)")
    return GateResult("size", True, f"{lines} líneas, {len(stats)} archivos")


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #

_DEP_FILES = ("requirements", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "poetry.lock")
_PINNED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?==([A-Za-z0-9.+!-]+)\s*(#.*)?$")


def _file_hunks(diff: str, path: str) -> list[str]:
    lines, inside = [], False
    for line in (diff or "").splitlines():
        if line.startswith("diff --git "):
            inside = line.endswith(f" b/{path}")
            continue
        if inside and not line.startswith(("+++", "---", "@@", "index ", "new file", "deleted file")):
            lines.append(line)
    return lines


def _dep_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(name.startswith(prefix) for prefix in _DEP_FILES)


def pure_python_on_pypi(name: str, version: str, timeout: float = 10.0) -> bool:
    """True when PyPI ships *name*==*version* as a ``py3-none-any`` wheel."""
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed host
            data = json.load(resp)
    except Exception:
        return False
    files = data.get("urls") or []
    return any(str(f.get("filename", "")).endswith("-none-any.whl") for f in files)


def gate_dependencies(stats: list[FileStat], diff: str, *, capability: bool,
                      pure_python: Callable[[str, str], bool] = pure_python_on_pypi) -> GateResult:
    """No new dependency, except a pinned pure-Python one for a capability."""
    touched = [s.path for s in stats if _dep_file(s.path)]
    if not touched:
        return GateResult("dependencies", True)
    added = []
    for path in touched:
        for line in _file_hunks(diff, path):
            if line.startswith("+") and line[1:].strip() and not line[1:].strip().startswith("#"):
                added.append((path, line[1:].strip()))
    if not added:
        return GateResult("dependencies", True)
    if not capability:
        return GateResult("dependencies", False, "un arreglo no puede agregar dependencias",
                          [f"{p}: {d}" for p, d in added])
    bad = []
    for path, dep in added:
        m = _PINNED.match(dep)
        if not path.rsplit("/", 1)[-1].startswith("requirements") or m is None:
            bad.append(f"{path}: {dep} (no fijada con ==)")
        elif not pure_python(m.group(1), m.group(3)):
            bad.append(f"{dep} (no es pura-Python)")
    if bad:
        return GateResult("dependencies", False, "dependencia no permitida", bad)
    return GateResult("dependencies", True, "dependencia pura-Python fijada",
                      [d for _, d in added])


def gate_secrets(diff: str) -> GateResult:
    found = scan_diff(diff)
    if found:
        return GateResult("secrets", False, "el diff tiene algo secreto o personal", found)
    return GateResult("secrets", True)


def gate_tests(code: int, timed_out: bool, output: str) -> GateResult:
    tail = [line for line in (output or "").splitlines()[-15:] if line.strip()]
    if timed_out:
        return GateResult("tests", False, "los tests no terminaron a tiempo", tail[-5:])
    if code != 0:
        return GateResult("tests", False, "los tests fallan", tail[-8:])
    return GateResult("tests", True, (tail[-1] if tail else "").strip()[:120])


def gate_review(approved: bool, reasons: str, engine: str) -> GateResult:
    lines = [line.strip() for line in (reasons or "").splitlines() if line.strip()][-8:]
    if not approved:
        return GateResult("review", False, f"el revisor ({engine}) lo rechazó", lines)
    return GateResult("review", True, f"aprobado por {engine}", lines[-3:])
