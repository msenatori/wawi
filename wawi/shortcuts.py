"""Voice-safe access to macOS Shortcuts through its public CLI."""

from __future__ import annotations

import difflib
import json
import logging
import re
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)
CLI = "/usr/bin/shortcuts"
CACHE_SECONDS = 300
CONFIRM_SECONDS = 180
MAX_OUTPUT = 3000


def normalize(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value.casefold())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", plain))


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    allow: tuple[str, ...] = ()
    confirm: tuple[str, ...] = ()
    local_file: Path = Path("shortcuts_local.json")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Settings":
        data = raw or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            allow=tuple(str(x) for x in data.get("allow", []) or []),
            confirm=tuple(str(x) for x in data.get("confirm", []) or []),
            local_file=Path(data.get("local_file") or "shortcuts_local.json").expanduser(),
        )


Runner = Callable[[list[str], float], subprocess.CompletedProcess[str]]


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


class ShortcutsBridge:
    def __init__(self, settings: Settings, runner: Runner = _run,
                 clock: Callable[[], float] = time.time) -> None:
        self.settings = settings
        self.runner = runner
        self.clock = clock
        self._names: list[str] = []
        self._expires = 0.0
        self._pending: tuple[str, str, float] | None = None
        self._confirmation_id: str | None = None
        self._confirm_result: tuple[bool, str, dict] | None = None
        try:
            saved = json.loads(settings.local_file.read_text())
            if isinstance(saved.get("names"), list):
                self._names = [str(x) for x in saved["names"]]
                self._expires = float(saved.get("expires", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            pass

    def _save(self) -> None:
        try:
            path = self.settings.local_file
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".tmp")
            temp.write_text(json.dumps({"names": self._names, "expires": self._expires}))
            temp.replace(path)
        except OSError:
            log.warning("shortcuts: could not save cache", exc_info=True)

    def _call(self, argv: list[str], timeout: float) -> tuple[bool, str]:
        try:
            result = self.runner(argv, timeout)
        except subprocess.TimeoutExpired:
            return False, "El atajo tardó demasiado y lo detuve."
        except OSError:
            return False, "No pude abrir la app Atajos en esta Mac."
        if result.returncode:
            detail = (result.stderr or result.stdout or "error desconocido").strip()[:250]
            return False, f"Atajos falló: {detail}"
        return True, result.stdout

    def list(self) -> tuple[bool, str, dict]:
        if self.clock() >= self._expires:
            ok, output = self._call([CLI, "list"], 15)
            if not ok:
                return False, output, {"error": "list_failed"}
            self._names = [line.strip() for line in output.splitlines() if line.strip()]
            self._expires = self.clock() + CACHE_SECONDS
            self._save()
        visible = self._allowed(self._names)
        return True, f"Tengo {len(visible)} atajos disponibles.", {"names": visible}

    def _allowed(self, names: list[str]) -> list[str]:
        allow = {normalize(x) for x in self.settings.allow}
        return [name for name in names if not allow or normalize(name) in allow]

    def _match(self, spoken: str, names: list[str]) -> tuple[str | None, list[str]]:
        target = normalize(spoken)
        if not target:
            return None, []
        exact = [name for name in names if normalize(name) == target]
        if len(exact) == 1:
            return exact[0], []
        scored = sorted(((difflib.SequenceMatcher(None, target, normalize(name)).ratio(), name)
                         for name in names), reverse=True)
        if not scored or scored[0][0] < 0.65:
            return None, []
        candidates = [name for score, name in scored if score >= max(0.65, scored[0][0] - 0.12)][:3]
        if len(candidates) > 1:
            return None, candidates
        return scored[0][1], []

    def run(self, name: str, input_text: str = "") -> tuple[bool, str, dict]:
        ok, speak, data = self.list()
        if not ok:
            return ok, speak, data
        chosen, candidates = self._match(name, data["names"])
        if chosen is None:
            if candidates:
                return False, "¿Cuál atajo querés: " + ", ".join(candidates) + "?", {"candidates": candidates}
            return False, "No encontré ese atajo entre los disponibles.", {"error": "not_found"}
        if normalize(chosen) in {normalize(x) for x in self.settings.confirm}:
            from wawi.confirm import REGISTRY
            if self._confirmation_id:
                REGISTRY.discard(self._confirmation_id)
            pending = (chosen, input_text, self.clock())
            self._pending = pending
            def clear() -> None:
                if self._pending is pending:
                    self._pending = None
            def accepted() -> None:
                clear()
                self._confirm_result = self._execute(chosen, input_text)
            self._confirmation_id = REGISTRY.add(
                "Ejecutar atajo", chosen, "shortcuts", accepted, clear)
            return True, f"¿Corro el atajo «{chosen}»?", {"pending": True, "run": False}
        return self._execute(chosen, input_text)

    def _execute(self, name: str, input_text: str) -> tuple[bool, str, dict]:
        path: Path | None = None
        try:
            argv = [CLI, "run", name]
            if input_text:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as stream:
                    stream.write(input_text)
                    path = Path(stream.name)
                argv += ["-i", str(path)]
            ok, output = self._call(argv, 60)
            if not ok:
                return False, output, {"error": "run_failed"}
            return True, f"Corrí «{name}».", {"name": name, "output": output[:MAX_OUTPUT],
                                         "truncated": len(output) > MAX_OUTPUT}
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def hear(self, text: str) -> str | None:
        from wawi.confirm import REGISTRY
        pending = self._pending
        if pending is None:
            return None
        if self.clock() - pending[2] > CONFIRM_SECONDS:
            REGISTRY.discard(self._confirmation_id or "")
            self._pending = None
            return None
        from wawi import lang
        words = text.lower().split()
        answer = lang.affirmative(text)
        if answer is True and len(words) <= 8 and "pero" not in words:
            if REGISTRY.resolve(self._confirmation_id or "", True):
                return self._confirm_result[1] if self._confirm_result else None
        if answer is False and len(words) <= 4:
            if REGISTRY.resolve(self._confirmation_id or "", False):
                return "Bueno, no corro el atajo."
        return None


_BRIDGE: ShortcutsBridge | None = None


def install(bridge: ShortcutsBridge | None) -> None:
    global _BRIDGE
    _BRIDGE = bridge


def get() -> ShortcutsBridge | None:
    return _BRIDGE


def build(cfg: Any) -> ShortcutsBridge | None:
    settings = Settings.from_dict(getattr(cfg, "shortcuts", None))
    return ShortcutsBridge(settings) if settings.enabled and Path(CLI).exists() else None


def hear(text: str) -> str | None:
    if _BRIDGE is None:
        return None
    try:
        return _BRIDGE.hear(text)
    except Exception:
        log.warning("shortcuts: confirmation failed", exc_info=True)
        return "No pude ejecutar ese atajo."
