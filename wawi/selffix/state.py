"""What she remembers about fixing herself: ``selffix_local.json`` (gitignored).

- ``enabled``: the menu switch (None = follow config).
- ``last_good``: main's HEAD before the last self-applied merge.
- ``issues``: per issue id -> attempts, outcome, why, when it last failed,
  won't-fix, blocked on the owner.
- ``attempts``: every attempt, newest last (capped).
- ``pending``: a finished change waiting for the owner's Aplicar/Descartar.
- ``owner_requests``: what she needs from him to do blocked work.
- ``announce``: bubbles owed after a restart ("me arreglé…").
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 200


class Store:
    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self.data: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        raw: Any = {}
        if self.path is not None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raw = {}
        if not isinstance(raw, dict):
            raw = {}
        raw.setdefault("enabled", None)
        raw.setdefault("last_good", "")
        raw.setdefault("issues", {})
        raw.setdefault("attempts", [])
        raw.setdefault("pending", None)
        raw.setdefault("owner_requests", [])
        raw.setdefault("announce", [])
        with self._lock:
            self.data = raw

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            text = json.dumps(self.data, ensure_ascii=False, indent=2)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            log.warning("selffix: could not save %s", self.path, exc_info=True)

    # -- helpers ----------------------------------------------------------------
    def issue(self, issue_id: str) -> dict:
        with self._lock:
            return self.data["issues"].setdefault(issue_id, {"attempts": 0, "reasons": []})

    def issues(self) -> dict[str, dict]:
        with self._lock:
            return dict(self.data["issues"])

    def attempts(self) -> list[dict]:
        with self._lock:
            return list(self.data["attempts"])

    def add_attempt(self, attempt: dict) -> None:
        with self._lock:
            self.data["attempts"].append(attempt)
            del self.data["attempts"][:-MAX_ATTEMPTS]

    def find_attempt(self, attempt_id: str) -> dict | None:
        with self._lock:
            for row in reversed(self.data["attempts"]):
                if row.get("id") == attempt_id:
                    return row
        return None

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self.data[key] = value
