"""``games_state.json``: scores, the owner's habits, what she learned.

Gitignored: it records when the owner plays and how they throw at
piedra-papel-tijera. Written atomically, at most every few seconds unless
forced, and never raises — a full disk costs her the scoreboard, not the app.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VERSION = 1


class GameStore:
    SAVE_EVERY = 5.0

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = {"version": VERSION, "scores": {}, "rps": [],
                                     "bandit": {}, "proposals": {}, "fetches": 0}
        self._dirty = False
        self._saved_at = 0.0

    def load(self) -> "GameStore":
        if self.path is None or not self.path.is_file():
            return self
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict):
                for key in self.data:
                    if key in raw and isinstance(raw[key], type(self.data[key])):
                        self.data[key] = raw[key]
        except Exception:
            log.warning("Could not read %s; scores start fresh", self.path)
        return self

    # -- scores -------------------------------------------------------------------
    def record(self, game: str, result: str) -> None:
        if result not in ("owner", "wawi", "draw"):
            return
        row = self.data["scores"].setdefault(game, {"owner": 0, "wawi": 0, "draw": 0})
        row[result] = int(row.get(result, 0)) + 1
        self._dirty = True

    def score(self, game: str | None = None) -> dict[str, int]:
        rows = [self.data["scores"].get(game, {})] if game else list(self.data["scores"].values())
        out = {"owner": 0, "wawi": 0, "draw": 0}
        for row in rows:
            for k in out:
                try:
                    out[k] += int(row.get(k, 0))
                except (TypeError, ValueError):
                    pass
        return out

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._dirty = True

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    # -- disk -----------------------------------------------------------------------
    def save(self, force: bool = False, now: float | None = None) -> bool:
        if self.path is None or not (self._dirty or force):
            return False
        now = time.monotonic() if now is None else now
        if not force and now - self._saved_at < self.SAVE_EVERY:
            return False
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
            os.replace(tmp, self.path)
        except Exception:
            log.debug("Could not save %s", self.path, exc_info=True)
            return False
        self._dirty = False
        self._saved_at = now
        return True
