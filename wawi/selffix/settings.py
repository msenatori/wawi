"""The ``selffix:`` config section, parsed once into a frozen object.

Every number is clamped: a typo in config.yaml must not turn a 400-line diff
limit into 40 000 or a 30-minute budget into a day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime
from pathlib import Path
from typing import Any

from wawi.config import SELFFIX_DEFAULTS

ENGINES = ("codex", "claude")
AUTONOMY = ("auto", "ask")


def _num(d: dict, key: str, lo: float, hi: float) -> float:
    try:
        return min(hi, max(lo, float(d[key])))
    except (KeyError, TypeError, ValueError):
        return float(SELFFIX_DEFAULTS[key])


def _engine(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in ENGINES else default


def parse_window(text: str) -> tuple[dtime, dtime] | None:
    """``"02:00-06:00"`` -> (02:00, 06:00). None when empty or malformed."""
    try:
        a, b = str(text or "").split("-", 1)
        ha, ma = a.strip().split(":")
        hb, mb = b.strip().split(":")
        return dtime(int(ha), int(ma)), dtime(int(hb), int(mb))
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class RemoteSettings:
    enabled: bool = False
    host: str = ""
    base_dir: str = "~/work"
    min_free_mem_mb: int = 4096
    max_load_ratio: float = 0.6
    #: Name prefixes of things on the host that are never ours to touch (the
    #: owner's production services). Kept in the local, untracked settings.
    protected: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any) -> "RemoteSettings":
        d = {**SELFFIX_DEFAULTS["remote"], **(raw if isinstance(raw, dict) else {})}
        host = str(d.get("host") or "").strip()
        try:
            mem = max(0, int(d.get("min_free_mem_mb", 4096)))
        except (TypeError, ValueError):
            mem = 4096
        try:
            load = min(1.0, max(0.05, float(d.get("max_load_ratio", 0.6))))
        except (TypeError, ValueError):
            load = 0.6
        return cls(enabled=bool(d.get("enabled")) and bool(host), host=host,
                   base_dir=str(d.get("base_dir") or "~/work").strip() or "~/work",
                   min_free_mem_mb=mem, max_load_ratio=load,
                   protected=tuple(str(p).strip() for p in (d.get("protected") or ())
                                   if str(p).strip()))


@dataclass(frozen=True)
class SelffixSettings:
    enabled: bool = True
    autonomy: str = "auto"
    max_attempts_per_day: int = 2
    min_count: int = 3
    capability_min_count: int = 2
    retry_after_days: float = 7.0
    engine: str = "codex"
    fallback_engine: str = "claude"
    reviewer_engine: str = ""
    codex_model: str = ""
    claude_model: str = ""
    time_budget_s: float = 1800.0
    max_budget_usd: float = 3.0
    review_timeout_s: float = 600.0
    test_timeout_s: float = 900.0
    max_changed_lines: int = 400
    max_files: int = 12
    idle_after_s: float = 900.0
    night_window: tuple[dtime, dtime] | None = (dtime(2, 0), dtime(6, 0))
    poll_s: float = 120.0
    watchdog_startup_s: float = 90.0
    watchdog_watch_s: float = 1800.0
    log_excerpt_lines: int = 40
    work_dir: Path = Path("~/.wawi/selffix").expanduser()
    use_orca: bool = True
    pending_days: float = 3.0
    remote: RemoteSettings = field(default_factory=RemoteSettings)
    local_file: str = "selffix_local.json"

    @classmethod
    def from_dict(cls, raw: Any) -> "SelffixSettings":
        d = {**SELFFIX_DEFAULTS, **(raw if isinstance(raw, dict) else {})}
        engine = _engine(d.get("engine"), "codex")
        fallback = _engine(d.get("fallback_engine"), "claude" if engine == "codex" else "codex")
        reviewer = str(d.get("reviewer_engine") or "").strip().lower()
        autonomy = str(d.get("autonomy") or "auto").strip().lower()
        return cls(
            enabled=bool(d.get("enabled", True)),
            autonomy=autonomy if autonomy in AUTONOMY else "ask",
            max_attempts_per_day=int(_num(d, "max_attempts_per_day", 0, 10)),
            min_count=int(_num(d, "min_count", 1, 1000)),
            capability_min_count=int(_num(d, "capability_min_count", 1, 1000)),
            retry_after_days=_num(d, "retry_after_days", 0, 365),
            engine=engine,
            fallback_engine=fallback,
            reviewer_engine=reviewer if reviewer in ENGINES else "",
            codex_model=str(d.get("codex_model") or "").strip(),
            claude_model=str(d.get("claude_model") or "").strip(),
            time_budget_s=_num(d, "time_budget_s", 60, 3 * 3600),
            max_budget_usd=_num(d, "max_budget_usd", 0.1, 50),
            review_timeout_s=_num(d, "review_timeout_s", 30, 3600),
            test_timeout_s=_num(d, "test_timeout_s", 30, 3600),
            max_changed_lines=int(_num(d, "max_changed_lines", 1, 2000)),
            max_files=int(_num(d, "max_files", 1, 50)),
            idle_after_s=_num(d, "idle_after_s", 0, 24 * 3600),
            night_window=parse_window(str(d.get("night_window") or "")),
            poll_s=_num(d, "poll_s", 5, 3600),
            watchdog_startup_s=_num(d, "watchdog_startup_s", 20, 600),
            watchdog_watch_s=_num(d, "watchdog_watch_s", 60, 6 * 3600),
            log_excerpt_lines=int(_num(d, "log_excerpt_lines", 0, 400)),
            work_dir=Path(str(d.get("work_dir") or "~/.wawi/selffix")).expanduser(),
            use_orca=bool(d.get("use_orca", True)),
            pending_days=_num(d, "pending_days", 0.1, 60),
            remote=RemoteSettings.from_dict(d.get("remote")),
            local_file=str(d.get("local_file") or ""),
        )

    def reviewer(self, author: str) -> str:
        """The reviewing engine: configured, or the one that did NOT write it."""
        if self.reviewer_engine:
            return self.reviewer_engine
        return "claude" if author == "codex" else "codex"
