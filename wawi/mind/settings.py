"""The ``mind:`` section of config.yaml, parsed and validated.

Kept apart from :mod:`wawi.config` on purpose: the mind is an optional layer
and its knobs evolve with it. ``wawi.config`` only stores the parsed object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Event kinds Claude may be woken for when nobody configured anything.
#: Only web.down: it is urgent, rare, and a diagnosis is worth a cent. Info
#: events never reach Claude by default -- there are too many of them.
DEFAULT_CLAUDE_EVENTS = ("web.down",)

LOCAL_BACKENDS = ("auto", "mlx", "ollama")

#: Smallest model that writes acceptable Rioplatense one-liners on this Mac
#: (Apache-2.0, ~1 GB). Not the default: the default downloads nothing. See
#: the benchmark numbers in wawi/mind/local.py.
SUGGESTED_LOCAL_MODEL = "mlx-community/Qwen3-1.7B-4bit"


@dataclass(frozen=True)
class MindSettings:
    """Everything the mind reads from config. Defaults are the safe choice."""

    #: Master switch. Off means no subscription to the bus, no tools, no bubbles.
    enabled: bool = True
    #: Hand fetch_page / whats_new (and the pet and sensors tools when those
    #: modules exist) to the brain. Each schema is paid for on every turn.
    tools: bool = True
    #: "none" disables the local model entirely; bubbles then come from the
    #: curated list and triage defers to the reflex rules. Otherwise a model id
    #: (an MLX repo, or an Ollama tag with local_backend: ollama).
    local_model: str = "none"
    #: "auto" = Ollama when the brain backend is ollama, MLX otherwise.
    local_backend: str = "auto"
    #: Seconds to load the local model before giving up for this run.
    local_load_timeout_s: float = 60.0
    #: Seconds per local generation. Past this the canned fallback wins.
    local_timeout_s: float = 4.0
    #: Personality bubbles on their own initiative.
    bubbles: bool = True
    #: The gap between two unprompted bubbles is drawn from this range.
    bubble_min_interval_s: float = 600.0
    bubble_max_interval_s: float = 1200.0
    #: Let the local model triage info/notice events.
    triage: bool = True
    #: Event kinds that may wake Claude (rate-limited).
    claude_events: tuple[str, ...] = DEFAULT_CLAUDE_EVENTS
    #: Claude model for the mind. Empty = the brain's models.on_demand.
    claude_model: str = ""
    #: Seconds per Claude call made by the mind.
    claude_timeout_s: float = 45.0
    #: At most one Claude call per event kind+key in this window.
    claude_min_interval_s: float = 600.0
    #: Hard ceiling on unprompted Claude calls per hour (all kinds together).
    claude_max_per_hour: int = 6
    #: fetch_page may reach private / loopback addresses. Off: a misheard
    #: "localhost" should not poke at the router admin page.
    fetch_allow_private: bool = False
    #: Bytes read from a page before cutting it off.
    fetch_max_bytes: int = 1_500_000
    fetch_timeout_s: float = 12.0
    #: Characters of page text sent to Claude.
    fetch_max_chars: int = 12_000
    #: Reachability probe used by the web.down investigation to tell "your
    #: internet is down" from "that site is down".
    probe_url: str = "https://www.gstatic.com/generate_204"


def _as_tuple(raw: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(x).strip() for x in raw if str(x).strip())


def parse(raw: Any) -> MindSettings:
    """Build :class:`MindSettings` from the raw ``mind:`` mapping.

    Raises ValueError on a wrong type or backend: a typo should be loud, not
    silently fall back to a default that does something else.
    """
    if raw is None:
        return MindSettings()
    if not isinstance(raw, dict):
        raise ValueError("mind must be a mapping")
    d = MindSettings()
    backend = str(raw.get("local_backend", d.local_backend)).strip().lower()
    if backend not in LOCAL_BACKENDS:
        raise ValueError(
            f"mind.local_backend={backend!r} is not valid; use one of {LOCAL_BACKENDS}"
        )
    local_model = str(raw.get("local_model", d.local_model) or "none").strip()
    lo = float(raw.get("bubble_min_interval_s", d.bubble_min_interval_s))
    hi = float(raw.get("bubble_max_interval_s", d.bubble_max_interval_s))
    return MindSettings(
        enabled=bool(raw.get("enabled", d.enabled)),
        tools=bool(raw.get("tools", d.tools)),
        local_model=local_model if local_model.lower() not in ("", "none", "off") else "none",
        local_backend=backend,
        local_load_timeout_s=float(raw.get("local_load_timeout_s", d.local_load_timeout_s)),
        local_timeout_s=float(raw.get("local_timeout_s", d.local_timeout_s)),
        bubbles=bool(raw.get("bubbles", d.bubbles)),
        bubble_min_interval_s=max(30.0, lo),
        bubble_max_interval_s=max(30.0, lo, hi),
        triage=bool(raw.get("triage", d.triage)),
        claude_events=_as_tuple(raw.get("claude_events"), d.claude_events),
        claude_model=str(raw.get("claude_model", "") or "").strip(),
        claude_timeout_s=float(raw.get("claude_timeout_s", d.claude_timeout_s)),
        claude_min_interval_s=float(raw.get("claude_min_interval_s", d.claude_min_interval_s)),
        claude_max_per_hour=max(0, int(raw.get("claude_max_per_hour", d.claude_max_per_hour))),
        fetch_allow_private=bool(raw.get("fetch_allow_private", d.fetch_allow_private)),
        fetch_max_bytes=max(10_000, int(raw.get("fetch_max_bytes", d.fetch_max_bytes))),
        fetch_timeout_s=float(raw.get("fetch_timeout_s", d.fetch_timeout_s)),
        fetch_max_chars=max(1000, int(raw.get("fetch_max_chars", d.fetch_max_chars))),
        probe_url=str(raw.get("probe_url", d.probe_url) or d.probe_url),
    )
