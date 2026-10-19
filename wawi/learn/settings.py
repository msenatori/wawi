"""The ``learn:`` section of config.yaml, parsed once and validated.

Everything here is cheap on purpose: the bandit, the alert counts and the
tiny brain are a few kilobytes of numpy and a JSON file, so they are on by
default. Only the embedding model behind news relevance is heavy, and that
one needs an optional package (``model2vec``) the user has to install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wawi import paths as _paths

#: The multilingual static-embedding model (MIT, ~512 MB float32 on disk, 128
#: MB in memory once quantized to int8). Only used when model2vec is installed.
DEFAULT_EMBED_MODEL = "minishlab/potion-multilingual-128M"


@dataclass(frozen=True)
class LearnSettings:
    #: Master switch. Off = nothing is recorded, nothing is re-weighted.
    enabled: bool = True
    #: Bandit, alert scores and relevance state (gitignored).
    state_file: Path | None = field(default_factory=lambda: _paths.private("learn_state.json"))
    #: Learn which idle behaviours the owner enjoys.
    behaviours: bool = True
    #: Learn which alerts the owner engages with, and batch the ignored ones.
    alerts: bool = True
    #: Seconds after a behaviour or a bubble in which a touch / a pet / a
    #: voice turn counts as the owner's answer to it.
    reward_window_s: float = 20.0
    #: A click on a bubble sooner than this after it appeared is a dismissal.
    dismiss_quick_s: float = 4.0
    #: Bandit multipliers are clamped to this range: tastes tilt the dice,
    #: they never remove a behaviour.
    factor_min: float = 0.5
    factor_max: float = 2.0
    #: Evidence halves over this many days, so tastes can change.
    half_life_days: float = 21.0
    #: Alerts whose learned score is at or below this get batched (info and
    #: notice only; urgent always passes).
    batch_below: float = -0.45
    #: Minimum evidence (weighted outcomes) before the alert score is used.
    alert_min_evidence: float = 5.0
    #: News relevance (rss.item): the owner's interests, in their words.
    interests: tuple[str, ...] = ()
    relevance: bool = True
    #: model2vec model for relevance; "" or "none" = keywords only.
    embed_model: str = DEFAULT_EMBED_MODEL
    #: rss.item at or above this relevance gets its own bubble.
    own_bubble_above: float = 0.7
    #: rss.item below this (with confidence) is dropped instead of batched.
    drop_below: float = 0.2
    #: The tiny numpy brain (an MLP trained while the machine is idle).
    brain: bool = True
    brain_file: Path | None = field(default_factory=lambda: _paths.private("learn_brain.npz"))
    life_log: Path | None = field(default_factory=lambda: _paths.private("learn_life.jsonl"))
    #: Rolling life log: kept this many days, and at most this many bytes.
    life_log_days: float = 30.0
    life_log_max_bytes: int = 2_000_000
    #: Below this many logged decisions the brain only whispers.
    brain_min_samples: int = 300
    #: Hard ceiling on the brain's influence (0..1): the rules always win
    #: at least the rest of the vote.
    brain_max_weight: float = 0.5
    #: Retrain at most this often, and only after this long without a key.
    retrain_every_h: float = 12.0
    retrain_idle_s: float = 600.0

    @classmethod
    def from_dict(cls, raw: Any) -> "LearnSettings":
        """Parse the section; anything malformed falls back to its default
        (a learning typo is never worth stopping the app over)."""
        d = cls()
        if not isinstance(raw, dict):
            return d

        def num(key: str, default: float, lo: float = 0.0, hi: float = float("inf")) -> float:
            try:
                return min(hi, max(lo, float(raw.get(key, default))))
            except (TypeError, ValueError):
                return default

        def path(key: str, default: Path | None) -> Path | None:
            if key not in raw:
                return default
            value = raw.get(key)
            return _paths.resolve(value) if value else None

        interests = raw.get("interests") or ()
        if isinstance(interests, str):
            interests = [interests]
        embed = str(raw.get("embed_model", d.embed_model) or "").strip()
        lo = num("factor_min", d.factor_min, 0.05, 1.0)
        hi = num("factor_max", d.factor_max, 1.0, 20.0)
        return cls(
            enabled=bool(raw.get("enabled", d.enabled)),
            state_file=path("state_file", d.state_file),
            behaviours=bool(raw.get("behaviours", d.behaviours)),
            alerts=bool(raw.get("alerts", d.alerts)),
            reward_window_s=num("reward_window_s", d.reward_window_s, 1.0, 600.0),
            dismiss_quick_s=num("dismiss_quick_s", d.dismiss_quick_s, 0.0, 60.0),
            factor_min=lo,
            factor_max=hi,
            half_life_days=num("half_life_days", d.half_life_days, 0.5, 3650.0),
            batch_below=num("batch_below", d.batch_below, -1.0, 1.0),
            alert_min_evidence=num("alert_min_evidence", d.alert_min_evidence, 0.0, 1000.0),
            interests=tuple(str(i).strip() for i in interests if str(i).strip()),
            relevance=bool(raw.get("relevance", d.relevance)),
            embed_model="" if embed.lower() in ("", "none", "off") else embed,
            own_bubble_above=num("own_bubble_above", d.own_bubble_above, 0.0, 1.0),
            drop_below=num("drop_below", d.drop_below, 0.0, 1.0),
            brain=bool(raw.get("brain", d.brain)),
            brain_file=path("brain_file", d.brain_file),
            life_log=path("life_log", d.life_log),
            life_log_days=num("life_log_days", d.life_log_days, 1.0, 3650.0),
            life_log_max_bytes=int(num("life_log_max_bytes", d.life_log_max_bytes, 10_000, 1e9)),
            brain_min_samples=int(num("brain_min_samples", d.brain_min_samples, 10, 1e7)),
            brain_max_weight=num("brain_max_weight", d.brain_max_weight, 0.0, 0.9),
            retrain_every_h=num("retrain_every_h", d.retrain_every_h, 0.1, 24.0 * 30),
            retrain_idle_s=num("retrain_idle_s", d.retrain_idle_s, 0.0, 86400.0),
        )
