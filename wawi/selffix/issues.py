"""Reading the issue journal (``issues/*.md``) and choosing what to work on.

The journal is wawi/selfimprove.py's; this module only reads it. Two rules:

- **What qualifies.** A user-reported issue or a capability the owner asked
  for, always. An observed failure once it has recurred (``min_count``). A
  capability the model merely said it lacked once it was asked twice. Never
  one marked won't-fix, and never one that failed less than
  ``retry_after_days`` ago.
- **What comes first** is what helps the owner most: what he reported, what
  makes her ignore him or answer slowly, then the most frequent; cosmetic
  things last (:func:`benefit`).
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from wawi.selfimprove import (
    KIND_AUDIO_BEHIND,
    KIND_CAPABILITY,
    KIND_SILENT_AUDIO,
    KIND_SLOW_TURN,
    KIND_TOOL_FAILED,
    KIND_TURN_FAILED,
    KIND_USER_REPORTED,
    KIND_UTTERANCE_DROPPED,
)

#: A line in an issue file that keeps it out of reach for good.
WONTFIX_RE = re.compile(r"won'?t[- ]?fix|wontfix|no arreglar", re.I)

_FIELD_RE = re.compile(r"^- \*\*([a-z ]+)\*\*: `?([^`\n]*)`?$", re.M)
_TIME_FMT = "%Y-%m-%d %H:%M:%S"


def plain(text: str) -> str:
    """Lowercase ASCII, accents dropped: «Ignoró» -> «ignoro»."""
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()


@dataclass
class IssueRecord:
    """One ``issues/*.md`` file, as selfimprove wrote it."""

    path: Path
    kind: str
    title: str
    count: int = 1
    first_seen: str = ""
    last_seen: str = ""
    fingerprint: str = ""
    detail: str = ""
    context: dict = field(default_factory=dict)
    wontfix: bool = False

    @property
    def id(self) -> str:
        return self.fingerprint or self.path.stem

    @property
    def from_user(self) -> bool:
        return self.kind == KIND_USER_REPORTED or str(self.context.get("source", "")) == "user"

    @property
    def is_capability(self) -> bool:
        return self.kind == KIND_CAPABILITY

    def rate_per_hour(self, now: datetime | None = None) -> float:
        """Occurrences per hour since it was first seen (at least one hour)."""
        try:
            first = datetime.strptime(self.first_seen, _TIME_FMT)
        except ValueError:
            return 0.0
        hours = max(1.0, ((now or datetime.now()) - first).total_seconds() / 3600.0)
        return self.count / hours

    def text(self) -> str:
        return f"{self.title}\n{self.detail}"


def parse_issue(path: Path) -> IssueRecord | None:
    """Parse one issue file; None when it is not one."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    title_match = re.search(r"^# (.+)$", text, re.M)
    if not title_match:
        return None
    fields = dict(_FIELD_RE.findall(text))
    try:
        count = max(1, int(fields.get("occurrences", "1")))
    except ValueError:
        count = 1
    detail = ""
    match = re.search(r"^## What happened\n\n(.*?)(?=\n## |\n---|\Z)", text, re.S | re.M)
    if match:
        detail = match.group(1).strip()
    context: dict = {}
    match = re.search(r"^## Context\n\n```json\n(.*?)\n```", text, re.S | re.M)
    if match:
        try:
            value = json.loads(match.group(1))
            context = value if isinstance(value, dict) else {}
        except ValueError:
            context = {}
    return IssueRecord(
        path=path,
        kind=fields.get("kind", "").strip() or path.stem.split("-")[0],
        title=title_match.group(1).strip(),
        count=count,
        first_seen=fields.get("first seen", ""),
        last_seen=fields.get("last seen", ""),
        fingerprint=fields.get("fingerprint", "").strip(),
        detail=detail,
        context=context,
        wontfix=bool(WONTFIX_RE.search(text)),
    )


def load_issues(directory: Path | None) -> list[IssueRecord]:
    """Every open issue in *directory* (``closed/`` and the like are skipped)."""
    if directory is None or not Path(directory).is_dir():
        return []
    out = []
    for path in sorted(Path(directory).glob("*.md")):
        record = parse_issue(path)
        if record is not None:
            out.append(record)
    return out


# --------------------------------------------------------------------------- #
# Benefit to the owner
# --------------------------------------------------------------------------- #

#: Kinds by how much they cost HIM. Being ignored beats being slow beats a
#: failed tool beats a capability the model alone thought of.
_KIND_BENEFIT = {
    KIND_USER_REPORTED: 100.0,
    KIND_UTTERANCE_DROPPED: 80.0,
    KIND_SILENT_AUDIO: 75.0,
    KIND_TURN_FAILED: 72.0,
    KIND_SLOW_TURN: 65.0,
    KIND_AUDIO_BEHIND: 55.0,
    KIND_TOOL_FAILED: 50.0,
    KIND_CAPABILITY: 45.0,
}

#: She ignores him, does not answer, or answers slowly: the worst there is.
_URGENT_WORDS = (
    "ignor", "no me escucha", "no escucha", "no contesta", "no responde", "no me contest",
    "no me oye", "sordo", "se corta", "cortad", "lent", "latencia", "tarda", "demora",
    "se cuelga", "crash", "se cae", "no anda", "falla",
)
#: Looks, not function.
_COSMETIC_WORDS = (
    "color", "icono", "ícono", "animacion", "estetic", "aspecto", "fuente", "tipografia",
    "sombra", "borde", "pixel", "sprite", "disfraz", "bonit", "lindo", "cosmetic",
)


def benefit(issue: IssueRecord) -> float:
    """Expected benefit of fixing *issue* to the owner. Higher first."""
    score = _KIND_BENEFIT.get(issue.kind, 40.0)
    if issue.is_capability and issue.from_user:
        score = 90.0
    low = plain(issue.text())
    if any(word in low for word in _URGENT_WORDS):
        score += 30.0
    if any(plain(word) in low for word in _COSMETIC_WORDS):
        score -= 45.0
    return score + 6.0 * math.log1p(max(0, issue.count))


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Verdict:
    eligible: bool
    reason: str = ""


def eligibility(issue: IssueRecord, settings: Any, history: dict[str, Any] | None,
                now: datetime | None = None) -> Verdict:
    """Whether *issue* may be attempted now. *history* is the store's entry
    for it (attempts, last outcome, won't-fix)."""
    now = now or datetime.now()
    history = history or {}
    if issue.wontfix or history.get("wontfix"):
        return Verdict(False, "wontfix")
    if history.get("blocked_on_owner"):
        return Verdict(False, "blocked_on_owner")
    if history.get("pending"):
        return Verdict(False, "pending_owner")
    if history.get("outcome") in ("applied", "watching"):
        # Fixed, unless it kept happening well after the fix went in.
        if issue.from_user or issue.count < int(history.get("count_at_apply", 0)) + settings.min_count:
            return Verdict(False, "already_fixed")
    last_failed = str(history.get("last_failed") or "")
    if last_failed:
        try:
            when = datetime.strptime(last_failed, _TIME_FMT)
        except ValueError:
            when = None
        if when is not None and now - when < timedelta(days=float(settings.retry_after_days)):
            return Verdict(False, "backoff")
    if issue.is_capability:
        if issue.from_user or issue.count >= settings.capability_min_count:
            return Verdict(True)
        return Verdict(False, "too_rare")
    if issue.from_user or issue.count >= settings.min_count:
        return Verdict(True)
    return Verdict(False, "too_rare")


def rank(issues: Iterable[IssueRecord], settings: Any, histories: dict[str, dict],
         now: datetime | None = None) -> list[IssueRecord]:
    """The eligible issues, most useful to the owner first."""
    eligible = [i for i in issues
                if eligibility(i, settings, histories.get(i.id), now).eligible]
    return sorted(eligible, key=lambda i: (-benefit(i), -i.count, i.title))


def match(issues: Iterable[IssueRecord], words: str) -> IssueRecord | None:
    """The issue the owner meant by *words* ("arreglate lo de la latencia")."""
    wanted = [w for w in re.findall(r"[a-z0-9]{3,}", plain(words))
              if w not in {"arreglate", "arregla", "que", "del", "las", "los", "eso", "esto"}]
    if not wanted:
        return None
    best, best_score = None, 0
    for issue in issues:
        hay = plain(issue.text())
        score = sum(1 for w in wanted if w in hay)
        if score > best_score:
            best, best_score = issue, score
    return best
