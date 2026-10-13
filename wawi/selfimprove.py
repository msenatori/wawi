"""Self-improvement by filing issues -- never by editing code.

The boundary, and why it is structural
--------------------------------------
An assistant that edits its own source is how you get a system that breaks
itself in a way you cannot review, because the component that would have told
you is the one that broke. So this module cannot do that. It has no editor, no
patcher, no shell: the only verbs it knows are "write a markdown file" and
"call ``gh issue create``". Making the boundary a property of the code, rather
than a rule in a prompt, is the whole point -- a prompt can be argued with.

Changing code stays a thing the user asks for explicitly, in a normal session,
looking at a diff.

What it reports
---------------
Two sources, and both matter:

- **Observed**: the pipeline reporting on itself. A turn that errored, audio
  that came back silent, an utterance dropped because a turn was already in
  flight, a wake check that took ten times its usual budget. These are the
  failures the user never sees -- they show up as "no me contestó" and then
  vanish, because nothing wrote them down.
- **Reported**: the user saying that an answer was wrong. This is the signal
  that cannot be derived from telemetry: the turn *succeeded* mechanically and
  was still useless. Without it there is nothing to optimize against.

Deduplication
-------------
The same failure recurs every few minutes, and an assistant that files an issue
each time is noise that gets muted -- which costs you the signal too. Issues
are keyed by kind plus a fingerprint of their subject; a repeat bumps a counter
on the existing file instead of creating another one. The count is the useful
part anyway: "this happened 40 times today" is a different bug report from
"this happened once".
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_DIR = Path("issues")

# Kinds the pipeline can observe about itself. Keeping them closed makes the
# journal groupable and stops free-text kinds from fragmenting the dedup key.
KIND_TURN_FAILED = "turn-failed"
KIND_SILENT_AUDIO = "silent-audio"
KIND_UTTERANCE_DROPPED = "utterance-dropped"
KIND_SLOW_TURN = "slow-turn"
KIND_TOOL_FAILED = "tool-failed"
KIND_AUDIO_BEHIND = "audio-behind"
KIND_USER_REPORTED = "user-reported"

_KNOWN_KINDS = frozenset(
    {
        KIND_TURN_FAILED,
        KIND_SILENT_AUDIO,
        KIND_UTTERANCE_DROPPED,
        KIND_SLOW_TURN,
        KIND_TOOL_FAILED,
        KIND_AUDIO_BEHIND,
        KIND_USER_REPORTED,
    }
)

# A turn slower than this is worth recording. Set from measurement, not taste:
# observed turns ran 4.4-16.8 s, so this flags the tail without crying about
# an ordinary answer.
SLOW_TURN_S = 20.0

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, limit: int = 40) -> str:
    slug = _SLUG_RE.sub("-", (text or "").lower()).strip("-")
    return (slug[:limit].rstrip("-")) or "issue"


def _fingerprint(kind: str, subject: str) -> str:
    """Stable short key for deduplication."""
    raw = f"{kind}|{' '.join((subject or '').lower().split())}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


@dataclass
class Issue:
    """One recorded problem."""

    kind: str
    title: str
    detail: str = ""
    context: dict = field(default_factory=dict)
    count: int = 1
    first_seen: str = ""
    last_seen: str = ""
    fingerprint: str = ""

    def to_markdown(self, assistant_name: str = "Wawi") -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- **kind**: `{self.kind}`",
            f"- **occurrences**: {self.count}",
            f"- **first seen**: {self.first_seen}",
            f"- **last seen**: {self.last_seen}",
            f"- **fingerprint**: `{self.fingerprint}`",
            "",
        ]
        if self.detail:
            lines += ["## What happened", "", self.detail, ""]
        if self.context:
            lines += [
                "## Context",
                "",
                "```json",
                json.dumps(self.context, indent=2, ensure_ascii=False, default=str),
                "```",
                "",
            ]
        lines += [
            "---",
            f"_Filed automatically by {assistant_name}. No code was changed._",
            "",
        ]
        return "\n".join(lines)


class IssueJournal:
    """Writes and de-duplicates issues on disk. Optionally files them on GitHub.

    Thread-safe: observations arrive from the pipeline worker, the listener
    worker and the refresh thread.
    """

    def __init__(
        self,
        directory: Path | None = DEFAULT_DIR,
        assistant_name: str = "Wawi",
        github: bool = False,
    ) -> None:
        self.directory = Path(directory) if directory else None
        self.assistant_name = assistant_name
        self.github = github
        self._lock = threading.Lock()
        self._issues: dict[str, Issue] = {}

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def issues(self) -> list[Issue]:
        with self._lock:
            return sorted(self._issues.values(), key=lambda i: (-i.count, i.kind))

    def record(
        self,
        kind: str,
        title: str,
        detail: str = "",
        context: dict | None = None,
        subject: str = "",
    ) -> Issue | None:
        """Record one problem, merging into an existing issue if it repeats.

        Args:
            kind: One of the ``KIND_*`` constants. Unknown kinds are accepted
                but logged, so a typo surfaces instead of silently creating a
                parallel category.
            title: One line, human first.
            detail: Free text; what happened and what was expected.
            context: Structured extras (timings, transcripts, tool names).
            subject: What makes this occurrence the *same* as another. Defaults
                to the title. Pass something stable -- a tool name, an error
                class -- not something that varies per occurrence.

        Returns:
            The stored issue, or None when journalling is disabled.
        """
        if not self.enabled:
            return None
        if kind not in _KNOWN_KINDS:
            log.warning("Issue journal: unknown kind %r (still recorded)", kind)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        key = _fingerprint(kind, subject or title)

        with self._lock:
            issue = self._issues.get(key)
            if issue is None:
                issue = Issue(
                    kind=kind,
                    title=title.strip() or kind,
                    detail=detail.strip(),
                    context=dict(context or {}),
                    count=1,
                    first_seen=now,
                    last_seen=now,
                    fingerprint=key,
                )
                self._issues[key] = issue
                new = True
            else:
                issue.count += 1
                issue.last_seen = now
                if context:
                    issue.context.update(context)
                new = False

        self._write(issue)
        if new:
            log.info("Issue filed: [%s] %s", issue.kind, issue.title)
            if self.github:
                self._file_on_github(issue)
        else:
            log.debug("Issue recurred (%dx): %s", issue.count, issue.title)
        return issue

    def _path_for(self, issue: Issue) -> Path:
        assert self.directory is not None
        return self.directory / f"{issue.kind}-{_slug(issue.title)}-{issue.fingerprint}.md"

    def _write(self, issue: Issue) -> None:
        """Write the issue file. Never raises: a failed write is not fatal."""
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._path_for(issue).write_text(
                issue.to_markdown(self.assistant_name), encoding="utf-8"
            )
        except Exception:
            log.warning("Could not write issue file for %r", issue.title, exc_info=True)

    def _file_on_github(self, issue: Issue) -> None:
        """Open a GitHub issue via `gh`. Best effort, never blocking for long.

        Only ever CREATES an issue. It cannot push, edit files, or open a pull
        request -- the boundary this module exists to hold.
        """
        try:
            result = subprocess.run(
                [
                    "gh", "issue", "create",
                    "--title", issue.title,
                    "--body", issue.to_markdown(self.assistant_name),
                    "--label", f"wawi:{issue.kind}",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode == 0:
                log.info("GitHub issue created: %s", result.stdout.strip())
            else:
                log.warning("gh issue create failed: %s", result.stderr.strip()[:200])
        except FileNotFoundError:
            log.warning("gh CLI not installed; issue kept on disk only")
        except Exception:
            log.warning("gh issue create errored; issue kept on disk only", exc_info=True)

    def prompt_hint(self, limit: int = 3) -> str:
        """Top recurring issues, for the system prompt. "" when there are none.

        Gives the assistant awareness of its OWN failures, which is what makes
        "¿querés que levante un agente para arreglar esto?" a grounded offer
        rather than an invented one. Kept to a few lines: this is re-sent every
        turn, and the long tail is in the files.
        """
        issues = self.issues()[:limit]
        if not issues:
            return ""
        body = "\n".join(f"- ({i.count}x) {i.title}" for i in issues)
        return (
            "Problemas que vos mismo registraste (están como issues en disco):\n"
            f"{body}"
        )

    def summary(self) -> str:
        """One line per issue, most frequent first. For the user to read."""
        issues = self.issues()
        if not issues:
            return "Sin problemas registrados."
        return "\n".join(f"{i.count:>3}x  [{i.kind}] {i.title}" for i in issues)
