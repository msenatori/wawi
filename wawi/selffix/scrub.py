"""Secrets out of what she sends, and out of what she merges.

Two directions, one list of patterns:

- :func:`scrub` before a log excerpt goes into a coding agent's brief: tokens,
  keys, e-mail addresses and the owner's home path become placeholders.
- :func:`scan_diff` over the ``+`` lines of a candidate diff: anything that
  looks like a credential or a personal path rejects the attempt.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: (name, pattern). Ordered from most to least specific.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("assigned_secret", re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|passw(?:or)?d|auth)\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]")),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@")),
)

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ANY_HOME = re.compile(r"/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+")
#: Private LAN addresses: the owner's network layout is personal too.
_LAN_IP = re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b")

#: Addresses that are fine in code (examples, the attribution trailer).
_EMAIL_ALLOW = ("noreply@anthropic.com", "@example.com", "@example.org", "@users.noreply.github.com")


def home() -> str:
    return str(Path.home())


def scrub(text: str, *, home_dir: str | None = None) -> str:
    """*text* with credentials, e-mail addresses, LAN IPs and home paths
    replaced by placeholders. For anything that leaves for a coding agent."""
    out = text or ""
    for name, pattern in SECRET_PATTERNS:
        out = pattern.sub(f"<{name}>", out)
    out = _EMAIL.sub("<email>", out)
    h = home_dir or home()
    if h and h != "/":
        out = out.replace(h, "~")
    out = _ANY_HOME.sub("~", out)
    return _LAN_IP.sub("<lan-ip>", out)


def added_lines(diff: str) -> list[tuple[str, str]]:
    """(file, line) for every added line of a unified diff."""
    current = ""
    rows = []
    for line in (diff or "").splitlines():
        if line.startswith("+++ "):
            current = line[4:].strip()
            if current.startswith("b/"):
                current = current[2:]
            continue
        if line.startswith("+") and not line.startswith("+++"):
            rows.append((current, line[1:]))
    return rows


def scan_diff(diff: str, *, home_dir: str | None = None, user: str | None = None) -> list[str]:
    """What looks secret or personal in the added lines. [] = clean."""
    found = []
    h = home_dir or home()
    user = user or os.environ.get("USER", "") or Path(h).name
    for path, line in added_lines(diff):
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                found.append(f"{path}: {name}")
        for email in _EMAIL.findall(line):
            if not any(email.endswith(a) or email == a for a in _EMAIL_ALLOW):
                found.append(f"{path}: email")
        if (h and h != "/" and h in line) or _ANY_HOME.search(line):
            found.append(f"{path}: ruta personal")
        elif user and len(user) >= 4 and re.search(rf"\b{re.escape(user)}\b", line):
            found.append(f"{path}: nombre de usuario")
        if _LAN_IP.search(line):
            found.append(f"{path}: IP de la red local")
    return list(dict.fromkeys(found))
