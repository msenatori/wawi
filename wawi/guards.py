"""Permission tiers for the executor: ask about the dangerous few, not about everything.

Before this, every tool call the CLI consulted us about went to the user as a
spoken question. An ``ls`` and an ``rm -rf`` got the same sentence, and a
question asked about everything is a question answered "sí" without listening.
So the executor's calls are sorted here, in harness code, into four tiers:

* **READ** -- never asks. Read/Glob/Grep anywhere off the blocklist, and Bash
  built only from a closed list of read-only commands (``ls``, ``cat``,
  ``mdfind``, ``open -a``, ``git status``...). Logged, not announced.
* **WORK** -- local, reversible work: an edit inside the project, ``mkdir``,
  ``pytest``, a Bash command nobody listed. Asked ONCE; the "sí" then covers
  every WORK call of that same task for ``grant_minutes``.
* **RISKY** -- send, pay, delete, submit, transfer, push, overwrite, write
  outside the project, and their Spanish verbs. Asked EVERY time, with a
  sentence short enough to hear ("¿Borro 3 archivos?"), grant or no grant.
* **BLOCKED** -- password managers, the Keychain, System Settings, ``sudo``,
  and Wawi's own config. Denied without asking. Nothing turns this off.

Yolo skips RISKY questions for an hour and then switches itself off. It lives
in memory only, and the only ways to turn it on are code paths the user owns
(today: ``dangerous_skip_permissions`` at startup). No tool the executor has
reaches :meth:`PermissionGate.enable_yolo`, and an executor trying to flip the
config key is BLOCKED by the same classifier.

Why this is code and never prompt: the executor is the thing being guarded.
A rule it is told about is a rule it can talk itself out of. Unknown tools and
shapes the parser cannot read fall to the asking side, never to READ.
"""

from __future__ import annotations

import enum
import logging
import os
import pathlib
import re
import shlex
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

#: How long a "sí" to a WORK call keeps covering the rest of that task.
DEFAULT_GRANT_MINUTES = 30.0

#: Yolo is not configurable on purpose: an hour is long enough for a session
#: of work and short enough that nobody forgets it is on overnight.
YOLO_MINUTES = 60.0

#: Cap on a spoken RISKY question. Heard, not read: past this it is a speech.
SPOKEN_MAX_CHARS = 60


class Tier(enum.Enum):
    READ = "READ"
    WORK = "WORK"
    RISKY = "RISKY"
    BLOCKED = "BLOCKED"


class Verdict(enum.Enum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


@dataclass(frozen=True)
class Classification:
    """What a tool call is, before anyone's grants are taken into account."""

    tier: Tier
    why: str = ""
    #: The spoken question for RISKY, as ``{"es": ..., "en": ...}``.
    prompt: dict[str, str] | None = None


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    tier: Tier
    reason: str = ""
    prompt: dict[str, str] | None = None
    #: Which task the decision was made for. A "sí" that arrives after the
    #: task changed must not grant anything to the next one.
    epoch: int = 0


DENIED_BLOCKED = (
    "Denied: this is on Wawi's fixed blocklist (password managers, Keychain, "
    "System Settings, sudo, Wawi's own config). Do not retry it or work "
    "around it; tell the user it needs to be done by hand."
)


# --------------------------------------------------------------------------- #
# Spoken questions for RISKY calls -- each fits SPOKEN_MAX_CHARS
# --------------------------------------------------------------------------- #


def _q(es: str, en: str) -> dict[str, str]:
    return {"es": _fit(es), "en": _fit(en)}


def _fit(text: str) -> str:
    """Cut a question to SPOKEN_MAX_CHARS, keeping the closing "?"."""
    if len(text) <= SPOKEN_MAX_CHARS:
        return text
    return text[: SPOKEN_MAX_CHARS - 2].rstrip(" ,.") + "…?"


def _ask_delete(targets: list[str]) -> dict[str, str]:
    if len(targets) == 1:
        name = _speakable(targets[0])
        return _q(f"¿Borro {name}?", f"Should I delete {name}?")
    if targets:
        n = len(targets)
        return _q(f"¿Borro {n} archivos?", f"Should I delete {n} files?")
    return _q("¿Borro archivos?", "Should I delete files?")


_ASK_SEND = _q("¿Lo mando?", "Should I send it?")
_ASK_PAY = _q("¿Hago el pago?", "Should I make the payment?")
_ASK_BUY = _q("¿Hago la compra?", "Should I buy it?")
_ASK_TRANSFER = _q("¿Hago la transferencia?", "Should I make the transfer?")
_ASK_SUBMIT = _q("¿Envío el formulario?", "Should I submit it?")
_ASK_CONFIRM = _q("¿Confirmo?", "Should I confirm?")
_ASK_PUBLISH = _q("¿Lo publico?", "Should I publish it?")
_ASK_PUSH = _q("¿Hago git push?", "Should I git push?")
_ASK_DISCARD = _q("¿Descarto cambios de git?", "Should I discard git changes?")
_ASK_KILL = _q("¿Cierro un programa?", "Should I close a program?")
_ASK_POWER = _q("¿Apago o reinicio la Mac?", "Should I shut down or restart?")
_ASK_SCRIPT = _q("¿Corro un script sin leerlo?", "Should I run an unread script?")
_ASK_DISK = _q("¿Toco un disco?", "Should I touch a disk?")
_ASK_SYSTEM = _q("¿Cambio algo del sistema?", "Should I change a system setting?")


def _ask_overwrite(target: str) -> dict[str, str]:
    name = _speakable(target)
    return _q(f"¿Sobrescribo {name}?", f"Should I overwrite {name}?")


def _ask_outside(target: str) -> dict[str, str]:
    name = _speakable(target)
    return _q(f"¿Escribo {name} fuera del proyecto?", f"Should I write {name} outside the project?")


def _ask_network(host: str) -> dict[str, str]:
    host = _speakable(host) or "internet"
    return _q(f"¿Envío datos a {host}?", f"Should I send data to {host}?")


def _ask_tool(tool: str) -> dict[str, str]:
    return _q(f"¿Le dejo usar {tool}?", f"Should I let it use {tool}?")


def _speakable(path: str) -> str:
    """The part of a path worth saying: its last component, short."""
    name = os.path.basename(path.rstrip("/")) or path
    return name[:28]


#: Verb -> question, in English and Spanish (infinitive, voseo imperative and
#: the usual conjugations). Matched as whole words. Only applied to commands
#: whose arguments ARE behaviour (inline scripts, osascript, curl, unknown
#: binaries) -- never to a commit message or a grep pattern.
_RISK_VERBS: tuple[tuple[re.Pattern[str], dict[str, str]], ...] = tuple(
    (re.compile(rf"\b(?:{words})\b", re.IGNORECASE), question)
    for words, question in (
        (r"send|sends|sendmail|mandar|mandá|manda|mande|mando|enviar|envía|envia"
         r"|envíe|envie|envío|envio", _ASK_SEND),
        (r"pay|pays|payment|pagar|pagá|paga|pague|pago", _ASK_PAY),
        (r"buy|purchase|checkout|comprar|comprá|compra|compre", _ASK_BUY),
        (r"transfer|transferir|transferí|transfiere|transferencia", _ASK_TRANSFER),
        (r"submit|post", _ASK_SUBMIT),
        (r"confirm|confirmar|confirmá|confirma|confirme", _ASK_CONFIRM),
        (r"publish|publicar|publicá|publica|publique", _ASK_PUBLISH),
        (r"delete|remove|erase|destroy|unlink|rmtree|trash|borrar|borrá|borra"
         r"|borre|eliminar|eliminá|elimina|elimine|suprimir", _ask_delete([])),
        (r"shutdown|shut\s+down|restart|reboot|apagar|reiniciar", _ASK_POWER),
    )
)


# --------------------------------------------------------------------------- #
# The blocklist -- fixed, and checked before anything else
# --------------------------------------------------------------------------- #

#: Substrings (lowercased) that put any path or command on the blocklist.
_BLOCKED_TEXT: tuple[str, ...] = (
    "library/keychains",
    ".keychain",
    "keychain access",
    "1password",
    "bitwarden",
    "system settings",
    "system preferences",
    "systempreferences",
    "systemsettings",
    "com.apple.tcc",
    "tcc.db",
    # The config key behind yolo. An executor writing it is an executor
    # trying to grant itself the thing only the user may grant.
    "dangerous_skip_permissions",
)

#: Bash commands refused outright, by program name.
_BLOCKED_COMMANDS = frozenset({
    "sudo", "su", "doas", "security", "op", "bw", "tccutil", "csrutil",
    "passwd", "dscl", "systemsetup",
})

#: ``defaults`` is fine to read and blocked to write: it is System Settings
#: without the window.
_DEFAULTS_WRITES = frozenset({"write", "delete", "import", "rename"})


def _blocked_text(text: str) -> str:
    low = text.lower()
    for needle in _BLOCKED_TEXT:
        if needle in low:
            return needle
    return ""


# --------------------------------------------------------------------------- #
# Bash
# --------------------------------------------------------------------------- #

#: Programs that only look. Every segment of a pipeline must be one of these,
#: with no output redirection, for the command to be READ.
_READ_ONLY_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "file", "stat", "du", "df", "pwd",
    "echo", "printf", "which", "whereis", "type", "whoami", "id", "date", "cal",
    "uname", "sw_vers", "hostname", "uptime", "ps", "pgrep", "lsof", "env",
    "printenv", "grep", "egrep", "fgrep", "rg", "ag", "mdfind", "mdls", "tree",
    "sort", "uniq", "cut", "tr", "jq", "basename", "dirname", "realpath",
    "readlink", "less", "more", "diff", "cmp", "md5", "shasum", "true", "test",
    "[", "column", "nl", "fold", "strings", "otool", "system_profiler",
    "sysctl", "vm_stat", "pbpaste", "plutil", "xxd", "hexdump", "open", "cd",
    "ioreg", "networksetup", "scutil", "pmset", "launchctl", "defaults",
    "mdutil", "lipo", "codesign", "spctl",
})

#: Subcommands that make an otherwise-listed program stop being read-only.
_NOT_READ_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "launchctl": frozenset({"load", "unload", "bootstrap", "bootout", "kickstart",
                            "kill", "remove", "submit", "enable", "disable", "start",
                            "stop", "setenv", "unsetenv"}),
    "defaults": _DEFAULTS_WRITES,
    "pmset": frozenset({"-a", "-b", "-c", "-u", "sleepnow", "restoredefaults",
                        "schedule", "repeat", "displaysleepnow"}),
    "networksetup": frozenset(),  # every -set* is caught by the prefix check below
    "scutil": frozenset({"--set"}),
    "mdutil": frozenset({"-i", "-E", "-X"}),
    "plutil": frozenset({"-convert", "-insert", "-replace", "-remove", "-extract"}),
    "spctl": frozenset({"--master-disable", "--global-disable", "--add", "--remove",
                        "--enable", "--disable"}),
}

_GIT_READ = frozenset({
    "status", "log", "diff", "show", "rev-parse", "blame", "ls-files",
    "describe", "shortlog", "grep", "reflog", "remote", "branch", "tag",
    "config", "stash",
})

#: Programs whose arguments are data (a commit message, a sed script), so the
#: risk-verb scan is not run over them. Their risky shapes are listed by hand.
_LOCAL_WORK_COMMANDS = frozenset({
    "git", "mkdir", "touch", "cp", "mv", "ln", "chmod", "sed", "awk", "tee",
    "tar", "zip", "unzip", "gzip", "gunzip", "make", "pytest", "npm", "pnpm",
    "yarn", "npx", "pip", "pip3", "uv", "brew", "export", "cd", "ruff", "black",
    "mypy", "cargo", "go", "swift", "xcodebuild",
})

_DELETE_COMMANDS = frozenset({"rm", "rmdir", "unlink", "shred", "srm", "trash"})
_KILL_COMMANDS = frozenset({"kill", "killall", "pkill"})
_POWER_COMMANDS = frozenset({"shutdown", "reboot", "halt"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "fish"})
#: Wrappers that run the NEXT word as the command.
_WRAPPERS = frozenset({"nohup", "time", "command", "builtin", "exec", "nice",
                       "caffeinate", "timeout", "gtimeout"})

_SEPARATORS = ("&&", "||", "$(", ";", "|", "&", "\n", "(", ")", "`")
_REDIRECT = re.compile(r"(?<![<>&\d])(\d*)(>>?|&>)\s*([^\s;&|<>()]+)")
_FD_DUP = re.compile(r"\d*>&\d+-?")


def _segments(command: str) -> list[str]:
    """Split a shell line into simple commands, quote-aware.

    Not a shell parser, and it does not need to be one: every place it can
    be wrong (``$(`` inside double quotes, here-docs) only yields MORE
    segments, and an odd-looking segment classifies as WORK, not as READ.
    """
    segments: list[str] = []
    buf: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        c = command[i]
        if quote == "'":
            buf.append(c)
            if c == "'":
                quote = ""
            i += 1
            continue
        if c == "\\" and i + 1 < len(command):
            buf.append(command[i : i + 2])
            i += 2
            continue
        if quote == '"' and c == '"':
            quote = ""
            buf.append(c)
            i += 1
            continue
        if not quote and c in "'\"":
            quote = c
            buf.append(c)
            i += 1
            continue
        if quote == '"' and not (command.startswith("$(", i) or c == "`"):
            buf.append(c)
            i += 1
            continue
        for sep in _SEPARATORS:
            if command.startswith(sep, i):
                segments.append("".join(buf))
                buf = []
                quote = ""
                i += len(sep)
                break
        else:
            buf.append(c)
            i += 1
    segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()]


def _words(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.replace('"', " ").replace("'", " ").split()


def _program(words: list[str]) -> tuple[str, list[str]]:
    """The program a simple command really runs, and its arguments."""
    i = 0
    while i < len(words):
        word = words[i]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word):
            i += 1
            continue
        name = os.path.basename(word)
        if name in _WRAPPERS or name == "env":
            i += 1
            # Skip the wrapper's own flags and numeric args (timeout 10, nice -n 5).
            while i < len(words) and (words[i].startswith("-") or words[i].isdigit()):
                i += 1
            continue
        return name, words[i + 1 :]
    return "", []


def _args(args: list[str]) -> list[str]:
    return [a for a in args if not a.startswith("-")]


class _Bash:
    """Classify one Bash command. Worst segment wins."""

    def __init__(self, cwd: str, protected: tuple[pathlib.Path, ...]) -> None:
        self.cwd = cwd
        self.protected = protected

    def classify(self, command: str, depth: int = 0) -> Classification:
        if depth > 4:
            return Classification(Tier.RISKY, "nested too deep", _ASK_SCRIPT)
        needle = _blocked_text(command) or ("keychain" if "keychain" in command.lower() else "")
        if needle:
            return Classification(Tier.BLOCKED, f"mentions {needle!r}")

        worst = Classification(Tier.READ, "read-only command")
        writes = self._redirects(command)
        for target in writes:
            found = self._write_target(target, overwrite_risky=True)
            worst = _worse(worst, found)

        segments = _segments(_FD_DUP.sub(" ", command))
        for index, segment in enumerate(segments):
            piped = index > 0
            found = self._segment(_REDIRECT.sub(" ", segment), piped, depth)
            worst = _worse(worst, found)
            if worst.tier is Tier.BLOCKED:
                return worst

        if worst.tier is not Tier.READ:
            for token in _words(_REDIRECT.sub(" ", command)):
                if self._is_protected(token):
                    return Classification(Tier.BLOCKED, "touches Wawi's config")
        return worst

    # -- pieces ---------------------------------------------------------

    def _redirects(self, command: str) -> list[str]:
        targets = []
        for _fd, _op, target in _REDIRECT.findall(_FD_DUP.sub(" ", command)):
            target = target.strip("'\"")
            if target not in ("/dev/null", "/dev/stdout", "/dev/stderr"):
                targets.append(target)
        return targets

    def _segment(self, segment: str, piped: bool, depth: int) -> Classification:
        words = _words(segment)
        program, args = _program(words)
        if not program:
            return Classification(Tier.READ)
        plain = _args(args)

        if program in _BLOCKED_COMMANDS:
            return Classification(Tier.BLOCKED, f"{program} is blocklisted")
        if program == "defaults" and args and args[0] in _DEFAULTS_WRITES:
            return Classification(Tier.BLOCKED, "defaults write is System Settings")

        if program in _DELETE_COMMANDS:
            return Classification(Tier.RISKY, f"{program} deletes", _ask_delete(plain))
        if program in _KILL_COMMANDS:
            return Classification(Tier.RISKY, f"{program} closes programs", _ASK_KILL)
        if program in _POWER_COMMANDS:
            return Classification(Tier.RISKY, f"{program} powers off", _ASK_POWER)
        if program in ("dd", "mkfs", "fdisk", "newfs_apfs", "asr"):
            return Classification(Tier.RISKY, f"{program} writes disks", _ASK_DISK)
        if program == "diskutil":
            if args and args[0] in ("list", "info", "activity"):
                return Classification(Tier.READ)
            return Classification(Tier.RISKY, "diskutil changes disks", _ASK_DISK)
        if program == "crontab" and "-l" not in args:
            return Classification(Tier.RISKY, "crontab changes the system", _ASK_SYSTEM)

        if program in _SHELLS:
            if "-c" in args:
                inner = args[args.index("-c") + 1 :]
                return self.classify(inner[0] if inner else "", depth + 1)
            if not plain:
                # `curl ... | sh`: a script nobody read, straight into a shell.
                return Classification(Tier.RISKY, "shell reading stdin", _ASK_SCRIPT)
            return self._verbs(segment, Classification(Tier.WORK, f"runs {plain[0]}"))
        if program == "eval":
            return self.classify(" ".join(args), depth + 1)
        if program == "xargs":
            rest = args
            while rest and rest[0].startswith("-"):
                # Flags that take a value (-I {}, -n 1) eat the next word too.
                rest = rest[2:] if rest[0] in ("-I", "-n", "-P", "-L", "-J") else rest[1:]
            return self.classify(shlex.join(rest), depth + 1) if rest else Classification(Tier.READ)
        if program == "find":
            return self._find(args, depth)
        if program == "git":
            return self._git(args)
        if program in ("curl", "wget", "http", "https"):
            return self._curl(program, args)
        if program in ("cp", "mv", "ln", "install", "rsync", "ditto"):
            return self._copy(program, plain)
        if program in ("mkdir", "touch", "chmod", "chown", "tee", "truncate"):
            found = Classification(Tier.WORK, f"{program} writes")
            for target in plain:
                found = _worse(found, self._write_target(target, overwrite_risky=False))
            return found
        if program == "sed" and any(a.startswith("-i") for a in args):
            return Classification(Tier.WORK, "sed -i edits in place")

        if program in _READ_ONLY_COMMANDS:
            banned = _NOT_READ_SUBCOMMANDS.get(program, frozenset())
            if any(a in banned for a in args) or (
                program == "networksetup" and any(a.startswith("-set") for a in args)
            ):
                return Classification(Tier.RISKY, f"{program} changes the system", _ASK_SYSTEM)
            if program == "open":
                # Opening a URL with data in it is still opening a URL; the
                # risky part would be what happens next, which is the user's.
                return Classification(Tier.READ, "opens something")
            return Classification(Tier.READ)

        if program in _LOCAL_WORK_COMMANDS:
            return Classification(Tier.WORK, f"runs {program}")
        return self._verbs(segment, Classification(Tier.WORK, f"runs {program}"))

    def _verbs(self, text: str, default: Classification) -> Classification:
        for pattern, question in _RISK_VERBS:
            match = pattern.search(text)
            if match:
                return Classification(Tier.RISKY, f"says {match.group(0)!r}", question)
        return default

    def _find(self, args: list[str], depth: int) -> Classification:
        if "-delete" in args:
            return Classification(Tier.RISKY, "find -delete", _ask_delete([]))
        for flag in ("-exec", "-execdir", "-ok", "-okdir"):
            if flag in args:
                inner = args[args.index(flag) + 1 :]
                inner = [w for w in inner if w not in (";", "\\;", "+", "{}")]
                return self.classify(shlex.join(inner), depth + 1)
        return Classification(Tier.READ)

    def _git(self, args: list[str]) -> Classification:
        plain = _args(args)
        sub = plain[0] if plain else ""
        if sub == "push":
            return Classification(Tier.RISKY, "git push", _ASK_PUSH)
        if sub == "reset" and "--hard" in args:
            return Classification(Tier.RISKY, "git reset --hard", _ASK_DISCARD)
        if sub == "clean" or (sub in ("checkout", "restore") and ("--" in args or "." in plain[1:])):
            return Classification(Tier.RISKY, f"git {sub} discards work", _ASK_DISCARD)
        if sub == "branch" and any(a in ("-d", "-D", "--delete", "-m", "-M") for a in args):
            return Classification(Tier.RISKY, "git branch delete", _ASK_DISCARD)
        if sub == "stash" and len(plain) > 1 and plain[1] in ("drop", "clear"):
            return Classification(Tier.RISKY, "git stash drop", _ASK_DISCARD)
        if sub in ("filter-branch", "filter-repo"):
            return Classification(Tier.RISKY, f"git {sub}", _ASK_DISCARD)
        if sub in _GIT_READ:
            mutating = {
                "remote": len(plain) > 1,
                "tag": len(plain) > 1,
                "branch": len(plain) > 1,
                "config": not any(a in ("--get", "--list", "-l", "--get-all") for a in args),
                "stash": not (len(plain) > 1 and plain[1] in ("list", "show")),
            }
            if not mutating.get(sub, False):
                return Classification(Tier.READ)
        return Classification(Tier.WORK, f"git {sub}")

    def _curl(self, program: str, args: list[str]) -> Classification:
        host = next((re.sub(r"^\w+://", "", a).split("/")[0] for a in args
                     if "://" in a), "")
        sends = {"-d", "--data", "--data-raw", "--data-binary", "--data-urlencode",
                 "-F", "--form", "-T", "--upload-file", "--json", "--post-data",
                 "--post-file"}
        for i, arg in enumerate(args):
            key = arg.split("=", 1)[0]
            if key in sends:
                return Classification(Tier.RISKY, f"{program} sends data", _ask_network(host))
            if key in ("-X", "--request", "--method"):
                method = (args[i + 1] if "=" not in arg and i + 1 < len(args)
                          else arg.split("=", 1)[-1]).upper()
                if method not in ("GET", "HEAD", "OPTIONS"):
                    return Classification(Tier.RISKY, f"{program} {method}", _ask_network(host))
        if any(a in ("-o", "-O", "--output", "--remote-name") or a.startswith("-o")
               for a in args) or program == "wget":
            return Classification(Tier.WORK, f"{program} downloads a file")
        return Classification(Tier.READ, f"{program} reads a page")

    def _copy(self, program: str, plain: list[str]) -> Classification:
        if len(plain) < 2:
            return Classification(Tier.WORK, f"{program}")
        found = self._write_target(plain[-1], overwrite_risky=True)
        if program == "mv":
            # A move empties its source: outside the project, that is a delete.
            for source in plain[:-1]:
                found = _worse(found, self._write_target(source, overwrite_risky=False))
        return found

    def _write_target(self, target: str, *, overwrite_risky: bool) -> Classification:
        if _blocked_text(target) or self._is_protected(target):
            return Classification(Tier.BLOCKED, f"writes {target}")
        path = _resolve(target, self.cwd)
        if path is None or not _inside_workspace(path, self.cwd):
            return Classification(Tier.RISKY, f"writes outside: {target}", _ask_outside(target))
        if overwrite_risky and path.is_file():
            return Classification(Tier.RISKY, f"overwrites {target}", _ask_overwrite(target))
        return Classification(Tier.WORK, f"writes {target}")

    def _is_protected(self, token: str) -> bool:
        if not token or token.startswith("-"):
            return False
        path = _resolve(token, self.cwd)
        return path is not None and path in self.protected


_ORDER = {Tier.READ: 0, Tier.WORK: 1, Tier.RISKY: 2, Tier.BLOCKED: 3}


def _worse(a: Classification, b: Classification) -> Classification:
    return b if _ORDER[b.tier] > _ORDER[a.tier] else a


def _resolve(target: str, cwd: str) -> pathlib.Path | None:
    try:
        path = pathlib.Path(os.path.expanduser(target))
        if not path.is_absolute():
            path = pathlib.Path(cwd or os.getcwd()) / path
        return path.resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def _temp_roots() -> tuple[pathlib.Path, ...]:
    roots = {pathlib.Path("/tmp"), pathlib.Path(tempfile.gettempdir())}
    out = []
    for root in roots:
        try:
            out.append(root.resolve())
        except OSError:
            continue
    return tuple(out)


def _inside_workspace(path: pathlib.Path, cwd: str) -> bool:
    """Inside the project, or in a temp dir -- scratch space is not "outside"."""
    roots = list(_temp_roots())
    if cwd:
        try:
            roots.append(pathlib.Path(cwd).resolve())
        except OSError:
            pass
    return any(path == root or path.is_relative_to(root) for root in roots)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

_READ_TOOLS = frozenset({"Read", "Glob", "Grep", "LS", "WebFetch", "WebSearch",
                         "TodoWrite"})
_WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})


def default_protected() -> tuple[pathlib.Path, ...]:
    """Wawi's own config files: where it was started, and next to the code,
    both in the private folder (wawi/paths.py) and where older versions kept it."""
    from wawi import paths as _paths

    cwd = pathlib.Path.cwd()
    repo = pathlib.Path(__file__).resolve().parents[1]
    candidates = {
        cwd / "config.yaml",
        repo / "config.yaml",
        _paths.under(cwd) / "config.yaml",
        _paths.under(repo) / "config.yaml",
    }
    out = []
    for candidate in candidates:
        try:
            out.append(candidate.resolve())
        except OSError:
            continue
    return tuple(out)


def classify(
    tool: str,
    tool_input: dict,
    cwd: str,
    protected: tuple[pathlib.Path, ...] = (),
) -> Classification:
    """Sort one executor tool call into a :class:`Tier`. Pure; no grants."""
    tool_input = tool_input or {}
    if tool.startswith("mcp__wawi_task__"):
        return Classification(Tier.READ, "the executor talking to us")

    if tool == "Bash":
        command = str(tool_input.get("command", "") or "")
        return _Bash(cwd, protected).classify(command)

    if tool in _READ_TOOLS:
        # Grep's own pattern is a regex, not a target: searching FOR the word
        # "1Password" is not opening 1Password.
        keys = ("file_path", "path", "glob", "url") + (("pattern",) if tool == "Glob" else ())
        for key in keys:
            value = tool_input.get(key)
            if isinstance(value, str) and _blocked_text(value):
                return Classification(Tier.BLOCKED, f"{tool} on the blocklist")
        return Classification(Tier.READ)

    if tool in _WRITE_TOOLS:
        target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        bash = _Bash(cwd, protected)
        if not target:
            return Classification(Tier.WORK, f"{tool} with no path")
        if _blocked_text(target) or bash._is_protected(target):
            return Classification(Tier.BLOCKED, f"{tool} {target}")
        if _blocked_text(str(tool_input.get("content", "")) + str(tool_input.get("new_string", ""))):
            return Classification(Tier.BLOCKED, f"{tool} writes a blocklisted key")
        return bash._write_target(target, overwrite_risky=False)

    # A tool nobody classified asks every time: fail closed, not open.
    return Classification(Tier.RISKY, f"unknown tool {tool}", _ask_tool(tool))


# --------------------------------------------------------------------------- #
# The gate -- classification plus the two kinds of standing permission
# --------------------------------------------------------------------------- #


class PermissionGate:
    """Holds the task-scoped grant and the yolo switch, and decides.

    Thread-safe: :meth:`decide` runs on the executor's thread, while
    :meth:`revoke_all` may come from the listener or the orchestrator.

    *clock* is injectable so expiry is testable without sleeping for an hour.
    """

    def __init__(
        self,
        *,
        grant_minutes: float = DEFAULT_GRANT_MINUTES,
        protected: tuple[pathlib.Path, ...] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._grant_s = max(0.0, float(grant_minutes)) * 60.0
        self._protected = default_protected() if protected is None else tuple(protected)
        self._clock = clock
        self._lock = threading.Lock()
        self._epoch = 0
        self._grant_until = 0.0
        self._yolo_until = 0.0

    # -- standing permissions -------------------------------------------

    def begin_task(self) -> None:
        """A new task starts with no grant. A "sí" never carries over."""
        with self._lock:
            self._epoch += 1
            self._grant_until = 0.0

    def end_task(self) -> None:
        with self._lock:
            self._epoch += 1
            self._grant_until = 0.0

    def enable_yolo(self) -> None:
        """Skip RISKY questions for YOLO_MINUTES. USER-OWNED code paths only.

        Nothing the executor can call reaches this; see the module docstring.
        """
        with self._lock:
            self._yolo_until = self._clock() + YOLO_MINUTES * 60.0
        log.warning("permissions: YOLO on for %.0f min -- risky actions run "
                    "without asking (the blocklist still applies)", YOLO_MINUTES)

    def disable_yolo(self) -> None:
        with self._lock:
            self._yolo_until = 0.0

    def revoke_all(self) -> None:
        """Panic: drop the grant and yolo at once."""
        with self._lock:
            self._epoch += 1
            self._grant_until = 0.0
            self._yolo_until = 0.0
        log.info("permissions: every standing grant revoked")

    @property
    def yolo_active(self) -> bool:
        with self._lock:
            return self._clock() < self._yolo_until

    @property
    def grant_active(self) -> bool:
        with self._lock:
            return self._clock() < self._grant_until

    # -- deciding -------------------------------------------------------

    def decide(self, tool: str, tool_input: dict, cwd: str) -> Decision:
        found = classify(tool, tool_input, cwd, self._protected)
        with self._lock:
            now = self._clock()
            epoch = self._epoch
            yolo = now < self._yolo_until
            granted = now < self._grant_until
        if found.tier is Tier.BLOCKED:
            log.warning("permissions: BLOCKED %s (%s)", tool, found.why)
            return Decision(Verdict.DENY, found.tier, DENIED_BLOCKED, epoch=epoch)
        if found.tier is Tier.READ:
            log.info("permissions: read %s (%s)", tool, found.why or "read-only")
            return Decision(Verdict.ALLOW, found.tier, epoch=epoch)
        if yolo:
            log.warning("permissions: yolo allowed %s %s (%s)", found.tier.value, tool, found.why)
            return Decision(Verdict.ALLOW, found.tier, epoch=epoch)
        if found.tier is Tier.WORK and granted:
            log.info("permissions: granted %s (%s)", tool, found.why)
            return Decision(Verdict.ALLOW, found.tier, epoch=epoch)
        return Decision(Verdict.ASK, found.tier, found.why, found.prompt, epoch)

    def approved(self, decision: Decision) -> None:
        """The user said yes. A WORK "sí" opens the timed grant; RISKY never does."""
        if decision.tier is not Tier.WORK or self._grant_s <= 0:
            return
        with self._lock:
            if decision.epoch != self._epoch:
                return
            self._grant_until = self._clock() + self._grant_s
