"""Git, for the self-fix loop: worktrees out of the repo, one squash commit in.

Every write to the owner's checkout goes through :func:`squash_apply` (after
checking it is on main with nothing uncommitted) or through
:func:`wawi.selffix.watchdog.revert_selffix`. Nothing here resets, forces,
rebases main or pushes; the only branch it ever deletes is its own
``selffix/<id>`` once the attempt is over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from wawi.selffix.watchdog import REVERT_TRAILER, TRAILER, Git, run_git

BRANCH_PREFIX = "selffix/"


class GitError(Exception):
    pass


def _ok(git: Git, repo: str | Path, *args: str) -> str:
    code, out, err = git(str(repo), *args)
    if code != 0:
        raise GitError(f"git {' '.join(args[:2])}: {(err or out).strip()[:300]}")
    return out


def head(repo: str | Path, git: Git = run_git) -> str:
    return _ok(git, repo, "rev-parse", "HEAD").strip()


def current_branch(repo: str | Path, git: Git = run_git) -> str:
    return _ok(git, repo, "rev-parse", "--abbrev-ref", "HEAD").strip()


def dirty(repo: str | Path, git: Git = run_git) -> list[str]:
    """Uncommitted changes, untracked files included (ignored ones are not)."""
    out = _ok(git, repo, "status", "--porcelain")
    return [line for line in out.splitlines() if line.strip()]


def can_apply(repo: str | Path, branch: str = "main", git: Git = run_git) -> str:
    """"" when the checkout may receive a self-fix, else why not."""
    try:
        if current_branch(repo, git) != branch:
            return "not_on_main"
        if dirty(repo, git):
            return "dirty_tree"
    except GitError as exc:
        return f"git_error: {exc}"
    return ""


def create_worktree(repo: str | Path, path: Path, branch: str, base: str,
                    git: Git = run_git) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ok(git, repo, "worktree", "add", "-b", branch, str(path), base)


def remove_worktree(repo: str | Path, path: Path, branch: str, git: Git = run_git) -> None:
    """Remove the attempt's worktree and its branch. Only ever ``selffix/*``."""
    if not branch.startswith(BRANCH_PREFIX):
        raise GitError(f"refusing to delete {branch!r}: not a self-fix branch")
    if path.exists():
        git(str(repo), "worktree", "remove", "--force", str(path))
    git(str(repo), "worktree", "prune")
    git(str(repo), "branch", "-D", branch)


def commit_all(worktree: Path, message: str, git: Git = run_git) -> str | None:
    """Commit everything the agent left in *worktree*. None when it changed nothing."""
    _ok(git, worktree, "add", "-A")
    if git(str(worktree), "diff", "--cached", "--quiet")[0] == 0:
        return None
    _ok(git, worktree, "commit", "--no-verify", "-m", message)
    return head(worktree, git)


def diff(repo: str | Path, base: str, tip: str, git: Git = run_git) -> str:
    return _ok(git, repo, "diff", "--no-color", "--no-ext-diff", f"{base}..{tip}")


@dataclass(frozen=True)
class FileStat:
    path: str
    added: int
    removed: int
    binary: bool = False


def numstat(repo: str | Path, base: str, tip: str, git: Git = run_git) -> list[FileStat]:
    out = _ok(git, repo, "diff", "--numstat", "--no-renames", f"{base}..{tip}")
    stats = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        a, r, path = parts
        binary = a == "-" or r == "-"
        stats.append(FileStat(path, 0 if binary else int(a), 0 if binary else int(r), binary))
    return stats


def rebase(worktree: Path, onto: str, git: Git = run_git) -> bool:
    """Bring the attempt's branch onto the current main. False (and the
    branch untouched) when it does not apply cleanly."""
    code, _, _ = git(str(worktree), "rebase", onto)
    if code != 0:
        git(str(worktree), "rebase", "--abort")
        return False
    return True


def is_ancestor(repo: str | Path, a: str, b: str, git: Git = run_git) -> bool:
    return git(str(repo), "merge-base", "--is-ancestor", a, b)[0] == 0


def commit_message(title: str, issue_id: str, summary: str = "") -> str:
    """The squash commit: Spanish, the issue named, and the trailer."""
    subject = f"Autoarreglo: {title}".strip()
    if len(subject) > 72:
        subject = subject[:71].rstrip() + "…"
    body = summary.strip()
    lines = [subject, ""]
    if body:
        lines += [body, ""]
    lines += [f"Arreglo hecho por Wawi sola para el issue «{title}».", "",
              f"{TRAILER}: {issue_id}"]
    return "\n".join(lines) + "\n"


def squash_apply(repo: str | Path, branch: str, message: str, *, main: str = "main",
                 git: Git = run_git) -> str:
    """Squash *branch* onto *main* as ONE commit. Returns its sha.

    Preconditions (checked, never forced): on *main*, nothing uncommitted, and
    *branch* already based on main's HEAD so the squash cannot conflict.
    """
    why = can_apply(repo, main, git)
    if why:
        raise GitError(why)
    if not is_ancestor(repo, "HEAD", branch, git):
        raise GitError("branch_not_on_main")
    _ok(git, repo, "merge", "--squash", "--no-commit", branch)
    code, _, err = git(str(repo), "commit", "--no-verify", "-m", message)
    if code != 0:
        raise GitError(f"commit_failed: {err.strip()[:200]}")
    return head(repo, git)


@dataclass(frozen=True)
class SelffixCommit:
    sha: str
    issue: str
    subject: str


def selffix_commits(repo: str | Path, limit: int = 200, git: Git = run_git) -> list[SelffixCommit]:
    """Self-fix commits on HEAD that were not reverted, newest first."""
    out = _ok(git, repo, "log", f"-{limit}", "--format=%H%x00%s%x00%B%x1e", "HEAD")
    rows, reverted = [], set()
    for record in out.split("\x1e"):
        parts = record.strip("\n").split("\x00", 2)
        if len(parts) != 3:
            continue
        sha, subject, body = parts
        for m in re.finditer(rf"^{REVERT_TRAILER}:\s*([0-9a-f]+)", body, re.M):
            reverted.add(m.group(1))
        m = re.search(rf"^{TRAILER}:\s*(\S+)", body, re.M)
        if m:
            rows.append(SelffixCommit(sha.strip(), m.group(1), subject))
    return [c for c in rows if c.sha not in reverted]
