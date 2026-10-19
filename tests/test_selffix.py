"""Autoarreglo (wawi/selffix/): selection, brief, gates, classifier, apply,
watchdog, learning, controls and the optional executors.

Every git operation runs against a throwaway repository under tmp_path, and
every coding agent, reviewer, test run, Orca call and ssh call is a fake: no
test here touches the live repository, starts an agent or opens a socket.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from wawi import selfimprove
from wawi.selffix import brief as B
from wawi.selffix import classify as C
from wawi.selffix import gates as G
from wawi.selffix import gitops, issues as I, report as R
from wawi.selffix import tools as T
from wawi.selffix import watchdog as W
from wawi.selffix.engines import Job, JobResult, LocalExecutor, OrcaExecutor
from wawi.selffix.remote import RemoteError, RemoteExecutor
from wawi.selffix.scrub import scan_diff, scrub
from wawi.selffix.service import SelffixService
from wawi.selffix.settings import RemoteSettings, SelffixSettings
from wawi.selffix.state import Store
from wawi.selffix.ui import SelffixMenu

NOW = datetime(2026, 10, 17, 15, 0, 0)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repo(tmp_path: Path, _template_repo: Path) -> Path:
    """A fresh copy of a one-commit repo on main (copying beats re-running git)."""
    root = tmp_path / "repo"
    shutil.copytree(_template_repo, root)
    return root


@pytest.fixture(scope="session")
def _template_repo(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("template") / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "wawi@example.com")
    git(root, "config", "user.name", "Wawi Test")
    git(root, "config", "commit.gpgsign", "false")
    (root / "wawi").mkdir()
    (root / "tests").mkdir()
    (root / "wawi" / "thing.py").write_text("def answer():\n    return 41\n")
    (root / "tests" / "test_thing.py").write_text("def test_ok():\n    assert True\n")
    (root / "requirements.txt").write_text("numpy==2.0.0\n")
    (root / ".gitignore").write_text("*.log\nselffix_local.json\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "Inicial")
    return root


def issue_file(directory: Path, kind: str, title: str, count: int = 1, detail: str = "",
               source: str = "", first_seen: str = "2026-10-16 10:00:00") -> I.IssueRecord:
    journal = selfimprove.IssueJournal(directory=directory)
    issue = None
    for _ in range(count):
        issue = journal.record(kind, title, detail=detail, subject=title,
                               context={"source": source} if source else None)
    path = journal._path_for(issue)
    text = path.read_text().replace(f"- **first seen**: {issue.first_seen}",
                                    f"- **first seen**: {first_seen}")
    path.write_text(text)
    return I.parse_issue(path)


class FakeAgents:
    """Plays author, test runner and reviewer.

    *edit(worktree)* is what the author does; *tests_code* what pytest
    returns; *verdict* what the reviewer says; *report* the author's report.
    """

    def __init__(self, edit=None, tests_code=0, verdict="VERDICT: APPROVE",
                 report="COMMIT: Arreglar la respuesta\nCambié el número.\nOWNER_IMPACT: none") -> None:
        self.edit = edit or (lambda wt: (wt / "wawi" / "thing.py").write_text(
            "def answer():\n    return 42\n"))
        self.tests_code = tests_code
        self.verdict = verdict
        self.report = report
        self.jobs: list[Job] = []

    def run(self, job: Job) -> JobResult:
        self.jobs.append(job)
        job.dir.mkdir(parents=True, exist_ok=True)
        last = Path(job.argv[job.argv.index("-o") + 1]) if "-o" in job.argv else None
        if job.name == "author":
            self.edit(job.cwd)
            if last:
                last.write_text(self.report)
            return JobResult(0, output=self.report)
        if job.name == "tests":
            return JobResult(self.tests_code, output="1 passed" if self.tests_code == 0 else "1 failed")
        if job.name == "review":
            if last:
                last.write_text(f"- todo bien\n{self.verdict}")
            return JobResult(0, output=f"- revisado\n{self.verdict}")
        raise AssertionError(job.name)


def make_service(tmp_path: Path, repo: Path, agents: FakeAgents | None = None, *,
                 settings: dict | None = None, restart=("/x/wawictl", "restart"),
                 **kw) -> SelffixService:
    agents = agents or FakeAgents()
    s = SelffixSettings.from_dict({"work_dir": str(tmp_path / "work"), "use_orca": False,
                                   **(settings or {})})
    bubbles: list[str] = []
    spawned: list[list[str]] = []
    service = SelffixService(
        s, repo=repo, issues_dir=tmp_path / "issues", log_path=tmp_path / "wawi.log",
        store=Store(tmp_path / "selffix_local.json"), python="python3",
        author_executor=agents, local=agents, which=lambda e: f"/usr/bin/{e}",
        notify=bubbles.append, restart_cmd=lambda: list(restart) if restart else None,
        spawn=spawned.append, power=lambda: True, meeting=lambda: False,
        keyboard_idle=lambda: 10_000.0, pure_python=lambda n, v: True,
        now=lambda: NOW, clock=lambda: 1_000_000.0, **kw)
    service._last_activity = -1e9   # no turn for ages
    service.bubbles = bubbles
    service.spawned = spawned
    service.agents = agents
    return service


def user_issue(tmp_path: Path, title="Respuestas demasiado lentas", **kw) -> I.IssueRecord:
    return issue_file(tmp_path / "issues", selfimprove.KIND_USER_REPORTED, title,
                      source="user", **kw)


# --------------------------------------------------------------------------- #
# Selection and quotas
# --------------------------------------------------------------------------- #


class TestSelection:
    def settings(self):
        return SelffixSettings.from_dict({})

    def test_user_reported_qualifies_at_once(self, tmp_path):
        issue = user_issue(tmp_path)
        assert I.eligibility(issue, self.settings(), {}, NOW).eligible

    def test_observed_needs_three(self, tmp_path):
        d = tmp_path / "issues"
        two = issue_file(d, selfimprove.KIND_TURN_FAILED, "Turno fallado A", count=2)
        three = issue_file(d, selfimprove.KIND_TURN_FAILED, "Turno fallado B", count=3)
        assert I.eligibility(two, self.settings(), {}, NOW).reason == "too_rare"
        assert I.eligibility(three, self.settings(), {}, NOW).eligible

    def test_wontfix_in_file_or_history(self, tmp_path):
        issue = user_issue(tmp_path)
        assert I.eligibility(issue, self.settings(), {"wontfix": True}, NOW).reason == "wontfix"
        issue.path.write_text(issue.path.read_text() + "\nwon't fix\n")
        assert I.parse_issue(issue.path).wontfix

    def test_backoff_seven_days(self, tmp_path):
        issue = user_issue(tmp_path)
        recent = {"last_failed": (NOW - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")}
        old = {"last_failed": (NOW - timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S")}
        assert I.eligibility(issue, self.settings(), recent, NOW).reason == "backoff"
        assert I.eligibility(issue, self.settings(), old, NOW).eligible

    def test_capabilities(self, tmp_path):
        d = tmp_path / "issues"
        asked = issue_file(d, selfimprove.KIND_CAPABILITY, "Aprender: clima", source="user")
        guessed = issue_file(d, selfimprove.KIND_CAPABILITY, "Aprender: chistes", source="model")
        assert I.eligibility(asked, self.settings(), {}, NOW).eligible
        assert not I.eligibility(guessed, self.settings(), {}, NOW).eligible

    def test_ranking_puts_the_owner_first_and_cosmetics_last(self, tmp_path):
        d = tmp_path / "issues"
        issue_file(d, selfimprove.KIND_TOOL_FAILED, "Falla la herramienta X", count=9)
        issue_file(d, selfimprove.KIND_USER_REPORTED, "Cambiar el color del icono", source="user")
        issue_file(d, selfimprove.KIND_UTTERANCE_DROPPED, "Me ignoró una frase", count=3)
        issue_file(d, selfimprove.KIND_USER_REPORTED, "Responde lento", source="user")
        ranked = I.rank(I.load_issues(d), self.settings(), {}, NOW)
        titles = [i.title for i in ranked]
        assert titles[0] == "Responde lento"
        assert titles.index("Me ignoró una frase") < titles.index("Falla la herramienta X")
        assert titles[-1] == "Cambiar el color del icono"

    def test_match_by_words(self, tmp_path):
        user_issue(tmp_path, "Latencia alta en la interacción")
        user_issue(tmp_path, "Permitir elegir monitor")
        found = I.match(I.load_issues(tmp_path / "issues"), "arreglate lo de la latencia")
        assert found.title == "Latencia alta en la interacción"

    def test_quota_and_one_at_a_time(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        assert service.can_work() == (True, "")
        for n in range(2):
            service.store.add_attempt({"id": str(n), "started": "2026-10-17 01:00:00"})
        assert service.can_work() == (False, "cupo")
        assert service.can_work(manual=True) == (False, "cupo")
        service.store["attempts"] = []
        service.running = {"id": "x"}
        assert service.can_work() == (False, "ocupada")

    def test_idle_power_meeting_and_night(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        service.keyboard_idle = lambda: 5.0
        assert service.can_work() == (False, "no_idle")
        assert service.can_work(manual=True) == (True, "")
        service.keyboard_idle = lambda: 10_000.0
        service.power = lambda: False
        assert service.can_work() == (False, "bateria")
        service.power = lambda: None      # a desktop Mac: unknown is fine
        service.meeting = lambda: True
        assert service.can_work() == (False, "reunion")
        service.meeting = lambda: False
        # In the night window a short idle is enough.
        service.now = lambda: NOW.replace(hour=3)
        service.keyboard_idle = lambda: 120.0
        assert service.can_work() == (True, "")

    def test_external_work_gate(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        service.add_work_gate(lambda: (False, "iniciativa"))
        assert service.can_work() == (False, "iniciativa")

    def test_disabled(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        service.set_enabled(False)
        assert service.can_work() == (False, "apagado")
        assert Store(tmp_path / "selffix_local.json")["enabled"] is False


# --------------------------------------------------------------------------- #
# Brief and scrubbing
# --------------------------------------------------------------------------- #


class TestBrief:
    def test_scrub(self):
        text = ("token sk-ant-api03-abcdefghijklmnopqrstuv en /Users/pepe/x.py, "
                "mail pepe@gmail.com, ghp_" + "a" * 36 + " server 10.0.0.42")
        out = scrub(text, home_dir="/Users/pepe")
        assert "sk-ant" not in out and "pepe" not in out and "ghp_" not in out
        assert "192.168" not in out and "<email>" in out

    def test_log_excerpt_matches_and_scrubs(self, tmp_path):
        issue = issue_file(tmp_path / "issues", selfimprove.KIND_TURN_FAILED, "Turno fallado BrainError", 3)
        log = tmp_path / "wawi.log"
        log.write_text("\n".join([
            "2026 INFO x: nada que ver",
            f"2026 INFO wawi.selfimprove: Issue recurred (3x): [turn-failed] T fp={issue.fingerprint}",
            "2026 ERROR wawi.brain: turno fallado brainerror token=sk-ant-api03-abcdefghijklmnopqrstu",
        ]))
        excerpt = B.log_excerpt(log, issue, 10)
        assert f"fp={issue.fingerprint}" in excerpt
        assert "brainerror" in excerpt.lower() and "sk-ant" not in excerpt
        assert "nada que ver" not in excerpt

    def test_bug_brief_demands_a_failing_test_first(self, tmp_path):
        issue = user_issue(tmp_path, detail="Tarda mucho; mi mail es pepe@gmail.com")
        text = B.build_brief(issue, excerpt="linea", previous_failures=["los tests fallan"])
        assert "FALLE" in text and "No agregues dependencias" in text
        assert "pepe@gmail.com" not in text
        assert "los tests fallan" in text
        for path in B.FORBIDDEN_PATHS:
            assert path in text
        assert "OWNER_IMPACT" in text and "COMMIT:" in text

    def test_capability_brief_asks_for_a_plugin(self, tmp_path):
        issue = issue_file(tmp_path / "issues", selfimprove.KIND_CAPABILITY, "Aprender: clima",
                           source="user")
        text = B.build_brief(issue)
        assert "agregado" in text and "pura-Python" in text

    def test_parse_verdict_is_strict(self):
        assert B.parse_verdict("ok\nVERDICT: APPROVE")[0]
        assert not B.parse_verdict("VERDICT: APPROVE\nmmm VERDICT: REJECT")[0]
        assert not B.parse_verdict("me parece bien")[0]

    def test_parse_report(self):
        subject, summary = B.parse_report("COMMIT: Arreglar X\nHice Y.\nOWNER_IMPACT: none")
        assert subject == "Arreglar X" and "Hice Y." in summary and "OWNER" not in summary


# --------------------------------------------------------------------------- #
# Gates (units)
# --------------------------------------------------------------------------- #


def stats(*rows):
    return [gitops.FileStat(p, a, r) for p, a, r in rows]


class TestGates:
    def test_forbidden_paths(self):
        for path in ("config.yaml", "orca_local.json", "wawi/selffix/gates.py",
                     "scripts/wawictl", "LICENSE"):
            assert not G.gate_forbidden(stats((path, 1, 0)), "").ok, path
        assert G.gate_forbidden(stats(("wawi/thing.py", 1, 0)), "").ok

    def test_gitignore_may_grow_but_not_shrink(self):
        add = "diff --git a/.gitignore b/.gitignore\n+++ b/.gitignore\n+new.json\n"
        drop = "diff --git a/.gitignore b/.gitignore\n+++ b/.gitignore\n-selffix_local.json\n"
        assert G.gate_forbidden(stats((".gitignore", 1, 0)), add).ok
        assert not G.gate_forbidden(stats((".gitignore", 0, 1)), drop).ok

    def test_size(self):
        assert not G.gate_size(stats(("a.py", 300, 101)), 400, 12).ok
        assert not G.gate_size(stats(*[(f"f{i}.py", 1, 0) for i in range(13)]), 400, 12).ok
        assert not G.gate_size([], 400, 12).ok
        assert G.gate_size(stats(("a.py", 10, 2)), 400, 12).ok

    def test_dependencies(self):
        diff = "diff --git a/requirements.txt b/requirements.txt\n+++ b/requirements.txt\n+tinydb==4.8.0\n"
        st = stats(("requirements.txt", 1, 0))
        assert not G.gate_dependencies(st, diff, capability=False).ok
        assert G.gate_dependencies(st, diff, capability=True, pure_python=lambda n, v: True).ok
        assert not G.gate_dependencies(st, diff, capability=True, pure_python=lambda n, v: False).ok
        loose = diff.replace("==4.8.0", ">=4")
        assert not G.gate_dependencies(st, loose, capability=True, pure_python=lambda n, v: True).ok

    def test_secrets(self):
        diff = "+++ b/wawi/x.py\n+KEY = 'sk-ant-api03-abcdefghijklmnopqrstuvwx'\n"
        assert not G.gate_secrets(diff).ok
        assert scan_diff("+++ b/x.py\n+p = '/Users/someone/file'\n", home_dir="/Users/me")
        assert scan_diff("+++ b/x.py\n+ip = '10.0.0.42'\n")
        assert not scan_diff("+++ b/x.py\n+Co-Authored-By: Claude <noreply@anthropic.com>\n",
                             home_dir="/Users/me", user="me")

    def test_tests_and_review(self):
        assert not G.gate_tests(1, False, "1 failed").ok
        assert not G.gate_tests(0, True, "").ok
        assert G.gate_tests(0, False, "5 passed").ok
        assert not G.gate_review(False, "rompe X", "claude").ok


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #


class TestClassifier:
    def test_internal_change_applies_alone(self):
        diff = "+++ b/wawi/thing.py\n+    return 42\n"
        out = C.classify_change(["wawi/thing.py", "tests/test_thing.py"], diff,
                                agent_report="OWNER_IMPACT: none")
        assert out.verdict == C.AUTO and out.reasons == []

    @pytest.mark.parametrize("path", ["wawi/plugins.py", "wawi/connectors.py", "wawi/brain.py",
                                      "wawi/petvoice.py", "wawi/mind/x.py", "wawi/initiative/y.py",
                                      "requirements.txt", "Makefile"])
    def test_paths_that_involve_him(self, path):
        out = C.classify_change([path], "", agent_report="OWNER_IMPACT: none")
        assert out.verdict == C.ASK

    @pytest.mark.parametrize("line", [
        "urllib.request.urlopen(url)", "requests.post(x)", "Reaction(bubble='x', speak=True)",
        "subprocess.run(['security', 'find-generic-password'])", "AVCaptureDevice.requestAccess",
        "open('~/Documents/x')",
    ])
    def test_added_lines_that_involve_him(self, line):
        out = C.classify_change(["wawi/thing.py"], f"+++ b/wawi/thing.py\n+{line}\n",
                                agent_report="OWNER_IMPACT: none")
        assert out.verdict == C.ASK

    def test_doubt_means_ask(self):
        assert C.classify_change(["wawi/thing.py"], "", agent_report="").verdict == C.ASK
        assert C.classify_change(["wawi/thing.py"], "",
                                 agent_report="OWNER_IMPACT: interaction").verdict == C.ASK
        assert C.classify_change(["wawi/thing.py"], "", agent_report="OWNER_IMPACT: none",
                                 new_deps=True).verdict == C.ASK
        assert C.classify_change(["wawi/thing.py"], "", agent_report="OWNER_IMPACT: none",
                                 autonomy="ask").verdict == C.ASK

    def test_blocked_on_the_owner(self, tmp_path):
        voice = user_issue(tmp_path, "Verificar hablante: solo tu voz abre un turno")
        check = C.precheck(voice)
        assert check.verdict == C.BLOCKED and check.need == "voice"
        assert "grabes 50 veces «Wawi»" in check.request
        token = user_issue(tmp_path, "Conectar Spotify", detail="Necesita un token de la API")
        assert C.precheck(token).need == "token"
        assert C.precheck(user_issue(tmp_path, "Respuestas lentas")).verdict == "work"

    def test_blocked_issue_becomes_a_request_not_an_attempt(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        user_issue(tmp_path, "Verificar hablante: solo tu voz abre un turno")
        lento = user_issue(tmp_path, "Responde lento")
        assert [i.id for i in service.candidates()] == [lento.id]
        requests = service.owner_requests()
        assert len(requests) == 1 and requests[0]["need"] == "voice"
        service.candidates()  # not filed twice
        assert len(service.owner_requests()) == 1


# --------------------------------------------------------------------------- #
# The pipeline: gates in context, apply, ask
# --------------------------------------------------------------------------- #


class TestPipeline:
    def test_she_shows_her_dots_while_she_fixes_herself(self, tmp_path, repo, monkeypatch):
        from wawi import working

        service = make_service(tmp_path, repo)
        seen = []

        def pipeline(*args, **kwargs):
            seen.append(working.musing())
            raise RuntimeError("boom")

        monkeypatch.setattr(service, "_pipeline", pipeline)
        service.run_attempt(user_issue(tmp_path))
        assert seen == [True]
        assert working.musing() is False  # off again, even after a crash

    def test_happy_path_merges_with_trailer_and_hands_over_to_the_watchdog(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        issue = user_issue(tmp_path)
        before = git(repo, "rev-parse", "HEAD").strip()
        attempt = service.run_attempt(issue)
        assert attempt["outcome"] == "watching", attempt["reasons"]
        message = git(repo, "log", "-1", "--format=%B")
        assert f"Wawi-Selffix: {issue.id}" in message and "Autoarreglo:" in message
        assert (repo / "wawi" / "thing.py").read_text().endswith("return 42\n")
        assert service.store["last_good"] == before
        assert not Path(attempt["worktree"]).exists()
        assert "selffix/" not in git(repo, "branch")
        # The frozen watchdog is spawned with its job; it does the restart.
        argv = service.spawned[-1]
        job = json.loads(Path(argv[-1]).read_text())
        assert Path(argv[1]).name == "watchdog.py" and job["commit"] == attempt["commit"]
        assert job["wawictl"] == "/x/wawictl" and job["fingerprint"] == issue.id
        assert any("Me estoy arreglando" in b for b in service.bubbles)
        assert service.store["announce"]
        names = [j.name for j in service.agents.jobs]
        assert names == ["author", "tests", "review"]
        # The reviewer is the other engine.
        review = service.agents.jobs[-1]
        assert review.argv[0] == "claude"

    def _rejected(self, tmp_path, repo, agents, **kw):
        service = make_service(tmp_path, repo, agents, **kw)
        head = git(repo, "rev-parse", "HEAD")
        attempt = service.run_attempt(user_issue(tmp_path))
        assert attempt["outcome"] == "rejected"
        assert git(repo, "rev-parse", "HEAD") == head, "main must be untouched"
        assert not Path(attempt["worktree"]).exists()
        hist = service.store.issue(attempt["issue"])
        assert hist["last_failed"] and hist["reasons"]
        return attempt

    def test_tests_fail(self, tmp_path, repo):
        attempt = self._rejected(tmp_path, repo, FakeAgents(tests_code=1))
        assert "tests fallan" in attempt["reasons"][0]

    def test_diff_too_big(self, tmp_path, repo):
        big = lambda wt: (wt / "wawi" / "big.py").write_text("x = 1\n" * 500)  # noqa: E731
        attempt = self._rejected(tmp_path, repo, FakeAgents(edit=big))
        assert "demasiado grande" in attempt["reasons"][0]

    def test_forbidden_path(self, tmp_path, repo):
        def edit(wt):
            (wt / "wawi" / "selffix").mkdir(parents=True)
            (wt / "wawi" / "selffix" / "gates.py").write_text("ALL_OK = True\n")
        attempt = self._rejected(tmp_path, repo, FakeAgents(edit=edit))
        assert "prohibidos" in attempt["reasons"][0]

    def test_new_dependency_in_a_bug_fix(self, tmp_path, repo):
        def edit(wt):
            (wt / "requirements.txt").write_text("numpy==2.0.0\nleftpad==1.0.0\n")
        attempt = self._rejected(tmp_path, repo, FakeAgents(edit=edit))
        assert "dependencias" in attempt["reasons"][0]

    def test_reviewer_rejects(self, tmp_path, repo):
        attempt = self._rejected(tmp_path, repo, FakeAgents(verdict="- rompe X\nVERDICT: REJECT"))
        assert "revisor" in attempt["reasons"][0]

    def test_secret_found(self, tmp_path, repo):
        def edit(wt):
            (wt / "wawi" / "thing.py").write_text("KEY = 'sk-ant-api03-abcdefghijklmnopqrstuvwx'\n")
        attempt = self._rejected(tmp_path, repo, FakeAgents(edit=edit))
        assert "secreto" in attempt["reasons"][0]

    def test_agent_changes_nothing(self, tmp_path, repo):
        attempt = self._rejected(tmp_path, repo, FakeAgents(edit=lambda wt: None))
        assert "no cambió nada" in attempt["reasons"][0]

    def test_dirty_tree_waits_and_applies_later(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        (repo / "wawi" / "thing.py").write_text("# trabajo del humano\n")
        head = git(repo, "rev-parse", "HEAD")
        attempt = service.run_attempt(user_issue(tmp_path))
        assert attempt["outcome"] == "waiting_clean"
        assert git(repo, "rev-parse", "HEAD") == head
        assert (repo / "wawi" / "thing.py").read_text() == "# trabajo del humano\n"
        assert Path(attempt["worktree"]).exists()
        assert service.tick() == "waiting_clean"
        git(repo, "checkout", "--", "wawi/thing.py")      # the human finished
        assert service.tick() == "applied"
        assert "Wawi-Selffix:" in git(repo, "log", "-1", "--format=%B")
        assert service.store["pending"] is None

    def test_not_on_main_waits(self, tmp_path, repo):
        git(repo, "checkout", "-q", "-b", "feature")
        service = make_service(tmp_path, repo)
        attempt = service.run_attempt(user_issue(tmp_path))
        assert attempt["outcome"] == "waiting_clean" and attempt["wait_reason"] == "not_on_main"

    def test_main_moved_rebases_and_retests(self, tmp_path, repo):
        agents = FakeAgents()
        service = make_service(tmp_path, repo, agents)
        (repo / "wawi" / "thing.py").write_text("# sucio\n")
        attempt = service.run_attempt(user_issue(tmp_path))
        git(repo, "checkout", "--", "wawi/thing.py")
        (repo / "README.md").write_text("hola\n")
        git(repo, "add", "README.md")
        git(repo, "commit", "-q", "-m", "Commit humano")
        assert service.tick() == "applied"
        assert [j.name for j in agents.jobs].count("tests") == 2
        log = git(repo, "log", "--format=%s")
        assert "Commit humano" in log and "Autoarreglo" in log.splitlines()[0]
        assert attempt["outcome"] == "watching"

    def test_ask_offers_on_the_board_and_waits(self, tmp_path, repo):
        class Board:
            pages = []

            def show(self, title, content, fmt):
                self.pages.append((title, content))

        def edit(wt):
            (wt / "wawi" / "plugins.py").write_text("ENABLED = True\n")
        service = make_service(tmp_path, repo, FakeAgents(edit=edit), board=Board())
        head = git(repo, "rev-parse", "HEAD")
        attempt = service.run_attempt(user_issue(tmp_path))
        assert attempt["outcome"] == "pending_owner"
        assert git(repo, "rev-parse", "HEAD") == head
        title, content = service.board.pages[-1]
        assert "```diff" in content and "plugins.py" in content and "aplicá el arreglo" in content
        assert any("te toca a vos" in b for b in service.bubbles)
        assert service.can_work() == (False, "pendiente")
        ok, speak = service.decide_pending(True)
        assert ok and "aplico" in speak
        assert "Wawi-Selffix:" in git(repo, "log", "-1", "--format=%B")

    def test_discard_marks_wontfix(self, tmp_path, repo):
        def edit(wt):
            (wt / "wawi" / "plugins.py").write_text("ENABLED = True\n")
        service = make_service(tmp_path, repo, FakeAgents(edit=edit))
        attempt = service.run_attempt(user_issue(tmp_path))
        ok, _ = service.decide_pending(False)
        assert ok and attempt["outcome"] == "discarded"
        assert service.store.issue(attempt["issue"])["wontfix"]
        assert not Path(attempt["worktree"]).exists()
        assert service.candidates() == []

    def test_no_restart_path_means_no_apply(self, tmp_path, repo):
        service = make_service(tmp_path, repo, restart=None)
        attempt = service.run_attempt(user_issue(tmp_path))
        assert attempt["outcome"] == "waiting_clean" and attempt["wait_reason"] == "no_restart"

    def test_fallback_engine_when_codex_is_missing(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        service.which = lambda e: "/usr/bin/claude" if e == "claude" else None
        attempt = service.run_attempt(user_issue(tmp_path))
        assert attempt["engine"] == "claude"
        assert service.agents.jobs[0].argv[0] == "claude"


# --------------------------------------------------------------------------- #
# Watchdog
# --------------------------------------------------------------------------- #


def selffix_commit(repo: Path, issue: str = "abc123") -> str:
    branch = "selffix/t"
    git(repo, "checkout", "-q", "-b", branch)
    (repo / "wawi" / "thing.py").write_text("def answer():\n    return 42\n")
    git(repo, "commit", "-qam", "wip")
    git(repo, "checkout", "-q", "main")
    return gitops.squash_apply(repo, branch, gitops.commit_message("Arreglo", issue))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def watch(tmp_path, repo, sha, on_restart, alive=lambda: True, **job):
    log = tmp_path / "wawi.log"
    log.write_text("2026-10-17 10:00:00,000 INFO old: viejo\n")
    clock = FakeClock()
    restarts = []

    def restart():
        restarts.append(clock.t)
        on_restart(log, len(restarts))

    w = W.Watch({"repo": str(repo), "commit": sha, "log_path": str(log), "startup_s": 60,
                 "watch_s": 1800, "poll_s": 10, "result_path": str(tmp_path / "result.json"),
                 **job}, restart=restart, alive=alive, clock=clock, sleep=clock.sleep)
    result = w.run()
    return result, restarts


def append(log: Path, *lines: str) -> None:
    with log.open("a") as fh:
        fh.write("".join(line + "\n" for line in lines))


READY = "2026-10-17 10:00:05,000 INFO wawi.orchestrator: Wawi ready. Say wawi"


class TestWatchdog:
    def test_healthy(self, tmp_path, repo):
        sha = selffix_commit(repo)
        result, restarts = watch(tmp_path, repo, sha, lambda log, n: append(log, READY))
        assert result["outcome"] == "healthy" and len(restarts) == 1
        assert git(repo, "rev-parse", "HEAD").strip() == sha
        assert json.loads((tmp_path / "result.json").read_text())["outcome"] == "healthy"

    def test_startup_crash_reverts_and_restarts(self, tmp_path, repo):
        sha = selffix_commit(repo)

        def crash(log, n):
            if n == 1:
                append(log, "Traceback (most recent call last):", "  File x", "NameError: boom")
        result, restarts = watch(tmp_path, repo, sha, crash)
        assert result["outcome"] == "rolled_back" and len(restarts) == 2
        assert (repo / "wawi" / "thing.py").read_text().endswith("return 41\n")
        body = git(repo, "log", "-1", "--format=%B")
        assert f"Wawi-Selffix-Revert: {sha}" in body and body.startswith("Revertir")
        assert sha in git(repo, "log", "--format=%H")    # reverted, never reset

    def test_error_lines_before_ready_revert(self, tmp_path, repo):
        sha = selffix_commit(repo)
        result, _ = watch(tmp_path, repo, sha, lambda log, n: append(
            log, "2026-10-17 10:00:01,000 ERROR wawi.panel: broke", READY) if n == 1 else None)
        assert result["outcome"] == "rolled_back"

    def test_recurrence_reverts(self, tmp_path, repo):
        sha = selffix_commit(repo, "deadbeef01")

        def again(log, n):
            if n == 1:
                append(log, READY,
                       "2026 INFO wawi.selfimprove: Issue recurred (4x): [slow-turn] T fp=deadbeef01",
                       "2026 INFO wawi.selfimprove: Issue recurred (5x): [slow-turn] T fp=deadbeef01")
        result, _ = watch(tmp_path, repo, sha, again, fingerprint="deadbeef01",
                          baseline_per_hour=0.5)
        assert result["outcome"] == "rolled_back" and "volvió" in result["reason"]

    def test_crash_loop_reverts_but_a_clean_restart_does_not(self, tmp_path, repo):
        assert W.unclean_restarts(f"{READY}\n...\n{READY}\n") == 1
        assert W.unclean_restarts(f"{READY}\nINFO Wawi stopped.\n{READY}\n") == 0
        sha = selffix_commit(repo)
        alive = iter([True] + [False] * 400)
        result, _ = watch(tmp_path, repo, sha, lambda log, n: append(log, READY),
                          alive=lambda: next(alive))
        assert result["outcome"] == "rolled_back" and "se cayó" in result["reason"]

    def test_owner_quitting_is_not_a_crash(self, tmp_path, repo):
        sha = selffix_commit(repo)
        state = {"alive": True}

        def ready(log, n):
            append(log, READY, "2026 INFO wawi.orchestrator: Wawi stopped.")
            state["alive"] = False
        alive = iter([True])
        result, _ = watch(tmp_path, repo, sha, ready, alive=lambda: next(alive, False))
        assert result["outcome"] == "inconclusive"
        assert git(repo, "rev-parse", "HEAD").strip() == sha

    def test_never_reverts_a_human_commit(self, tmp_path, repo):
        human = git(repo, "rev-parse", "HEAD").strip()
        ok, why = W.revert_selffix(str(repo), human, "x")
        assert not ok and why == "not_a_selffix_commit"
        result, _ = watch(tmp_path, repo, human, lambda log, n: None)
        assert result["outcome"] == "revert_blocked"
        assert git(repo, "rev-parse", "HEAD").strip() == human

    def test_never_touches_uncommitted_work(self, tmp_path, repo):
        sha = selffix_commit(repo)
        (repo / "wawi" / "thing.py").write_text("# humano\n")
        ok, why = W.revert_selffix(str(repo), sha, "x")
        assert not ok and why == "dirty_tree"
        assert (repo / "wawi" / "thing.py").read_text() == "# humano\n"

    def test_not_twice(self, tmp_path, repo):
        sha = selffix_commit(repo)
        assert W.revert_selffix(str(repo), sha, "x")[0]
        assert W.revert_selffix(str(repo), sha, "x") == (False, "already_reverted")

    def test_runs_standalone_with_the_stdlib(self, tmp_path):
        # The frozen copy must not import wawi.
        source = Path(W.__file__).read_text()
        assert "from wawi" not in source and "import wawi" not in source
        out = subprocess.run([sys.executable, "-I", W.__file__], capture_output=True, text=True)
        assert out.returncode == 2 and "usage" in out.stderr


# --------------------------------------------------------------------------- #
# Learning, backoff and reporting
# --------------------------------------------------------------------------- #


class TestLearning:
    def test_failure_feeds_the_next_brief_and_backs_off(self, tmp_path, repo):
        service = make_service(tmp_path, repo, FakeAgents(tests_code=1))
        issue = user_issue(tmp_path)
        service.run_attempt(issue)
        assert service.candidates() == []           # backoff
        later = NOW + timedelta(days=8)
        service.now = lambda: later
        assert [i.id for i in service.candidates()] == [issue.id]
        service.agents.tests_code = 0
        service.run_attempt(issue)
        brief = service.agents.jobs[-3].stdin_text
        assert "Intentos anteriores fallaron" in brief and "tests fallan" in brief

    def test_watch_verdicts_fold_into_history(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        attempt = service.run_attempt(user_issue(tmp_path))
        Path(attempt["watch_result"]).write_text(json.dumps(
            {"outcome": "rolled_back", "reason": "errores al arrancar", "evidence": ["ERROR x"],
             "revert": "abc"}))
        service.fold_watch_results()
        assert attempt["outcome"] == "rolled_back"
        assert service.store.issue(attempt["issue"])["last_failed"]
        assert any("Deshice" in b for b in service.bubbles)

    def test_healthy_verdict_marks_applied_and_not_retried(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        attempt = service.run_attempt(user_issue(tmp_path))
        Path(attempt["watch_result"]).write_text(json.dumps({"outcome": "healthy"}))
        service.fold_watch_results()
        assert attempt["outcome"] == "applied"
        service.store["attempts"] = []   # quota out of the way
        assert service.candidates() == []

    def test_no_new_work_while_a_fix_is_watched(self, tmp_path, repo):
        service = make_service(tmp_path, repo, settings={"max_attempts_per_day": 5})
        attempt = service.run_attempt(user_issue(tmp_path))
        assert service.can_work() == (False, "vigilando")
        # A watchdog that never reports stops blocking after the watch + an hour.
        service.now = lambda: NOW + timedelta(hours=3)
        service.fold_watch_results()
        assert attempt["outcome"] == "applied" and service.can_work() == (True, "")

    def test_today_markdown(self, tmp_path, repo):
        service = make_service(tmp_path, repo, FakeAgents(tests_code=1))
        service.run_attempt(user_issue(tmp_path))
        user_issue(tmp_path, "Verificar hablante: solo tu voz")
        service.candidates()
        text = service.today_markdown()
        assert text.startswith("## Lo que me arreglé hoy")
        assert "no pasó mis controles" in text and "grabes 50 veces" in text
        assert "Respuestas demasiado lentas" in service.history_markdown()

    def test_start_announces_after_a_self_restart(self, tmp_path, repo):
        service = make_service(tmp_path, repo, settings={"poll_s": 3600})
        service.run_attempt(user_issue(tmp_path))
        again = make_service(tmp_path, repo, settings={"poll_s": 3600})
        again.start()
        again.stop()
        assert any("Me arreglé lo de" in b for b in again.bubbles)
        assert again.store["announce"] == []


# --------------------------------------------------------------------------- #
# Undo, menu and voice
# --------------------------------------------------------------------------- #


class TestControls:
    def test_undo_last_reverts_only_selffix_and_restarts(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        attempt = service.run_attempt(user_issue(tmp_path))
        (repo / "README.md").write_text("humano\n")
        git(repo, "add", "README.md")
        git(repo, "commit", "-q", "-m", "Commit humano")
        ok, speak = service.undo_last()
        assert ok and "deshice" in speak.lower()
        assert (repo / "README.md").exists()                # the human commit stays
        assert (repo / "wawi" / "thing.py").read_text().endswith("return 41\n")
        assert service.spawned[-1] == ["/x/wawictl", "restart"]
        assert attempt["outcome"] == "undone"
        ok, speak = service.undo_last()
        assert not ok and "ningún arreglo" in speak

    def test_undo_with_dirty_tree_touches_nothing(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        service.run_attempt(user_issue(tmp_path))
        (repo / "wawi" / "thing.py").write_text("# humano\n")
        ok, speak = service.undo_last()
        assert not ok and "sin commitear" in speak
        assert (repo / "wawi" / "thing.py").read_text() == "# humano\n"

    def test_menu(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        ran = []
        menu = SelffixMenu(service, notify=service.bubbles.append, run=lambda fn: (ran.append(fn), fn()))
        titles = [t for _, t, _, visible in menu.entries() if visible]
        assert titles == ["Autoarreglo activado", "Arreglarme ahora",
                          "Deshacer el último arreglo", "Ver historial"]
        menu.click("toggle")
        assert not service.enabled and service.bubbles[-1] == "Autoarreglo apagado."
        menu.click("fix")
        assert service.bubbles[-1] == "Tengo el autoarreglo apagado."
        menu.click("toggle")
        menu.click("undo")
        assert "ningún arreglo" in service.bubbles[-1]
        service.store["pending"] = {"attempt": "x", "title": "t", "approved": False}
        visible = [a for a, _, _, v in menu.entries() if v]
        assert "apply" in visible and "discard" in visible

    def test_fix_now_by_name(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        user_issue(tmp_path, "Latencia alta")
        issue_file(tmp_path / "issues", selfimprove.KIND_SLOW_TURN, "Turno lento extremo", count=1)
        started = []
        service.start_attempt = lambda issue, wait=False: started.append(issue.title)
        ok, speak = service.fix_now("arreglate lo del turno lento extremo")
        assert ok and started == ["Turno lento extremo"]     # named = no count threshold
        ok, speak = service.fix_now("lo de los pingüinos voladores")
        assert not ok and "No encuentro" in speak
        user_issue(tmp_path, "Verificar hablante: solo tu voz")
        ok, speak = service.fix_now("verificar hablante")
        assert not ok and "grabes" in speak

    def test_voice_tools(self, tmp_path, repo):
        service = make_service(tmp_path, repo)
        assert T.self_fix(None, "report")[0] is False
        ok, speak, _ = T.self_fix(service, "report")
        assert ok and "no me arreglé nada" in speak
        ok, speak, _ = T.self_fix(service, "undo")
        assert not ok
        ok, speak, data = T.self_fix(service, "requests")
        assert ok and data["requests"] == []
        assert T.self_fix(service, "bailar")[0] is False

    def test_learn_capability_files_an_issue(self, tmp_path):
        journal = selfimprove.IssueJournal(directory=tmp_path / "issues")
        ok, speak, _ = T.learn_capability(journal, "leerme el clima", from_user=True)
        assert ok and "aprendo" in speak
        ok, speak, _ = T.learn_capability(journal, "contar chistes")
        assert speak == "No sé hacer eso todavía, me lo armo."
        rows = I.load_issues(tmp_path / "issues")
        kinds = {r.title: (r.kind, r.from_user) for r in rows}
        assert kinds["Aprender: leerme el clima"] == ("capability", True)
        assert kinds["Aprender: contar chistes"] == ("capability", False)

    def test_tools_registry(self, tmp_path, repo, monkeypatch):
        from wawi import selffix, tools

        journal = selfimprove.IssueJournal(directory=tmp_path / "issues")
        monkeypatch.setattr(tools, "_JOURNAL", journal)
        monkeypatch.setattr(selffix, "_SERVICE", None)
        names = {t["name"] for t in tools.active_tools()}
        assert "learn_capability" in names and "self_fix" not in names
        monkeypatch.setattr(selffix, "_SERVICE", make_service(tmp_path, repo))
        names = {t["name"] for t in tools.active_tools()}
        assert "self_fix" in names
        assert tools.dispatch("self_fix", {"action": "report"}).ok
        assert all("animation" not in t for t in tools.active_tools())

    def test_initiative_hooks(self, tmp_path, repo, monkeypatch):
        from wawi import selffix

        monkeypatch.setattr(selffix, "_SERVICE", None)
        assert selffix.can_work() == (False, "sin_servicio")
        assert selffix.today_markdown() == "" and selffix.owner_requests() == []
        service = make_service(tmp_path, repo)
        monkeypatch.setattr(selffix, "_SERVICE", service)
        assert selffix.add_work_gate(lambda: False)
        assert selffix.can_work() == (False, "gate")


# --------------------------------------------------------------------------- #
# Executors
# --------------------------------------------------------------------------- #


class TestExecutors:
    def test_runjob_timeout_kills_and_reports(self, tmp_path):
        job = Job(name="slow", argv=[sys.executable, "-c", "import time; time.sleep(30)"],
                  cwd=tmp_path, timeout_s=0.5, dir=tmp_path)
        result = LocalExecutor().run(job)
        assert result.timed_out and not result.ok

    def test_runjob_ok_with_stdin(self, tmp_path):
        job = Job(name="echo", argv=[sys.executable, "-c", "import sys; print(sys.stdin.read())"],
                  cwd=tmp_path, timeout_s=20, dir=tmp_path, stdin_text="hola")
        result = LocalExecutor().run(job)
        assert result.ok and "hola" in result.output

    def test_orca_shows_the_author_in_a_terminal(self, tmp_path):
        calls = []

        class CLI:
            def run(self, *args, timeout=60.0):
                calls.append(args)
                if args[0] == "status":
                    return {"runtime": {"reachable": True}}
                if args[:2] == ("terminal", "create"):
                    spec = json.loads(Path(args[-1].split()[-1]).read_text())
                    Path(spec["result_file"]).write_text('{"code": 0}')
                    return {"terminal": {"handle": "t1"}}
                return {}

        job = Job(name="author", argv=["true"], cwd=tmp_path, timeout_s=60, dir=tmp_path,
                  visible=True, title="Wawi · Autoarreglo · x")
        result = OrcaExecutor(CLI(), tmp_path, sleep=lambda s: None).run(job)
        assert result.ok and result.where == "orca"
        assert ("terminal", "close", "--terminal", "t1") in calls

    def test_orca_down_runs_locally(self, tmp_path):
        class Down:
            def run(self, *args, timeout=60.0):
                raise RuntimeError("down")

        class Local:
            def run(self, job):
                return JobResult(0, where="local")

        job = Job(name="author", argv=["true"], cwd=tmp_path, timeout_s=60, dir=tmp_path, visible=True)
        assert OrcaExecutor(Down(), tmp_path, fallback=Local()).run(job).where == "local"


class FakeSsh:
    def __init__(self, cpus=16, load=1.0, mem=12000):
        self.calls: list[tuple[str, str]] = []
        self.res = f"{cpus}\n{load}\n{mem}\n"

    def __call__(self, command, stdin_text="", timeout=60.0):
        self.calls.append((command, stdin_text))
        if command.startswith("nproc"):
            return 0, self.res, ""
        if "nohup setsid" in command:
            return 0, "4242\n", ""
        if command.startswith("cat ") and "exit_code" in command:
            return 0, "0\n", ""
        return 0, "", ""


class TestRemote:
    def settings(self):
        return RemoteSettings(enabled=True, host="buildbox", base_dir="~/work")

    def test_off_by_default(self):
        assert not SelffixSettings.from_dict({}).remote.enabled
        with pytest.raises(RemoteError):
            RemoteExecutor(RemoteSettings())

    def test_refuses_sudo_apt_docker_and_production(self):
        settings = RemoteSettings(enabled=True, host="buildbox", base_dir="~/work",
                                  protected=("prod-",))
        ex = RemoteExecutor(settings, ssh=FakeSsh())
        for command in ("sudo make", "apt-get install x", "docker rm -f a", "echo prod-db"):
            with pytest.raises(RemoteError):
                ex.start("job", command)

    def test_checks_headroom(self):
        with pytest.raises(RemoteError, match="ocupado"):
            RemoteExecutor(self.settings(), ssh=FakeSsh(load=14.0)).start("train", "python t.py")
        with pytest.raises(RemoteError, match="memoria"):
            RemoteExecutor(self.settings(), ssh=FakeSsh(mem=1000)).start("train", "python t.py")

    def test_labelled_detached_job_with_secret_env_file(self):
        ssh = FakeSsh()
        ex = RemoteExecutor(self.settings(), ssh=ssh, project="wawi-wakeword")
        job = ex.start("train", "python train.py", env={"HF_TOKEN": "s3cr3t"})
        assert job.pid == 4242 and job.label.startswith("wawi-wawi-wakeword-train-")
        assert job.dir.startswith("~/work/wawi-wakeword/.wawi-jobs/")
        commands = " ".join(c for c, _ in ssh.calls)
        assert "s3cr3t" not in commands                   # only via stdin
        env_write = next(s for c, s in ssh.calls if c.endswith("chmod 600 " + job.dir + "/env"))
        assert "HF_TOKEN=s3cr3t" in env_write
        assert any("nohup setsid" in c for c, _ in ssh.calls)

    def test_kill_by_pid_after_checking_the_label_and_clean_only_own(self):
        ssh = FakeSsh()
        ex = RemoteExecutor(self.settings(), ssh=ssh)
        job = ex.start("t", "true", heavy=False)
        ex.stop(job)
        kill = ssh.calls[-1][0]
        assert f"-p {job.pid}" in kill and job.label in kill and kill.endswith(f"kill {job.pid}")
        assert "pkill" not in kill and "killall" not in kill
        from wawi.selffix.remote import RemoteJob

        assert not ex.cleanup(RemoteJob("other", "~/work/x", 1))
        assert ex.cleanup(job)
        assert ssh.calls[-1][0] == f"rm -rf -- {job.dir}"

    def test_run_interface(self, tmp_path):
        ex = RemoteExecutor(self.settings(), ssh=FakeSsh(), sleep=lambda s: None)
        result = ex.run(Job(name="tests", argv=["pytest", "-q"], cwd=tmp_path, timeout_s=60, dir=tmp_path))
        assert result.ok and result.where == "remote"


# --------------------------------------------------------------------------- #
# The journal side and settings
# --------------------------------------------------------------------------- #


class TestJournal:
    def test_a_restart_keeps_counting(self, tmp_path):
        d = tmp_path / "issues"
        first = selfimprove.IssueJournal(directory=d)
        for _ in range(3):
            first.record(selfimprove.KIND_SLOW_TURN, "Turno lento", subject="slow")
        second = selfimprove.IssueJournal(directory=d)
        issue = second.record(selfimprove.KIND_SLOW_TURN, "Turno lento", subject="slow")
        assert issue.count == 4
        assert I.load_issues(d)[0].count == 4

    def test_recurrences_are_logged_with_the_fingerprint(self, tmp_path, caplog):
        journal = selfimprove.IssueJournal(directory=tmp_path / "issues")
        with caplog.at_level("INFO", logger="wawi.selfimprove"):
            a = journal.record(selfimprove.KIND_SLOW_TURN, "Turno lento")
            journal.record(selfimprove.KIND_SLOW_TURN, "Turno lento")
        assert W.count_issue(caplog.text, a.fingerprint) == 2


class TestSettings:
    def test_defaults_and_clamps(self):
        s = SelffixSettings.from_dict({"max_changed_lines": 99999, "engine": "gpt",
                                       "autonomy": "yolo", "time_budget_s": "x"})
        assert s.max_changed_lines == 2000 and s.engine == "codex" and s.autonomy == "ask"
        assert s.time_budget_s == 1800 and s.max_attempts_per_day == 2
        assert s.reviewer("codex") == "claude" and s.reviewer("claude") == "codex"
        assert s.night_window is not None

    def test_config_section(self, tmp_path):
        from wawi import config

        (tmp_path / "config.yaml").write_text("selffix:\n  max_files: 5\n")
        cfg = config.load(tmp_path / "config.yaml")
        assert cfg.selffix["max_files"] == 5 and cfg.selffix["enabled"] is True
        assert cfg.selffix["local_file"] == str(tmp_path / "selffix_local.json")

    def test_report_lines(self):
        line = R.attempt_line({"outcome": "applied", "title": "X", "commit": "0123456789"})
        assert line == "Me arreglé «X». (commit `01234567`)"


def test_model_tokens_are_not_a_login():
    """Observed: an issue about image tokens was blocked as needing a token."""
    from wawi.selffix import classify

    class _Issue:
        def text(self):
            return "Reescalar a 1280px: ~1.370 tokens por imagen en vez de ~2.000 tokens."

    assert classify.precheck(_Issue()).verdict == "work"
    class _Login:
        def text(self):
            return "Para leer Gmail necesito tu token de acceso."

    assert classify.precheck(_Login()).need == "token"
