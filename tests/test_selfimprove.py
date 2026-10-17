"""Tests for issue-filing self-improvement.

The property that matters most is the BOUNDARY: this module records problems
and cannot change code. The rest is deduplication, which is what keeps a
recurring failure from turning the journal into noise.
"""

from __future__ import annotations

import pytest

from wawi.selfimprove import (
    KIND_SLOW_TURN,
    KIND_TURN_FAILED,
    KIND_USER_REPORTED,
    IssueJournal,
)


@pytest.fixture
def journal(tmp_path):
    return IssueJournal(directory=tmp_path / "issues", assistant_name="Wawi")


class TestBoundary:
    def test_the_journal_has_no_way_to_change_code(self, journal):
        """Structural, not a promise in a prompt: there is no such verb."""
        forbidden = ("write_file", "edit", "patch", "apply", "commit", "push", "run")
        surface = {n for n in dir(journal) if not n.startswith("_")}
        assert not surface & set(forbidden)

    def test_recording_only_ever_writes_inside_the_issues_directory(self, journal, tmp_path):
        journal.record(KIND_TURN_FAILED, "algo falló")
        written = list(tmp_path.rglob("*"))
        assert written, "expected an issue file"
        for path in written:
            assert str(path).startswith(str(tmp_path / "issues"))

    def test_the_filed_issue_states_that_no_code_changed(self, journal, tmp_path):
        journal.record(KIND_TURN_FAILED, "algo falló")
        body = next((tmp_path / "issues").glob("*.md")).read_text()
        assert "No code was changed" in body


class TestRecording:
    def test_an_issue_is_written_to_disk(self, journal, tmp_path):
        journal.record(KIND_TURN_FAILED, "Turno fallado: TimeoutError", detail="tardó")
        files = list((tmp_path / "issues").glob("*.md"))
        assert len(files) == 1
        body = files[0].read_text()
        assert "Turno fallado: TimeoutError" in body
        assert "tardó" in body

    def test_context_is_recorded_as_json(self, journal, tmp_path):
        journal.record(KIND_SLOW_TURN, "lento", context={"total_s": 22.5})
        body = next((tmp_path / "issues").glob("*.md")).read_text()
        assert "total_s" in body and "22.5" in body

    def test_disabled_journal_records_nothing(self, tmp_path):
        j = IssueJournal(directory=None)
        assert j.record(KIND_TURN_FAILED, "algo") is None
        assert not j.enabled
        assert list(tmp_path.iterdir()) == []

    def test_an_unwritable_directory_never_raises(self, tmp_path):
        blocked = tmp_path / "afile"
        blocked.write_text("not a directory")
        j = IssueJournal(directory=blocked / "issues")
        j.record(KIND_TURN_FAILED, "algo")  # must not raise
        assert len(j.issues()) == 1

    def test_an_unknown_kind_is_still_recorded(self, journal):
        assert journal.record("something-new", "raro") is not None


class TestDeduplication:
    def test_the_same_failure_bumps_a_counter_instead_of_piling_up(self, journal, tmp_path):
        for _ in range(4):
            journal.record(KIND_TURN_FAILED, "Turno fallado: TimeoutError",
                           subject="TimeoutError")
        assert len(list((tmp_path / "issues").glob("*.md"))) == 1
        assert journal.issues()[0].count == 4

    def test_different_subjects_stay_separate(self, journal, tmp_path):
        journal.record(KIND_TURN_FAILED, "a", subject="TimeoutError")
        journal.record(KIND_TURN_FAILED, "b", subject="ConnectionError")
        assert len(list((tmp_path / "issues").glob("*.md"))) == 2

    def test_the_subject_not_the_title_decides_sameness(self, journal):
        """Titles carry timings; grouping by them would defeat dedup entirely."""
        journal.record(KIND_SLOW_TURN, "Turno lento: 21s", subject="slow turn")
        journal.record(KIND_SLOW_TURN, "Turno lento: 34s", subject="slow turn")
        assert len(journal.issues()) == 1
        assert journal.issues()[0].count == 2

    def test_the_same_kind_and_subject_across_kinds_do_not_collide(self, journal):
        journal.record(KIND_TURN_FAILED, "x", subject="same")
        journal.record(KIND_SLOW_TURN, "x", subject="same")
        assert len(journal.issues()) == 2

    def test_context_from_a_repeat_updates_the_issue(self, journal):
        journal.record(KIND_SLOW_TURN, "lento", subject="s", context={"total_s": 21})
        journal.record(KIND_SLOW_TURN, "lento", subject="s", context={"total_s": 40})
        assert journal.issues()[0].context["total_s"] == 40


class TestReporting:
    def test_issues_are_ranked_by_how_often_they_happen(self, journal):
        journal.record(KIND_SLOW_TURN, "raro", subject="a")
        for _ in range(3):
            journal.record(KIND_TURN_FAILED, "frecuente", subject="b")
        assert journal.issues()[0].title == "frecuente"

    def test_prompt_hint_is_empty_with_no_issues(self, journal):
        assert journal.prompt_hint() == ""

    def test_prompt_hint_is_bounded(self, journal):
        for i in range(10):
            journal.record(KIND_TURN_FAILED, f"problema {i}", subject=f"s{i}")
        assert len(journal.prompt_hint(limit=3).splitlines()) == 4  # header + 3

    def test_prompt_hint_is_stable_across_repeats(self, journal):
        """A repeat must not change the prompt: that drops the cached session."""
        journal.record(KIND_USER_REPORTED, "mala respuesta", subject="x")
        first = journal.prompt_hint()
        journal.record(KIND_USER_REPORTED, "mala respuesta", subject="x")
        assert journal.prompt_hint() == first and "mala respuesta" in first

    def test_summary_says_so_when_there_is_nothing(self, journal):
        assert "Sin problemas" in journal.summary()

    def test_summary_lists_counts(self, journal):
        journal.record(KIND_TURN_FAILED, "algo", subject="x")
        assert "1x" in journal.summary()


class TestGitHub:
    def test_github_is_off_by_default(self, tmp_path):
        assert IssueJournal(directory=tmp_path).github is False

    def test_a_missing_gh_binary_keeps_the_issue_on_disk(self, tmp_path, monkeypatch):
        import subprocess

        def boom(*a, **k):
            raise FileNotFoundError("no gh here")

        monkeypatch.setattr(subprocess, "run", boom)
        j = IssueJournal(directory=tmp_path / "issues", github=True)
        j.record(KIND_TURN_FAILED, "algo")  # must not raise
        assert list((tmp_path / "issues").glob("*.md"))

    def test_only_a_repeat_does_not_refile_on_github(self, tmp_path, monkeypatch):
        """Filing the same issue every occurrence is how a repo gets muted."""
        calls = []
        import subprocess

        class Result:
            returncode = 0
            stdout = "https://github.com/x/y/issues/1"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or Result())
        j = IssueJournal(directory=tmp_path / "issues", github=True)
        for _ in range(3):
            j.record(KIND_TURN_FAILED, "algo", subject="same")
        assert len(calls) == 1
