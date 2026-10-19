"""Tests for picking up a turn or task that a crash or restart cut off.

A "crash" here is simply a run that never reached the line that clears its
record: the file it left behind is read by a second store, exactly as the next
process would read it. Nothing talks to the SDK — :func:`wawi.task._drive_session`
and :func:`wawi.brain.ask_sync` are the seams, as in the rest of the suite.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass

import pytest

from wawi import brain
from wawi import config as config_mod
from wawi import orchestrator as orch_mod
from wawi import task as task_module
from wawi.inflight import KIND_TASK, KIND_TURN, Inflight
from wawi.task import TaskOptions, TaskRunner, TaskStatus
from tests.test_orchestrator import AUDIO, make_turn_orch

HOUR = 3600.0


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"condition never held within {timeout}s")


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


class TestStore:
    def test_completed_conversation_session_survives_restart(self, tmp_path):
        path = tmp_path / "inflight.json"
        first = Inflight(path)
        first.remember_session("sess-conversation")

        restarted = Inflight(path)
        assert restarted.conversation_session() == "sess-conversation"
        assert restarted.take(12 * HOUR) is None
        assert Inflight(path).conversation_session() == "sess-conversation"

    def test_taking_interrupted_work_preserves_completed_session(self, tmp_path):
        path = tmp_path / "inflight.json"
        first = Inflight(path)
        first.remember_session("sess-conversation")
        first.mark_task("t1", "ordená los imports", "sess-executor")

        restarted = Inflight(path)
        assert restarted.take(12 * HOUR).kind == KIND_TASK
        assert restarted.conversation_session() == "sess-conversation"
        assert Inflight(path).take(12 * HOUR) is None

    def test_a_turn_in_flight_is_on_disk_until_it_ends(self, tmp_path):
        path = tmp_path / "inflight.json"
        store = Inflight(path)

        store.mark_turn("qué hora es", "sess-c")
        on_disk = json.loads(path.read_text())
        assert on_disk["turn"]["text"] == "qué hora es"
        assert on_disk["turn"]["session_id"] == "sess-c"

        store.clear_turn()
        assert not path.exists()

    def test_a_clean_run_leaves_nothing_to_offer(self, tmp_path):
        path = tmp_path / "inflight.json"
        store = Inflight(path)
        store.mark_turn("hola", None)
        store.clear_turn()
        store.mark_task("t1", "ordená los imports")
        store.clear_task("t1")

        assert not path.exists()
        assert Inflight(path).take(12 * HOUR) is None

    def test_take_returns_the_record_once_and_deletes_it(self, tmp_path):
        path = tmp_path / "inflight.json"
        Inflight(path).mark_task("t1", "ordená los imports", "sess-x")

        record = Inflight(path).take(12 * HOUR)
        assert record is not None
        assert (record.kind, record.text, record.session_id) == (
            KIND_TASK, "ordená los imports", "sess-x",
        )
        # Never asked twice, even if the next run dies before the answer.
        assert not path.exists()
        assert Inflight(path).take(12 * HOUR) is None

    def test_the_task_wins_over_a_turn_that_overlapped_it(self, tmp_path):
        path = tmp_path / "inflight.json"
        store = Inflight(path)
        store.mark_task("t1", "migrá la base")
        store.mark_turn("cómo va", None)

        assert Inflight(path).take(12 * HOUR).kind == KIND_TASK

    def test_finishing_a_turn_keeps_the_task_running_beside_it(self, tmp_path):
        path = tmp_path / "inflight.json"
        store = Inflight(path)
        store.mark_task("t1", "migrá la base")
        store.mark_turn("cómo va", None)
        store.clear_turn()

        record = Inflight(path).take(12 * HOUR)
        assert record.kind == KIND_TASK

    def test_a_late_session_report_does_not_resurrect_a_finished_task(self, tmp_path):
        path = tmp_path / "inflight.json"
        store = Inflight(path)
        store.mark_task("t1", "algo")
        store.clear_task("t1")
        store.set_task_session("t1", "sess-late")

        assert not path.exists()

    def test_a_stale_record_is_ignored_and_removed(self, tmp_path):
        path = tmp_path / "inflight.json"
        old = time.time() - 13 * HOUR
        path.write_text(json.dumps({"task": {"text": "vieja", "ts": old, "id": "t1"}}))

        assert Inflight(path).take(12 * HOUR) is None
        assert not path.exists()

    @pytest.mark.parametrize(
        "body",
        [
            "{not json",
            "[1, 2]",
            json.dumps({"task": {"text": "", "ts": time.time()}}),
            json.dumps({"turn": {"text": "hola", "ts": "ayer"}}),
            json.dumps({"turn": {"text": "hola", "ts": time.time(), "session_id": 3}}),
        ],
    )
    def test_a_corrupt_file_degrades_to_nothing(self, tmp_path, body):
        path = tmp_path / "inflight.json"
        path.write_text(body)

        assert Inflight(path).take(12 * HOUR) is None
        assert not path.exists()

    def test_a_disabled_store_writes_nothing(self, tmp_path):
        store = Inflight(None)
        store.mark_turn("hola", None)
        store.mark_task("t1", "algo")

        assert store.take(12 * HOUR) is None
        assert list(tmp_path.iterdir()) == []


class TestConfig:
    def _load(self, tmp_path, body):
        path = tmp_path / "config.yaml"
        path.write_text("language: es\n" + body)
        return config_mod.load(path)

    def test_defaults(self, tmp_path):
        cfg = self._load(tmp_path, "")
        assert cfg.inflight_file is not None
        assert cfg.inflight_file.name == "inflight.json"
        assert cfg.resume_max_age_h == 12.0

    def test_an_empty_value_disables_it(self, tmp_path):
        cfg = self._load(tmp_path, 'inflight_file: ""\nresume_max_age_h: 2\n')
        assert cfg.inflight_file is None
        assert cfg.resume_max_age_h == 2.0


# --------------------------------------------------------------------------- #
# The executor
# --------------------------------------------------------------------------- #


class _Session:
    """Stands in for the SDK. Reports a session id, then blocks or fails."""

    def __init__(self, fail_resume: bool = False, fail_all: bool = False) -> None:
        self.calls: list[dict] = []
        self.release = threading.Event()
        self.reported = threading.Event()
        self.fail_resume = fail_resume
        self.fail_all = fail_all

    async def __call__(self, goal, options, on_client, *, resume=None, on_session=None):
        self.calls.append({"goal": goal, "resume": resume})
        if resume and self.fail_resume:
            raise RuntimeError("No conversation found with session ID")
        if self.fail_all:
            raise RuntimeError("boom")
        if on_session is not None:
            on_session("sess-exec")
        self.reported.set()
        await asyncio.to_thread(self.release.wait, 10.0)
        return type("R", (), {"text": "listo"})()


@pytest.fixture
def make_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(TaskRunner, "_build_options", lambda self: object())
    path = tmp_path / "inflight.json"

    def _make(**kw):
        session = _Session(**kw)
        monkeypatch.setattr(task_module, "_drive_session", session)
        runner = TaskRunner(TaskOptions(model="m", cwd=str(tmp_path)), inflight=Inflight(path))
        return runner, session, path

    return _make


class TestExecutorRecord:
    def test_a_killed_task_leaves_its_goal_and_session(self, make_runner):
        runner, session, path = make_runner()
        runner.start("ordená los imports")
        assert session.reported.wait(5.0)

        # kill -9 here: nothing else runs. What the next start reads:
        on_disk = json.loads(path.read_text())["task"]
        assert on_disk["text"] == "ordená los imports"
        assert on_disk["session_id"] == "sess-exec"
        session.release.set()

    def test_shutting_down_mid_task_keeps_the_record(self, make_runner):
        runner, session, path = make_runner()
        task = runner.start("ordená los imports")
        assert session.reported.wait(5.0)

        runner.close()
        session.release.set()
        _wait_for(lambda: task.finished_at is not None)

        assert json.loads(path.read_text())["task"]["session_id"] == "sess-exec"

    def test_a_finished_task_leaves_nothing(self, make_runner):
        runner, session, path = make_runner()
        task = runner.start("ordená los imports")
        session.release.set()
        _wait_for(lambda: task.finished_at is not None)

        assert task.status is TaskStatus.DONE
        assert not path.exists()

    def test_a_failed_task_leaves_nothing(self, make_runner):
        runner, session, path = make_runner(fail_all=True)
        task = runner.start("ordená los imports")
        _wait_for(lambda: task.finished_at is not None)

        assert task.status is TaskStatus.FAILED
        assert not path.exists()

    def test_a_task_the_user_stopped_leaves_nothing(self, make_runner):
        runner, session, path = make_runner()
        task = runner.start("ordená los imports")
        assert session.reported.wait(5.0)

        runner.stop()
        session.release.set()
        _wait_for(lambda: task.finished_at is not None)

        assert not path.exists()


class TestExecutorResume:
    def test_resume_continues_the_saved_session(self, make_runner):
        runner, session, path = make_runner()
        task = runner.start("ordená los imports", resume="sess-old")
        assert session.reported.wait(5.0)
        session.release.set()
        _wait_for(lambda: task.finished_at is not None)

        assert len(session.calls) == 1
        assert session.calls[0]["resume"] == "sess-old"
        assert "ordená los imports" in session.calls[0]["goal"]
        assert task.status is TaskStatus.DONE

    def test_a_session_that_cannot_be_resumed_starts_the_goal_over(self, make_runner):
        runner, session, path = make_runner(fail_resume=True)
        task = runner.start("ordená los imports", resume="sess-gone")
        assert session.reported.wait(5.0)
        session.release.set()
        _wait_for(lambda: task.finished_at is not None)

        assert [c["resume"] for c in session.calls] == ["sess-gone", None]
        assert session.calls[1]["goal"] == "ordená los imports"
        assert task.status is TaskStatus.DONE

    def test_when_the_retry_fails_too_the_task_fails_and_nothing_raises(
        self, make_runner
    ):
        runner, session, path = make_runner(fail_resume=True, fail_all=True)
        task = runner.start("ordená los imports", resume="sess-gone")
        _wait_for(lambda: task.finished_at is not None)

        assert len(session.calls) == 2
        assert task.status is TaskStatus.FAILED


class TestSessionWatch:
    def test_only_the_first_session_id_is_reported(self):
        @dataclass
        class Msg:
            session_id: str | None = None
            data: dict | None = None

        async def stream():
            yield Msg(data={"session_id": "s-init"})
            yield Msg(session_id="s-init")
            yield Msg(session_id="s-other")

        seen: list[str] = []

        async def drain():
            return [m async for m in task_module._watch_session(stream(), seen.append)]

        messages = asyncio.run(drain())
        assert len(messages) == 3
        assert seen == ["s-init"]


# --------------------------------------------------------------------------- #
# The orchestrator: one question on startup, answered by an ordinary turn
# --------------------------------------------------------------------------- #


class FakeTasks:
    def __init__(self) -> None:
        self.started: list[tuple[str, str | None]] = []
        self.is_running = False

    def start(self, goal, resume=None):
        self.started.append((goal, resume))
        return task_module.Task(id="t1", goal=goal)


def _say(o, text):
    o.stt = type("T", (), {"transcribe": staticmethod(lambda *a, **kw: text)})()


def _restarted_orch(path, **kw):
    """The next process: a turn-ready Orchestrator that has read *path*."""
    o = make_turn_orch()
    o.inflight = Inflight(path)
    o._resume_offer = o.inflight.take(12 * HOUR)
    for name, value in kw.items():
        setattr(o, name, value)
    return o


def _ask_and_wait(o):
    """Speak the startup question, then hand the turn lock back to the test."""
    o._turn_lock.release()
    o._offer_resume()
    _wait_for(lambda: bool(o.voice.said))
    _wait_for(lambda: not o._turn_lock.locked())
    o._turn_lock.acquire()


def _no_brain(monkeypatch):
    calls: list = []

    def fake(user_text, **kw):
        calls.append((user_text, kw.get("resume")))
        return brain.BrainResult(text="respuesta", tool_calls=(), session_id="sess-new")

    monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
    return calls


class TestStartupOffer:
    def test_a_killed_task_is_offered_once_on_the_next_start(self, tmp_path):
        path = tmp_path / "inflight.json"
        Inflight(path).mark_task("t1", "ordená los imports", "sess-exec")

        o = _restarted_orch(path, tasks=FakeTasks())
        _ask_and_wait(o)

        assert o.voice.said == ["Me cortaron mientras hacía ordená los imports, ¿lo retomo?"]
        assert o.panel.texts[-1] == o.voice.said[0]
        # Already off disk: a second restart asks nothing.
        assert Inflight(path).take(12 * HOUR) is None

    def test_yes_resumes_the_task_in_its_saved_session(self, tmp_path, monkeypatch):
        path = tmp_path / "inflight.json"
        Inflight(path).mark_task("t1", "ordená los imports", "sess-exec")
        calls = _no_brain(monkeypatch)

        o = _restarted_orch(path, tasks=FakeTasks())
        _ask_and_wait(o)
        _say(o, "sí, dale")
        o._do_turn(AUDIO)

        assert o.tasks.started == [("ordená los imports", "sess-exec")]
        assert o.voice.said[-1] == "Dale, lo retomo."
        assert calls == []

    def test_no_drops_it_silently(self, tmp_path, monkeypatch):
        path = tmp_path / "inflight.json"
        Inflight(path).mark_task("t1", "ordená los imports", "sess-exec")
        calls = _no_brain(monkeypatch)

        o = _restarted_orch(path, tasks=FakeTasks())
        _ask_and_wait(o)
        _say(o, "no")
        o._do_turn(AUDIO)

        assert o.tasks.started == []
        assert calls == []
        assert len(o.voice.said) == 1
        assert not path.exists()

    def test_talking_about_something_else_drops_it_and_answers(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "inflight.json"
        Inflight(path).mark_task("t1", "ordená los imports", "sess-exec")
        calls = _no_brain(monkeypatch)

        o = _restarted_orch(path, tasks=FakeTasks())
        _ask_and_wait(o)
        _say(o, "qué hora es")
        o._do_turn(AUDIO)
        _say(o, "sí")
        o._turn_lock.acquire()
        o._do_turn(AUDIO)

        assert o.tasks.started == []
        assert [c[0] for c in calls] == ["qué hora es", "sí"]

    def test_a_yes_long_after_the_question_is_not_an_answer(self, tmp_path, monkeypatch):
        path = tmp_path / "inflight.json"
        Inflight(path).mark_task("t1", "ordená los imports", "sess-exec")
        calls = _no_brain(monkeypatch)

        o = _restarted_orch(path, tasks=FakeTasks())
        _ask_and_wait(o)
        o._resume_offer_until = time.monotonic() - 1
        _say(o, "sí")
        o._do_turn(AUDIO)

        assert o.tasks.started == []
        assert calls == [("sí", None)]

    def test_a_task_is_not_offered_when_tasks_are_off(self, tmp_path):
        path = tmp_path / "inflight.json"
        Inflight(path).mark_task("t1", "ordená los imports", "sess-exec")

        o = _restarted_orch(path, tasks=None)
        o._turn_lock.release()
        o._offer_resume()
        time.sleep(0.05)

        assert o.voice.said == []
        assert o._resume_offer is None

    def test_a_clean_start_asks_nothing(self, tmp_path):
        o = _restarted_orch(tmp_path / "inflight.json", tasks=FakeTasks())
        o._turn_lock.release()
        o._offer_resume()
        time.sleep(0.05)

        assert o.voice.said == []

    def test_a_stale_or_corrupt_file_asks_nothing(self, tmp_path):
        path = tmp_path / "inflight.json"
        path.write_text("{garbage")
        o = _restarted_orch(path, tasks=FakeTasks())
        o._turn_lock.release()
        o._offer_resume()
        time.sleep(0.05)

        assert o.voice.said == []


class TestInterruptedTurn:
    def test_completed_turn_keeps_session_for_next_process(self, tmp_path):
        path = tmp_path / "inflight.json"
        o = make_turn_orch()
        o.inflight = Inflight(path)
        o._remember_session("sess-conversation")

        assert Inflight(path).conversation_session() == "sess-conversation"
        o._forget_session()
        assert Inflight(path).conversation_session() is None

    def test_the_turn_is_on_disk_only_while_the_brain_works(self, tmp_path, monkeypatch):
        path = tmp_path / "inflight.json"
        seen: list = []

        def fake(user_text, **kw):
            seen.append(json.loads(path.read_text())["turn"]["text"])
            return brain.BrainResult(text="listo", tool_calls=())

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o = make_turn_orch()
        o.inflight = Inflight(path)
        o._do_turn(AUDIO)

        assert seen == ["cómo va"]
        assert not path.exists()

    def test_a_failed_turn_leaves_nothing(self, tmp_path, monkeypatch):
        path = tmp_path / "inflight.json"

        def fake(user_text, **kw):
            raise brain.BrainError("down")

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o = make_turn_orch()
        o.inflight = Inflight(path)
        o._do_turn(AUDIO)

        assert not path.exists()

    def test_yes_replays_the_cut_turn_in_its_session(self, tmp_path, monkeypatch):
        path = tmp_path / "inflight.json"
        Inflight(path).mark_turn("poné música tranquila", "sess-conv")
        calls = _no_brain(monkeypatch)

        o = _restarted_orch(path)
        _ask_and_wait(o)
        assert o.voice.said == [
            "Me cortaron antes de responderte «poné música tranquila», ¿lo retomo?"
        ]
        _say(o, "sí")
        o._do_turn(AUDIO)

        assert calls == [("poné música tranquila", "sess-conv")]
        assert o._session_id == "sess-new"

    def test_a_session_that_cannot_be_resumed_is_retried_fresh(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "inflight.json"
        Inflight(path).mark_turn("poné música tranquila", "sess-gone")
        calls: list = []

        def fake(user_text, **kw):
            calls.append(kw.get("resume"))
            if kw.get("resume"):
                raise brain.BrainError("No conversation found")
            return brain.BrainResult(text="listo", tool_calls=())

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o = _restarted_orch(path)
        _ask_and_wait(o)
        _say(o, "sí")
        o._do_turn(AUDIO)

        assert calls == ["sess-gone", None]
        assert o._last_response == "listo"
