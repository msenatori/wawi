"""Delegated work: two agents that only ever touch through the mailbox.

What is pinned down here is the separation, because every way it can break is
silent. The executor blocking forever looks like a finished task. An approval
that defaults to yes looks like an approval. A second executor started over the
first looks like the first one is still running. None of those raise, and none
of them are visible from the outside until the damage is done.

Nothing here talks to the SDK: :func:`wawi.task._drive_session` is the seam,
and it is replaced wholesale.
"""

from __future__ import annotations

import threading
import time

import pytest

from wawi import task as task_module
from wawi.task import Mailbox, RequestKind, TaskOptions, TaskRunner, TaskStatus


def _options(**kw) -> TaskOptions:
    base = dict(model="claude-test", cwd="/tmp", ask_timeout_s=0.2,
                approval_timeout_s=0.2)
    base.update(kw)
    return TaskOptions(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The mailbox
# --------------------------------------------------------------------------- #


class TestAskRoundTrip:
    """The executor asks, the ORCHESTRATOR answers. Neither touches the mic."""

    def test_an_answer_reaches_the_waiting_executor(self):
        box = Mailbox()
        got: list[str] = []

        def _executor():
            got.append(box.ask("¿En qué carpeta?", timeout=5.0))

        thread = threading.Thread(target=_executor, daemon=True)
        thread.start()
        _wait_for(lambda: box.pending() is not None)

        assert box.answer("en tienda") is True
        thread.join(timeout=5.0)

        assert got == ["en tienda"]

    def test_the_pending_question_is_readable_while_it_waits(self):
        # This is what makes "¿cómo va?" answerable: the orchestrator can see
        # what the executor is stuck on without waking it.
        box = Mailbox()
        threading.Thread(target=lambda: box.ask("¿Borro el branch?", 5.0),
                         daemon=True).start()
        _wait_for(lambda: box.pending() is not None)

        pending = box.pending()
        assert pending is not None
        assert pending.kind is RequestKind.QUESTION
        assert pending.text == "¿Borro el branch?"

    def test_the_slot_is_free_again_once_answered(self):
        box = Mailbox()
        threading.Thread(target=lambda: box.ask("¿Sigo?", 5.0), daemon=True).start()
        _wait_for(lambda: box.pending() is not None)
        box.answer("dale")

        _wait_for(lambda: box.pending() is None)
        # And a late answer to a question nobody is asking is refused rather
        # than left to resolve the NEXT one.
        assert box.answer("dale otra vez") is False

    def test_an_empty_question_is_not_posted(self):
        box = Mailbox()
        assert box.ask("   ", timeout=5.0) == task_module.NO_ANSWER
        assert box.pending() is None


class TestAskTimeout:
    """Nobody answered. That is information, not a failure."""

    def test_it_returns_the_decide_alone_string(self):
        box = Mailbox()
        started = time.perf_counter()

        answer = box.ask("¿Uso pytest o unittest?", timeout=0.1)

        assert answer == task_module.NO_ANSWER
        assert time.perf_counter() - started < 2.0

    def test_it_never_raises(self):
        # Raising would end half an hour of work over a user who walked away.
        box = Mailbox()
        assert isinstance(box.ask("¿Y?", timeout=0.05), str)

    def test_a_timed_out_question_stops_being_pending(self):
        box = Mailbox()
        box.ask("¿Y?", timeout=0.05)
        assert box.pending() is None


class TestApproval:
    """Deny is the only safe default, and there are three ways to reach it."""

    def test_a_timeout_denies(self):
        allowed, reason = Mailbox().approval("Bash: rm -rf build", timeout=0.05)

        assert allowed is False
        assert reason == task_module.DENIED_TIMEOUT

    def test_a_yes_allows(self):
        box = Mailbox()
        out: list[tuple[bool, str]] = []
        threading.Thread(
            target=lambda: out.append(box.approval("Bash: pytest", 5.0)), daemon=True
        ).start()
        _wait_for(lambda: box.pending() is not None)
        box.answer("dale, sí")

        _wait_for(lambda: bool(out))
        assert out[0][0] is True

    def test_a_no_denies(self):
        box = Mailbox()
        out: list[tuple[bool, str]] = []
        threading.Thread(
            target=lambda: out.append(box.approval("Bash: rm -rf /", 5.0)), daemon=True
        ).start()
        _wait_for(lambda: box.pending() is not None)
        box.answer("no, ni loco")

        _wait_for(lambda: bool(out))
        assert out[0] == (False, task_module.DENIED_BY_USER)

    def test_an_answer_that_is_neither_denies(self):
        """A misheard yes costs a repeated question; a misheard no costs the repo."""
        box = Mailbox()
        out: list[tuple[bool, str]] = []
        threading.Thread(
            target=lambda: out.append(box.approval("Bash: git push --force", 5.0)),
            daemon=True,
        ).start()
        _wait_for(lambda: box.pending() is not None)
        box.answer("mmm no sé, fijate vos")

        _wait_for(lambda: bool(out))
        assert out[0][0] is False

    def test_cancelling_releases_the_waiter_denied(self):
        box = Mailbox()
        out: list[tuple[bool, str]] = []
        threading.Thread(
            target=lambda: out.append(box.approval("Bash: make", 5.0)), daemon=True
        ).start()
        _wait_for(lambda: box.pending() is not None)
        box.cancel()

        _wait_for(lambda: bool(out))
        assert out[0] == (False, task_module.DENIED_CANCELLED)


class TestSpeakingOnce:
    """A question is read out once, however many observers notice it."""

    def test_the_second_claim_loses(self):
        box = Mailbox()
        threading.Thread(target=lambda: box.ask("¿Sigo?", 5.0), daemon=True).start()
        _wait_for(lambda: box.pending() is not None)
        request = box.pending()
        assert request is not None

        assert box.mark_spoken(request.id) is True
        assert box.mark_spoken(request.id) is False

    def test_a_stale_id_never_claims_the_current_one(self):
        box = Mailbox()
        assert box.mark_spoken("q999") is False


class TestProgress:
    """Progress is shown, not spoken. It is also the answer to '¿cómo va?'."""

    def test_lines_are_kept_in_order_and_bounded(self):
        box = Mailbox()
        for i in range(task_module._PROGRESS_MAX + 5):
            box.progress(f"paso {i}")

        recent = box.recent_progress(limit=3)
        assert recent == ["paso 22", "paso 23", "paso 24"]
        assert box.latest() == "paso 24"

    def test_empty_lines_are_dropped(self):
        box = Mailbox()
        box.progress("   ")
        assert box.recent_progress() == []

    def test_the_observer_is_told_but_cannot_break_the_executor(self):
        seen: list[str] = []

        def _observer(event: str) -> None:
            seen.append(event)
            raise RuntimeError("the panel exploded")

        box = Mailbox(observer=_observer)
        box.progress("compilando")  # must not raise

        assert seen == [task_module.EVENT_PROGRESS]


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


class _FakeSession:
    """Stands in for the SDK: blocks until released, then returns a result."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.text = "listo"

    async def __call__(self, goal, options, on_client, **kw):
        self.started.set()
        import asyncio

        await asyncio.to_thread(self.release.wait, 10.0)
        return type("R", (), {"text": self.text})()


@pytest.fixture
def fake_sdk(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(task_module, "_drive_session", session)
    # Building real ClaudeAgentOptions spawns nothing, but it is still the SDK's
    # code and not what these tests are about.
    monkeypatch.setattr(TaskRunner, "_build_options", lambda self: object())
    return session


class TestOneAtATime:
    """Two executors on one mailbox cannot be told apart by the person answering."""

    def test_a_second_task_is_refused_while_one_runs(self, fake_sdk):
        runner = TaskRunner(_options())
        first = runner.start("arreglá los tests")
        assert first is not None
        assert fake_sdk.started.wait(5.0)

        assert runner.start("hacé otra cosa") is None
        assert runner.current is first

        fake_sdk.release.set()

    def test_a_new_one_is_allowed_once_the_first_ends(self, fake_sdk):
        runner = TaskRunner(_options())
        runner.start("uno")
        assert fake_sdk.started.wait(5.0)
        fake_sdk.release.set()
        _wait_for(lambda: not runner.is_running)

        second = runner.start("dos")
        assert second is not None and second.id == "t2"
        fake_sdk.release.set()

    def test_an_empty_goal_starts_nothing(self, fake_sdk):
        runner = TaskRunner(_options())
        assert runner.start("   ") is None
        assert runner.current is None


class TestStatusWhileRunning:
    """'¿Cómo va?' must be answerable without waking the executor."""

    def test_it_reports_the_goal_and_the_progress(self, fake_sdk):
        runner = TaskRunner(_options())
        runner.start("migrá la config")
        assert fake_sdk.started.wait(5.0)
        runner.mailbox.progress("leyendo config.py")

        status = runner.status()

        assert status["running"] is True
        assert status["id"] == "t1"
        assert status["goal"] == "migrá la config"
        assert status["progress"][-1] == "leyendo config.py"
        assert status["waiting_for_you"] is None

        fake_sdk.release.set()

    def test_it_says_when_the_executor_is_stuck_on_you(self, fake_sdk):
        runner = TaskRunner(_options(ask_timeout_s=5.0))
        runner.start("migrá la config")
        assert fake_sdk.started.wait(5.0)
        threading.Thread(
            target=lambda: runner.mailbox.ask("¿Piso el archivo?", 5.0), daemon=True
        ).start()
        _wait_for(lambda: runner.mailbox.pending() is not None)

        waiting = runner.status()["waiting_for_you"]

        assert waiting == {"kind": "QUESTION", "text": "¿Piso el archivo?"}
        fake_sdk.release.set()

    def test_nothing_running_reports_nothing(self):
        assert TaskRunner(_options()).status() == {"running": False, "task": None}

    def test_the_result_survives_the_end_of_the_task(self, fake_sdk):
        runner = TaskRunner(_options())
        runner.start("contá los tests")
        assert fake_sdk.started.wait(5.0)
        fake_sdk.release.set()
        _wait_for(lambda: not runner.is_running)

        status = runner.status()
        assert status["status"] == TaskStatus.DONE.value
        assert status["result"] == "listo"


class TestStop:
    """Stopping releases the blocked executor as well as interrupting it."""

    def test_it_frees_anything_waiting_on_an_answer(self, fake_sdk):
        runner = TaskRunner(_options(ask_timeout_s=30.0))
        runner.start("hacé algo largo")
        assert fake_sdk.started.wait(5.0)
        out: list[str] = []
        threading.Thread(
            target=lambda: out.append(runner.mailbox.ask("¿Sigo?", 30.0)), daemon=True
        ).start()
        _wait_for(lambda: runner.mailbox.pending() is not None)

        assert runner.stop() is True

        # Without the mailbox half of stop() this thread would sit on its Event
        # for the full 30 seconds after the task was already dead.
        _wait_for(lambda: bool(out))
        assert out[0] == task_module.NO_ANSWER
        assert runner.current.status is TaskStatus.STOPPED
        fake_sdk.release.set()

    def test_stopping_nothing_is_not_an_error(self):
        assert TaskRunner(_options()).stop() is False


class TestExecutorOptions:
    """The two agents' options differ on every axis that matters."""

    def test_the_builtins_are_never_auto_approved(self):
        """An allowed_tools entry naming a whole tool shadows can_use_tool.

        Listing "Bash" there would silently disable the approval question that
        is the only thing between a mistranscription and a shell.
        """
        options = TaskRunner(_options())._build_options()

        assert options.can_use_tool is not None
        assert options.permission_mode == "default"
        for entry in options.allowed_tools:
            assert entry.startswith("mcp__wawi_task__")

    def test_the_executor_gets_real_tools_and_no_turn_cap(self):
        options = TaskRunner(_options())._build_options()

        assert "Bash" in options.tools
        assert options.max_turns is None
        assert options.cwd == "/tmp"


# --------------------------------------------------------------------------- #
# The conversational side
# --------------------------------------------------------------------------- #


class TestConversationalTools:
    """Wawi drives the executor. The executor never drives Wawi."""

    @pytest.fixture(autouse=True)
    def _uninstall(self):
        from wawi import tools

        yield
        tools.set_task_runner(None)

    def test_the_task_schemas_are_absent_until_one_is_installed(self):
        from wawi import tools

        tools.set_task_runner(None)
        assert {t["name"] for t in tools.active_tools()} == {
            t["name"] for t in tools.TOOLS
        }

        tools.set_task_runner(TaskRunner(_options()))
        assert "start_task" in {t["name"] for t in tools.active_tools()}

    def test_every_task_tool_refuses_cleanly_when_tasks_are_off(self):
        from wawi import tools

        tools.set_task_runner(None)
        for name in ("start_task", "task_status", "answer_task", "stop_task"):
            result = tools.dispatch(name, {"goal": "x", "text": "x"})
            assert result.ok is False
            assert result.data["error"] == "tasks_disabled"

    def test_answer_task_resolves_the_pending_question(self, fake_sdk):
        from wawi import tools

        runner = TaskRunner(_options(ask_timeout_s=5.0))
        tools.set_task_runner(runner)
        runner.start("hacé algo")
        assert fake_sdk.started.wait(5.0)
        got: list[str] = []
        threading.Thread(
            target=lambda: got.append(runner.mailbox.ask("¿Cuál branch?", 5.0)),
            daemon=True,
        ).start()
        _wait_for(lambda: runner.mailbox.pending() is not None)

        result = tools.dispatch("answer_task", {"text": "el de main"})

        assert result.ok is True
        _wait_for(lambda: bool(got))
        assert got[0] == "el de main"
        fake_sdk.release.set()

    def test_answer_task_says_so_when_nothing_is_waiting(self):
        from wawi import tools

        tools.set_task_runner(TaskRunner(_options()))
        result = tools.dispatch("answer_task", {"text": "dale"})

        assert result.ok is False
        assert result.data["error"] == "nothing_pending"

    def test_start_task_refuses_a_second_one_out_loud(self, fake_sdk):
        from wawi import tools

        runner = TaskRunner(_options())
        tools.set_task_runner(runner)
        assert tools.dispatch("start_task", {"goal": "uno"}).ok is True
        assert fake_sdk.started.wait(5.0)

        result = tools.dispatch("start_task", {"goal": "dos"})

        assert result.ok is False
        assert result.data["error"] == "already_running"
        assert result.speak  # the user has to HEAR why nothing happened
        fake_sdk.release.set()


class TestSpokenDescription:
    """An approval question is heard, so it has to be short and specific."""

    def test_it_names_the_command_not_the_json(self):
        assert task_module._describe_call(
            "Bash", {"command": "pytest tests/ -q", "description": "run tests"}
        ) == "Bash: pytest tests/ -q"

    def test_it_names_the_file_for_an_edit(self):
        described = task_module._describe_call(
            "Write", {"file_path": "wawi/task.py", "content": "x" * 5000}
        )
        assert described == "Write: wawi/task.py"

    def test_a_long_detail_is_cut_rather_than_read_for_a_minute(self):
        described = task_module._describe_call("Bash", {"command": "echo " + "a" * 400})
        assert len(described) < 120
        assert described.endswith("…")

    def test_a_tool_with_nothing_to_show_is_named_alone(self):
        assert task_module._describe_call("Glob", {}) == "Glob"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _wait_for(predicate, timeout: float = 5.0) -> None:
    """Spin until *predicate* holds. Fails the test rather than hanging it."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"condition never held within {timeout}s")


# --------------------------------------------------------------------------- #
# The orchestrator seam
# --------------------------------------------------------------------------- #


class FakeCfg:
    conversation_mode = False
    wake_follow_up_s = 20.0
    wake_follow_up_idle_s = 0.0
    barge_in = False
    wake_engine = "whisper"
    wake_phrases = ("wawi",)


class FakeVoice:
    def __init__(self):
        self.said: list[str] = []

    def say(self, text):
        self.said.append(text)


class FakeSink:
    """Stands in for the TUI, the printer and the panel at once."""

    def __init__(self):
        self.task_calls: list[tuple[bool, str]] = []

    def set_texts(self, **kw):
        pass

    def answer(self, text):
        pass

    def set_task(self, active, line=""):
        self.task_calls.append((active, line))


def _orchestrator(runner):
    """An Orchestrator with only what the delegation seam touches.

    Built without __init__ on purpose, exactly as tests/test_orchestrator.py
    does: the real constructor loads Whisper, Kokoro and the microphone, none
    of which any of this depends on.
    """
    from wawi.orchestrator import Orchestrator
    from wawi.state import StateMachine

    orch = object.__new__(Orchestrator)
    orch._cfg = FakeCfg()
    orch._language = "es"
    orch._last_response = ""
    orch._barge_in_armed = False
    orch._turn_lock = threading.Lock()
    orch._stop_refresh = threading.Event()
    orch.listener = None
    orch.state = StateMachine()
    orch.voice = FakeVoice()
    orch.tui = FakeSink()
    orch.printer = FakeSink()
    orch.panel = FakeSink()
    orch.tasks = runner
    return orch


class TestOrchestratorSeam:
    """Everything the executor says reaches the user through Wawi's own voice."""

    def test_a_pending_question_is_spoken_once_and_prefixed(self, fake_sdk):
        runner = TaskRunner(_options(ask_timeout_s=5.0))
        orch = _orchestrator(runner)
        runner.mailbox.set_observer(orch._on_task_event)
        runner.start("hacé algo")
        assert fake_sdk.started.wait(5.0)

        threading.Thread(
            target=lambda: runner.mailbox.ask("¿Cuál branch?", 5.0), daemon=True
        ).start()
        _wait_for(lambda: bool(orch.voice.said))

        assert orch.voice.said == ["El agente pregunta: ¿Cuál branch?"]
        fake_sdk.release.set()

    def test_speaking_never_takes_the_mic_from_a_live_turn(self, fake_sdk):
        # The turn lock is what keeps the two agents from talking over each
        # other. Held, the executor's question waits rather than cutting in.
        runner = TaskRunner(_options(ask_timeout_s=5.0))
        orch = _orchestrator(runner)
        runner.mailbox.set_observer(orch._on_task_event)
        orch._turn_lock.acquire()

        threading.Thread(
            target=lambda: runner.mailbox.ask("¿Sigo?", 1.0), daemon=True
        ).start()
        _wait_for(lambda: runner.mailbox.pending() is not None)
        time.sleep(0.1)

        assert orch.voice.said == []
        orch._turn_lock.release()

    def test_speaking_returns_the_state_machine_to_idle(self, fake_sdk):
        from wawi.state import State

        runner = TaskRunner(_options())
        orch = _orchestrator(runner)
        orch._speak_for_task("terminé")
        _wait_for(lambda: bool(orch.voice.said))
        _wait_for(lambda: orch.state.state is State.IDLE)

        # A task that left the machine in SPEAKING would leave the microphone
        # paused for good — the listener is only resumed on the way OUT of it.
        assert orch.state.state is State.IDLE
        assert not orch._turn_lock.locked()

    def test_progress_reaches_the_hud_and_not_the_speaker(self, fake_sdk):
        runner = TaskRunner(_options())
        orch = _orchestrator(runner)
        runner.mailbox.set_observer(orch._on_task_event)
        runner.start("hacé algo")
        assert fake_sdk.started.wait(5.0)

        runner.mailbox.progress("corriendo pytest")

        assert ("corriendo pytest") in [line for _, line in orch.panel.task_calls]
        assert orch.voice.said == []
        fake_sdk.release.set()

    def test_the_outcome_is_spoken_when_the_executor_stops(self, fake_sdk):
        runner = TaskRunner(_options())
        orch = _orchestrator(runner)
        runner.mailbox.set_observer(orch._on_task_event)
        runner.start("contá los tests")
        assert fake_sdk.started.wait(5.0)
        fake_sdk.release.set()

        _wait_for(lambda: bool(orch.voice.said))
        assert orch.voice.said[0] == "Terminé la tarea t1. listo"


class TestOutcomeSentences:
    def test_each_status_has_its_own_line(self):
        from wawi.orchestrator import _outcome_sentence
        from wawi.task import Task

        done = Task(id="t1", goal="x", status=TaskStatus.DONE, result="quedó andando")
        failed = Task(id="t2", goal="x", status=TaskStatus.FAILED, error="boom")
        stopped = Task(id="t3", goal="x", status=TaskStatus.STOPPED)

        assert _outcome_sentence(done, "es") == "Terminé la tarea t1. quedó andando"
        assert _outcome_sentence(failed, "en") == "Task t2 failed."
        assert _outcome_sentence(stopped, "es") == "Corté la tarea t3."

    def test_a_task_still_running_says_nothing(self):
        from wawi.orchestrator import _outcome_sentence
        from wawi.task import Task

        assert _outcome_sentence(Task(id="t1", goal="x"), "es") == ""
        assert _outcome_sentence(None, "es") == ""

    def test_an_approval_is_worded_as_a_permission_not_a_question(self):
        from wawi.orchestrator import _request_sentence
        from wawi.task import Request

        request = Request(id="q1", kind=RequestKind.APPROVAL, text="Bash: pytest")
        assert _request_sentence(request, "es") == (
            "El agente quiere hacer esto: Bash: pytest. ¿Lo dejo?"
        )


class TestAnsweredRequestIsNotPending:
    """The window between an answer and the executor waking up.

    ``answer`` resolves the request but the slot stays filled until
    ``_collect`` runs on the EXECUTOR's thread, which does not happen until it
    wakes. Anything reading the mailbox in between used to find the question
    still waiting -- and an orchestrator that reacts to EVENT_RESOLVED reads it
    exactly then, which is how the same question got asked out loud twice.
    """

    def test_pending_goes_quiet_the_moment_it_is_answered(self):
        from wawi.task import Mailbox, RequestKind

        mb = Mailbox()
        mb._post(RequestKind.QUESTION, "Which file?")
        assert mb.pending() is not None
        mb.answer("stt.py")
        assert mb.pending() is None

    def test_the_resolved_event_does_not_re_offer_the_question(self):
        from wawi.task import EVENT_RESOLVED, Mailbox, RequestKind

        seen = []
        mb = Mailbox()

        def observer(event):
            if event == EVENT_RESOLVED:
                seen.append(mb.pending())

        mb.set_observer(observer)
        mb._post(RequestKind.QUESTION, "Which file?")
        mb.answer("stt.py")
        assert seen == [None]

    def test_status_stops_claiming_it_is_waiting_on_you(self):
        from wawi.task import Mailbox, RequestKind

        mb = Mailbox()
        request = mb._post(RequestKind.APPROVAL, "Bash: rm -rf /")
        assert mb.pending() is request
        mb.answer("no")
        # waiting_for_you is built from pending(); a stale one tells the model
        # the executor is blocked when it is already moving again.
        assert mb.pending() is None

    def test_the_executor_still_collects_its_answer(self):
        # The slot must stay filled for _collect to match identity -- the fix
        # hides it from readers, it does not drop it.
        from wawi.task import Mailbox, RequestKind

        mb = Mailbox()
        request = mb._post(RequestKind.QUESTION, "Which file?")
        mb.answer("stt.py")
        assert mb._collect(request) == "stt.py"


class TestSpokenDescription:
    """What a person hears before approving, and the path that is not theirs."""

    CWD = "/Users/someone/Work/wawi"

    def test_a_path_outside_the_project_is_called_out(self):
        # Measured behaviour, not a hypothetical: the model produced
        # "/Users/ana/wawi/stt.py" on a machine whose user is someone
        # else, and that invented path is exactly what reaches approval.
        from wawi.task import _describe_call

        said = _describe_call("Read", {"file_path": "/etc/passwd"}, self.CWD)
        assert "OUTSIDE" in said

    def test_a_path_inside_the_project_is_not(self):
        from wawi.task import _describe_call

        said = _describe_call("Read", {"file_path": f"{self.CWD}/wawi/stt.py"}, self.CWD)
        assert "OUTSIDE" not in said

    def test_a_relative_path_is_not_judged(self):
        # It resolves against the executor's cwd by definition.
        from wawi.task import _describe_call

        assert "OUTSIDE" not in _describe_call("Read", {"file_path": "wawi/stt.py"}, self.CWD)

    def test_a_grep_pattern_is_not_treated_as_a_path(self):
        from wawi.task import _describe_call

        assert "OUTSIDE" not in _describe_call("Grep", {"pattern": "mailbox"}, self.CWD)

    def test_no_cwd_means_nothing_can_be_judged(self):
        from wawi.task import _describe_call

        assert "OUTSIDE" not in _describe_call("Read", {"file_path": "/etc/passwd"}, "")


class TestShorten:
    """Progress lines are read aloud and drawn in the HUD."""

    def test_it_does_not_cut_mid_word(self):
        from wawi.task import _shorten

        text = "Read wawi/stt.py and report which arguments it passes. Before you start"
        assert not _shorten(text, 40).rstrip("…").endswith("Befor")

    def test_short_text_is_returned_whole_and_unmarked(self):
        from wawi.task import _shorten

        assert _shorten("done", 40) == "done"

    def test_whitespace_is_collapsed(self):
        from wawi.task import _shorten

        assert _shorten("a\n\n  b", 40) == "a b"

    def test_a_single_overlong_word_still_gets_cut(self):
        from wawi.task import _shorten

        assert len(_shorten("x" * 200, 40)) <= 40
