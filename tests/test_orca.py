"""Wawi sending coding tasks to Orca (wawi/orca.py).

Orca itself is never run: a fake runner answers the CLI the way the real one
did when it was checked by hand, and records every call it got."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wawi import orca, tools
from wawi.orca import OrcaBridge, OrcaCLI, OrcaError, OrcaSettings

GOOD_SPEC = (
    "Target: wawi/panel.py. Change: un botón de silencio. Constraints: no "
    "romper el menú. Ownership: solo wawi/panel.py y sus tests. Acceptance: "
    "pytest pasa y el botón aparece."
)


def ok(result: dict) -> tuple[int, str, str]:
    return 0, json.dumps({"id": "x", "ok": True, "result": result}), ""


def fail(code: str, rc: int = 1, **extra) -> tuple[int, str, str]:
    body = {"id": "x", "ok": False, "error": {"code": code, "message": code}}
    body.update(extra)
    return rc, json.dumps(body), ""


class FakeOrca:
    """Answers `orca ...` like the real CLI. Handlers are keyed by the first
    words of the command; a list of answers is consumed one per call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.terminals = 0
        self.alive: set[str] = set()
        self.answers: dict[str, list] = {}
        self.default = {
            "status": ok({"runtime": {"reachable": True, "state": "ready"}}),
            "orchestration run-create": ok({"run": {"id": "run_1"}}),
            "orchestration run-use": ok({"run": {"id": "run_1"}}),
            "orchestration worker-release": ok({"state": "released"}),
            "orchestration reply": ok({"message": {"id": "msg_r"}}),
            "orchestration worker-stop": ok({"state": "stopped"}),
            "orchestration task-update": ok({}),
            "orchestration worker-list": ok({"workers": []}),
            "orchestration worker-show": ok({}),
            "repo list": ok({"repos": [
                {"id": "r1", "path": "/Users/x/orca/projects/bruno", "displayName": "bruno"},
            ]}),
            "orchestration worker-start": ok({
                "taskId": "task_1", "dispatchId": "ctx_1", "state": "ready",
                "worktree": {"path": "/tmp/wt/wawi-boton", "branch": "refs/heads/wawi-boton"},
            }),
            "orchestration check": ok({"deliveryId": None, "messages": [], "timedOut": True}),
            "terminal read": ok({"terminal": {"tail": ["Ready", "Working on task"], "source": "screen"}}),
            "terminal send": ok({"accepted": True}),
            "terminal wait": ok({"wait": {"satisfied": True}}),
        }

    def queue(self, key: str, *answers) -> None:
        self.answers.setdefault(key, []).extend(answers)

    def __call__(self, argv, env, timeout):
        args = argv[1:]
        self.calls.append(args)
        self.envs.append(env)
        key2 = " ".join(args[:2])
        if key2 == "terminal create":
            self.terminals += 1
            handle = f"term_{self.terminals}"
            self.alive.add(handle)
            return ok({"terminal": {"handle": handle}})
        if key2 == "terminal show":
            handle = args[args.index("--terminal") + 1]
            if handle in self.alive:
                return ok({"terminal": {"handle": handle, "connected": True}})
            return fail("terminal_handle_stale")
        for key in (key2, args[0]):
            if self.answers.get(key):
                answer = self.answers[key].pop(0)
                if isinstance(answer, Exception):
                    raise answer
                return answer
            if key in self.default:
                return self.default[key]
        return fail("unknown_command")

    def commands(self, key: str) -> list[list[str]]:
        n = len(key.split())
        return [c for c in self.calls if " ".join(c[:n]) == key]


@pytest.fixture
def fake() -> FakeOrca:
    return FakeOrca()


def make(tmp_path: Path, fake: FakeOrca, notes: list | None = None, **over) -> OrcaBridge:
    raw = {
        "local_file": str(tmp_path / "orca_local.json"),
        "spec_dir": str(tmp_path / "issues"),
        "default_repo": str(tmp_path / "jarvis"),
        **over,
    }
    sink = notes if notes is not None else []
    clock = over.pop("clock", None) or (lambda: 1000.0)
    return OrcaBridge(
        OrcaSettings.from_dict(raw), OrcaCLI("/bin/orca", fake),
        notify=lambda kind, text: sink.append((kind, text)),
        clock=clock, sleep=lambda s: False, background=False,
    )


def saved(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "orca_local.json").read_text())


# --------------------------------------------------------------------------- #


class TestCli:
    def test_resolution_order(self):
        assert orca.resolve_cli("/opt/orca", exists=lambda p: p == "/opt/orca",
                                which=lambda n: None) == "/opt/orca"
        assert orca.resolve_cli("", which=lambda n: "/usr/local/bin/orca",
                                exists=lambda p: True) == "/usr/local/bin/orca"
        assert orca.resolve_cli("", which=lambda n: None,
                                exists=lambda p: p == orca.APP_CLI) == orca.APP_CLI
        assert orca.resolve_cli("", which=lambda n: None, exists=lambda p: False) is None

    def test_orca_env_is_stripped(self, fake, monkeypatch):
        monkeypatch.setenv("ORCA_TERMINAL_HANDLE", "term_someone_else")
        monkeypatch.setenv("ORCA_WORKTREE_ID", "x")
        monkeypatch.setenv("HOME_SWEET", "kept")
        OrcaCLI("/bin/orca", fake).run("status")
        env = fake.envs[-1]
        assert not any(k.startswith("ORCA_") for k in env)
        assert env["HOME_SWEET"] == "kept"
        assert fake.calls[-1][-1] == "--json"

    def test_error_carries_code_and_payload(self, fake):
        fake.queue("status", fail("boom", failedStage="launch"))
        with pytest.raises(OrcaError) as err:
            OrcaCLI("/bin/orca", fake).run("status")
        assert err.value.code == "boom"
        assert err.value.payload["failedStage"] == "launch"

    def test_garbage_output(self):
        cli = OrcaCLI("/bin/orca", lambda a, e, t: (0, "not json", ""))
        with pytest.raises(OrcaError) as err:
            cli.run("status")
        assert err.value.code == "bad_output"


class TestTerminalAndRun:
    def test_created_lazily_and_persisted(self, tmp_path, fake):
        bridge = make(tmp_path, fake)
        assert fake.calls == []
        handle, run = bridge.ensure_run()
        assert (handle, run) == ("term_1", "run_1")
        create = fake.commands("terminal create")[0]
        assert f"path:{tmp_path / 'jarvis'}" in create and "Wawi" in create
        rc = fake.commands("orchestration run-create")[0]
        assert rc[rc.index("--from") + 1] == "term_1"
        assert saved(tmp_path)["terminal"] == "term_1"
        assert saved(tmp_path)["run"] == "run_1"

    def test_reused_from_disk(self, tmp_path, fake):
        make(tmp_path, fake).ensure_run()
        again = FakeOrca()
        again.alive.add("term_1")
        assert make(tmp_path, again).ensure_run() == ("term_1", "run_1")
        assert again.commands("terminal create") == []
        assert again.commands("orchestration run-create") == []

    def test_gone_terminal_is_recreated_and_run_rebound(self, tmp_path, fake):
        make(tmp_path, fake).ensure_run()
        later = FakeOrca()  # term_1 is not alive here
        later.terminals = 1
        handle, run = make(tmp_path, later).ensure_run()
        assert handle == "term_2" and run == "run_1"
        use = later.commands("orchestration run-use")[0]
        assert use[use.index("--id") + 1] == "run_1"
        assert use[use.index("--from") + 1] == "term_2"
        assert saved(tmp_path)["terminal"] == "term_2"

    def test_unreachable_runtime(self, tmp_path, fake):
        fake.queue("status", ok({"runtime": {"reachable": False}}))
        assert make(tmp_path, fake).available() is False


class TestVisibleTerminals:
    def test_empty_shell_opens_without_confirmation_or_run(self, tmp_path, fake):
        bridge = make(tmp_path, fake)
        ok_, _, data = bridge.open_terminal("shell")
        assert ok_ and data["handle"] == "term_1"
        assert bridge.pending_kind() is None
        create = fake.commands("terminal create")[0]
        assert "--focus" in create and "--command" not in create
        assert fake.commands("orchestration run-create") == []
        assert saved(tmp_path)["opened_terminals"][0]["repo"] == "jarvis"
        assert bridge.list_terminals()[2]["terminals"][0]["handle"] == "term_1"
        assert make(tmp_path, fake).list_terminals()[2]["terminals"][0]["handle"] == "term_1"

    def test_paid_agent_and_prompt_wait_for_voice_yes(self, tmp_path, fake):
        bridge = make(tmp_path, fake)
        ok_, _, data = bridge.open_terminal("codex", "bruno", "Fix the bug")
        assert ok_ and data["sent"] is False
        assert fake.commands("terminal create") == []
        assert bridge.hear("sí") == "Dale, la abro en Orca."
        create = fake.commands("terminal create")[0]
        assert create[create.index("--worktree") + 1] == "path:/Users/x/orca/projects/bruno"
        assert create[create.index("--command") + 1] == "codex"
        sent = fake.commands("terminal send")[0]
        assert len(fake.commands("terminal wait")) == 1
        assert sent[sent.index("--text") + 1] == "Fix the bug"
        assert fake.commands("orchestration run-create") == []

    def test_shell_prompt_and_terminal_send_require_voice_yes(self, tmp_path, fake):
        bridge = make(tmp_path, fake)
        bridge.open_terminal("shell", prompt="echo hi")
        assert fake.commands("terminal create") == []
        assert bridge.hear("no") == "Bueno, no la abro."
        bridge.open_terminal("shell")
        ok_, _, data = bridge.terminal_send("", "echo hi")
        assert ok_ and data["sent"] is False
        assert fake.commands("terminal send") == []
        assert bridge.hear("sí") == "Dale, se lo escribo."
        assert len(fake.commands("terminal send")) == 1

    def test_read_only_opened_terminal_and_schema(self, tmp_path, fake):
        bridge = make(tmp_path, fake)
        assert bridge.terminal_read("term_other")[2]["error"] == "nothing_opened"
        bridge.open_terminal("shell")
        ok_, speak, data = bridge.terminal_read("")
        assert ok_ and "Working on task" in speak and data["source"] == "screen"
        assert "--screen" in fake.commands("terminal read")[0]
        assert {s["name"] for s in tools.ORCA_TERMINAL_TOOLS} == {
            "open_terminal", "terminal_read", "terminal_send", "list_terminals"}
        assert all(s["animation"] == "laptop" for s in tools.ORCA_TERMINAL_TOOLS)

    def test_disabled_handoff(self, tmp_path, fake):
        bridge = make(tmp_path, fake, terminal_handoff_enabled=False)
        assert bridge.open_terminal("shell")[2]["error"] == "disabled"


class TestSendFlow:
    def test_incomplete_spec_is_refused(self, tmp_path, fake):
        ok_, speak, data = make(tmp_path, fake).draft("Botón", "hacé un botón")
        assert not ok_ and data["error"] == "incomplete_spec"
        assert "target" in data["missing"]

    def test_draft_waits_for_a_spoken_yes(self, tmp_path, fake):
        bridge = make(tmp_path, fake)
        ok_, speak, data = bridge.draft("Botón de silencio", GOOD_SPEC)
        assert ok_ and "¿La mando a Orca?" in speak and data["sent"] is False
        assert fake.commands("orchestration worker-start") == []
        reply = bridge.hear("sí, dale")
        assert reply == "Dale, se la mando a Orca."
        start = fake.commands("orchestration worker-start")
        assert len(start) == 1
        argv = start[0]
        assert argv[argv.index("--from") + 1] == "term_1"
        assert argv[argv.index("--run") + 1] == "run_1"
        assert argv[argv.index("--worktree") + 1] == "new-top-level"
        assert argv[argv.index("--repo") + 1] == f"path:{tmp_path / 'jarvis'}"
        assert argv[argv.index("--name") + 1] == "wawi-boton-de-silencio"
        assert argv[argv.index("--agent") + 1] == "claude"
        spec = argv[argv.index("--spec") + 1]
        assert GOOD_SPEC in spec and "Co-Authored-By: Claude Opus 5.5" in spec
        assert "NO hagas merge" in spec and "config.yaml" in spec
        entry = saved(tmp_path)["dispatches"]["ctx_1"]
        assert entry["branch"] == "wawi-boton" and entry["task"] == "task_1"
        record = (tmp_path / "issues" / "boton-de-silencio.md").read_text()
        assert "Botón de silencio" in record and "ctx_1" in record

    def test_no_drops_the_draft(self, tmp_path, fake):
        bridge = make(tmp_path, fake)
        bridge.draft("Botón", GOOD_SPEC)
        assert bridge.hear("no") == "Bueno, no la mando."
        assert bridge.hear("sí") is None  # nothing pending any more
        assert fake.commands("orchestration worker-start") == []

    def test_an_edit_is_not_a_yes(self, tmp_path, fake):
        bridge = make(tmp_path, fake)
        bridge.draft("Botón", GOOD_SPEC)
        assert bridge.hear("sí, pero cambiale el título a otra cosa") is None
        assert fake.commands("orchestration worker-start") == []

    def test_a_late_yes_is_not_a_yes(self, tmp_path, fake):
        now = [1000.0]
        bridge = make(tmp_path, fake, clock=lambda: now[0])
        bridge.draft("Botón", GOOD_SPEC)
        now[0] += orca.CONFIRM_TIMEOUT_S + 1
        assert bridge.hear("sí") is None
        assert fake.commands("orchestration worker-start") == []

    def test_failed_start_is_reported_never_relaunched(self, tmp_path, fake):
        fake.queue("orchestration worker-start", fail(
            "runtime_error", result={"failedStage": "agent_launch",
                                     "residualResources": ["term_x"]}))
        notes: list = []
        bridge = make(tmp_path, fake, notes)
        bridge.draft("Botón", GOOD_SPEC)
        bridge.hear("dale")
        assert len(fake.commands("orchestration worker-start")) == 1
        assert notes[-1][0] == "failed" and "agent_launch" in notes[-1][1]
        assert "No lo reintento" in notes[-1][1]
        assert saved(tmp_path)["dispatches"] == {}

    def test_other_repo_by_name(self, tmp_path, fake):
        bridge = make(tmp_path, fake)
        ok_, speak, _ = bridge.draft("Arreglo", GOOD_SPEC, repo="Bruno")
        assert ok_ and "bruno" in speak
        bridge.hear("sí")
        argv = fake.commands("orchestration worker-start")[0]
        assert argv[argv.index("--repo") + 1] == "path:/Users/x/orca/projects/bruno"
        spec = argv[argv.index("--spec") + 1]
        assert "test_listener.py" not in spec and "NO hagas merge" in spec

    def test_unknown_repo(self, tmp_path, fake):
        ok_, speak, data = make(tmp_path, fake).draft("X", GOOD_SPEC, repo="nada")
        assert not ok_ and data["error"] == "unknown_repo" and "bruno" in speak

    def test_existing_worktree_gets_no_creation_flags(self, tmp_path, fake):
        bridge = make(tmp_path, fake, worktree="path:/tmp/here")
        bridge.draft("X", GOOD_SPEC)
        bridge.hear("sí")
        argv = fake.commands("orchestration worker-start")[0]
        assert "--repo" not in argv and "--name" not in argv
        assert argv[argv.index("--worktree") + 1] == "path:/tmp/here"

    def test_orca_closed(self, tmp_path, fake):
        fake.queue("status", fail("runtime_unavailable"))
        ok_, speak, data = make(tmp_path, fake).draft("X", GOOD_SPEC)
        assert not ok_ and data["error"] == "orca_down"


def done(dispatch: str, outcome: str, mid: str = "m1", body: str = "Hecho. Todo bien.") -> dict:
    return {"id": mid, "type": "worker_done", "subject": "done", "body": body,
            "payload": json.dumps({"taskId": "task_1", "dispatchId": dispatch, "outcome": outcome})}


def delivery(did: str, *messages) -> tuple[int, str, str]:
    return ok({"deliveryId": did, "messages": list(messages), "timedOut": False})


class TestWatcher:
    def started(self, tmp_path, fake, notes) -> OrcaBridge:
        bridge = make(tmp_path, fake, notes)
        bridge.draft("Botón", GOOD_SPEC)
        bridge.hear("sí")
        notes.clear()
        return bridge

    def test_done_announces_releases_then_acks(self, tmp_path, fake):
        notes: list = []
        bridge = self.started(tmp_path, fake, notes)
        fake.queue("orchestration check", delivery("d1", done("ctx_1", "succeeded")))
        bridge.watch()
        assert notes == [("done", "¡Terminé lo de «Botón»! Quedó en la rama wawi-boton.")]
        checks = fake.commands("orchestration check")
        first, last = checks[0], checks[-1]
        assert "--wait" in first and first[first.index("--terminal") + 1] == "term_1"
        assert "--ack" in last and last[last.index("--ack") + 1] == "d1"
        release = fake.commands("orchestration worker-release")
        assert release == [["orchestration", "worker-release", "--dispatch", "ctx_1", "--json"]]
        # release came before the ack
        assert fake.calls.index(release[0]) < fake.calls.index(last)
        assert saved(tmp_path)["dispatches"] == {}

    def test_failed_is_sad_with_the_summary(self, tmp_path, fake):
        notes: list = []
        bridge = self.started(tmp_path, fake, notes)
        fake.queue("orchestration check", delivery(
            "d1", done("ctx_1", "failed", body="Los tests no pasan. Detalle largo.")))
        bridge.watch()
        assert notes[0][0] == "failed" and "Los tests no pasan." in notes[0][1]
        assert len(fake.commands("orchestration worker-release")) == 1

    def test_question_is_kept_and_answered(self, tmp_path, fake):
        notes: list = []
        bridge = self.started(tmp_path, fake, notes)
        q = {"id": "msg_q", "type": "question", "subject": "¿Rojo o azul?",
             "body": "¿El botón va rojo o azul?",
             "payload": json.dumps({"dispatchId": "ctx_1"})}
        fake.queue("orchestration check",
                   delivery("d1", q),
                   delivery("d1", q),  # replayed before the ack landed
                   delivery("d2", done("ctx_1", "succeeded", mid="m2")))
        bridge.watch()
        questions = [n for n in notes if n[0] == "question"]
        assert len(questions) == 1 and "rojo o azul" in questions[0][1]
        # no release for a question
        assert fake.calls.index(fake.commands("orchestration worker-release")[0]) > \
            fake.calls.index(fake.commands("orchestration check")[2])
        assert saved(tmp_path)["questions"] == []  # its task is done

    def test_answer_replies_to_the_pending_question(self, tmp_path, fake):
        notes: list = []
        bridge = self.started(tmp_path, fake, notes)
        bridge.handle_message({"id": "msg_q", "type": "question", "body": "¿Rojo o azul?",
                               "payload": json.dumps({"dispatchId": "ctx_1"})})
        assert any("Pregunta pendiente" in t["description"]
                   for t in _with(bridge, tools.active_tools) if t["name"] == "answer_orca")
        ok_, speak, _ = bridge.answer("azul")
        assert ok_ and speak == "Se lo paso."
        reply = fake.commands("orchestration reply")[0]
        assert reply[reply.index("--id") + 1] == "msg_q"
        assert reply[reply.index("--body") + 1] == "azul"
        assert reply[reply.index("--from") + 1] == "term_1"
        assert bridge.answer("otra")[2]["error"] == "nothing_pending"

    def test_errors_back_off_and_keep_the_ack(self, tmp_path, fake):
        notes: list = []
        bridge = self.started(tmp_path, fake, notes)
        waits: list[float] = []
        bridge._sleep = lambda s: waits.append(s) or False
        fake.queue("orchestration check",
                   delivery("d1", done("ctx_1", "succeeded")),
                   fail("connection_lost"), fail("connection_lost"))
        bridge.watch()
        assert waits == [5.0, 10.0]
        acks = [c for c in fake.commands("orchestration check") if "--ack" in c]
        assert len(acks) == 3 and all(c[c.index("--ack") + 1] == "d1" for c in acks)

    def test_timeout_is_only_a_checkpoint(self, tmp_path, fake):
        notes: list = []
        bridge = self.started(tmp_path, fake, notes)
        fake.queue("orchestration check",
                   ok({"deliveryId": None, "messages": [], "timedOut": True}),
                   ok({"deliveryId": None, "messages": [], "timedOut": True}),
                   delivery("d1", done("ctx_1", "succeeded")))
        bridge.watch()
        assert notes[0][0] == "done"
        assert len(fake.commands("orchestration worker-release")) == 1
        assert fake.commands("orchestration worker-stop") == []

    def test_unknown_dispatch_is_never_released(self, tmp_path, fake):
        notes: list = []
        bridge = self.started(tmp_path, fake, notes)
        bridge.handle_message(done("ctx_other", "succeeded"))
        assert fake.commands("orchestration worker-release") == []
        assert notes == []

    def test_resumes_from_disk(self, tmp_path, fake):
        notes: list = []
        self.started(tmp_path, fake, notes)
        again = FakeOrca()
        again.alive.add("term_1")
        again.queue("orchestration check", delivery("d1", done("ctx_1", "succeeded")))
        bridge = make(tmp_path, again, notes)
        assert list(bridge.state.dispatches) == ["ctx_1"]
        bridge.watch()
        assert notes[-1][0] == "done"


class TestCancelAndStatus:
    def test_cancel_needs_a_yes_and_stops(self, tmp_path, fake):
        notes: list = []
        bridge = make(tmp_path, fake, notes)
        bridge.draft("Botón", GOOD_SPEC)
        bridge.hear("sí")
        ok_, speak, _ = bridge.ask_cancel("botón")
        assert ok_ and "¿Corto" in speak
        assert fake.commands("orchestration worker-stop") == []
        assert bridge.hear("dale") == "Dale, la corto."
        assert fake.commands("orchestration worker-stop")[0][3] == "ctx_1"
        upd = fake.commands("orchestration task-update")[0]
        assert "failed" in upd and "cancelled" in upd
        assert saved(tmp_path)["dispatches"] == {}

    def test_status(self, tmp_path, fake):
        now = [1000.0]
        bridge = make(tmp_path, fake, clock=lambda: now[0])
        assert bridge.status_summary()[0] == "No hay tareas en Orca."
        bridge.draft("Botón", GOOD_SPEC)
        bridge.hear("sí")
        now[0] += 600
        speak, _ = bridge.status_summary()
        assert speak == "Hay una tarea en Orca: «Botón» lleva 10 min."


def _with(bridge, fn):
    old = orca.get()
    orca.install(bridge)
    try:
        return fn()
    finally:
        orca.install(old)


class TestTools:
    def test_hidden_and_harmless_when_disabled(self, monkeypatch):
        monkeypatch.setattr(orca, "_BRIDGE", None)
        names = {t["name"] for t in tools.active_tools()}
        assert not names & {"send_to_orca", "orca_status", "answer_orca", "cancel_orca_task"}
        r = tools.dispatch("send_to_orca", {"title": "x", "spec": GOOD_SPEC})
        assert r.ok is False and r.data["error"] == "orca_disabled"
        assert orca.hear("sí") is None

    def test_offered_when_installed(self, tmp_path, fake, monkeypatch):
        monkeypatch.setattr(orca, "_BRIDGE", make(tmp_path, fake))
        names = {t["name"] for t in tools.active_tools()}
        assert {"send_to_orca", "orca_status", "answer_orca", "cancel_orca_task"} <= names
        send = next(t for t in tools.active_tools() if t["name"] == "send_to_orca")
        assert "mandalo a Orca" in send["description"]

    def test_tool_flow_through_dispatch(self, tmp_path, fake, monkeypatch):
        monkeypatch.setattr(orca, "_BRIDGE", make(tmp_path, fake))
        r = tools.dispatch("send_to_orca", {"title": "Botón", "spec": GOOD_SPEC})
        assert r.ok and r.speak.endswith("¿La mando a Orca?")
        assert fake.commands("orchestration worker-start") == []
        assert orca.hear("sí") == "Dale, se la mando a Orca."
        assert len(fake.commands("orchestration worker-start")) == 1
        assert "una tarea" in tools.dispatch("orca_status").speak

    def test_build_disabled_paths(self, tmp_path, monkeypatch):
        class Cfg:
            orca = {"enabled": False}

        assert orca.build(Cfg()) is None
        Cfg.orca = {"cli": str(tmp_path / "no-such-orca")}
        assert orca.build(Cfg()) is None
        exe = tmp_path / "orca"
        exe.write_text("#!/bin/sh\n")
        Cfg.orca = {"cli": str(exe), "local_file": str(tmp_path / "s.json")}
        bridge = orca.build(Cfg())
        assert bridge is not None and bridge.cli.cli == str(exe)


class TestConfig:
    def test_orca_section_and_local_file_beside_config(self, tmp_path):
        from wawi import config

        path = tmp_path / "config.yaml"
        path.write_text("orca:\n  agent: codex\n")
        cfg = config.load(path)
        assert cfg.orca["agent"] == "codex"
        assert cfg.orca["local_file"] == str(tmp_path / "orca_local.json")
        settings = OrcaSettings.from_dict(cfg.orca)
        assert settings.agent == "codex" and settings.worktree == "new-top-level"
        assert settings.default_repo == orca.REPO_ROOT


def test_visible_confirmation_click_dispatches_and_closes(tmp_path, fake):
    from wawi.confirm import REGISTRY
    bridge = make(tmp_path, fake)
    bridge.draft("Botón", GOOD_SPEC)
    pending = bridge._pending
    assert pending is not None and pending.confirmation_id
    assert REGISTRY.resolve(pending.confirmation_id, True)
    assert bridge.pending_kind() is None
    assert len(fake.commands("orchestration worker-start")) == 1
    assert bridge.hear("sí") is None


def test_unrelated_speech_keeps_orca_draft(tmp_path, fake):
    bridge = make(tmp_path, fake)
    bridge.draft("Botón", GOOD_SPEC)
    assert bridge.hear("estoy hablando con otra persona") is None
    assert bridge.pending_kind() == "send"
    assert bridge.hear("no") == "Bueno, no la mando."
