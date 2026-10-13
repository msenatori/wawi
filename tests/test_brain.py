"""Tests for the bounded tool-use brain.

Fully hermetic: the SDK query is replaced by a fake async generator and
``wawi.tools.dispatch`` is monkeypatched, so nothing here touches the network,
or a real model.
"""

from __future__ import annotations

import asyncio
import threading
import json

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage
from claude_agent_sdk.types import TextBlock

from wawi import brain
from wawi.tools import ToolResult


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model="test-model")


def _result(**over) -> ResultMessage:
    base = dict(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        total_cost_usd=0.01,
        usage={"input_tokens": 1200, "output_tokens": 60},
    )
    base.update(over)
    return ResultMessage(**base)


@pytest.fixture
def captured_bridge(monkeypatch):
    """Capture the bridge ``ask`` builds so a fake SDK can drive its handlers."""
    box: dict = {}
    original = brain._build_bridge

    def spy(state, screen_options):
        bridge = original(state, screen_options)
        box["bridge"] = bridge
        return bridge

    monkeypatch.setattr(brain, "_build_bridge", spy)
    return box


def _fake_query(script):
    """Build a fake ``_sdk_query``.

    *script* is an async callable receiving ``(prompt, options)`` and returning
    the messages to yield. Being async, it may await tool handlers first —
    exactly what the real SDK loop does between assistant turns.
    """

    async def fake(prompt, options):
        for message in await script(prompt, options):
            yield message

    return fake


def _tool(bridge, name):
    return bridge.handlers[f"mcp__{brain.MCP_SERVER_NAME}__{name}"]


SCREEN = f"mcp__{brain.MCP_SERVER_NAME}__{brain.SCREEN_TOOL}"


# --------------------------------------------------------------------------- #
# Tool round-trip
# --------------------------------------------------------------------------- #


def test_tool_round_trip_reaches_dispatch_and_feeds_the_answer(
    monkeypatch, captured_bridge
):
    calls: list[tuple[str, dict]] = []

    def fake_dispatch(name, args=None):
        calls.append((name, dict(args or {})))
        return ToolResult(
            True,
            "Anotado.",
            {"title": "se cortó a mitad", "count": 3, "detail": "x" * 500},
        )

    monkeypatch.setattr(brain.tools, "dispatch", fake_dispatch)

    async def script(prompt, options):
        bridge = captured_bridge["bridge"]
        out = await _tool(bridge, "report_issue")({"title": "se cortó a mitad"})
        payload = json.loads(out["content"][0]["text"])
        assert payload["ok"] is True
        return [_assistant("Blog terminó."), _result()]

    monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))

    res = brain.ask_sync("anotá que se cortó a mitad", model="m")

    assert calls == [("report_issue", {"title": "se cortó a mitad"})]
    assert res.tool_calls == ("report_issue",)
    assert res.text == "Blog terminó."
    assert res.cost_usd == 0.01
    assert res.input_tokens == 1200
    assert res.output_tokens == 60


def test_every_tool_from_tools_module_is_exposed():
    bridge = brain._build_bridge(brain._TurnState(), None)
    from wawi import tools as tools_mod

    expected = {f"mcp__{brain.MCP_SERVER_NAME}__{s['name']}" for s in tools_mod.TOOLS}
    assert expected == set(bridge.handlers)


def test_screen_tool_is_absent_by_default_and_opt_in_when_asked():
    assert SCREEN not in brain._build_bridge(brain._TurnState(), None).handlers
    with_screen = brain._build_bridge(brain._TurnState(), brain.ScreenOptions())
    assert SCREEN in with_screen.handlers


# --------------------------------------------------------------------------- #
# Iteration cap
# --------------------------------------------------------------------------- #


def test_iteration_cap_stops_dispatch_after_the_budget(monkeypatch):
    seen: list[str] = []

    def fake_dispatch(name, args=None):
        seen.append(name)
        return ToolResult(True, "ok", {})

    monkeypatch.setattr(brain.tools, "dispatch", fake_dispatch)

    state = brain._TurnState(max_iterations=3)
    bridge = brain._build_bridge(state, None)
    handler = _tool(bridge, "report_issue")

    outs = [asyncio.run(handler({})) for _ in range(6)]

    assert len(seen) == 3, "dispatch must stop at the cap"
    assert state.calls == ["report_issue"] * 3
    for out in outs[3:]:
        assert "budget exhausted" in out["content"][0]["text"].lower()


def test_default_cap_is_wired_into_max_turns(monkeypatch, captured_bridge):
    monkeypatch.setattr(brain.tools, "dispatch", lambda n, a=None: ToolResult(True, "ok"))
    seen: dict = {}

    async def script(prompt, options):
        seen["max_turns"] = options.max_turns
        seen["tools"] = options.tools
        seen["mcp"] = list(options.mcp_servers)
        return [_assistant("listo"), _result()]

    monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
    brain.ask_sync("hola", model="m")

    # +2: one turn for the refused call that spends the budget, one to answer.
    assert seen["max_turns"] == brain.MAX_TOOL_ITERATIONS + 2
    assert seen["tools"] == [], "built-in tool schemas must stay out of context"
    assert seen["mcp"] == [brain.MCP_SERVER_NAME], "no extra MCP servers"


# --------------------------------------------------------------------------- #
# Speech sanitization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, forbidden",
    [
        ("**blog** terminó", "*"),
        ("- uno\n- dos", "-"),
        ("## Estado", "#"),
        ("mirá `npm run dev`", "`"),
        ("listo 🎉✅", "🎉"),
        ("está en https://github.com/x/y ahora", "https"),
    ],
)
def test_sanitize_removes_unspeakable_markup(raw, forbidden):
    out = brain.sanitize_for_speech(raw)
    assert forbidden not in out
    assert "\n" not in out


def test_sanitize_drops_code_blocks_entirely():
    out = brain.sanitize_for_speech("Falló acá:\n```py\nraise ValueError()\n```\nNada más.")
    assert "ValueError" not in out
    assert out == "Falló acá: Nada más."


def test_sanitize_keeps_spanish_punctuation_and_accents():
    assert brain.sanitize_for_speech("¿Qué pasó con notas? Terminó.") == (
        "¿Qué pasó con notas? Terminó."
    )


def test_final_answer_is_sanitized_before_it_reaches_tts(monkeypatch, captured_bridge):
    async def script(prompt, options):
        return [_assistant("**blog** terminó 🎉\n- ver `logs`"), _result()]

    monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
    text = brain.ask_sync("y?", model="m").text

    for bad in ("*", "`", "🎉", "\n"):
        assert bad not in text


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #


def test_system_prompt_tells_the_model_to_admit_ignorance():
    """The one honesty rule that survives without tools to check anything.

    Wawi answers from the model's own knowledge now, so the failure mode is
    no longer "claimed a crashed agent finished" — it is a confident invented
    answer, and the only defence in the prompt is permission to say "no sé".
    """
    from wawi.prompts import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    assert "si no sabés algo, decilo" in lowered


def test_brain_never_invents_success_wording_of_its_own(monkeypatch, captured_bridge):
    """Wawi adds no wording of its own; whatever it says comes from the model."""

    async def script(prompt, options):
        return [_assistant("El agente de blog terminó."), _result()]

    monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
    text = brain.ask_sync("y blog?", model="m").text

    assert text == "El agente de blog terminó."
    for claim in ("salió todo bien", "terminó bien", "sin problemas", "exitosamente"):
        assert claim not in text.lower()


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_is_error_result_raises_brain_error(monkeypatch, captured_bridge):
    async def script(prompt, options):
        return [_result(is_error=True, subtype="error_max_turns")]

    monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
    with pytest.raises(brain.BrainError):
        brain.ask_sync("hola", model="m")


def test_max_turns_with_something_already_said_is_spoken_not_raised(
    monkeypatch, captured_bridge
):
    """A long turn that ran out of turns still has an answer. Say it."""

    async def script(prompt, options):
        return [
            _assistant("Ya lo mandé al 3."),
            _result(is_error=True, subtype="error_max_turns"),
        ]

    monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
    result = brain.ask_sync("hola", model="m")

    assert result.text == "Ya lo mandé al 3."


def test_max_turns_with_nothing_said_still_raises(monkeypatch, captured_bridge):
    """Silence plus an error is a failed turn, not an empty answer."""

    async def script(prompt, options):
        return [_assistant("   "), _result(is_error=True, subtype="error_max_turns")]

    monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
    with pytest.raises(brain.BrainError):
        brain.ask_sync("hola", model="m")


# --------------------------------------------------------------------------- #
# Ollama degradation
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_answers_without_tools_and_says_so(monkeypatch):
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"message": {"content": "**Ni idea**, no tengo acceso."}})

    monkeypatch.setattr(brain.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        brain.tools, "dispatch", lambda *a, **k: pytest.fail("ollama must not use tools")
    )

    res = brain.ask_sync("qué pasa con blog", model="m", backend="ollama")

    assert res.degraded is True
    assert res.tool_calls == ()
    assert "modo local" in res.text
    assert "*" not in res.text
    assert "tools" not in sent["body"], "no tool schemas may be sent to Ollama"


def test_ollama_degradation_notice_survives_an_empty_answer(monkeypatch):
    monkeypatch.setattr(
        brain.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse({"message": {"content": "   "}}),
    )
    res = brain.ask_ollama("hola", model="m", url="http://x/api/chat")
    assert res.text == brain._ES_DEGRADED


def test_ollama_unreachable_raises_brain_error(monkeypatch):
    def boom(req, timeout=None):
        raise brain.urllib.error.URLError("connection refused")

    monkeypatch.setattr(brain.urllib.request, "urlopen", boom)
    with pytest.raises(brain.BrainError):
        brain.ask_ollama("hola", model="m", url="http://x/api/chat")


def test_the_prompt_forbids_reading_paths_aloud():
    """A path is for the tools, not for the speaker.

    The screenshot flow puts an absolute path in the model's context on
    purpose — the user never knows it and never should. Reading it out is
    thirty seconds of slashes nobody can use.

    It applies in every language: ``capture_screen`` returns a path in all of
    them.
    """
    from wawi.prompts import system_prompt

    prompt = system_prompt("Wawi").lower()
    assert "nunca los digas en voz alta" in prompt


def test_the_prompt_names_the_assistant_everywhere():
    from wawi.prompts import system_prompt, system_prompt_no_tools

    for text in (system_prompt("Viernes"), system_prompt_no_tools("Viernes")):
        assert "Viernes" in text
        assert "Wawi" not in text


# --------------------------------------------------------------------------- #
# Session continuity
# --------------------------------------------------------------------------- #


def test_the_session_id_is_returned_so_the_next_turn_can_resume(
    monkeypatch, captured_bridge
):
    async def script(prompt, options):  # noqa: ARG001
        yield _assistant("Sun.")
        yield _result(session_id="sess-abc123")

    monkeypatch.setattr(brain, "_sdk_query", script)

    res = brain.ask_sync("cómo se escribe sol en inglés", model="m")

    assert res.session_id == "sess-abc123"


def test_resuming_stops_re_sending_our_own_copy_of_the_history(
    monkeypatch, captured_bridge
):
    """The server already holds the conversation.

    Sending our rendered summary as well pays twice to say the same thing, and
    the two disagree the moment our window slides past what the server kept.
    """
    seen = {}

    async def script(prompt, options):
        seen["prompt"] = prompt
        seen["resume"] = getattr(options, "resume", None)
        yield _assistant("ok")
        yield _result()

    monkeypatch.setattr(brain, "_sdk_query", script)

    brain.ask_sync(
        "y luna",
        model="m",
        history="Vos: cómo se escribe sol\nWawi: Sun.\n",
        resume="sess-abc123",
    )

    assert seen["prompt"] == "y luna"
    assert seen["resume"] == "sess-abc123"


def test_without_a_session_the_history_is_still_sent(monkeypatch, captured_bridge):
    """The first turn of a conversation has no session to inherit."""
    seen = {}

    async def script(prompt, options):  # noqa: ARG001
        seen["prompt"] = prompt
        yield _assistant("ok")
        yield _result()

    monkeypatch.setattr(brain, "_sdk_query", script)

    brain.ask_sync("y luna", model="m", history="Vos: sol\nWawi: Sun.\n")

    assert "Wawi: Sun." in seen["prompt"]


# --------------------------------------------------------------------------- #
# Persistent engine
# --------------------------------------------------------------------------- #


class _FakeClient:
    """Stands in for ClaudeSDKClient. Records what it was asked to do."""

    instances: list["_FakeClient"] = []

    def __init__(self, options=None):
        self.options = options
        self.connected = False
        self.disconnected = False
        self.prompts: list[str] = []
        self.fail_on_query = False
        _FakeClient.instances.append(self)

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def query(self, prompt, session_id="default"):  # noqa: ARG002
        if self.fail_on_query:
            raise RuntimeError("the CLI died")
        self.prompts.append(prompt)

    async def receive_response(self):
        yield _assistant("Sun.")
        yield _result(session_id="sess-live")


@pytest.fixture
def fake_client(monkeypatch):
    _FakeClient.instances.clear()
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClient)
    return _FakeClient


def _engine_ask(engine, prompt="hola", **over):
    kwargs = dict(
        model="m",
        system="SYS",
        max_iterations=7,
        screen_options=None,
        bypass_permissions=False,
    )
    kwargs.update(over)
    return engine.ask(prompt, **kwargs)


class TestEngine:
    """The connection is kept so a question does not pay to start the CLI.

    Measured on this machine: ~1.3s of process start-up per turn, against ~12ms
    of harness inside the turn. The win is in not launching it again.
    """

    def test_a_second_turn_reuses_the_same_connection(self, fake_client):
        engine = brain.Engine()
        try:
            _engine_ask(engine, "sol")
            _engine_ask(engine, "luna")
        finally:
            engine.close()

        assert len(fake_client.instances) == 1
        assert fake_client.instances[0].prompts == ["sol", "luna"]

    def test_a_changed_system_prompt_forces_a_reconnect(self, fake_client):
        """A live client holds the prompt it connected with.

        Teaching a preference or switching language rewrites that prompt, and
        answering under the old one is worse than paying to reconnect.
        """
        engine = brain.Engine()
        try:
            _engine_ask(engine, "sol")
            _engine_ask(engine, "sol", system="OTRO SYSTEM")
        finally:
            engine.close()

        assert len(fake_client.instances) == 2
        assert fake_client.instances[0].disconnected is True

    def test_a_changed_model_forces_a_reconnect(self, fake_client):
        engine = brain.Engine()
        try:
            _engine_ask(engine, "sol")
            _engine_ask(engine, "sol", model="otro-modelo")
        finally:
            engine.close()

        assert len(fake_client.instances) == 2

    def test_the_tool_budget_is_reset_between_turns(self, fake_client):
        # The handlers close over ONE state object; a fresh one would leave the
        # live handlers writing into the old, and the budget would never reset.
        engine = brain.Engine()
        try:
            _engine_ask(engine, "sol")
            engine._state.calls.extend(["a", "b", "c"])
            _engine_ask(engine, "luna")
            assert engine._state.calls == []
        finally:
            engine.close()

    def test_a_failed_turn_drops_the_connection_and_raises(self, fake_client):
        """A half-dead client answers nothing on every later turn."""
        engine = brain.Engine()
        try:
            _engine_ask(engine, "sol")
            fake_client.instances[0].fail_on_query = True

            with pytest.raises(RuntimeError):
                _engine_ask(engine, "luna")

            assert engine._client is None
        finally:
            engine.close()

    def test_closing_twice_is_harmless(self, fake_client):
        engine = brain.Engine()
        _engine_ask(engine, "sol")
        engine.close()
        engine.close()
        assert engine._client is None


class TestEngineFallback:
    def test_a_broken_engine_costs_latency_not_the_answer(
        self, monkeypatch, captured_bridge
    ):
        """The whole point of the fallback: never lose a turn to an optimisation."""

        class _Broken:
            def ask(self, *a, **kw):
                raise RuntimeError("connection refused")

        async def script(prompt, options):  # noqa: ARG001
            yield _assistant("Sun.")
            yield _result()

        monkeypatch.setattr(brain, "_sdk_query", script)

        res = brain.ask_sync("cómo se escribe sol", model="m", engine=_Broken())

        assert res.text == "Sun."


# --------------------------------------------------------------------------- #
# Streaming — speak the first sentence instead of waiting for the last
# --------------------------------------------------------------------------- #


def _delta(text: str):
    """One raw Anthropic text delta, wrapped the way the SDK yields it."""
    from claude_agent_sdk.types import StreamEvent

    return StreamEvent(
        uuid="u",
        session_id="s",
        event={
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    )


class TestSentenceStreaming:
    """A finished sentence goes out at once; a half-written one waits."""

    def test_only_finished_sentences_leave_the_buffer(self):
        said: list[str] = []
        emitter = brain._SentenceEmitter(said.append)

        emitter.feed("Ya lo mandé al tres. Está corri")
        assert said == ["Ya lo mandé al tres."]

        emitter.feed("endo ahora. Y termina")
        assert said == ["Ya lo mandé al tres.", "Está corriendo ahora."]

        emitter.flush()
        assert said == [
            "Ya lo mandé al tres.",
            "Está corriendo ahora.",
            "Y termina",
        ]

    def test_a_decimal_is_not_a_sentence_boundary(self):
        """"3." only ends a sentence when whitespace follows it."""
        said: list[str] = []
        emitter = brain._SentenceEmitter(said.append)
        emitter.feed("Tarda 3.")
        assert said == []
        emitter.feed("5 segundos.")
        assert said == []
        emitter.flush()
        assert said == ["Tarda 3.5 segundos."]

    def test_an_open_code_fence_holds_everything_back(self):
        """sanitize_for_speech deletes an open fence, so emitting mid-fence
        would speak a truncation of a block the finished answer drops."""
        said: list[str] = []
        emitter = brain._SentenceEmitter(said.append)

        emitter.feed("Mirá esto. ```python\nx = 1. y = 2.\n")
        assert said == ["Mirá esto."]

        emitter.feed("```\nListo. ")
        assert said == ["Mirá esto.", "Listo."]

    def test_each_sentence_is_sanitized_before_it_is_spoken(self):
        said: list[str] = []
        emitter = brain._SentenceEmitter(said.append)
        emitter.feed("Mirá **esto** en https://example.com/x ahora. ")
        assert said == ["Mirá esto en el enlace ahora."]

    def test_a_sentence_with_nothing_speakable_is_skipped(self):
        said: list[str] = []
        emitter = brain._SentenceEmitter(said.append)
        emitter.feed("```\ncode\n```\n")
        emitter.flush()
        assert said == []

    def test_a_broken_consumer_does_not_kill_the_turn(self):
        calls: list[str] = []

        def boom(sentence):
            calls.append(sentence)
            raise RuntimeError("the speaker is on fire")

        emitter = brain._SentenceEmitter(boom)
        emitter.feed("Una. Dos. Tres. ")
        emitter.flush()
        # It is dropped after the first failure rather than retried forever.
        assert calls == ["Una."]


class TestStreamingThroughTheBrain:
    def test_token_deltas_reach_the_speaker_before_the_answer_ends(
        self, monkeypatch, captured_bridge
    ):
        said: list[str] = []

        async def script(prompt, options):  # noqa: ARG001
            return [
                _delta("Hay tres agentes. "),
                _delta("Dos terminaron."),
                _assistant("Hay tres agentes. Dos terminaron."),
                _result(),
            ]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
        res = brain.ask_sync("cómo va", model="m", on_sentence=said.append)

        assert said == ["Hay tres agentes.", "Dos terminaron."]
        assert res.text == "Hay tres agentes. Dos terminaron."

    def test_the_recorded_answer_is_the_whole_answer(
        self, monkeypatch, captured_bridge
    ):
        """Streaming changes WHEN text is spoken, never what the turn returns."""
        said: list[str] = []

        async def script(prompt, options):  # noqa: ARG001
            return [
                _delta("Uno. Dos."),
                _assistant("Uno. Dos."),
                _result(total_cost_usd=0.5, session_id="sess-x"),
            ]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
        res = brain.ask_sync("hola", model="m", on_sentence=said.append)

        assert res.text == "Uno. Dos."
        assert res.cost_usd == 0.5
        assert res.session_id == "sess-x"
        assert " ".join(said) == res.text

    def test_without_deltas_it_streams_per_message_instead(
        self, monkeypatch, captured_bridge
    ):
        """A CLI too old for partial messages still speaks before the result."""
        said: list[str] = []

        async def script(prompt, options):  # noqa: ARG001
            return [
                _assistant("Voy a mirar."),
                _assistant("Listo, eran tres."),
                _result(),
            ]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
        res = brain.ask_sync("hola", model="m", on_sentence=said.append)

        assert said == ["Voy a mirar.", "Listo, eran tres."]
        assert res.text == "Voy a mirar. Listo, eran tres."

    def test_deltas_and_their_message_are_never_spoken_twice(
        self, monkeypatch, captured_bridge
    ):
        said: list[str] = []

        async def script(prompt, options):  # noqa: ARG001
            return [
                _delta("Primero. "),
                _assistant("Primero."),
                _delta("Segundo."),
                _assistant("Segundo."),
                _result(),
            ]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
        brain.ask_sync("hola", model="m", on_sentence=said.append)

        assert said == ["Primero.", "Segundo."]

    def test_a_failed_turn_still_delivers_what_was_already_written(
        self, monkeypatch, captured_bridge
    ):
        """Dying mid-sentence and going quiet reads as a crash from the room."""
        said: list[str] = []

        async def script(prompt, options):  # noqa: ARG001
            return [
                _delta("Empecé a mirar. Y ahí"),
                _result(is_error=True, subtype="error_during_execution"),
            ]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
        with pytest.raises(brain.BrainError):
            brain.ask_sync("hola", model="m", on_sentence=said.append)

        assert said == ["Empecé a mirar.", "Y ahí"]

    def test_partial_messages_are_only_requested_when_streaming(
        self, monkeypatch, captured_bridge
    ):
        seen: dict = {}

        async def script(prompt, options):  # noqa: ARG001
            seen["partial"] = getattr(options, "include_partial_messages", None)
            return [_assistant("Sí."), _result()]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))

        brain.ask_sync("hola", model="m")
        assert seen["partial"] is False

        brain.ask_sync("hola", model="m", on_sentence=lambda _s: None)
        assert seen["partial"] is True


class TestStreamingFallback:
    def test_a_retried_turn_stops_streaming_and_says_so(
        self, monkeypatch, captured_bridge
    ):
        """The retry writes a NEW answer; streaming it on top of the abandoned
        fragment would speak two versions of the same turn."""
        said: list[str] = []
        restarted: list[bool] = []

        class _BrokenAfterSpeaking:
            def ask(self, *a, on_sentence=None, **kw):  # noqa: ARG002
                on_sentence("Voy a mirar.")
                raise RuntimeError("the CLI died")

        async def script(prompt, options):  # noqa: ARG001
            return [_assistant("Eran tres."), _result()]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
        res = brain.ask_sync(
            "hola",
            model="m",
            engine=_BrokenAfterSpeaking(),
            on_sentence=said.append,
            on_restart=lambda: restarted.append(True),
        )

        assert said == ["Voy a mirar."]
        assert restarted == [True]
        assert res.text == "Eran tres."

    def test_a_retry_that_never_spoke_keeps_streaming(
        self, monkeypatch, captured_bridge
    ):
        said: list[str] = []
        restarted: list[bool] = []

        class _BrokenImmediately:
            def ask(self, *a, **kw):
                raise RuntimeError("connection refused")

        async def script(prompt, options):  # noqa: ARG001
            return [_delta("Eran tres."), _assistant("Eran tres."), _result()]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
        brain.ask_sync(
            "hola",
            model="m",
            engine=_BrokenImmediately(),
            on_sentence=said.append,
            on_restart=lambda: restarted.append(True),
        )

        assert said == ["Eran tres."]
        assert restarted == []

    def test_ollama_never_pretends_to_stream(self, monkeypatch):
        """It is called with stream: False — there is nothing to hand over."""
        said: list[str] = []
        monkeypatch.setattr(
            brain.urllib.request,
            "urlopen",
            lambda req, timeout=None: _FakeResponse({"message": {"content": "Hola."}}),
        )

        res = brain.ask_sync(
            "hola", model="m", backend="ollama", on_sentence=said.append
        )

        assert said == []
        assert res.text.startswith("Hola.")


class TestStreamingOnTheLiveEngine:
    def test_the_persistent_path_streams_too(self, fake_client, monkeypatch):
        said: list[str] = []

        async def receive(self):
            yield _delta("Sun. ")
            yield _assistant("Sun.")
            yield _result(session_id="sess-live")

        monkeypatch.setattr(_FakeClient, "receive_response", receive)

        engine = brain.Engine()
        try:
            res = _engine_ask(engine, "sol", on_sentence=said.append)
        finally:
            engine.close()

        assert said == ["Sun."]
        assert res.text == "Sun."

    def test_turning_streaming_on_reconnects(self, fake_client):
        """The flag is baked into the connected client's options: reusing a
        connection made without it would silently yield no deltas."""
        engine = brain.Engine()
        try:
            _engine_ask(engine, "sol")
            _engine_ask(engine, "luna", on_sentence=lambda _s: None)
        finally:
            engine.close()

        assert len(fake_client.instances) == 2


class TestTurnBudgetFailure:
    """error_max_turns is not a connection problem and must not sound like one."""

    def test_a_spent_budget_raises_its_own_error(self, monkeypatch, captured_bridge):
        async def script(prompt, options):  # noqa: ARG001
            return [_result(is_error=True, subtype="error_max_turns")]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
        with pytest.raises(brain.BrainBudgetError):
            brain.ask_sync("hola", model="m")

    def test_it_is_still_a_brain_error(self, monkeypatch, captured_bridge):
        """Everything that catches BrainError today must keep catching it."""
        assert issubclass(brain.BrainBudgetError, brain.BrainError)

    def test_other_failures_are_not_budget_failures(
        self, monkeypatch, captured_bridge
    ):
        async def script(prompt, options):  # noqa: ARG001
            return [_result(is_error=True, subtype="error_during_execution")]

        monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))
        with pytest.raises(brain.BrainError) as caught:
            brain.ask_sync("hola", model="m")
        assert not isinstance(caught.value, brain.BrainBudgetError)


# --------------------------------------------------------------------------- #
# Screenshot hygiene — only the latest image travels
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_screen(monkeypatch):
    """A 2880x1800 'screen' (15-inch Retina) without touching the real one."""
    from PIL import Image

    from wawi import screen

    monkeypatch.setattr(
        screen, "capture", lambda monitor=1: Image.new("RGB", (2880, 1800))
    )


def _images(content) -> list[dict]:
    return [b for b in content if b.get("type") == "image"]


def test_a_look_returns_the_image_and_its_one_line_note(
    monkeypatch, captured_bridge, fake_screen
):
    seen = {}

    async def script(prompt, options):  # noqa: ARG001
        seen["out"] = await _tool(captured_bridge["bridge"], brain.SCREEN_TOOL)({})
        return [_assistant("Veo el editor."), _result()]

    monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))

    res = brain.ask_sync("qué ves", model="m", screen_options=brain.ScreenOptions())

    content = seen["out"]["content"]
    assert len(_images(content)) == 1
    note = content[-1]["text"]
    assert note.startswith("[captura ") and note.endswith("1280x800, monitor 1]")
    assert res.screen_notes == (note,)


def test_two_looks_in_one_turn_send_a_single_image(
    monkeypatch, captured_bridge, fake_screen
):
    """Nothing the model can call moves a window, so a second look is the same
    picture again at ~1,400 tokens. It gets the first one's note instead."""
    outs = []

    async def script(prompt, options):  # noqa: ARG001
        handler = _tool(captured_bridge["bridge"], brain.SCREEN_TOOL)
        outs.append(await handler({}))
        outs.append(await handler({}))
        return [_assistant("Veo el editor."), _result()]

    monkeypatch.setattr(brain, "_sdk_query", _fake_query(script))

    res = brain.ask_sync("qué ves", model="m", screen_options=brain.ScreenOptions())

    assert sum(len(_images(o["content"])) for o in outs) == 1
    assert _images(outs[1]["content"]) == []
    assert res.screen_notes[0] in outs[1]["content"][0]["text"]
    assert len(res.screen_notes) == 1
    # The repeat still counts against the budget, so it cannot loop for free.
    assert res.tool_calls == (brain.SCREEN_TOOL, brain.SCREEN_TOOL)


def test_the_notes_preamble_is_empty_without_captures():
    assert brain.screen_notes_preamble(()) == ""
    text = brain.screen_notes_preamble(("[captura 14:50:08, 1280x800, monitor 1]",))
    assert "[captura 14:50:08, 1280x800, monitor 1]" in text
    assert brain.SCREEN_TOOL in text


class _TranscriptClient(_FakeClient):
    """A fake CLI that keeps what it was sent, the way the real one does.

    The real ``claude`` process re-sends its whole in-memory transcript on
    every turn, so ``transcript`` at the start of a turn IS what that turn
    ships. A prompt containing "pantalla" makes the model look.
    """

    engine: "brain.Engine | None" = None

    def __init__(self, options=None):
        super().__init__(options)
        self.transcript: list[dict] = []

    async def query(self, prompt, session_id="default"):  # noqa: ARG002
        await super().query(prompt)
        self.transcript.append({"type": "text", "text": prompt})

    async def receive_response(self):
        if "pantalla" in self.prompts[-1]:
            handler = self.engine._bridge.handlers[SCREEN]
            self.transcript.extend((await handler({}))["content"])
        yield _assistant("Listo.")
        yield _result(session_id=f"sess-{id(self)}")


class TestOnlyTheLatestImageTravels:
    """Four turns, looking at the screen on the 1st and 3rd.

    Before: the live client kept every image and re-sent both on turn 4
    (~4,100 tokens at 1568px). Now a turn that looked rotates the client, so
    turn 4 ships text only and the old captures survive as one-line notes.
    """

    def _ask(self, engine, prompt, history=""):
        result = _engine_ask(
            engine, prompt, screen_options=brain.ScreenOptions(), history=history
        )
        if engine._prewarm_thread is not None:
            engine._prewarm_thread.join(timeout=5)
        return result

    def test_a_text_turn_after_two_looks_carries_no_image(
        self, monkeypatch, fake_screen
    ):
        _FakeClient.instances.clear()
        import claude_agent_sdk

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _TranscriptClient)
        engine = brain.Engine()
        _TranscriptClient.engine = engine
        try:
            r1 = self._ask(engine, "mirá la pantalla")
            r2 = self._ask(engine, "y eso qué es", history="HIST2|")
            r3 = self._ask(engine, "mirá la pantalla otra vez", history="HIST3|")
            self._ask(engine, "y el 3?", history="HIST4|")
            last = engine._client
        finally:
            engine.close()

        assert len(r1.screen_notes) == 1 and r2.screen_notes == ()
        assert len(r3.screen_notes) == 1
        # Turn 4 runs on a client that never held an image.
        assert _images(last.transcript) == []
        assert last.prompts == ["HIST4|y el 3?"]
        # The clients that looked were dropped, images and all.
        lookers = [c for c in _FakeClient.instances if _images(c.transcript)]
        assert len(lookers) == 2 and all(c.disconnected for c in lookers)

    def test_the_conversation_rides_along_once_after_a_rotation(
        self, monkeypatch, fake_screen
    ):
        """Continuity: the fresh client gets the history as text on its first
        prompt — so "eso" still resolves — and never again after that."""
        _FakeClient.instances.clear()
        import claude_agent_sdk

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _TranscriptClient)
        engine = brain.Engine()
        _TranscriptClient.engine = engine
        try:
            self._ask(engine, "mirá la pantalla")
            self._ask(engine, "y eso qué es", history="HIST|")
            self._ask(engine, "gracias", history="HIST|")
        finally:
            engine.close()

        assert _FakeClient.instances[-1].prompts == [
            "HIST|y eso qué es",
            "gracias",
        ]

    def test_a_text_only_turn_keeps_the_connection(self, monkeypatch, fake_screen):
        _FakeClient.instances.clear()
        import claude_agent_sdk

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _TranscriptClient)
        engine = brain.Engine()
        _TranscriptClient.engine = engine
        try:
            self._ask(engine, "hola")
            self._ask(engine, "chau", history="HIST|")
        finally:
            engine.close()

        assert len(_FakeClient.instances) == 1
        assert _FakeClient.instances[0].prompts == ["hola", "chau"]

    def test_a_question_that_beats_the_background_reconnect_never_reuses_the_image(
        self, monkeypatch, fake_screen
    ):
        """If the next turn wins the lock before the rotation thread, it must
        reconnect itself rather than answer on the client holding the image."""
        _FakeClient.instances.clear()
        import claude_agent_sdk

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _TranscriptClient)
        real_start = threading.Thread.start

        def start(thread):
            if thread.name != "wawi-brain-prewarm":  # the engine loop must run
                real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", start)
        engine = brain.Engine()
        _TranscriptClient.engine = engine
        try:
            _engine_ask(engine, "mirá la pantalla", screen_options=brain.ScreenOptions())
            _engine_ask(
                engine, "y eso?", screen_options=brain.ScreenOptions(), history="H|"
            )
            last = engine._client
        finally:
            engine.close()

        assert _images(last.transcript) == []
        assert last.prompts == ["H|y eso?"]
        assert _FakeClient.instances[0].disconnected is True
