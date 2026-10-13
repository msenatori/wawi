"""Tests for rolling conversation memory and the readable transcript."""

from __future__ import annotations

import pytest

from wawi.conversation import DEFAULT_MAX_TURNS, Conversation, Turn


class TestWindow:
    def test_remembers_what_was_said(self):
        c = Conversation()
        c.add("cómo viene el 3", "El 3 es tienda, un agente trabajando")
        assert len(c) == 1
        assert c.turns()[0].user == "cómo viene el 3"

    def test_drops_the_oldest_turn_past_the_limit(self):
        """The window is bounded because every turn is re-sent and re-paid for."""
        c = Conversation(max_turns=3)
        for i in range(5):
            c.add(f"pregunta {i}", f"respuesta {i}")
        assert len(c) == 3
        assert [t.user for t in c.turns()] == ["pregunta 2", "pregunta 3", "pregunta 4"]

    def test_an_empty_exchange_is_not_remembered(self):
        """A dropped transcription must not waste a slot in the window."""
        c = Conversation()
        c.add("", "")
        c.add("   ", "  ")
        assert len(c) == 0

    def test_a_half_empty_exchange_is_still_remembered(self):
        c = Conversation()
        c.add("hola wawi", "")
        assert len(c) == 1

    def test_clear_forgets_everything(self):
        c = Conversation()
        c.add("algo", "otra cosa")
        c.clear()
        assert len(c) == 0
        assert c.as_prompt_context("Wawi") == ""

    def test_default_window_size(self):
        assert Conversation().max_turns == DEFAULT_MAX_TURNS


class TestPromptContext:
    def test_empty_conversation_adds_nothing_to_the_prompt(self):
        """No history must cost no tokens."""
        assert Conversation().as_prompt_context("Wawi") == ""

    def test_context_carries_both_sides_in_order(self):
        c = Conversation()
        c.add("cuántos proyectos hay", "Hay nueve")
        c.add("y el 3", "El 3 es tienda")
        context = c.as_prompt_context("Wawi")
        assert context.index("cuántos proyectos hay") < context.index("y el 3")
        assert "Hay nueve" in context
        assert "Wawi:" in context

    def test_context_uses_the_configured_name(self):
        c = Conversation()
        c.add("hola", "qué tal")
        assert "Wawi:" in c.as_prompt_context("Wawi")
        assert "Viernes:" in c.as_prompt_context("Viernes")

    def test_context_ends_by_framing_the_new_message(self):
        """Without this the model answers ABOUT the transcript, not from it."""
        c = Conversation()
        c.add("hola", "qué tal")
        assert c.as_prompt_context("Wawi").rstrip().endswith("Mensaje nuevo del usuario:")

    def test_a_very_long_answer_is_clipped_in_memory_only(self):
        c = Conversation()
        long_answer = "palabra " * 500
        c.add("contame todo", long_answer)
        rendered = c.as_prompt_context("Wawi")
        assert "…" in rendered
        assert len(rendered) < len(long_answer)
        # The stored turn keeps the full text; only the rendering is clipped.
        assert c.turns()[0].assistant == long_answer.strip()


class TestTranscriptLog:
    def test_nothing_is_written_without_a_configured_path(self, tmp_path):
        c = Conversation()
        c.add("hola", "qué tal")
        assert list(tmp_path.iterdir()) == []

    def test_each_turn_is_appended(self, tmp_path):
        path = tmp_path / "conv.log"
        c = Conversation(log_path=path, assistant_name="Wawi")
        c.add("hola wawi", "qué necesitás")
        c.add("nada", "dale")
        body = path.read_text()
        assert "hola wawi" in body
        assert "qué necesitás" in body
        assert body.index("hola wawi") < body.index("nada")

    def test_tools_used_are_recorded(self, tmp_path):
        path = tmp_path / "conv.log"
        c = Conversation(log_path=path, assistant_name="Wawi")
        c.add("cómo va", "un agente trabajando", ["project_status"])
        assert "project_status" in path.read_text()

    def test_the_log_appends_across_sessions(self, tmp_path):
        path = tmp_path / "conv.log"
        Conversation(log_path=path).add("primera", "sesión")
        Conversation(log_path=path).add("segunda", "sesión")
        body = path.read_text()
        assert "primera" in body and "segunda" in body

    def test_a_turn_beyond_the_window_still_reaches_the_log(self, tmp_path):
        """The window bounds the PROMPT, never the transcript."""
        path = tmp_path / "conv.log"
        c = Conversation(max_turns=2, log_path=path)
        for i in range(5):
            c.add(f"pregunta {i}", f"respuesta {i}")
        body = path.read_text()
        assert len(c) == 2
        for i in range(5):
            assert f"pregunta {i}" in body

    def test_an_unwritable_path_never_costs_the_user_the_turn(self, tmp_path):
        blocked = tmp_path / "afile"
        blocked.write_text("not a directory")
        c = Conversation(log_path=blocked / "nested" / "conv.log")
        c.add("hola", "qué tal")   # must not raise
        assert len(c) == 1


class TestTurn:
    def test_render_normalizes_whitespace(self):
        turn = Turn("hola   \n  wawi", "qué  tal")
        assert turn.render("Wawi") == "Usuario: hola wawi\nWawi: qué tal"


@pytest.mark.parametrize("max_turns", [0, 1, 50])
def test_any_window_size_is_accepted(max_turns):
    c = Conversation(max_turns=max_turns)
    for i in range(5):
        c.add(f"p{i}", f"r{i}")
    assert len(c) == min(max_turns, 5)
