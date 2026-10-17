from pathlib import Path

import pytest

from wawi.board import Board, render_page, render_abilities
from wawi import board as board_module
from wawi import tools


def test_markdown_table_and_escaping():
    page = render_page("A < B", "| A | B |\n|---|---|\n| 1 | 2 |", "markdown")
    assert "<table>" in page.html
    assert "A &lt; B" in page.html
    assert "color-scheme:light dark" in page.html


def test_html_keeps_inline_js():
    page = render_page("Converter", "<button onclick=\"convert()\">Go</button><script>function convert(){}</script>", "html")
    assert "onclick" in page.html and "<script>" in page.html


def test_board_theme_covers_markdown_and_converter_controls():
    markdown = render_page("Notes", "# Heading\n\n- Item\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint(1)\n```")
    converter = render_page("Converter", "<input><button>Convert</button><textarea></textarea>", "html")
    for page in (markdown, converter):
        assert '"Chalkboard SE"' in page.html
        assert "--paper:#fffbee" in page.html
        assert "--paper:#263b34" in page.html
        assert "button,input,textarea,select" in page.html
    assert "<table>" in markdown.html
    assert "<pre>" in markdown.html


def test_invalid_content():
    with pytest.raises(ValueError):
        render_page("x", "", "markdown")
    with pytest.raises(ValueError):
        render_page("x", "ok", "pdf")


def test_history_and_persistence(tmp_path):
    board = Board(directory=tmp_path)
    for i in range(12):
        board.show(f"Board {i}", f"Item {i}")
    assert len(board.history) == 10
    assert len(list(tmp_path.glob("board-*.html"))) == 10
    assert board.previous().title == "Board 10"
    assert board.current.title == "Board 10"
    assert board.hide()
    assert board.current is None


def test_tools_are_conditional_and_spoken_text_short(tmp_path):
    board = Board(directory=tmp_path)
    old = tools._BOARD
    try:
        tools.set_board(board)
        names = {spec["name"] for spec in tools.active_tools()}
        assert {"show_board", "hide_board", "previous_board", "show_abilities"} <= names
        result = tools.dispatch("show_board", {"title": "Test", "content": "# Huge\n" * 100})
        assert result.ok and result.speak == "Te lo dejo en la pizarra."
        assert tools.dispatch("previous_board").ok is False
        assert tools.dispatch("hide_board").ok
    finally:
        tools.set_board(old)


def test_abilities_cover_every_active_tool_with_voice_example(tmp_path):
    old = tools._BOARD
    try:
        tools.set_board(Board(directory=tmp_path))
        active = {spec["name"] for spec in tools.active_tools()}
        groups = tools.ability_groups()
        entries = [entry for _heading, items in groups for entry in items]
        assert active <= {name for name, _, _ in entries}
        assert all(description and example.startswith("Wawi,") for _, description, example in entries)
        assert all("example" not in spec and "ability" not in spec for spec in tools.active_tools())
        result = tools.dispatch("show_abilities")
        assert result.ok and result.speak == "Mirá todo lo que sé hacer."
        assert tools._BOARD.current.format == "markdown"  # no live window: ordinary board
        assert board_module.render_abilities(groups).format == "abilities"
    finally:
        tools.set_board(old)


def test_abilities_scroll_escapes_tool_text_and_falls_back_without_art(tmp_path, monkeypatch):
    groups = [("Grupo", [("new_tool", "<script>bad</script>", "Wawi, probá <eso>")])]
    page = render_abilities(groups)
    assert "overflow-y:auto" in page.html and "@keyframes unroll" in page.html
    assert "&lt;script&gt;" in page.html and "&lt;eso&gt;" in page.html
    monkeypatch.setattr(board_module, "ROLL_ART", tmp_path / "missing.png")
    board = Board(directory=tmp_path)
    assert board.show_abilities(groups).format == "markdown"
    assert "<script>" not in board.current.html


def test_new_active_tool_appears_without_a_catalog_edit(monkeypatch):
    original = tools.active_tools
    monkeypatch.setattr(tools, "active_tools", lambda: [*original(), {
        "name": "new_plugin_tool", "description": "Probá un conector nuevo.",
        "input_schema": {"type": "object", "properties": {}}}])
    groups = tools.ability_groups()
    assert any(name == "new_plugin_tool" and example.startswith("Wawi,")
               for _, items in groups for name, _, example in items)


def test_enabled_obsidian_connector_appears_only_while_installed():
    class Connector:
        def __bool__(self):
            return True

        def allowed_tools(self):
            return ["mcp__obsidian__obsidian_simple_search"]

    old = tools._CONNECTORS
    try:
        tools.set_connectors(Connector())
        assert any(name == "mcp__obsidian__obsidian_simple_search"
                   for _, items in tools.ability_groups() for name, _, _ in items)
        tools.set_connectors(None)
        assert not any(name.startswith("mcp__obsidian")
                       for _, items in tools.ability_groups() for name, _, _ in items)
    finally:
        tools.set_connectors(old)


def test_scroll_waits_for_ladder_then_falls_back_if_climb_stalls(tmp_path, monkeypatch):
    from wawi.pet import api
    board = Board(directory=tmp_path)
    shown = []
    monkeypatch.setattr(board, "_show_native", lambda page, window: shown.append(page.format))
    monkeypatch.setattr(api, "at_window", lambda: False)
    board.show_abilities([("Cosas", [("x", "Algo", "Wawi, algo")])])
    board.tick()
    assert shown == []
    monkeypatch.setattr(api, "at_window", lambda: True)
    board.tick()
    assert shown == ["abilities"]
    board.show_abilities([("Cosas", [("x", "Algo", "Wawi, algo")])])
    monkeypatch.setattr(api, "at_window", lambda: False)
    board._abilities_deadline = 0
    board.tick()
    assert shown[-1] == "markdown"


def test_an_embedded_photo_does_not_count_against_the_text_limit():
    """Observed: a curiosity with its Wikipedia photo was refused as too long."""
    photo = "data:image/jpeg;base64," + "A" * 400_000
    page = render_page("Lapacho", f"<img src='{photo}'><p>Florece en primavera.</p>", "html")
    assert photo in page.html
    with pytest.raises(ValueError):
        render_page("x", f"<img src='{photo}'>" + "x" * 250_000, "html")
