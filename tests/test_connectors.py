"""Tests for borrowed MCP servers (wawi/connectors.py).

Fully hermetic: every source file lives in a fake HOME under tmp_path, the SDK
status is a dict, and the "MCP server" is a counter behind the gate. The user's
real ~/.claude.json is never opened — the secrets below are canaries.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from wawi import brain, connectors
from wawi.config import ConfigError, _parse_connectors
from wawi.connectors import Connectors, discover

CANARY = "sk-canary-7f3a9e1d"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """A HOME with all three sources, each holding a canary secret."""
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    desktop = connectors.desktop_path(home)
    desktop.parent.mkdir(parents=True)
    cwd.mkdir()
    monkeypatch.setenv("NOTES_DIR", "/notas")

    claude = {
        "mcpServers": {
            "obsidian": {
                "type": "stdio",
                "command": "uvx",
                "args": ["mcp-obsidian"],
                "env": {"OBSIDIAN_API_KEY": CANARY},
            },
            "broken": {"type": "ws", "url": "ws://x"},
        },
        "projects": {
            "/some/project": {
                "mcpServers": {
                    "linear": {
                        "type": "http",
                        "url": "https://mcp.linear.app/mcp",
                        "headers": {"Authorization": f"Bearer {CANARY}"},
                    }
                }
            }
        },
    }
    mcp_json = {
        "mcpServers": {
            "notes": {"command": "notes-mcp", "args": ["${NOTES_DIR}", "${MISSING:-def}"]}
        }
    }
    desktop_cfg = {
        "mcpServers": {
            "slack": {"command": "slack-mcp", "env": {"SLACK_TOKEN": CANARY}},
            "obsidian": {"command": "shadowed"},
        }
    }
    files = {
        connectors.claude_code_path(home): claude,
        cwd / ".mcp.json": mcp_json,
        desktop: desktop_cfg,
    }
    for path, data in files.items():
        path.write_text(json.dumps(data))
    return home, cwd, list(files)


def _status(server: str, tools: dict[str, dict | None]) -> dict:
    """A get_mcp_status() response: tool name -> annotations (None = absent)."""
    entries = []
    for name, notes in tools.items():
        entry = {"name": name}
        if notes is not None:
            entry["annotations"] = notes
        entries.append(entry)
    return {"mcpServers": [{"name": server, "status": "connected", "tools": entries}]}


def _servers(fake_home):
    home, cwd, _ = fake_home
    return discover(home, cwd)


async def _call(conns: Connectors, name: str, args: dict, server_calls: list) -> dict:
    """What the CLI does with one tool call: ask the hook, run only if not denied."""
    out = await conns._pre_tool_use({"tool_name": name, "tool_input": args}, "id", None)
    decision = out.get("hookSpecificOutput", {}).get("permissionDecision")
    if decision != "deny":
        server_calls.append((name, args))
    return out


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


class TestImport:
    def test_every_source_is_listed_by_name(self, fake_home):
        found = _servers(fake_home)
        assert set(found) == {"obsidian", "linear", "notes", "slack"}
        assert found["obsidian"].source == "claude_code"
        assert found["obsidian"].config["command"] == "uvx"  # not the Desktop copy
        assert found["linear"].transport == "http"
        assert found["notes"].source == "mcp.json"
        assert found["slack"].source == "claude_desktop"

    def test_env_vars_are_expanded_like_claude_code(self, fake_home):
        assert _servers(fake_home)["notes"].config["args"] == ["/notas", "def"]

    def test_the_source_files_are_left_exactly_as_they_were(self, fake_home):
        home, cwd, paths = fake_home
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths}
        conns = Connectors.from_config(
            {"obsidian": {"enabled": True}}, home=home, cwd=cwd
        )
        conns.mcp_servers()
        assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths} == before

    def test_a_broken_file_imports_nothing_and_says_so_without_its_content(
        self, tmp_path, caplog
    ):
        home = tmp_path
        connectors.claude_code_path(home).write_text('{"mcpServers": {"x": "' + CANARY)
        with caplog.at_level(logging.DEBUG):
            assert discover(home, tmp_path) == {}
        assert CANARY not in caplog.text

    def test_nothing_is_read_when_no_connector_is_enabled(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("read a config file with nothing enabled")

        monkeypatch.setattr(connectors, "discover", boom)
        assert not Connectors.from_config(None)
        assert not Connectors.from_config({"obsidian": {"enabled": False}})


# --------------------------------------------------------------------------- #
# Enable by name
# --------------------------------------------------------------------------- #


class TestEnable:
    def test_only_servers_named_in_config_are_handed_to_the_sdk(self, fake_home):
        conns = Connectors(_servers(fake_home), {"obsidian": {"enabled": True}})
        assert set(conns.mcp_servers()) == {"obsidian"}

    def test_an_imported_server_is_off_by_default(self, fake_home):
        assert not Connectors(_servers(fake_home), {})

    def test_a_name_no_app_declares_is_skipped(self, fake_home):
        conns = Connectors(_servers(fake_home), {"jira": {"enabled": True}})
        assert not conns

    def test_a_server_named_like_wawi_own_cannot_shadow_it(self, fake_home):
        servers = _servers(fake_home)
        servers["wawi"] = servers["obsidian"]
        conns = Connectors(servers, {"wawi": {"enabled": True}})
        assert "wawi" not in conns.mcp_servers()


# --------------------------------------------------------------------------- #
# Read-only rule
# --------------------------------------------------------------------------- #


class TestReadOnly:
    @pytest.fixture
    def conns(self, fake_home):
        return Connectors(
            _servers(fake_home),
            {"obsidian": {
                "enabled": True,
                "mode": "read_only",
                "read_tools": ["get_note", "search", "no_hint"],
            }},
        )

    def test_allowed_tools_lists_only_the_curated_reads(self, conns):
        assert sorted(conns.allowed_tools()) == [
            "mcp__obsidian__get_note",
            "mcp__obsidian__no_hint",
            "mcp__obsidian__search",
        ]

    def test_nothing_is_a_read_before_the_server_vouches_for_it(self, conns):
        allowed, _ = conns.decide("mcp__obsidian__get_note", {})
        assert allowed is False

    def test_hint_and_list_together_make_a_read(self, conns):
        conns.learn(_status("obsidian", {
            "get_note": {"readOnly": True},
            "search": {"readOnlyHint": True},
            "no_hint": None,
            "delete_note": {"readOnly": True},
            "append_note": {"readOnly": False},
        }))
        decide = lambda tool: conns.decide(f"mcp__obsidian__{tool}", {})[0]  # noqa: E731
        assert decide("get_note") and decide("search")
        assert not decide("no_hint")        # curated, but no annotation
        assert not decide("delete_note")    # annotated, but not curated
        assert not decide("append_note")

    def test_non_reads_are_hidden_from_the_next_connection(self, conns):
        conns.learn(_status("obsidian", {
            "get_note": {"readOnly": True},
            "no_hint": None,
            "delete_note": {"readOnly": True},
        }))
        assert conns.disallowed_tools() == [
            "mcp__obsidian__delete_note",
            "mcp__obsidian__no_hint",
        ]

    def test_without_a_curated_list_a_server_has_no_reads(self, fake_home):
        conns = Connectors(_servers(fake_home), {"slack": {"enabled": True}})
        conns.learn(_status("slack", {"read_channel": {"readOnly": True}}))
        assert conns.allowed_tools() == []
        assert conns.decide("mcp__slack__read_channel", {})[0] is False

    def test_a_denied_call_never_reaches_the_server(self, conns):
        calls: list = []
        out = asyncio.run(_call(conns, "mcp__obsidian__delete_note", {"p": "x"}, calls))
        assert calls == []
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_an_unproven_read_asks_the_live_status_once(self, conns):
        asked = []

        async def status():
            asked.append(1)
            return _status("obsidian", {"get_note": {"readOnly": True}})

        conns.set_status_source(status)
        calls: list = []
        asyncio.run(_call(conns, "mcp__obsidian__get_note", {}, calls))
        asyncio.run(_call(conns, "mcp__obsidian__get_note", {}, calls))
        assert len(calls) == 2 and len(asked) == 1

    def test_a_status_that_fails_denies(self, conns):
        async def status():
            raise RuntimeError("cli gone")

        conns.set_status_source(status)
        calls: list = []
        asyncio.run(_call(conns, "mcp__obsidian__get_note", {}, calls))
        assert calls == []

    def test_other_tools_are_none_of_its_business(self, conns):
        assert conns.decide("mcp__wawi__remember", {}) is None
        assert asyncio.run(conns._pre_tool_use(
            {"tool_name": "mcp__wawi__remember", "tool_input": {}}, None, None
        )) == {}


# --------------------------------------------------------------------------- #
# Read and send: one spoken yes
# --------------------------------------------------------------------------- #


class TestReadSend:
    SEND = "mcp__slack__send_message"
    ARGS = {"channel": "juan", "text": "llego tarde"}

    @pytest.fixture
    def conns(self, fake_home):
        conns = Connectors(
            _servers(fake_home),
            {"slack": {"enabled": True, "mode": "read_send", "read_tools": ["read_channel"]}},
        )
        conns.learn(_status("slack", {
            "read_channel": {"readOnly": True},
            "send_message": {"readOnly": False},
        }))
        return conns

    def test_the_server_is_pre_approved_and_the_gate_decides(self, conns):
        assert conns.allowed_tools() == ["mcp__slack"]

    def test_a_read_does_not_ask(self, conns):
        conns.begin_turn("¿qué hay en el canal?")
        calls: list = []
        asyncio.run(_call(conns, "mcp__slack__read_channel", {"c": "x"}, calls))
        assert len(calls) == 1

    def test_the_first_send_is_refused_with_every_argument_in_the_preview(self, conns):
        conns.begin_turn("mandale a juan que llego tarde")
        calls: list = []
        out = asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        reason = out["hookSpecificOutput"]["permissionDecisionReason"]
        assert calls == []
        assert "channel: juan" in reason and "text: llego tarde" in reason
        assert "slack" in reason and "¿Lo mando?" in reason

    def test_a_spoken_yes_lets_the_same_call_through_once(self, conns):
        calls: list = []
        conns.begin_turn("mandale a juan que llego tarde")
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        conns.begin_turn("sí, mandalo")
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))  # no second freebie
        assert calls == [(self.SEND, self.ARGS)]

    @pytest.mark.parametrize("reply", ["no", "mejor no", "esperá", ""])
    def test_anything_but_a_yes_sends_nothing(self, conns, reply):
        calls: list = []
        conns.begin_turn("mandale a juan que llego tarde")
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        conns.begin_turn(reply)
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        assert calls == []

    def test_a_yes_to_different_arguments_is_not_a_yes(self, conns):
        calls: list = []
        conns.begin_turn("mandale a juan")
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        conns.begin_turn("sí")
        asyncio.run(_call(conns, self.SEND, {**self.ARGS, "channel": "general"}, calls))
        assert calls == []

    def test_a_yes_that_comes_too_late_sends_nothing(self, fake_home):
        conns = Connectors(
            _servers(fake_home),
            {"slack": {"enabled": True, "mode": "read_send"}},
            confirm_timeout_s=0.0,
        )
        calls: list = []
        conns.begin_turn("mandale a juan")
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        conns.begin_turn("sí")
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        assert calls == []

    def test_a_yes_two_turns_later_is_about_something_else(self, conns):
        calls: list = []
        conns.begin_turn("mandale a juan")
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        conns.begin_turn("¿qué hora es?")
        conns.begin_turn("sí")
        asyncio.run(_call(conns, self.SEND, self.ARGS, calls))
        assert calls == []


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


def test_no_secret_reaches_a_log_a_repr_or_a_spoken_message(fake_home, caplog):
    home, cwd, _ = fake_home
    with caplog.at_level(logging.DEBUG):
        conns = Connectors.from_config(
            {
                "slack": {"enabled": True, "mode": "read_send"},
                "linear": {"enabled": True, "mode": "read_only"},
                "obsidian": {"enabled": True},
            },
            home=home,
            cwd=cwd,
        )
        conns.learn(_status("slack", {"send_message": None}))
        conns.begin_turn("mandale")
        outs = [
            asyncio.run(conns._pre_tool_use(
                {"tool_name": name, "tool_input": {"text": "hola"}}, None, None
            ))
            for name in ("mcp__slack__send_message", "mcp__linear__create_issue")
        ]
    found = discover(home, cwd)
    shown = " ".join([
        caplog.text,
        json.dumps(outs),
        *(repr(s) for s in found.values()),
        *(s.describe() for s in found.values()),
    ])
    assert CANARY not in shown
    # The names are what the panel and the log are for.
    assert "SLACK_TOKEN" in found["slack"].describe()
    # ...while the SDK still gets the real value: it is the one place it goes.
    assert conns.mcp_servers()["slack"]["env"]["SLACK_TOKEN"] == CANARY


# --------------------------------------------------------------------------- #
# Brain wiring
# --------------------------------------------------------------------------- #


def _options(conns=None):
    bridge = brain._build_bridge(brain._TurnState(), None)
    return brain._build_options(bridge, model="m", max_iterations=3, connectors=conns)


class TestBrainWiring:
    def test_with_no_connector_the_options_are_unchanged(self, fake_home):
        base = _options()
        for conns in (None, Connectors(_servers(fake_home), {})):
            opts = _options(conns)
            assert list(opts.mcp_servers) == ["wawi"] == list(base.mcp_servers)
            assert opts.allowed_tools == base.allowed_tools
            assert not opts.hooks and not opts.disallowed_tools
            assert opts.strict_mcp_config is True

    def test_enabled_connectors_add_servers_reads_and_the_gate(self, fake_home):
        conns = Connectors(
            _servers(fake_home),
            {"obsidian": {"enabled": True, "read_tools": ["get_note"]}},
        )
        opts = _options(conns)
        assert set(opts.mcp_servers) == {"wawi", "obsidian"}
        assert opts.allowed_tools == [*_options().allowed_tools, "mcp__obsidian__get_note"]
        assert opts.hooks["PreToolUse"][0].hooks == [conns._pre_tool_use]
        assert opts.strict_mcp_config is True

    def test_the_engine_proves_reads_at_connect_and_forgets_the_client_on_close(
        self, fake_home, monkeypatch
    ):
        import claude_agent_sdk

        class Client:
            def __init__(self, options=None):
                self.options = options

            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def get_mcp_status(self):
                return _status("obsidian", {"get_note": {"readOnly": True}})

            async def query(self, prompt, session_id="default"):  # noqa: ARG002
                pass

            async def receive_response(self):
                from claude_agent_sdk import ResultMessage

                yield ResultMessage(
                    subtype="success", duration_ms=1, duration_api_ms=1,
                    is_error=False, num_turns=1, session_id="s",
                )

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
        conns = Connectors(
            _servers(fake_home),
            {"obsidian": {"enabled": True, "read_tools": ["get_note"]}},
        )
        engine = brain.Engine()
        try:
            engine.ask("hola", model="m", system="S", max_iterations=3,
                       screen_options=None, bypass_permissions=False,
                       connectors=conns)
            assert conns.is_read("obsidian", "get_note")
            assert conns._status_source is not None
        finally:
            engine.close()
        assert conns._status_source is None

    def test_ask_sync_hands_the_transcript_to_the_gate(self, fake_home, monkeypatch):
        seen = []
        conns = Connectors(_servers(fake_home), {"slack": {"enabled": True}})
        monkeypatch.setattr(conns, "begin_turn", seen.append)

        async def fake_ask(*a, **k):
            return brain.BrainResult(text="ok")

        monkeypatch.setattr(brain, "ask", fake_ask)
        brain.ask_sync("sí", model="m", connectors=conns)
        assert seen == ["sí"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_absent_means_none(self):
        assert _parse_connectors(None) == {}

    def test_mode_defaults_to_read_only(self):
        parsed = _parse_connectors({"linear": {"enabled": True}})
        assert parsed == {"linear": {"enabled": True, "mode": "read_only", "read_tools": None}}

    def test_a_typo_in_the_mode_is_an_error_not_a_send(self):
        with pytest.raises(ConfigError):
            _parse_connectors({"linear": {"enabled": True, "mode": "read-send"}})
