"""Agregados (wawi/plugins.py, wawi/plugins_ui.py): the owner's own services.

Hermetic: the registry lives in tmp_path, the Keychain is a dict behind the
``_security`` seam, and every HTTP call goes to a fake ``_urlopen``. The
token below is a canary: it must never show up in the JSON, a log line, an
argv, the CLI output or the page state.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import subprocess
import threading
import time
import urllib.error

import pytest

from wawi import brain, connectors, plugins, plugins_ui, tools
from wawi.connectors import Connectors, looks_like_read, plugin_tool_is_read
from wawi.reactions import ReactionBus

CANARY = "tok-canary-5b2e91"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


class FakeKeychain:
    """Stands in for /usr/bin/security; records every argv it was given."""

    def __init__(self):
        self.items: dict[tuple[str, str], str] = {}
        self.argv: list[list[str]] = []

    def __call__(self, args, stdin=None):
        self.argv.append(list(args))
        if args == ["-i"]:
            # add-generic-password -U -s "svc" -a "acct" -w "value"
            import shlex

            parts = shlex.split(stdin)
            svc = parts[parts.index("-s") + 1]
            acct = parts[parts.index("-a") + 1]
            self.items[(svc, acct)] = parts[parts.index("-w") + 1]
            return subprocess.CompletedProcess(args, 0, "", "")
        svc = args[args.index("-s") + 1]
        acct = args[args.index("-a") + 1]
        if args[0] == "find-generic-password":
            value = self.items.get((svc, acct))
            return subprocess.CompletedProcess(args, 0 if value else 44, (value or "") + "\n", "")
        if args[0] == "delete-generic-password":
            self.items.pop((svc, acct), None)
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


class FakeHTTP:
    """A fake urlopen: records requests, answers from a queue."""

    def __init__(self, answers=None):
        self.requests = []
        self.answers = list(answers or [])

    def __call__(self, req, timeout=None):
        self.requests.append((req, timeout))
        answer = self.answers.pop(0) if self.answers else (200, b'{"ok": true}')
        if isinstance(answer, BaseException):
            raise answer
        status, body = answer
        if status >= 400:
            raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(body))
        return _Resp(status, body)


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body
        self.headers = {"Content-Type": "application/json"}

    def read(self, n=-1):
        return self._body[:n] if n and n > 0 else self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def keychain(monkeypatch):
    fake = FakeKeychain()
    monkeypatch.setattr(plugins, "_security", fake)
    return fake


@pytest.fixture
def http(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(plugins, "_urlopen", fake)
    return fake


@pytest.fixture
def store(tmp_path):
    return plugins.Store(tmp_path / "plugins_local.json")


@pytest.fixture
def manager(store, monkeypatch):
    m = plugins.Manager(store)
    monkeypatch.setattr(plugins, "_MANAGER", m)
    return m


def rest_plugin(**over):
    raw = {
        "id": "tienda", "name": "Mi Tienda", "kind": "rest",
        "description": "pedidos y stock", "base_url": "https://api.tienda.test/v1/",
        "endpoints": [
            {"name": "pedidos", "path": "/orders", "query": "estado", "description": "pedidos"},
            {"name": "pedido", "path": "/orders/{id}"},
            {"name": "crear", "method": "POST", "path": "/orders", "description": "crea uno"},
        ],
    }
    raw.update(over)
    return raw


def mcp_plugin(**over):
    raw = {"id": "notas", "name": "Notas", "kind": "mcp", "description": "mis notas",
           "mcp": {"command": "npx", "args": ["-y", "notas-mcp"]}}
    raw.update(over)
    return raw


def run_cli(store, *argv, token=CANARY):
    out = io.StringIO()
    code = plugins.main(["--file", str(store.path), *argv], read_token=lambda prompt: token, out=out)
    return code, out.getvalue()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_add_persists_and_reloads(self, store):
        store.add(rest_plugin())
        again = plugins.Store(store.path)
        [plugin] = again.plugins()
        assert plugin["id"] == "tienda"
        assert plugin["base_url"] == "https://api.tienda.test/v1"
        assert plugin["endpoints"][0]["query"] == ["estado"]
        assert plugin["enabled"] is True and plugin["allow_writes"] is False

    def test_file_is_private(self, store):
        store.add(rest_plugin())
        assert store.path.stat().st_mode & 0o077 == 0

    def test_refresh_notices_external_edits(self, store):
        store.add(rest_plugin())
        store.refresh()
        generation = store.generation
        assert store.refresh() is False
        data = json.loads(store.path.read_text())
        data["plugins"][0]["enabled"] = False
        time.sleep(0.01)
        store.path.write_text(json.dumps(data) + " ")
        assert store.refresh() is True
        assert store.generation > generation
        assert store.plugins()[0]["enabled"] is False

    def test_bad_entries_are_skipped_not_fatal(self, store, caplog):
        store.path.write_text(json.dumps({"plugins": [
            {"id": "BAD ID", "kind": "rest"}, rest_plugin(), {"kind": "ftp"}]}))
        assert [p["id"] for p in store.plugins()] == ["tienda"]

    def test_unreadable_file_is_empty(self, store):
        store.path.write_text("{not json")
        assert store.plugins() == []

    def test_duplicate_id_refused(self, store):
        store.add(rest_plugin())
        with pytest.raises(plugins.PluginError):
            store.add(rest_plugin())

    def test_enable_disable_remove(self, store, keychain):
        store.add(rest_plugin())
        assert store.set_enabled("tienda", False)["enabled"] is False
        assert store.remove("tienda") is True
        assert store.plugins() == []
        assert store.remove("tienda") is False

    def test_slug(self):
        assert plugins.slug("Mi API (prod)") == "mi_api_prod"
        assert plugins.slug("Ñandú café") == "nandu_cafe"

    def test_validation_messages_are_spanish(self):
        with pytest.raises(plugins.PluginError, match="URL"):
            plugins.normalize(rest_plugin(base_url="ftp://x"))
        with pytest.raises(plugins.PluginError, match="usuario"):
            plugins.normalize(rest_plugin(base_url="https://user:pw@x.test"))
        with pytest.raises(plugins.PluginError, match="relativo"):
            plugins.normalize(rest_plugin(endpoints=[{"name": "x", "path": "https://evil/x"}]))
        with pytest.raises(plugins.PluginError, match="token"):
            plugins.normalize(rest_plugin(auth={"type": "bearer"}))

    def test_health_endpoint_must_be_a_plain_get(self):
        with pytest.raises(plugins.PluginError):
            plugins.normalize(rest_plugin(health={"endpoint": "pedido"}))
        ok = plugins.normalize(rest_plugin(health={"endpoint": "pedidos", "interval": 5}))
        assert ok["health"] == {"endpoint": "pedidos", "expect": 200, "interval": 30}


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


class TestSecrets:
    def test_plain_credentials_are_refused(self):
        with pytest.raises(plugins.PluginError, match="credencial"):
            plugins.normalize(rest_plugin(headers={"Authorization": f"Bearer {CANARY}"}))
        with pytest.raises(plugins.PluginError, match="credencial"):
            plugins.normalize(mcp_plugin(mcp={"command": "x", "env": {"API_KEY": CANARY}}))

    def test_refs_must_be_names(self):
        with pytest.raises(plugins.PluginError):
            plugins.normalize(rest_plugin(auth={"type": "bearer", "secret": {"value": CANARY}}))
        with pytest.raises(plugins.PluginError):
            plugins.normalize(rest_plugin(auth={"type": "bearer", "secret": {"env": "has space"}}))

    def test_keychain_value_never_in_argv(self, keychain):
        plugins.keychain_set("tienda", "token", CANARY)
        assert keychain.items[("wawi-plugin-tienda", "token")] == CANARY
        assert all(CANARY not in " ".join(argv) for argv in keychain.argv)
        assert plugins.keychain_get("tienda", "token") == CANARY

    def test_keychain_quotes_survive(self, keychain):
        tricky = 'a b"c\\d'
        plugins.keychain_set("tienda", "token", tricky)
        assert keychain.items[("wawi-plugin-tienda", "token")] == tricky

    def test_newlines_refused(self, keychain):
        with pytest.raises(plugins.PluginError):
            plugins.keychain_set("tienda", "token", "a\nb")

    def test_env_reference(self, monkeypatch):
        monkeypatch.setenv("TIENDA_TOKEN", CANARY)
        assert plugins.resolve_secret("tienda", {"env": "TIENDA_TOKEN"}) == CANARY
        assert plugins.resolve_secret("tienda", {"env": "NOPE_NOT_SET"}) is None

    def test_cli_add_with_token_keeps_it_out_of_everything(self, store, keychain, http, caplog):
        caplog.set_level(logging.DEBUG)
        code, out = run_cli(store, "add", "Mi Tienda", "--rest", "https://api.tienda.test",
                            "--auth", "bearer", "--token")
        assert code == 0
        assert CANARY not in store.path.read_text()
        assert CANARY not in out
        assert CANARY not in caplog.text
        assert keychain.items[("wawi-plugin-mi_tienda", "token")] == CANARY
        assert json.loads(store.path.read_text())["plugins"][0]["auth"]["secret"] == {"keychain": "token"}
        code, out = run_cli(store, "show", "mi_tienda")
        assert "Llavero (token)" in out and CANARY not in out
        # The request carries it; the log line about it does not.
        store.update("mi_tienda", lambda p: p["endpoints"].append({"name": "ping", "path": "/ping"}))
        ok, _, _ = plugins.call_rest(store.get("mi_tienda"), {"endpoint": "ping"})
        assert ok
        req, _ = http.requests[-1]
        assert req.get_header("Authorization") == f"Bearer {CANARY}"
        assert CANARY not in caplog.text
        assert CANARY not in repr(plugins.build_request(store.get("mi_tienda"), {"endpoint": "ping"}))

    def test_missing_token_is_a_short_error(self, store, keychain, http):
        store.add(rest_plugin(auth={"type": "bearer", "secret": {"keychain": "token"}}))
        ok, speak, data = plugins.call_rest(store.get("tienda"), {"endpoint": "pedidos"})
        assert not ok and "token" in speak and not http.requests

    def test_remove_forgets_keychain_items(self, store, keychain):
        run_cli(store, "add", "Mi Tienda", "--rest", "https://api.tienda.test", "--token")
        assert keychain.items
        run_cli(store, "remove", "mi_tienda")
        assert not keychain.items

    def test_mcp_secret_env_resolved_in_memory_only(self, store, keychain, manager, caplog):
        caplog.set_level(logging.DEBUG)
        code, _ = run_cli(store, "add", "Notas", "--mcp-cmd", "npx -y notas-mcp", "--secret-env", "NOTAS_KEY")
        assert code == 0 and CANARY not in store.path.read_text()
        [server] = manager.mcp_servers()
        assert server.config["env"]["NOTAS_KEY"] == CANARY
        assert CANARY not in repr(server)
        conns = Connectors({}, {})
        conns.set_plugins([server])
        assert CANARY not in repr(conns.signature())
        assert CANARY not in caplog.text


# --------------------------------------------------------------------------- #
# REST tools
# --------------------------------------------------------------------------- #


class TestRestTools:
    def test_spec_is_get_only_by_default(self):
        spec = plugins.rest_spec(plugins.normalize(rest_plugin()))
        assert spec["name"] == "svc_tienda"
        assert spec["input_schema"]["properties"]["endpoint"]["enum"] == ["pedidos", "pedido"]
        assert "crear" not in spec["description"]
        assert "pedido(id)" in spec["description"] and "pedidos(estado?)" in spec["description"]
        model = tools.for_model(spec)
        assert set(model) == {"name", "description", "input_schema"}

    def test_spec_with_writes_names_them(self):
        spec = plugins.rest_spec(plugins.normalize(rest_plugin(allow_writes=True)))
        assert "crear" in spec["input_schema"]["properties"]["endpoint"]["enum"]
        assert "confirmación" in spec["description"]

    def test_spec_without_endpoints_takes_a_path(self):
        spec = plugins.rest_spec(plugins.normalize(rest_plugin(endpoints=[])))
        assert spec["input_schema"]["required"] == ["path"]

    def test_description_is_bounded(self):
        many = [{"name": f"e{i}", "path": f"/x/{i}", "description": "d" * 150} for i in range(20)]
        spec = plugins.rest_spec(plugins.normalize(rest_plugin(endpoints=many)))
        assert len(spec["description"]) <= plugins.MAX_TOOL_DESCRIPTION

    def test_build_request(self):
        plugin = plugins.normalize(rest_plugin())
        req = plugins.build_request(plugin, {"endpoint": "pedido", "params": {"id": "a/b", "x": 1}})
        assert req.url == "https://api.tienda.test/v1/orders/a%2Fb?x=1"
        assert req.method == "GET" and req.body is None
        req = plugins.build_request(plugin, {"endpoint": "pedidos", "params": {"estado": "abierto"}})
        assert req.url.endswith("/orders?estado=abierto")

    def test_free_path_cannot_escape(self):
        plugin = plugins.normalize(rest_plugin(endpoints=[]))
        for bad in ("https://evil.test/", "../../etc", "a\\b"):
            with pytest.raises(plugins.PluginError):
                plugins.build_request(plugin, {"path": bad})
        assert plugins.build_request(plugin, {"path": "status"}).url == "https://api.tienda.test/v1/status"

    def test_truncation_and_shape(self):
        big = {"items": [{"n": i, "text": "x" * 1000} for i in range(100)]}
        text, cut = plugins.render_body(json.dumps(big).encode())
        assert len(text) <= plugins.MAX_RESULT_CHARS
        assert "… 80 más" in text or cut
        small, cut = plugins.render_body(b'{"a": 1}')
        assert small == '{\n "a": 1\n}' and not cut
        html, _ = plugins.render_body(b"<html><script>x()</script><p>Hola   mundo</p></html>")
        assert html == "Hola mundo"
        long, cut = plugins.render_body(b"y" * 20_000)
        assert cut and len(long) == plugins.MAX_RESULT_CHARS

    def test_errors_are_short_lines(self, http):
        plugin = plugins.normalize(rest_plugin())
        http.answers = [urllib.error.URLError(TimeoutError("timed out")), (503, b"busy " * 1000)]
        ok, speak, data = plugins.call_rest(plugin, {"endpoint": "pedidos"})
        assert not ok and speak == "No pude conectar con Mi Tienda." and "Traceback" not in data["detail"]
        ok, speak, data = plugins.call_rest(plugin, {"endpoint": "pedidos"})
        assert not ok and "503" in speak and len(data["body"]) <= 600
        assert http.requests[0][1] == plugins.TIMEOUT_S

    def test_unknown_endpoint(self, http):
        ok, speak, _ = plugins.call_rest(plugins.normalize(rest_plugin()), {"endpoint": "nada"})
        assert not ok and "nada" in speak and not http.requests


class TestWrites:
    def test_non_get_refused_without_allow_writes(self, http, monkeypatch):
        monkeypatch.setattr(plugins, "_confirm", lambda *a: pytest.fail("must not ask"))
        ok, speak, _ = plugins.call_rest(plugins.normalize(rest_plugin()), {"endpoint": "crear"})
        assert not ok and "solo lectura" in speak and not http.requests

    def test_declined_on_screen_sends_nothing(self, http, monkeypatch):
        asked = []
        monkeypatch.setattr(plugins, "_confirm", lambda t, d: asked.append((t, d)) or False)
        plugin = plugins.normalize(rest_plugin(allow_writes=True))
        ok, speak, data = plugins.call_rest(plugin, {"endpoint": "crear", "params": {"item": "pan"}})
        assert not ok and data["error"] == "not_confirmed" and not http.requests
        assert asked and "POST" in asked[0][0] and "item=pan" in asked[0][1]

    def test_accepted_on_screen_sends_json_body(self, http, monkeypatch):
        monkeypatch.setattr(plugins, "_confirm", lambda t, d: True)
        plugin = plugins.normalize(rest_plugin(allow_writes=True))
        ok, _, _ = plugins.call_rest(plugin, {"endpoint": "crear", "params": {"item": "pan"}})
        req, _ = http.requests[0]
        assert ok and req.get_method() == "POST" and json.loads(req.data) == {"item": "pan"}

    def test_get_never_asks(self, http, monkeypatch):
        monkeypatch.setattr(plugins, "_confirm", lambda *a: pytest.fail("must not ask"))
        assert plugins.call_rest(plugins.normalize(rest_plugin(allow_writes=True)), {"endpoint": "pedidos"})[0]

    def test_confirm_on_screen_uses_the_card(self, monkeypatch):
        from wawi import confirm

        registry = confirm.Registry(presenter=object())
        monkeypatch.setattr(confirm, "REGISTRY", registry)

        def click():
            for _ in range(200):
                if registry.pending():
                    registry.resolve(registry.pending()[0].id, True)
                    return
                time.sleep(0.01)

        threading.Thread(target=click, daemon=True).start()
        assert plugins.confirm_on_screen("Mi Tienda: POST", "POST /orders", timeout=3) is True
        threading.Thread(target=lambda: (time.sleep(0.05), registry.resolve(registry.pending()[0].id, False)),
                         daemon=True).start()
        assert plugins.confirm_on_screen("Mi Tienda: POST", "POST /orders", timeout=3) is False

    def test_no_presenter_means_no(self, monkeypatch):
        from wawi import confirm

        monkeypatch.setattr(confirm, "REGISTRY", confirm.Registry())
        assert plugins.confirm_on_screen("t", "d", timeout=0.1) is False


# --------------------------------------------------------------------------- #
# MCP plugins through the connectors' gate
# --------------------------------------------------------------------------- #


def _server(allow_writes=False, prefix="plug_notas"):
    return plugins.PluginServer(prefix, "Notas", "stdio", {"type": "stdio", "command": "npx"},
                                allow_writes)


STATUS = {"mcpServers": [{"name": "plug_notas", "tools": [
    {"name": "mcp__plug_notas__list_notes", "annotations": {}},
    {"name": "mcp__plug_notas__get_note", "annotations": {"readOnlyHint": True}},
    {"name": "mcp__plug_notas__delete_note", "annotations": {}},
    {"name": "mcp__plug_notas__get_or_create", "annotations": {}},
    {"name": "mcp__plug_notas__notes", "annotations": {"readOnlyHint": True}},
    {"name": "mcp__plug_notas__search_and_wipe", "annotations": {"destructiveHint": True}},
]}]}


class TestMcpReadFilter:
    @pytest.mark.parametrize("name,read", [
        ("list_issues", True), ("getStatus", True), ("search", True), ("get_or_create_issue", False),
        ("issues", False), ("create_issue", False), ("send_message", False), ("fetch-page", True),
        ("deleteThing", False), ("", False),
    ])
    def test_looks_like_read(self, name, read):
        assert looks_like_read(name) is read

    def test_annotations_win(self):
        assert plugin_tool_is_read("notes", {"readOnlyHint": True})
        assert not plugin_tool_is_read("get_x", {"readOnlyHint": False})
        assert not plugin_tool_is_read("list_x", {"destructiveHint": True})

    def test_read_only_plugin(self):
        conns = Connectors({}, {})
        assert not conns
        conns.set_plugins([_server()])
        assert conns and conns.allowed_tools() == ["mcp__plug_notas"]
        assert conns.mcp_servers() == {"plug_notas": {"type": "stdio", "command": "npx"}}
        conns.learn(STATUS)
        assert conns.decide("mcp__plug_notas__list_notes", {}) == (True, "")
        assert conns.decide("mcp__plug_notas__notes", {}) == (True, "")
        for write in ("delete_note", "get_or_create", "search_and_wipe"):
            allowed, reason = conns.decide(f"mcp__plug_notas__{write}", {})
            assert not allowed and "read-only" in reason
        assert conns.disallowed_tools() == [
            "mcp__plug_notas__delete_note", "mcp__plug_notas__get_or_create",
            "mcp__plug_notas__search_and_wipe"]

    def test_borrowed_connectors_survive_set_plugins(self):
        imported = {"obsidian": connectors.ImportedServer("obsidian", "claude_code", "stdio", {"type": "stdio"})}
        conns = Connectors(imported, {"obsidian": {"enabled": True}})
        conns.set_plugins([_server()])
        conns.set_plugins([])
        assert list(conns.mcp_servers()) == ["obsidian"]

    def test_signature_changes_with_plugins(self):
        conns = Connectors({}, {})
        empty = conns.signature()
        conns.set_plugins([_server()])
        one = conns.signature()
        conns.set_plugins([_server(allow_writes=True)])
        assert len({empty, one, conns.signature()}) == 3

    def _hook(self, conns, tool):
        return asyncio.run(conns._pre_tool_use(
            {"tool_name": f"mcp__plug_notas__{tool}", "tool_input": {"id": "7"}}, None, None))

    def test_writes_wait_for_the_card(self, monkeypatch):
        asked = []
        monkeypatch.setattr(plugins, "_confirm", lambda t, d: asked.append((t, d)) or True)
        monkeypatch.setattr(connectors, "_show_working", lambda *a: None)
        conns = Connectors({}, {})
        conns.set_plugins([_server(allow_writes=True)])
        conns.learn(STATUS)
        assert conns.disallowed_tools() == []
        assert self._hook(conns, "delete_note") == {}
        assert asked and "delete note" in asked[0][0] and "id: 7" in asked[0][1]
        monkeypatch.setattr(plugins, "_confirm", lambda t, d: False)
        denied = self._hook(conns, "delete_note")
        assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
        # Reads never ask.
        monkeypatch.setattr(plugins, "_confirm", lambda *a: pytest.fail("must not ask"))
        assert self._hook(conns, "list_notes") == {}

    def test_read_only_write_is_denied_without_asking(self, monkeypatch):
        monkeypatch.setattr(plugins, "_confirm", lambda *a: pytest.fail("must not ask"))
        conns = Connectors({}, {})
        conns.set_plugins([_server()])
        conns.learn(STATUS)
        assert self._hook(conns, "delete_note")["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_manager_sync_feeds_the_gate(self, store, manager, keychain):
        store.add(mcp_plugin())
        conns = Connectors({}, {})
        plugins.sync(conns)
        assert "plug_notas" in conns.mcp_servers()
        store.set_enabled("notas", False)
        plugins.sync(conns)
        assert not conns

    def test_options_carry_plugin_servers(self, store, manager):
        store.add(mcp_plugin())
        conns = Connectors({}, {})
        plugins.sync(conns)
        state = brain._TurnState()
        options = brain._build_options(brain._build_bridge(state, None), model="m",
                                       max_iterations=3, connectors=conns)
        assert "plug_notas" in options.mcp_servers
        assert "mcp__plug_notas" in options.allowed_tools


# --------------------------------------------------------------------------- #
# Tools, abilities, brain signature
# --------------------------------------------------------------------------- #


class TestToolsAndAbilities:
    def test_nothing_without_plugins(self, manager):
        names = [s["name"] for s in tools.active_tools()]
        assert "my_services" not in names and not any(n.startswith("svc_") for n in names)

    def test_rest_plugin_becomes_one_tool(self, store, manager):
        store.add(rest_plugin())
        store.add(mcp_plugin())
        active = {s["name"]: s for s in tools.active_tools()}
        assert "svc_tienda" in active and "my_services" in active
        assert "ability" not in active["svc_tienda"] and "example" not in active["svc_tienda"]
        assert tools.animation_for("svc_tienda") == "binoculars"

    def test_disabled_rest_plugin_has_no_tool(self, store, manager):
        store.add(rest_plugin(enabled=False))
        names = [s["name"] for s in tools.active_tools()]
        assert "svc_tienda" not in names and "my_services" in names

    def test_dispatch(self, store, manager, http):
        store.add(rest_plugin())
        result = tools.dispatch("svc_tienda", {"endpoint": "pedidos"})
        assert result.ok and result.data["status"] == 200
        services = tools.dispatch("my_services", {})
        assert services.ok and services.speak == "Tengo Mi Tienda."
        assert services.data["services"][0]["endpoints"] == ["pedidos", "pedido"]

    def test_abilities_list_services(self, store, manager, monkeypatch):
        store.add(rest_plugin())
        store.add(mcp_plugin())
        monkeypatch.setattr(tools, "_CONNECTORS", None)
        groups = dict(tools.ability_groups())
        rows = groups["Tus servicios (agregados)"]
        assert ("svc_tienda", "Consultar Mi Tienda", "Wawi, ¿cómo está Mi Tienda?") in rows
        assert ("plug_notas", "Consultar Notas", "Wawi, ¿cómo está Notas?") in rows
        assert any(r[0] == "my_services" and r[2] == "Wawi, ¿qué servicios tenés?" for r in rows)

    def test_mcp_plugins_not_listed_twice_as_connectors(self, store, manager, monkeypatch):
        store.add(mcp_plugin())
        conns = Connectors({}, {})
        plugins.sync(conns)
        monkeypatch.setattr(tools, "_CONNECTORS", conns)
        headings = [h for h, _ in tools.ability_groups()]
        assert not any(h.startswith("Conector") for h in headings)

    def test_brain_signature_follows_the_registry(self, store, manager):
        before = brain._extras_signature(None)
        store.add(rest_plugin())
        after = brain._extras_signature(None)
        assert before != after
        assert brain._extras_signature(None) == after  # stable: no reconnect per turn
        store.set_enabled("tienda", False)
        assert brain._extras_signature(None) != after

    def test_spec_text_is_stable(self, store, manager):
        store.add(rest_plugin())
        assert manager.tool_specs() == manager.tool_specs()


# --------------------------------------------------------------------------- #
# Health -> reactions
# --------------------------------------------------------------------------- #


class TestHealth:
    def _sensor(self, manager, results, clock=None, cancelled=None):
        answers = list(results)
        return plugins.PluginHealthSensor(
            manager, check=lambda p: answers.pop(0),
            clock=clock or (lambda: 0.0),
            cancel=(cancelled.append if cancelled is not None else lambda k: None))

    def test_down_then_up_emits_web_events(self, store, manager):
        store.add(rest_plugin(health={"url": "https://api.tienda.test/health", "interval": 30}))
        now = [0.0]
        url = "https://api.tienda.test/health"
        sensor = self._sensor(manager, [(False, "x", url), (False, "x", url), (True, "ok", url)],
                              clock=lambda: now[0])
        assert sensor.poll() == []
        assert sensor.poll() == []  # not due yet
        now[0] = 31
        [down] = sensor.poll()
        assert down.kind == "web.down" and down.key == f"web:{url}"
        assert down.data["host"] == "Mi Tienda" and manager.health["tienda"] == "down"
        assert sensor.summary() == "está caído Mi Tienda"
        now[0] = 62
        [up] = sensor.poll()
        assert up.kind == "web.up" and up.key == down.key
        assert sensor.summary() == "tus agregados andan"

    def test_down_reaches_the_firefighter(self, store, manager):
        from wawi.reflexes import QuietState, Reflexes

        class Quiet:
            def check(self):
                return QuietState()

        bus = ReactionBus()
        rx = Reflexes({}, bus=bus, quiet=Quiet(), triage=lambda e: None)
        rx.attach()
        store.add(rest_plugin(health={"url": "https://api.tienda.test/health"}))
        url = "https://api.tienda.test/health"
        sensor = self._sensor(manager, [(False, "x", url)] * 2)
        sensor._due = {}
        events = sensor.poll()
        sensor._due = {}
        events += sensor.poll()
        for event in events:
            bus.emit(event)
        reaction = bus.pop()
        assert reaction is not None and reaction.costume == "firefighter"
        assert "Mi Tienda" in reaction.bubble

    def test_removed_while_down_cancels(self, store, manager):
        store.add(rest_plugin(health={"url": "https://api.tienda.test/health"}))
        cancelled = []
        url = "https://api.tienda.test/health"
        sensor = self._sensor(manager, [(False, "x", url)] * 2, cancelled=cancelled)
        sensor.poll()
        sensor._due = {}
        sensor.poll()
        store.remove("tienda", forget_secrets=False)
        sensor.poll()
        assert cancelled == [f"web:{url}"]
        assert "tienda" not in manager.health

    def test_check_health_with_fake_http(self, http):
        plugin = plugins.normalize(rest_plugin(health={"endpoint": "pedidos"}))
        assert plugins.check_health(plugin)[0] is True
        http.answers = [(500, b"")]
        ok, detail, url = plugins.check_health(plugin)
        assert not ok and "500" in detail and url == "https://api.tienda.test/v1/orders"

    def test_health_url_elsewhere_gets_no_token(self, http, keychain):
        plugins.keychain_set("tienda", "token", CANARY)
        plugin = plugins.normalize(rest_plugin(
            auth={"type": "bearer", "secret": {"keychain": "token"}},
            health={"url": "https://status.other.test/ping"}))
        plugins.check_health(plugin)
        req, _ = http.requests[0]
        assert req.get_header("Authorization") is None

    def test_attach_health_needs_running_sensors(self, manager, monkeypatch):
        from wawi import sensors

        monkeypatch.setattr(sensors, "_RUNNER", None)
        assert plugins.attach_health() is False

        class Runner:
            def __init__(self):
                self.got = []

            def replace(self, sensor):
                self.got.append(sensor)

        runner = Runner()
        monkeypatch.setattr(sensors, "_RUNNER", runner)
        assert plugins.attach_health() is True
        assert isinstance(runner.got[0], plugins.PluginHealthSensor)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestCli:
    def test_full_cycle(self, store, keychain, http):
        assert run_cli(store, "list")[1].startswith("No hay agregados")
        code, out = run_cli(store, "add", "Mi Tienda", "--rest", "https://api.tienda.test",
                            "--desc", "pedidos", "--header", "Accept-Language=es")
        assert code == 0 and "mi_tienda" in out and "reiniciar" in out
        code, _ = run_cli(store, "endpoint", "add", "mi_tienda", "--name", "pedidos",
                          "--path", "/orders", "--desc", "los pedidos", "--query", "estado,limite", "--health")
        assert code == 0
        plugin = store.get("mi_tienda")
        assert plugin["endpoints"][0]["query"] == ["estado", "limite"]
        assert plugin["health"]["endpoint"] == "pedidos"
        code, out = run_cli(store, "show", "mi_tienda")
        assert "endpoint GET pedidos /orders" in out and "salud: pedidos" in out
        code, out = run_cli(store, "test", "mi_tienda")
        assert code == 0 and out.startswith("OK")
        assert run_cli(store, "disable", "mi_tienda")[1].startswith("[off]")
        assert run_cli(store, "enable", "mi_tienda")[1].startswith("[on ]")
        code, _ = run_cli(store, "endpoint", "remove", "mi_tienda", "pedidos")
        assert code == 0 and "health" not in store.get("mi_tienda")
        assert run_cli(store, "remove", "mi_tienda")[0] == 0
        assert store.plugins() == []

    def test_test_without_health_uses_first_plain_get(self, store, http):
        store.add(rest_plugin())
        http.answers = [(200, b'{"orders": []}')]
        code, out = run_cli(store, "test", "tienda")
        assert code == 0 and "HTTP 200" in out
        assert http.requests[0][0].full_url == "https://api.tienda.test/v1/orders"

    def test_test_failure_exit_code(self, store, http):
        store.add(rest_plugin())
        http.answers = [urllib.error.URLError("refused")]
        code, out = run_cli(store, "test", "tienda")
        assert code == 1 and out.startswith("MAL")

    def test_mcp_add_variants(self, store):
        assert run_cli(store, "add", "Remoto", "--mcp-url", "https://mcp.test/mcp", "--sse")[0] == 0
        assert store.get("remoto")["mcp"]["transport"] == "sse"
        assert run_cli(store, "add", "Local", "--mcp-cmd", "uvx local-mcp --ro", "--env", "MODE=ro")[0] == 0
        local = store.get("local")["mcp"]
        assert local["command"] == "uvx" and local["args"] == ["local-mcp", "--ro"] and local["env"] == {"MODE": "ro"}

    def test_token_env_reference(self, store):
        code, _ = run_cli(store, "add", "Api", "--rest", "https://a.test", "--token-env", "API_TOKEN")
        assert code == 0
        assert store.get("api")["auth"] == {"type": "bearer", "secret": {"env": "API_TOKEN"}}

    def test_errors_go_to_stderr_with_exit_1(self, store, capsys):
        assert run_cli(store, "add", "X")[0] == 1
        assert run_cli(store, "remove", "nada")[0] == 1
        assert "error:" in capsys.readouterr().err

    def test_set_token(self, store, keychain):
        run_cli(store, "add", "Api", "--rest", "https://a.test", "--token")
        run_cli(store, "set-token", "api", token="nuevo-valor")
        assert keychain.items[("wawi-plugin-api", "token")] == "nuevo-valor"

    def test_mcp_test_lists_tools(self, store, monkeypatch):
        store.add(mcp_plugin())
        monkeypatch.setattr(plugins, "list_mcp_tools", lambda p: (True, "2 herramienta(s)."))
        code, out = run_cli(store, "test", "notas")
        assert code == 0 and "2 herramienta" in out


# --------------------------------------------------------------------------- #
# The window's controller
# --------------------------------------------------------------------------- #


class TestWindowController:
    def test_add_with_token(self, manager, keychain):
        ctl = plugins_ui.Controller(manager)
        state = ctl.handle(json.dumps({"action": "add", "name": "Mi Tienda", "kind": "rest",
                                       "target": "https://api.tienda.test", "auth": "bearer",
                                       "token": CANARY, "description": "pedidos"}))
        assert not state["error"], state["error"]
        assert CANARY not in json.dumps(state)
        assert CANARY not in manager.store.path.read_text()
        assert keychain.items[("wawi-plugin-mi_tienda", "token")] == CANARY
        [view] = state["plugins"]
        assert view["auth"] == "bearer · Llavero (token)" and view["kind"] == "rest"

    def test_add_mcp_command_with_token_needs_a_variable(self, manager, keychain):
        ctl = plugins_ui.Controller(manager)
        form = {"action": "add", "name": "Notas", "kind": "mcp", "target": "npx -y notas-mcp", "token": CANARY}
        assert "variable" in ctl.handle(form)["error"]
        state = ctl.handle({**form, "variable": "NOTAS_KEY"})
        assert not state["error"]
        assert manager.store.get("notas")["mcp"]["secret_env"] == {"NOTAS_KEY": {"keychain": "NOTAS_KEY"}}

    def test_toggle_test_remove(self, manager, http, keychain):
        manager.store.add(rest_plugin())
        ctl = plugins_ui.Controller(manager)
        state = ctl.handle({"action": "toggle", "id": "tienda", "enabled": False})
        assert state["plugins"][0]["enabled"] is False and "apagado" in state["message"]
        state = ctl.handle({"action": "test", "id": "tienda"})
        assert state["test"]["ok"] is True
        state = ctl.handle({"action": "remove", "id": "tienda"})
        assert state["plugins"] == []

    def test_bad_input_is_an_error_not_a_crash(self, manager):
        ctl = plugins_ui.Controller(manager)
        assert ctl.handle("{nope")["error"]
        assert ctl.handle({"action": "add", "name": "", "kind": "rest"})["error"]
        assert ctl.handle({"action": "add", "name": "X", "kind": "rest", "target": "nada"})["error"]
        assert ctl.handle({"action": "explode"})["error"]

    def test_page_has_the_bridge(self):
        page = plugins_ui.page_html()
        assert "messageHandlers.wawi" in page and "Chalkboard" in page and 'type="password"' in page

    def test_window_round_trip_without_appkit(self, manager):
        manager.store.add(rest_plugin())
        rendered = []
        done = threading.Event()
        window = plugins_ui.PluginsWindow(plugins_ui.Controller(manager),
                                          call_after=lambda fn: (fn(), done.set()))

        class Web:
            def evaluateJavaScript_completionHandler_(self, js, handler):
                rendered.append(js)

        window.web = Web()
        window.receive(json.dumps({"action": "list"}))
        assert done.wait(5)
        assert rendered[0].startswith("wawiRender(") and "Mi Tienda" in rendered[0]
