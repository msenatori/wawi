"""Her own little computer (wawi/webvm): settings, Keychain, client,
lifecycle, safety gates, tools, brain wiring, menu and the view's math.

Nothing here touches Docker, the network, the Keychain or AppKit: every seam
(runner, transport, security, clock) is faked.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from wawi import webvm
from wawi.webvm import keychain, safety
from wawi.webvm import menu as webvm_menu
from wawi.webvm import tools as webvm_tools
from wawi.webvm import view
from wawi.webvm.client import Client, Response, Unreachable, WebVMError
from wawi.webvm.lifecycle import LABEL, Lifecycle
from wawi.webvm.settings import ASK_KINDS, SettingsError, WebVMSettings

TOKEN = "tok-SECRET-123"
VNC = "vncSECR8"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


#: A made-up server for the tests (the real one lives in private/config.yaml).
BOX = {"context": "buildbox", "host": "10.0.0.42"}


class TestSettings:
    def test_defaults(self):
        s = WebVMSettings.from_dict(None)
        assert s.enabled is False
        # No server of anyone's baked in: that goes in private/config.yaml.
        assert s.context == "" and s.host == ""
        assert s.idle_stop_min == 15
        assert s.ask_before == ASK_KINDS
        assert s.persist_profile is False

    def test_overrides(self):
        s = WebVMSettings.from_dict({**BOX, "enabled": True, "host": "10.0.0.5", "api_port": 7000,
                                     "idle_stop_min": 5, "ask_before": ["payment"]})
        assert s.enabled and s.api_url == "http://10.0.0.5:7000"
        assert s.idle_stop_min == 5 and s.ask_before == ("payment",)

    @pytest.mark.parametrize("raw", [
        {"bogus": 1},
        {"api_port": 0},
        {"api_port": 6180, "view_port": 6180},
        {"idle_stop_min": 0},
        {"max_steps": 0},
        {"ask_before": "payment"},
        {"ask_before": ["everything"]},
        {"enabled": True, "context": "x", "host": ""},
        {"enabled": True, "host": "10.0.0.5"},
        [],
    ])
    def test_rejects_what_cannot_work(self, raw):
        with pytest.raises(SettingsError):
            WebVMSettings.from_dict(raw)

    def test_build_is_none_when_disabled_or_broken(self):
        assert webvm.build(SimpleNamespace(webvm={})) is None
        assert webvm.build(SimpleNamespace(webvm={**BOX, "enabled": True, "api_port": "x"})) is None
        assert isinstance(webvm.build(SimpleNamespace(webvm={**BOX, "enabled": True})), webvm.WebVM)

    def test_config_keeps_the_section(self, tmp_path):
        from wawi import config

        path = tmp_path / "config.yaml"
        path.write_text("webvm:\n  enabled: true\n  idle_stop_min: 7\n", encoding="utf-8")
        cfg = config.load(path)
        assert cfg.webvm == {"enabled": True, "idle_stop_min": 7}


# --------------------------------------------------------------------------- #
# Keychain
# --------------------------------------------------------------------------- #


class FakeSecurity:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.argv: list[list[str]] = []

    def __call__(self, args, stdin=None):
        self.argv.append(list(args))
        if args[0] == "find-generic-password":
            account = args[args.index("-a") + 1]
            assert args[args.index("-s") + 1] == keychain.SERVICE
            value = self.store.get(account)
            return subprocess.CompletedProcess(args, 0 if value else 44, (value or "") + "\n", "")
        if args == ["-i"]:
            parts = stdin.split('"')
            self.store[parts[3]] = parts[5]
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


class TestKeychain:
    def test_creates_once_then_reuses(self):
        sec = FakeSecurity()
        token, vnc = keychain.credentials(sec)
        assert len(token) >= 32 and len(vnc) == keychain.VNC_LENGTH
        assert keychain.credentials(sec) == (token, vnc)
        assert sec.store == {keychain.TOKEN_ACCOUNT: token, keychain.VNC_ACCOUNT: vnc}

    def test_the_secret_never_travels_in_argv(self):
        sec = FakeSecurity()
        token, vnc = keychain.credentials(sec)
        flat = " ".join(" ".join(a) for a in sec.argv)
        assert token not in flat and vnc not in flat

    def test_a_failed_store_says_so(self):
        def broken(args, stdin=None):
            return subprocess.CompletedProcess(args, 1, "", "denied")

        with pytest.raises(keychain.SecretError):
            keychain.get_or_create(keychain.TOKEN_ACCOUNT, broken)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class FakeTransport:
    def __init__(self, replies=None, fail=False):
        self.calls: list[tuple] = []
        self.replies = list(replies or [])
        self.fail = fail

    def __call__(self, method, url, headers, data, timeout):
        self.calls.append((method, url, headers, data, timeout))
        if self.fail:
            raise Unreachable("No llego a mi compu (refused).")
        if self.replies:
            status, body = self.replies.pop(0)
        else:
            status, body = 200, {"ok": True, "url": "https://x.test/", "title": "X"}
        return Response(status, json.dumps(body).encode())


class TestClient:
    def test_sends_the_bearer_and_json(self):
        t = FakeTransport()
        c = Client("http://h:1/", TOKEN, t)
        c.click(10, 20)
        method, url, headers, data, _ = t.calls[0]
        assert (method, url) == ("POST", "http://h:1/click")
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        assert json.loads(data) == {"x": 10, "y": 20, "double": False}

    def test_query_parameters(self):
        t = FakeTransport()
        Client("http://h:1", TOKEN, t).element_at(3, 4)
        assert t.calls[0][1] == "http://h:1/element?x=3&y=4"

    def test_repr_hides_the_token(self):
        assert TOKEN not in repr(Client("http://h:1", TOKEN, FakeTransport()))

    @pytest.mark.parametrize("status,needle", [(401, "clave"), (403, "dirección"), (502, "nav")])
    def test_errors_become_sentences(self, status, needle):
        t = FakeTransport([(status, {"ok": False, "error": "navigation failed"})])
        with pytest.raises(WebVMError) as err:
            Client("http://h:1", TOKEN, t).open("https://a.test")
        assert needle in str(err.value)
        assert TOKEN not in str(err.value)

    def test_unreachable(self):
        with pytest.raises(Unreachable):
            Client("http://h:1", TOKEN, FakeTransport(fail=True)).health()


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


class FakeRunner:
    """Answers docker commands from a little script of states."""

    def __init__(self, state="", image=True, run_rc=0, stderr=""):
        self.state = state
        self.image = image
        self.run_rc = run_rc
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict | None]] = []

    def __call__(self, argv, env=None, timeout=60):
        self.calls.append((list(argv), env))
        if self.stderr:
            return subprocess.CompletedProcess(argv, 1, "", self.stderr)
        sub = argv[3:] if argv[:3] == ["docker", "--context", "buildbox"] else argv
        if sub[:2] == ["ps", "-a"]:
            return subprocess.CompletedProcess(argv, 0, self.state + "\n" if self.state else "", "")
        if sub[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0 if self.image else 1, "",
                                               "" if self.image else "Error: No such image: x")
        if sub[0] == "run":
            if self.run_rc == 0:
                self.state = "running"
            return subprocess.CompletedProcess(argv, self.run_rc, "abc\n",
                                               "" if self.run_rc == 0 else "port is already allocated")
        if sub[0] in ("rm", "stop"):
            self.state = ""
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def docker_calls(self):
        return [c[0][3:] for c in self.calls]


class FakeClient:
    """A control API that answers like the real one, in memory."""

    def __init__(self, healthy=True):
        self.healthy = healthy
        self.calls: list[tuple] = []
        self.element: dict | None = {"tag": "a", "text": "Vuelos baratos", "form": None}
        self.focused_element: dict | None = {"tag": "input", "type": "text", "form": None}
        self.url = "https://example.com/"

    def _page(self):
        return {"ok": True, "url": self.url, "title": "Example"}

    def health(self, timeout=4.0):
        self.calls.append(("health",))
        if not self.healthy:
            raise Unreachable("No llego a mi compu.")
        return self._page()

    def info(self):
        return self._page()

    def open(self, url):
        self.calls.append(("open", url))
        self.url = url
        return {**self._page(), "status": 200}

    def screenshot(self, quality=65):
        self.calls.append(("screenshot",))
        return {**self._page(), "image_b64": "SU1H", "width": 1280, "height": 800,
                "elements": [{"x": 5, "y": 6, "tag": "a", "type": "", "text": "Más"}]}

    def click(self, x, y, double=False):
        self.calls.append(("click", x, y))
        return self._page()

    def type(self, text, submit=False):
        self.calls.append(("type", text, submit))
        return self._page()

    def key(self, name):
        self.calls.append(("key", name))
        return self._page()

    def scroll(self, dx=0, dy=0):
        self.calls.append(("scroll", dx, dy))
        return self._page()

    def back(self):
        self.calls.append(("back",))
        return self._page()

    def text(self, max_chars=6000):
        return {**self._page(), "text": "Hola " * 5, "truncated": False}

    def element_at(self, x, y):
        return {**self._page(), "element": self.element}

    def focused(self):
        return {**self._page(), "element": self.focused_element}


def _lifecycle(runner, client=None, settings=None, clock=None):
    client = client or FakeClient()
    ticks = iter(range(0, 10_000))
    return Lifecycle(settings or WebVMSettings.from_dict({**BOX, "enabled": True}),
                     lambda: (TOKEN, VNC), runner=runner,
                     client_factory=lambda url, token: client,
                     clock=clock or (lambda: next(ticks)), sleep=lambda s: None)


class TestLifecycle:
    def test_a_healthy_computer_costs_no_docker_call(self):
        runner = FakeRunner()
        _lifecycle(runner).ensure_running()
        assert runner.calls == []

    def test_starts_it_with_labels_limits_ports_and_no_secret_in_argv(self):
        client = FakeClient(healthy=False)
        runner = FakeRunner()

        def heal(argv, env=None, timeout=60):
            out = FakeRunner.__call__(runner, argv, env, timeout)
            if "run" in argv:
                client.healthy = True
            return out

        life = _lifecycle(heal, client)
        life.ensure_running()
        run_argv, env = next((a, e) for a, e in runner.calls if "run" in a)
        assert run_argv[:3] == ["docker", "--context", "buildbox"]
        joined = " ".join(run_argv)
        assert TOKEN not in joined and VNC not in joined
        assert env == {"WEBVM_TOKEN": TOKEN, "WEBVM_VNC_PASSWORD": VNC}
        for piece in ("--rm", f"--label {LABEL}", "--name wawi-webvm", "--memory 2g",
                      "--cpus 2", "--shm-size 1g", "-p 10.0.0.42:6180:8800",
                      "-p 10.0.0.42:6181:6080", "-e WEBVM_TOKEN", "wawi-webvm:1"):
            assert piece in joined
        assert "WEBVM_IDLE_EXIT_MIN=20" in joined
        assert "-v" not in run_argv  # ephemeral profile by default

    def test_persistent_profile_uses_its_own_volume(self):
        life = _lifecycle(FakeRunner(), settings=WebVMSettings.from_dict(
            {**BOX, "enabled": True, "persist_profile": True}))
        argv = life.run_argv()
        assert "wawi-webvm-profile:/home/pwuser/profile" in argv

    def test_a_leftover_container_is_removed_by_exact_name(self):
        client = FakeClient(healthy=False)
        runner = FakeRunner(state="exited")
        life = _lifecycle(runner, client)
        with pytest.raises(WebVMError):  # never becomes healthy in this test
            life.ensure_running(wait_s=3)
        assert ["rm", "-f", "wawi-webvm"] in runner.docker_calls()

    def test_missing_image_says_how_to_build(self):
        life = _lifecycle(FakeRunner(image=False), FakeClient(healthy=False))
        with pytest.raises(WebVMError, match="wawictl webvm build"):
            life.ensure_running()

    def test_server_unreachable(self):
        life = _lifecycle(FakeRunner(stderr="ssh: connect to host buildbox port 22: No route to host"),
                          FakeClient(healthy=False))
        with pytest.raises(WebVMError, match="No llego al servidor"):
            life.ensure_running()

    def test_busy_ports(self):
        life = _lifecycle(FakeRunner(run_rc=1), FakeClient(healthy=False))
        with pytest.raises(WebVMError, match="puertos"):
            life.ensure_running()

    def test_stop_touches_only_its_container(self):
        runner = FakeRunner(state="running")
        assert _lifecycle(runner).stop() is True
        assert runner.docker_calls()[-1] == ["stop", "-t", "5", "wawi-webvm"]
        ps = runner.docker_calls()[0]
        assert "name=^/wawi-webvm$" in ps and f"label={LABEL}" in ps

    def test_stop_without_a_container(self):
        runner = FakeRunner(state="")
        assert _lifecycle(runner).stop() is False
        assert all(c[0] != "stop" for c in runner.docker_calls())

    def test_status(self):
        assert _lifecycle(FakeRunner()).status().healthy
        down = _lifecycle(FakeRunner(state=""), FakeClient(healthy=False)).status()
        assert not down.running and "apagada" in down.describe()
        dead = _lifecycle(FakeRunner(stderr="Cannot connect to the Docker daemon"),
                          FakeClient(healthy=False)).status()
        assert "servidor" in dead.describe()

    def test_build_rsyncs_then_builds_on_the_server(self):
        cmds = _lifecycle(FakeRunner()).build_commands()
        assert cmds[0] == ["ssh", "buildbox", "mkdir -p ~/work/wawi-webvm"]
        assert cmds[1][0] == "rsync" and cmds[1][-1] == "buildbox:work/wawi-webvm/"
        assert f"docker build --label {LABEL} -t wawi-webvm:1" in cmds[2][2]


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


def _el(tag="button", text="", type_="", form=None, **kw):
    return {"tag": tag, "text": text, "type": type_, "form": form, **kw}


class TestSafety:
    @pytest.mark.parametrize("element,kinds", [
        (_el(text="Comprar ahora"), {"purchase"}),
        (_el(text="Pagar $ 120.000"), {"payment"}),
        (_el(tag="a", text="Iniciar sesión"), {"login"}),
        (_el(text="Enviar"), {"message"}),
        (_el(text="Publicar"), {"post"}),
        (_el(text="Buscar vuelos"), set()),
        (_el(tag="a", text="Ver más resultados"), set()),
        (_el(tag="div", text="Comprar"), set()),  # not clickable as such
        (_el(text="Continuar", form={"has_password": True}), {"login"}),
        (_el(text="Continuar", form={"names": ["cc-number", "cvv"]}), {"payment"}),
        (_el(text="Buscar", form={"input_types": ["text"], "names": ["q"]}), set()),
    ])
    def test_click_kinds(self, element, kinds):
        assert safety.click_kinds(element) == kinds

    def test_enter_in_a_login_form(self):
        focused = _el(tag="input", type_="text", form={"has_password": True})
        assert "login" in safety.submit_kinds(focused)
        assert safety.submit_kinds(_el(tag="input", form={"names": ["q"]})) == set()
        assert safety.submit_kinds(_el(tag="textarea", form={"has_password": True})) == set()

    def test_password_fields(self):
        assert safety.is_password_field({"type": "password"})
        assert safety.is_password_field({"type": "text", "autocomplete": "current-password"})
        assert not safety.is_password_field({"type": "search"})

    @pytest.mark.parametrize("text,element,personal", [
        ("ana@example.com", None, True),
        ("4111 1111 1111 1111", None, True),
        ("+54 9 351 555 1234", None, True),
        ("Córdoba", None, False),
        ("vuelos a córdoba el viernes", None, False),
        ("Ana", {"autocomplete": "given-name"}, True),
        ("Ana", {"type": "email"}, True),
    ])
    def test_personal_data(self, text, element, personal):
        assert ("personal_data" in safety.text_kinds(text, element)) is personal

    def test_only_kinds_the_owner_wants_asked(self):
        assert safety.needs_asking({"purchase", "post"}, ("post",)) == ["post"]
        assert safety.needs_asking({"payment", "login"}, ASK_KINDS) == ["payment", "login"]
        assert safety.needs_asking(set(), ASK_KINDS) == []

    @pytest.mark.parametrize("url,ok", [
        ("https://www.google.com/search?q=vuelos", True),
        ("despegar.com.ar", True),
        ("file:///etc/passwd", False),
        ("javascript:alert(1)", False),
        ("http://192.168.0.1/", False),
        ("http://127.0.0.1:8080", False),
        ("http://localhost:3000", False),
        ("http://buildbox:6180/health", False),
        ("http://nas.local/", False),
        ("http://10.1.2.3", False),
        ("", False),
    ])
    def test_check_url(self, url, ok):
        assert (safety.check_url(url)[0] is not None) is ok

    def test_lan_when_allowed(self):
        assert safety.check_url("http://192.168.0.1/", allow_lan=True)[0]


# --------------------------------------------------------------------------- #
# Service + tools (through wawi.tools, as the brain calls them)
# --------------------------------------------------------------------------- #


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def vm(monkeypatch):
    """An installed service over a fake API and a fake docker."""
    from wawi.confirm import REGISTRY

    client = FakeClient()
    runner = FakeRunner(state="running")
    clock = Clock()
    settings = WebVMSettings.from_dict({**BOX, "enabled": True})
    life = Lifecycle(settings, lambda: (TOKEN, VNC), runner=runner,
                     client_factory=lambda url, token: client)
    service = webvm.WebVM(settings, life, clock=clock)
    monkeypatch.setattr(service, "_start_watcher", lambda: None)
    monkeypatch.setattr(service, "_set_busy", lambda on: None)
    webvm.install(service)
    yield SimpleNamespace(service=service, client=client, runner=runner, clock=clock)
    webvm.install(None)
    for item in REGISTRY.pending():
        REGISTRY.discard(item.id)


class TestTools:
    def test_hidden_without_the_service(self):
        from wawi import tools

        webvm.install(None)
        names = {t["name"] for t in tools.active_tools()}
        assert not names & set(webvm_tools.NAMES)
        assert tools.web_step_budget("web_open") == 0
        assert tools.dispatch("web_open", {"url": "a.test"}).ok is False

    def test_active_with_the_service_and_without_local_fields(self, vm):
        from wawi import tools

        active = {t["name"]: t for t in tools.active_tools()}
        assert set(webvm_tools.NAMES) <= set(active)
        assert "animation" not in active["web_open"]
        assert tools.animation_for("web_click") == "laptop"
        assert tools.has_no_animation("web_done")
        assert tools.web_step_budget("web_look") == 20
        assert tools.web_step_budget("show_board") == 0

    def test_open_starts_and_shows(self, vm):
        from wawi import tools

        r = tools.dispatch("web_open", {"url": "vuelos.test"})
        assert r.ok and ("open", "https://vuelos.test") in vm.client.calls
        assert vm.service.running and vm.service.view_wanted

    def test_open_refuses_the_lan(self, vm):
        from wawi import tools

        r = tools.dispatch("web_open", {"url": "http://192.168.0.10/admin"})
        assert not r.ok and not any(c[0] == "open" for c in vm.client.calls)

    def test_look_returns_the_image_apart_from_the_data(self, vm):
        from wawi import tools

        r = tools.dispatch("web_look", {})
        assert r.ok and r.image_b64 == "SU1H"
        assert "SU1H" not in json.dumps(r.data)
        assert r.data["elements"][0] == {"x": 5, "y": 6, "tag": "a", "text": "Más"}
        assert "SU1H" not in repr(r)

    def test_a_harmless_click_just_clicks(self, vm):
        from wawi import tools

        assert tools.dispatch("web_click", {"x": 1, "y": 2}).ok
        assert ("click", 1.0, 2.0) in vm.client.calls

    def test_a_purchase_click_waits_for_the_owner(self, vm):
        from wawi import tools

        vm.client.element = _el(text="Comprar pasaje")
        r = tools.dispatch("web_click", {"x": 7, "y": 8})
        assert r.ok and r.data["pending"] is True and "¿Lo hago?" in r.speak
        assert not any(c[0] == "click" for c in vm.client.calls)
        # The owner's spoken yes, read before the brain runs, does it.
        said = webvm.hear("sí, dale")
        assert said and "click" in said.lower()
        assert ("click", 7.0, 8.0) in vm.client.calls
        assert vm.service.pending is None

    def test_a_no_drops_it(self, vm):
        from wawi import tools

        vm.client.element = _el(text="Pagar")
        tools.dispatch("web_click", {"x": 7, "y": 8})
        assert webvm.hear("no") == "Bueno, no lo hago."
        assert not any(c[0] == "click" for c in vm.client.calls)
        assert webvm.hear("sí") is None  # nothing pending any more

    def test_a_changed_page_is_not_clicked_blindly(self, vm):
        from wawi import tools

        vm.client.element = _el(text="Comprar")
        tools.dispatch("web_click", {"x": 7, "y": 8})
        vm.client.url = "https://other.test/"
        assert "cambió" in webvm.hear("sí")
        assert not any(c[0] == "click" for c in vm.client.calls)

    def test_the_pending_question_expires(self, vm):
        from wawi import tools

        vm.client.element = _el(text="Comprar")
        tools.dispatch("web_click", {"x": 7, "y": 8})
        vm.clock.t += webvm.CONFIRM_SECONDS + 1
        assert webvm.hear("sí") is None
        assert not any(c[0] == "click" for c in vm.client.calls)

    def test_never_types_into_a_password_field(self, vm):
        from wawi import tools

        vm.client.focused_element = {"tag": "input", "type": "password", "form": None}
        r = tools.dispatch("web_type", {"text": "hunter2"})
        assert not r.ok and r.data["error"] == "password_field"
        assert not any(c[0] == "type" for c in vm.client.calls)

    def test_personal_data_waits(self, vm):
        from wawi import tools

        r = tools.dispatch("web_type", {"text": "ana@example.com"})
        assert r.data.get("pending")
        assert not any(c[0] == "type" for c in vm.client.calls)

    def test_plain_search_types_and_submits(self, vm):
        from wawi import tools

        vm.client.focused_element = {"tag": "input", "type": "search",
                                     "form": {"names": ["q"], "input_types": ["search"]}}
        assert tools.dispatch("web_type", {"text": "vuelos a córdoba", "submit": True}).ok
        assert ("type", "vuelos a córdoba", True) in vm.client.calls

    def test_enter_in_a_login_form_waits(self, vm):
        from wawi import tools

        vm.client.focused_element = {"tag": "input", "type": "email",
                                     "form": {"has_password": True}}
        r = tools.dispatch("web_key", {"name": "Enter"})
        assert r.data.get("pending")
        assert not any(c[0] == "key" for c in vm.client.calls)
        assert tools.dispatch("web_key", {"name": "PageDown"}).ok

    def test_ask_before_can_be_narrowed(self, vm, monkeypatch):
        from wawi import tools

        monkeypatch.setattr(vm.service, "settings", WebVMSettings.from_dict(
            {**BOX, "enabled": True, "ask_before": ["payment"]}))
        vm.client.element = _el(text="Publicar")
        assert tools.dispatch("web_click", {"x": 1, "y": 1}).data.get("pending") is None

    def test_read_scroll_back_done(self, vm):
        from wawi import tools

        assert "Hola" in tools.dispatch("web_read", {}).data["text"]
        assert tools.dispatch("web_scroll", {"dy": 600}).ok
        assert tools.dispatch("web_back", {}).ok
        tools.dispatch("web_open", {"url": "a.test"})
        assert vm.service.view_wanted
        assert tools.dispatch("web_done", {}).ok
        assert not vm.service.view_wanted

    def test_unreachable_restarts_once(self, vm):
        from wawi import tools

        tools.dispatch("web_open", {"url": "a.test"})
        calls = {"n": 0}
        real = vm.client.text

        def flaky(max_chars=6000):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Unreachable("No llego a mi compu.")
            return real(max_chars)

        vm.client.text = flaky
        assert tools.dispatch("web_read", {}).ok
        assert calls["n"] == 2

    def test_errors_are_sentences(self, vm, monkeypatch):
        from wawi import tools

        def boom():
            raise WebVMError("No llego al servidor de mi compu: ¿está prendido y en la red?")

        monkeypatch.setattr(vm.service.lifecycle, "ensure_running", boom)
        r = tools.dispatch("web_open", {"url": "a.test"})
        assert not r.ok and "servidor" in r.speak


class TestService:
    def test_idle_stop(self, vm):
        vm.service.ensure()
        vm.clock.t += 14 * 60
        assert vm.service.check_idle() is False
        vm.clock.t += 2 * 60
        assert vm.service.check_idle() is True
        assert not vm.service.running and not vm.service.view_wanted
        assert ["stop", "-t", "5", "wawi-webvm"] in vm.runner.docker_calls()

    def test_owner_hid_it_stays_hidden_while_she_browses(self, vm):
        vm.service.ensure()
        vm.service.show(False, by_owner=True)
        vm.service.ensure()
        assert not vm.service.view_wanted
        vm.service.done()
        vm.service.ensure()
        assert vm.service.view_wanted

    def test_laptop_pose_follows_use(self, vm):
        vm.service.ensure()
        assert vm.service.busy_now()
        vm.clock.t += webvm.BUSY_AFTER_USE_S + 1
        assert not vm.service.busy_now()


# --------------------------------------------------------------------------- #
# Brain wiring: the web budget and the screenshot as an image block
# --------------------------------------------------------------------------- #


class TestBrain:
    def test_web_steps_have_their_own_budget(self, vm):
        from wawi import brain

        state = brain._TurnState(max_iterations=2)
        assert all(state.note("web_look") for _ in range(20))
        assert not state.note("web_look")
        assert state.note("show_board") and state.note("show_board")
        assert not state.note("show_board")

    def test_without_the_service_web_tools_count_as_usual(self):
        from wawi import brain

        webvm.install(None)
        state = brain._TurnState(max_iterations=1)
        assert state.note("web_look") and not state.note("web_look")

    def test_look_comes_back_as_an_image_and_rotates_the_session(self, vm):
        from wawi import brain

        state = brain._TurnState()
        out = brain._run_tool("web_look", {}, state)
        image, text = out["content"]
        assert image == {"type": "image", "data": "SU1H", "mimeType": "image/jpeg"}
        assert json.loads(text["text"])["data"]["url"] == "https://example.com/"
        assert state.shots  # Engine.ask rotates a session that holds images

    def test_other_tools_stay_text(self, vm):
        from wawi import brain

        out = brain._run_tool("web_read", {}, brain._TurnState())
        assert [c["type"] for c in out["content"]] == ["text"]

    def test_budget_is_part_of_the_connection_signature(self, vm):
        from wawi import brain

        with_vm = brain._extras_signature(None)
        webvm.install(None)
        assert brain._extras_signature(None) != with_vm


# --------------------------------------------------------------------------- #
# Menu and the view's math
# --------------------------------------------------------------------------- #


class TestMenu:
    def test_entries_follow_the_state(self):
        off = SimpleNamespace(running=False, view_wanted=False, page={})
        rows = webvm_menu.entries(off)
        assert rows[0][0] == "Apagada"
        assert ("Mostrar", "show", True) in rows and ("Apagar", "stop", False) in rows
        on = SimpleNamespace(running=True, view_wanted=True, page={"title": "Vuelos"})
        rows = webvm_menu.entries(on)
        assert rows[0][0] == "Prendida: Vuelos"
        assert ("Ocultar", "hide", True) in rows and ("Apagar", "stop", True) in rows

    def test_tag_is_free(self):
        from wawi.panel import PanelObserver

        taken = {PanelObserver._WAKE_MODE_TAG, PanelObserver._ORCA_TAG, PanelObserver._NOTES_TAG,
                 PanelObserver._CURIOSITY_TAG, PanelObserver._PLUGINS_TAG, PanelObserver._MUSIC_TAG}
        assert webvm_menu.TAG == PanelObserver._WEBVM_TAG and webvm_menu.TAG not in taken


class TestViewMath:
    VISIBLE = (0.0, 0.0, 1440.0, 900.0)

    def test_beside_her_top_aligned(self):
        x, y = view.place_mini((100.0, 300.0, 200.0, 200.0), self.VISIBLE)
        assert x == 100 + 200 + view.GAP
        assert y + view.HEIGHT == 500.0

    def test_flips_to_her_left_at_the_screen_edge(self):
        x, _ = view.place_mini((1300.0, 300.0, 120.0, 120.0), self.VISIBLE)
        assert x == 1300 - view.GAP - view.WIDTH

    def test_always_on_screen(self):
        x, y = view.place_mini((-50.0, -80.0, 100.0, 100.0), self.VISIBLE)
        assert 0 <= x <= 1440 - view.WIDTH and 0 <= y <= 900 - view.HEIGHT

    def test_no_anchor_bottom_right(self):
        x, y = view.place_mini(None, self.VISIBLE)
        assert x > 1000 and y < 100

    def test_big_window_is_centred_and_fits(self):
        x, y, w, h = view.place_big(self.VISIBLE)
        assert (w, h) == (view.BIG_W, view.BIG_H)
        assert x == (1440 - w) / 2 and y == (900 - h) / 2
        _, _, w, h = view.place_big((0.0, 0.0, 800.0, 500.0))
        assert w <= 760 and h <= 460

    def test_buttons(self):
        bx, by, bw, bh = view.EXPAND_BUTTON
        assert view.hit(bx + bw / 2, by + bh / 2) == "expand"
        cx, cy, cw, ch = view.CLOSE_BUTTON
        assert view.hit(cx + cw / 2, cy + ch / 2) == "close"
        assert view.hit(50, 100) is None

    def test_fade(self):
        assert view.fade(0.0, 1.0, view.FADE_S / 2) == pytest.approx(0.5)
        assert view.fade(0.9, 1.0, 1.0) == 1.0
        assert view.fade(1.0, 0.0, 10.0) == 0.0

    def test_viewer_page_keeps_the_password_out_of_urls(self):
        page = view.viewer_html("10.0.0.42", 6181, 'p"w</script>', view_only=True, big=False)
        assert "credentials" in page and "rfb.viewOnly = true" in page
        assert 'p"w</script>' not in page  # JSON-escaped, never raw
        assert "password=" not in page  # not in any URL
        big = view.viewer_html("h", 1, "x", view_only=True, big=True)
        assert "Tomar el control" in big
