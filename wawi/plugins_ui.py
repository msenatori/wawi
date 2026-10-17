"""The "Agregados…" window: list, switch on/off, test, remove and add plugins.

A WKWebView page in the board's paper-and-crayon style (wawi/board.py's CSS)
instead of raw AppKit forms: a form with a password field is a few lines of
HTML and hundreds of lines of NSTextField. The page talks to Python through
ONE script-message handler (``window.webkit.messageHandlers.wawi``); every
message is a JSON string, handled by :class:`Controller` (pure Python, so the
tests need no window) on a worker thread, and the answer goes back to the
page on the main thread as ``wawiRender(state)``.

The token typed in the form goes straight to the Keychain and is never sent
back to the page, logged or written to the JSON.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

from wawi import plugins

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Controller: what the page can ask
# --------------------------------------------------------------------------- #


def _view(plugin: dict[str, Any], health: dict[str, str]) -> dict[str, Any]:
    """What the page shows about one plugin. References, never values."""
    mcp = plugin.get("mcp") or {}
    auth = plugin.get("auth") or {}
    where = plugin.get("base_url") or mcp.get("url") \
        or " ".join([mcp.get("command", ""), *mcp.get("args", [])]).strip()
    return {
        "id": plugin["id"],
        "name": plugin["name"],
        "kind": plugin["kind"],
        "enabled": plugin["enabled"],
        "writes": plugin["allow_writes"],
        "description": plugin["description"],
        "where": where[:120],
        "auth": "sin token" if auth.get("type", "none") == "none"
                else f"{auth['type']} · {plugins.describe_ref(auth.get('secret'))}",
        "endpoints": [f"{e['method']} {e['name']}" for e in plugin.get("endpoints", [])],
        "health": health.get(plugin["id"], ""),
    }


class Controller:
    """Handles one page message: ``{"action": …}`` -> new page state."""

    def __init__(self, manager: plugins.Manager) -> None:
        self.manager = manager

    def state(self, message: str = "", error: str = "", test: dict | None = None) -> dict[str, Any]:
        return {
            "plugins": [_view(p, self.manager.health) for p in self.manager.plugins()],
            "message": message,
            "error": error,
            "test": test,
            "file": str(self.manager.store.path),
        }

    def handle(self, raw: Any) -> dict[str, Any]:
        try:
            message = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (TypeError, ValueError):
            return self.state(error="Mensaje inválido.")
        action = str(message.get("action") or "list")
        pid = str(message.get("id") or "")
        store = self.manager.store
        try:
            if action == "list":
                return self.state()
            if action == "toggle":
                plugin = store.set_enabled(pid, bool(message.get("enabled")))
                word = "prendido" if plugin["enabled"] else "apagado"
                return self.state(f"{plugin['name']} {word}. Lo tomo en tu próxima pregunta.")
            if action == "remove":
                if not store.remove(pid):
                    return self.state(error="Ese agregado ya no está.")
                return self.state("Listo, lo quité (y su token del Llavero).")
            if action == "test":
                plugin = store.get(pid)
                if plugin is None:
                    return self.state(error="Ese agregado ya no está.")
                ok, line = plugins.test_plugin(plugin)
                return self.state(test={"id": pid, "ok": ok, "line": line})
            if action == "add":
                plugin = self.add(message)
                return self.state(f"Agregué {plugin['name']}. Lo tomo en tu próxima pregunta.")
        except plugins.PluginError as exc:
            return self.state(error=str(exc))
        except Exception:
            log.warning("plugins window: %s failed", action, exc_info=True)
            return self.state(error="Algo salió mal; mirá wawi.log.")
        return self.state(error=f"No sé hacer «{action}».")

    def add(self, form: dict[str, Any]) -> dict[str, Any]:
        """The "Agregar…" form. The token goes to the Keychain first."""
        name = " ".join(str(form.get("name") or "").split())
        if not name:
            raise plugins.PluginError("Ponele un nombre.")
        kind = str(form.get("kind") or "rest")
        target = str(form.get("target") or "").strip()
        auth_type = str(form.get("auth") or "none")
        token = str(form.get("token") or "").strip()
        pid = plugins.slug(name)
        if any(p["id"] == pid for p in self.manager.store.plugins()):
            raise plugins.PluginError(f"Ya hay un agregado «{pid}». Cambiale el nombre.")
        raw: dict[str, Any] = {"id": pid, "name": name, "kind": kind,
                               "description": str(form.get("description") or ""),
                               "allow_writes": bool(form.get("writes")),
                               "auth": {"type": auth_type, "header": str(form.get("header") or "")}}
        account: str | None = None
        if kind == "rest":
            raw["base_url"] = target
        elif "://" in target:
            raw["mcp"] = {"transport": "sse" if form.get("sse") else "http", "url": target}
        else:
            import shlex

            try:
                parts = shlex.split(target)
            except ValueError:
                raise plugins.PluginError("No entiendo el comando (¿comillas sin cerrar?).") from None
            if not parts:
                raise plugins.PluginError("Falta el comando del servidor MCP.")
            raw["mcp"] = {"transport": "stdio", "command": parts[0], "args": parts[1:]}
            raw["auth"] = {"type": "none"}
            if token:
                var = str(form.get("variable") or "").strip()
                if not var:
                    raise plugins.PluginError("Decime en qué variable de entorno va el token (p. ej. API_KEY).")
                raw["mcp"]["secret_env"] = {var: {"keychain": var}}
                account = var
        if raw["auth"]["type"] != "none":
            if not token:
                raise plugins.PluginError("Con autenticación hace falta el token.")
            raw["auth"]["secret"] = {"keychain": "token"}
            account = "token"
        plugin = plugins.normalize(raw)
        if account is not None:
            plugins.keychain_set(pid, account, token)
        return self.manager.store.add(plugin)


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

_PAGE_CSS = """
body{padding:16px 18px 28px}
h1{margin-bottom:.2em}.hint{color:var(--muted);font-size:.86rem;margin:0 0 12px}
.card{border:2px solid var(--ink);border-radius:12px 9px 13px 10px;padding:10px 12px;margin:0 0 10px;
  background:var(--paper);box-shadow:2px 3px 0 #292f6826}
.card.off{opacity:.62}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.name{font-family:"Chalkboard SE","Marker Felt",sans-serif;font-weight:700;font-size:1.08rem;flex:1}
.tag{font-size:.75rem;border:1.5px solid var(--ink);border-radius:9px;padding:0 6px}
.meta{font-size:.82rem;color:var(--muted);margin:4px 0;word-break:break-all}
.result{font-size:.82rem;margin-top:6px;padding:5px 8px;border-radius:7px;background:var(--wash);white-space:pre-wrap}
.result.bad{border-left:4px solid #b3261e}.result.ok{border-left:4px solid #2e7d32}
.switch{display:flex;align-items:center;gap:4px;font-size:.85rem}
.flash{padding:6px 10px;border:2px dashed var(--ink);border-radius:9px;margin:0 0 10px}
.flash.err{border-color:#b3261e;color:#b3261e}
form{display:grid;gap:7px;border:2px dashed var(--ink);border-radius:12px;padding:12px;margin-top:12px}
form label{display:grid;gap:3px;font-size:.86rem}
.down{color:#b3261e;font-weight:700}.up{color:#2e7d32}
"""

_PAGE_JS = r"""
const send = (msg) => window.webkit.messageHandlers.wawi.postMessage(JSON.stringify(msg));
let tests = {};
function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') n.className = v; else if (k.startsWith('on')) n[k] = v; else n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
}
function wawiRender(state) {
  if (state.test) tests[state.test.id] = state.test;
  const list = document.getElementById('list');
  list.replaceChildren();
  const flash = document.getElementById('flash');
  flash.replaceChildren();
  if (state.error) flash.append(el('div', {class: 'flash err'}, state.error));
  else if (state.message) flash.append(el('div', {class: 'flash'}, state.message));
  if (!state.plugins.length) list.append(el('p', {class: 'hint'}, 'Todavía no hay agregados. Sumá uno abajo.'));
  for (const p of state.plugins) {
    const toggle = el('input', {type: 'checkbox'});
    toggle.checked = p.enabled;
    toggle.onchange = () => send({action: 'toggle', id: p.id, enabled: toggle.checked});
    const test = el('button', {onclick: () => { test.textContent = 'Probando…'; send({action: 'test', id: p.id}); }}, 'Probar');
    // Two clicks instead of confirm(): WKWebView has no JS dialogs without a UI delegate.
    const remove = el('button', {onclick: () => {
      if (remove.dataset.armed) send({action: 'remove', id: p.id});
      else { remove.dataset.armed = '1'; remove.textContent = '¿Seguro? Quitar'; }
    }}, 'Quitar');
    const health = p.health === 'down' ? el('span', {class: 'down'}, '● caído') : p.health === 'up' ? el('span', {class: 'up'}, '● anda') : null;
    const card = el('div', {class: 'card' + (p.enabled ? '' : ' off')},
      el('div', {class: 'row'}, el('span', {class: 'name'}, p.name), el('span', {class: 'tag'}, p.kind.toUpperCase()),
         p.writes ? el('span', {class: 'tag'}, 'escritura con confirmación') : el('span', {class: 'tag'}, 'solo lectura'),
         health, el('label', {class: 'switch'}, toggle, 'prendido'), test, remove),
      p.description ? el('div', {class: 'meta'}, p.description) : null,
      el('div', {class: 'meta'}, p.where + ' · ' + p.auth + (p.endpoints.length ? ' · ' + p.endpoints.join(', ') : '')));
    const t = tests[p.id];
    if (t) card.append(el('div', {class: 'result ' + (t.ok ? 'ok' : 'bad')}, t.line));
    list.append(card);
  }
  document.getElementById('file').textContent = state.file;
  const add = document.getElementById('add-btn');
  add.disabled = false; add.textContent = 'Agregar';
  if (state.message && state.message.startsWith('Agregué')) document.getElementById('form').reset();
  sync();
}
// form.name / form.target are the FORM's own properties: fields by namedItem.
const field = (n) => document.getElementById('form').elements.namedItem(n);
function sync() {
  const rest = field('kind').value === 'rest';
  const command = !rest && !field('target').value.includes('://');
  const f = document.getElementById('form');
  f.querySelector('.auth').hidden = command;
  f.querySelector('.header').hidden = command || field('auth').value !== 'header';
  f.querySelector('.token').hidden = !command && field('auth').value === 'none';
  f.querySelector('.variable').hidden = !command;
  field('target').placeholder = rest ? 'https://mi-api.ejemplo.com/v1' : 'npx -y mi-servidor-mcp   o   https://…/mcp';
}
document.addEventListener('DOMContentLoaded', () => {
  const f = document.getElementById('form');
  f.oninput = sync;
  f.onsubmit = (ev) => {
    ev.preventDefault();
    const b = document.getElementById('add-btn');
    b.disabled = true; b.textContent = 'Agregando…';
    const command = field('kind').value === 'mcp' && !field('target').value.includes('://');
    send({action: 'add', name: field('name').value, kind: field('kind').value, target: field('target').value,
          auth: command ? 'none' : field('auth').value, header: field('header').value,
          token: field('token').value, variable: field('variable').value,
          description: field('description').value, writes: field('writes').checked});
    field('token').value = '';
  };
  sync();
  send({action: 'list'});
});
"""

_PAGE_BODY = """
<h1>Agregados</h1>
<p class="hint">Servicios tuyos que Wawi puede leer: APIs REST o servidores MCP.
Los tokens van al Llavero de macOS, nunca al archivo. Los cambios valen desde tu próxima pregunta.</p>
<div id="flash"></div>
<div id="list"></div>
<form id="form" autocomplete="off">
  <strong>Agregar…</strong>
  <label>Nombre <input name="name" required maxlength="60" placeholder="Mi API"></label>
  <label>Tipo <select name="kind"><option value="rest">REST</option><option value="mcp">MCP</option></select></label>
  <label>URL o comando <input name="target" required></label>
  <label class="auth">Autenticación <select name="auth">
    <option value="none">Ninguna</option><option value="bearer">Bearer token</option>
    <option value="header">Encabezado propio</option></select></label>
  <label class="header">Nombre del encabezado <input name="header" placeholder="X-Api-Key"></label>
  <label class="variable">Variable del token (MCP por comando) <input name="variable" placeholder="API_KEY"></label>
  <label class="token">Token <input name="token" type="password" autocomplete="new-password"></label>
  <label>Qué es (lo lee Wawi) <textarea name="description" rows="2" maxlength="300"
    placeholder="Mi tienda: pedidos, stock y ventas del día"></textarea></label>
  <label class="switch"><input type="checkbox" name="writes"> Permitir cambios (cada uno lo confirmás en pantalla)</label>
  <button id="add-btn" type="submit">Agregar</button>
</form>
<p class="hint">Endpoints y chequeos de salud: <code>scripts/wawictl plugins endpoint add &lt;id&gt; --name … --path …</code><br>
Registro: <code id="file"></code></p>
"""


def page_html() -> str:
    from wawi.board import _CSS

    return ("<!doctype html><html lang='es'><head><meta charset='utf-8'>"
            f"<style>{_CSS}{_PAGE_CSS}</style><script>{_PAGE_JS}</script></head>"
            f"<body>{_PAGE_BODY}</body></html>")


# --------------------------------------------------------------------------- #
# AppKit
# --------------------------------------------------------------------------- #

_WINDOW: "PluginsWindow | None" = None


def show() -> None:
    """Open (or bring forward) the window. Main thread: a menu click."""
    global _WINDOW
    manager = plugins.get()
    if manager is None:
        log.info("plugins window: no manager installed")
        return
    if _WINDOW is None:
        _WINDOW = PluginsWindow(Controller(manager))
    _WINDOW.show()


class PluginsWindow:
    def __init__(self, controller: Controller,
                 call_after: Callable[[Callable[[], None]], None] | None = None) -> None:
        self.controller = controller
        self.window: Any = None
        self.web: Any = None
        self._handler: Any = None
        if call_after is None:
            from PyObjCTools import AppHelper  # type: ignore[import]

            call_after = AppHelper.callAfter
        self._call_after = call_after

    def show(self) -> None:
        from AppKit import (NSApp, NSBackingStoreBuffered, NSColor, NSMakeRect,  # type: ignore[import]
                            NSViewHeightSizable, NSViewWidthSizable, NSWindow,
                            NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable,
                            NSWindowStyleMaskResizable, NSWindowStyleMaskTitled)

        if self.window is None:
            from WebKit import WKWebView, WKWebViewConfiguration, WKWebsiteDataStore  # type: ignore[import]

            style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                     | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
            window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 520, 620), style, NSBackingStoreBuffered, False)
            window.setTitle_("Agregados")
            window.setReleasedWhenClosed_(False)
            window.setMinSize_((420, 380))
            window.setTitlebarAppearsTransparent_(True)
            window.setBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, .984, .933, 1.0))
            config = WKWebViewConfiguration.alloc().init()
            config.setWebsiteDataStore_(WKWebsiteDataStore.nonPersistentDataStore())
            self._handler = _make_handler(self)
            config.userContentController().addScriptMessageHandler_name_(self._handler, "wawi")
            web = WKWebView.alloc().initWithFrame_configuration_(window.contentView().bounds(), config)
            web.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            window.contentView().addSubview_(web)
            web.loadHTMLString_baseURL_(page_html(), None)
            window.center()
            self.window, self.web = window, web
        else:
            self.receive(json.dumps({"action": "list"}))
        app = NSApp()
        if app is not None:
            app.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    def receive(self, body: str) -> None:
        """A message from the page (main thread). Handled on a worker: a
        test can take seconds and the Keychain spawns a process."""
        def _work() -> None:
            state = self.controller.handle(body)
            payload = json.dumps(state, ensure_ascii=True)
            self._call_after(lambda: self._render(payload))

        threading.Thread(target=_work, name="wawi-plugins-ui", daemon=True).start()

    def _render(self, payload: str) -> None:
        if self.web is not None:
            self.web.evaluateJavaScript_completionHandler_(f"wawiRender({payload})", None)


def _make_handler(owner: PluginsWindow) -> Any:
    import objc  # type: ignore[import]
    from Foundation import NSObject  # type: ignore[import]

    global _Handler
    if _Handler is None:
        protocol = objc.protocolNamed("WKScriptMessageHandler")

        class WawiPluginsMessageHandler(NSObject, protocols=[protocol]):  # type: ignore[misc]
            def userContentController_didReceiveScriptMessage_(self, _controller, message):  # noqa: N802
                target = getattr(self, "_owner", None)
                if target is not None:
                    target.receive(str(message.body()))

        _Handler = WawiPluginsMessageHandler
    handler = _Handler.alloc().init()
    handler._owner = owner
    return handler


_Handler: Any = None
