"""Her own little computer: a sandboxed Chromium in Docker on a LAN server that she
drives to browse the web FOR the owner, and that the owner can watch in a
small window beside her (bigger on demand) and take over.

Pieces:

* :mod:`wawi.webvm.image` — the container (Playwright + Xvfb + noVNC + API).
* :mod:`wawi.webvm.settings` — the ``webvm:`` config section.
* :mod:`wawi.webvm.keychain` — the API token and VNC password (Keychain).
* :mod:`wawi.webvm.client` — the control API client.
* :mod:`wawi.webvm.lifecycle` — docker start / stop / status / build.
* :mod:`wawi.webvm.safety` — what she asks before doing.
* :mod:`wawi.webvm.tools` — the brain's ``web_*`` tools.
* :mod:`wawi.webvm.view` — the mini and expanded windows (AppKit).
* :mod:`wawi.webvm.menu` — "Su compu ▸" in the menu bar.

:class:`WebVM` is the one service object (``install`` / ``get``), shared by
the tools (brain thread), the view and menu (main thread) and its own idle
watcher thread; its state is behind a lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from wawi.webvm.client import Client, WebVMError
from wawi.webvm.lifecycle import Lifecycle, Status
from wawi.webvm.settings import WebVMSettings

log = logging.getLogger(__name__)

#: How often the idle watcher looks (seconds).
WATCH_EVERY_S = 30.0
#: She keeps the laptop open this long after her last web step.
BUSY_AFTER_USE_S = 45.0
#: A pending «¿lo hago?» expires after this long.
CONFIRM_SECONDS = 180.0


@dataclass
class Pending:
    """An action waiting for the owner's yes."""

    kinds: list[str]
    what: str
    url: str
    action: Callable[[], tuple[bool, str, dict]]
    created: float
    confirmation_id: str = ""


class WebVM:
    def __init__(self, settings: WebVMSettings, lifecycle: Lifecycle | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        from wawi.webvm import keychain

        self.settings = settings
        self.lifecycle = lifecycle or Lifecycle(settings, keychain.credentials)
        self.clock = clock
        self._lock = threading.RLock()
        self.last_used = 0.0
        self.running = False
        #: The small view should be on screen (a session is under way).
        self.view_wanted = False
        #: The owner closed the small view: tools do not bring it back.
        self.owner_hid = False
        #: The owner asked for the big window (consumed by the view).
        self.expand_requested = False
        self.pending: Pending | None = None
        self._confirm_result: tuple[bool, str, dict] | None = None
        self._watcher: threading.Thread | None = None
        self._stop_watch = threading.Event()
        self._busy_shown = False
        #: What the last action reported (url/title), for the view's caption.
        self.page: dict[str, str] = {}

    # -- credentials for the view (main thread; cached) --------------------
    def view_password(self) -> str:
        from wawi.webvm import keychain

        return keychain.credentials()[1]

    # -- lifecycle --------------------------------------------------------
    def client(self) -> Client:
        return self.lifecycle.client()

    def touch(self) -> None:
        with self._lock:
            self.last_used = self.clock()

    def ensure(self) -> Client:
        """The client of a running computer, started if needed. Raises
        WebVMError with a sentence she can say."""
        self.touch()
        if not self.running:
            got = self.lifecycle.ensure_running()
            with self._lock:
                self.running = True
                self.page = {"url": str(got.get("url", "")), "title": str(got.get("title", ""))}
            self._start_watcher()
        with self._lock:
            if self.settings.show_view and not self.owner_hid:
                self.view_wanted = True
        return self.client()

    def note_page(self, payload: dict) -> None:
        with self._lock:
            self.page = {"url": str(payload.get("url", "")), "title": str(payload.get("title", ""))}

    def lost(self) -> None:
        """The computer stopped answering: next use starts it again."""
        with self._lock:
            self.running = False

    def done(self) -> None:
        """She is done showing it (web_done). The container stays up until idle."""
        with self._lock:
            self.view_wanted = False
            self.expand_requested = False
            self.owner_hid = False

    def show(self, on: bool | None = None, by_owner: bool = False) -> None:
        """Show / hide (None: toggle) the small view. Hidden *by_owner* (its ✕,
        the menu), it stays hidden while she keeps browsing, until shown again
        or she is done."""
        with self._lock:
            self.view_wanted = (not self.view_wanted) if on is None else bool(on)
            if by_owner:
                self.owner_hid = not self.view_wanted

    def expand(self) -> None:
        with self._lock:
            self.view_wanted = True
            self.expand_requested = True

    def open_from_menu(self, big: bool = False) -> None:
        """The menu's «Mostrar» / «Ventana grande»: start it if needed (on a
        thread of its own: that takes seconds), then show it."""
        with self._lock:
            self.owner_hid = False

        def _go() -> None:
            try:
                self.ensure()
            except WebVMError as exc:
                log.warning("webvm: could not start from the menu: %s", exc)
                return
            if big:
                self.expand()
            else:
                self.show(True)

        threading.Thread(target=_go, name="wawi-webvm-open", daemon=True).start()

    def stop(self) -> bool:
        """Stop the container now (menu «Apagar», idle). Any thread; blocking."""
        with self._lock:
            self.view_wanted = False
            self.expand_requested = False
            self.running = False
            self._drop_pending()
        self._stop_watch.set()
        self._set_busy(False)
        try:
            return self.lifecycle.stop()
        except WebVMError as exc:
            log.warning("webvm: stop failed: %s", exc)
            return False

    def status(self) -> Status:
        return self.lifecycle.status()

    def idle_for(self) -> float:
        with self._lock:
            return self.clock() - self.last_used if self.last_used else 0.0

    def check_idle(self) -> bool:
        """Stop the container when unused for idle_stop_min. True when stopped."""
        with self._lock:
            running = self.running
        if running and self.idle_for() >= self.settings.idle_stop_min * 60.0:
            log.info("webvm: idle for %.0f min; stopping", self.idle_for() / 60.0)
            self.stop()
            return True
        return False

    def _start_watcher(self) -> None:
        with self._lock:
            if self._watcher is not None and self._watcher.is_alive():
                return
            self._stop_watch = threading.Event()
            stop = self._stop_watch

            def _loop() -> None:
                while not stop.wait(WATCH_EVERY_S):
                    try:
                        if self.check_idle():
                            return
                    except Exception:
                        log.debug("webvm: idle watcher failed", exc_info=True)

            self._watcher = threading.Thread(target=_loop, name="wawi-webvm-idle", daemon=True)
            self._watcher.start()

    # -- the laptop pose ----------------------------------------------------
    def busy_now(self) -> bool:
        """She sits at her laptop: the view is up and she used it lately."""
        with self._lock:
            wanted = self.view_wanted and self.running
            recent = bool(self.last_used) and self.clock() - self.last_used < BUSY_AFTER_USE_S
        return wanted and recent

    def poll(self) -> None:
        """Called by the view tick (main thread): keep the laptop pose in sync."""
        self._set_busy(self.busy_now())

    def _set_busy(self, on: bool) -> None:
        if on == self._busy_shown:
            return
        self._busy_shown = on
        try:
            from wawi import working

            working.set_busy("webvm", on, "background")
        except Exception:
            log.debug("webvm: set_busy failed", exc_info=True)

    # -- confirmation («¿lo hago?») --------------------------------------
    def ask(self, kinds: list[str], what: str, url: str,
            action: Callable[[], tuple[bool, str, dict]]) -> None:
        """Park *action* until the owner says yes (voice: :meth:`hear`, or the
        Aceptar card)."""
        from wawi.confirm import REGISTRY
        from wawi.webvm.safety import KIND_WORDS

        with self._lock:
            self._drop_pending()
            pending = Pending(kinds, what, url, action, self.clock())
            self.pending = pending
            self._confirm_result = None

        def _clear() -> None:
            with self._lock:
                if self.pending is pending:
                    self.pending = None

        def _accept() -> None:
            _clear()
            self._confirm_result = self._run_pending(pending)

        title = f"Mi compu: {KIND_WORDS.get(kinds[0], 'confirmar')}"
        pending.confirmation_id = REGISTRY.add(title[:45], (what or url)[:95], "webvm",
                                               _accept, _clear, timeout=CONFIRM_SECONDS)

    def _drop_pending(self) -> None:
        if self.pending is not None and self.pending.confirmation_id:
            try:
                from wawi.confirm import REGISTRY

                REGISTRY.discard(self.pending.confirmation_id)
            except Exception:
                pass
        self.pending = None

    def _run_pending(self, pending: Pending) -> tuple[bool, str, dict]:
        try:
            client = self.ensure()
            now = client.info()
            if pending.url and str(now.get("url", "")) != pending.url:
                return False, "La página cambió; pedímelo de nuevo.", {"error": "page_changed"}
            ok, speak, data = pending.action()
            self.touch()
            # The turn that asked is over: she continues when told to.
            return ok, (speak + " ¿Sigo?") if ok else speak, data
        except WebVMError as exc:
            return False, str(exc), {"error": "webvm"}

    def hear(self, text: str) -> str | None:
        """The owner's answer to «¿lo hago?», read from the transcript before
        the brain runs. None when there is nothing pending or it is no answer."""
        from wawi import lang
        from wawi.confirm import REGISTRY

        with self._lock:
            pending = self.pending
        if pending is None:
            return None
        if self.clock() - pending.created > CONFIRM_SECONDS:
            with self._lock:
                self._drop_pending()
            return None
        words = text.lower().split()
        answer = lang.affirmative(text)
        if answer is True and len(words) <= 8 and "pero" not in words:
            if REGISTRY.resolve(pending.confirmation_id, True):
                result = self._confirm_result
                return result[1] if result else None
        if answer is False and len(words) <= 4:
            if REGISTRY.resolve(pending.confirmation_id, False):
                return "Bueno, no lo hago."
        return None


_SERVICE: WebVM | None = None


def build(cfg: Any) -> WebVM | None:
    """The service from config, or None when ``webvm.enabled`` is off."""
    raw = getattr(cfg, "webvm", None) or {}
    try:
        settings = WebVMSettings.from_dict(raw)
    except ValueError as exc:
        log.warning("webvm: config ignored: %s", exc)
        return None
    if not settings.enabled:
        return None
    return WebVM(settings)


def install(service: WebVM | None) -> None:
    global _SERVICE
    _SERVICE = service


def get() -> WebVM | None:
    return _SERVICE


def hear(text: str) -> str | None:
    service = _SERVICE
    return service.hear(text) if service is not None else None


def shutdown() -> None:
    """Wawi is quitting: stop the container in the background (best effort)."""
    service = _SERVICE
    if service is None or not service.running:
        return
    t = threading.Thread(target=service.stop, name="wawi-webvm-stop", daemon=True)
    t.start()
    t.join(timeout=8.0)
