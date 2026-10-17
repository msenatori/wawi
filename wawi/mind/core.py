"""The mind: listens to events, thinks a little, pushes reactions.

Most of what Wawi does needs no model at all -- the sensors' reflex rules
put on the firefighter helmet the instant a site goes down. The mind adds
judgement on top, at the right moments only:

* now and then a personality bubble (local model, or a curated line),
* triage of low-severity events for the reflex layer (local model),
* a detective diagnosis after ``web.down`` (Claude, opt-in, rate-limited),
* the "¿qué hay de nuevo?" digest and the "andá a ver esta página" spaceship
  (Claude, because the user asked).

Threading: :meth:`Mind.on_event` runs on whatever thread emitted the event,
so it only records and hands slow work to a thread of its own. Nothing here
touches AppKit; everything visible goes out as a :class:`Reaction` on the bus.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
import uuid
from typing import Any, Callable

from wawi.mind import fetch as fetch_mod
from wawi.mind import peers
from wawi.mind.claude import (
    DIGEST_SYSTEM,
    PAGE_SYSTEM,
    ClaudeMind,
    ClaudeUnavailable,
    diagnose,
    gather_evidence,
)
from wawi.mind.local import LocalModel, Personality, Triage, summarize_state
from wawi.mind.settings import MindSettings
from wawi.reactions import BUS, Event, Reaction, ReactionBus

log = logging.getLogger(__name__)

#: How often the ticker wakes to see whether a bubble is due.
TICK_S = 30.0
#: Seconds a personality bubble stays up.
BUBBLE_S = 6.0
#: Head start for the reflex: the firefighter runs first, then the detective.
INVESTIGATE_DELAY_S = 2.0
#: Clip on any bubble the mind pushes; the body may clip harder.
MAX_BUBBLE_CHARS = 160


def _clip(text: str, limit: int = MAX_BUBBLE_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class Mind:
    def __init__(
        self,
        settings: MindSettings,
        *,
        bus: ReactionBus = BUS,
        claude: ClaudeMind | None = None,
        local: LocalModel | None = None,
        quiet: Callable[[], bool | None] = peers.is_quiet,
        clock: Callable[[], float] = time.time,
        investigate_delay_s: float = INVESTIGATE_DELAY_S,
        gather: Callable[..., dict[str, Any]] = gather_evidence,
        fetcher: Callable[..., fetch_mod.Page] = fetch_mod.fetch,
    ) -> None:
        self.settings = settings
        self.bus = bus
        self.claude = claude
        self.local = local
        self._quiet = quiet
        self._clock = clock
        self._delay = investigate_delay_s
        self._gather = gather
        self._fetcher = fetcher
        self.events: collections.deque[Event] = collections.deque(maxlen=300)
        self._events_lock = threading.Lock()
        self._investigating: set[str] = set()
        self._inv_lock = threading.Lock()
        self._last_digest_at = 0.0
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self.personality = Personality(
            local,
            min_interval_s=settings.bubble_min_interval_s,
            max_interval_s=settings.bubble_max_interval_s,
            clock=clock,
            quiet=quiet,
            timeout_s=settings.local_timeout_s,
        )
        self.triager = Triage(local if settings.triage else None,
                              timeout_s=min(2.0, settings.local_timeout_s))

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self.bus.subscribe(self.on_event)
        if self.settings.bubbles and self._ticker is None:
            self._ticker = threading.Thread(target=self._tick_loop, name="wawi-mind-tick", daemon=True)
            self._ticker.start()

    def stop(self) -> None:
        self._stop.set()
        if self.local is not None:
            self.local.close()

    def _tick_loop(self) -> None:
        while not self._stop.wait(TICK_S):
            try:
                self.tick()
            except Exception:
                log.warning("mind: tick failed", exc_info=True)

    # -- events -------------------------------------------------------------
    def recent_events(self, kind_prefix: str = "", since: float = 0.0) -> list[Event]:
        with self._events_lock:
            return [e for e in self.events if e.kind.startswith(kind_prefix) and e.at >= since]

    def on_event(self, event: Event) -> None:
        """Bus listener. Must stay quick: it runs on the emitter's thread."""
        with self._events_lock:
            self.events.append(event)
        # The investigation itself is free (DNS + two HTTP probes) and runs on
        # every web.down; only the Claude half of it is opt-in by kind.
        if event.kind == "web.down":
            self._start_investigation(event)

    def _start_investigation(self, event: Event) -> None:
        key = event.key or str(event.data.get("url", ""))
        with self._inv_lock:
            if key in self._investigating:
                return
            self._investigating.add(key)

        def _run() -> None:
            try:
                if self._delay:
                    time.sleep(self._delay)
                self.investigate_web_down(event)
            except Exception:
                log.warning("mind: web.down investigation failed", exc_info=True)
            finally:
                with self._inv_lock:
                    self._investigating.discard(key)

        threading.Thread(target=_run, name="wawi-mind-detective", daemon=True).start()

    def investigate_web_down(self, event: Event) -> str:
        """Gather evidence, diagnose, and push the detective. Returns the diagnosis."""
        url = str(event.data.get("url") or event.key or "")
        if not url:
            return ""
        evidence = self._gather(url, probe_url=self.settings.probe_url)
        claude = self.claude if "web.down" in self.settings.claude_events else None
        diagnosis, from_claude = diagnose(evidence, claude, f"web.down:{url}")
        quiet = self._quiet()
        # Unknown quiet state means silent: a voice out of nowhere in a meeting
        # costs more than a bubble nobody read.
        speak = quiet is False
        self.bus.push(
            Reaction(
                behaviour="detective",
                costume="detective",
                bubble=_clip(diagnosis),
                speak=speak,
                priority=3,
                duration=15.0,
                source=event,
            )
        )
        log.info("mind: web.down %s diagnosed (%s): %s", url,
                 "claude" if from_claude else "rules", diagnosis)
        return diagnosis

    # -- bubbles and triage -------------------------------------------------
    def tick(self) -> str | None:
        """Push a personality bubble if one is due. Returns the line pushed."""
        if not self.settings.bubbles or not self.personality.due():
            return None
        summary = summarize_state(self.recent_events(since=self._clock() - 3600)[-10:],
                                  peers.pet_status(), self._clock(),
                                  self_model=peers.self_model())
        line = self.personality.maybe_bubble(summary)
        if line:
            self.bus.push(Reaction(bubble=line, priority=0, duration=BUBBLE_S))
        return line

    def triage(self, event: Event, timeout: float | None = None) -> str | None:
        """now / later / ignore, or None = "let the reflex rules decide"."""
        try:
            return self.triager.decide(event, timeout)
        except Exception:
            log.debug("mind: triage failed", exc_info=True)
            return None

    # -- user-initiated Claude moments ----------------------------------------
    def digest(self) -> tuple[bool, str, dict[str, Any]]:
        """Answer "¿qué hay de nuevo?" from recent rss items and sensor status."""
        now = self._clock()
        since = max(self._last_digest_at, now - 24 * 3600)
        items = self.recent_events("rss.", since=since)[-15:]
        if not items and self._last_digest_at:
            items = self.recent_events("rss.", since=now - 24 * 3600)[-15:]
        sensors = peers.sensors_status()
        notable = [e for e in self.recent_events(since=now - 6 * 3600)
                   if e.severity in ("warning", "urgent")][-5:]
        data: dict[str, Any] = {
            "items": [{"title": e.title, "source": e.data.get("feed", "")} for e in items],
            "alerts": [e.title or e.kind for e in notable],
        }
        if not items and not notable and not sensors:
            return True, "No hay nada nuevo por ahora.", data
        payload = json.dumps(
            {"novedades": data["items"], "alertas": data["alerts"],
             "sensores": sensors if isinstance(sensors, (dict, list, str)) else str(sensors)},
            ensure_ascii=False,
        )[:6000]
        self._last_digest_at = now
        if self.claude is not None:
            try:
                return True, self.claude.ask(payload, DIGEST_SYSTEM), data
            except ClaudeUnavailable as exc:
                log.info("mind: digest without Claude (%s)", exc)
        if items:
            titles = "; ".join((e.title or "sin título")[:70] for e in items[:3])
            more = f", y {len(items) - 3} más" if len(items) > 3 else ""
            return True, f"Hay {len(items)} novedades: {titles}{more}.", data
        return True, "Nada nuevo en los feeds; " + (
            f"último aviso: {data['alerts'][-1]}." if notable else "los sensores están tranquilos."
        ), data

    def fetch_page(self, url: str, question: str = "") -> tuple[bool, str, dict[str, Any]]:
        """The spaceship: fly to the page, read it, come back with the answer.

        The ship reaction is ALWAYS closed -- celebrate on success, sad on any
        failure -- so the body is never left waiting for a ship that crashed.
        """
        key = f"fetch:{uuid.uuid4().hex[:8]}"
        self.bus.push(Reaction(behaviour="spaceship", until_key=key, priority=2))
        closed = False

        def _close(behaviour: str, bubble: str | None = None) -> None:
            nonlocal closed
            if not closed:
                closed = True
                self.bus.push(Reaction(behaviour=behaviour, cancel_key=key, priority=2,
                                       bubble=_clip(bubble, 60) if bubble else None,
                                       duration=3.0))

        try:
            try:
                page = self._fetcher(
                    url,
                    allow_private=self.settings.fetch_allow_private,
                    max_bytes=self.settings.fetch_max_bytes,
                    timeout=self.settings.fetch_timeout_s,
                )
            except fetch_mod.FetchError as exc:
                _close("sad", exc.spoken)
                return False, exc.spoken, {"error": exc.code, "url": url}
            text = page.text[: self.settings.fetch_max_chars]
            data: dict[str, Any] = {"url": page.url, "title": page.title,
                                    "truncated": page.truncated or len(page.text) > len(text)}
            if len(text.strip()) < 40:
                msg = "La página no tiene texto que pueda leer; capaz necesita JavaScript."
                _close("sad", "No hay nada para leer.")
                return False, msg, {**data, "error": "no_text"}
            if self.claude is None:
                _close("celebrate")
                return True, "Traje la página.", {**data, "excerpt": text[:2000]}
            prompt = (
                f"Pregunta: {question.strip() or '(ninguna: resumila)'}\n"
                f"Título: {page.title}\n"
                f"<pagina>\n{text}\n</pagina>"
            )
            try:
                answer = self.claude.ask(prompt, PAGE_SYSTEM)
            except ClaudeUnavailable as exc:
                log.info("mind: page fetched but not read (%s)", exc)
                _close("sad", "No pude leerla ahora.")
                # The brain is Claude too: hand it an excerpt so it can still try.
                return False, "Traje la página pero no pude leerla ahora.", {
                    **data, "error": "read_failed", "excerpt": text[:1500]}
            _close("celebrate")
            return True, answer, data
        except Exception:
            log.warning("mind: fetch_page failed", exc_info=True)
            _close("sad", "Algo salió mal.")
            return False, "Algo salió mal yendo a esa página.", {"error": "failed", "url": url}
        finally:
            _close("sad")
