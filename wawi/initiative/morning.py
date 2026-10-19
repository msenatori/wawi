"""Resumen de la mañana: when the owner arrives, the day on one board page.

The moment: the first activity of the day (or the first after
``away_hours`` without any), between ``from_hour`` and ``until_hour``, once
per day -- persisted, so a restart never brings it back. "Arrived" comes
from the engine's own persisted "last seen", never from Wawi starting.

The page is prepared quietly from what Wawi already has, all local:

* the weather the sensor already fetched (no new request);
* Orca agents working, waiting, or finished / failed in the last hours (the
  Orca watcher's last look);
* reminders due today (wawi/reminders);
* the last note's title and length (wawi/notes, newest file);
* the issues she filed herself (wawi/selfimprove);
* today's calendar events, only when the owner turned "Agenda en el resumen"
  on in the menu (EventKit, :mod:`wawi.initiative.calendar`), else nothing.

Then an offer, "Te dejé el resumen del día", whose link opens the board; a
kind the owner always opens is promoted to just opening it.
"""

from __future__ import annotations

import html
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from wawi.initiative.core import Act, Ctx, Moment, register

log = logging.getLogger(__name__)

TITLE = "Buen día"
OFFER = "Te dejé el resumen del día"
LINK = "Ver resumen…"
AUTO = "Te dejé el resumen del día en la pizarra."
DAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre")


@dataclass
class MorningFacts:
    weather: str | None = None
    orca: dict[str, list[str]] = field(default_factory=dict)
    #: (HH:MM, text) due today, oldest first.
    reminders: list[tuple[str, str]] = field(default_factory=list)
    #: (heading, lines) of the newest note.
    note: tuple[str, int] | None = None
    issues: list[str] = field(default_factory=list)
    #: (HH:MM or "todo el día", title); None = no calendar source.
    events: list[tuple[str, str]] | None = None


def _e(text: str) -> str:
    """Owner-local text on the board, never its own markup."""
    return html.escape(" ".join(str(text).split()), quote=False)


def assemble(facts: MorningFacts, now: float) -> tuple[str, str]:
    """(title, markdown) of the page. Pure."""
    d = datetime.fromtimestamp(now)
    lines = [f"# {TITLE}", "", f"_{DAYS[d.weekday()].capitalize()} {d.day} de {MONTHS[d.month - 1]}_", ""]
    if facts.weather:
        w = facts.weather
        lines += ["## Clima", "", w[0].upper() + w[1:] + ".", ""]
    if facts.events is not None:
        lines += ["## Agenda", ""]
        lines += [f"- **{_e(t)}** {_e(title)}" for t, title in facts.events] or ["Nada en el calendario."]
        lines.append("")
    o = facts.orca or {}
    if any(o.get(k) for k in ("working", "waiting", "done", "failed")):
        lines += ["## Agentes de Orca", ""]
        for key, label in (("waiting", "Te espera"), ("failed", "Falló"),
                           ("done", "Terminó"), ("working", "Trabajando")):
            titles = list(o.get(key) or [])
            lines += [f"- {label}: «{_e(t)}»" for t in titles[:5]]
            if len(titles) > 5:
                lines.append(f"- …y {len(titles) - 5} más ({label.lower()})")
        lines.append("")
    if facts.reminders:
        lines += ["## Recordatorios de hoy", ""]
        lines += [f"- **{_e(t)}** {_e(text)}" for t, text in facts.reminders]
        lines.append("")
    if facts.note is not None:
        heading, count = facts.note
        n = "sin líneas" if not count else ("1 línea" if count == 1 else f"{count} líneas")
        lines += ["## Última nota", "", f"{_e(heading)} ({n}).", ""]
    if facts.issues:
        lines += ["## Problemas que anoté", ""]
        lines += [f"- {_e(t)}" for t in facts.issues[:3]]
        lines.append("")
    if len(lines) <= 4:
        lines += ["Nada pendiente. Arrancá tranquilo.", ""]
    return TITLE, "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Gathering (each source guarded: a missing one is just left out)
# --------------------------------------------------------------------------- #


def weather_line() -> str | None:
    try:
        from wawi import sensors
        from wawi.sensors.morning import weather_clause

        runner = sensors._RUNNER
        for s in (runner.sensors if runner is not None else []):
            if s.name == "weather":
                return weather_clause(s)
    except Exception:
        log.debug("initiative: no weather for the morning", exc_info=True)
    return None


def reminders_today(now: float) -> list[tuple[str, str]]:
    try:
        from wawi import reminders

        service = reminders.get()
        if service is None:
            return []
        today = datetime.fromtimestamp(now).date()
        out = []
        for item in service.list_items("reminder"):
            try:
                due = datetime.fromisoformat(str(item.get("due")))
            except ValueError:
                continue
            if due.date() <= today:
                out.append((due, f"{due:%H:%M}", str(item.get("text") or "")))
        return [(t, text) for _, t, text in sorted(out)]
    except Exception:
        log.debug("initiative: no reminders for the morning", exc_info=True)
        return []


def last_note(directory: Any) -> tuple[str, int] | None:
    if not directory:
        return None
    try:
        folder = Path(str(directory)).expanduser()
        notes = sorted(folder.glob("*.md"), key=lambda p: p.stat().st_mtime)
        if not notes:
            return None
        text = notes[-1].read_text(encoding="utf-8", errors="replace")
        first = next((ln for ln in text.splitlines() if ln.strip()), notes[-1].stem)
        heading = first.lstrip("#").strip() or notes[-1].stem
        return heading, sum(1 for ln in text.splitlines() if ln.startswith("- **"))
    except OSError:
        return None


class MorningWatcher:
    kinds = ("morning",)

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.page: tuple[str, str] | None = None

    def poll(self, ctx: Ctx) -> list[Moment]:
        if not ctx.arrived:
            return []
        s = self.engine.settings
        lt = time.localtime(ctx.now)
        hour = lt.tm_hour + lt.tm_min / 60.0
        if not (s.morning_from_h <= hour < s.morning_until_h):
            return []
        from wawi.instinct.observer import day_key

        today = day_key(ctx.now)
        store = self.engine.store
        if store.morning_day == today:
            return []
        store.morning_day = today
        store.save()
        return [Moment("morning", f"morning:{today}")]

    def facts(self, now: float) -> MorningFacts:
        from wawi.initiative import calendar
        from wawi.initiative.orca_watch import OrcaWatcher, overnight

        e = self.engine
        orca = next((w for w in e.watchers if isinstance(w, OrcaWatcher)), None)
        try:
            issues = [str(t) for t in e.hooks.issues()][:3]
        except Exception:
            issues = []
        events = calendar.today_events(now) if e.store.calendar else None
        return MorningFacts(weather=weather_line(), orca=overnight(orca, now),
                            reminders=reminders_today(now), note=last_note(e.hooks.notes_dir),
                            issues=issues, events=events)

    def act(self, moment: Moment, ctx: Ctx) -> Act | None:
        board = self.engine.hooks.board
        if board is None:
            return None
        s = self.engine.settings
        lt = time.localtime(ctx.now)
        left = (s.morning_until_h - (lt.tm_hour + lt.tm_min / 60.0)) * 3600.0

        def prepare() -> None:
            self.page = assemble(self.facts(ctx.now), ctx.now)
            log.info("initiative: morning summary ready")

        def open_it() -> None:
            if self.page is not None:
                board.show(self.page[0], self.page[1])

        return Act("morning", OFFER, link=LINK, accept=open_it, prepare=prepare,
                   auto_text=AUTO, ttl_s=max(600.0, left), show_s=40.0, key=moment.key)


@register("morning", order=30)
def _morning_watcher(engine: Any) -> MorningWatcher:
    return MorningWatcher(engine)
