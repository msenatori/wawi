"""Local, clock-injectable reminders and recurring instructions."""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_DAYS = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3,
         "viernes": 4, "sabado": 5, "domingo": 6}
_NUMBER = {"uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
           "cinco": 5, "diez": 10, "quince": 15, "veinte": 20,
           "treinta": 30, "sesenta": 60}
_CONFIRM_S = 180


def local_zone() -> tzinfo:
    """Use the system's named zone when available, not today's fixed UTC offset."""
    try:
        path = Path('/etc/localtime').resolve()
        marker = '/zoneinfo/'
        if marker in str(path):
            return ZoneInfo(str(path).split(marker, 1)[1])
        with Path('/etc/localtime').open('rb') as stream:
            return ZoneInfo.from_file(stream)
    except (OSError, ValueError):
        return datetime.now().astimezone().tzinfo or timezone.utc


def _number(value: str) -> int:
    value = value.lower()
    if value.isdigit():
        return int(value)
    if value in _NUMBER:
        return _NUMBER[value]
    raise ValueError('No entendí el número de tiempo.')


def _time(text: str) -> tuple[int, int]:
    match = re.search(r'\b(?:a\s+las?\s+)?(\d{1,2})(?::(\d{2}))?\s*(?:hs?|horas?)?\b', text.lower())
    if not match:
        raise ValueError('Decime una hora, por ejemplo «a las 9».')
    hour, minute = int(match[1]), int(match[2] or 0)
    if hour > 23 or minute > 59:
        raise ValueError('La hora tiene que estar entre 0:00 y 23:59.')
    return hour, minute


def parse_when(text: str, now: datetime) -> datetime:
    """Spanish relative, today/next occurrence and tomorrow expressions."""
    text = ' '.join(str(text or '').lower().split())
    if not text:
        raise ValueError('Decime cuándo querés que te avise.')
    match = re.fullmatch(r'(?:en|dentro de)\s+(\d+|\w+)\s+(segundos?|minutos?|horas?|dias?|días?)', text)
    if match:
        amount = _number(match[1])
        if amount <= 0:
            raise ValueError('El tiempo tiene que ser mayor que cero.')
        unit = match[2]
        seconds = amount * (86400 if unit.startswith(('dia', 'día')) else
                            3600 if unit.startswith('hora') else
                            60 if unit.startswith('minuto') else 1)
        return now + timedelta(seconds=seconds)
    if text.startswith('mañana') or text.startswith('manana'):
        hour, minute = _time(text.split(' ', 1)[1] if ' ' in text else '')
        day = now.date() + timedelta(days=1)
        return datetime(day.year, day.month, day.day, hour, minute, tzinfo=now.tzinfo)
    if text.startswith(('a las ', 'a la ')):
        hour, minute = _time(text)
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return candidate if candidate > now else candidate + timedelta(days=1)
    raise ValueError('Probá «en 20 minutos», «a las 18» o «mañana a las 9».')


def parse_schedule(text: str) -> dict[str, Any]:
    """Return normalized daily/weekly schedule, leaving instruction separate."""
    import unicodedata
    plain = unicodedata.normalize('NFKD', str(text or '').lower())
    plain = ''.join(c for c in plain if not unicodedata.combining(c))
    daily = re.search(r'\b(?:todos los dias|todas las mananas|cada manana|diariamente)\b', plain)
    weekly = re.search(r'\b(?:los|cada)\s+(lunes|martes|miercoles|jueves|viernes|sabado|domingo)\b', plain)
    if not daily and not weekly:
        raise ValueError('Usá «todas las mañanas a las 9» o «los viernes a las 18».')
    hour, minute = _time(plain)
    return {'frequency': 'daily' if daily else 'weekly', 'weekday': None if daily else _DAYS[weekly[1]],
            'hour': hour, 'minute': minute}


def next_occurrence(schedule: dict[str, Any], now: datetime) -> datetime:
    candidate = now.replace(hour=schedule['hour'], minute=schedule['minute'], second=0, microsecond=0)
    if schedule['frequency'] == 'weekly':
        candidate += timedelta(days=(schedule['weekday'] - now.weekday()) % 7)
        if candidate <= now:
            candidate += timedelta(days=7)
    elif candidate <= now:
        candidate += timedelta(days=1)
    return candidate


class ReminderService:
    """Durable scheduler. `step` is deterministic; the thread merely calls it."""

    def __init__(self, path: Path | str, *, clock: Callable[[], datetime] | None = None,
                 announce: Callable[[str], bool] | None = None,
                 run_routine: Callable[[str], bool] | None = None,
                 poll_s: float = 1.0) -> None:
        self.path = Path(path)
        self._clock = clock or (lambda: datetime.now(local_zone()))
        self._announce = announce or (lambda _: False)
        self._run_routine = run_routine or (lambda _: False)
        self._poll_s = poll_s
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.items: list[dict[str, Any]] = []
        self._pending: tuple[datetime, dict[str, Any]] | None = None
        self._confirmation_id: str | None = None
        self._due: list[dict[str, Any]] = []
        self._started_at = self.now()
        self._load()

    def now(self) -> datetime:
        now = self._clock()
        return now if now.tzinfo else now.replace(tzinfo=local_zone())

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict) or not isinstance(raw.get('items'), list):
                raise ValueError('Invalid reminders file')
            self.items = [item for item in raw['items'] if isinstance(item, dict) and
                          item.get('kind') in ('reminder', 'routine') and item.get('due')]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError):
            log.warning('Could not read reminders from %s', self.path, exc_info=True)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + '.tmp')
        tmp.write_text(json.dumps({'version': 1, 'items': self.items}, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(self.path)

    def _add(self, kind: str, due: datetime, text: str, **extra: Any) -> dict[str, Any]:
        if not text.strip():
            raise ValueError('Decime qué querés que recuerde o haga.')
        item = {'id': uuid.uuid4().hex[:8], 'kind': kind, 'due': due.isoformat(),
                'text': text.strip(), **extra}
        with self._lock:
            self.items.append(item)
            self._save()
        self._wake.set()
        return dict(item)

    def set_timer(self, *, seconds: int = 0, minutes: int = 0, label: str = '') -> dict[str, Any]:
        duration = int(seconds) + int(minutes) * 60
        if not 0 < duration <= 365 * 86400:
            raise ValueError('Elegí entre un segundo y un año.')
        return self._add('reminder', self.now() + timedelta(seconds=duration), label or 'Temporizador')

    def remind_me(self, when: str, text: str) -> dict[str, Any]:
        return self._add('reminder', parse_when(when, self.now()), text)

    def add_routine(self, schedule: str, instruction: str) -> dict[str, Any]:
        parsed = parse_schedule(schedule)
        if not instruction.strip():
            raise ValueError('Decime qué querés que haga en esa rutina.')
        with self._lock:
            self._pending = (self.now(), {'schedule': schedule, 'text': instruction, 'parsed': parsed})
        from wawi.confirm import REGISTRY
        if self._confirmation_id:
            REGISTRY.discard(self._confirmation_id)
        pending = self._pending
        def clear() -> None:
            with self._lock:
                if self._pending is pending:
                    self._pending = None
        def accepted() -> None:
            clear()
            draft = pending[1]
            self._add('routine', next_occurrence(draft['parsed'], self.now()),
                      draft['text'], schedule=draft['schedule'], parsed=draft['parsed'])
        self._confirmation_id = REGISTRY.add(
            "Programar rutina", f"{schedule}: {instruction[:65]}", "reminders", accepted, clear)
        return {'pending': True, 'schedule': schedule, 'text': instruction}

    def hear(self, text: str) -> str | None:
        """Only the owner's next transcript can authorize a routine."""
        from wawi.confirm import REGISTRY
        with self._lock:
            pending = self._pending
        if pending is None:
            return None
        if (self.now() - pending[0]).total_seconds() > _CONFIRM_S:
            REGISTRY.discard(self._confirmation_id or "")
            with self._lock:
                self._pending = None
            return None
        from wawi.lang import affirmative
        words = text.lower().split()
        answer = affirmative(text)
        if answer is True and len(words) <= 4 and 'pero' not in words:
            if REGISTRY.resolve(self._confirmation_id or "", True):
                return 'Listo, programé la rutina.'
        if answer is False and len(words) <= 4:
            if REGISTRY.resolve(self._confirmation_id or "", False):
                return 'Bueno, no la programo.'
        return None

    def list_items(self, kind: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self.items if item['kind'] == kind]

    def remove(self, identifier: str, kind: str) -> bool:
        with self._lock:
            matches = [item for item in self.items if item['kind'] == kind and
                       (item['id'] == identifier or item['text'].casefold() == identifier.casefold())]
            if len(matches) != 1:
                return False
            self.items.remove(matches[0])
            self._due = [item for item in self._due if item['id'] != matches[0]['id']]
            self._save()
            return True

    def step(self) -> str:
        """Collect due items once, then retry delivery without losing them when busy."""
        now = self.now()
        with self._lock:
            for item in list(self.items):
                if datetime.fromisoformat(item['due']) > now:
                    continue
                if not any(queued['id'] == item['id'] for queued in self._due):
                    overdue = datetime.fromisoformat(item['due']) < self._started_at
                    self._due.append({**item, 'overdue': overdue})
            if not self._due:
                return 'empty'
            item = self._due[0]
        if item['kind'] == 'routine':
            delivered = self._run_routine(item['text'])
        else:
            prefix = 'Se te pasó: ' if item['overdue'] else 'Recordatorio: '
            delivered = self._announce(prefix + item['text'])
        if not delivered:
            return 'busy'
        with self._lock:
            if self._due and self._due[0]['id'] == item['id']:
                self._due.pop(0)
                current = next((x for x in self.items if x['id'] == item['id']), None)
                if current is not None:
                    if current['kind'] == 'routine':
                        current['due'] = next_occurrence(current['parsed'], now).isoformat()
                    else:
                        self.items.remove(current)
                    self._save()
        return 'delivered'

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name='wawi-reminders', daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                log.warning('Reminder tick failed', exc_info=True)
            self._wake.wait(self._poll_s)
            self._wake.clear()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2)

_SERVICE: ReminderService | None = None


def install(service: ReminderService | None) -> None:
    global _SERVICE
    _SERVICE = service


def get() -> ReminderService | None:
    return _SERVICE


def hear(text: str) -> str | None:
    return _SERVICE.hear(text) if _SERVICE else None
