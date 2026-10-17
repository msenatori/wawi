"""The morning greeting: one short bubble about the day, made locally.

``clock.morning`` used to say just "¡Buen día!". With the other senses
running it can say something worth reading — "Buen día. Hoy 24°, todas tus
webs andan." — from what they ALREADY know: the last forecast, the last
uptime check, the last disk reading. No model, no request: the greeting is
built from state in memory, so it costs nothing and cannot be slow.

Only what is known and worth saying goes in. A healthy disk is not news; a
site never checked yet is not "up".
"""

from __future__ import annotations

from typing import Any, Iterable

from wawi.sensors.base import Sensor, human_bytes

#: The bubble is clipped at 140 by the reflexes anyway; aim well under it.
MAX_CHARS = 120


def _by_name(sensors: Iterable[Sensor]) -> dict[str, Sensor]:
    return {s.name: s for s in sensors}


def weather_clause(sensor: Any) -> str | None:
    today = sensor.today() if hasattr(sensor, "today") else None
    if not today:
        return None
    temp = today.get("max") if today.get("max") is not None else today.get("temp")
    if temp is None:
        return None
    text = f"hoy {temp}°"
    if today.get("rain_at"):
        text += f" y llueve a las {today['rain_at']}"
    elif isinstance(today.get("rain"), (int, float)) and today["rain"] >= 60:
        text += " y puede llover"
    return text


def web_clause(sensor: Any) -> str | None:
    urls = list(getattr(sensor, "urls", []) or [])
    if not urls or not getattr(sensor, "checked", False):
        return None
    from wawi.sensors.web import _host

    down = sensor.down()
    if not down:
        return "todas tus webs andan" if len(urls) > 1 else f"{_host(urls[0])} anda"
    hosts = ", ".join(_host(u) for u in down)
    up = len(urls) - len(down)
    if len(down) == 1:
        tail = f" (las otras {up} andan)" if up > 1 else (" (la otra anda)" if up == 1 else "")
        return f"está caída {hosts}{tail}"
    return f"están caídas {hosts}"


def disk_clause(sensor: Any) -> str | None:
    alert = sensor.disk_alert() if hasattr(sensor, "disk_alert") else None
    if not alert:
        return None
    _vol, level, free = alert
    if level >= 2:
        return f"el disco está casi lleno ({human_bytes(free)} libres)"
    return f"el disco anda justo ({human_bytes(free)} libres)"


def compose(sensors: Iterable[Sensor]) -> str:
    """"Buen día. Hoy 24°, todas tus webs andan." — or "¡Buen día!" alone."""
    found = _by_name(sensors)
    clauses = []
    for name, fn in (("weather", weather_clause), ("web", web_clause), ("system", disk_clause)):
        sensor = found.get(name)
        if sensor is None:
            continue
        try:
            clause = fn(sensor)
        except Exception:
            clause = None
        if clause:
            clauses.append(clause)
    if not clauses:
        return "¡Buen día!"
    text = "Buen día."
    body = ""
    for clause in clauses:
        candidate = clause if not body else f"{body}, {clause}"
        if len(text) + 1 + len(candidate) + 1 > MAX_CHARS:
            break
        body = candidate
    if not body:
        return "¡Buen día!"
    return f"{text} {body[0].upper()}{body[1:]}."
