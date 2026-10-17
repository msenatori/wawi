"""What Wawi watches, changed by voice, persisted beside the config.

"Wawi, vigilá example.com" must work now (no restart) and still be true
tomorrow. Rewriting the user's hand-written ``config.yaml`` to get there
would be rude (comments, ordering, the noise floor they tuned by hand), so
spoken changes go to a separate, gitignored ``sensors_local.json`` that is
merged OVER ``config.yaml``'s ``sensors:`` section at startup and on every
change.

The file stores **deltas**, not a copy of the lists::

    {"version": 1, "sensors": {
        "web":      {"add": {"urls": ["https://example.com"]},
                     "remove": {"urls": ["https://old.example"]}},
        "weather":  {"set": {"enabled": true, "city": "Córdoba", "lat": -31.4, "lon": -64.2}}}}

Deltas because the user keeps editing ``config.yaml`` too: a stored full
list would silently shadow every URL they add there later.

Everything here is pure (dicts in, dicts out) except :func:`load_local`,
:func:`save_local` and :func:`geocode`, so the rules are testable without a
runner, a network or a disk.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlparse

from wawi.sensors.base import as_list

log = logging.getLogger(__name__)

DEFAULT_LOCAL_FILE = "sensors_local.json"

#: Turned on when ``sensors:`` names no sensor at all: the set that needs no
#: personal input and makes no network request. Weather, web and rss wait
#: for a city or a URL; services wait for something to watch.
DEFAULT_SENSORS: dict[str, dict[str, Any]] = {
    "system": {"enabled": True},  # disk + battery (memory/cpu stay opt-in)
    "clock": {"enabled": True},  # day parts + "¿una pausa?"
}

SENSOR_NAMES = ("web", "system", "services", "weather", "rss", "clock")

#: Spoken kind -> (sensor, list field). Weather is a setting, not a list.
KINDS: dict[str, tuple[str, str]] = {
    "web": ("web", "urls"),
    "feed": ("rss", "feeds"),
    "port": ("services", "ports"),
    "container": ("services", "docker"),
    "process": ("services", "processes"),
}
ALL_KINDS = (*KINDS, "weather")


class Invalid(ValueError):
    """A target that must not be watched; the message is spoken (Spanish)."""


# --------------------------------------------------------------------------- #
# Defaults and merging
# --------------------------------------------------------------------------- #


def with_defaults(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """*cfg* plus :data:`DEFAULT_SENSORS` when it names no sensor.

    ``sensors: {}`` (the template's value, and what a config without the
    section gets) means "nobody chose", so Wawi starts with the safe set.
    ``enabled: false`` or ``defaults: false`` opts out.
    """
    cfg = dict(cfg or {})
    if cfg.get("enabled", True) is False or cfg.get("defaults", True) is False:
        return cfg
    if any(name in cfg for name in SENSOR_NAMES):
        return cfg
    out = copy.deepcopy(DEFAULT_SENSORS)
    out.update(cfg)
    return out


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    raw = cfg.get(name)
    if raw is True:
        return {"enabled": True}
    return dict(raw) if isinstance(raw, dict) else {}


def ident(field: str, item: Any) -> str:
    """Comparable identity of a list entry (URLs and ports have many spellings)."""
    if isinstance(item, dict):
        if field == "ports":
            item = f"{item.get('host', '127.0.0.1')}:{item.get('port', '')}"
        else:
            item = item.get("url") or item.get("name") or ""
    text = str(item).strip()
    if field == "ports":
        host, _, port = text.rpartition(":")
        host = (host or "127.0.0.1").lower()
        if host == "localhost":
            host = "127.0.0.1"
        return f"{host}:{port}"
    if field in ("urls", "feeds"):
        return text.rstrip("/").lower()
    return text.lower()


def merge_local(cfg: dict[str, Any], local: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the stored deltas over *cfg* (already including defaults)."""
    out = copy.deepcopy(cfg)
    for name, delta in (local or {}).items():
        if not isinstance(delta, dict) or name not in SENSOR_NAMES:
            continue
        section = _section(out, name)
        if isinstance(delta.get("set"), dict):
            section.update(delta["set"])
            if {"lat", "lon"} & set(delta["set"]) and "city" not in delta["set"]:
                section.pop("city", None)
        for field, items in (delta.get("add") or {}).items():
            if field == "docker" and section.get("docker") is True:
                continue  # already watching every container
            current = as_list(section.get(field)) if section.get(field) is not True else []
            seen = {ident(field, x) for x in current}
            for item in as_list(items):
                if ident(field, item) not in seen:
                    current.append(item)
                    seen.add(ident(field, item))
            section[field] = current
        for field, items in (delta.get("remove") or {}).items():
            if section.get(field) is True:
                continue
            gone = {ident(field, x) for x in as_list(items)}
            section[field] = [x for x in as_list(section.get(field)) if ident(field, x) not in gone]
        out[name] = section
    return out


def effective(cfg: dict[str, Any] | None, local: dict[str, Any] | None) -> dict[str, Any]:
    base = dict(cfg or {})
    if base.get("enabled", True) is False:
        return base  # the master switch beats anything said out loud
    return merge_local(with_defaults(base), local)


# --------------------------------------------------------------------------- #
# Editing the deltas
# --------------------------------------------------------------------------- #


def _delta(local: dict[str, Any], name: str) -> dict[str, Any]:
    return local.setdefault(name, {})


def _drop(lst: list[Any], field: str, key: str) -> bool:
    before = len(lst)
    lst[:] = [x for x in lst if ident(field, x) != key]
    return len(lst) != before


def add_item(local: dict[str, Any], name: str, field: str, item: Any) -> None:
    delta = _delta(local, name)
    key = ident(field, item)
    _drop(delta.setdefault("remove", {}).setdefault(field, []), field, key)
    adds = delta.setdefault("add", {}).setdefault(field, [])
    if all(ident(field, x) != key for x in adds):
        adds.append(item)
    delta.setdefault("set", {})["enabled"] = True
    _tidy(local, name)


def remove_item(local: dict[str, Any], name: str, field: str, item: Any) -> None:
    delta = _delta(local, name)
    key = ident(field, item)
    _drop(delta.setdefault("add", {}).setdefault(field, []), field, key)
    removes = delta.setdefault("remove", {}).setdefault(field, [])
    if all(ident(field, x) != key for x in removes):
        removes.append(item)
    _tidy(local, name)


def set_values(local: dict[str, Any], name: str, values: dict[str, Any]) -> None:
    _delta(local, name).setdefault("set", {}).update(values)
    _tidy(local, name)


def _tidy(local: dict[str, Any], name: str) -> None:
    delta = local.get(name) or {}
    for part in ("add", "remove"):
        fields = {f: v for f, v in (delta.get(part) or {}).items() if v}
        if fields:
            delta[part] = fields
        else:
            delta.pop(part, None)
    if not delta:
        local.pop(name, None)


def contains(cfg: dict[str, Any], name: str, field: str, item: Any) -> bool:
    section = cfg.get(name)
    if not isinstance(section, dict) or section.get("enabled") is False:
        return False
    value = section.get(field)
    if value is True:
        return True
    return ident(field, item) in {ident(field, x) for x in as_list(value)}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

_NAME_RE = re.compile(r"^[\w][\w .+\-]{0,63}$", re.UNICODE)
_CITY_RE = re.compile(r"^[^\W\d_][\w .,'\-]{0,79}$", re.UNICODE)


def clean_url(text: str, *, what: str = "página") -> str:
    """A safe http(s) URL from what the model passed, or :class:`Invalid`.

    A bare host ("example.com") gets ``https://``. Plain ``http://`` is
    allowed (dev servers), and so is localhost and the LAN: watching your own
    dev server is a legitimate request, and every URL here is one the user
    named out loud. What is refused is anything that is not a web address
    (``file:``, ``ftp:``…) and credentials in the URL, which would end up
    stored in plain text.
    """
    raw = (text or "").strip().strip("<>\"'")
    if not raw or len(raw) > 2048 or any(c.isspace() for c in raw):
        raise Invalid(f"Esa dirección de {what} no me sirve.")
    if "://" not in raw:
        if ":" in raw.split("/", 1)[0] and not re.match(r"^[\w.\-]+:\d+", raw):
            raise Invalid("Sólo vigilo direcciones web, http o https.")
        raw = ("http://" if _is_local(raw.split("/", 1)[0].split(":")[0]) else "https://") + raw
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise Invalid("Sólo vigilo direcciones web, http o https.")
    if parsed.username or parsed.password:
        raise Invalid("No guardo direcciones con usuario y contraseña adentro.")
    host = parsed.hostname or ""
    if not host or not re.match(r"^[A-Za-z0-9.\-\[\]:_]+$", host):
        raise Invalid(f"Esa dirección de {what} no me sirve.")
    try:
        port = parsed.port
    except ValueError:
        raise Invalid("Ese puerto no existe.") from None
    if port is not None and not 1 <= port <= 65535:
        raise Invalid("Ese puerto no existe.")
    if "." not in host and not _is_local(host):
        raise Invalid(f"Esa dirección de {what} no me sirve.")
    return raw


def _is_local(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in ("localhost", "") or host.endswith(".local"):
        return True
    try:
        return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def clean_port(text: Any) -> str:
    """"3000", "puerto 3000", "localhost:3000" -> "3000" / "host:3000"."""
    raw = str(text or "").strip().lower()
    raw = re.sub(r"^(el\s+)?puerto\s+", "", raw)
    host, _, port = raw.rpartition(":")
    try:
        number = int(port.strip())
    except ValueError:
        raise Invalid("Decime el número de puerto, del 1 al 65535.") from None
    if not 1 <= number <= 65535:
        raise Invalid("Ese puerto no existe: va del 1 al 65535.")
    host = host.strip()
    if host in ("", "localhost", "127.0.0.1"):
        return str(number)
    if not re.match(r"^[a-z0-9.\-]{1,253}$", host):
        raise Invalid("Ese host no me sirve.")
    return f"{host}:{number}"


def clean_name(text: Any, what: str) -> str:
    name = " ".join(str(text or "").split())
    if not _NAME_RE.match(name):
        raise Invalid(f"Ese nombre de {what} no me sirve.")
    return name


def clean_city(text: Any) -> str:
    city = " ".join(str(text or "").split())
    if not _CITY_RE.match(city):
        raise Invalid("No entendí la ciudad.")
    return city


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def load_local(path: Path) -> dict[str, Any]:
    """The stored deltas, or {} (a missing or broken file is not fatal)."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        log.warning("%s is unreadable; ignoring the spoken sensor changes", path, exc_info=True)
        return {}
    sensors = data.get("sensors") if isinstance(data, dict) else None
    return sensors if isinstance(sensors, dict) else {}


def save_local(path: Path, local: dict[str, Any]) -> None:
    """Atomic write: a crash mid-save must not lose what was already there."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "sensors": local}, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode(city: str, fetch_json: Callable[[str], Any] | None = None) -> dict[str, Any] | None:
    """{name, lat, lon, country} for *city* via Open-Meteo, or None if unknown.

    Raises OSError when the service cannot be reached (the caller says so).
    """
    if fetch_json is None:
        from wawi.sensors import net

        fetch_json = lambda url: net.fetch_json(url, timeout=8.0)  # noqa: E731
    url = GEOCODE_URL + "?" + urlencode({"name": city, "count": 1, "language": "es"})
    results = (fetch_json(url) or {}).get("results") or []
    if not results:
        return None
    hit = results[0]
    return {
        "name": str(hit.get("name") or city),
        "lat": round(float(hit["latitude"]), 4),
        "lon": round(float(hit["longitude"]), 4),
        "country": str(hit.get("country") or ""),
    }
