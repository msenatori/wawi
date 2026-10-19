"""Wawi's senses: small pollers that notice things and emit Events.

Built-in sensors (enabled in ``config.yaml`` → ``sensors:``). When that
section names no sensor at all, a safe default set runs: ``system`` (disk +
battery) and ``clock`` (day parts + breaks) — no personal input, no network.
``sensors: {defaults: false}`` opts out, ``enabled: false`` turns all off.

What Wawi watches can also be changed by voice ("vigilá example.com"); those
changes live in a gitignored ``sensors_local.json`` merged over the config
(see :mod:`wawi.sensors.live`) and reach the running sensors immediately.

========== ==============================================================
web        URLs answering? ``web.down`` / ``web.up``
system     disk, battery, memory pressure, CPU load/heat
services   processes, TCP ports, Docker containers
weather    Open-Meteo forecast: rain soon, hot, cold, sunny
rss        new feed items
clock      day parts and "you've been at it for hours" → ``break.suggest``
========== ==============================================================

User plugins
------------
With ``sensors.user_plugins: true``, every ``*.py`` in ``sensors.plugins_dir``
(default ``~/.wawi/plugins``) is imported at startup. A plugin exposes either

- ``SENSOR``: a :class:`~wawi.sensors.base.Sensor` subclass or instance, or
- ``make_sensor(config) -> Sensor``.

Its config is ``sensors.plugins.<file stem>`` (``enabled: false`` skips it).
Plugins are ordinary Python running inside Wawi with your permissions, which
is why loading them is opt-in and files writable by other users are refused.

Example ``~/.wawi/plugins/hello.py``::

    from wawi.reactions import Event
    from wawi.sensors.base import Sensor

    class Hello(Sensor):
        name = "hello"
        default_interval = 3600
        def poll(self):
            return [Event("hello.ping", "info", "hello", "Hola")]

    SENSOR = Hello
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import os
import stat
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from wawi.reactions import BUS, Reaction, ReactionBus
from wawi.sensors import live
from wawi.sensors.base import Sensor
from wawi.sensors.clock import ClockSensor
from wawi.sensors.rss import RssSensor
from wawi.sensors.runner import SensorRunner
from wawi.sensors.services import ServicesSensor
from wawi.sensors.system import SystemSensor
from wawi.sensors.weather import WeatherSensor
from wawi.sensors.web import WebSensor

log = logging.getLogger(__name__)

REGISTRY: dict[str, type[Sensor]] = {
    "web": WebSensor,
    "system": SystemSensor,
    "services": ServicesSensor,
    "weather": WeatherSensor,
    "rss": RssSensor,
    "clock": ClockSensor,
}

DEFAULT_PLUGINS_DIR = "~/.wawi/plugins"

_lock = threading.RLock()
_RUNNER: SensorRunner | None = None
_REFLEXES: Any = None
#: What the running sensors were built from: config.yaml's section, the
#: spoken deltas, and the merge of the two (with defaults).
_BASE: dict[str, Any] = {}
_LOCAL: dict[str, Any] = {}
_LOCAL_PATH: Path = Path(live.DEFAULT_LOCAL_FILE)
_EFFECTIVE: dict[str, Any] = {}
_BUS: ReactionBus = BUS


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any] | None:
    """The config for one sensor, or None when it is not enabled.

    ``web: {enabled: true, urls: [...]}`` and the shorthand ``web: true``
    both enable it; a missing section or ``enabled: false`` does not.
    """
    raw = cfg.get(name)
    if raw is True:
        return {}
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    return raw


def build_sensors(cfg: dict[str, Any] | None) -> list[Sensor]:
    cfg = dict(cfg or {})
    if cfg.get("enabled", True) is False:
        return []
    sensors: list[Sensor] = []
    for name, cls in REGISTRY.items():
        section = _section(cfg, name)
        if section is None:
            continue
        try:
            sensors.append(cls(section))
        except Exception:
            log.warning("Sensor %s misconfigured; skipped", name, exc_info=True)
    if cfg.get("user_plugins", False):
        sensors += load_user_plugins(
            Path(str(cfg.get("plugins_dir") or DEFAULT_PLUGINS_DIR)).expanduser(),
            cfg.get("plugins") or {},
        )
    return sensors


def _unsafe(path: Path) -> str | None:
    try:
        st = path.stat()
    except OSError as exc:
        return str(exc)
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return "writable by other users"
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        return "not owned by you"
    return None


def load_user_plugins(folder: Path, configs: dict[str, Any] | None = None) -> list[Sensor]:
    """Import ``folder/*.py`` and build one sensor per file. Never raises."""
    configs = configs or {}
    sensors: list[Sensor] = []
    if not folder.is_dir():
        return sensors
    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):
            continue
        conf = configs.get(path.stem) or {}
        if not isinstance(conf, dict):
            conf = {}
        if conf.get("enabled", True) is False:
            continue
        why = _unsafe(path)
        if why:
            log.warning("Sensor plugin %s refused: %s", path, why)
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"wawi_user_plugin_{path.stem}", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "make_sensor"):
                sensor = module.make_sensor(conf)
            else:
                target = getattr(module, "SENSOR", None)
                sensor = target(conf) if isinstance(target, type) else target
            if not isinstance(sensor, Sensor):
                log.warning("Sensor plugin %s exposes no SENSOR / make_sensor", path)
                continue
            sensors.append(sensor)
            log.info("Loaded sensor plugin %s (%s)", path.name, sensor.name)
        except Exception:
            log.warning("Sensor plugin %s failed to load", path, exc_info=True)
    return sensors


def start(cfg: Any, bus: ReactionBus = BUS, *, local_path: str | Path | None = None) -> SensorRunner | None:
    """Build sensors + reflexes from a Config and start polling. Idempotent.

    Returns None (and starts nothing) only when ``sensors.enabled`` is false.
    With nothing to watch it still starts an idle runner, so "vigilá
    example.com" can add the first sensor without a restart.
    """
    global _RUNNER, _REFLEXES, _BASE, _LOCAL, _LOCAL_PATH, _EFFECTIVE, _BUS
    with _lock:
        if _RUNNER is not None:
            return _RUNNER
        base = dict(getattr(cfg, "sensors", None) or {})
        if base.get("enabled", True) is False:
            return None
        from wawi import paths as _paths

        path = (Path(str(local_path)).expanduser() if local_path
                else _paths.resolve(base.get("local_file") or live.DEFAULT_LOCAL_FILE))
        local = live.load_local(path)
        merged = live.effective(base, local)
        sensors = build_sensors(merged)
        from wawi.reflexes import Reflexes

        reflex_cfg = getattr(cfg, "reflexes", None) or {}
        hooks = []
        if reflex_cfg.get("enabled", True) is not False:
            _REFLEXES = Reflexes(reflex_cfg, bus=bus)
            _REFLEXES.set_composer("clock.morning", _morning_bubble)
            _REFLEXES.attach()
            hooks.append(_REFLEXES.tick)
        _BASE, _LOCAL, _LOCAL_PATH, _EFFECTIVE, _BUS = base, local, path, merged, bus
        _RUNNER = SensorRunner(sensors, bus, hooks=hooks)
        _RUNNER.start()
        return _RUNNER


def _morning_bubble(_event: Any) -> str:
    """The clock.morning bubble, from what the other sensors already know."""
    from wawi.sensors.morning import compose

    runner = _RUNNER
    return compose(runner.sensors if runner is not None else [])


def stop() -> None:
    global _RUNNER, _REFLEXES
    with _lock:
        runner, _RUNNER = _RUNNER, None
        reflexes, _REFLEXES = _REFLEXES, None
    if reflexes is not None:
        reflexes.detach()
    if runner is not None:
        runner.stop()


def status(runner: SensorRunner | None = None) -> str:
    """Short Spanish summary of what the sensors know, for "¿cómo está todo?".

    Read-only: it reports the last poll's state and never triggers I/O.
    """
    runner = runner if runner is not None else _RUNNER
    if runner is None:
        return "No estoy vigilando nada: los sensores están apagados."
    parts = []
    for sensor in runner.sensors:
        try:
            text = sensor.summary()
        except Exception:
            log.debug("summary() failed for %s", sensor.name, exc_info=True)
            text = None
        if text:
            parts.append(text.strip().rstrip("."))
    if not parts:
        return "Todavía no tengo novedades: recién empiezo a mirar."
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


# --------------------------------------------------------------------------- #
# Live reconfiguration ("vigilá example.com")
# --------------------------------------------------------------------------- #

#: (ok, sentence to say, data for the model)
Outcome = tuple[bool, str, dict[str, Any]]

_OFF: Outcome = (False, "Tengo los sensores apagados en la configuración.", {"error": "sensors_off"})


def _apply(local: dict[str, Any]) -> None:
    """Persist *local*, then rebuild only the sensors whose section changed."""
    global _LOCAL, _EFFECTIVE
    runner = _RUNNER
    if runner is None:
        return
    merged = live.effective(_BASE, local)
    live.save_local(_LOCAL_PATH, local)
    for name, cls in REGISTRY.items():
        old, new = _section(_EFFECTIVE, name), _section(merged, name)
        if old == new:
            continue
        if new is None:
            runner.remove(name)
            continue
        try:
            runner.replace(cls(new))
        except Exception:
            log.warning("Sensor %s misconfigured after a change; removed", name, exc_info=True)
            runner.remove(name)
    _LOCAL, _EFFECTIVE = local, merged


def _cancel(key: str) -> None:
    # Whatever she was doing about the thing (a firefighter run) stops with it.
    _BUS.push(Reaction(cancel_key=key, priority=0))


def _label(kind: str, item: str) -> str:
    if kind in ("web", "feed"):
        parsed = urlparse(item)
        host = parsed.hostname or item
        if parsed.port and host in ("localhost", "127.0.0.1"):
            return f"{host}:{parsed.port}"
        return host
    if kind == "port":
        return f"el puerto {item}"
    if kind == "container":
        return f"el contenedor {item}"
    if kind == "process":
        return f"el proceso {item}"
    return item


def _clean(kind: str, target: str) -> str:
    if kind == "web":
        return live.clean_url(target, what="página")
    if kind == "feed":
        return live.clean_url(target, what="feed")
    if kind == "port":
        return live.clean_port(target)
    if kind == "container":
        return live.clean_name(target, "contenedor")
    if kind == "process":
        return live.clean_name(target, "proceso")
    raise live.Invalid("No sé vigilar eso.")


def _stored(kind: str, item: str) -> Any:
    # Local ports are stored as plain numbers, the way config.yaml writes them.
    return int(item) if kind == "port" and item.isdigit() else item


def _docker_all() -> bool:
    return (_EFFECTIVE.get("services") or {}).get("docker") is True


def watch(kind: str, target: str, *, geocode: Any = None) -> Outcome:
    """Start watching *target*. Kinds: web, feed, port, container, process, weather."""
    kind = (kind or "").strip().lower()
    with _lock:
        if _RUNNER is None:
            return _OFF
        if kind not in live.ALL_KINDS:
            return False, "No sé vigilar eso.", {"error": "unknown_kind", "kinds": list(live.ALL_KINDS)}
        try:
            if kind == "weather":
                return _watch_weather(target, geocode)
            item = _clean(kind, target)
        except live.Invalid as exc:
            return False, str(exc), {"error": "invalid", "target": target}
        name, field = live.KINDS[kind]
        label = _label(kind, item)
        if kind == "container" and _docker_all():
            return True, "Ya vigilo todos tus contenedores.", {"already": True}
        if live.contains(_EFFECTIVE, name, field, item):
            return True, f"Ya estaba vigilando {label}.", {"already": True, "target": item}
        local = copy.deepcopy(_LOCAL)
        live.add_item(local, name, field, _stored(kind, item))
        _apply(local)
        if kind == "feed":
            return True, f"Listo, sigo el feed de {label}.", {"watching": item, "kind": kind}
        return True, f"Listo, vigilo {label}.", {"watching": item, "kind": kind}


def _watch_weather(target: str, geocode: Any) -> Outcome:
    city = live.clean_city(target)
    try:
        place = live.geocode(city, geocode)
    except Exception:
        log.warning("Geocoding %r failed", city, exc_info=True)
        return False, "No pude buscar esa ciudad: ¿hay internet?", {"error": "geocode_failed"}
    if place is None:
        return False, f"No encontré {city}.", {"error": "city_not_found", "city": city}
    local = copy.deepcopy(_LOCAL)
    live.set_values(local, "weather", {"enabled": True, "city": place["name"],
                                       "lat": place["lat"], "lon": place["lon"]})
    _apply(local)
    return True, f"Listo, te aviso del clima de {place['name']}.", {"weather": place}


def unwatch(kind: str, target: str = "") -> Outcome:
    """Stop watching *target* (also things that came from config.yaml)."""
    kind = (kind or "").strip().lower()
    with _lock:
        if _RUNNER is None:
            return _OFF
        if kind == "weather":
            if _section(_EFFECTIVE, "weather") is None:
                return True, "No estaba mirando el clima.", {"already": True}
            local = copy.deepcopy(_LOCAL)
            live.set_values(local, "weather", {"enabled": False})
            _apply(local)
            return True, "Listo, dejo de mirar el clima.", {"weather": None}
        if kind not in live.KINDS:
            return False, "No sé qué dejar de vigilar.", {"error": "unknown_kind"}
        try:
            item = _clean(kind, target)
        except live.Invalid as exc:
            return False, str(exc), {"error": "invalid", "target": target}
        name, field = live.KINDS[kind]
        label = _label(kind, item)
        if kind == "container" and _docker_all():
            return False, "Vigilo todos los contenedores; eso se cambia en la configuración.", \
                {"error": "docker_all"}
        if not live.contains(_EFFECTIVE, name, field, item):
            return True, f"No estaba vigilando {label}.", {"already": True}
        local = copy.deepcopy(_LOCAL)
        live.remove_item(local, name, field, _stored(kind, item))
        _apply(local)
        if kind == "web":
            # Every spelling the sensor may have keyed it under.
            for url in {item, item.rstrip("/"), item.rstrip("/") + "/"}:
                _cancel(f"web:{url}")
        else:
            svc = kind if kind != "feed" else ""
            if svc:
                shown = f"puerto {item}" if kind == "port" and item.isdigit() else item
                _cancel(f"service:{svc}:{shown}")
        return True, f"Listo, dejé de vigilar {label}.", {"unwatched": item, "kind": kind}


def _urls(items: Any) -> list[str]:
    return [str(i.get("url")) if isinstance(i, dict) else str(i) for i in live.as_list(items)]


def watching() -> Outcome:
    """"¿Qué estás vigilando?" — from the effective config, never I/O."""
    with _lock:
        if _RUNNER is None:
            return _OFF
        cfg = _EFFECTIVE
    things: list[str] = []
    data: dict[str, Any] = {}

    web = _section(cfg, "web")
    if web is not None:
        data["web"] = _urls(web.get("urls"))
        things += [_label("web", u) for u in data["web"]]
    svc = _section(cfg, "services")
    if svc is not None:
        ports = [str(p) for p in live.as_list(svc.get("ports"))]
        procs = [str(p) for p in live.as_list(svc.get("processes"))]
        docker = svc.get("docker", False)
        data["services"] = {"ports": ports, "processes": procs, "docker": docker}
        things += [_label("port", p) for p in ports] + [_label("process", p) for p in procs]
        if docker is True:
            things.append("tus contenedores")
        elif docker:
            things += [_label("container", str(c)) for c in live.as_list(docker)]
    rss = _section(cfg, "rss")
    if rss is not None:
        data["feeds"] = _urls(rss.get("feeds"))
        if data["feeds"]:
            n = len(data["feeds"])
            things.append("un feed de noticias" if n == 1 else f"{n} feeds de noticias")
    weather = _section(cfg, "weather")
    if weather is not None:
        data["weather"] = weather.get("city") or "tu zona"
        things.append(f"el clima de {data['weather']}")
    system = _section(cfg, "system")
    if system is not None:
        data["system"] = True
        things.append("el disco y la batería" if system.get("battery", True) else "el disco")
    if _section(cfg, "clock") is not None:
        data["clock"] = True
        things.append("la hora y tus pausas")
    if not things:
        return True, "Por ahora no estoy vigilando nada.", data
    sentence = things[0] if len(things) == 1 else ", ".join(things[:-1]) + " y " + things[-1]
    return True, f"Vigilo {sentence}.", data
