"""Weather from Open-Meteo (free, no API key, no account).

Every event fires at most once per day per kind (rain once per rain spell),
so a forecast refreshed every half hour does not make Wawi put the umbrella
on twelve times.

Times come back in the location's local time (``timezone=auto``), and the
"now" used for the look-ahead is the API's own ``current.time``, which keeps
the logic independent of the Mac's clock and makes fixtures deterministic.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable
from urllib.parse import urlencode

from wawi.reactions import Event
from wawi.sensors import net
from wawi.sensors.base import Sensor

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

#: WMO weather codes that read as "sunny" (clear, mainly clear).
SUNNY_CODES = {0, 1}


def forecast_url(lat: float, lon: float) -> str:
    return FORECAST_URL + "?" + urlencode({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "current": "temperature_2m,weather_code",
        "hourly": "temperature_2m,precipitation_probability",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
        "timezone": "auto",
        "forecast_days": 2,
    })


def _parse_time(text: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text))
    except ValueError:
        return None


def _num(seq: Any, i: int) -> float | None:
    try:
        value = seq[i]
    except (IndexError, TypeError, KeyError):
        return None
    return None if value is None else float(value)


def parse_forecast(payload: dict[str, Any], *, hours: int = 3, threshold: float = 60.0) -> dict[str, Any]:
    """Distil the Open-Meteo answer into the handful of facts the sensor uses.

    Returns {now, temp, code, today: {date,max,min,rain,code},
    tomorrow: {...} or None, rain_at: "HH:MM" or None, rain_prob}.
    """
    current = payload.get("current") or {}
    now = _parse_time(current.get("time")) or datetime.now()
    out: dict[str, Any] = {
        "now": now,
        "temp": _num([current.get("temperature_2m")], 0),
        "code": current.get("weather_code"),
        "rain_at": None,
        "rain_prob": None,
    }
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    probs = hourly.get("precipitation_probability") or []
    for i, t in enumerate(times):
        at = _parse_time(t)
        if at is None:
            continue
        ahead = (at - now).total_seconds() / 3600.0
        # The hour slot containing "now" counts (ahead > -1).
        if ahead <= -1 or ahead > hours:
            continue
        prob = _num(probs, i)
        if prob is not None and prob >= threshold:
            out["rain_at"] = at.strftime("%H:%M")
            out["rain_prob"] = int(prob)
            break
    daily = payload.get("daily") or {}
    days = []
    for i, d in enumerate(daily.get("time") or []):
        days.append({
            "date": str(d),
            "max": _num(daily.get("temperature_2m_max"), i),
            "min": _num(daily.get("temperature_2m_min"), i),
            "rain": _num(daily.get("precipitation_probability_max"), i),
            "code": _num(daily.get("weather_code"), i),
        })
    today = now.date().isoformat()
    out["today"] = next((d for d in days if d["date"] == today), days[0] if days else None)
    later = [d for d in days if d["date"] > today]
    out["tomorrow"] = later[0] if later else None
    return out


class WeatherSensor(Sensor):
    name = "weather"
    default_interval = 1800.0
    min_interval = 300.0  # Open-Meteo is free; be a polite guest.

    def __init__(self, config: dict[str, Any] | None = None, fetch_json: Callable[[str], Any] | None = None) -> None:
        super().__init__(config)
        c = self.config
        self._fetch_json = fetch_json or (lambda url: net.fetch_json(url, timeout=10.0))
        self.lat = c.get("lat", c.get("latitude"))
        self.lon = c.get("lon", c.get("longitude"))
        self.city = str(c.get("city") or "").strip()
        self.rain_hours = int(c.get("rain_hours", 3))
        self.rain_threshold = float(c.get("rain_threshold", 60))
        self.hot_c = float(c.get("hot_c", 30))
        self.cold_c = float(c.get("cold_c", 5))
        self.morning_hour = int(c.get("morning_hour", 8))
        self._fired: dict[str, str] = {}  # kind -> date it last fired
        self._raining = False
        self._last: dict[str, Any] | None = None

    def setup(self) -> None:
        if self.lat is not None and self.lon is not None:
            self.lat, self.lon = float(self.lat), float(self.lon)
            return
        if not self.city:
            raise ValueError("weather: set lat/lon or city")
        url = GEOCODE_URL + "?" + urlencode({"name": self.city, "count": 1, "language": "es"})
        results = (self._fetch_json(url) or {}).get("results") or []
        if not results:
            raise ValueError(f"weather: city {self.city!r} not found")
        self.lat, self.lon = float(results[0]["latitude"]), float(results[0]["longitude"])
        log.info("Weather: %s -> %.3f, %.3f", self.city, self.lat, self.lon)

    def _once_per_day(self, kind: str, date: str) -> bool:
        if self._fired.get(kind) == date:
            return False
        self._fired[kind] = date
        return True

    def poll(self) -> list[Event]:
        payload = self._fetch_json(forecast_url(float(self.lat), float(self.lon)))
        f = parse_forecast(payload, hours=self.rain_hours, threshold=self.rain_threshold)
        self._last = f
        now: datetime = f["now"]
        date = now.date().isoformat()
        today = f["today"] or {}
        events: list[Event] = []
        place = self.city or ""

        if f["rain_at"]:
            if not self._raining:
                self._raining = True
                events.append(Event(
                    "weather.rain_soon", "notice", f"weather:rain:{date}",
                    f"Va a llover a las {f['rain_at']}",
                    {"hour": f["rain_at"], "probability": f["rain_prob"], "place": place},
                ))
        else:
            # Re-arm only once the window is dry, so one spell = one event.
            self._raining = False

        # Daily facts wait for the morning, so a 3 a.m. restart doesn't open
        # with "hoy hace calor".
        if now.hour >= self.morning_hour:
            tmax, tmin = today.get("max"), today.get("min")
            if tmax is not None and tmax >= self.hot_c and self._once_per_day("hot", date):
                events.append(Event("weather.hot", "info", f"weather:hot:{date}", "Hoy hace calor",
                                    {"max": round(tmax), "min": _round(tmin), "place": place}))
            if tmin is not None and tmin <= self.cold_c and self._once_per_day("cold", date):
                events.append(Event("weather.cold", "info", f"weather:cold:{date}", "Hoy hace frío",
                                    {"min": round(tmin), "max": _round(tmax), "place": place}))
            code = f["code"] if f["code"] is not None else today.get("code")
            if (
                code is not None and int(code) in SUNNY_CODES
                and not f["rain_at"] and self._once_per_day("sunny", date)
            ):
                events.append(Event("weather.sunny", "info", f"weather:sunny:{date}", "Día de sol",
                                    {"temp": _round(f["temp"]), "max": _round(tmax), "place": place}))
        return events

    def inherit(self, old: Sensor) -> None:
        # Same place: keep the once-a-day memory so a reconfiguration does not
        # put the umbrella on twice. A new place starts fresh.
        if isinstance(old, WeatherSensor) and (old.city, old.lat, old.lon) == (self.city, self.lat, self.lon):
            self._fired, self._raining, self._last = dict(old._fired), old._raining, old._last

    def today(self) -> dict[str, Any] | None:
        """The last forecast's facts for the morning greeting, or None.

        {max, min, temp, rain_at} — numbers rounded; never does I/O.
        """
        f = self._last
        if not f:
            return None
        today = f.get("today") or {}
        return {
            "temp": _round(f.get("temp")),
            "max": _round(today.get("max")),
            "min": _round(today.get("min")),
            "rain_at": f.get("rain_at"),
            "rain": today.get("rain"),
        }

    def summary(self) -> str | None:
        f = self._last
        if not f:
            return None
        parts = []
        if f["temp"] is not None:
            parts.append(f"afuera hay {round(f['temp'])} °C")
        if f["rain_at"]:
            parts.append(f"va a llover a las {f['rain_at']}")
        tomorrow = f.get("tomorrow")
        if tomorrow and tomorrow.get("rain") is not None:
            parts.append("mañana llueve" if tomorrow["rain"] >= self.rain_threshold else "mañana no llueve")
        return ", ".join(parts) or None


def _round(value: float | None) -> int | None:
    return None if value is None else round(value)
