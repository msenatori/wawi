"""Website uptime: is each configured URL answering?

Down means no HTTP answer at all (DNS, refused, TLS, timeout) or a 5xx, for
``fail_after`` consecutive checks. A 4xx counts as UP: the server is alive and
answering, and a 404 on a health URL is a config problem, not an outage.
"""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlparse

from wawi.reactions import Event
from wawi.sensors import net
from wawi.sensors.base import Debouncer, Sensor, as_list


class WebSensor(Sensor):
    name = "web"
    default_interval = 120.0

    def __init__(self, config: dict[str, Any] | None = None, fetch: Callable[..., net.Response] | None = None) -> None:
        super().__init__(config)
        self._fetch = fetch or net.fetch
        self.timeout = float(self.config.get("timeout", 10.0))
        self.method = str(self.config.get("method", "GET")).upper()
        self.urls: list[str] = []
        for item in as_list(self.config.get("urls")):
            url = item.get("url") if isinstance(item, dict) else item
            if url and str(url).startswith(("http://", "https://")):
                self.urls.append(str(url))
        self._deb = Debouncer(
            fail_after=int(self.config.get("fail_after", 2)),
            recover_after=int(self.config.get("recover_after", 1)),
            alert_initial=True,
        )
        self._last: dict[str, net.Response] = {}

    def poll(self) -> list[Event]:
        events: list[Event] = []
        for url in self.urls:
            resp = self._fetch(url, timeout=self.timeout, method=self.method)
            self._last[url] = resp
            bad = resp.status is None or resp.status >= 500
            change = self._deb.update(url, bad)
            if change is None:
                continue
            host = urlparse(url).hostname or url
            data = {
                "url": url,
                "host": host,
                "status": resp.status,
                "latency_ms": resp.latency_ms,
                "error": resp.error,
            }
            if change == "down":
                events.append(Event("web.down", "urgent", f"web:{url}", f"Se cayó {host}", data))
            else:
                events.append(Event("web.up", "notice", f"web:{url}", f"{host} volvió", data))
        return events

    def inherit(self, old: Sensor) -> None:
        if not isinstance(old, WebSensor):
            return
        deb = old._deb.copy()
        deb.forget(self.urls)
        deb.fail_after, deb.recover_after = self._deb.fail_after, self._deb.recover_after
        self._deb = deb
        self._last = {u: r for u, r in old._last.items() if u in self.urls}

    @property
    def checked(self) -> bool:
        """True once every URL has an answer (or lack of one) on record."""
        return bool(self.urls) and all(u in self._last for u in self.urls)

    def down(self) -> list[str]:
        return [u for u, st in self._deb.states().items() if st == "down"]

    def summary(self) -> str | None:
        if not self.urls:
            return None
        if not self._last:
            return "todavía no revisé tus webs"
        down = self.down()
        if not down:
            return "todas tus webs andan" if len(self.urls) > 1 else f"{_host(self.urls[0])} anda"
        hosts = ", ".join(_host(u) for u in down)
        return f"está caída {hosts}" if len(down) == 1 else f"están caídas {hosts}"


def _host(url: str) -> str:
    return urlparse(url).hostname or url
