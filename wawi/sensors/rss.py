"""RSS / Atom feeds: new items become ``rss.item`` events.

The first successful read of a feed only marks what is already there as
seen. Without that, adding a feed would dump its last 50 headlines on the
user at once — the opposite of a friendly pet.

Parsing is stdlib ``xml.etree``. The expat bundled with Python 3.12 caps
entity amplification (billion laughs), and bodies are size-capped in
:mod:`wawi.sensors.net`, so a hostile feed can at worst be ignored.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Any, Callable

from wawi.reactions import Event
from wawi.sensors import net
from wawi.sensors.base import Sensor, as_list

ATOM = "{http://www.w3.org/2005/Atom}"
#: Remembered ids per feed. Feeds show the latest ~20-100 items; this keeps
#: an item that drops off and comes back from looking new.
SEEN_LIMIT = 1000


def _text(el: ET.Element | None) -> str:
    return " ".join((el.text or "").split()) if el is not None else ""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_feed(body: bytes | str) -> tuple[str, list[dict[str, str]]]:
    """Return (feed title, items newest-first as {id, title, link})."""
    root = ET.fromstring(body)
    items: list[dict[str, str]] = []
    feed_title = ""
    if _local(root.tag) == "feed":  # Atom
        feed_title = _text(root.find(f"{ATOM}title"))
        for entry in root.findall(f"{ATOM}entry"):
            link = ""
            for l in entry.findall(f"{ATOM}link"):
                if l.get("rel", "alternate") == "alternate":
                    link = l.get("href", "")
                    break
            title = _text(entry.find(f"{ATOM}title"))
            ident = _text(entry.find(f"{ATOM}id")) or link or title
            items.append({"id": ident, "title": title, "link": link})
    else:  # RSS 2.0 / 0.9x / RDF: <item> anywhere under the root
        channel = root.find("channel")
        feed_title = _text(channel.find("title")) if channel is not None else ""
        for item in root.iter():
            if _local(item.tag) != "item":
                continue
            fields = {_local(child.tag): child for child in item}
            title = _text(fields.get("title"))
            link = _text(fields.get("link"))
            ident = _text(fields.get("guid")) or link or title
            items.append({"id": ident, "title": title, "link": link})
    return feed_title, [i for i in items if i["id"]]


class RssSensor(Sensor):
    name = "rss"
    default_interval = 900.0
    min_interval = 120.0

    def __init__(self, config: dict[str, Any] | None = None, fetch: Callable[..., net.Response] | None = None) -> None:
        super().__init__(config)
        self._fetch = fetch or net.fetch
        self.feeds: list[tuple[str, str]] = []
        for item in as_list(self.config.get("feeds")):
            if isinstance(item, dict):
                url, name = str(item.get("url", "")), str(item.get("name", ""))
            else:
                url, name = str(item), ""
            if url.startswith(("http://", "https://")):
                self.feeds.append((url, name))
        self.max_per_poll = int(self.config.get("max_per_poll", 10))
        self._seen: dict[str, OrderedDict[str, None]] = {}
        self.new_count = 0

    def poll(self) -> list[Event]:
        events: list[Event] = []
        failures = 0
        for url, name in self.feeds:
            resp = self._fetch(url, timeout=10.0, method="GET")
            if resp.status != 200 or not resp.body:
                failures += 1
                continue
            try:
                feed_title, items = parse_feed(resp.body)
            except ET.ParseError:
                failures += 1
                continue
            seen = self._seen.get(url)
            first = seen is None
            if first:
                seen = self._seen[url] = OrderedDict()
            fresh = [i for i in items if i["id"] not in seen]
            for item in items:
                seen[item["id"]] = None
            while len(seen) > SEEN_LIMIT:
                seen.popitem(last=False)
            if first:
                continue
            for item in fresh[: self.max_per_poll]:
                digest = hashlib.sha1(item["id"].encode("utf-8")).hexdigest()[:12]
                events.append(Event(
                    "rss.item", "info", f"rss:{digest}", item["title"],
                    {"title": item["title"] or "(sin título)", "link": item["link"],
                     "feed": name or feed_title or url},
                ))
        if self.feeds and failures == len(self.feeds):
            # Every feed failed: let the runner back off instead of retrying
            # on schedule against a network that is not there.
            raise OSError("rss: no feed could be read")
        self.new_count += len(events)
        return events

    def inherit(self, old: Sensor) -> None:
        if not isinstance(old, RssSensor):
            return
        urls = {u for u, _ in self.feeds}
        # A kept feed must not re-announce what it already showed; a NEW feed
        # starts unseen, so its first read is the usual silent baseline.
        self._seen = {u: seen for u, seen in old._seen.items() if u in urls}
        self.new_count = old.new_count

    def summary(self) -> str | None:
        if not self.feeds or not self.new_count:
            return None
        n = self.new_count
        return "llegó 1 noticia nueva" if n == 1 else f"llegaron {n} noticias nuevas"
