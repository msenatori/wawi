"""Curiosidades: now and then, while she idles, a tiny fact in her bubble.

The owner's idea: during the day, with nothing going on, Wawi shows one short
teaser ("¿Sabías que la berenjena se cosecha a fines del verano?") with a
"Ver más…" link that opens the board with a short article. Silent on purpose
-- the owner is often in a call -- so only the bubble and a wag.

Where the facts come from, alternating batch by batch:

* **Wikipedia en español**: the featured feed of the day (``onthisday`` --
  "un día como hoy" --, the featured article and the most read), topped up
  with random summaries that are not stubs. The teaser is the extract,
  shortened, never invented; Haiku may only rephrase it into one sentence.
* **Claude Haiku**: a one-shot, tool-less call (the mind's lean SDK path, see
  :mod:`wawi.mind.claude`) asking for well-established light facts as strict
  JSON. Never the live brain session: a batch of articles riding in that
  session's history would be re-sent on every later turn.

Batches are cached in a small local JSON file (gitignored) with the on/off
switch and the titles already used, refilled in the background when fewer
than a few remain, at most once a day. Nothing here touches AppKit: the panel
polls :meth:`CuriosityService.poll` from its render tick and draws the bubble.

The last curiosity reaches the brain as ONE line prepended to the next user
turn (:meth:`CuriosityService.with_note`), never in the system prompt: the
live SDK client is keyed on that prompt, and changing it would reconnect and
throw the prompt cache away for a fact about eggplants.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from wawi import working as _working

log = logging.getLogger(__name__)

#: The link line under the teaser; the panel underlines it and makes it clickable.
LINK = "Ver más…"

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # One every ~hour at most, ± jitter, and the first one not right at start.
    "every_min": 60.0,
    "jitter": 0.35,
    "first_after_min": 20.0,
    # Never right after a conversation.
    "quiet_after_talk_s": 300.0,
    # How long the bubble stays up.
    "show_s": 25.0,
    # How long "contame más" can still refer to the last one.
    "recent_min": 20.0,
    "batch": 10,
    "refill_below": 3,
    # The cheapest, fastest model: a batch of trivia does not need more.
    "model": "claude-haiku-5-5",
    "timeout_s": 120.0,
    "local_file": "curiosities_local.json",
}

#: Between random summaries: Wikimedia answers a burst with 429.
_RANDOM_PAUSE_S = 1.0
TEASER_MAX = 110          # hard limit; the prompt asks for <= 90
TEASER_TARGET = 90
_USED_MAX = 400
_RETRY_AFTER_FAIL_S = 3600.0
_RETRY_WHEN_EMPTY_S = 600.0


@dataclass(frozen=True)
class CuriositySettings:
    enabled: bool = True
    every_min: float = 60.0
    jitter: float = 0.35
    first_after_min: float = 20.0
    quiet_after_talk_s: float = 300.0
    show_s: float = 25.0
    recent_min: float = 20.0
    batch: int = 10
    refill_below: int = 3
    model: str = "claude-haiku-5-5"
    timeout_s: float = 120.0
    local_file: str = "curiosities_local.json"

    @classmethod
    def from_dict(cls, raw: Any) -> "CuriositySettings":
        d = {**DEFAULTS, **(raw if isinstance(raw, dict) else {})}

        def num(key: str, lo: float, hi: float) -> float:
            try:
                return min(hi, max(lo, float(d[key])))
            except (TypeError, ValueError):
                return float(DEFAULTS[key])

        return cls(
            enabled=bool(d.get("enabled", True)),
            every_min=num("every_min", 5.0, 24 * 60.0),
            jitter=num("jitter", 0.0, 0.9),
            first_after_min=num("first_after_min", 0.0, 24 * 60.0),
            quiet_after_talk_s=num("quiet_after_talk_s", 0.0, 3600.0),
            show_s=num("show_s", 5.0, 120.0),
            recent_min=num("recent_min", 1.0, 240.0),
            batch=int(num("batch", 1, 30)),
            refill_below=int(num("refill_below", 0, 30)),
            model=str(d.get("model") or DEFAULTS["model"]),
            timeout_s=num("timeout_s", 10.0, 600.0),
            local_file=str(d.get("local_file") or ""),
        )


@dataclass(frozen=True)
class Curiosity:
    teaser: str
    title: str
    body: str                 # markdown, 2-4 short paragraphs
    image_query: str = ""
    source: str = "claude"    # "claude" | "wikipedia"
    url: str = ""
    #: ISO date the item is only valid on ("un día como hoy"), or "".
    day: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "Curiosity | None":
        if not isinstance(d, dict):
            return None
        try:
            item = cls(**{k: str(d.get(k) or "") for k in
                          ("teaser", "title", "body", "image_query", "source", "url", "day")})
        except TypeError:
            return None
        return item if item.teaser and item.title and item.body else None


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _key(title: str) -> str:
    """A title as compared for "already used": no accents, case or punctuation."""
    import unicodedata

    t = unicodedata.normalize("NFKD", title.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def shorten(text: str, limit: int = TEASER_TARGET) -> str:
    """*text* on one line, cut on a word boundary with an ellipsis."""
    t = _clean(text)
    if len(t) <= limit:
        return t
    cut = t[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:-–—(") + "…"


def first_sentence(text: str) -> str:
    """The first sentence of an extract ("El ornitorrinco es … Australia.")."""
    t = _clean(text)
    # A period after a lowercase letter or a closing bracket, then a capital.
    m = re.search(r"(?<=[a-záéíóúñ0-9\)\]])\.\s+(?=[A-ZÁÉÍÓÚÑ¿])", t)
    return t[: m.start() + 1] if m else t


def season(day: date) -> str:
    """The season in Argentina (southern hemisphere)."""
    return {12: "verano", 1: "verano", 2: "verano", 3: "otoño", 4: "otoño", 5: "otoño",
            6: "invierno", 7: "invierno", 8: "invierno"}.get(day.month, "primavera")


# --------------------------------------------------------------------------- #
# Claude batches
# --------------------------------------------------------------------------- #

GENERATE_SYSTEM = (
    "Escribís curiosidades cortas para Wawi, una perrita asistente de escritorio "
    "que habla en español rioplatense. Solo datos bien establecidos y "
    "verificables, livianos y simpáticos; nada dudoso, polémico, de salud ni "
    "leyendas urbanas. Si no estás seguro de un dato, no lo uses. Respondés "
    "SOLO con un array JSON válido, sin texto antes ni después."
)


def generation_prompt(n: int, today: date, avoid: list[str]) -> str:
    avoid_txt = "; ".join(avoid[-60:]) or "(ninguno)"
    return (
        f"Hoy es {today.isoformat()} en Argentina ({season(today)}). "
        f"Escribí {n} curiosidades variadas mezclando temas: comida y estaciones "
        "(hemisferio sur), animales (sobre todo perros), ciencia, historia, "
        "idioma y palabras, Argentina. Algunas pueden tener que ver con la "
        "estación o la fecha.\n"
        f"No repitas estos temas ya usados: {avoid_txt}.\n"
        "Cada elemento es un objeto con exactamente estas claves:\n"
        '- "teaser": UNA frase de hasta 90 caracteres, en rioplatense, que '
        'invite a saber más (ej.: "¿Sabías que la berenjena se cosecha a fines '
        'del verano?").\n'
        '- "title": título corto del tema (hasta 60 caracteres).\n'
        '- "body_markdown": 2 a 4 párrafos cortos en markdown que explican el '
        "dato, sin encabezados ni enlaces.\n"
        '- "image_query": 1 a 3 palabras para buscar una foto libre del tema '
        'en Wikimedia (ej.: "berenjena"), o "" si no aplica.\n'
        "Devolvé solo el array JSON."
    )


def parse_items(text: str, *, used: set[str] | None = None) -> list[Curiosity]:
    """Validate a model's JSON answer; bad or repeated items are dropped."""
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    seen = set(used or ())
    out: list[Curiosity] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        teaser = _clean(d.get("teaser"))
        title = _clean(d.get("title"))
        body = str(d.get("body_markdown") or "").strip()
        query = _clean(d.get("image_query"))[:80]
        if not (12 <= len(teaser) <= TEASER_MAX) or not (2 <= len(title) <= 80):
            continue
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        if not (60 <= len(body) <= 3000) or not 1 <= len(paragraphs) <= 6:
            continue
        if _key(title) in seen:
            continue
        seen.add(_key(title))
        out.append(Curiosity(teaser=teaser, title=title, body=body,
                             image_query=query, source="claude"))
    return out


def ask_claude(prompt: str, system: str, *, model: str, timeout: float) -> str:
    """One lean, tool-less SDK call on its own thread (wawi.mind.claude's path:
    subscription login, no CLAUDE.md, no MCP). Raises on failure."""
    from wawi.mind import claude as mc

    options = mc._options(model, system)
    return mc.run_in_thread(lambda: asyncio.run(mc._ask_async(prompt, options)), timeout) or ""


# --------------------------------------------------------------------------- #
# Wikipedia
# --------------------------------------------------------------------------- #

WIKI = "https://es.wikipedia.org/api/rest_v1"
MIN_EXTRACT = 300
#: Descriptions that mark the dull end of a random article: small places,
#: lists, taxonomy entries, disambiguations.
_STUB_WORDS = re.compile(
    r"\b(localidad|municipio|comuna|pueblo|aldea|villa|parroquia|distrito|"
    r"departamento|cant[oó]n|condado|barrio|corregimiento|pedan[ií]a|"
    r"especie de|g[eé]nero de|desambiguaci[oó]n|p[aá]gina de|anexo|lista de|"
    r"futbolista|episodio)\b", re.I)

REWRITE_SYSTEM = (
    "Reescribís un dato en UNA sola frase en español rioplatense de hasta 90 "
    "caracteres, que empiece con «¿Sabías que» y termine con «?». Usá solo lo "
    "que dice el texto, sin agregar ningún dato. Respondé solo la frase."
)


def _json_get(url: str) -> dict:
    """GET JSON through the board's own bounded, redirect-checked helper."""
    from wawi import images

    return images._json(url)


def is_stub(summary: dict) -> bool:
    """A summary not worth a curiosity: short, a disambiguation, a small place."""
    if not isinstance(summary, dict) or summary.get("type", "standard") != "standard":
        return True
    extract = _clean(summary.get("extract"))
    if len(extract) < MIN_EXTRACT:
        return True
    description = _clean(summary.get("description"))
    return bool(_STUB_WORDS.search(description) or _STUB_WORDS.search(first_sentence(extract)))


def _page_url(summary: dict) -> str:
    return str(((summary.get("content_urls") or {}).get("desktop") or {}).get("page") or "")


#: First words of an efeméride that read lowercase after "un día como hoy,"
#: ("Muere…", "El cardenal…"); anything else may be a name ("Henry Kissinger").
_LOWER_FIRST = frozenset("el la los las lo un una unos unas se en al del tras durante "
                         "nace nacen muere mueren es son fue fueron comienza comienzan "
                         "termina finaliza inicia".split())


def _lower_first(text: str) -> str:
    word = text.split(" ", 1)[0]
    if word.lower() in _LOWER_FIRST or re.search(r"(ó|aron|ieron)$", word):
        return text[:1].lower() + text[1:]
    return text


def _title(page: dict) -> str:
    """A page's readable title ("Juan Pablo II", not "Juan_Pablo_II")."""
    titles = page.get("titles") if isinstance(page.get("titles"), dict) else {}
    return _clean(titles.get("normalized") or str(page.get("title") or "").replace("_", " "))


def _no_asides(text: str) -> str:
    """Drop parentheticals: dates and spellings that only make a teaser long."""
    return _clean(re.sub(r"\s*\([^()]*\)", "", text))


def _teaser(candidate: str, extract: str, rewrite: Callable[[str], str] | None) -> str:
    """*candidate* if it fits, else Haiku's one-sentence rephrase of the
    extract (validated), else *candidate* shortened. Never invents."""
    candidate = _clean(candidate)
    if len(candidate) <= TEASER_TARGET:
        return candidate
    if rewrite is not None:
        try:
            text = _clean(rewrite(f"Dato: {candidate}\n\nTexto: {extract[:1200]}"))
            text = text.strip("«»\"")
            if 12 <= len(text) <= TEASER_MAX:
                return text
        except Exception as exc:
            log.info("Curiosity teaser rewrite failed: %s", exc)
    return shorten(candidate)


def summary_item(summary: dict, rewrite: Callable[[str], str] | None = None) -> Curiosity | None:
    """A curiosity from a page summary (featured or random), or None."""
    title = _title(summary)
    extract = str(summary.get("extract") or "").strip()
    if not title or not extract:
        return None
    lead = _no_asides(first_sentence(extract))
    return Curiosity(
        teaser=_teaser(lead, extract, rewrite), title=title[:80],
        body=extract, image_query=title, source="wikipedia", url=_page_url(summary))


def onthisday_item(entry: dict, today: date,
                   rewrite: Callable[[str], str] | None = None) -> Curiosity | None:
    """"¿Sabías que un día como hoy, en 1923, …?" -- valid only today."""
    text = _clean(entry.get("text")).rstrip(".")
    short = _no_asides(text)
    year = entry.get("year")
    pages = [p for p in entry.get("pages") or [] if isinstance(p, dict)]
    page = next((p for p in pages if len(_clean(p.get("extract"))) >= MIN_EXTRACT), None)
    if not text or not isinstance(year, int) or page is None:
        return None
    title = _title(page)
    teaser = _teaser(f"¿Sabías que un día como hoy, en {year}, {_lower_first(short)}?",
                     text, rewrite)
    body = (f"**Un día como hoy, en {year}:** {text}.\n\n"
            + str(page.get("extract") or "").strip())
    return Curiosity(teaser=teaser, title=title[:80] or text[:80], body=body,
                     image_query=title, source="wikipedia", url=_page_url(page),
                     day=today.isoformat())


def wikipedia_batch(n: int, today: date, *, used: set[str] | None = None,
                    rewrite: Callable[[str], str] | None = None,
                    fetch: Callable[[str], dict] = _json_get,
                    rng: random.Random | None = None) -> list[Curiosity]:
    """Up to *n* curiosities from es.wikipedia: the day's featured feed first
    (efemérides, the featured article, the most read), then random summaries
    that are not stubs. Raises only when nothing at all could be fetched."""
    rng = rng or random.Random()
    seen = set(used or ())
    out: list[Curiosity] = []
    errors = 0

    def add(item: Curiosity | None) -> None:
        if item is not None and len(out) < n and _key(item.title) not in seen:
            seen.add(_key(item.title))
            out.append(item)

    try:
        feed = fetch(f"{WIKI}/feed/featured/{today:%Y/%m/%d}")
    except Exception as exc:
        log.info("Wikipedia featured feed unavailable: %s", exc)
        feed, errors = {}, errors + 1
    days = [e for e in feed.get("onthisday") or [] if isinstance(e, dict)]
    rng.shuffle(days)
    for entry in days[: max(1, n // 2)]:
        add(onthisday_item(entry, today, rewrite))
    tfa = feed.get("tfa")
    if isinstance(tfa, dict) and not is_stub(tfa):
        add(summary_item(tfa, rewrite))
    read = [a for a in ((feed.get("mostread") or {}).get("articles") or [])
            if isinstance(a, dict) and not is_stub(a)]
    for article in read[:3]:
        add(summary_item(article, rewrite))
    tries = 0
    while len(out) < n and tries < 3 * n + 5:
        tries += 1
        try:
            summary = fetch(f"{WIKI}/page/random/summary")
        except Exception as exc:
            errors += 1
            if errors >= 3:
                log.info("Wikipedia random summary unavailable: %s", exc)
                break
            continue
        if not is_stub(summary):
            add(summary_item(summary, rewrite))
        time.sleep(_RANDOM_PAUSE_S)
    if not out and errors:
        raise OSError("Wikipedia no disponible")
    return out


# --------------------------------------------------------------------------- #
# The article on the board
# --------------------------------------------------------------------------- #


def _safe_markdown(text: str) -> str:
    """Generated text never brings its own HTML onto the board."""
    import markdown

    return markdown.markdown(html.escape(text, quote=False))


def article_html(item: Curiosity, card: Any = None) -> tuple[str, str]:
    """(content, format) for Board.show: the image-card layout when a free
    image was found (wawi/images.py), plain markdown otherwise."""
    source = ""
    if item.url:
        source = f"[Fuente: Wikipedia]({item.url})" if item.source == "wikipedia" else f"[Fuente]({item.url})"
    if card is None or not str(getattr(card, "data_uri", "")).startswith((
            "data:image/jpeg;base64,", "data:image/png;base64,",
            "data:image/gif;base64,", "data:image/webp;base64,")):
        md = f"# {item.title}\n\n{item.body}"
        if source:
            md += f"\n\n---\n\n{source}"
        return md, "markdown"
    title = html.escape(item.title)
    credit = (f"<p>Crédito: {html.escape(card.credit)}</p>"
              f"<p>Licencia: {html.escape(card.license)}</p>")
    links = []
    if item.url:
        links.append(f"<a href='{html.escape(item.url, quote=True)}'>"
                     + ("Fuente: Wikipedia" if item.source == "wikipedia" else "Fuente") + "</a>")
    if card.source_url:
        links.append(f"<a href='{html.escape(card.source_url, quote=True)}'>Ver la foto</a>")
    body = ("<article class='image-card'><figure class='image-frame'>"
            f"<img src='{card.data_uri}' alt='{title}'></figure>"
            f"<section class='image-details'><h1>{title}</h1>{_safe_markdown(item.body)}"
            f"<div class='image-credit'>{credit}<p>{' · '.join(links)}</p></div>"
            "</section></article>")
    return body, "html"


# --------------------------------------------------------------------------- #
# Local state: the switch, the queue, what was used
# --------------------------------------------------------------------------- #


class Store:
    """``curiosities_local.json``: enabled, queue, used titles, last refill day."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.enabled: bool | None = None
        self.queue: list[Curiosity] = []
        self.used: list[str] = []
        self.refilled_on = ""
        self.next_source = "wikipedia"
        self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            log.warning("Ignoring unreadable %s", self.path)
            return
        if not isinstance(data, dict):
            return
        if isinstance(data.get("enabled"), bool):
            self.enabled = data["enabled"]
        self.queue = [c for c in (Curiosity.from_dict(d) for d in data.get("queue") or []) if c]
        self.used = [str(t) for t in data.get("used") or [] if t][-_USED_MAX:]
        self.refilled_on = str(data.get("refilled_on") or "")
        if data.get("next_source") in ("wikipedia", "claude"):
            self.next_source = data["next_source"]

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = {"enabled": self.enabled, "queue": [c.to_dict() for c in self.queue],
                    "used": self.used[-_USED_MAX:], "refilled_on": self.refilled_on,
                    "next_source": self.next_source}
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Could not save curiosities to %s: %s", self.path, exc)

    def used_keys(self) -> set[str]:
        with self._lock:
            return {_key(t) for t in self.used} | {_key(c.title) for c in self.queue}

    def pending(self, today: date) -> int:
        with self._lock:
            return sum(1 for c in self.queue if not c.day or c.day == today.isoformat())

    def pop(self, today: date) -> Curiosity | None:
        """The next item still valid today (stale "un día como hoy" are dropped)."""
        with self._lock:
            self.queue = [c for c in self.queue if not c.day or c.day == today.isoformat()]
            if not self.queue:
                return None
            item = self.queue.pop(0)
            self.used.append(item.title)
        self.save()
        return item

    def add(self, items: list[Curiosity], today: date, source: str) -> None:
        with self._lock:
            self.queue.extend(items)
            self.refilled_on = today.isoformat()
            self.next_source = "claude" if source == "wikipedia" else "wikipedia"
        self.save()


# --------------------------------------------------------------------------- #
# When
# --------------------------------------------------------------------------- #


class Schedule:
    """Whether a curiosity may be shown now. Pure: the clocks are passed in."""

    def __init__(self, settings: CuriositySettings, *, night_start: float = 21.0,
                 night_end: float = 7.0, rng: random.Random | None = None,
                 now: float | None = None) -> None:
        self.settings = settings
        self.night_start, self.night_end = night_start, night_end
        self.rng = rng or random.Random()
        start = time.monotonic() if now is None else now
        self.next_at = start + settings.first_after_min * 60.0 * self._spread()

    def _spread(self) -> float:
        j = self.settings.jitter
        return self.rng.uniform(1.0 - j, 1.0 + j)

    def night(self, wall: float) -> bool:
        from wawi.pet.rhythm import profile

        return profile(wall, night_start=self.night_start, night_end=self.night_end)[4]

    def due(self, now: float, wall: float, *, idle: bool, asleep: bool, busy: bool,
            channel_open: bool, last_talk: float | None) -> bool:
        if now < self.next_at:
            return False
        if not idle or asleep or busy or channel_open:
            return False
        if last_talk is not None and now - last_talk < self.settings.quiet_after_talk_s:
            return False
        return not self.night(wall)

    def fired(self, now: float) -> None:
        self.next_at = now + self.settings.every_min * 60.0 * self._spread()

    def later(self, now: float, seconds: float) -> None:
        self.next_at = max(self.next_at, now + seconds)


def _in_meeting() -> bool:
    try:
        from wawi.initiative.meeting import in_meeting

        return in_meeting()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


class CuriosityService:
    """Owns the queue, the schedule, the bubble and the "Ver más" article.

    :meth:`poll`, :meth:`bubble_text`, :meth:`dismiss` and :meth:`more` are
    called by the panel on the main thread and never block: refills, image
    searches and the board's document are built on daemon threads.
    """

    def __init__(self, settings: CuriositySettings, *, state_path: str | Path | None = None,
                 board: Any = None, night_start: float = 21.0, night_end: float = 7.0,
                 claude: Callable[[str, str], str] | None = None,
                 wiki_fetch: Callable[[str], dict] = _json_get,
                 find_image: Callable[[str], Any] | None = None,
                 rng: random.Random | None = None, now: float | None = None,
                 background: bool = True) -> None:
        self.settings = settings
        self.store = Store(state_path)
        self.board = board
        self.schedule = Schedule(settings, night_start=night_start, night_end=night_end,
                                 rng=rng, now=now)
        self._claude = claude
        self._wiki_fetch = wiki_fetch
        self._find_image = find_image
        self._background = background
        self._lock = threading.Lock()
        self._refilling = False
        self._failed_at: float | None = None
        self.last_talk: float | None = None
        self.current: Curiosity | None = None
        self._shown_at = 0.0
        self._until = 0.0
        self._noted: Curiosity | None = None

    # -- switch ---------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.settings.enabled if self.store.enabled is None else self.store.enabled

    def set_enabled(self, on: bool) -> None:
        """The menu's checkmark. Persisted; switching off hides the bubble."""
        self.store.enabled = bool(on)
        self.store.save()
        if not on:
            self.dismiss()
        log.info("Curiosities %s", "on" if on else "off")

    # -- inputs ---------------------------------------------------------------
    def observe_state(self, old: Any, new: Any) -> None:
        """StateMachine observer: any turn counts as talk (quiet afterwards)."""
        if getattr(new, "value", new) != "IDLE" or getattr(old, "value", old) != "IDLE":
            self.last_talk = time.monotonic()

    # -- the bubble -------------------------------------------------------------
    def poll(self, now: float, wall: float, *, idle: bool, asleep: bool, busy: bool,
             channel_open: bool) -> Curiosity | None:
        """A new curiosity to show right now, or None. Main thread, cheap."""
        if not self.enabled or self.current is not None and now < self._until:
            return None
        if _in_meeting():
            return None  # not even a silent one: the owner is on a call
        if not self.schedule.due(now, wall, idle=idle, asleep=asleep, busy=busy,
                                 channel_open=channel_open, last_talk=self.last_talk):
            if now >= self.schedule.next_at:
                self.schedule.later(now, 5.0)  # re-check in a while, not every frame
            return None
        today = date.fromtimestamp(wall)
        item = self.store.pop(today)
        if item is None:
            self.schedule.later(now, _RETRY_WHEN_EMPTY_S)
            self.maybe_refill(today)
            return None
        self.schedule.fired(now)
        self.current, self._shown_at, self._until = item, now, now + self.settings.show_s
        log.info("Curiosity shown (%s): %s", item.source, item.teaser)
        self.maybe_refill(today)
        return item

    def bubble_text(self, now: float) -> str | None:
        item = self.current
        if item is None or now >= self._until:
            return None
        return f"{item.teaser}\n{LINK}"

    def dismiss(self) -> None:
        self._until = 0.0

    def more(self) -> bool:
        """"Ver más…" was clicked: open the article on the board."""
        item = self.current
        if item is None:
            return False
        self.dismiss()
        log.info("Curiosity 'ver más': %s", item.title)
        self._spawn(self._open_article, item, name="wawi-curiosity-more")
        return True

    def _open_article(self, item: Curiosity) -> None:
        card = None
        if item.image_query:
            try:
                finder = self._find_image
                if finder is None:
                    from wawi.images import find_image as finder
                card = finder(item.image_query)
            except Exception as exc:
                log.info("No image for curiosity %r: %s", item.title, exc)
        content, fmt = article_html(item, card)
        try:
            if self.board is not None:
                self.board.show(item.title, content, fmt)
        except Exception as exc:
            log.warning("Could not show the curiosity on the board: %s", exc)

    # -- the brain --------------------------------------------------------------
    def with_note(self, user_text: str, now: float | None = None) -> str:
        """*user_text* with one line about the last curiosity, while it is
        recent and only once (the session keeps it after that). Goes in the
        USER turn: the system prompt must not change (prompt cache)."""
        item = self.current
        now = time.monotonic() if now is None else now
        if item is None or item is self._noted \
                or now - self._shown_at > self.settings.recent_min * 60.0:
            return user_text
        self._noted = item
        return (f"(Contexto: hace un rato mostraste en tu globito la curiosidad "
                f"«{item.teaser}» sobre «{item.title}». Si te piden «contame más», "
                f"es sobre esto.)\n\n{user_text}")

    # -- refills ----------------------------------------------------------------
    def needs_refill(self, today: date, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if self._failed_at is not None and now - self._failed_at < _RETRY_AFTER_FAIL_S:
            return False
        return (self.store.pending(today) < self.settings.refill_below
                and self.store.refilled_on != today.isoformat())

    def maybe_refill(self, today: date | None = None) -> bool:
        today = today or date.today()
        if not self.enabled or not self.needs_refill(today):
            return False
        with self._lock:
            if self._refilling:
                return False
            self._refilling = True
        self._spawn(self._refill, today, name="wawi-curiosity-refill")
        return True

    def _refill(self, today: date) -> None:
        # Her little "…" while she looks something up (wawi/working.py).
        _working.set_musing("curiosities", True)
        try:
            first = self.store.next_source
            for source in (first, "claude" if first == "wikipedia" else "wikipedia"):
                try:
                    items = self._generate(source, today)
                except Exception as exc:
                    # One line, no traceback: offline is not a bug.
                    log.info("Curiosities from %s failed: %s", source, exc)
                    continue
                if items:
                    self.store.add(items, today, source)
                    self._failed_at = None
                    log.info("Curiosities: %d new from %s", len(items), source)
                    return
            self._failed_at = time.monotonic()
        finally:
            _working.set_musing("curiosities", False)
            with self._lock:
                self._refilling = False

    def _generate(self, source: str, today: date) -> list[Curiosity]:
        n = self.settings.batch
        used = self.store.used_keys()
        if source == "claude":
            if self._claude is None:
                return []
            text = self._claude(generation_prompt(n, today, self.store.used), GENERATE_SYSTEM)
            return parse_items(text, used=used)[:n]
        rewrite = None
        if self._claude is not None:
            claude = self._claude
            rewrite = lambda prompt: claude(prompt, REWRITE_SYSTEM)  # noqa: E731
        return wikipedia_batch(n, today, used=used, rewrite=rewrite, fetch=self._wiki_fetch)

    def _spawn(self, fn: Callable[..., None], *args: Any, name: str) -> None:
        if not self._background:
            fn(*args)
            return
        threading.Thread(target=self._safe, args=(fn, *args), name=name, daemon=True).start()

    @staticmethod
    def _safe(fn: Callable[..., None], *args: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:
            log.warning("Curiosities: %s", exc)

    def start(self) -> None:
        """Top up the queue in the background if it is low."""
        self.maybe_refill()


# --------------------------------------------------------------------------- #
# Process-wide instance (like wawi.games / wawi.reminders)
# --------------------------------------------------------------------------- #

_SERVICE: CuriosityService | None = None


def install(service: CuriosityService | None) -> None:
    global _SERVICE
    _SERVICE = service


def get() -> CuriosityService | None:
    return _SERVICE


def with_note(user_text: str) -> str:
    """The installed service's :meth:`CuriosityService.with_note`; never raises."""
    service = _SERVICE
    if service is None:
        return user_text
    try:
        return service.with_note(user_text)
    except Exception:
        return user_text


def build(cfg: Any, board: Any = None) -> CuriosityService | None:
    """The service for *cfg* (config ``curiosities:``), or None without a board."""
    settings = CuriositySettings.from_dict(getattr(cfg, "curiosities", {}) or {})
    try:
        from wawi.pet.settings import PetSettings

        pet = PetSettings.from_dict(getattr(cfg, "pet", {}) or {})
        night = (pet.night_start, pet.night_end)
    except Exception:
        night = (21.0, 7.0)
    claude = None
    if getattr(cfg, "backend", "claude") == "claude":
        model, timeout = settings.model, settings.timeout_s
        claude = lambda prompt, system: ask_claude(  # noqa: E731
            prompt, system, model=model, timeout=timeout)
    from wawi import paths as _paths

    state = _paths.resolve(settings.local_file) if settings.local_file else None
    return CuriosityService(settings, state_path=state, board=board,
                            night_start=night[0], night_end=night[1], claude=claude)
