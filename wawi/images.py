"""Find a freely reusable image, then bring its bytes into the offline board."""
from __future__ import annotations

import base64
import html
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

from wawi.mind.fetch import FetchError, USER_AGENT, _CheckedRedirects, _resolve, check_url

log = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 6_000_000
MAX_PAGE_BYTES = 1_000_000
TIMEOUT = 12
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


class ImageError(Exception):
    """A short, user-facing failure."""


@dataclass(frozen=True)
class ImageCard:
    title: str
    description: str
    source_url: str
    credit: str
    license: str
    data_uri: str


def _request(url: str, limit: int) -> tuple[bytes, str]:
    """Bound a public HTTP(S) response, checking each redirect as well."""
    try:
        check_url(url)
        opener = urllib.request.build_opener(_CheckedRedirects(False, _resolve))
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept-Language": "es,en;q=0.8"})
        with opener.open(req, timeout=TIMEOUT) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > limit:
                raise ImageError("La imagen es demasiado grande.")
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ImageError("La imagen es demasiado grande.")
            return body, response.headers.get("Content-Type", "").split(";", 1)[0].lower()
    except ImageError:
        raise
    except (OSError, ValueError, FetchError) as exc:
        log.info("Image fetch failed: %s", exc)
        raise ImageError("No pude traer esa imagen.") from exc


def _json(url: str) -> dict:
    body, _ = _request(url, MAX_PAGE_BYTES)
    try:
        return json.loads(body)
    except (UnicodeError, ValueError) as exc:
        raise ImageError("No pude leer la información de la imagen.") from exc


def _commons_info(query: str, *, file_title: str = "") -> dict:
    params = {"action": "query", "format": "json", "prop": "imageinfo",
              "iiprop": "url|extmetadata", "iiurlwidth": "1200"}
    if file_title:
        params["titles"] = file_title
    else:
        params.update({"generator": "search", "gsrsearch": f"filetype:bitmap {query}",
                       "gsrnamespace": "6", "gsrlimit": "10"})
    data = _json(COMMONS_API + "?" + urllib.parse.urlencode(params))
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url", "")
        if url and not url.lower().split("?", 1)[0].endswith(".svg"):
            return {"url": url, "title": page.get("title", ""),
                    "page_url": info.get("descriptionurl", ""),
                    "metadata": info.get("extmetadata", {})}
    return {}


def _meta_text(metadata: dict, key: str) -> str:
    # Commons metadata can contain simple HTML links and markup.
    raw = str(metadata.get(key, {}).get("value", ""))
    class Text(HTMLParser):
        parts: list[str]
        def __init__(self):
            super().__init__(); self.parts = []
        def handle_data(self, data):
            self.parts.append(data)
    parser = Text()
    parser.feed(raw)
    return html.unescape(" ".join("".join(parser.parts).split()))[:600]


class _PageImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        values = dict(attrs)
        key = (values.get("property") or values.get("name") or "").lower()
        if key in {"og:image", "twitter:image", "og:title", "og:description", "description"}:
            self.meta.setdefault(key, values.get("content") or "")


def _page_info(url: str) -> dict:
    body, content_type = _request(url, MAX_PAGE_BYTES)
    if content_type not in {"text/html", "application/xhtml+xml"}:
        raise ImageError("Esa página no tiene una foto disponible.")
    parser = _PageImageParser()
    parser.feed(body.decode("utf-8", "replace"))
    meta = parser.meta
    image_url = meta.get("og:image") or meta.get("twitter:image")
    if not image_url:
        raise ImageError("Esa página no tiene una foto disponible.")
    return {"url": urllib.parse.urljoin(url, image_url), "title": meta.get("og:title", ""),
            "description": meta.get("og:description") or meta.get("description", ""),
            "page_url": url, "credit": urllib.parse.urlsplit(url).hostname or "",
            "license": "Ver derechos en la fuente"}


def _wiki_info(query: str) -> dict:
    for language in ("es", "en"):
        url = (f"https://{language}.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(query.replace(" ", "_"), safe=""))
        try:
            data = _json(url)
        except ImageError:
            continue
        image = data.get("thumbnail") or data.get("originalimage") or {}
        if image.get("source"):
            return {"url": image["source"], "title": data.get("title", query),
                    "description": data.get("extract", ""),
                    "page_url": data.get("content_urls", {}).get("desktop", {}).get("page", url)}
    return {}


def _commons_title(image_url: str) -> str:
    parts = urllib.parse.urlsplit(image_url)
    if not parts.hostname or not parts.hostname.endswith(".wikimedia.org") or "/wikipedia/commons/" not in parts.path:
        return ""
    path = parts.path.split("/")
    filename = path[-2] if "thumb" in path else path[-1]
    return "File:" + urllib.parse.unquote(filename) if filename else ""


def _data_uri(url: str) -> str:
    body, mime = _request(url, MAX_IMAGE_BYTES)
    signatures = {"image/jpeg": body.startswith(b"\xff\xd8\xff"),
                  "image/png": body.startswith(b"\x89PNG\r\n\x1a\n"),
                  "image/gif": body.startswith((b"GIF87a", b"GIF89a")),
                  "image/webp": body.startswith(b"RIFF") and body[8:12] == b"WEBP"}
    if not signatures.get(mime, False):
        raise ImageError("Ese archivo no es una foto compatible.")
    return f"data:{mime};base64,{base64.b64encode(body).decode('ascii')}"


def find_image(query: str, url: str = "", title: str = "") -> ImageCard:
    query = (query or "").strip()[:160]
    url = (url or "").strip()
    if not query and not url:
        raise ImageError("Decime qué foto querés ver.")
    if url:
        info = _page_info(url)
    else:
        info = _wiki_info(query)
        if info and not _commons_title(info["url"]):
            info = {}  # Wikipedia's locally hosted fair-use files are not free.
        commons = {}
        try:
            file_title = _commons_title(info["url"]) if info else ""
            commons = _commons_info(query, file_title=file_title) if file_title else {}
            if not info:
                commons = _commons_info(query)
        except ImageError:
            pass
        if commons:
            # Commons supplies explicit attribution and a free-license record.
            meta = commons["metadata"]
            info = {**info, **commons,
                    "title": info.get("title") or commons["title"],
                    "description": info.get("description") or _meta_text(meta, "ImageDescription"),
                    "credit": _meta_text(meta, "Artist") or _meta_text(meta, "Credit"),
                    "license": _meta_text(meta, "LicenseShortName") or "Ver licencia en Wikimedia Commons"}
    if not info or not info.get("url"):
        raise ImageError("No encontré una foto de eso.")
    try:
        data_uri = _data_uri(info["url"])
    except ImageError:
        raise
    return ImageCard(title=(title or info.get("title") or query or "Imagen")[:120],
                     description=(info.get("description") or "")[:1200],
                     source_url=info.get("page_url") or url,
                     credit=(info.get("credit") or "Wikimedia Commons")[:300],
                     license=(info.get("license") or "Ver licencia en la fuente")[:150],
                     data_uri=data_uri)
