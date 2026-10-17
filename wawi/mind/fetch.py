"""Fetch a web page the user named, safely enough for a voice command.

stdlib only, no JavaScript. Three guards, because the URL arrives through a
speech recogniser and a model:

* **Scheme.** http and https only. ``file://`` would read the disk.
* **Address.** The host is resolved first and refused when it lands on a
  private, loopback, link-local or reserved address (unless config allows
  it), and every redirect is checked again -- a public page that 302s to
  ``http://192.168.0.1/`` is the classic way around a check made only once.
* **Size and time.** The body is read up to a byte cap with a timeout; a
  huge download cannot wedge the brain's tool loop.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Macintosh; Wawi desktop assistant) AppleWebKit/537.36"


class FetchError(Exception):
    """A fetch that failed, with a short Spanish reason fit to be spoken."""

    def __init__(self, code: str, spoken: str) -> None:
        super().__init__(code)
        self.code = code
        self.spoken = spoken


@dataclass
class Page:
    url: str
    status: int
    content_type: str
    text: str
    title: str = ""
    truncated: bool = False
    headers: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# URL checks
# --------------------------------------------------------------------------- #


def normalise_url(raw: str) -> str:
    """Accept what a person says: "example.com/x" becomes https://example.com/x."""
    url = (raw or "").strip().strip("<>\"' ")
    if not url:
        raise FetchError("empty", "No me dijiste qué página.")
    if "://" not in url:
        url = "https://" + url
    return url


Resolver = Callable[[str], list[str]]


def _resolve(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def _is_private(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
        or ip.is_multicast or ip.is_unspecified
    )


def check_url(url: str, *, allow_private: bool = False, resolver: Resolver = _resolve) -> str:
    """Raise :class:`FetchError` unless *url* is http(s) to a public host."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise FetchError("scheme", "Solo puedo abrir páginas web, con http o https.")
    host = parts.hostname
    if not host:
        raise FetchError("no_host", "Esa dirección no tiene un sitio.")
    if allow_private:
        return url
    if host.lower() in ("localhost",) or host.lower().endswith((".local", ".localhost", ".internal")):
        raise FetchError("private", "Esa dirección es de tu red local y no tengo permiso para entrar.")
    try:
        addrs = [host] if _looks_like_ip(host) else resolver(host)
    except (socket.gaierror, UnicodeError, OSError):
        raise FetchError("dns", "No encontré ese sitio; puede que el nombre esté mal.")
    if not addrs:
        raise FetchError("dns", "No encontré ese sitio; puede que el nombre esté mal.")
    for addr in addrs:
        try:
            private = _is_private(addr)
        except ValueError:
            private = True
        if private:
            raise FetchError("private", "Esa dirección es de tu red local y no tengo permiso para entrar.")
    return url


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# HTML -> text
# --------------------------------------------------------------------------- #

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head", "iframe", "canvas"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "h1", "h2", "h3", "h4",
    "h5", "h6", "section", "article", "header", "footer", "pre", "blockquote",
    "table", "nav", "main", "aside", "dd", "dt", "figcaption",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "title":
            self._in_title = True
        elif tag == "body":
            # HTML5 lets </head> (and unclosed junk before it) be omitted; the
            # body starting is the one sure sign that we are past it.
            self._skip = 0
        elif tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """(title, readable text) of an HTML document. Never raises."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup still yields what was parsed
        log.debug("html_to_text: parser gave up early", exc_info=True)
    title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip()
    raw = "".join(parser.parts)
    lines = [re.sub(r"[ \t\r\f\v ]+", " ", ln).strip() for ln in raw.split("\n")]
    text = "\n".join(ln for ln in lines if ln)
    return title, text


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 5

    def __init__(self, allow_private: bool, resolver: Resolver) -> None:
        self._allow_private = allow_private
        self._resolver = resolver

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        check_url(newurl, allow_private=self._allow_private, resolver=self._resolver)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _charset(content_type: str, head: bytes) -> str:
    m = re.search(r"charset=([\w-]+)", content_type or "", re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", head[:4096], re.IGNORECASE)
    return m.group(1).decode("ascii", "ignore") if m else "utf-8"


def fetch(
    raw_url: str,
    *,
    allow_private: bool = False,
    max_bytes: int = 1_500_000,
    timeout: float = 12.0,
    resolver: Resolver = _resolve,
    opener: urllib.request.OpenerDirector | None = None,
) -> Page:
    """Fetch *raw_url* and return its readable text. Raises :class:`FetchError`."""
    url = check_url(normalise_url(raw_url), allow_private=allow_private, resolver=resolver)
    if opener is None:
        opener = urllib.request.build_opener(_CheckedRedirects(allow_private, resolver))
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
        },
    )
    try:
        with opener.open(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode() or 200
            final_url = resp.geturl() or url
            headers = {k.lower(): v for k, v in resp.headers.items()}
            body = resp.read(max_bytes + 1)
    except FetchError:
        raise
    except urllib.error.HTTPError as exc:
        raise FetchError(f"http_{exc.code}", f"La página respondió con error {exc.code}.")
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, FetchError):
            raise reason
        if isinstance(reason, socket.timeout) or "timed out" in str(reason):
            raise FetchError("timeout", "La página tardó demasiado en contestar.")
        raise FetchError("unreachable", "No pude llegar a esa página.")
    except (socket.timeout, TimeoutError):
        raise FetchError("timeout", "La página tardó demasiado en contestar.")
    except (OSError, ValueError) as exc:
        log.info("fetch failed: %s", exc)
        raise FetchError("unreachable", "No pude llegar a esa página.")

    truncated = len(body) > max_bytes
    body = body[:max_bytes]
    ctype = headers.get("content-type", "")
    main_type = ctype.split(";", 1)[0].strip().lower()
    if main_type and not (main_type.startswith("text/") or "html" in main_type
                          or "xml" in main_type or "json" in main_type):
        raise FetchError("not_text", "Eso no es una página de texto; no la puedo leer.")
    decoded = body.decode(_charset(ctype, body), errors="replace")
    if "html" in main_type or (not main_type and "<html" in decoded[:2000].lower()):
        title, text = html_to_text(decoded)
    else:
        title, text = "", decoded.strip()
    return Page(final_url, int(status), ctype, text, title, truncated, headers)
