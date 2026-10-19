"""Things the owner drags onto Wawi: she catches them and waits for the order.

Files, folders, links, text and pictures dragged straight out of an app are
dropped on her window (wawi/panel.py, the NSDraggingDestination half). Here:

* :func:`classify` turns what the pasteboard holds into :class:`DropItem` s
  (a picture with no file behind it is saved to a temp file first),
* :func:`catch_line` is what she says on catching them, by kind,
* :class:`Drop` is the thing "in her mouth": the items, until when the
  owner's order is awaited, and until when she still holds them for
  follow-ups ("y ahora traducilo") before she buries them and forgets,
* :meth:`Drop.context` renders them for the brain, ONLY when the owner has
  spoken an order: a one-line header per item and what can be read of it
  (text inline, a file's excerpt, a PDF's text, an audio file's local
  transcript) prefixed to the USER turn, and pictures as image attachments
  (wawi.brain.ImageAttachment) so the model actually sees them.

Nothing here talks to the network or the model. The orchestrator owns the
microphone window and the turn (see Orchestrator._handle_drop).
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.parse import unquote, urlparse

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Settings (config.yaml ``drops:``; defaults in wawi/config.py)
# --------------------------------------------------------------------------- #

#: Seconds the owner has to START saying what to do with it.
WINDOW_S = 15.0
#: Seconds she keeps it "in her mouth" for follow-ups, then buries it.
HOLD_S = 180.0
#: Longest text, excerpt or transcript inlined per item (bytes of UTF-8).
MAX_TEXT = 8 * 1024
#: Biggest plain-text file read whole for an excerpt.
MAX_FILE = 200 * 1024
#: Pictures attached to one turn; more are listed by name only.
MAX_IMAGES = 4
#: Longest audio file transcribed (seconds); longer ones are cut there.
MAX_AUDIO_S = 15 * 60.0


@dataclass(frozen=True)
class DropSettings:
    window_s: float = WINDOW_S
    hold_s: float = HOLD_S
    max_text: int = MAX_TEXT
    max_file: int = MAX_FILE

    @classmethod
    def from_config(cls, raw: Any) -> "DropSettings":
        section = raw if isinstance(raw, dict) else {}

        def num(key: str, default: float) -> float:
            try:
                value = float(section.get(key, default))
            except (TypeError, ValueError):
                return default
            return value if value > 0 else default

        return cls(
            window_s=num("window_s", WINDOW_S),
            hold_s=num("hold_s", HOLD_S),
            max_text=int(num("max_text_kb", MAX_TEXT / 1024) * 1024),
            max_file=int(num("max_file_kb", MAX_FILE / 1024) * 1024),
        )


# --------------------------------------------------------------------------- #
# What she says (Spanish, rioplatense; short: it is spoken)
# --------------------------------------------------------------------------- #

LINES = {
    "image": "¡Una foto! ¿Qué hago?",
    "images": "¡Fotos! ¿Qué hago?",
    "pdf": "Uh, un papel largo. ¿Qué hago?",
    "doc": "Uh, un papel largo. ¿Qué hago?",
    "url": "¿Voy a ver?",
    "text": "Lo tengo.",
    "audio": "¿Lo paso a texto?",
    "sheet": "Muchos números…",
    "code": "Huele a código.",
    "folder": "¡Cuántas cosas!",
    "archive": "Huele raro…",
    "unknown": "Huele raro…",
    "many": "¡Cuántas cosas! ¿Qué hago?",
}
#: A turn is in flight: she does not catch it.
BUSY_TEXT = "Ahora no puedo, tiramelo después."
#: The window ran out with no order: she leaves it there, quietly.
LEFT_TEXT = "Lo dejo acá por si lo querés."
#: Held long enough: she buries it and forgets.
BURIED_TEXT = "Lo enterré."
#: Taking notes: the item went into the note instead.
NOTED_TEXT = "Anotado."

# --------------------------------------------------------------------------- #
# Items and their kinds
# --------------------------------------------------------------------------- #

_EXT_KINDS: dict[str, tuple[str, ...]] = {
    "image": ("png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "tif", "tiff", "bmp"),
    "pdf": ("pdf",),
    "doc": ("txt", "md", "markdown", "rtf", "rtfd", "doc", "docx", "odt", "pages", "html",
            "htm", "log"),
    "sheet": ("csv", "tsv", "xls", "xlsx", "numbers", "ods"),
    "code": ("py", "js", "ts", "tsx", "jsx", "json", "yaml", "yml", "toml", "sh", "zsh",
             "bash", "go", "rs", "java", "kt", "c", "h", "cc", "cpp", "hpp", "m", "mm",
             "swift", "rb", "php", "sql", "css", "scss", "xml", "ini", "cfg", "lua", "r",
             "ipynb"),
    "audio": ("m4a", "mp3", "wav", "aiff", "aif", "flac", "ogg", "caf", "aac", "opus"),
    "archive": ("zip", "tar", "gz", "tgz", "bz2", "xz", "7z", "rar", "dmg"),
}
_KIND_BY_EXT = {ext: kind for kind, exts in _EXT_KINDS.items() for ext in exts}
#: Read as plain text for an excerpt (small files only, see MAX_FILE).
_PLAIN_TEXT = frozenset(("txt", "md", "markdown", "log", "csv", "tsv", *_EXT_KINDS["code"]))
#: Converted to text by macOS's own textutil (no extra dependency).
_TEXTUTIL = frozenset(("rtf", "rtfd", "doc", "docx", "odt", "html", "htm"))

#: How each kind is named in the header the model reads.
_KIND_LABEL = {
    "image": "imagen", "pdf": "PDF", "doc": "documento", "sheet": "planilla",
    "code": "código", "audio": "audio", "folder": "carpeta", "archive": "archivo comprimido",
    "unknown": "archivo", "url": "link", "text": "texto",
}


@dataclass
class DropItem:
    """One thing dropped. *path* for files and folders (a saved temp file for
    a picture with no file behind it), *text* for a link or plain text."""

    kind: str
    name: str = ""
    path: str | None = None
    size: int | None = None
    text: str = ""
    #: Where a picture without a file came from (the browser's link), if known.
    source: str = ""

    def header(self) -> str:
        """``imagen /path/x.png (x.png, 1.2 MB)``: what the model is told."""
        label = _KIND_LABEL.get(self.kind, "archivo")
        if self.kind == "url":
            return f"{label} {self.text}"
        if self.kind == "text":
            return f"{label} ({len(self.text)} caracteres)"
        bits = [self.name] if self.name else []
        if self.size is not None and self.kind != "folder":
            bits.append(human_size(self.size))
        if self.source:
            bits.append(f"de {self.source}")
        detail = f" ({', '.join(bits)})" if bits else ""
        return f"{label} {self.path or self.name}{detail}"


def human_size(n: int) -> str:
    """``1.2 MB``, ``340 KB``, ``12 B``."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{round(n / 1024)} KB"
    return f"{n} B"


def kind_for_path(path: str) -> str:
    """The kind of a dropped file or folder, by what it is and its extension."""
    p = Path(path)
    if p.is_dir() and p.suffix.lower() not in (".rtfd", ".pages", ".numbers"):
        return "folder"
    return _KIND_BY_EXT.get(p.suffix.lower().lstrip("."), "unknown")


def item_for_path(path: str) -> DropItem:
    p = Path(path)
    kind = kind_for_path(path)
    size = None
    if kind != "folder":
        try:
            size = p.stat().st_size
        except OSError:
            size = None
    return DropItem(kind=kind, name=p.name, path=str(p), size=size)


# --------------------------------------------------------------------------- #
# The pasteboard (an NSPasteboard in the app, a fake in the tests)
# --------------------------------------------------------------------------- #

#: Pasteboard types she accepts (registerForDraggedTypes_).
FILE_URL = "public.file-url"
URL = "public.url"
STRING = "public.utf8-plain-text"
PNG = "public.png"
TIFF = "public.tiff"
JPEG = "public.jpeg"
DRAG_TYPES = (FILE_URL, URL, STRING, PNG, TIFF, JPEG)


class Pasteboard(Protocol):
    def types(self) -> list[str]: ...
    def file_paths(self) -> list[str]: ...
    def string(self, kind: str) -> str | None: ...
    def data(self, kind: str) -> bytes | None: ...


def _web_url(text: str | None) -> str:
    """*text* if it is one http(s) link and nothing else, else ""."""
    text = (text or "").strip()
    if not text or any(c.isspace() for c in text):
        return ""
    parsed = urlparse(text)
    return text if parsed.scheme in ("http", "https") and parsed.netloc else ""


def _drop_dir() -> Path:
    folder = Path(tempfile.gettempdir()) / "wawi-drops"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_picture(data: bytes, kind: str, directory: Path | None = None) -> Path:
    """Write a dragged picture's bytes to a temp file. TIFF (what most apps
    put on the pasteboard) becomes PNG when Pillow can read it."""
    folder = directory or _drop_dir()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if kind == TIFF:
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            path = folder / f"wawi-drop-{stamp}.png"
            img.save(path, format="PNG")
            return path
        except Exception:
            log.debug("TIFF drop kept as TIFF", exc_info=True)
    ext = {PNG: "png", JPEG: "jpg", TIFF: "tiff"}.get(kind, "png")
    fd, name = tempfile.mkstemp(prefix=f"wawi-drop-{stamp}-", suffix=f".{ext}", dir=folder)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return Path(name)


def classify(pb: Pasteboard, *, picture_dir: Path | None = None) -> list[DropItem]:
    """What was dropped, as items. Files win (dragging a file also puts its
    icon and name on the pasteboard); then a picture dragged out of an app
    (saved to a temp file; the link it came from kept as its source); then a
    link; then plain text (one that is just a link counts as a link)."""
    types = set(pb.types() or ())
    paths = [p for p in (pb.file_paths() or []) if p]
    if paths:
        return [item_for_path(p) for p in paths]
    url = ""
    if URL in types:
        url = _web_url(pb.string(URL))
    for kind in (PNG, JPEG, TIFF):
        if kind in types:
            data = pb.data(kind)
            if data:
                path = save_picture(data, kind, picture_dir)
                return [DropItem(kind="image", name=path.name, path=str(path),
                                 size=len(data), source=url)]
    if url:
        return [DropItem(kind="url", name=url, text=url)]
    if STRING in types:
        text = pb.string(STRING) or ""
        link = _web_url(text)
        if link:
            return [DropItem(kind="url", name=link, text=link)]
        if text.strip():
            return [DropItem(kind="text", name="texto", text=text)]
    return []


def paths_from_file_urls(urls: Iterable[str]) -> list[str]:
    """``file:///a%20b`` -> ``/a b``; anything else dropped."""
    out = []
    for u in urls:
        parsed = urlparse(str(u))
        if parsed.scheme == "file":
            out.append(unquote(parsed.path))
    return out


# --------------------------------------------------------------------------- #
# What she says, and what the log says
# --------------------------------------------------------------------------- #


def catch_line(items: list[DropItem]) -> str:
    """Her line on catching *items*: by kind; several of one kind say the
    same (pictures have a plural); a mix is "¡Cuántas cosas!"."""
    kinds = {i.kind for i in items}
    if not kinds:
        return LINES["unknown"]
    if len(kinds) > 1:
        return LINES["many"]
    kind = kinds.pop()
    if kind == "image" and len(items) > 1:
        return LINES["images"]
    return LINES.get(kind, LINES["unknown"])


def summary(items: list[DropItem]) -> str:
    """For the log: kinds, count and sizes; never a name's contents."""
    counts: dict[str, int] = {}
    for i in items:
        counts[i.kind] = counts.get(i.kind, 0) + 1
    kinds = ", ".join(f"{k}x{n}" if n > 1 else k for k, n in counts.items())
    sizes = [human_size(i.size) for i in items if i.size is not None]
    texts = [f"{len(i.text)} chars" for i in items if i.kind in ("text", "url")]
    return (f"{len(items)} item(s): {kinds}"
            + (f"; sizes {', '.join(sizes)}" if sizes else "")
            + (f"; {', '.join(texts)}" if texts else ""))


def note_line(item: DropItem) -> str:
    """The note line for an item dropped while she takes notes."""
    if item.kind == "text":
        return f"[texto] {item.text}"
    if item.kind == "url":
        return f"[link: {item.text}]"
    return f"[archivo: {item.name or item.path}]"


# --------------------------------------------------------------------------- #
# Reading items for the brain
# --------------------------------------------------------------------------- #


def clip_text(text: str, limit: int = MAX_TEXT) -> tuple[str, bool]:
    """*text* cut to *limit* bytes of UTF-8 (on a character boundary), and
    whether it was cut."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text, False
    return raw[:limit].decode("utf-8", "ignore").rstrip() + " […]", True


def read_text_file(path: str, max_file: int = MAX_FILE) -> str | None:
    """A small plain-text file's contents, or None (too big, binary, gone)."""
    try:
        p = Path(path)
        if p.stat().st_size > max_file:
            return None
        raw = p.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", "replace")


def textutil_text(path: str, timeout: float = 20.0) -> str | None:
    """A Word/RTF/HTML document's text, by macOS's own ``textutil``."""
    try:
        out = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", path],
            capture_output=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", "replace")


def pdf_text(path: str, max_chars: int = MAX_TEXT * 2) -> str | None:
    """A PDF's text by PDFKit (already here through pyobjc's Quartz), or None
    when it cannot be read (no PDFKit, scanned pages, encrypted)."""
    try:
        from Foundation import NSURL  # type: ignore[import]
        from Quartz import PDFDocument  # type: ignore[import]
    except Exception:
        return None
    try:
        doc = PDFDocument.alloc().initWithURL_(NSURL.fileURLWithPath_(path))
        if doc is None:
            return None
        parts: list[str] = []
        total = 0
        for i in range(int(doc.pageCount())):
            page = doc.pageAtIndex_(i)
            text = str(page.string() or "") if page is not None else ""
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        text = "\n".join(parts).strip()
        return text or None
    except Exception:
        log.debug("PDF text failed", exc_info=True)
        return None


def load_audio(path: str, sample_rate: int = 16000) -> "Any | None":
    """Decode an audio file to float32 mono at *sample_rate*, or None.

    soundfile reads wav/flac/ogg/mp3; anything else (m4a, aac, caf) goes
    through macOS's own ``afconvert`` to a temp WAV first.
    """
    import numpy as np

    try:
        import soundfile as sf
    except Exception:
        return None

    def _read(p: str):
        data, sr = sf.read(p, dtype="float32", always_2d=True,
                           frames=int(MAX_AUDIO_S * 48000))
        return data.mean(axis=1), int(sr)

    try:
        mono, sr = _read(path)
    except Exception:
        fd, wav = tempfile.mkstemp(suffix=".wav", dir=_drop_dir())
        os.close(fd)
        try:
            subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}",
                            "-c", "1", path, wav],
                           capture_output=True, timeout=120, check=True)
            mono, sr = _read(wav)
        except Exception:
            log.debug("Could not decode %s", path, exc_info=True)
            return None
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass
    mono = mono[: int(MAX_AUDIO_S * sr)]
    if sr != sample_rate and mono.size:
        n = int(round(mono.size * sample_rate / sr))
        mono = np.interp(np.linspace(0, mono.size - 1, n), np.arange(mono.size), mono)
    return np.asarray(mono, dtype=np.float32)


def encode_picture(path: str, max_edge: int = 1280, jpeg_quality: int = 65):
    """``(base64_jpeg, (w, h))`` for a picture file, the way a screenshot is
    sent (wawi.screen.encode_jpeg); HEIC and friends go through ``sips``.
    None when it cannot be read."""
    from PIL import Image

    from wawi import screen

    def _open(p: str):
        img = Image.open(p)
        img.load()
        return img

    try:
        img = _open(path)
    except Exception:
        fd, jpg = tempfile.mkstemp(suffix=".jpg", dir=_drop_dir())
        os.close(fd)
        try:
            subprocess.run(["sips", "-s", "format", "jpeg", path, "--out", jpg],
                           capture_output=True, timeout=30, check=True)
            img = _open(jpg)
        except Exception:
            log.debug("Could not read picture %s", path, exc_info=True)
            return None
        finally:
            try:
                os.unlink(jpg)
            except OSError:
                pass
    if getattr(img, "is_animated", False):
        img.seek(0)
    if img.mode not in ("RGB", "L"):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        img = flat
    return screen.encode_jpeg(img, max_edge=max_edge, jpeg_quality=jpeg_quality)


def folder_listing(path: str, limit: int = 40) -> str | None:
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return None
    shown = [n + ("/" if os.path.isdir(os.path.join(path, n)) else "")
             for n in names if not n.startswith(".")][:limit]
    more = len(names) - len(shown)
    return "\n".join(shown) + (f"\n… y {more} más" if more > 0 else "")


Transcriber = Callable[["Any"], str]


def _excerpt(item: DropItem, settings: DropSettings,
             transcribe: Transcriber | None) -> tuple[str, str] | None:
    """``(title, text)`` of what can be read of *item*, or None."""
    path = item.path or ""
    ext = Path(path).suffix.lower().lstrip(".")
    if item.kind == "text":
        return "Texto", item.text
    if item.kind == "folder":
        listing = folder_listing(path)
        return (f"Contenido de {item.name}", listing) if listing else None
    if item.kind == "pdf":
        text = pdf_text(path)
        return (f"Texto de {item.name}", text) if text else (
            f"Sobre {item.name}", "(No pude leer el texto de este PDF: solo ves su nombre.)")
    if item.kind == "audio":
        if transcribe is None:
            return None
        audio = load_audio(path)
        if audio is None:
            return f"Sobre {item.name}", "(No pude abrir este audio.)"
        try:
            text = (transcribe(audio) or "").strip()
        except Exception:
            log.warning("Could not transcribe a dropped audio file", exc_info=True)
            return f"Sobre {item.name}", "(No pude pasar este audio a texto.)"
        return f"Transcripción de {item.name} (local, Whisper)", text or "(sin habla)"
    if ext in _PLAIN_TEXT:
        text = read_text_file(path, settings.max_file)
        return (f"Contenido de {item.name}", text) if text is not None else None
    if ext in _TEXTUTIL:
        text = textutil_text(path)
        return (f"Texto de {item.name}", text) if text else None
    return None


@dataclass
class Drop:
    """What she is holding: the items, and the clock.

    *awaiting* is True while the window for the order is open; the first
    turn that carries the drop closes it. The drop itself stays until
    :meth:`expired` (``hold_s`` after it was caught).
    """

    items: list[DropItem]
    settings: DropSettings = field(default_factory=DropSettings)
    at: float = field(default_factory=time.monotonic)
    awaiting: bool = True
    turns: int = 0
    _context: "tuple[str, tuple[Any, ...]] | None" = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def expired(self, now: float | None = None) -> bool:
        return ((time.monotonic() if now is None else now) - self.at) >= self.settings.hold_s

    def context(self, transcribe: Transcriber | None = None) -> "tuple[str, tuple[Any, ...]]":
        """``(prefix, images)`` for a turn that carries this drop: the text to
        put before what the owner said, and the pictures to attach. Read once
        and kept: a follow-up does not read the files (or transcribe) again.
        """
        with self._lock:
            if self._context is None:
                self._context = self._build(transcribe)
            return self._context

    def _build(self, transcribe: Transcriber | None) -> "tuple[str, tuple[Any, ...]]":
        from wawi.brain import ImageAttachment

        images: list[Any] = []
        headers: list[str] = []
        blocks: list[str] = []
        budget = self.settings.max_text * 3  # all items together
        for item in self.items:
            header = item.header()
            if item.kind == "image" and item.path:
                if len(images) < MAX_IMAGES:
                    encoded = encode_picture(item.path)
                    if encoded is not None:
                        data, (w, h) = encoded
                        images.append(ImageAttachment(
                            data=data, note=f"[imagen que te pasó: {item.name}, {w}x{h}]"))
                        header += " — la imagen va adjunta"
                    else:
                        header += " — no pude abrirla"
                else:
                    header += " — no adjunta (demasiadas)"
            else:
                got = _excerpt(item, self.settings, transcribe)
                if got is not None and budget > 0:
                    title, text = got
                    text, cut = clip_text(text, min(self.settings.max_text, budget))
                    budget -= len(text.encode("utf-8"))
                    blocks.append(f"[{title}{' (recortado)' if cut else ''}:\n{text}\n]")
            headers.append(header)
        if len(headers) == 1:
            head = f"[Te pasé: {headers[0]}]"
        else:
            head = f"[Te pasé {len(headers)} cosas:\n" + "\n".join(f"- {h}" for h in headers) + "]"
        prefix = "\n".join([head, *blocks]) + "\n"
        return prefix, tuple(images)
