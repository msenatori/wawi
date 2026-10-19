"""The brain's ``web_*`` tools: she browses in her own little computer.

Every handler returns ``(ok, speak, data)`` plus, for :func:`web_look`, a
JPEG (base64) the brain hands the model as an image block (wawi/brain.py).
Coordinates are pixels of that screenshot (1:1 with the page's CSS pixels).

Safety lives HERE, not in the prompt (see wawi/webvm/safety.py): a click or
an Enter that logs in, buys, pays, sends or posts, and typing personal data,
park the action and ask the owner; password fields are never typed into; only
public http(s) addresses open; nothing is ever downloaded to the Mac (the
container refuses downloads, and only screenshots and text come back).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from wawi.webvm import safety
from wawi.webvm.client import Unreachable, WebVMError

log = logging.getLogger(__name__)

PREFIX = "web_"
MAX_TEXT = 4000
MAX_ELEMENTS = 40

_EMPTY = {"type": "object", "properties": {}}

SPECS: list[dict[str, Any]] = [
    {"name": "web_open", "animation": "laptop",
     "example": "Wawi, buscame vuelos a Córdoba para el viernes",
     "ability": "Navegar la web por vos en mi compu",
     "description": (
         "Tu propia compu (un navegador en un servidor): navegá la web POR el dueño "
         "cuando pida buscar, comparar, reservar o mirar algo en un sitio. Abrí una URL "
         "(o https://www.google.com/search?q=...). Después web_look para ver. Antes de "
         "empezar decí una frase corta. Al terminar, contá lo que encontraste (o usá la "
         "pizarra) y llamá web_done."),
     "input_schema": {"type": "object", "properties": {"url": {"type": "string"}},
                      "required": ["url"]}},
    {"name": "web_look", "animation": "laptop",
     "description": "Mirá tu compu: imagen de la página, url, título y elementos clicables con su x,y (píxeles de la imagen).",
     "input_schema": _EMPTY},
    {"name": "web_click", "animation": "laptop",
     "description": "Click en x,y de la última imagen. Si es iniciar sesión, comprar, pagar, mandar o publicar, pide el sí del dueño.",
     "input_schema": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}},
    {"name": "web_type", "animation": "laptop",
     "description": "Escribí en el campo con foco (hacé click antes). submit=true aprieta Enter. Nunca contraseñas.",
     "input_schema": {"type": "object", "properties": {
         "text": {"type": "string"}, "submit": {"type": "boolean"}}, "required": ["text"]}},
    {"name": "web_key", "animation": "laptop",
     "description": "Apretá una tecla: Enter, Tab, Escape, PageDown, ArrowDown…",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
    {"name": "web_scroll", "animation": "laptop",
     "description": "Scroll en píxeles: dy>0 baja, dy<0 sube.",
     "input_schema": {"type": "object", "properties": {
         "dy": {"type": "number"}, "dx": {"type": "number"}}, "required": ["dy"]}},
    {"name": "web_back", "animation": "laptop", "description": "Volvé a la página anterior.",
     "input_schema": _EMPTY},
    {"name": "web_read", "animation": "laptop",
     "description": "Leé el texto de la página actual (más barato que mirar).",
     "input_schema": _EMPTY},
    {"name": "web_done", "animation": None,
     "description": "Terminaste con tu compu: dejá de mostrarla.", "input_schema": _EMPTY},
]

NAMES = tuple(spec["name"] for spec in SPECS)


class Result:
    """``(ok, speak, data)`` and an optional screenshot (base64 JPEG)."""

    __slots__ = ("ok", "speak", "data", "image_b64")

    def __init__(self, ok: bool, speak: str, data: dict | None = None,
                 image_b64: str = "") -> None:
        self.ok, self.speak, self.data, self.image_b64 = ok, speak, data, image_b64


def _service():
    from wawi import webvm

    return webvm.get()


def _page(payload: dict) -> dict:
    return {"url": str(payload.get("url", ""))[:300], "title": str(payload.get("title", ""))[:120]}


def _guarded(fn: Callable[[Any, Any], Result]) -> Callable[[dict], Result]:
    """Common shape: a service, a running computer, errors as sentences."""

    def run(args: dict) -> Result:
        service = _service()
        if service is None:
            return Result(False, "No tengo mi compu configurada.", {"error": "webvm_disabled"})
        try:
            try:
                result = fn(service, service.ensure(), args)
            except Unreachable:
                # It may have stopped on its own (idle): start it once more.
                service.lost()
                result = fn(service, service.ensure(), args)
        except Unreachable as exc:
            service.lost()
            return Result(False, str(exc) or "No llego a mi compu.", {"error": "unreachable"})
        except WebVMError as exc:
            return Result(False, str(exc), {"error": "webvm"})
        service.touch()
        if isinstance(result.data, dict) and result.data.get("url"):
            service.note_page(result.data)
        return result

    return run


def _num(args: dict, key: str, default: float | None = None) -> float:
    value = args.get(key, default)
    if value is None:
        raise WebVMError(f"Me falta {key}.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise WebVMError(f"{key} tiene que ser un número.") from None


def _open(service, client, args: dict) -> Result:
    url, reason = safety.check_url(str(args.get("url", "")), service.settings.allow_lan)
    if url is None:
        return Result(False, reason, {"error": "url_refused"})
    got = client.open(url)
    return Result(True, "Abrí la página.", {**_page(got), "status": got.get("status"),
                                             "next": "web_look o web_read"})


def _look(service, client, args: dict) -> Result:
    got = client.screenshot()
    elements = [{k: e.get(k) for k in ("x", "y", "tag", "text") if e.get(k) not in (None, "")}
                for e in (got.get("elements") or [])[:MAX_ELEMENTS]]
    data = {**_page(got), "width": got.get("width"), "height": got.get("height"),
            "elements": elements}
    return Result(True, "Miro mi compu.", data, str(got.get("image_b64", "")))


def _click(service, client, args: dict) -> Result:
    x, y = _num(args, "x"), _num(args, "y")
    info = client.element_at(x, y)
    element = info.get("element")
    ask = safety.needs_asking(safety.click_kinds(element), service.settings.ask_before)
    if ask:
        what = safety.describe(element)
        service.ask(ask, f"Click en «{what}»" if what else "Click", str(info.get("url", "")),
                    lambda: _do_click(client, x, y))
        return _asked(ask, what)
    ok, speak, data = _do_click(client, x, y)
    return Result(ok, speak, {**data, "clicked": safety.describe(element)})


def _do_click(client, x: float, y: float) -> tuple[bool, str, dict]:
    got = client.click(x, y)
    return True, "Listo, hice click.", _page(got)


def _asked(kinds: list[str], what: str) -> Result:
    return Result(True, safety.question(kinds, what),
                  {"pending": True, "done": False, "kinds": kinds,
                   "note": "Decí la pregunta y esperá el sí del dueño en su próxima frase; no sigas."})


def _type(service, client, args: dict) -> Result:
    text = str(args.get("text", ""))
    submit = bool(args.get("submit", False))
    if not text and not submit:
        return Result(False, "No me dijiste qué escribir.", {"error": "empty"})
    focus = client.focused()
    element = focus.get("element")
    if safety.is_password_field(element):
        return Result(False, "Las contraseñas no las escribo yo: si querés, tomá el control "
                             "en la ventana grande de mi compu.", {"error": "password_field"})
    kinds = safety.text_kinds(text, element)
    if submit:
        kinds |= safety.submit_kinds(element)
    ask = safety.needs_asking(kinds, service.settings.ask_before)
    if ask:
        service.ask(ask, f"Escribir «{text[:40]}»" + (" y enviar" if submit else ""),
                    str(focus.get("url", "")), lambda: _do_type(client, text, submit))
        return _asked(ask, text[:40])
    ok, speak, data = _do_type(client, text, submit)
    return Result(ok, speak, data)


def _do_type(client, text: str, submit: bool) -> tuple[bool, str, dict]:
    got = client.type(text, submit)
    return True, "Listo, lo escribí." + (" Y apreté Enter." if submit else ""), _page(got)


_SUBMIT_KEYS = {"enter", "return", "numpadenter"}


def _key(service, client, args: dict) -> Result:
    name = str(args.get("name", "")).strip()
    if not name:
        return Result(False, "¿Qué tecla?", {"error": "empty"})
    if name.lower() in _SUBMIT_KEYS:
        focus = client.focused()
        ask = safety.needs_asking(safety.submit_kinds(focus.get("element")),
                                  service.settings.ask_before)
        if ask:
            service.ask(ask, "Apretar Enter en el formulario", str(focus.get("url", "")),
                        lambda: _do_key(client, name))
            return _asked(ask, "")
    ok, speak, data = _do_key(client, name)
    return Result(ok, speak, data)


def _do_key(client, name: str) -> tuple[bool, str, dict]:
    got = client.key(name)
    return True, f"Apreté {name}.", _page(got)


def _scroll(service, client, args: dict) -> Result:
    got = client.scroll(_num(args, "dx", 0), _num(args, "dy", 0))
    return Result(True, "Moví la página.", _page(got))


def _back(service, client, args: dict) -> Result:
    got = client.back()
    return Result(True, "Volví atrás.", _page(got))


def _read(service, client, args: dict) -> Result:
    got = client.text(MAX_TEXT)
    return Result(True, "Leí la página.", {**_page(got), "text": str(got.get("text", ""))[:MAX_TEXT],
                                           "truncated": bool(got.get("truncated"))})


def web_done(args: dict) -> Result:
    service = _service()
    if service is None:
        return Result(False, "No tengo mi compu configurada.", {"error": "webvm_disabled"})
    service.done()
    return Result(True, "Cierro mi compu.", None)


HANDLERS: dict[str, Callable[[dict], Result]] = {
    "web_open": _guarded(_open),
    "web_look": _guarded(_look),
    "web_click": _guarded(_click),
    "web_type": _guarded(_type),
    "web_key": _guarded(_key),
    "web_scroll": _guarded(_scroll),
    "web_back": _guarded(_back),
    "web_read": _guarded(_read),
    "web_done": web_done,
}


def is_web_tool(name: str) -> bool:
    return name in HANDLERS
