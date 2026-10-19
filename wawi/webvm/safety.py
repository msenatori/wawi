"""What she may do on her own in her little computer, and what she asks first.

Pure functions over what the control API reports about an element (its tag,
text, form…), so every rule here is testable without a browser. The tool
layer (wawi/webvm/tools.py) calls them BEFORE acting:

* a click on something that logs in, buys, pays, sends a message or posts in
  public, or Enter / submit in such a form, waits for the owner's yes;
* typing what looks like personal data (an email, a phone, a card, a DNI)
  waits for the owner's yes;
* a password field is never typed into, whoever asks: the owner takes over in
  the big window for that;
* only http(s) addresses on the internet open (no file:, no LAN, no localhost).

Which kinds ask is ``webvm.ask_before`` (all of them by default).
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from typing import Any
from urllib.parse import urlsplit

#: Spoken name of each kind, for «¿… ?» questions.
KIND_WORDS = {
    "login": "iniciar sesión",
    "purchase": "comprar o reservar",
    "payment": "pagar",
    "message": "mandar un mensaje",
    "post": "publicar",
    "personal_data": "escribir datos personales",
}

# Words on a button / link that mean the click does the thing. Checked against
# the element's own text (accent-folded, lower case), as whole words.
_WORDS: dict[str, tuple[str, ...]] = {
    "payment": ("pagar", "pago", "abonar", "pay", "payment", "checkout", "finalizar compra",
                "confirmar pago", "place order", "realizar pedido"),
    "purchase": ("comprar", "compra", "buy", "purchase", "reservar", "reserva", "book",
                 "book now", "confirmar compra", "confirmar reserva", "suscribirme",
                 "suscribirse", "subscribe", "contratar", "order now", "pedir ahora"),
    "login": ("iniciar sesion", "ingresar", "log in", "login", "sign in", "signin",
              "acceder", "entrar", "registrarme", "registrarse", "crear cuenta", "sign up",
              "signup", "continuar con google", "continue with google"),
    "message": ("enviar", "send", "mandar", "responder", "reply", "enviar mensaje",
                "send message", "enviar correo", "send email"),
    "post": ("publicar", "post", "postear", "tweet", "twittear", "compartir", "share",
             "comentar", "comment", "opinar", "enviar resena", "submit review", "subir"),
}

# Form fields that mean money or identity.
_PAYMENT_FIELDS = ("cc-", "card", "tarjeta", "cvv", "cvc", "expir", "vencimiento", "iban")
_PERSONAL_FIELDS = ("email", "tel", "phone", "telefono", "dni", "cuit", "cuil", "passport",
                    "pasaporte", "address", "direccion", "street", "postal", "birth",
                    "nacimiento", "given-name", "family-name", "apellido")
_PERSONAL_AUTOCOMPLETE = ("email", "tel", "name", "given-name", "family-name", "street-address",
                          "address-line1", "postal-code", "bday", "cc-number", "cc-csc")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d[\s-]?){8,}")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_DNI = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}\b")
_CUIT = re.compile(r"\b\d{2}-?\d{8}-?\d\b")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9@.+ -]", " ", text.lower()).split())


def _has_word(text: str, words: tuple[str, ...]) -> bool:
    folded = f" {_fold(text)} "
    return any(f" {w} " in folded for w in words)


def _form_kinds(form: dict | None) -> set[str]:
    """What submitting *form* would do, from its fields alone."""
    if not form:
        return set()
    kinds: set[str] = set()
    names = " ".join(map(str, form.get("names") or ()))
    auto = " ".join(map(str, form.get("autocomplete") or ()))
    types = {str(t) for t in form.get("input_types") or ()}
    blob = f"{names} {auto}".lower()
    if form.get("has_password") or "password" in types or "current-password" in blob:
        kinds.add("login")
    if any(k in blob for k in _PAYMENT_FIELDS):
        kinds.add("payment")
    if "email" in types or "tel" in types or any(k in blob for k in _PERSONAL_FIELDS):
        kinds.add("personal_data")
    for submit in form.get("submits") or ():
        kinds |= _text_kinds(str(submit))
    return kinds


def _text_kinds(text: str) -> set[str]:
    return {kind for kind, words in _WORDS.items() if _has_word(text, words)}


def click_kinds(element: dict | None) -> set[str]:
    """What clicking *element* would do that needs the owner's yes."""
    if not element:
        return set()
    tag = str(element.get("tag", ""))
    etype = str(element.get("type", ""))
    role = str(element.get("role", ""))
    text = " ".join(str(element.get(k, "")) for k in ("text", "name", "id"))
    actionable = tag in ("button", "a") or role in ("button", "link", "menuitem") or \
        (tag == "input" and etype in ("submit", "button", "image"))
    if not actionable:
        return set()
    kinds = _text_kinds(text)
    submits = tag == "button" and etype in ("", "submit") or etype in ("submit", "image")
    if submits and element.get("form"):
        form_kinds = _form_kinds(element.get("form"))
        # A search box's form has no email/password: nothing there to ask about.
        kinds |= form_kinds & {"login", "payment"}
        if "personal_data" in form_kinds and kinds & {"message", "post", "purchase"}:
            kinds.add("personal_data")
    return kinds


def submit_kinds(element: dict | None) -> set[str]:
    """What pressing Enter in *element* (the focused one) would submit."""
    if not element:
        return set()
    form = element.get("form")
    if not form:
        return set()
    kinds = _form_kinds(form)
    if str(element.get("tag")) == "textarea":
        return set()  # Enter in a textarea is a new line
    return kinds


def is_password_field(element: dict | None) -> bool:
    if not element:
        return False
    return str(element.get("type", "")) == "password" or \
        "password" in str(element.get("autocomplete", ""))


def text_kinds(text: str, element: dict | None = None) -> set[str]:
    """Typing *text* (into *element*) is personal data?"""
    kinds: set[str] = set()
    t = str(text or "")
    if _EMAIL.search(t) or _CARD.search(t) or _DNI.search(t) or _CUIT.search(t) or \
            _PHONE.search(t):
        kinds.add("personal_data")
    if element:
        auto = str(element.get("autocomplete", ""))
        etype = str(element.get("type", ""))
        field = f"{element.get('name', '')} {element.get('id', '')}".lower()
        if etype in ("email", "tel") or any(a == auto or auto.startswith(a) for a in _PERSONAL_AUTOCOMPLETE) \
                or any(k in field for k in _PERSONAL_FIELDS):
            kinds.add("personal_data")
        if any(k in f"{field} {auto}" for k in _PAYMENT_FIELDS):
            kinds.add("payment")
    return kinds


def needs_asking(kinds: set[str], ask_before: tuple[str, ...] | list[str]) -> list[str]:
    """The kinds in *kinds* the owner wants to be asked about, in a fixed order."""
    order = ("payment", "purchase", "login", "message", "post", "personal_data")
    return [k for k in order if k in kinds and k in ask_before]


def question(kinds: list[str], what: str = "") -> str:
    words = " y ".join(KIND_WORDS[k] for k in kinds[:2])
    target = f" ({what[:60]})" if what else ""
    return f"Esto es para {words}{target}. ¿Lo hago?"


_PRIVATE_NAMES = ("localhost", "host.docker.internal")
_PRIVATE_SUFFIXES = (".local", ".localhost", ".lan", ".internal", ".home.arpa")


def check_url(url: str, allow_lan: bool = False) -> tuple[str | None, str]:
    """``(normalized url, "")`` or ``(None, reason in Spanish)``."""
    url = str(url or "").strip()
    if not url:
        return None, "Decime qué página abro."
    if "://" not in url:
        if url.split(":", 1)[0].lower() in ("file", "javascript", "data", "chrome", "about"):
            return None, "Solo abro páginas web (http o https)."
        url = "https://" + url
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return None, "Solo abro páginas web (http o https)."
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return None, "Esa dirección no tiene sitio."
    if not allow_lan:
        if host in _PRIVATE_NAMES or host.endswith(_PRIVATE_SUFFIXES) or "." not in host:
            return None, "No abro direcciones de la red de casa desde mi compu."
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local
                               or ip.is_reserved or ip.is_unspecified):
            return None, "No abro direcciones de la red de casa desde mi compu."
    return url, ""


def describe(element: dict[str, Any] | None) -> str:
    """A few words naming *element*, for the question and the card."""
    if not element:
        return ""
    return " ".join(str(element.get("text") or element.get("name") or element.get("tag") or "").split())[:60]
