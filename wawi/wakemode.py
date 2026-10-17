"""Always-on vs wake phrase: the owner's live switch, by voice or from the menu.

``config.yaml`` says how Wawi starts (``wake.require_phrase``). Saying
"Wawi, usá palabra clave" or "escuchá siempre", or the menu-bar item, flips it
while she runs, and the choice is kept in a small gitignored file
(``wake_local.json`` by default, ``wake.local_file``) that is read back over
config.yaml on the next start -- the same arrangement as sensors_local.json:
the tracked config is the template, the local file is what YOU last said.

The voice command is recognised HERE, before any model is asked: switching a
microphone mode is not a question, it costs nothing, and it must work even
when the network does not.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from wawi.wake import normalize

log = logging.getLogger(__name__)

DEFAULT_LOCAL_FILE = "wake_local.json"

#: Longer than this and it is a sentence that mentions the mode, not the
#: command itself ("¿qué pasa si uso palabra clave y no me escucha?").
_MAX_COMMAND_WORDS = 9

#: Phrase mode on. Matched on normalized text (no accents, lowercase).
_TO_PHRASE = (
    r"\b(usa|activa|pone|prende|usemos|quiero)( la)? palabra (clave|de activacion)\b",
    r"\bmodo palabra (clave|de activacion)\b",
    r"\b(solo|nada mas) (cuando|si) te (llamo|nombro|digo wawi)\b",
    r"\bespera (a )?que te llame\b",
)
#: Always-on back.
_TO_ALWAYS = (
    r"\bescucha(me)? siempre\b",
    r"\bmodo siempre\b",
    r"\bsiempre escuchando\b",
    r"\b(saca|desactiva|apaga|deja)( la)? palabra (clave|de activacion)\b",
    r"\bsin palabra (clave|de activacion)\b",
)


def parse_command(text: str) -> bool | None:
    """True = wake-phrase mode, False = always-on, None = not this command."""
    words = normalize(text)
    if not words or len(words.split()) > _MAX_COMMAND_WORDS:
        return None
    for pattern in _TO_ALWAYS:
        if re.search(pattern, words):
            return False
    for pattern in _TO_PHRASE:
        if re.search(pattern, words):
            return True
    return None


def confirmation(require_phrase: bool, name: str = "Wawi") -> str:
    """What she says after switching."""
    if require_phrase:
        return f"Listo, ahora contesto solo cuando me decís {name}."
    return "Listo, te escucho siempre, sin palabra clave."


def read_local(path: Path | None) -> dict:
    """The saved mode, or {} when there is none (or it is unreadable)."""
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        log.warning("Ignoring unreadable %s", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    if isinstance(data.get("require_phrase"), bool):
        out["require_phrase"] = data["require_phrase"]
    return out


def write_local(path: Path | None, require_phrase: bool) -> bool:
    """Persist the mode atomically. Never raises; False when it could not."""
    if path is None:
        return False
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps({"require_phrase": bool(require_phrase)}) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        log.warning("Could not save the listening mode to %s", path, exc_info=True)
        return False
