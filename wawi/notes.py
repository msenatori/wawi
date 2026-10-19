"""Taking notes: Wawi as a dictation recorder, no model involved.

"Wawi, tomá nota" puts her in note mode: from then on the microphone listens
with no wake word, and every utterance the listener closes is transcribed by
the same Whisper STT as a turn and appended, with its time, to a Markdown
file. Nothing goes to the brain, nothing is spoken back, no follow-up window
opens. It ends on the giant STOP button, the menu item, "Wawi, pará", the
microphone switch, or after :data:`IDLE_STOP_S` of silence.

The commands are recognised HERE, on the transcript, before any model is
asked -- the same arrangement as :mod:`wawi.wakemode`: starting a recorder is
not a question, it costs nothing, and it must work offline.

The file is written as it goes (append, flush and fsync per line), so a crash
in the middle of a meeting costs at most the sentence being transcribed.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from wawi.wake import normalize

log = logging.getLogger(__name__)

#: Where notes go when config.yaml says nothing (``notes.dir``).
DEFAULT_DIR = "~/Documents/Wawi/Notas"
#: A session with no speech for this long ends on its own (``notes.idle_stop_min``).
IDLE_STOP_S = 30 * 60.0

#: Said when the session starts, and when it ends. Short on purpose: in a
#: meeting anything longer is an interruption.
START_TEXT = "Te escucho, anoto todo."
STOP_TEXT = "Listo, guardé la nota."
EMPTY_TEXT = "Listo, no escuché nada para anotar."
BUSY_TEXT = "Ya estoy anotando."
FAILED_TEXT = "No pude abrir el archivo de notas."
_OWN_LINES = frozenset(normalize(t) for t in (START_TEXT, STOP_TEXT, EMPTY_TEXT, BUSY_TEXT))


def is_own_line(text: str) -> bool:
    """*text* is one of her own note sentences, heard back through the
    speakers: never a line of the note."""
    return normalize(text) in _OWN_LINES


# Politeness around a command: "dale, tomá nota por favor".
_LEAD = r"(?:(?:che|dale|bueno|ok|okay|ahora|wawi|por favor|porfa)\s+)*"
_TAIL = r"(?:\s+(?:por favor|porfa|wawi|ahora|ya|dale|gracias))*"

#: Note mode on. Matched against the WHOLE normalized utterance (no accents,
#: lowercase): "anotá esto: comprar pan" is a request for the model, only
#: "anotá esto" on its own starts the recorder.
_START = (
    r"(?:podes |podrias |quiero que )?(?:toma|tomame|tomes|tomar) (?:una |unas )?notas?"
    r"(?: de (?:todo|esto|la charla|la reunion|lo que (?:digo|hablamos)))?",
    r"anota(?:me)? (?:esto|todo|lo que (?:digo|hablamos))",
    r"(?:en |pasa a |activa (?:el )?)?modo notas?",
    r"graba(?:me)? (?:la charla|esto|la reunion|la conversacion|todo|lo que (?:digo|hablamos))",
    r"(?:empeza|arranca|ponete) a grabar",
)

#: Stop commands that cannot be anything else, even with words after them
#: ("Wawi, listo por hoy").
_STOP_CLEAR = (
    r"(?:para|pare|deja|termina) de (?:anotar|grabar|escuchar|tomar notas?)",
    r"termina(?:mos)?(?: la nota| la grabacion)?",
    r"basta", r"listo", r"frena", r"corta(?:la)?", r"stop", r"fin de la nota",
)
#: Stop commands that are also ordinary words ("para mañana tengo…"): only
#: when they are the whole utterance. Not "ya está": said in passing in the
#: middle of a dictation, it ended a note after fifteen seconds.
_STOP_ALONE = (r"para", r"pare", r"eso es todo")
#: Words allowed after a clear stop command when the name was said first.
_STOP_TAIL_WORDS = 3


def _whole(patterns: tuple[str, ...], words: str) -> bool:
    return any(re.fullmatch(_LEAD + p + _TAIL, words) for p in patterns)


def parse_start(text: str) -> bool:
    """True when *text* (wake word already taken off) asks to take notes."""
    words = normalize(text)
    return bool(words) and _whole(_START, words)


def parse_stop(text: str, named: bool = False) -> bool:
    """True when *text* ends the note session.

    *text* has the wake word already taken off; *named* says whether it was
    there. Without the name only the bare command counts ("pará", "listo"):
    anything longer is dictation. With it, a clear command may run on a few
    words ("Wawi, listo por hoy") -- but never the ambiguous "para".
    """
    words = normalize(text)
    if not words:
        return False
    if _whole(_STOP_CLEAR + _STOP_ALONE, words):
        return True
    if not named:
        return False
    for p in _STOP_CLEAR:
        m = re.match(_LEAD + p + r"\b", words)
        if m and len(words[m.end():].split()) <= _STOP_TAIL_WORDS:
            return True
    return False


class NoteFile:
    """One Markdown note on disk, written line by line.

    ``YYYY-MM-DD HH-MM.md`` in *directory* (created if needed); a second note
    in the same minute gets `` (2)`` and so on rather than overwriting the
    first. Every :meth:`append` is flushed and fsynced before it returns.
    """

    def __init__(self, directory: str | Path, *, now: datetime | None = None) -> None:
        stamp = now or datetime.now()
        folder = Path(directory).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        base = stamp.strftime("%Y-%m-%d %H-%M")
        path, n = folder / f"{base}.md", 2
        while path.exists():
            path, n = folder / f"{base} ({n}).md", n + 1
        self.path = path
        self.lines = 0
        self._lock = threading.Lock()
        # "x": never clobber a note that appeared between the check and here.
        self._fh = open(path, "x", encoding="utf-8")
        self._write(f"# Nota del {stamp:%d/%m/%Y} a las {stamp:%H:%M}\n\n")

    def _write(self, text: str) -> None:
        self._fh.write(text)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def append(self, text: str, when: datetime | None = None) -> bool:
        """Add one timestamped line. False (and nothing written) when closed
        or when *text* is blank."""
        text = " ".join(str(text).split())
        if not text:
            return False
        stamp = when or datetime.now()
        with self._lock:
            if self._fh is None:
                return False
            self._write(f"- **{stamp:%H:%M:%S}** {text}\n")
            self.lines += 1
        return True

    def close(self, when: datetime | None = None) -> None:
        """Write the closing line and close the file. Twice is harmless."""
        stamp = when or datetime.now()
        with self._lock:
            if self._fh is None:
                return
            try:
                if not self.lines:
                    self._write("_No se escuchó nada para anotar._\n")
                self._write(f"\n_Terminada a las {stamp:%H:%M:%S}._\n")
            finally:
                self._fh.close()
                self._fh = None

    @property
    def closed(self) -> bool:
        return self._fh is None

    def read(self) -> str:
        """The note as it is on disk ("" when it cannot be read)."""
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            return ""


class NoteSession:
    """A note being taken: its file, its clock, and what to put back after.

    *restore* is whatever the caller needs to undo when it ends (the
    listener's previous mode); this class only keeps it.
    """

    def __init__(self, note: NoteFile, *, restore: object = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.note = note
        self.restore = restore
        self._clock = clock
        self.started = clock()
        self.last_heard = self.started
        #: Set when the session ends; its worker and watchdog wait on it.
        self.ended = threading.Event()
        #: Whether ending it says "listo, guardé la nota".
        self.speak_on_end = True

    @property
    def lines(self) -> int:
        return self.note.lines

    def elapsed(self) -> float:
        return max(0.0, self._clock() - self.started)

    def idle_for(self) -> float:
        return max(0.0, self._clock() - self.last_heard)

    def heard(self) -> None:
        self.last_heard = self._clock()


def format_elapsed(seconds: float) -> str:
    """``MM:SS``, or ``H:MM:SS`` past the hour: the STOP button's timer."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
