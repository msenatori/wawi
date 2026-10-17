"""Everything that changes when the assistant switches language.

Switching Wawi to English is not one setting, and that is the whole reason
this module exists. Four things have to move together or the result is worse
than not switching at all:

1. **STT.** Whisper is *told* the language (``stt.transcribe(language=...)``).
   Left at ``es`` it will map English audio onto Spanish phonetics rather than
   fail honestly.
2. **The system prompt**, which orders the answer's language out loud.
3. **The TTS voice.** Kokoro's ``ef_dora`` is Spanish. Handed English text it
   reads it with Spanish phonemes — the single most audible way to get this
   half-done.
4. **The fixed strings the assistant says itself** — the connection error and
   the "I didn't hear you". These never reach the model, so nothing else
   would translate them.

What deliberately does NOT move: the Spanish text inside :mod:`wawi.tools`.
Those strings are fed to the MODEL, not spoken, and it re-renders them in
whatever language the prompt ordered. Translating them would buy nothing and
double the surface to maintain.
"""

from __future__ import annotations

#: Languages with a complete set of strings below. Whisper understands far
#: more, but a language listed here is one the assistant can actually *speak*,
#: which is a stronger claim.
SUPPORTED = ("es", "en")

DEFAULT = "es"

#: Kokoro voice used when ``tts.voice`` is left at ``auto``. Picking the voice
#: from the language is what keeps a half-configured switch from producing
#: English read aloud by a Spanish speaker.
VOICE = {
    "es": "ef_dora",
    "en": "af_heart",
}

#: Kokoro's phonemiser code. Separate from our own language code because the
#: two vocabularies differ: Kokoro says "en-us", we say "en". Passing the wrong
#: one does not fail — it reads English words with Spanish phonetics, which is
#: exactly the bug the voice table above exists to prevent.
TTS_LANG = {
    "es": "es",
    "en": "en-us",
}

#: Spoken when the brain call fails. Short on purpose: it is heard, not read.
ERROR_TEXT = {
    "es": "no pude conectarme",
    "en": "I could not connect",
}

#: Spoken when the turn ran out of steps before it reached an answer. Distinct
#: from :data:`ERROR_TEXT` because the connection was never the problem: saying
#: "I could not connect" here sends the user to check their network over a
#: model that spent its budget on tools and never spoke.
BUDGET_TEXT = {
    "es": "Me enredé con las herramientas y me quedé sin pasos. Preguntame de nuevo.",
    "en": "I got stuck using tools and ran out of steps. Ask me again.",
}

#: Spoken when the microphone opened but captured silence.
NO_AUDIO_TEXT = {
    "es": "No te escuché. ¿Probás de nuevo?",
    "en": "I did not hear you. Try again?",
}

#: Said when she is called by name and nothing else ("Wawi", "che Wawi"):
#: a question, so the channel stays open for the actual request.
SUMMON_TEXT = {
    "es": "¿Sí?",
    "en": "Yes?",
}

#: Printed (never spoken) when a turn is dropped for silence.
NO_AUDIO_NOTE = {
    "es": "(no se escuchó nada)",
    "en": "(heard nothing)",
}

#: Seeds Whisper's decoder with a sample of the orthography we want back.
#: Whisper continues the style of whatever precedes it, so this is not a
#: keyword list -- it is a written example. It exists because of a real bug:
#: Whisper transcribed the accented "si" as bare "si", which in Spanish is
#: the word "if", and the approval parser read "mira si te parece" as consent
#: to run a shell command. Showing the accent here is what makes the decoder
#: emit it. The assistant's own name is appended by :func:`stt_prompt`, since
#: it is configurable and a name the decoder has never seen comes back mangled.
#:
#: No questions in the seed: it used to contain "¿Está bien así?", and Whisper
#: handed silence or noise returned exactly that ("¿Está bien?") as if someone
#: had said it. A seed sentence is what the decoder falls back to on empty
#: audio, so it must not be something that reads as a real turn.
STT_PROMPT = {
    "es": (
        "S\u00ed, dale, corr\u00e9 los tests. No, mejor no. "
        "Abr\u00ed el archivo y revis\u00e1 el c\u00f3digo."
    ),
    "en": (
        "Yes, go ahead, run the tests. No, better not. "
        "Open the file and review the code."
    ),
}


def stt_prompt(language: str, assistant_name: str = "") -> str:
    """Return the decoder seed for *language*, naming the assistant.

    The name is worth its tokens: a wake word the decoder has never seen comes
    back as whatever it sounds like, and every downstream matcher then misses.
    """
    text = STT_PROMPT.get(language, STT_PROMPT[DEFAULT])
    name = (assistant_name or "").strip()
    return f"{text} {name}." if name else text


#: Appended by the Ollama backend, which has no tools and must say so rather
#: than let a blind answer look like a full one.
DEGRADED_SUFFIX = {
    "es": "Aclaro que estoy en modo local, sin herramientas.",
    "en": "Note that I am on a local model, with no tools.",
}


def normalise(language: str | None) -> str:
    """Return a supported language code, falling back to :data:`DEFAULT`.

    Callers pass whatever is in config. An unknown code degrades to Spanish
    rather than raising: config validation is the place that refuses a typo,
    and by the time a turn is running, answering in the wrong language beats
    not answering at all.
    """
    code = (language or "").strip().lower()
    return code if code in SUPPORTED else DEFAULT


def get(table: dict[str, str], language: str) -> str:
    """Look one string up, tolerating an unsupported language."""
    return table[normalise(language)]


def voice_for(language: str) -> str:
    """The Kokoro voice that matches *language*."""
    return VOICE[normalise(language)]


#: Spoken by the ORCHESTRATOR when the executor agent wants approval for a
#: tool call. The tool name and its target are substituted verbatim, so the
#: sentence has to survive ending in a shell command read out loud.
APPROVAL_ASK = {
    "es": "El agente quiere hacer esto: {action}. ¿Lo dejo?",
    "en": "The agent wants to do this: {action}. Should I let it?",
}

#: Spoken when the executor asks a real question. Prefixed rather than said
#: bare so the user can tell it apart from Wawi asking on its own account —
#: the answer goes somewhere else, and that difference is audible or it is not
#: there at all.
TASK_ASK = {
    "es": "El agente pregunta: {question}",
    "en": "The agent asks: {question}",
}

#: Spoken when the work ends. Nothing else reports it: a task produces no turn,
#: so without this a finished run is indistinguishable from a stalled one.
TASK_DONE = {
    "es": "Terminé la tarea {id}.",
    "en": "Task {id} is done.",
}

TASK_FAILED = {
    "es": "La tarea {id} falló.",
    "en": "Task {id} failed.",
}

TASK_STOPPED = {
    "es": "Corté la tarea {id}.",
    "en": "Task {id} was stopped.",
}

#: Asked ONCE on startup when the last run died with work in flight. {what}
#: is the task's goal or the user's own words, clipped to be heard.
RESUME_ASK_TASK = {
    "es": "Me cortaron mientras hacía {what}, ¿lo retomo?",
    "en": "I was cut off while doing {what}. Shall I pick it up?",
}

RESUME_ASK_TURN = {
    "es": "Me cortaron antes de responderte «{what}», ¿lo retomo?",
    "en": "I was cut off before answering \"{what}\". Shall I pick it up?",
}

#: The whole reply to a yes for a task: the executor speaks for itself when
#: it finishes, as any task does.
RESUME_TASK_OK = {
    "es": "Dale, lo retomo.",
    "en": "Okay, picking it up.",
}

#: Spoken words accepted as yes and no. Both languages are accepted regardless
#: of the configured one, on purpose: people answering a machine out loud
#: code-switch constantly ("dale", "ok", "yes"), and the cost of not
#: recognising a yes is only a repeated question.
_AFFIRMATIVE = frozenset(
    {
        "si", "sí", "sip", "claro", "dale", "obvio", "correcto", "exacto",
        "ok", "okay", "oka", "bueno", "listo", "hacelo", "hazlo", "adelante",
        "va", "vale", "afirmativo", "por supuesto", "yes", "yeah", "yep",
        "yup", "sure", "go", "do", "please", "confirm", "approved", "allow",
    }
)

_NEGATIVE = frozenset(
    {
        "no", "nop", "nope", "negativo", "para", "pará", "cancelá", "cancela",
        "cancel", "stop", "don't", "dont", "never", "nunca", "mejor no",
        "abortá", "aborta", "abort", "deny", "denied",
    }
)

#: Words that only count as an answer when they OPEN the sentence. Unaccented
#: "si" is the Spanish conjunction "if", and Whisper drops accents constantly —
#: so "mirá si te parece" would otherwise read as approval. As the first word
#: it is almost always the answer; anywhere else it is almost always the
#: conjunction, and guessing wrong in that direction is what runs the command.
_FIRST_WORD_ONLY = frozenset({"si"})

_YESNO_STRIP = ",.;:!?¡¿\"'“”…"


def affirmative(text: str) -> bool | None:
    """Read a spoken yes/no. None when it is neither.

    None is not a failure mode to smooth over: an answer that is neither is
    exactly when the caller must NOT guess. Everything that acts on this
    treats None as a no, which is why this reports it separately instead of
    returning a bare bool.
    """
    words = [w.strip(_YESNO_STRIP) for w in (text or "").lower().split()]
    words = [w for w in words if w]
    if not words:
        return None
    # Only the opening words are read. A yes buried in the middle of a
    # sentence ("no, en el proyecto sí está") is not an answer to the question
    # that was asked, and treating it as one is how a denial becomes approval.
    for index, word in enumerate(words[:3]):
        if word in _FIRST_WORD_ONLY and index > 0:
            continue
        if word in _AFFIRMATIVE:
            return True
        if word in _NEGATIVE:
            return False
    return None
