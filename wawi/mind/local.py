"""The small local model: personality bubbles and event triage.

Most of Wawi's life runs on rules. A tiny model on the machine makes two of
those moments a little smarter, and nothing depends on it:

* **Bubbles.** Now and then she says something short and dog-like in a
  speech bubble. The model writes it from a compact state summary; a curated
  list writes it when the model is missing, slow, or says something unusable.
* **Triage.** For info/notice events, "now / later / ignore". Urgent always
  passes, and any failure answers None, which means "the reflex rules decide".

Every generation runs on ONE private worker thread. MLX binds its GPU stream
to a thread and is not safe to drive from several at once, and the callers
(bus listeners, the bubble ticker, the brain's tool loop) must never wait on
a model load. A call that would queue behind a running one is refused rather
than queued: a bubble that arrives late is worse than none.

Measured on the development Mac (M-series, mlx-lm 0.32, warm cache):
see :data:`BENCHMARK` -- the numbers are kept next to the code that relies on
them so a regression in a dependency has something to be compared with.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from wawi.reactions import Event

log = logging.getLogger(__name__)

#: scripts/bench_mind.py on the development Mac (Apple Silicon, mlx-lm 0.32,
#: weights already cached; first-ever load adds the download). All Apache-2.0.
#: The 0.6B models are fast but write broken Spanish and triage everything the
#: same way; 1.7B is the smallest that reads like a dog and not a help desk.
BENCHMARK = {
    "mlx-community/Qwen3-1.7B-4bit": "load 1.5 s, bubble ~190 ms, triage ~110 ms",
    "mlx-community/Qwen3-0.6B-4bit": "load 1.8 s, bubble ~70 ms, triage ~45 ms",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit": "load 0.6 s, bubble ~85 ms, triage ~40 ms",
    # Gemma terms of use (not Apache), ~190 MB. Fast, but NOT usable here:
    # over two runs 11 of 12 bubbles failed the parser (it echoes the system
    # prompt or rambles in made-up Spanish) and 0 of 8 triages gave a valid
    # word. Selectable for experiments; the canned lines win almost always.
    "mlx-community/gemma-3-270m-it-4bit": "load 2.7 s, bubble ~100 ms, triage ~25 ms (unusable)",
}

#: Local models offered in config.yaml.example, smallest first, with the
#: verdict of scripts/bench_mind.py. Any other MLX repo id also works.
LOCAL_MODEL_CHOICES = {
    "mlx-community/gemma-3-270m-it-4bit": "Gemma terms, ~190 MB: fast but writes nonsense",
    "mlx-community/Qwen3-0.6B-4bit": "Apache-2.0, ~350 MB: broken Spanish",
    "mlx-community/Qwen3-1.7B-4bit": "Apache-2.0, ~1 GB: the smallest that sounds like her",
}

BUBBLE_MAX_CHARS = 60

# --------------------------------------------------------------------------- #
# The model runner
# --------------------------------------------------------------------------- #


def _strip_thinking(text: str) -> str:
    """Qwen3 may still emit an (empty) think block; it is never the answer."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


class _MLXRunner:
    """mlx-lm, imported lazily: it is an OPTIONAL dependency."""

    def __init__(self, repo: str) -> None:
        self.repo = repo
        self._model: Any = None
        self._tok: Any = None

    def load(self) -> None:
        from mlx_lm import load  # type: ignore[import-not-found]

        self._model, self._tok = load(self.repo)

    def generate(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        from mlx_lm import generate  # type: ignore[import-not-found]
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            # Qwen3 thinks out loud by default; for a 60-character line that
            # is all latency and no value.
            prompt = self._tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            prompt = self._tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        return generate(
            self._model,
            self._tok,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        )


class _OllamaRunner:
    """The user's Ollama, when they already run the brain on it."""

    def __init__(self, model: str, url: str, timeout_s: float) -> None:
        self.model = model
        self.url = url
        self.timeout_s = timeout_s

    def load(self) -> None:  # Ollama loads on first request by itself
        return None

    def generate(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(((data or {}).get("message") or {}).get("content") or "")


class LocalModel:
    """A small model behind a single worker thread, with hard timeouts.

    *runner* is anything with ``load()`` and ``generate(system, user,
    max_tokens, temperature) -> str``; tests pass a fake. ``generate`` here
    never raises and never blocks longer than its timeout: it returns the
    text, or None.
    """

    def __init__(
        self,
        runner: Any,
        *,
        name: str = "",
        load_timeout_s: float = 60.0,
        call_timeout_s: float = 4.0,
    ) -> None:
        self.name = name or getattr(runner, "repo", "") or getattr(runner, "model", "")
        self._runner = runner
        self._load_timeout_s = load_timeout_s
        self._call_timeout_s = call_timeout_s
        self._jobs: queue.Queue[tuple[Callable[[], Any], Future] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._loaded = False
        self._load_future: Future | None = None
        self._busy = False
        #: Set once loading failed; the model is then off for the rest of the run.
        self.broken = False
        self.stats: dict[str, float] = {}

    # -- worker ---------------------------------------------------------
    def _ensure_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._work, name="wawi-mind-local", daemon=True)
            self._thread.start()

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            fn, fut = job
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - handed to the caller
                fut.set_exception(exc)

    def _submit(self, fn: Callable[[], Any]) -> Future:
        fut: Future = Future()
        self._ensure_thread()
        self._jobs.put((fn, fut))
        return fut

    def _do_load(self) -> None:
        t0 = time.perf_counter()
        self._runner.load()
        self.stats["load_s"] = time.perf_counter() - t0
        self._loaded = True
        log.info("mind: local model %s loaded in %.2fs", self.name, self.stats["load_s"])

    def warm_up(self) -> None:
        """Start loading in the background; returns at once."""
        with self._lock:
            if self._load_future is None and not self.broken:
                self._load_future = self._submit(self._do_load)

    def close(self) -> None:
        if self._thread is not None:
            self._jobs.put(None)

    # -- the one public call ---------------------------------------------
    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 40,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> str | None:
        if self.broken:
            return None
        self.warm_up()
        load_future = self._load_future
        assert load_future is not None
        if not load_future.done():
            # The FIRST call pays for the load, bounded. Later callers while
            # it is still loading get None at once instead of piling up.
            try:
                load_future.result(timeout=self._load_timeout_s)
            except FutureTimeout:
                log.warning("mind: local model still loading after %.0fs", self._load_timeout_s)
                return None
            except Exception:
                pass
        if load_future.exception() is not None:
            if not self.broken:
                log.warning("mind: local model %s failed to load; using canned lines "
                            "for the rest of this run", self.name,
                            exc_info=load_future.exception())
            self.broken = True
            return None

        with self._lock:
            if self._busy:
                return None
            self._busy = True

        def _run() -> str:
            t0 = time.perf_counter()
            try:
                return self._runner.generate(system, user, max_tokens, temperature)
            finally:
                self.stats["last_call_s"] = time.perf_counter() - t0
                with self._lock:
                    self._busy = False

        fut = self._submit(_run)
        try:
            out = fut.result(timeout=timeout if timeout is not None else self._call_timeout_s)
        except FutureTimeout:
            # The generation keeps going on the worker and clears _busy when
            # it ends; this caller simply stops waiting for it.
            log.info("mind: local model timed out")
            return None
        except Exception:
            log.warning("mind: local generation failed", exc_info=True)
            return None
        return _strip_thinking(str(out or ""))


def build_local_model(
    model_id: str,
    backend: str = "auto",
    *,
    brain_backend: str = "claude",
    ollama_model: str = "",
    ollama_url: str = "http://localhost:11434/api/chat",
    load_timeout_s: float = 60.0,
    call_timeout_s: float = 4.0,
) -> LocalModel | None:
    """The configured local model, or None when there is none to have.

    None for ``local_model: none`` and for MLX without mlx-lm installed --
    both mean "canned bubbles, rule triage", which is a complete Wawi.
    """
    if not model_id or model_id.lower() == "none":
        return None
    if backend == "auto":
        backend = "ollama" if brain_backend == "ollama" else "mlx"
    if backend == "ollama":
        runner: Any = _OllamaRunner(model_id or ollama_model, ollama_url, call_timeout_s + 1)
    else:
        try:
            import importlib.util

            if importlib.util.find_spec("mlx_lm") is None:
                raise ImportError("mlx_lm")
        except (ImportError, ValueError):
            log.warning(
                "mind: local_model=%s needs mlx-lm (pip install mlx-lm); using "
                "canned bubbles and rule triage", model_id,
            )
            return None
        runner = _MLXRunner(model_id)
    return LocalModel(
        runner, name=model_id, load_timeout_s=load_timeout_s, call_timeout_s=call_timeout_s
    )


# --------------------------------------------------------------------------- #
# State summary
# --------------------------------------------------------------------------- #


def part_of_day(hour: int) -> str:
    if 6 <= hour < 12:
        return "mañana"
    if 12 <= hour < 20:
        return "tarde"
    if 20 <= hour < 24:
        return "noche"
    return "madrugada"


def _level(value: Any) -> str | None:
    """0..1 needs as words; strings pass through."""
    if isinstance(value, (int, float)):
        return "alta" if value >= 0.66 else "media" if value >= 0.33 else "baja"
    if isinstance(value, str) and value.strip():
        return value.strip()[:20]
    return None


_NEED_NAMES = {
    "mood": "ánimo",
    "hunger": "hambre",
    "thirst": "sed",
    "energy": "energía",
    "boredom": "aburrimiento",
}


@dataclass(frozen=True)
class StateSummary:
    hour: int
    minute: int
    pet: dict[str, Any]
    recent: tuple[str, ...]
    #: Her self-model (wawi/instinct): what she noticed today about the
    #: owner, and what she knows of their routines. Short Spanish phrases.
    notes: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    #: Ready-made bubbles grounded in today, for the canned path.
    lines: tuple[str, ...] = ()

    def render(self) -> str:
        parts = [f"Son las {self.hour:02d}:{self.minute:02d} ({part_of_day(self.hour)})."]
        for key, label in _NEED_NAMES.items():
            level = _level(self.pet.get(key))
            if level:
                parts.append(f"{label.capitalize()}: {level}.")
        if self.recent:
            parts.append("Pasó hace poco: " + "; ".join(self.recent[-3:]) + ".")
        if self.notes:
            parts.append("Hoy notaste: " + "; ".join(self.notes[:3]) + ".")
        if self.facts:
            parts.append("Sabés de tu dueño: " + "; ".join(self.facts[:2]) + ".")
        return " ".join(parts)


def _phrases(value: Any, limit: int, chars: int = 80) -> tuple[str, ...]:
    """At most *limit* clean one-line strings (the self-model is untrusted
    shape-wise: it comes from another package)."""
    if not isinstance(value, (list, tuple)):
        return ()
    out = []
    for v in value:
        text = " ".join(str(v).split())[:chars]
        if text and not _RE_EMOJI_OR_MARKUP.search(text) and not _RE_URL.search(text):
            out.append(text)
    return tuple(out[:limit])


def summarize_state(events: Iterable[Event], pet: Any, now: float | None = None,
                    self_model: dict[str, Any] | None = None) -> StateSummary:
    lt = time.localtime(now if now is not None else time.time())
    recent = tuple((e.title or e.kind)[:50] for e in events if e.severity != "info" or e.kind.startswith("weather."))
    sm = self_model if isinstance(self_model, dict) else {}
    return StateSummary(lt.tm_hour, lt.tm_min, pet if isinstance(pet, dict) else {}, recent,
                        notes=_phrases(sm.get("notes"), 3), facts=_phrases(sm.get("facts"), 2),
                        lines=tuple(ln for ln in _phrases(sm.get("lines"), 3, 400)
                                    if len(ln) <= BUBBLE_MAX_CHARS and parse_bubble(ln)))


# --------------------------------------------------------------------------- #
# Bubbles
# --------------------------------------------------------------------------- #

#: Curated lines, by situation. Short on purpose: a bubble is read at a glance.
CANNED: dict[str, tuple[str, ...]] = {
    "mañana": (
        "¿Ya desayunaste? Yo sí. Bueno, no.",
        "Buen día. El café no se toma solo.",
        "Arrancamos. Yo superviso desde acá.",
    ),
    "tarde": (
        "Hora de la siesta perruna. Vos seguí.",
        "¿Un mate? Yo cebo con la mirada.",
        "Sigo acá, vigilando todo. Bah, casi.",
    ),
    "noche": (
        "Ya es de noche, eh. Yo lo digo nomás.",
        "¿Cerramos por hoy? Pregunto por un amigo.",
        "Las estrellas no se miran solas.",
    ),
    "madrugada": (
        "¿Vos no dormís? Yo tampoco, parece.",
        "A esta hora ni los gatos trabajan.",
    ),
    "hambre": (
        "Che, ¿y mi comida? Pregunto sin presionar.",
        "Mi plato está vacío. Lo dejo ahí.",
    ),
    "sed": ("Un poquito de agua no vendría mal.",),
    "aburrimiento": (
        "¿Jugamos? Una pelotita, una sola.",
        "Me aburro con mucha elegancia.",
    ),
    "lluvia": ("Llueve. Hoy no salgo ni loca.",),
    "general": (
        "Todo tranqui por acá.",
        "Guau. Digo, hola.",
        "Estoy acá por si me necesitás.",
        "Vos tranqui, yo cuido el escritorio.",
    ),
}

_RE_EMOJI_OR_MARKUP = re.compile(
    "[\U0001f000-\U0001ffff☀-➿️<>{}\\[\\]#*`|]"
)
_RE_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_LEAKS = ("como modelo", "asistente", "assistant", "language model", "lo siento")


def parse_bubble(raw: str | None) -> str | None:
    """Accept one short, clean, speakable line or nothing.

    Strict on purpose: a 0.6B model says odd things, and a bubble on the
    desktop is in plain view. Anything doubtful falls back to the curated list.
    """
    text = _strip_thinking(raw or "")
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = re.sub(r"^(wawi|respuesta|frase)\s*:\s*", "", line, flags=re.IGNORECASE)
    line = line.strip(" \t\"'“”«»‘’")
    if not (3 <= len(line) <= BUBBLE_MAX_CHARS):
        return None
    if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", line):
        return None
    if _RE_URL.search(line) or _RE_EMOJI_OR_MARKUP.search(line):
        return None
    low = line.lower()
    if any(leak in low for leak in _LEAKS):
        return None
    return line


# Few-shot on purpose: a 0.6B model without examples answers like a help
# desk ("¿Te gustaría algo más?"). Three lines of the right voice fix most of it.
BUBBLE_SYSTEM = (
    "Sos Wawi, una perrita mascota de escritorio: simpática, un poco irónica, "
    "cariñosa. Hablás en primera persona, como perra. Escribí UNA sola frase "
    "corta (máximo 60 caracteres) en español rioplatense, con voseo, sin "
    "emojis, sin comillas. Ejemplos:\n"
    "Che, ¿y mi comida? Pregunto sin presionar.\n"
    "Llueve. Hoy no salgo ni loca.\n"
    "Vos tranqui, yo cuido el escritorio.\n"
    "Si te cuento algo que notaste hoy de tu dueño, podés comentarlo con "
    "cariño (por ejemplo: Hoy codeaste 3 horas seguidas, ¿pausa?). "
    "Respondé solo la frase nueva."
)


class Personality:
    """Decides WHEN a bubble may appear and writes it.

    Rate-limited by a randomized gap (min..max interval) so she does not talk
    on a clock. Never during quiet mode; an unknown quiet state allows it,
    because a bubble is silent and easy to ignore.
    """

    def __init__(
        self,
        model: LocalModel | None,
        *,
        min_interval_s: float = 600.0,
        max_interval_s: float = 1200.0,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
        quiet: Callable[[], bool | None] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._model = model
        self._min = min_interval_s
        self._max = max(min_interval_s, max_interval_s)
        self._clock = clock
        self._rng = rng or random.Random()
        self._quiet = quiet
        self._timeout_s = timeout_s
        self._next_at = clock() + self._gap()
        self._recent_lines: list[str] = []

    def _gap(self) -> float:
        return self._rng.uniform(self._min, self._max)

    def due(self) -> bool:
        return self._clock() >= self._next_at

    def is_quiet(self) -> bool:
        if self._quiet is None:
            return False
        try:
            return bool(self._quiet())
        except Exception:
            return False

    def maybe_bubble(self, summary: StateSummary) -> str | None:
        """A line if one is due and allowed now, else None. Re-arms the timer."""
        if not self.due() or self.is_quiet():
            return None
        self._next_at = self._clock() + self._gap()
        return self.line(summary)

    def line(self, summary: StateSummary) -> str:
        """Always returns a usable line: model first, curated list second."""
        if self._model is not None:
            raw = self._model.generate(
                BUBBLE_SYSTEM,
                summary.render(),
                max_tokens=40,
                temperature=0.8,
                timeout=self._timeout_s,
            )
            parsed = parse_bubble(raw)
            if parsed and parsed not in self._recent_lines:
                self._remember(parsed)
                return parsed
        return self.canned(summary)

    def canned(self, summary: StateSummary) -> str:
        # What she noticed today beats the hour of the day, half the time:
        # it is the most "she knows me" thing a canned line can say.
        grounded = [ln for ln in summary.lines if ln not in self._recent_lines]
        if grounded and self._rng.random() < 0.5:
            choice = self._rng.choice(grounded)
            self._remember(choice)
            return choice
        pools: list[str] = []
        for need in ("hambre", "sed", "aburrimiento"):
            key = next(k for k, v in _NEED_NAMES.items() if v == need)
            if _level(summary.pet.get(key)) == "alta":
                pools.append(need)
        if any("lluv" in r.lower() or "rain" in r.lower() for r in summary.recent):
            pools.append("lluvia")
        pools += [part_of_day(summary.hour), "general"]
        candidates = [ln for pool in pools for ln in CANNED[pool] if ln not in self._recent_lines]
        # Needs first: being hungry is a better reason to talk than the hour.
        if pools[0] in ("hambre", "sed", "aburrimiento", "lluvia"):
            urgent = [ln for ln in CANNED[pools[0]] if ln not in self._recent_lines]
            candidates = urgent or candidates
        choice = self._rng.choice(candidates or list(CANNED["general"]))
        self._remember(choice)
        return choice

    def _remember(self, line: str) -> None:
        self._recent_lines.append(line)
        del self._recent_lines[:-6]


# --------------------------------------------------------------------------- #
# Triage
# --------------------------------------------------------------------------- #

NOW, LATER, IGNORE = "now", "later", "ignore"

_TRIAGE_WORDS = {
    "now": NOW, "ahora": NOW, "ya": NOW,
    "later": LATER, "después": LATER, "despues": LATER, "luego": LATER, "más": LATER,
    "ignore": IGNORE, "ignorar": IGNORE, "ignoralo": IGNORE, "nada": IGNORE,
}

TRIAGE_SYSTEM = (
    "Clasificá un aviso para alguien que está trabajando en la computadora. "
    "now = afecta lo que hace ahora mismo; later = puede esperar a un resumen "
    "(noticias, novedades); ignore = repetido o sin importancia. Ejemplos:\n"
    "rss.item: Nueva versión de una librería -> later\n"
    "weather.rain: Llueve en 10 minutos -> now\n"
    "clock.hour: Son las 15 -> ignore\n"
    "Respondé SOLO una palabra: now, later o ignore."
)


def parse_triage(raw: str | None) -> str | None:
    text = _strip_thinking(raw or "").strip().lower()
    match = re.match(r"[a-záéíóúñ]+", text)
    if not match:
        return None
    return _TRIAGE_WORDS.get(match.group(0))


class Triage:
    """now / later / ignore for info and notice events.

    Deterministic-safe: urgent is always "now" without asking anything;
    warnings are left to the rules (None); a missing, slow or confused model
    answers None, which the reflex layer reads as "use your own rule".
    """

    def __init__(self, model: LocalModel | None, *, timeout_s: float = 2.0) -> None:
        self._model = model
        self._timeout_s = timeout_s

    def decide(self, event: Event, timeout: float | None = None) -> str | None:
        if event.severity == "urgent":
            return NOW
        if event.severity not in ("info", "notice") or self._model is None:
            return None
        hour = time.localtime(event.at).tm_hour
        user = (
            f"Aviso: {event.kind} (severidad {event.severity}) — "
            f"{(event.title or '')[:160]}. Hora: {hour}hs."
        )
        raw = self._model.generate(
            TRIAGE_SYSTEM, user, max_tokens=4, temperature=0.0,
            timeout=timeout if timeout is not None else self._timeout_s,
        )
        verdict = parse_triage(raw)
        if verdict == IGNORE and event.severity == "notice":
            # A 1-2B model is not trusted to DROP anything above info: measured,
            # it calls a low battery and an RSS item the same. Worst case here
            # is a notice that waits for the batch.
            return LATER
        return verdict
