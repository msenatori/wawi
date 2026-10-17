"""Is this headline something the owner cares about?

Three signals, each optional, combined into one relevance in [0, 1]:

* **Keywords** (always there, free): the owner's ``learn.interests`` against
  the title, accent-folded and crudely stemmed, so "programación" also
  catches "programadores". A hit is strong; a miss is weak (synonyms exist).
* **Learned words** (pure numpy): an online logistic regression over hashed
  word stems, trained from what the owner did with past headlines (engaged
  = 1, dismissed or ignored = 0). Trusted only after a handful of labels.
* **Static embeddings** (optional): model2vec's multilingual potion model,
  when the ``model2vec`` package is installed. Cosine similarity to the
  interests and to a centroid of headlines the owner engaged with. Loaded in
  the background; until it is ready (or forever, without the package) the
  other two decide alone.

Measured on the development Mac (M-series): potion-multilingual-128M loads in
~6 s from cache (int8: 128 MB resident; the first run downloads ~530 MB) and
embeds a headline in ~0.03 ms. The English potion-base models are smaller
but useless for Spanish, so they are not offered.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
import zlib
from typing import Any, Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)

HASH_DIM = 2048
STEM = 6
#: Labels before the learned words get a say.
MIN_LABELS = 10
#: An embedding similarity mapped to [0, 1]: below LO is unrelated, above
#: LO + SPAN is clearly on topic (measured on Spanish headlines).
SIM_LO = 0.1
SIM_SPAN = 0.4
KEYWORD_HIT = 0.8
KEYWORD_MISS = 0.35

_STOP = frozenset(
    "the and for with from that this what una uno unos unas los las del con por para que "
    "como mas pero sus son fue han hay sin sobre entre desde hasta este esta estos estas "
    "nuevo nueva nuevos nuevas sale salio".split()
)


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def stems(text: str) -> list[str]:
    """Content-word stems: accent-folded, lowercased, cut to STEM letters.
    Two-letter words survive only in capitals in the original (IA, AI, UX)."""
    out = []
    for raw in re.findall(r"[\w+#]+", text or ""):
        word = fold(raw)
        if len(word) < 3 and not (len(raw) >= 2 and raw.isupper()):
            continue
        if word in _STOP or word.isdigit():
            continue
        out.append(word[:STEM])
    return out


def _bucket(stem: str) -> int:
    return zlib.crc32(stem.encode("utf-8")) % HASH_DIM


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class Relevance:
    def __init__(
        self,
        interests: Iterable[str] = (),
        *,
        embedder: Any = None,
        lr: float = 0.3,
    ) -> None:
        self.interests = [i for i in interests if i and i.strip()]
        self._interest_stems = [set(stems(i)) or {fold(i)[:STEM]} for i in self.interests]
        self.embedder = embedder
        self._interest_vecs: np.ndarray | None = None
        self.centroid: np.ndarray | None = None
        self.w = np.zeros(HASH_DIM, dtype=np.float32)
        self.b = 0.0
        self.labels = 0
        self.lr = lr
        self._lock = threading.Lock()

    # -- the three signals ----------------------------------------------------
    def keyword(self, title: str) -> float | None:
        if not self._interest_stems:
            return None
        have = set(stems(title))
        for want in self._interest_stems:
            if want and want <= have:
                return 1.0
        return 0.0

    def _features(self, title: str) -> np.ndarray:
        idx = sorted({_bucket(s) for s in stems(title)})
        return np.asarray(idx, dtype=np.int64)

    def learned(self, title: str) -> float | None:
        if self.labels < MIN_LABELS:
            return None
        idx = self._features(title)
        z = float(self.w[idx].sum()) + self.b if idx.size else self.b
        return float(1.0 / (1.0 + np.exp(-z)))

    def embedded(self, title: str) -> float | None:
        emb = self.embedder
        if emb is None or not getattr(emb, "ready", True):
            return None
        try:
            vec = _unit(np.asarray(emb.encode([title])[0], dtype=np.float32))
            sims = []
            if self.interests:
                if self._interest_vecs is None:
                    m = np.asarray(emb.encode(self.interests), dtype=np.float32)
                    self._interest_vecs = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
                sims.append(float((self._interest_vecs @ vec).max()))
            if self.centroid is not None and self.centroid.shape == vec.shape:
                sims.append(float(_unit(self.centroid) @ vec))
            if not sims:
                return None
            return float(np.clip((max(sims) - SIM_LO) / SIM_SPAN, 0.0, 1.0))
        except Exception:
            log.debug("relevance: embedding failed", exc_info=True)
            return None

    # -- the verdict ------------------------------------------------------------
    def score(self, title: str) -> tuple[float | None, bool]:
        """(relevance in [0, 1] or None when nothing is known, confident).

        *confident* means a learned or embedded signal took part; a keyword
        miss alone is never enough to drop a headline.
        """
        with self._lock:
            kw = self.keyword(title)
            lw = self.learned(title)
            em = self.embedded(title)
        parts: list[tuple[float, float]] = []
        if em is not None:
            parts.append((0.5, em))
        if lw is not None:
            parts.append((0.3, lw))
        if kw is not None:
            parts.append((0.3, KEYWORD_HIT if kw else KEYWORD_MISS))
        if not parts:
            return None, False
        total = sum(w for w, _ in parts)
        value = sum(w * v for w, v in parts) / total
        if kw:
            value = max(value, KEYWORD_HIT)  # the owner said so in words
        return float(value), (em is not None or lw is not None)

    # -- learning -------------------------------------------------------------------
    def learn(self, title: str, positive: bool) -> None:
        """One label from the owner's behaviour. A few SGD steps, microseconds."""
        if not title:
            return
        y = 1.0 if positive else 0.0
        with self._lock:
            idx = self._features(title)
            for _ in range(3):
                z = (float(self.w[idx].sum()) if idx.size else 0.0) + self.b
                p = 1.0 / (1.0 + np.exp(-z))
                g = p - y
                if idx.size:
                    self.w[idx] -= self.lr * g
                self.b -= self.lr * 0.2 * g
            np.clip(self.w, -4.0, 4.0, out=self.w)
            self.b = float(np.clip(self.b, -3.0, 3.0))
            self.labels += 1
            emb = self.embedder
            if positive and emb is not None and getattr(emb, "ready", True):
                try:
                    vec = _unit(np.asarray(emb.encode([title])[0], dtype=np.float32))
                    if self.centroid is None or self.centroid.shape != vec.shape:
                        self.centroid = vec
                    else:
                        self.centroid = 0.8 * self.centroid + 0.2 * vec
                except Exception:
                    log.debug("relevance: centroid update failed", exc_info=True)

    # -- persistence ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        nz = np.nonzero(np.abs(self.w) > 1e-4)[0]
        return {
            "w": {str(int(i)): round(float(self.w[i]), 4) for i in nz},
            "b": round(self.b, 4),
            "labels": self.labels,
            "centroid": None if self.centroid is None else [round(float(x), 5) for x in self.centroid],
        }

    def load(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        try:
            for k, v in (raw.get("w") or {}).items():
                i = int(k)
                if 0 <= i < HASH_DIM:
                    self.w[i] = float(v)
            self.b = float(raw.get("b", 0.0))
            self.labels = int(raw.get("labels", 0))
            c = raw.get("centroid")
            self.centroid = np.asarray(c, dtype=np.float32) if c else None
        except (TypeError, ValueError, AttributeError):
            log.warning("learn: relevance state unreadable; starting fresh")
            self.w[:] = 0.0
            self.b, self.labels, self.centroid = 0.0, 0, None


class BackgroundEmbedder:
    """model2vec loaded on a background thread; ``ready`` once it is.

    *loader* is injectable for tests; the default imports model2vec, which
    is an optional extra. A failed import or load leaves ``ready`` False
    forever, and relevance quietly runs on keywords and learned words.
    """

    def __init__(self, model_id: str, loader: Callable[[str], Any] | None = None) -> None:
        self.model_id = model_id
        self.ready = False
        self.failed = False
        self._model: Any = None
        self._loader = loader or _load_model2vec
        self._thread = threading.Thread(target=self._load, name="wawi-learn-embed", daemon=True)

    def start(self) -> "BackgroundEmbedder":
        self._thread.start()
        return self

    def _load(self) -> None:
        try:
            self._model = self._loader(self.model_id)
            self.ready = True
            log.info("learn: relevance embeddings ready (%s)", self.model_id)
        except Exception:
            self.failed = True
            log.info("learn: no embeddings (%s); keywords only", self.model_id, exc_info=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts)), dtype=np.float32)


def model2vec_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("model2vec") is not None
    except (ImportError, ValueError):
        return False


def _load_model2vec(model_id: str) -> Any:
    from model2vec import StaticModel  # type: ignore[import-not-found]

    try:
        # int8 keeps the 128M-token table at 128 MB instead of 512 MB, with
        # the same similarities to two decimals (measured).
        return StaticModel.from_pretrained(model_id, quantize_to="int8")
    except TypeError:
        return StaticModel.from_pretrained(model_id)
