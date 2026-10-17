#!/usr/bin/env python3
"""Download real speech/background audio for the wake-word negatives.

Synthetic negatives (gen_negatives.py) teach the confusable words; real
recordings teach "a room where people talk". Everything fetched by default is
freely licensed (CC BY 4.0) and totals ~1.5 GB compressed:

  name                    license      size     use
  librispeech-dev-clean   CC BY 4.0    338 MB   train negatives (English read speech, 5.4 h)
  fleurs-es419-dev        CC BY 4.0    255 MB   train negatives (Latin-American Spanish)
  librispeech-test-clean  CC BY 4.0    347 MB   held-out false accepts/hour (5.4 h)
  fleurs-es419-test       CC BY 4.0    582 MB   held-out false accepts/hour (Spanish)

Also used for the shipped model (run_all.sh fetches them; --sets to pick):

  fleurs-es419-train      CC BY 4.0    1.44 GB  train negatives (more Spanish talk)
  librispeech-dev-other   CC BY 4.0    314 MB   train negatives (harder English)

Opt-in (--sets NAME), NOT downloaded by default:

  oww-validation          CC BY-NC-SA 4.0  ~185 MB*  openWakeWord's ~11 h validation
                          features (davidscripka/openwakeword_features). Non-
                          commercial: fine to MEASURE with, do not train a model
                          you might redistribute on it.
  oww-acav100m            CC BY-NC-SA 4.0  ~17 GB*   2000 h of precomputed negative
                          features: what the official openWakeWord recipe trains on.
                          Same license caveat; big.
  musan                   CC BY 4.0    ~11 GB    noise/music/speech corpus (openslr 17).

(*) sizes read from the Hugging Face API in October 2026; verify before relying
on them. Licenses are the ones the dataset pages state: LibriSpeech (openslr 12)
CC BY 4.0; FLEURS (google/fleurs) CC BY 4.0; MUSAN (openslr 17) CC BY 4.0.

After downloading, --make-noise writes a float32 16 kHz .npy of the TRAIN
sets concatenated (a few hours), which gen_positives/gen_negatives mix in as
real background babble.

    python scripts/wakeword/fetch_background.py --dest data/bg
    python scripts/wakeword/fetch_background.py --dest data/bg --make-noise data/bg/babble.npy --noise-hours 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

HF = "https://huggingface.co/datasets"
SETS = {
    "librispeech-dev-clean": ("https://www.openslr.org/resources/12/dev-clean.tar.gz", "train"),
    "fleurs-es419-dev": (f"{HF}/google/fleurs/resolve/main/data/es_419/audio/dev.tar.gz", "train"),
    "librispeech-test-clean": ("https://www.openslr.org/resources/12/test-clean.tar.gz", "test"),
    "fleurs-es419-test": (f"{HF}/google/fleurs/resolve/main/data/es_419/audio/test.tar.gz", "test"),
    # More real talk for the negatives (round 2 of the shipped model).
    "fleurs-es419-train": (f"{HF}/google/fleurs/resolve/main/data/es_419/audio/train.tar.gz", "train"),
    "librispeech-dev-other": ("https://www.openslr.org/resources/12/dev-other.tar.gz", "train"),
    "oww-validation": (f"{HF}/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy", "test"),
    "oww-acav100m": (f"{HF}/davidscripka/openwakeword_features/resolve/main/"
                     "openwakeword_features_ACAV100M_2000_hrs_16bit.npy", "train"),
    "musan": ("https://www.openslr.org/resources/17/musan.tar.gz", "train"),
}
DEFAULT = ("librispeech-dev-clean", "fleurs-es419-dev", "librispeech-test-clean",
           "fleurs-es419-test")


def fetch(name: str, dest: Path) -> Path:
    url, _ = SETS[name]
    dest.mkdir(parents=True, exist_ok=True)
    fname = dest / (name + ("".join(Path(url).suffixes[-2:]) if url.endswith(".tar.gz") else ".npy"))
    if not fname.exists():
        print(f"downloading {name} <- {url}", flush=True)
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(fname) + ".part", url], check=True)
        Path(str(fname) + ".part").rename(fname)
    if fname.suffixes[-2:] == [".tar", ".gz"]:
        out = dest / name
        if not out.exists():
            print(f"extracting {fname.name}", flush=True)
            with tarfile.open(fname) as tar:
                tar.extractall(out, filter="data")
        return out
    return fname


def make_noise(dirs: list[Path], out: Path, hours: float, seed: int) -> None:
    """Concatenate random files from *dirs* into one float32 array."""
    rng = np.random.default_rng(seed)
    files = [f for d in dirs for f in C.wavs(d)]
    rng.shuffle(files)
    target = int(hours * 3600 * C.SR)
    parts, total = [], 0
    for f in files:
        try:
            x = C.read_wav(f)
        except Exception:
            continue
        parts.append(x)
        total += x.size
        if total >= target:
            break
    arr = np.concatenate(parts)[:target].astype(np.float32)
    np.save(out, arr)
    print(f"{arr.size / C.SR / 3600:.2f} h of background -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--sets", default=",".join(DEFAULT))
    ap.add_argument("--make-noise", type=Path, default=None)
    ap.add_argument("--noise-hours", type=float, default=2.0)
    args = ap.parse_args()
    got = {n: fetch(n, args.dest) for n in [s.strip() for s in args.sets.split(",") if s.strip()]}
    if args.make_noise:
        train_dirs = [p for n, p in got.items() if SETS[n][1] == "train" and p.is_dir()]
        make_noise(train_dirs, args.make_noise, args.noise_hours, seed=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
