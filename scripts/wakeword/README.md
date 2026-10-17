# Wake word "Wawi": training the openWakeWord model

Wawi has two ways to hear its name (see the main README, "Palabra de activación"):

- **whisper** (default when `wake.require_phrase: true`): transcribe the first
  seconds of speech and match the text. No training. `measure_phrase.py`
  measures it.
- **openwakeword** with `model: wawi`: a small acoustic model that scores every
  80 ms. Cheaper and it does not need Whisper, but it has to be trained. This
  folder trains it.

The model is a small head (MLP, ~200k parameters) on openWakeWord's **frozen**
front-end (`melspectrogram.onnx` -> `embedding_model.onnx`, Google's speech
embedding). That is exactly what `openwakeword.Model` feeds a custom model:
16 embedding frames (1.28 s) -> one sigmoid score. Only the head is trained,
so it runs on a CPU in minutes. No GPU needed.

## Files

| script | what it does |
|---|---|
| `common.py` | Kokoro synthesis, augmentation (speed, pitch, noise at SNR, reverb), WAV IO |
| `synth_pool.py` | parallel Kokoro rendering (one pinned ONNX session per worker) |
| `gen_positives.py` | synthetic "Wawi", "che Wawi", "¿Wawi?"… across 21 voices |
| `gen_negatives.py` | confusables ("guau", "Huawei", "wow", "uy", "wifi"…), ordinary talk, non-speech noise |
| `fetch_background.py` | real speech for negatives (LibriSpeech, FLEURS es-419), all CC BY 4.0 |
| `record_wawi.py` | record YOUR voice saying "Wawi" (the most valuable data) |
| `features.py` | audio -> (N, 16, 96) feature windows |
| `train.py` | train the head, export `wawi.onnx` |
| `eval.py` | recall, false accepts/hour, confusable rate, through the real `openwakeword.Model` |
| `run_all.sh` | all of the above, skipping finished steps |
| `measure_phrase.py` | measures the whisper phrase mode instead |

## Requirements

Python 3.12 with:

```
numpy scipy soundfile onnx onnxruntime kokoro-onnx tqdm scikit-learn requests torch
openwakeword==0.6.0   # install with --no-deps on Linux (it pins tflite-runtime, unused here)
```

CPU-only torch is enough (`--index-url https://download.pytorch.org/whl/cpu`,
~200 MB instead of ~2 GB with CUDA). openWakeWord's front-end files
(`melspectrogram.onnx`, `embedding_model.onnx`) must be in
`openwakeword/resources/models/` (copy them from the Mac's venv, or
`python -c 'import openwakeword.utils as u; u.download_models([])'`), or set
`OWW_MODELS` to a folder holding them. Kokoro weights (`kokoro-v1.0.onnx`,
`voices-v1.0.bin`, the same files Wawi's voice uses) in `KOKORO_DIR`
(default `~/.local/share/kokoro`).

## Run it (Linux box or the Mac)

```
mkdir -p ~/work/wawi-wakeword && cd ~/work/wawi-wakeword
rsync -a <repo>/scripts/wakeword/ scripts/          # from the Mac
uv venv -p 3.12 .venv && . .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install numpy scipy soundfile onnx onnxruntime kokoro-onnx tqdm scikit-learn requests
uv pip install --no-deps openwakeword==0.6.0
KOKORO_DIR=$PWD/assets/kokoro PY=.venv/bin/python WORKERS=8 THREADS=2 NCPU=8 \
    nohup bash scripts/run_all.sh > run.log 2>&1 &
```

Then copy `out/wawi.onnx` to `wawi/assets/wakeword/wawi.onnx` in the repo.

### Cost (measured on a LAN server: 16-core x86 CPU, 15 GB RAM, no GPU)

| step | time | disk |
|---|---|---|
| downloads: 6 CC BY sets, 3.3 GB compressed | ~8 min (openslr ~1.5 MB/s, Hugging Face faster) | ~7 GB with extracted copies |
| Kokoro: 5 000 + 500 positives, 6 800 + 1 000 negatives (8 workers x 2 threads, nice 10) | 67 min (shared, busy box; ~45 min idle) | ~1.2 GB of WAVs |
| features (~23 h of real speech + all clips) | ~10 min total | 1.3 GB |
| training, 40 000 steps, 4 CPU threads | 4.5 min | 0.9 MB model |
| eval: 1 100 clips + 3.9 h negatives streamed through `openwakeword.Model` | ~5 min (8 workers) | - |

Peak RAM ~4.5 GB (8 Kokoro workers at ~550 MB each). End to end: 1 h 45 min
wall clock including a first training round, no GPU. On the 12-core M4 Mac
expect about the same (Kokoro is the bottleneck). A rerun with your own
recordings only redoes features + training + eval: ~15 min.

## Results of the shipped model (held out)

500 synthetic positives (21 voices, noise/reverb, incl. "Wawi, ¿qué hora
es?"), 600 synthetic confusables, 3.9 h of negatives (FLEURS es-419 test
1.28 h, LibriSpeech test-clean 2.20 h, Kokoro talk 0.33 h, noise 0.08 h).
"patience" = consecutive 80 ms frames that must be over the threshold.

| patience | threshold | recall | false accepts / h | confusables that wake |
|---|---|---|---|---|
| 1 | 0.5 | 97.8 % | 4.1 | 8.8 % |
| 1 | 0.9 | 97.4 % | 2.8 | 7.2 % |
| 2 | 0.8 | 95.8 % | 1.3 | 3.3 % |
| 3 | 0.5 | 94.0 % | 1.0 | 2.0 % |
| 3 | 0.9 | 93.0 % | 1.0 | 1.3 % |

With 3.9 h of negatives each false accept is 0.26/h, so treat the FA column
as +-0.5. The confusable that really wakes it is "Huawei" (mean peak 0.81,
it IS "wawi" plus a glide); then "wawa", "Rawi"; "guau", "uy", "bueno",
"agua", "wifi", "y vos" stay near 0. Cost at runtime: 1.2 ms per 80 ms chunk
on the M4 (plus openWakeWord's front-end), about 1.5 % of one core.

Caveat: every positive is synthetic. Real recall on your voice is unknown
until you record `pos_owner_test` and rerun eval.

## Retrain with your own voice (recommended)

On the Mac, with the microphone you use with Wawi:

```
.venv/bin/python scripts/wakeword/record_wawi.py --out data/pos_owner --n 80
.venv/bin/python scripts/wakeword/record_wawi.py --out data/pos_owner_test --n 20
# optional: the near-misses you actually say
.venv/bin/python scripts/wakeword/record_wawi.py --out data/neg_owner --n 40 --prompts negatives
rsync -a data/pos_owner data/pos_owner_test trainer:work/wawi-wakeword/data/
ssh trainer 'cd ~/work/wawi-wakeword && rm -f out/wawi.onnx && KOKORO_DIR=$PWD/assets/kokoro PY=.venv/bin/python nohup bash scripts/run_all.sh > run2.log 2>&1 &'
```

`trainer` is the machine that ran the first training: if its data, features
and venv are still in `~/work/wawi-wakeword` (~9.5 GB), this is all it takes there.
`run_all.sh` finds `data/pos_owner` and gives it 12 alignments per take (the
synthetic clips get 3), and evaluates on `data/pos_owner_test` too. Only
features, training and eval rerun (~10 min). For the owner's negatives, add
`feats/neg_owner.npy` (features.py `--mode aligned --jitters 6`) to `--hard`.

## Enable it

In `config.yaml`:

```yaml
wake:
  require_phrase: true
  engine: openwakeword
  model: wawi             # -> wawi/assets/wakeword/wawi.onnx
  wake_threshold: 0.9     # pick from the eval table: the FA/h you can live with
```

Recommended: 0.9 on a single frame (97 % recall, ~3 false wakes/h of
continuous talk). With `wake.patience: 2` (2 consecutive frames), 0.8 halves
the false wakes (~1.3/h) for ~2 points of recall. As a second-stage check
behind the whisper phrase mode (only scoring audio Whisper already thinks
said "Wawi"), a low bar like 0.5 is enough: it mainly vetoes "guau", "uy",
"wifi", "Wally" that the prompted Whisper turns into "Wawi".

If the file is missing Wawi falls back to the whisper phrase mode.

## Data and licenses

| source | license | used for |
|---|---|---|
| Kokoro-82M voices (synthetic speech) | Apache 2.0 | positives, confusables, talk |
| LibriSpeech dev-clean + dev-other / test-clean (openslr 12) | CC BY 4.0 | train / held-out negatives |
| FLEURS es_419 dev + train / test (google/fleurs) | CC BY 4.0 | train / held-out negatives |
| openWakeWord front-end models | Apache 2.0 (code); embedding model from Google (Apache 2.0) | frozen features |
| openWakeWord ACAV100M / validation features | CC BY-NC-SA 4.0 | NOT used by default (opt-in, non-commercial) |
| MUSAN (openslr 17) | CC BY 4.0, ~11 GB | not used by default (opt-in) |

The shipped `wawi.onnx` is trained only on the CC BY / Apache sources above;
see `wawi/assets/wakeword/README.md` for attribution.
