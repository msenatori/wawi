#!/usr/bin/env bash
# End-to-end "wawi" wake-word training: data -> features -> train -> ONNX -> eval.
#
# Run from a work directory that has these scripts in ./scripts (or set S),
# Kokoro weights in $KOKORO_DIR, and a Python with the packages listed in
# README.md. Every step skips work whose output already exists, so a rerun
# after adding your own recordings (data/pos_owner, see record_wawi.py) only
# redoes features + training + eval.
#
#   KOKORO_DIR=~/.local/share/kokoro WORKERS=8 bash scripts/run_all.sh
set -euo pipefail

S=${S:-scripts}
PY=${PY:-python}
WORKERS=${WORKERS:-8}
THREADS=${THREADS:-2}
NCPU=${NCPU:-8}
STEPS=${STEPS:-40000}
N_POS=${N_POS:-5000}
OUT=${OUT:-out/wawi.onnx}

step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step "background corpora (CC BY 4.0) + babble for augmentation"
$PY $S/fetch_background.py --dest data/bg \
    --sets librispeech-dev-clean,fleurs-es419-dev,librispeech-test-clean,fleurs-es419-test,fleurs-es419-train,librispeech-dev-other
[ -f data/bg/babble.npy ] || $PY $S/fetch_background.py --dest data/bg \
    --sets librispeech-dev-clean,fleurs-es419-dev --make-noise data/bg/babble.npy --noise-hours 1.5

step "synthetic positives"
[ -d data/pos_train ] || $PY $S/gen_positives.py --out data/pos_train --n "$N_POS" --seed 1 \
    --noise data/bg/babble.npy --workers "$WORKERS" --threads "$THREADS"
[ -d data/pos_test ] || $PY $S/gen_positives.py --out data/pos_test --n 500 --seed 99 --eval \
    --noise data/bg/babble.npy --workers "$WORKERS" --threads "$THREADS"

step "synthetic negatives"
[ -d data/neg_train ] || $PY $S/gen_negatives.py --out data/neg_train --hard 3000 --speech 3000 \
    --noise 800 --seed 2 --noise-bg data/bg/babble.npy --workers "$WORKERS" --threads "$THREADS"
[ -d data/neg_test ] || $PY $S/gen_negatives.py --out data/neg_test --hard 600 --speech 300 \
    --noise 100 --seed 98 --noise-bg data/bg/babble.npy --workers "$WORKERS" --threads "$THREADS"

step "features"
mkdir -p feats
f() { local out=$1; shift; [ -f "$out" ] || $PY $S/features.py --out "$out" --ncpu "$NCPU" "$@"; }
f feats/pos_train.npy --mode aligned --in data/pos_train --jitters 3
if [ -d data/pos_owner ]; then
    # The owner's own voice: more jitters, it is the most valuable data here.
    f feats/pos_owner.npy --mode aligned --in data/pos_owner --jitters 12
fi
f feats/neg_hard.npy --mode aligned --in data/neg_train/hard --jitters 3
f feats/neg_hard_stream.npy --mode stream --in data/neg_train/hard --stride 1
f feats/neg_speech.npy --mode stream --in data/neg_train/speech --stride 2
f feats/neg_noise.npy --mode stream --in data/neg_train/noise --stride 4
f feats/bg_libri.npy --mode stream --in data/bg/librispeech-dev-clean --stride 4
f feats/bg_fleurs.npy --mode stream --in data/bg/fleurs-es419-dev --stride 3
f feats/bg_fleurs_train.npy --mode stream --in data/bg/fleurs-es419-train --stride 4
f feats/bg_libri_other.npy --mode stream --in data/bg/librispeech-dev-other --stride 4

step "train"
POS="feats/pos_train.npy"
[ -f feats/pos_owner.npy ] && POS="$POS feats/pos_owner.npy"
$PY $S/train.py --pos $POS --hard feats/neg_hard.npy feats/neg_hard_stream.npy \
    --neg feats/neg_speech.npy feats/neg_noise.npy feats/bg_libri.npy feats/bg_fleurs.npy \
          feats/bg_fleurs_train.npy feats/bg_libri_other.npy \
    --out "$OUT" --steps "$STEPS"

step "eval (held out)"
POS_TEST="data/pos_test"
[ -d data/pos_owner_test ] && POS_TEST="$POS_TEST data/pos_owner_test"
$PY $S/eval.py --model "$OUT" --pos $POS_TEST --confusable data/neg_test/hard \
    --neg data/bg/fleurs-es419-test data/bg/librispeech-test-clean data/neg_test/speech data/neg_test/noise \
    --neg-hours "${NEG_HOURS:-10}" --workers "$NCPU" --json "${OUT%.onnx}.eval.json"
step "done"
