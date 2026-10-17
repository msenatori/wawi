# wawi.onnx: the "Wawi" wake-word model

An openWakeWord custom model: input `(1, 16, 96)` float32 (16 frames of
openWakeWord's speech embedding), output `(1, 1)` sigmoid score. Load it with

```python
openwakeword.Model(wakeword_models=["wawi/assets/wakeword/wawi.onnx"], inference_framework="onnx")
```

and read the score under the key `"wawi"`. Wawi does this when `config.yaml`
says `wake.engine: openwakeword` + `wake.model: wawi`.

**Status: experimental.** Trained only on synthetic voices (Kokoro) plus real
read speech as negatives. It has never heard the owner's voice. Retrain it
with your own recordings (`scripts/wakeword/README.md`, "Retrain with your
own voice") before relying on it alone.

Evaluation of this file: `wawi.eval.json` (held-out recall, false accepts per
hour by source and patience, confusable words) and `wawi.train.json`
(training summary). Headline, held out: threshold 0.9 single frame = 97.4 %
recall, 2.8 false accepts/h; 2 frames at 0.8 = 95.8 %, 1.3/h. "Huawei" still
wakes it. Full table in `scripts/wakeword/README.md`.

## How it was made

`scripts/wakeword/run_all.sh`: Kokoro-generated "Wawi" clips (21 voices,
speed/pitch/noise/reverb augmentation) as positives; Kokoro near-misses
("guau", "Huawei", "wow", "uy", "wifi", …), Kokoro talk, synthetic noise,
LibriSpeech dev-clean and FLEURS es_419 dev as negatives; a small MLP head
trained on the frozen openWakeWord front-end.

## License and attribution

The model weights are released under the repository's license. They were
derived from:

- **Kokoro-82M** (hexgrad), Apache License 2.0: synthetic speech.
- **LibriSpeech ASR corpus** (Panayotov et al., openslr.org/12), CC BY 4.0.
- **FLEURS** (Conneau et al., Google, huggingface.co/datasets/google/fleurs),
  CC BY 4.0.
- **openWakeWord** (David Scripka), Apache License 2.0: the feature
  front-end (`melspectrogram.onnx`, `embedding_model.onnx`, itself based on
  Google's speech_embedding model, Apache 2.0). Those front-end files are not
  in this folder; openWakeWord ships them.

No CC BY-NC data (openWakeWord's ACAV100M or validation features) was used
to train it.
