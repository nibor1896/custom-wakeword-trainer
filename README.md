# custom-wakeword-trainer

A **local trainer for your own [openWakeWord](https://github.com/dscripka/openWakeWord) wake words** — a reproducible recipe + scripts to train **any** wake word ("Hey Jarvis", "Computer", "Hey Horus", …). Runs **locally on Windows + Python 3.13**, where the official openWakeWord Colab notebook has been bit-rotted since 2023.

> **"Hey Horus"** is the first fully worked example model (built for the [Horus](https://github.com/nibor1896/Horus) project).

## What you get

A tiny `<phrase>.onnx` file (~840 KB). Runtime chain:

```
mic 16 kHz → melspectrogram.onnx → embedding_model.onnx → <phrase>.onnx → probability → threshold
```

The first two ONNX models are the generic feature extractors shared by **all** openWakeWord models (Apache-2.0). Only `<phrase>.onnx` is the trained, phrase-specific part → **standard openWakeWord format**, usable directly in Home Assistant / Rhasspy / ESPHome / your own runtimes.

## Why this exists (the decisive point)

The chain of reasoning that led exactly here:

1. **Requirement:** a wake word that runs **fully local / offline** and stays **free forever** (a hard constraint of the project this was built for).
2. **[Porcupine](https://picovoice.ai/platform/porcupine/)** — the obvious ready-made cross-platform wake-word engine — **discontinued its free tier on 2026-06-30**, which disqualified it under that constraint.
3. **First attempt, WITHOUT training:** take openWakeWord's generic embeddings and compare them (DTW / cosine **distance**) against a handful of enrolled reference recordings. This **demonstrably fails on a real voice** — there is *no* separation between the wake word and arbitrary speech (at best ~46 % false alarms at 89 % recall). The reason: these embeddings encode **speaker identity** far more strongly than phrase content, so with the same speaker (you enroll, you use it) your timbre matches the reference no matter *what* you say.
4. **Consequence:** you need a **properly trained** model — a small classifier learned over many speakers plus a lot of negative data — because only that separates the *phrase* from the *speaker*.
5. **But** openWakeWord's **official training pipeline (the Colab notebook) has been bit-rotted since 2023** and no longer runs in 2026 (Python 3.12 incompatibilities, changed package layouts, broken data downloads).

**➜ The decisive point is the combination of 3 and 5:** the simple distance-based DIY approach *doesn't work*, and the official training path *is broken*. A **working, local, reproducible trainer of your own** is therefore the only way to satisfy *free + local + reliable* at the same time — and that is exactly what this repo is.

(More detail, including the measured numbers, in [Issue #1](https://github.com/nibor1896/custom-wakeword-trainer/issues/1).)

## Requirements

- Python 3.13 (tested on Windows), NVIDIA GPU recommended (CPU works, just slower).
- ~20 GB free space (16 GB negative features + intermediate data).
- The two generic feature models from openWakeWord (v0.5.1):
  [`melspectrogram.onnx`](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx),
  [`embedding_model.onnx`](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx).

## Recipe

```bash
python -m venv venv && venv\Scripts\pip install -r requirements.txt
```

1. **Download negative features** (~16 GB, precomputed, ~2000 h of speech/music/noise):
   `python scripts/download_negatives.py`
2. **Generate synthetic positives** (many speakers, via [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)):
   ```bash
   # webrtcvad does not build on 3.13 -> use webrtcvad-wheels (in requirements.txt)
   set PYTHONIOENCODING=utf-8
   python -m piper_sample_generator "hey horus" --model models/en_US-libritts_r-medium.pt \
          --max-samples 20000 --batch-size 128 --output-dir data/positives
   ```
3. **Your own recordings — the key: TWO clean sessions** (no auto-labeling → no label noise):
   - Session A: **only** the wake word (~50×, normal voice, varied) → `data/session-pos/`
   - Session B: **only** negatives → `data/session-neg/`. Include normal talking, reading aloud,
     hard near-homophones, **and the real noises of your deployment recorded through the real path**
     — e.g. music played on your speakers and picked up by the mic, and your own keyboard (record
     *every* key). Clean noise files help far less than the actual mic/room path.
   - As 16 kHz mono WAV, ~2 s each.
4. **Train** (adjust the paths at the top of the file):
   `python scripts/train.py` → writes `<phrase>.onnx` and prints recall / false-alarm rate on held-out recordings.
5. **Iterate on false triggers.** Score *all* your negatives with the fresh model and sort by score.
   A single clip near 1.0 while its whole category sits near 0 is a **mislabeled positive** in your
   negatives — quarantine it and retrain (don't just add data). If a real sound genuinely scores high
   (single mechanical keys are the classic one), record that class densely. Details in
   [Issue #2](https://github.com/nibor1896/custom-wakeword-trainer/issues/2).

## Key learnings

Detailed in [Issue #2](https://github.com/nibor1896/custom-wakeword-trainer/issues/2). In one sentence: **data quality is everything** — purely synthetic positives give only ~8 % recall on a real voice, real user recordings + augmentation lift that to ~90 %, and Whisper auto-labeling of mixed sessions ruins it again (the two clean sessions above are the fix).

And when a trained model keeps false-triggering on something (music, keyboard), **score your own negatives first**: a lone high-scoring clip is usually a mislabeled positive (contamination), not a genuinely hard sound — fix the data, not the threshold.

## Licenses

- openWakeWord + feature models: Apache-2.0
- LibriTTS-R (piper positive voices): CC-BY-4.0 → attribution required
- ACAV100M: only precomputed features used, no audio ends up in the model
- Any real recordings you add are baked into the weights (not recoverable as audio)

## Status

See [Issue #1](https://github.com/nibor1896/custom-wakeword-trainer/issues/1). "Hey Horus" is trained, integrated into Horus, and confirmed live (90 % recall, 0 everyday false alarms).
