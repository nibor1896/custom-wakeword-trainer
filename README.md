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

1. **(Optional) Download negative features** (~16 GB, precomputed, ~2000 h of speech/music/noise):
   `python scripts/download_negatives.py`
   > ⚠️ **License warning:** this feature set (ACAV100M via `openwakeword_features`) is
   > **CC-BY-NC-SA 4.0 — NonCommercial**. A model trained with it inherits that restriction and
   > **cannot be used commercially**. If you need a commercially clean model, skip this download
   > (or set `WW_NO_ACAV=1`): the trainer then uses your recorded session negatives + generated
   > silence alone. Measured on a real wake word ("Anima", ~1700 own negatives): recall and false
   > alarms came out equal to the ACAV-trained model.
2. **Generate synthetic positives** (many speakers, via [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)):
   ```bash
   # webrtcvad does not build on 3.13 -> use webrtcvad-wheels (in requirements.txt)
   set PYTHONIOENCODING=utf-8
   python -m piper_sample_generator "hey horus" --model models/en_US-libritts_r-medium.pt \
          --max-samples 20000 --batch-size 128 --output-dir data/positives
   ```
3. **Your own recordings — the key: CLEAN, SEPARATE sessions** (no auto-labeling → no label noise).
   Use `scripts/record_sessions.py`:

   ```bash
   python scripts/record_sessions.py pos 50 8          # 50x the wake word, a cue every 8 s
   python scripts/record_sessions.py neg 240 keyboard  # then one category at a time:
   python scripts/record_sessions.py neg 180 tv
   python scripts/record_sessions.py neg 180 music
   python scripts/record_sessions.py neg 120 speech
   python scripts/record_sessions.py neg  60 silence
   ```

   - **Positives are recorded on an automatic cadence — you never touch the keyboard.**
     A keystroke right before each clip would end up *inside* the positives, and you are about
     to teach the model that keyboards are **not** the wake word. Don't poison that lesson.
   - **Negatives one category at a time.** The filenames keep the category, so when the model
     later false-triggers you can see *which kind of sound* did it and record more of exactly that.
   - Record the real noises **through the real mic in the real room**. Clean sample files help
     far less than the actual path the model will live in.
   - Cover: **keyboard** (the #1 false trigger — every key, fast and slow), **TV/YouTube**
     (real speech that isn't yours — no VAD will save you here), **music** bleeding out of your
     headphones, **your own speech + near-homophones** of the phrase, and plain **silence**.
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

### ⚠️ Never zero-pad the positives (fixed 2026-07-14)

The nastiest bug this recipe had. Your wake word is short (~0.8 s), the training window is 2 s — so the clip gets padded. If you pad with **digital silence**, while the bulk of your negatives (ACAV features) is **full-length audio**, then *"lots of leading silence"* becomes a near-perfect cue for the positive class. The model learns it happily — and then **fires at score 1.00 into a completely silent room**.

It stays invisible if your runtime has a VAD in front (the model is simply never asked while it's quiet). It bites the moment anyone reuses the model in a pipeline without one — which is exactly how it was found.

**What this trainer does now:**
- the clip is placed at a **random offset** in the window, and the rest is filled with a **faint noise floor** — so padding carries no class information,
- **explicit silence negatives** (digital zero up to quiet room tone) are added to the training set: silence must be learned as *not the wake word*,
- validation reports **silence false-alarms separately** and refuses to call a model good if they are above 0 %.

Measured on the old model: silence → score **1.00**; normal speech → **no false alarms at all** (peak 0.11). The model was excellent at speech and simply believed silence was the phrase.

**The fix is proven.** A second wake word ("Anima") trained with the current recipe — 50 positives, ~1700 real negatives (180 keyboard clips, TV, music, near-homophones, silence) plus the ACAV features:

```
threshold 0.5: recall 90%  false alarms 0.0%  SILENCE false alarms 0.0%
threshold 0.7: recall 90%  false alarms 0.0%  SILENCE false alarms 0.0%
threshold 0.9: recall 90%  false alarms 0.0%  SILENCE false alarms 0.0%
```

Still 90 % recall at threshold **0.9**, with zero false alarms on a keyboard being hammered, on TV speech, on music, and on silence. That headroom is what lets you run a conservative threshold in production and lose nothing.

## Licenses

- openWakeWord code + melspectrogram/embedding models: Apache-2.0
- LibriTTS-R (piper positive voices): CC-BY-4.0 → attribution required
- **ACAV100M feature set (`openwakeword_features`): CC-BY-NC-SA-4.0 — NonCommercial.** An
  earlier version of this README claimed "only features are used, no audio ends up in the
  model" — that was wrong as a license argument: the *feature files themselves* carry the NC
  license, and the openWakeWord author licenses models trained on them as NC too. **A model
  trained with this set must not be used commercially.** Train with `WW_NO_ACAV=1` (own
  negatives + silence only) for a commercially clean model.
- Any real recordings you add are baked into the weights (not recoverable as audio)

## Status

See [Issue #1](https://github.com/nibor1896/custom-wakeword-trainer/issues/1). "Hey Horus" is trained, integrated into Horus, and confirmed live (90 % recall, 0 everyday false alarms **on speech**).

> ⚠️ **The `hey_horus.onnx` in this repo predates the zero-padding fix (2026-07-14)** and therefore **fires on silence** (score 1.00 in a quiet room — see the learning above). Inside Horus it is harmless, because a VAD gates it. **Do not reuse it in a pipeline without a speech gate**, and retrain with the current recipe to get a model that is clean on silence as well.
