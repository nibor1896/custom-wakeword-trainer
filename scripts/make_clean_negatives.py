r"""
Build a COMMERCIALLY-CLEAN negative-feature file — a drop-in replacement for the NC-licensed
neg_ACAV100M_2000hrs.npy, so a trained wake word can be sold.

Why this exists: the openWakeWord ACAV100M feature dump is CC-BY-NC-SA 4.0 (NonCommercial),
which quietly makes any model trained on it non-commercial too. This script produces the SAME
kind of file — an (N, 16, 96) float16 array of embedding features — from audio YOU are allowed
to sell against. It uses only the openWakeWord melspectrogram + embedding models, which are
Apache-2.0, so the features it computes carry no license of their own; only the SOURCE AUDIO's
license matters. Point it at clean sources and the result is clean.

Verified commercially-usable sources (no NC):
  * MUSAN            openslr.org/17   — music+speech+noise, curated to permit commercial use
  * LibriTTS-R       openslr.org/141  — clean read speech, CC BY 4.0 (attribution)
  * Common Voice     commonvoice.mozilla.org — multilingual speech, CC0
  * your OWN session-neg recordings — your copyright, always fine, and the most on-target
(Do NOT feed it ACAV100M or anything CC-*-NC — that is the whole point.)

Usage (point it at a folder of clean audio; recurses; wav/flac/ogg):
    python scripts/make_clean_negatives.py  D:\clean_audio  [more_dirs ...]
    python scripts/make_clean_negatives.py  D:\musan  ..\data\session-neg
Options via env:
    WW_MODELS=...        folder with melspectrogram.onnx + embedding_model.onnx (default: data/models)
    OUT=...              output .npy (default: data/neg_clean.npy)
    STEP=16             frames between windows (16 = non-overlapping 1.28 s; 8 = 2x more, overlap)
    MAX_ROWS=0          cap the number of rows (0 = no cap)

Then train against it WITHOUT touching anything else:
    WW_NEG=..\data\neg_clean.npy  python scripts/train.py
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf
from openwakeword.utils import AudioFeatures

RATE = 16000
FRAME = 16                       # 16 embedding frames = 1.28 s, exactly one negative row
STEP = int(os.environ.get("STEP", "16"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "0"))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("WW_DATA") or os.path.join(HERE, "..", "data")
MODELS_DIR = os.environ.get("WW_MODELS") or os.path.join(DATA, "models")
OUT = os.environ.get("OUT") or os.path.join(DATA, "neg_clean.npy")
CHUNK_S = 30                     # embed the audio in 30 s pieces so memory stays bounded
EXTS = (".wav", ".flac", ".ogg", ".opus", ".aiff", ".aif")

F = AudioFeatures(
    melspec_model_path=os.path.join(MODELS_DIR, "melspectrogram.onnx"),
    embedding_model_path=os.path.join(MODELS_DIR, "embedding_model.onnx"),
)


def _resample(d, sr):
    if sr == RATE:
        return d.astype(np.float32)
    try:
        import torch
        import torchaudio
        return torchaudio.functional.resample(
            torch.from_numpy(d.astype(np.float32)), sr, RATE).numpy()
    except Exception:
        n = int(round(len(d) * RATE / sr))
        return np.interp(np.linspace(0, len(d) - 1, n), np.arange(len(d)), d).astype(np.float32)


def load16k(path):
    d, sr = sf.read(path, dtype="float32", always_2d=False)
    if d.ndim > 1:
        d = d.mean(1)
    return _resample(d, sr)


def windows_from(sig):
    """Embed one clip and cut its frame sequence into (16, 96) rows — the ACAV row format."""
    rows = []
    hop = CHUNK_S * RATE
    for i in range(0, len(sig), hop):
        seg = sig[i:i + hop]
        if len(seg) < RATE // 2:                 # < 0.5 s tail — nothing useful
            continue
        clip = (np.clip(seg, -1, 1) * 32767).astype(np.int16)[None, :]
        fe = F.embed_clips(clip, batch_size=1)   # (1, n_frames, 96) — same model as train.py
        fr = fe[0]
        for s in range(0, fr.shape[0] - FRAME + 1, STEP):
            rows.append(fr[s:s + FRAME, :])
    return rows


def main():
    dirs = [d for d in sys.argv[1:] if not d.startswith("-")]
    if not dirs:
        sys.exit("give one or more folders of CLEAN audio, e.g.:\n"
                 "  python scripts/make_clean_negatives.py D:\\musan ..\\data\\session-neg")

    files = []
    for d in dirs:
        if os.path.isfile(d):
            files.append(d)
        else:
            for e in EXTS:
                files += glob.glob(os.path.join(d, "**", "*" + e), recursive=True)
    files = sorted(set(files))
    if not files:
        sys.exit(f"no audio files ({', '.join(EXTS)}) found under: {dirs}")
    print(f"{len(files)} clean audio files from {len(dirs)} folder(s) -> {OUT}", flush=True)

    all_rows = []
    kept, skipped, secs = 0, 0, 0.0
    for k, f in enumerate(files, 1):
        try:
            sig = load16k(f)
            secs += len(sig) / RATE
            all_rows.extend(windows_from(sig))
            kept += 1
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  skip {os.path.basename(f)}: {e}", flush=True)
        if k % 200 == 0 or k == len(files):
            print(f"  {k}/{len(files)} files  ->  {len(all_rows):,} rows"
                  f"  ({secs/3600:.1f} h audio)", flush=True)
        if MAX_ROWS and len(all_rows) >= MAX_ROWS:
            print(f"  reached MAX_ROWS={MAX_ROWS:,}, stopping early", flush=True)
            break

    if not all_rows:
        sys.exit("produced 0 rows — were the files silent/corrupt?")
    arr = np.asarray(all_rows[:MAX_ROWS] if MAX_ROWS else all_rows, dtype=np.float16)
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.save(OUT, arr)
    print(f"\nwrote {OUT}", flush=True)
    print(f"  shape {arr.shape}  dtype {arr.dtype}  "
          f"({arr.nbytes/1e9:.2f} GB, {arr.shape[0]*1.28/3600:.1f} h of negative windows)")
    print(f"  from {kept} files ({secs/3600:.1f} h audio), {skipped} skipped")
    print("\nNext: WW_NEG=" + os.path.abspath(OUT) + "  python scripts/train.py")
    print("Then compare the validation table to the ACAV model — only ship if it holds.")


if __name__ == "__main__":
    main()
