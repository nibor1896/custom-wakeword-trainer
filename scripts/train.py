"""Trains an openWakeWord wake-word classifier from CLEAN two-session data.

Data philosophy (the whole trick): one recording session containing ONLY the wake word, one
containing ONLY negatives, stored SEPARATELY — no auto-labeling, hence no label noise. The
positives are heavily augmented (noise / gain / pitch / tempo) so the model does not simply
learn "clean audio = positive". A slice of the real POS/NEG is held back for validation, so
the reported numbers are honest.

The model outputs a probability 0..1 (sigmoid inside the model) -> the runtime needs no sigmoid
of its own. The architecture is byte-identical to openwakeword.train (model_type="dnn").
"""
import glob
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from audiomentations import AddGaussianSNR, Compose, Gain, PitchShift, TimeStretch
from openwakeword.utils import AudioFeatures

# ======================= CONFIG =========================
# Everything can be overridden by environment variables, so the same recipe can train a
# different phrase from a different data folder without editing this file:
#   WW_PHRASE=anima  WW_DATA=D:\some\where  python scripts/train.py
HERE = os.path.dirname(__file__)
DATA = os.environ.get("WW_DATA") or os.path.join(HERE, "..", "data")
PHRASE = os.environ.get("WW_PHRASE", "hey_horus")
MODELS_DIR = os.environ.get("WW_MODELS") or os.path.join(DATA, "models")   # melspectrogram.onnx + embedding_model.onnx (openWakeWord v0.5.1)
POS_DIR = os.path.join(DATA, "session-pos")                     # clean session: ONLY the wake word, 16 kHz mono WAV
NEG_DIR = os.path.join(DATA, "session-neg")                     # clean session: ONLY negatives
NEG_FEATURES = os.environ.get("WW_NEG") or os.path.join(DATA, "neg_ACAV100M_2000hrs.npy")  # WW_NEG overrides with a commercially-clean neg_clean.npy (make_clean_negatives.py); default = NC ACAV dump
SYNTH_POS_NPY = os.path.join(DATA, "pos_synth.npy")             # optional: pre-embedded piper positives (N,16,96), or absent
OUT = os.path.join(HERE, "..", f"{PHRASE}.onnx")
TARGET = 32000            # 2 s @ 16 kHz -> exactly 16 embedding frames
N_NEG_SUBSET = 800_000
STEPS = 25000
# =================================================================

rng = np.random.default_rng(0)
F = AudioFeatures(
    melspec_model_path=os.path.join(MODELS_DIR, "melspectrogram.onnx"),
    embedding_model_path=os.path.join(MODELS_DIR, "embedding_model.onnx"),
)


def load_raw(path):
    """Load a clip as it is — no padding, no window. Augmentation happens on THIS."""
    d, sr = sf.read(path, dtype="float32", always_2d=False)
    if d.ndim > 1:
        d = d.mean(1)
    return d[:TARGET]


def place(d):
    """Drop a clip at a RANDOM offset into the TARGET window, on a faint noise floor.

    Do NOT zero-pad. The positives are short (~0.8 s) and were previously left-padded with
    DIGITAL SILENCE to fill the 2 s window, while the bulk of the negatives (the ACAV features)
    is full-length audio. That turns "lots of leading silence" into a near-perfect cue for the
    positive class — the model duly learns it and then fires at score 1.00 into a completely
    silent room (measured; see issue #2).

    Called AFTER augmentation, never before: stretching a clip that is already padded stretches
    the SILENCE too, and cropping back to length then cuts the word in half. The augmentation
    would be corrupting the very examples it is supposed to teach.
    """
    d = d[:TARGET]
    if len(d) == TARGET:
        return d
    out = (rng.standard_normal(TARGET) * 1e-3).astype(np.float32)   # faint room-tone floor
    off = int(rng.integers(0, TARGET - len(d) + 1))
    out[off:off + len(d)] += d
    return out


def load16k(path):
    """Unaugmented clip in a window — for validation and for the clean copy of each positive."""
    return place(load_raw(path))


def to_feat(sig):
    clip = (np.clip(sig, -1, 1) * 32767).astype(np.int16)[None, :]
    fe = F.embed_clips(clip, batch_size=1)
    n = fe.shape[1]
    s = max(0, (n - 16) // 2)
    w = fe[:, s:s + 16, :]
    if w.shape[1] < 16:
        w = np.pad(w, ((0, 0), (0, 16 - w.shape[1]), (0, 0)))
    return w[0]


def silence_feats(n_per_level=10):
    """Explicit SILENCE negatives — from digital zero up to quiet room tone.

    Direct insurance against the failure that was actually measured: the model fired at score
    1.00 into a silent room. Without these, the model can still drift towards "quiet = wake
    word", because the real negatives contain almost no silence. Silence must be LEARNED as
    "not the wake word", not merely gated away at runtime.
    """
    out = []
    for level in (0.0, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2):
        for _ in range(n_per_level):
            out.append(to_feat((rng.standard_normal(TARGET) * level).astype(np.float32)))
    return np.stack(out).astype(np.float32)


# How people ACTUALLY say the word, versus how they say it into a recording script.
#
# Recording sessions produce one single way of speaking: prompted, deliberate, evenly paced,
# same distance, same posture, same evening. Real life does not. The same person calling the
# same word across a room, half-distracted, mid-sentence, is easily 15-20% faster and a couple
# of semitones off — and a model trained only on the session simply does not know them anymore.
#
# This was measured, not assumed: a model with TimeStretch(0.9, 1.1) scored 1.00 on the
# recordings and 0.21 in the room. The same failing audio, slowed to 0.90, woke it instantly —
# the speaker was ~11% faster than anything the model had ever seen. He was standing just
# outside the augmentation range.
#
# So the range has to cover how a human varies, not how a recording session varies.
aug = Compose([
    AddGaussianSNR(min_snr_db=3.0, max_snr_db=30.0, p=0.9),
    Gain(min_gain_db=-12.0, max_gain_db=6.0, p=0.9),
    PitchShift(min_semitones=-3.0, max_semitones=3.0, p=0.7),
    # 0.75-1.35 and applied almost always. Speaking rate is the axis people vary MOST and the
    # one a recording session flattens completely. leave_length_unchanged=False: the word is
    # allowed to get longer or shorter, and place() re-windows it afterwards.
    TimeStretch(min_rate=0.75, max_rate=1.35, p=0.9, leave_length_unchanged=False),
])


def embed_aug(files, k, label=""):
    """One clean copy per clip, then k augmented ones — augmented BEFORE being windowed."""
    out = []
    for i, f in enumerate(files):
        raw = load_raw(f)
        out.append(to_feat(place(raw)))
        for _ in range(k):
            out.append(to_feat(place(aug(samples=raw, sample_rate=16000))))
        if label and (i % 50 == 0 or i == len(files) - 1):
            print(f"  embedding {label}: {i + 1}/{len(files)} clips...", flush=True)
    return np.stack(out).astype(np.float32)


pos = sorted(glob.glob(os.path.join(POS_DIR, "*.wav")))
rng.shuffle(pos)
neg = sorted(glob.glob(os.path.join(NEG_DIR, "*.wav")))
rng.shuffle(neg)
nvp, nvn = max(8, len(pos) // 5), max(20, len(neg) // 5)
vp_f, tp_f = pos[:nvp], pos[nvp:]
vn_f, tn_f = neg[:nvn], neg[nvn:]
print(f"POS {len(pos)} ({nvp} val) | NEG {len(neg)} ({nvn} val)", flush=True)

real_pos = embed_aug(tp_f, 20, "positives")
real_neg = embed_aug(tn_f, 8, "negatives (1724 clips — this is the slow part)")

# Silence MUST be learned as negative — otherwise the model mistakes an empty room for the
# wake word (see issue #2).
sil_train = silence_feats(10)
real_neg = np.concatenate([real_neg, sil_train])
print(f"silence negatives added: {len(sil_train)}", flush=True)

val_pos = np.stack([to_feat(load16k(f)) for f in vp_f]).astype(np.float32)
val_neg = np.stack([to_feat(load16k(f)) for f in vn_f]).astype(np.float32)
val_sil = silence_feats(5)          # held-out silence -> honest numbers


def tempo_feats(files, rate):
    """The held-out positives, spoken faster or slower — nothing else changed.

    THE question this trainer has to answer. Validating only on the original clips reports a
    proud 90% recall and tells you nothing: those clips come from the same session as the
    training data, so of course the model knows them. A model can score 90% here and still be
    deaf in the living room — that exact thing happened (issue #2).

    A wake word is worthless if it only answers to the one cadence someone used while a script
    was prompting them. If recall collapses a few rows down, the model has memorised a
    recording session, not learned a word.
    """
    out = []
    for f in files:
        d = load_raw(f)
        n = max(1, int(len(d) / rate))
        d = np.interp(np.linspace(0, len(d) - 1, n), np.arange(len(d)), d).astype(np.float32)
        out.append(to_feat(place(d)))
    return np.stack(out).astype(np.float32)


TEMPI = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
val_tempo = {r: tempo_feats(vp_f, r) for r in TEMPI}

# The bulk negatives are OPTIONAL. The shipped feature file (ACAV100M) is CC-BY-NC-SA — fine for
# private use, but it forbids commercial use, so a wake model trained WITH it cannot be sold.
# If the file is absent (or WW_NO_ACAV=1), train on the recorded session negatives + silence
# ALONE — 100% your own, commercially clean. The trade-off is diversity: fewer unseen sounds are
# covered, so watch the false-alarm numbers in the validation below to see if it is enough.
USE_ACAV = os.path.exists(NEG_FEATURES) and os.environ.get("WW_NO_ACAV", "0") != "1"
if USE_ACAV:
    neg_mm = np.load(NEG_FEATURES, mmap_mode="r")
    take = min(N_NEG_SUBSET, neg_mm.shape[0])
    acav = torch.from_numpy(
        np.asarray(neg_mm[np.sort(rng.choice(neg_mm.shape[0], take, replace=False))], dtype=np.float16)
    )
    print(f"bulk negatives: ACAV100M, {take} windows (NC-licensed — private use only)", flush=True)
else:
    acav = None
    print(f"bulk negatives: NONE — session negatives + silence only "
          f"({len(real_neg)} windows, commercially clean)", flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
realp_t, realn_t = torch.from_numpy(real_pos).to(dev), torch.from_numpy(real_neg).to(dev)
synth_t = (
    torch.from_numpy(np.load(SYNTH_POS_NPY).astype(np.float32)).to(dev)
    if os.path.exists(SYNTH_POS_NPY)
    else None
)
print(f"synthetic positives: {'yes' if synth_t is not None else 'no'} | device {dev}", flush=True)


class _FCN(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.fc, s.relu, s.ln = nn.Linear(d, d), nn.ReLU(), nn.LayerNorm(d)

    def forward(s, x):
        return s.relu(s.ln(s.fc(x)))


class Net(nn.Module):
    def __init__(s, d=128, nb=1):
        super().__init__()
        s.flat = nn.Flatten()
        s.l1 = nn.Linear(16 * 96, d)
        s.relu, s.ln = nn.ReLU(), nn.LayerNorm(d)
        s.blocks = nn.ModuleList([_FCN(d) for _ in range(nb)])
        s.last, s.act = nn.Linear(d, 1), nn.Sigmoid()

    def forward(s, x):
        x = s.relu(s.ln(s.l1(s.flat(x))))
        for b in s.blocks:
            x = b(x)
        return s.act(s.last(x))


net = Net().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
bce = torch.nn.functional.binary_cross_entropy
for step in range(1, STEPS + 1):
    parts, npos = [], 0
    if synth_t is not None:
        parts.append(synth_t[torch.randint(0, synth_t.shape[0], (128,), device=dev)])
        npos += 128
    parts.append(realp_t[torch.randint(0, realp_t.shape[0], (192,), device=dev)])
    npos += 192
    if acav is not None:
        parts.append(acav[torch.randint(0, acav.shape[0], (640,))].to(dev, torch.float32))
        parts.append(realn_t[torch.randint(0, realn_t.shape[0], (320,), device=dev)])
    else:
        # no bulk set — draw all 960 negatives from the recorded session negatives + silence
        parts.append(realn_t[torch.randint(0, realn_t.shape[0], (960,), device=dev)])
    xb = torch.cat(parts, 0)
    yb = torch.cat([torch.ones(npos, device=dev), torch.zeros(960, device=dev)])
    pred = net(xb).squeeze(1)
    keep = ((yb == 0) & (pred >= 0.001)) | ((yb == 1) & (pred < 0.999))   # hard-example mining
    p, y = pred[keep], yb[keep]
    if p.numel() == 0:
        continue
    loss = bce(p, y, weight=torch.where(y == 1, 3.0, 1.0))
    opt.zero_grad()
    loss.backward()
    opt.step()

net.eval()
with torch.no_grad():
    vp = net(torch.from_numpy(val_pos).to(dev)).squeeze(1).cpu().numpy()
    vn = net(torch.from_numpy(val_neg).to(dev)).squeeze(1).cpu().numpy()
    vs = net(torch.from_numpy(val_sil).to(dev)).squeeze(1).cpu().numpy()

print(f"\n=== VALIDATION (held out: {len(vp)} POS / {len(vn)} NEG / {len(vs)} SILENCE) ===", flush=True)
for t in [0.5, 0.7, 0.9]:
    print(
        f"  threshold {t}: recall {(vp >= t).mean() * 100:.0f}%  "
        f"false alarms {(vn >= t).mean() * 100:.1f}%  "
        f"SILENCE false alarms {(vs >= t).mean() * 100:.1f}%"
    )

# Silence MUST sit at 0%. Anything else means the model mistakes an empty room for the wake
# word — exactly the bug from issue #2, which used to slip through unnoticed.
if (vs >= 0.5).mean() > 0:
    print("\n  !! WARNING: the model fires on SILENCE. Do not ship it.", flush=True)

# --- THE REAL TEST: does it still know the word when it is spoken differently? --------------
print("\n=== ROBUSTNESS: the same held-out calls, spoken faster / slower ===", flush=True)
print("  (recall at threshold 0.7 — a model that only works at 1.00 has memorised a session)")
weak = []
with torch.no_grad():
    for r in TEMPI:
        v = net(torch.from_numpy(val_tempo[r]).to(dev)).squeeze(1).cpu().numpy()
        recall = (v >= 0.7).mean() * 100
        bar = "#" * int(recall / 5)
        tag = "  <- as recorded" if r == 1.00 else ""
        print(f"  speed {r:.2f}: {recall:3.0f}%  {bar}{tag}", flush=True)
        if 0.85 <= r <= 1.25 and recall < 60:
            weak.append(r)

if weak:
    print("\n  !! WARNING: recall collapses at normal speaking rates "
          f"({', '.join(f'{r:.2f}x' for r in weak)}).", flush=True)
    print("     This model will pass its own validation and then fail in the room. Record the")
    print("     wake word the way you ACTUALLY say it — casually, mid-sentence, across the")
    print("     room — not only the way a prompt tells you to say it.", flush=True)
else:
    print("\n  Recall holds across speaking rates: it learned the WORD, not the recording.", flush=True)

net.to("cpu")
torch.onnx.export(net, torch.rand(1, 16, 96), OUT, output_names=["probability"], dynamo=False)
print("exported:", OUT, flush=True)
