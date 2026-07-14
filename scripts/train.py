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

# ======================= CONFIG (adjust) =========================
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
PHRASE = "hey_horus"
MODELS_DIR = os.path.join(DATA, "models")                       # melspectrogram.onnx + embedding_model.onnx (openWakeWord v0.5.1)
POS_DIR = os.path.join(DATA, "session-pos")                     # clean session: ONLY the wake word, 16 kHz mono WAV
NEG_DIR = os.path.join(DATA, "session-neg")                     # clean session: ONLY negatives
NEG_FEATURES = os.path.join(DATA, "neg_ACAV100M_2000hrs.npy")   # from download_negatives.py
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


def load16k(path):
    """Load a clip into a TARGET-length window.

    Do NOT zero-pad. The positives are short (~0.8 s) and were previously left-padded with
    DIGITAL SILENCE to fill the 2 s window, while the bulk of the negatives (the ACAV features)
    is full-length audio. That turns "lots of leading silence" into a near-perfect cue for the
    positive class — the model duly learns it and then fires at score 1.00 into a completely
    silent room (measured; see issue #2).

    Instead: place the clip at a RANDOM offset inside the window and fill the rest with a faint
    noise floor, so the padding carries no class information.
    """
    d, sr = sf.read(path, dtype="float32", always_2d=False)
    if d.ndim > 1:
        d = d.mean(1)
    d = d[:TARGET]
    if len(d) == TARGET:
        return d
    out = (rng.standard_normal(TARGET) * 1e-3).astype(np.float32)   # faint room-tone floor
    off = int(rng.integers(0, TARGET - len(d) + 1))
    out[off:off + len(d)] += d
    return out


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


aug = Compose([
    AddGaussianSNR(min_snr_db=3.0, max_snr_db=30.0, p=0.9),
    Gain(min_gain_db=-10.0, max_gain_db=6.0, p=0.8),
    PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5, leave_length_unchanged=True),
])


def embed_aug(files, k):
    out = []
    for f in files:
        sig = load16k(f)
        out.append(to_feat(sig))
        for _ in range(k):
            out.append(to_feat(aug(samples=sig, sample_rate=16000)))
    return np.stack(out).astype(np.float32)


pos = sorted(glob.glob(os.path.join(POS_DIR, "*.wav")))
rng.shuffle(pos)
neg = sorted(glob.glob(os.path.join(NEG_DIR, "*.wav")))
rng.shuffle(neg)
nvp, nvn = max(8, len(pos) // 5), max(20, len(neg) // 5)
vp_f, tp_f = pos[:nvp], pos[nvp:]
vn_f, tn_f = neg[:nvn], neg[nvn:]
print(f"POS {len(pos)} ({nvp} val) | NEG {len(neg)} ({nvn} val)", flush=True)

real_pos = embed_aug(tp_f, 20)
real_neg = embed_aug(tn_f, 8)

# Silence MUST be learned as negative — otherwise the model mistakes an empty room for the
# wake word (see issue #2).
sil_train = silence_feats(10)
real_neg = np.concatenate([real_neg, sil_train])
print(f"silence negatives added: {len(sil_train)}", flush=True)

val_pos = np.stack([to_feat(load16k(f)) for f in vp_f]).astype(np.float32)
val_neg = np.stack([to_feat(load16k(f)) for f in vn_f]).astype(np.float32)
val_sil = silence_feats(5)          # held-out silence -> honest numbers

neg_mm = np.load(NEG_FEATURES, mmap_mode="r")
acav = torch.from_numpy(
    np.asarray(neg_mm[np.sort(rng.choice(neg_mm.shape[0], N_NEG_SUBSET, replace=False))], dtype=np.float16)
)
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
    parts.append(acav[torch.randint(0, acav.shape[0], (640,))].to(dev, torch.float32))
    parts.append(realn_t[torch.randint(0, realn_t.shape[0], (320,), device=dev)])
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

net.to("cpu")
torch.onnx.export(net, torch.rand(1, 16, 96), OUT, output_names=["probability"], dynamo=False)
print("exported:", OUT, flush=True)
