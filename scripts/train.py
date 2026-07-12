"""Trainiert einen openWakeWord-Wake-Word-Klassifikator aus SAUBEREN Zwei-Sessions-Daten.

Datenphilosophie (der ganze Trick): eine Aufnahme-Session nur mit dem Wake-Word, eine nur mit
Negatives, GETRENNT gespeichert - kein Auto-Labeling, kein Label-Rauschen. Positive werden stark
augmentiert (Rauschen/Pegel/Pitch/Tempo), damit das Modell nicht nur "sauberes Audio = positiv"
lernt. Ein Teil der echten POS/NEG wird als Validierung zurueckgehalten (ehrliche Zahlen).

Ausgang des Modells ist eine Wahrscheinlichkeit 0..1 (Sigmoid im Modell) -> die Laufzeit braucht
kein eigenes Sigmoid. Architektur byte-identisch zu openwakeword.train (model_type="dnn")."""
import os, glob, numpy as np, torch, soundfile as sf
import torch.nn as nn
from audiomentations import Compose, AddGaussianSNR, Gain, PitchShift, TimeStretch
from openwakeword.utils import AudioFeatures

# ======================= CONFIG (anpassen) =======================
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
PHRASE = "hey_horus"
MODELS_DIR = os.path.join(DATA, "models")                 # melspectrogram.onnx + embedding_model.onnx (openWakeWord v0.5.1)
POS_DIR = os.path.join(DATA, "session-pos")               # saubere Session: NUR Wake-Word, 16 kHz mono WAV
NEG_DIR = os.path.join(DATA, "session-neg")               # saubere Session: NUR Negatives
NEG_FEATURES = os.path.join(DATA, "neg_ACAV100M_2000hrs.npy")   # von download_negatives.py
SYNTH_POS_NPY = os.path.join(DATA, "pos_synth.npy")        # optional: vor-embeddete piper-Positive (N,16,96) oder fehlend
OUT = os.path.join(HERE, "..", f"{PHRASE}.onnx")
TARGET = 32000            # 2 s @ 16 kHz -> genau 16 Embedding-Frames
N_NEG_SUBSET = 800_000
STEPS = 25000
# =================================================================

rng = np.random.default_rng(0)
F = AudioFeatures(melspec_model_path=os.path.join(MODELS_DIR, "melspectrogram.onnx"),
                  embedding_model_path=os.path.join(MODELS_DIR, "embedding_model.onnx"))


def load16k(path):
    d, sr = sf.read(path, dtype="float32", always_2d=False)
    if d.ndim > 1: d = d.mean(1)
    if len(d) < TARGET: d = np.concatenate([np.zeros(TARGET - len(d), np.float32), d])
    return d[:TARGET]


def to_feat(sig):
    clip = (np.clip(sig, -1, 1) * 32767).astype(np.int16)[None, :]
    fe = F.embed_clips(clip, batch_size=1); n = fe.shape[1]; s = max(0, (n - 16) // 2)
    w = fe[:, s:s + 16, :]
    if w.shape[1] < 16: w = np.pad(w, ((0, 0), (0, 16 - w.shape[1]), (0, 0)))
    return w[0]


aug = Compose([
    AddGaussianSNR(min_snr_db=3.0, max_snr_db=30.0, p=0.9),
    Gain(min_gain_db=-10.0, max_gain_db=6.0, p=0.8),
    PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5, leave_length_unchanged=True),
])


def embed_aug(files, k):
    out = []
    for f in files:
        sig = load16k(f); out.append(to_feat(sig))
        for _ in range(k): out.append(to_feat(aug(samples=sig, sample_rate=16000)))
    return np.stack(out).astype(np.float32)


pos = sorted(glob.glob(os.path.join(POS_DIR, "*.wav"))); rng.shuffle(pos)
neg = sorted(glob.glob(os.path.join(NEG_DIR, "*.wav"))); rng.shuffle(neg)
nvp, nvn = max(8, len(pos) // 5), max(20, len(neg) // 5)
vp_f, tp_f = pos[:nvp], pos[nvp:]
vn_f, tn_f = neg[:nvn], neg[nvn:]
print(f"POS {len(pos)} ({nvp} val) | NEG {len(neg)} ({nvn} val)", flush=True)

real_pos = embed_aug(tp_f, 20); real_neg = embed_aug(tn_f, 8)
val_pos = np.stack([to_feat(load16k(f)) for f in vp_f]).astype(np.float32)
val_neg = np.stack([to_feat(load16k(f)) for f in vn_f]).astype(np.float32)

neg_mm = np.load(NEG_FEATURES, mmap_mode="r")
acav = torch.from_numpy(np.asarray(neg_mm[np.sort(rng.choice(neg_mm.shape[0], N_NEG_SUBSET, replace=False))], dtype=np.float16))
dev = "cuda" if torch.cuda.is_available() else "cpu"
realp_t, realn_t = torch.from_numpy(real_pos).to(dev), torch.from_numpy(real_neg).to(dev)
synth_t = torch.from_numpy(np.load(SYNTH_POS_NPY).astype(np.float32)).to(dev) if os.path.exists(SYNTH_POS_NPY) else None
print(f"synthetische Positive: {'ja' if synth_t is not None else 'nein'} | device {dev}", flush=True)


class _FCN(nn.Module):
    def __init__(s, d):
        super().__init__(); s.fc, s.relu, s.ln = nn.Linear(d, d), nn.ReLU(), nn.LayerNorm(d)
    def forward(s, x): return s.relu(s.ln(s.fc(x)))
class Net(nn.Module):
    def __init__(s, d=128, nb=1):
        super().__init__(); s.flat = nn.Flatten(); s.l1 = nn.Linear(16 * 96, d)
        s.relu, s.ln = nn.ReLU(), nn.LayerNorm(d); s.blocks = nn.ModuleList([_FCN(d) for _ in range(nb)])
        s.last, s.act = nn.Linear(d, 1), nn.Sigmoid()
    def forward(s, x):
        x = s.relu(s.ln(s.l1(s.flat(x))))
        for b in s.blocks: x = b(x)
        return s.act(s.last(x))


net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
bce = torch.nn.functional.binary_cross_entropy
for step in range(1, STEPS + 1):
    parts, npos = [], 0
    if synth_t is not None:
        parts.append(synth_t[torch.randint(0, synth_t.shape[0], (128,), device=dev)]); npos += 128
    parts.append(realp_t[torch.randint(0, realp_t.shape[0], (192,), device=dev)]); npos += 192
    parts.append(acav[torch.randint(0, acav.shape[0], (640,))].to(dev, torch.float32))
    parts.append(realn_t[torch.randint(0, realn_t.shape[0], (320,), device=dev)])
    xb = torch.cat(parts, 0)
    yb = torch.cat([torch.ones(npos, device=dev), torch.zeros(960, device=dev)])
    pred = net(xb).squeeze(1)
    keep = ((yb == 0) & (pred >= 0.001)) | ((yb == 1) & (pred < 0.999))   # Hard-Example-Mining
    p, y = pred[keep], yb[keep]
    if p.numel() == 0: continue
    loss = bce(p, y, weight=torch.where(y == 1, 3.0, 1.0)); opt.zero_grad(); loss.backward(); opt.step()

net.eval()
with torch.no_grad():
    vp = net(torch.from_numpy(val_pos).to(dev)).squeeze(1).cpu().numpy()
    vn = net(torch.from_numpy(val_neg).to(dev)).squeeze(1).cpu().numpy()
print(f"\n=== VALIDIERUNG (zurueckgehalten: {len(vp)} POS / {len(vn)} NEG) ===", flush=True)
for t in [0.5, 0.7, 0.9]:
    print(f"  Schwelle {t}: Recall {(vp>=t).mean()*100:.0f}%  Fehlalarme {(vn>=t).mean()*100:.1f}%")

net.to("cpu")
torch.onnx.export(net, torch.rand(1, 16, 96), OUT, output_names=["probability"], dynamo=False)
print("Exportiert:", OUT, flush=True)
