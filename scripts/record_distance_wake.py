"""
Record the WAKE WORD from across the room — sound-guided, because you can't see the screen.

The speaker check now recognises robin from the sofa, but the WAKE WORD itself still misses a
quiet, distant call: it was trained on the phrase spoken up close, and a smeared far call never
reaches the fire threshold. Same lesson as speaking rate, one axis over — a recording session
at the desk is not the whole room.

Unlike sentences with the name embedded (issue #2), a distant call is still an ISOLATED wake
word, centred in its clip, so the trainer's middle-window can use it directly. It is safe to
add — with ONE thing to verify after retraining: a quiet distant positive sits close to
background noise, so watch the false-alarm and SILENCE-false-alarm numbers do not rise.

Guided entirely by sound (you're on the sofa):
    two low beeps = get ready    one HIGH beep = say the wake word ONCE    one low beep = done
    three rising beeps = position done, walk back to the PC and press Enter for the next

    WW_PHRASE=anima WW_DATA=...\\data python scripts/record_distance_wake.py

Writes dist_*.wav into <data>/session-pos/ (distinct prefix, so a retrain can drop them again
if they hurt). Then retrain and check the ROBUSTNESS + false-alarm blocks.
"""
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

RATE = 16000
SECONDS = 2.0             # one "Anima" fits easily; the trainer windows the middle
CALLS_PER_POS = 8        # how many calls at each spot
WALK_TIME = 14           # seconds to walk to the spot after pressing Enter at the PC
PHRASE = os.environ.get("WW_PHRASE", "anima").capitalize()
DATA = os.environ.get("WW_DATA") or os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(DATA, "session-pos")

POSITIONS = [
    ("sofa_norm",  "SOFA, facing the mic, NORMAL volume."),
    ("sofa_quiet", "SOFA, QUIET — the way you'd say it half-asleep. This is the one that fails."),
    ("sofa_turn",  "SOFA, TURNED AWAY toward the TV."),
    ("far",        "STANDING far across the room."),
]


def beep(freq, ms, vol=0.35):
    t = np.linspace(0, ms / 1000, int(RATE * ms / 1000), False)
    tone = (vol * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    e = int(RATE * 0.005)
    tone[:e] *= np.linspace(0, 1, e)
    tone[-e:] *= np.linspace(1, 0, e)
    sd.play(tone, RATE)
    sd.wait()


def cue_ready():
    beep(430, 110); time.sleep(0.1); beep(430, 110)
def cue_go():   beep(880, 300)
def cue_stop(): beep(320, 200)
def cue_done():
    for f in (520, 660, 880):
        beep(f, 120)


def record(seconds):
    a = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16")
    sd.wait()
    return a.reshape(-1)


os.makedirs(OUT, exist_ok=True)
existing = len([f for f in os.listdir(OUT) if f.endswith(".wav")])
print(f'=== recording "{PHRASE}" FROM DISTANCE — guided by sound ===')
print(f"mic: {sd.query_devices(sd.default.device[0], 'input')['name']}")
print(f"{existing} positives already there — these are ADDED.")
print(f"{len(POSITIONS)} positions x {CALLS_PER_POS} calls = {len(POSITIONS) * CALLS_PER_POS} clips.\n")
print("Listen for the beeps:")
print(f"  two low = get ready    ONE HIGH = say \"{PHRASE}\" once    one low = done")
print("  three rising = come back, press Enter for the next\n")

n = 0
for tag, how in POSITIONS:
    print(f"\n--- {tag.upper()} --- {how}")
    input(f"    PRESS ENTER, then walk to the spot ({WALK_TIME}s)...")
    for left in range(WALK_TIME, 0, -1):
        print(f"    walk to your spot... {left:2d}   ", end="\r", flush=True)
        time.sleep(1.0)
    print("    >>> recording now — listen for the beeps <<<        ")
    beep(660, 200)
    time.sleep(0.8)

    for i in range(CALLS_PER_POS):
        cue_ready()
        time.sleep(0.6)
        cue_go()                      # <- say the wake word ONCE after this
        a = record(SECONDS)
        cue_stop()
        n += 1
        peak = int(np.abs(a).max())
        path = os.path.join(OUT, f"dist_{tag}_{existing + n:04d}.wav")
        sf.write(path, a, RATE, subtype="PCM_16")
        warn = "   ! nothing captured" if peak < 150 else ""
        print(f"    saved {os.path.basename(path)}  (peak {peak}){warn}")
        time.sleep(0.5)

    cue_done()
    print("    position done — walk back and press Enter for the next.")

print(f"\n{n} distance wake calls added. Now retrain and CHECK the false-alarm numbers:")
print("  WW_PHRASE=anima WW_DATA=... WW_MODELS=... python scripts/train.py")
