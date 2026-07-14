"""
Record the CLEAN sessions the recipe needs — no auto-labeling, hence no label noise.

POSITIVES — on a fixed cadence, WITHOUT touching the keyboard:

    python scripts/record_sessions.py pos 50 12     # 50 clips, one cue every 12 s

  The script counts you in and prints ">>> NOW <<<" — you speak, it records exactly then.
  Nobody presses Enter, so no keystroke can ever land inside a positive clip. That matters:
  we are about to teach the model that keyboards are NOT the wake word, so a keystroke in
  the positives would poison exactly that lesson.

NEGATIVES — recorded continuously and sliced into 2 s clips, one CATEGORY at a time:

    python scripts/record_sessions.py neg 180 keyboard   # only typing
    python scripts/record_sessions.py neg 180 tv         # TV / people talking / everyday sounds
    python scripts/record_sessions.py neg 180 music      # music bleeding out of your headphones
    python scripts/record_sessions.py neg 120 speech     # you talking, reading, near-homophones
    python scripts/record_sessions.py neg 60  silence    # just the room

  Separate sessions keep the categories clean and let you see later WHICH kind of sound
  causes false alarms. They all land in session-neg/ (the trainer reads the whole folder),
  but the filenames keep the category.

Everything is 16 kHz mono WAV, 2 s per clip. Data folder via WW_DATA (same as the trainer).
"""
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd

RATE = 16000
CLIP_S = 2.0                    # exactly what the trainer expects (TARGET = 32000)
CLIP_N = int(RATE * CLIP_S)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("WW_DATA") or os.path.join(HERE, "..", "data")

HINTS = {
    "keyboard": [
        "TYPE. Hammer on it, every key, fast and slow.",
        "Space, Enter, Backspace, the loud ones. Rolls of keys. Single hard strokes.",
        "This is the #1 false trigger — be generous.",
    ],
    "tv": [
        "TV / series / news / YouTube through your SPEAKERS — real speech that is not yours.",
        "The nastiest case: it IS speech, so no VAD will ever save you. Let people talk,",
        "argue, laugh, talk over each other. Plus everyday sounds: dishes, doors, footsteps,",
        "chair, mug on the desk, phone notifications, coughing.",
    ],
    "music": [
        "Music bleeding out of your HEADPHONES into the mic — exactly the leak we measured",
        "with the echo guard. Different genres, vocals included, a bit louder than usual.",
    ],
    "speech": [
        "YOU talking: normal chat, reading aloud, on the phone.",
        "And plenty of NEAR-HOMOPHONES of the wake word — for 'Anima':",
        "   Anna, Amina, an mir, Anime, Aroma, Marina, Amir, Animation ... say them a lot.",
    ],
    "silence": [
        "Nothing. Just the room, the fan, your breathing.",
        "The model must learn that an empty room is NOT the wake word.",
    ],
}


def save(path, pcm_int16):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm_int16.tobytes())


def record_positives(n, every):
    out = os.path.join(DATA, "session-pos")
    os.makedirs(out, exist_ok=True)
    start = len([f for f in os.listdir(out) if f.endswith(".wav")])

    print(f"\nSession POS — ONLY the wake word, {n} clips, a cue every {every:.0f}s -> {out}")
    print(f"Total time: about {n * every / 60:.0f} minutes.\n")
    print("  Say it on the cue, and VARY it deliberately:")
    print("   normal / quieter / louder / faster / slower / turned away / slightly mumbled.")
    print("  Do NOT touch the keyboard — the cadence is automatic.\n")
    input("Enter ONCE to start (then hands off the keyboard)...")
    time.sleep(2.0)

    for i in range(1, n + 1):
        for c in (3, 2, 1):
            print(f"   {c}...", end="\r", flush=True)
            time.sleep(0.6)
        print("   >>> NOW <<<          ", flush=True)
        audio = sd.rec(CLIP_N, samplerate=RATE, channels=1, dtype="int16")
        sd.wait()
        peak = int(np.abs(audio).max())
        path = os.path.join(out, f"pos_{start + i:03d}.wav")
        save(path, audio)
        flag = "ok" if peak > 1500 else "VERY QUIET"
        print(f"   [{i}/{n}] saved  level {peak}  ({flag})\n")

        rest = every - CLIP_S - 1.8
        if i < n and rest > 0:
            time.sleep(rest)

    print(f"Done — {n} positive clips (keyboard never touched).")


def record_negatives(seconds, label):
    out = os.path.join(DATA, "session-neg")
    os.makedirs(out, exist_ok=True)
    start = len([f for f in os.listdir(out) if f.startswith(f"neg_{label}_")])
    n_clips = int(seconds // CLIP_S)

    print(f"\nSession NEG [{label}] — {seconds:.0f}s -> {n_clips} clips -> {out}")
    print("NEVER say the wake word in this session.\n")
    for line in HINTS.get(label, ["(no hints for this category)"]):
        print(f"   {line}")
    print("\n  Record it through your REAL mic in your REAL room — clean sample files")
    print("  help far less than the actual path she will live in.\n")

    input("Enter to start recording...")
    print("\n   >>> RECORDING <<<\n")

    for i in range(1, n_clips + 1):
        audio = sd.rec(CLIP_N, samplerate=RATE, channels=1, dtype="int16")
        sd.wait()
        peak = int(np.abs(audio).max())
        path = os.path.join(out, f"neg_{label}_{start + i:04d}.wav")
        save(path, audio)
        left = (n_clips - i) * CLIP_S
        print(f"   {i}/{n_clips}  level {peak:5d}   {left:5.0f}s left", flush=True)

    print(f"\nDone — {n_clips} negative clips [{label}].")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode = sys.argv[1]
    print(f"data folder: {DATA}")

    if mode == "pos":
        n = int(sys.argv[2])
        every = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
        record_positives(n, every)
    elif mode == "neg":
        seconds = float(sys.argv[2])
        label = sys.argv[3] if len(sys.argv) > 3 else "misc"
        record_negatives(seconds, label)
    else:
        sys.exit(__doc__)
