"""
Record the wake word the way you ACTUALLY say it — not the way a prompt makes you say it.

A recording script asks for the word, you say the word, it beeps, you say it again. Fifty
times. The model learns that: a command, isolated, deliberate, evenly paced, one evening.

Then you talk to the thing, and it is deaf. Measured (ANIMA #19):

    "Anima."                     spoken alone      -> score 1.00, wakes every time
    "Anima, wie geht's dir?"     spoken naturally  -> score 0.00, three attempts, nothing

Not because a sentence follows — a sentence glued behind a clean recording still scores 0.99.
Because the NAME ITSELF changes when it lives inside a sentence: faster, flatter, swallowed.
A called name is a command. A spoken name is just one word among others.

So this records the name inside real sentences, in the ways a person actually uses it: at the
start, in the middle, at the end, as a question, as an aside. It is the same lesson as the
speaking-rate bug, one level up — the model must learn the WORD, not a recording session.

    python scripts/record_natural.py                 (defaults to WW_PHRASE)
    WW_PHRASE=anima python scripts/record_natural.py

Writes into <data>/session-pos/ alongside existing positives. Then retrain.
"""
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

RATE = 16000
SECONDS = 2.5               # a short natural utterance, same window the trainer uses
READ_TIME = 4               # seconds to read the line — no keypress near a recording
PHRASE = os.environ.get("WW_PHRASE", "anima").capitalize()
DATA = os.environ.get("WW_DATA") or os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(DATA, "session-pos")

# The name in every position it takes in real speech. NOT a list of commands — a list of ways
# a person drops a name into a sentence without thinking about it.
LINES = [
    "{p}, wie geht es dir?",
    "{p}, mach mal bitte das Licht an.",
    "Sag mal {p}, was hältst du davon?",
    "Hey {p}, hörst du mich?",
    "{p}? Bist du da?",
    "Ich glaub, {p} hat das schon erledigt.",
    "{p}, kannst du kurz helfen?",
    "Was meinst du, {p}?",
    "{p}, hör mal.",
    "Guck mal {p}, das ist interessant.",
    "{p}, wie spät ist es eigentlich?",
    "Ach {p}, das ist doch Quatsch.",
]

ROUNDS = 3                  # each line said three times — nobody says it the same way twice


def record(seconds):
    a = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16")
    sd.wait()
    return a.reshape(-1)


os.makedirs(OUT, exist_ok=True)
existing = sorted(f for f in os.listdir(OUT) if f.endswith(".wav"))
start = len(existing)
total = len(LINES) * ROUNDS

print(f"=== teaching her how you SPEAK her name, not how you call it ===")
print(f"mic: {sd.query_devices(sd.default.device[0], 'input')['name']}")
print(f"{start} positives already recorded — these are ADDED to them.")
print(f"{len(LINES)} sentences x {ROUNDS} rounds = {total} clips, "
      f"roughly {total * (SECONDS + READ_TIME + 1) / 60:.0f} minutes.\n")
print("Say them NORMALLY. Do not enunciate. Do not perform. Swallow the name if that is what")
print("you do — that swallowed version is exactly what she cannot hear today.\n")
input("PRESS ENTER to start, then keep your hands off the keyboard...")
print("running on its own from here.")
time.sleep(2.0)

n = 0
for r in range(ROUNDS):
    print(f"\n=== round {r + 1} of {ROUNDS} ===")
    for line in LINES:
        text = line.format(p=PHRASE)
        print(f'\n    say: "{text}"')
        for left in range(READ_TIME, 0, -1):
            print(f"    reading... {left}  ", end="\r", flush=True)
            time.sleep(1.0)
        print(f"    >>> NOW <<<                    ")
        a = record(SECONDS)
        n += 1
        peak = int(np.abs(a).max())
        path = os.path.join(OUT, f"pos_{start + n:04d}.wav")
        sf.write(path, a, RATE, subtype="PCM_16")
        warn = "   ! very quiet" if peak < 800 else ""
        print(f"    saved {os.path.basename(path)}  (peak {peak}){warn}")

print(f"\n{n} new positives. {start + n} in total.")
print("Now retrain — and watch the ROBUSTNESS block, not the recall number.")
