"""
Record TV (or any ambient) audio as NEGATIVE training material — so the wake word learns your
living room and stops firing on the television.

The clean way to fix false wakes: not fishing one moment out of a log (live wake detection does
not reproduce offline), but capturing minutes of the real enemy through the real mic, in the
real room. You stay QUIET; the TV plays as it normally does. The recording is chopped into
2-second clips (the trainer's window) and dropped into session-neg.

    cd C:\\Users\\robin\\dev\\oww-train
    .\\Scripts\\activate                 # or just system python — needs sounddevice+soundfile
    python C:\\Users\\robin\\dev\\custom-wakeword-trainer\\scripts\\record_tv_negatives.py

    WW_DATA=...  python ... record_tv_negatives.py          # custom data dir
    python record_tv_negatives.py 240                       # record 240s instead of 180

IMPORTANT: do NOT say the wake word during this. It is NEGATIVE data — anything you say here
teaches her that sound is "not the wake word". Just let the TV run and stay quiet.
"""
import os
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

RATE = 16000
CLIP = 2.0                    # seconds per negative clip — matches the trainer's window
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
DATA = os.environ.get("WW_DATA") or os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(DATA, "session-neg")


def main():
    os.makedirs(OUT, exist_ok=True)
    existing = len([f for f in os.listdir(OUT) if f.endswith(".wav")])
    dev = sd.query_devices(sd.default.device[0], "input")
    print("=== recording TV as NEGATIVE material ===")
    print(f"mic   : {dev['name']}")
    print(f"length: {DURATION:.0f}s  ->  ~{int(DURATION / CLIP)} clips of {CLIP:.0f}s")
    print(f"{existing} negatives already in session-neg/\n")
    print("Have the TV playing at a NORMAL volume. STAY QUIET — do not say her name.")
    input("Press Enter to start recording...")

    print(f"\n  recording {DURATION:.0f}s — stay quiet...")
    audio = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype="int16")
    for left in range(int(DURATION), 0, -5):
        print(f"    {left:4d}s left ", end="\r", flush=True)
        time.sleep(5)
    sd.wait()
    audio = audio.reshape(-1)

    peak = int(np.abs(audio).max())
    if peak < 400:
        print(f"\n  ! very quiet (peak {peak}) — was the TV actually audible to the mic? "
              "These clips would be near-silence, not TV. Consider turning it up and redoing.")

    n = int(len(audio) // (CLIP * RATE))
    kept = 0
    for i in range(n):
        clip = audio[int(i * CLIP * RATE):int((i + 1) * CLIP * RATE)]
        if int(np.abs(clip).max()) < 200:        # skip a silent gap — it teaches nothing
            continue
        kept += 1
        sf.write(os.path.join(OUT, f"tv_{existing + kept:05d}.wav"), clip, RATE, subtype="PCM_16")

    print(f"\n\n  saved {kept} TV negative clips (peak {peak}).")
    print(f"  session-neg now holds {existing + kept} negatives.")
    print("\n  Next: retrain, and check the wake word still catches YOU while ignoring the TV.")


if __name__ == "__main__":
    main()
