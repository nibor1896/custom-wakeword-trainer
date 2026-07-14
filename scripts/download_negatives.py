"""Downloads the pre-computed openWakeWord negative features (~16 GB).

Shape (N, 16, 96) float16 — already cut into 16-frame windows, so they can be used as
negatives directly, without any audio processing on your side.

Robust and resumable: HuggingFace over Windows schannel likes to drop the SSL connection
(curl fails with exit 35), so this uses requests (certifi) plus auto-resume via a Range
header inside a retry loop.

NOTE: the download comes from huggingface.co. Some networks / security suites block it
(connections get reset with WinError 10054). If this never gets past the first megabytes,
that is why — fetch the file manually in a browser and drop it at the DST path below.
"""
import os
import time

import requests

URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
DST = os.path.join(os.path.dirname(__file__), "..", "data", "neg_ACAV100M_2000hrs.npy")
TOTAL = 17_280_000_128

os.makedirs(os.path.dirname(DST), exist_ok=True)
for attempt in range(100000):
    pos = os.path.getsize(DST) if os.path.exists(DST) else 0
    if pos >= TOTAL:
        print(f"done: {pos} bytes", flush=True)
        break
    try:
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        with requests.get(URL, headers=headers, stream=True, timeout=(30, 90)) as r:
            r.raise_for_status()
            with open(DST, "ab" if pos else "wb") as f:
                for chunk in r.iter_content(chunk_size=4 << 20):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        have = os.path.getsize(DST) if os.path.exists(DST) else 0
        print(f"retry {attempt} at {have}/{TOTAL}: {e}", flush=True)
        time.sleep(3)
