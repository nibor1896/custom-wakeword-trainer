"""Laedt die vorberechneten openWakeWord-Negativ-Features (~16 GB, shape (N, 16, 96) float16 -
bereits in 16-Frame-Fenster geschnitten, direkt als Negative nutzbar).

Robust + resumierbar: HuggingFace/Windows-schannel bricht SSL gern ab (curl scheitert mit Exit 35),
deshalb requests (certifi) + Auto-Resume per Range-Header in einer Retry-Schleife."""
import os, time, requests

URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
DST = os.path.join(os.path.dirname(__file__), "..", "data", "neg_ACAV100M_2000hrs.npy")
TOTAL = 17_280_000_128

os.makedirs(os.path.dirname(DST), exist_ok=True)
for attempt in range(100000):
    pos = os.path.getsize(DST) if os.path.exists(DST) else 0
    if pos >= TOTAL:
        print(f"Fertig: {pos} bytes", flush=True)
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
        print(f"Retry {attempt} bei {os.path.getsize(DST) if os.path.exists(DST) else 0}/{TOTAL}: {e}", flush=True)
        time.sleep(3)
