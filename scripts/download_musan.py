"""Download MUSAN (music + speech + noise, ~109 h, 11 GB) — the commercially clean bulk-negative
source. MUSAN is a mix of US-public-domain and Creative Commons material, assembled for exactly
this purpose (Snyder et al., 2015, openslr.org/17); usable in commercial training, unlike the
CC-BY-NC ACAV100M feature set.

requests + Range auto-resume, same recipe as download_negatives.py (curl fails on some nets).

    python scripts/download_musan.py            # -> data/musan.tar.gz
"""
import os
import sys
import time

import requests

URL = "https://www.openslr.org/resources/17/musan.tar.gz"
DST = os.path.join(os.path.dirname(__file__), "..", "data", "musan.tar.gz")


def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    total = int(requests.head(URL, timeout=30, allow_redirects=True)
                .headers.get("Content-Length", 0))
    while True:
        have = os.path.getsize(DST) if os.path.exists(DST) else 0
        if total and have >= total:
            print(f"\ncomplete: {DST} ({have/1e9:.1f} GB)")
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=(30, 90)) as r:
                r.raise_for_status()
                mode = "ab" if have else "wb"
                t0, done0 = time.time(), have
                with open(DST, mode) as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        have += len(chunk)
                        if time.time() - t0 > 15:
                            rate = (have - done0) / (time.time() - t0) / 1e6
                            print(f"  {have/1e9:5.1f} / {total/1e9:.1f} GB  ({rate:.0f} MB/s)",
                                  flush=True)
                            t0, done0 = time.time(), have
        except Exception as e:
            print(f"  interrupted ({str(e)[:60]}) — resuming in 5 s", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
