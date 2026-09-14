"""Stage 0 of the DVC pipeline -- download a byte-range prefix of PMC-Patients.

The full CSV is 545MB on the Hub; we only need the first few thousand rows for
a bounded eval cut, so we pull a fixed-size prefix via an HTTP Range request
instead of the whole file. Scripted (not a one-off curl) so `dvc repro` can
reproduce it and DVC can track `pmc_head.csv` as a versioned pipeline output.

Usage:
  python eval/download_source.py --bytes 3500000 --out eval/data/pmc_head.csv
"""
from __future__ import annotations

import argparse
import os

import requests

URL = "https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=3_500_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "pmc_head.csv"))
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    headers = {"Range": f"bytes=0-{args.bytes - 1}"}
    r = requests.get(args.url, headers=headers, timeout=60)
    r.raise_for_status()
    with open(args.out, "wb") as f:
        f.write(r.content)
    print(f"wrote {len(r.content)} bytes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
