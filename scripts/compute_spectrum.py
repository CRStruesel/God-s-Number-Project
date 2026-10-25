#!/usr/bin/env python3
"""Second-largest / smallest adjacency eigenvalues of the undirected Cayley graphs, as CSV.

    python scripts/compute_spectrum.py --pmax 101 --out data/spectrum.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from godsnumber.spectral import spectrum  # noqa: E402
from compute_table import primes_up_to  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmin", type=int, default=3)
    ap.add_argument("--pmax", type=int, default=61)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "N", "degree", "lambda2", "lambda_min", "lambda", "ramanujan_threshold", "is_ramanujan", "chung_bound", "seconds"])
        for p in primes_up_to(args.pmax):
            if p < args.pmin:
                continue
            t = time.perf_counter()
            s = spectrum(p)
            dt = time.perf_counter() - t
            w.writerow([p, s.N, s.degree, f"{s.lambda2:.6f}", f"{s.lambda_min:.6f}", f"{s.lam:.6f}", f"{s.ramanujan_threshold:.6f}", s.is_ramanujan, s.diameter_bound, f"{dt:.1f}"])
            f.flush()
            print(f"p={p:4d} N={s.N:8d} lambda2={s.lambda2:.5f} lambda_min={s.lambda_min:.5f} chung={s.diameter_bound} {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
