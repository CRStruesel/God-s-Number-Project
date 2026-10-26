#!/usr/bin/env python3
"""lambda_2 and lambda_min of Cay(SL_2(F_p), {S^±1, T^±1}) via the representation theory of SL_2(F_p).

    python scripts/compute_spectrum_reps.py --pmax 400 --out data/spectrum_reps.csv

Also stores, for a few primes, the whole spectrum as a histogram (data/spectrum_hist_p<p>.json)
so that the empirical eigenvalue distribution can be compared with the spectral measure of the
infinite Cayley graph of SL_2(Z).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from godsnumber.reps import adjacency_blocks, full_spectrum  # noqa: E402
from compute_table import primes_up_to  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmin", type=int, default=5)
    ap.add_argument("--pmax", type=int, default=200)
    ap.add_argument("--hist-primes", type=int, nargs="*", default=[101, 199, 307])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {int(r["p"]) for r in csv.DictReader(f)}
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["p", "N", "lambda2", "lambda2_block", "lambda_min", "lambda_min_block", "seconds"])
        for p in primes_up_to(args.pmax):
            if p < args.pmin or p in done:
                continue
            t0 = time.perf_counter()
            best, worst, bl, wl = -np.inf, np.inf, "", ""
            spectra = []
            for A, mult, label in adjacency_blocks(p):
                ev = np.linalg.eigvalsh((A + A.conj().T) / 2)
                if isinstance(mult, tuple):
                    ev = np.delete(ev, int(np.argmin(np.abs(ev - 4))))
                    spectra.append((ev, p))
                else:
                    spectra.append((ev, mult))
                if ev.max() > best:
                    best, bl = float(ev.max()), label
                if ev.min() < worst:
                    worst, wl = float(ev.min()), label
            dt = time.perf_counter() - t0
            w.writerow([p, p**3 - p, f"{best:.8f}", bl, f"{worst:.8f}", wl, f"{dt:.1f}"])
            f.flush()
            if p in args.hist_primes:
                bins = np.linspace(-4, 4, 401)
                hist = np.zeros(400)
                for ev, mult in spectra:
                    h, _ = np.histogram(ev, bins=bins)
                    hist += h * mult
                with open(os.path.join(os.path.dirname(args.out), f"spectrum_hist_p{p}.json"), "w") as g:
                    json.dump({"p": p, "bins": bins.tolist(), "counts": hist.tolist(), "trivial_eigenvalue": 4}, g)
            print(f"{time.strftime('%H:%M:%S')} p={p:4d} lambda2={best:.6f} ({bl})  lambda_min={worst:.6f}  {dt:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
