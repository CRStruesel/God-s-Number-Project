#!/usr/bin/env python3
"""God's number for the Lubotzky-Phillips-Sarnak generating sets, folded to PSL_2(F_p).

    python scripts/compute_lps.py --q 5 --pmax 1009 --out data/lps_q5.jsonl

One JSON line per prime p ≡ 1 (mod 4) with q a square mod p:
    {"p": 29, "q": 5, "N_psl": 12180, "G_sl2": 9, "G_psl": 8, "layer_sizes_psl": [...], "seconds": 0.1}
``layer_sizes_psl[n]`` is the number of elements of PSL_2(F_p) at distance n (the SL_2 count halved).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from godsnumber import bfs, group_order  # noqa: E402
from godsnumber.lps import lps_generators, psl_distances  # noqa: E402
from compute_table import primes_up_to  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--pmin", type=int, default=5)
    ap.add_argument("--pmax", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    q = args.q
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["p"] for line in f if line.strip()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        for p in primes_up_to(args.pmax):
            if p < args.pmin or p == q or p % 4 != 1 or pow(q, (p - 1) // 2, p) != 1 or p in done:
                continue
            t0 = time.perf_counter()
            r = bfs(p, gens=lps_generators(p, q), keep_dist=True)
            dpsl = psl_distances(r.dist, p)
            counts = np.bincount(dpsl.astype(np.int64))
            assert all(c % 2 == 0 for c in counts)
            rec = {"p": p, "q": q, "N_psl": group_order(p) // 2, "G_sl2": r.diameter, "G_psl": int(len(counts) - 1),
                   "layer_sizes_psl": [int(c // 2) for c in counts], "seconds": round(time.perf_counter() - t0, 1)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} q={q} p={p:5d} N_psl={rec['N_psl']:>11,} G_sl2={r.diameter:3d} "
                  f"G_psl={rec['G_psl']:3d} log_q N={np.log(rec['N_psl'])/np.log(q):6.2f} {rec['seconds']:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
