#!/usr/bin/env python3
"""How many S-letters do the farthest elements need?  (direction: describe the farthest elements)

For every element gamma we compute  m(gamma) = the minimum number of letters S^{±1} over all
geodesic words for gamma  (dynamic programming along the BFS layers: a geodesic for gamma ends with
a generator x such that ell(gamma x^-1) = ell(gamma) - 1, so m(gamma) = min over such x of
m(gamma x^-1) + [x is an S-letter]).  The number of T-letters is then ell - m.

    python scripts/analyze_farthest.py --pmax 250 --out data/farthest_geodesics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from godsnumber import bfs, decode, encode, generators, group_order, inverse, inverse_table  # noqa: E402
from godsnumber.index import index_dtype  # noqa: E402
from compute_table import primes_up_to  # noqa: E402


def min_s_letters(p: int, conv: str):
    r = bfs(p, conv, keep_dist=True)
    dist = r.dist
    N = group_order(p)
    dt = index_dtype(p)
    inv = inverse_table(p).astype(dt)
    gens = generators(p, conv)
    s_like = [g in (tuple(x % p for x in (0, -1, 1, 0)), tuple(x % p for x in (0, 1, -1, 0))) for g in gens]
    ginv = [inverse(g, p) for g in gens]
    m = np.full(N, 127, dtype=np.int8)
    m[0] = 0  # identity has index 0
    order = np.argsort(dist, kind="stable")
    bounds = np.searchsorted(dist[order], np.arange(r.diameter + 2))
    for n in range(1, r.diameter + 1):
        idx = order[bounds[n] : bounds[n + 1]].astype(dt)
        a, b, c, d = decode(idx, p, inv)
        best = np.full(idx.size, 127, dtype=np.int8)
        for (g0, g1, g2, g3), cost in zip(ginv, s_like):
            pa = (a * g0 + b * g2) % p
            pb = (a * g1 + b * g3) % p
            pc = (c * g0 + d * g2) % p
            pd = (c * g1 + d * g3) % p
            pred = encode(pa, pb, pc, pd, p)
            ok = dist[pred] == n - 1
            cand = np.where(ok, m[pred] + (1 if cost else 0), 127).astype(np.int8)
            best = np.minimum(best, cand)
        m[idx] = best
    return r, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=int, default=250)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = []
    for p in primes_up_to(args.pmax):
        if p < 5:
            continue
        for conv in ["directed", "undirected"]:
            t0 = time.perf_counter()
            r, m = min_s_letters(p, conv)
            last = np.nonzero(r.dist == r.diameter)[0]
            ms = m[last].astype(int)
            mode_layer = int(np.argmax(r.layer_sizes))
            typical = m[r.dist == mode_layer].astype(int)
            rec = {"p": p, "conv": conv, "G": r.diameter, "n_far": int(last.size),
                   "min_S_letters_of_farthest": sorted(set(ms.tolist())),
                   "max_S_among_farthest": int(ms.max()),
                   "mean_S_at_mode_layer": round(float(typical.mean()), 2),
                   "mean_S_overall": round(float(m.astype(float).mean()), 2),
                   "seconds": round(time.perf_counter() - t0, 1)}
            results.append(rec)
            print(f"p={p:4d} {conv:10s} G={r.diameter:3d} farthest need S-letters in {rec['min_S_letters_of_farthest']} "
                  f"(typical element at the mode layer: {rec['mean_S_at_mode_layer']:.1f}) {rec['seconds']:.1f}s", flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f)


if __name__ == "__main__":
    main()
