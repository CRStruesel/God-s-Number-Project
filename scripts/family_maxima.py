#!/usr/bin/env python3
"""Is God's number attained on a few explicit one-parameter families?

For each prime and convention: G(p), and the maximum of ell over the families
    T^k,  U^k = [[1,0],[k,1]],  diag(x, 1/x),  T^k S,  -T^k,
together with the number of elements of the last layer that lie in the union of these families.

    python scripts/family_maxima.py --pmax 300 --out data/family_maxima.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from godsnumber import bfs, encode, group_order  # noqa: E402
from compute_table import primes_up_to  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmin", type=int, default=5)
    ap.add_argument("--pmax", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = []
    for p in primes_up_to(args.pmax):
        if p < args.pmin:
            continue
        k = np.arange(p)
        kk = np.arange(1, p)
        one, zero = np.ones(p, dtype=int), np.zeros(p, dtype=int)
        inv = np.array([pow(int(x), p - 2, p) for x in kk])
        fam = {
            "T^k": encode(one, k, zero, one, p),
            "-T^k": encode((p - 1) * one, (-k) % p, zero, (p - 1) * one, p),
            "U^k": encode(one, zero, k, one, p),
            "diag": encode(kk, np.zeros(p - 1, dtype=int), np.zeros(p - 1, dtype=int), inv, p),
            "T^k S": encode(k, (p - 1) * one, one, zero, p),
        }
        for conv in ["directed", "undirected"]:
            t0 = time.perf_counter()
            r = bfs(p, conv, keep_dist=True)
            last = set(np.nonzero(r.dist == r.diameter)[0].tolist())
            rec = {"p": p, "conv": conv, "G": r.diameter, "n_far": len(last)}
            union = set()
            for name, idx in fam.items():
                rec[f"max {name}"] = int(r.dist[idx].max())
                union |= set(idx.tolist())
            rec["farthest_in_union_of_families"] = len(last & union)
            rec["max_over_families"] = max(rec[f"max {n}"] for n in fam)
            rec["seconds"] = round(time.perf_counter() - t0, 1)
            out.append(rec)
            print(f"p={p:4d} {conv:10s} G={r.diameter:3d} | " + "  ".join(f"{n}:{rec['max ' + n]:3d}" for n in fam)
                  + f" | farthest in union: {rec['farthest_in_union_of_families']}/{len(last)}", flush=True)
            with open(args.out, "w") as f:
                json.dump(out, f)


if __name__ == "__main__":
    main()
