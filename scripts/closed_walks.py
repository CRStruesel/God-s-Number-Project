#!/usr/bin/env python3
"""Exact closed-walk counts c_{2n} at the identity of Cay(SL_2(Z), {S^±1, T^±1}).

    python scripts/closed_walks.py --nmax 28 --out data/closed_walks_sl2z.json

c_{2n} = sum_x mu_n(x)^2 where mu_n(x) is the number of walks of length n from I to x.
Every c_{2n}^{1/(2n)} is a rigorous lower bound for the spectral radius rho of the graph,
and c_{2n} ~ C rho^{2n} n^{-3/2} allows extrapolation.
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S, Si, T, Ti = (0, -1, 1, 0), (0, 1, -1, 0), (1, 1, 0, 1), (1, -1, 0, 1)
    gens = [S, Si, T, Ti]

    def mul(m, g):
        a, b, c, d = m
        e, f, gg, h = g
        return (a * e + b * gg, a * f + b * h, c * e + d * gg, c * f + d * h)

    mu = {(1, 0, 0, 1): 1}
    c = [1]
    sizes = [1]
    t0 = time.time()
    for n in range(1, args.nmax + 1):
        nxt = {}
        for x, w in mu.items():
            for g in gens:
                y = mul(x, g)
                nxt[y] = nxt.get(y, 0) + w
        mu = nxt
        c.append(sum(w * w for w in mu.values()))
        sizes.append(len(mu))
        print(f"n={n:2d} support={len(mu):>10,} c_2n^(1/2n)={c[-1] ** (1 / (2 * n)):.5f} {time.time() - t0:7.1f}s", flush=True)
        with open(args.out, "w") as f:
            json.dump({"nmax": n, "c_2n": [str(v) for v in c], "support_sizes": sizes}, f)


if __name__ == "__main__":
    main()
