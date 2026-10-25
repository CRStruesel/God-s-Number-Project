#!/usr/bin/env python3
"""Compute exact diameters of SL_2(Z/pZ) for many primes and append them to a JSONL file.

Example
-------
    python scripts/compute_table.py --pmax 500 --convention directed --out data/layers_directed.jsonl

Each output line is a JSON object::

    {"p": 101, "convention": "directed", "diameter": 41, "layer_sizes": [1, 2, 4, ...],
     "farthest": [1, 81, 0, 1], "seconds": 0.23}

Primes already present in the output file are skipped, so the script can be re-run to
extend a table.  Run it from the repository root (or put the repository on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from godsnumber import bfs  # noqa: E402


def primes_up_to(n: int) -> list[int]:
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(n + 1) if sieve[i]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmin", type=int, default=2)
    ap.add_argument("--pmax", type=int, default=100)
    ap.add_argument("--primes", type=int, nargs="*", help="explicit list of primes (overrides --pmin/--pmax)")
    ap.add_argument("--convention", choices=["directed", "undirected"], default="directed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    primes = args.primes if args.primes else [p for p in primes_up_to(args.pmax) if p >= args.pmin]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["p"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "a") as f:
        for p in primes:
            if p in done:
                continue
            r = bfs(p, args.convention)
            rec = {
                "p": p,
                "convention": args.convention,
                "order": r.order,
                "diameter": r.diameter,
                "layer_sizes": r.layer_sizes,
                "farthest": list(r.farthest),
                "seconds": round(r.seconds, 3),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                f"{time.strftime('%H:%M:%S')} p={p:5d} N={r.order:11d} diameter={r.diameter:3d} "
                f"farthest={r.farthest} {r.seconds:8.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
