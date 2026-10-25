#!/usr/bin/env python3
"""Flatten data/layers_*.jsonl into data/diameters.csv (one row per prime).

    python scripts/make_summary_csv.py

Columns: p, N, G_directed, G_undirected, the counting and SL_2(Z)-growth lower bounds, the
number of elements at maximal distance, the mean word length, and one farthest element per convention.
"""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from godsnumber import counting_lower_bound  # noqa: E402
from godsnumber.integer_growth import growth_lower_bound  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(conv):
    rows = {}
    with open(os.path.join(ROOT, "data", f"layers_{conv}.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["p"]] = r
    return rows


def main() -> None:
    data = {c: load(c) for c in ["directed", "undirected"]}
    primes = sorted(set(data["directed"]) & set(data["undirected"]))
    out = os.path.join(ROOT, "data", "diameters.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "N", "G_directed", "G_undirected",
                    "count_lb_directed", "growth_lb_directed", "count_lb_undirected", "growth_lb_undirected",
                    "n_farthest_directed", "n_farthest_undirected", "mean_directed", "mean_undirected",
                    "farthest_directed", "farthest_undirected"])
        for p in primes:
            row = [p, data["directed"][p]["order"]]
            for c in ["directed", "undirected"]:
                row.append(data[c][p]["diameter"])
            for c in ["directed", "undirected"]:
                row += [counting_lower_bound(p, c), growth_lower_bound(p, c)]
            for c in ["directed", "undirected"]:
                row.append(data[c][p]["layer_sizes"][-1])
            for c in ["directed", "undirected"]:
                L = data[c][p]["layer_sizes"]
                row.append(round(sum(n * s for n, s in enumerate(L)) / sum(L), 3))
            for c in ["directed", "undirected"]:
                row.append(" ".join(map(str, data[c][p]["farthest"])))
            w.writerow(row)
    print(f"wrote {out} with {len(primes)} primes")


if __name__ == "__main__":
    main()
