# God's number for SL₂(ℤ/pℤ)

How many multiplications by the generators

    S = [[0, -1], [1, 0]]        T = [[1, 1], [0, 1]]

are needed to reach every element of SL₂(𝔽_p)? The largest such number, the diameter of the
Cayley graph, is "God's number" G(p). It is known that G(p) = Θ(log p); the constant is not.
This repository computes G(p) exactly and studies the structure behind it.

**Start with the notebook: [`notebooks/gods_number.ipynb`](notebooks/gods_number.ipynb).** It contains the
background (with proofs), the algorithm, every result, and the open questions. The notebook is
generated and executed by `notebooks/build_notebook.py`, so its outputs are reproducible.

## Main results so far

* Exact God's numbers for every prime p ≤ 500 and for p = 521, 601, 701, 809, 907, 1009, in both
  conventions (words in {S, T} only — "directed" — and words in {S^±1, T^±1} — "undirected"), together
  with the whole distribution of word lengths. Data in `data/`.
* The small layers of the Cayley graph are those of SL₂(ℤ) and satisfy exact recurrences
  (undirected: 4·F_{n+2}, golden-ratio growth; directed: L_n = L_{n−1} + L_{n−3}), which give a lower bound
  that explains most of G(p).
* The farthest elements are (almost always) powers of T or close relatives.
* The graphs are expanders but, for p ≥ 5, **not** Ramanujan (λ₂ > 2√3), contrary to the proposal's wording.

## Layout

| path | contents |
|---|---|
| `proposal/` | the original LaTeX proposal (January 2026) |
| `godsnumber/` | the Python package: arithmetic, element indexing, vectorised BFS, SL₂(ℤ) growth, spectra |
| `scripts/` | batch computations that fill `data/` |
| `data/` | `layers_directed.jsonl`, `layers_undirected.jsonl` (one JSON object per prime), `spectrum.csv` |
| `tests/` | pytest suite (`python -m pytest tests`) |
| `notebooks/` | the project notebook and the script that builds it |

## Quick start

```
pip install -r requirements.txt
python -m pytest tests
python - <<'EOF'
from godsnumber import bfs
r = bfs(101, "undirected")
print(r.diameter, r.layer_sizes, r.farthest)
EOF
```

Conventions (matrices as tuples `(a, b, c, d)`, words built by right multiplication, `p` prime) are
documented in `godsnumber/__init__.py`.
