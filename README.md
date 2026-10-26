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
* **Theorem** (`docs/theorems.md`): the spheres of SL₂(ℤ) in these generators have generating functions
  (1+x)(1+2x+4x²+x³)/(1−x−x²) (undirected; |Lₙ| = 4F_{n+2}, golden-ratio growth) and
  (1+x)(1+x²)(1−x+x²)(1+x+x²)/(1−x−x³) (directed). Proof by amalgam normal forms, an explicit geodesic
  word, a finite Lipschitz check and a transfer matrix (`godsnumber/normal_forms.py`, `tests/test_normal_forms.py`).
  Since the small layers of the finite graphs are those of SL₂(ℤ), this gives G(p) ≥ 3 log_φ p − O(1)
  (undirected) and 3 log_ψ p − O(1) (directed), which explain 75–80 % of G(p).
* **Theorem**: the undirected Cayley graphs are not Ramanujan for all large p — λ₂ is asymptotically at least
  the spectral radius of the Cayley graph of SL₂(ℤ), certified ≥ 3.6158 > 2√3 (exact closed-walk counts) and
  extrapolated ≈ 3.93.
* The whole spectrum of the finite graphs is computed from the representation theory of SL₂(𝔽_p) (principal
  series and the Weil representation), verified against brute force to 10⁻¹³ (`godsnumber/reps.py`); λ₂ for
  every prime up to 400 stays at ≈ ρ(SL₂(ℤ)) ≈ 3.93.
* A power of T is among the farthest elements for most primes (about three quarters of them in the
  directed convention, a smaller majority in the undirected one); the other maximisers are structured
  (diagonal matrices, Tᵏ·S, powers of U, conjugates of S) but not yet understood.
* The second eigenvalue of the finite graphs sits at the spectral radius of the infinite Cayley graph of
  SL₂(ℤ) (≈ 3.93): no "exceptional eigenvalues" are visible — a graph analogue of Selberg's conjecture.
* With Lubotzky–Phillips–Sarnak (Ramanujan) generators instead of S, T (q = 5, 13, 17; p up to ~1000),
  God's number is ≈ (4/3)·log_q N, matching Sardari's bound; the notebook's §12 explains how modular forms enter.
* The graphs are expanders but, for p ≥ 5, **not** Ramanujan (λ₂ > 2√3), contrary to the proposal's wording.

## Layout

| path | contents |
|---|---|
| `proposal/` | the original LaTeX proposal (January 2026) |
| `docs/theorems.md` | the theorems with complete proofs (and exactly which finite checks they rely on) |
| `godsnumber/` | the Python package: arithmetic, element indexing, vectorised BFS, SL₂(ℤ) growth and normal forms, spectra (sparse and via representations), LPS generators |
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
