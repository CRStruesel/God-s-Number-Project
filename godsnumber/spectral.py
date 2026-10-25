"""Spectral quantities of the undirected Cayley graph of SL_2(Z/pZ).

For a connected ``k``-regular undirected graph on ``N`` vertices, the adjacency matrix
``A`` is symmetric with eigenvalues ``k = lambda_1 >= lambda_2 >= ... >= lambda_N >= -k``.

* ``lambda(G) = max(|lambda_2|, |lambda_N|)`` controls expansion: the smaller it is, the
  faster random walks mix and the smaller the diameter has to be.
* The graph is *Ramanujan* when ``lambda(G) <= 2 sqrt(k - 1)``  (Lubotzky-Phillips-Sarnak).
  For ``k = 4`` the threshold is ``2 sqrt 3 = 3.4641...``.
* Chung's diameter bound (F. R. K. Chung, "Diameters and eigenvalues", J. AMS 2 (1989)):
      diam(G) <= ceil( log(N - 1) / log(k / lambda(G)) ).

The adjacency matrix below is the one of the *multigraph*: if two generators move a
vertex to the same neighbour, that edge is counted twice, so the matrix is exactly
``A = sum_g P_g`` with ``P_g`` the permutation matrix of right multiplication by ``g``.
With the generating set ``{S, S^-1, T, T^-1}`` and ``p`` odd no such coincidence occurs
and the graph is a simple 4-regular graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .arith import Mat, generators
from .index import decode, encode, group_order, index_dtype, inverse_table


def adjacency(p: int, gens: list[Mat] | None = None) -> sp.csr_matrix:
    """Sparse adjacency matrix of the Cayley (multi)graph with edges ``x -> x g``."""
    gens = gens if gens is not None else generators(p, "undirected")
    N = group_order(p)
    dt = index_dtype(p)
    inv = inverse_table(p).astype(dt)
    idx = np.arange(N, dtype=dt)
    a, b, c, d = decode(idx, p, inv)
    rows, cols = [], []
    for g0, g1, g2, g3 in gens:
        na = (a * g0 + b * g2) % p
        nb = (a * g1 + b * g3) % p
        nc = (c * g0 + d * g2) % p
        nd = (c * g1 + d * g3) % p
        rows.append(idx)
        cols.append(encode(na, nb, nc, nd, p))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(N, N)).tocsr()
    return A


@dataclass
class Spectrum:
    p: int
    N: int
    degree: int
    lambda2: float  # second largest eigenvalue
    lambda_min: float  # smallest eigenvalue
    diameter_bound: int  # Chung's bound

    @property
    def lam(self) -> float:
        return max(abs(self.lambda2), abs(self.lambda_min))

    @property
    def ramanujan_threshold(self) -> float:
        return 2 * math.sqrt(self.degree - 1)

    @property
    def is_ramanujan(self) -> bool:
        return self.lam <= self.ramanujan_threshold + 1e-9


def spectrum(p: int, gens: list[Mat] | None = None, tol: float = 1e-8) -> Spectrum:
    """Second-largest and smallest adjacency eigenvalues (ARPACK) and Chung's bound."""
    A = adjacency(p, gens)
    N = A.shape[0]
    k = int(round(A[0].sum()))
    if N <= 2000:
        w = np.linalg.eigvalsh(A.toarray())
        lam2, lam_min = float(w[-2]), float(w[0])
    else:
        top = spla.eigsh(A, k=2, which="LA", tol=tol, return_eigenvectors=False)
        bot = spla.eigsh(A, k=1, which="SA", tol=tol, return_eigenvectors=False)
        top = np.sort(top)
        lam2, lam_min = float(top[0]), float(bot[0])
    lam = max(abs(lam2), abs(lam_min))
    bound = math.ceil(math.log(N - 1) / math.log(k / lam))
    return Spectrum(p, N, k, lam2, lam_min, bound)
