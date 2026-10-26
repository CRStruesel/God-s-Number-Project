"""Lubotzky-Phillips-Sarnak (Ramanujan) generating sets inside SL_2(F_p).

For a prime ``q ≡ 1 (mod 4)`` there are exactly ``q + 1`` integer quaternions
``alpha = a0 + a1 i + a2 j + a3 k`` of norm ``a0^2 + a1^2 + a2^2 + a3^2 = q`` with ``a0 > 0`` odd and
``a1, a2, a3`` even (a consequence of Jacobi's four-square theorem).  For ``q = 5`` they are
``1 ± 2i, 1 ± 2j, 1 ± 2k``; for ``q = 13`` the fourteen ``3 ± 2i, ...`` and ``1 ± 2i ± 2j ± 2k``.
For a prime ``p ≡ 1 (mod 4)`` pick ``x, y`` in ``F_p`` with ``x^2 + y^2 + 1 = 0``; then

    i -> [[x, y], [y, -x]],   j -> [[0, 1], [-1, 0]],   k = ij

defines a ring homomorphism from the quaternions to ``M_2(F_p)`` (``i^2 = j^2 = -1``, ``ij = -ji``),
and the images of the ``q + 1`` quaternions have determinant ``q``.  When ``q`` is a square mod ``p``
we divide by a square root of ``q`` to land in ``SL_2(F_p)``.  Lubotzky, Phillips and Sarnak proved
that the Cayley graph of ``PSL_2(F_p)`` with respect to these ``q + 1`` elements is a connected
``(q + 1)``-regular *Ramanujan* graph: every nontrivial adjacency eigenvalue has absolute value at
most ``2 sqrt(q)``.  (The proof uses the Ramanujan-Petersson bound for weight-2 modular forms,
i.e. Eichler/Igusa/Deligne, transported to the quaternion algebra by Jacquet-Langlands.)

Our BFS works in ``SL_2(F_p)``, whose Cayley graph is a double cover of the LPS graph on
``PSL_2(F_p) = SL_2(F_p)/{±I}``; :func:`psl_distances` folds the two sheets together.
"""

from __future__ import annotations

import itertools

import numpy as np

from .arith import Mat, det, inverse
from .index import decode, encode, group_order, index_dtype, inverse_table


def _sqrt_mod(a: int, p: int) -> int | None:
    a %= p
    for t in range(p):
        if t * t % p == a:
            return t
    return None


def norm_q_quaternions(q: int) -> list[tuple[int, int, int, int]]:
    """The ``q + 1`` quaternions of norm ``q`` with ``a0 > 0`` odd and ``a1, a2, a3`` even (``q ≡ 1 mod 4``)."""
    if q % 4 != 1:
        raise ValueError("q must be ≡ 1 (mod 4)")
    r = int(q**0.5) + 1
    out = []
    for a0 in range(1, r + 1, 2):
        for a1, a2, a3 in itertools.product(range(-r, r + 1), repeat=3):
            if a1 % 2 == 0 and a2 % 2 == 0 and a3 % 2 == 0 and a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 == q:
                out.append((a0, a1, a2, a3))
    assert len(out) == q + 1, (q, len(out))
    return out


def lps_generators(p: int, q: int = 5) -> list[Mat]:
    """The ``q + 1`` LPS generators as elements of ``SL_2(F_p)``.

    Requires ``p ≡ 1 (mod 4)``, ``q ≡ 1 (mod 4)`` prime, ``p ≠ q``, and ``q`` a nonzero square modulo ``p``
    (otherwise the LPS graph lives in ``PGL_2`` and is bipartite).
    """
    if p % 4 != 1:
        raise ValueError("p must be ≡ 1 (mod 4)")
    if pow(q, (p - 1) // 2, p) != 1:
        raise ValueError(f"{q} is not a square modulo {p}; the LPS graph then lives in PGL_2, not PSL_2")
    x = y = None
    for xx in range(p):
        yy = _sqrt_mod(-1 - xx * xx, p)
        if yy is not None:
            x, y = xx, yy
            break
    s = _sqrt_mod(q, p)
    si = pow(s, p - 2, p)
    gens: list[Mat] = []
    for a0, a1, a2, a3 in norm_q_quaternions(q):
        m = (
            (a0 + a1 * x + a3 * y) % p,
            (-a1 * y + a2 + a3 * x) % p,
            (-a1 * y - a2 + a3 * x) % p,
            (a0 - a1 * x - a3 * y) % p,
        )
        assert det(m, p) == q % p
        gens.append(tuple(v * si % p for v in m))
    assert len(set(gens)) == q + 1
    assert all(inverse(g, p) in gens for g in gens), "the LPS set must be closed under inverses"
    return gens


def psl_distances(dist: np.ndarray, p: int, chunk: int = 1 << 24) -> np.ndarray:
    """Given ``dist[idx] = ell(gamma)`` on ``SL_2(F_p)``, return ``min(ell(gamma), ell(-gamma))``,
    which is the word length of the image of ``gamma`` in ``PSL_2(F_p)`` (same generators).
    Processed in chunks so that ``p ≈ 1000`` (10^9 elements) fits in memory."""
    N = group_order(p)
    dt = index_dtype(p)
    inv = inverse_table(p).astype(dt)
    out = np.empty_like(dist)
    for s in range(0, N, chunk):
        idx = np.arange(s, min(N, s + chunk), dtype=dt)
        a, b, c, d = decode(idx, p, inv)
        neg = encode((-a) % p, (-b) % p, (-c) % p, (-d) % p, p)
        out[s : s + chunk] = np.minimum(dist[idx], dist[neg])
    return out
