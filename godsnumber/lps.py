"""Lubotzky-Phillips-Sarnak (Ramanujan) generating sets inside SL_2(F_p).

For a prime ``q`` with ``q ≡ 1 (mod 4)`` there are exactly ``q + 1`` integer quaternions
``alpha = a0 + a1 i + a2 j + a3 k`` of norm ``a0^2 + a1^2 + a2^2 + a3^2 = q`` with ``a0 > 0`` odd and
``a1, a2, a3`` even.  For ``q = 5`` they are ``1 ± 2i, 1 ± 2j, 1 ± 2k``.  For a prime ``p ≡ 1 (mod 4)``
pick ``x, y`` in ``F_p`` with ``x^2 + y^2 + 1 = 0``; then

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

import numpy as np

from .arith import Mat, det, inverse
from .index import decode, encode, group_order, inverse_table


def _sqrt_mod(a: int, p: int) -> int | None:
    a %= p
    for t in range(p):
        if t * t % p == a:
            return t
    return None


def lps_generators(p: int, q: int = 5) -> list[Mat]:
    """The ``q + 1`` LPS generators as elements of ``SL_2(F_p)``.

    Requires ``p ≡ 1 (mod 4)`` (so that ``x^2 + y^2 + 1 = 0`` is solvable with the recipe below;
    in fact it is solvable for every odd ``p``) and ``q`` a nonzero square modulo ``p``.
    Only ``q = 5`` is implemented explicitly.
    """
    if q != 5:
        raise NotImplementedError("only q = 5 is implemented")
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
    quats = []
    for sign in (1, -1):
        quats += [(1, 2 * sign, 0, 0), (1, 0, 2 * sign, 0), (1, 0, 0, 2 * sign)]
    gens: list[Mat] = []
    for a0, a1, a2, a3 in quats:
        m = (
            (a0 + a1 * x + a3 * y) % p,
            (-a1 * y + a2 + a3 * x) % p,
            (-a1 * y - a2 + a3 * x) % p,
            (a0 - a1 * x - a3 * y) % p,
        )
        assert det(m, p) == q % p
        gens.append(tuple(v * si % p for v in m))
    assert all(inverse(g, p) in gens for g in gens), "the LPS set must be closed under inverses"
    return gens


def psl_distances(dist: np.ndarray, p: int) -> np.ndarray:
    """Given ``dist[idx] = ell(gamma)`` on ``SL_2(F_p)``, return ``min(ell(gamma), ell(-gamma))``,
    which is the word length of the image of ``gamma`` in ``PSL_2(F_p)`` (same generators)."""
    idx = np.arange(group_order(p))
    a, b, c, d = decode(idx, p, inverse_table(p))
    neg = encode((-a) % p, (-b) % p, (-c) % p, (-d) % p, p)
    return np.minimum(dist, dist[neg])
