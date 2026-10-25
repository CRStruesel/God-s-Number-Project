"""Scalar arithmetic in SL_2(Z/pZ).

Matrices are 4-tuples ``(a, b, c, d)`` standing for ``[[a, b], [c, d]]``.  These
functions are deliberately simple: they are the reference implementation that the
fast vectorised code in :mod:`godsnumber.bfs` is tested against.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

Mat = Tuple[int, int, int, int]

# The two standard generators of SL_2(Z), read modulo p.
S: Mat = (0, -1, 1, 0)  # [[0, -1], [1, 0]]   S^2 = -I, S^4 = I
T: Mat = (1, 1, 0, 1)  # [[1,  1], [0, 1]]   T^k = [[1, k], [0, 1]], T^p = I mod p
# The lower-triangular partner of T; S T^{-1} S^{-1} = U.
U: Mat = (1, 0, 1, 1)  # [[1,  0], [1, 1]]


def reduce_mod(m: Sequence[int], p: int) -> Mat:
    """Reduce every entry of ``m`` into ``0 <= entry < p``."""
    a, b, c, d = m
    return (a % p, b % p, c % p, d % p)


def identity(p: int) -> Mat:
    return (1 % p, 0, 0, 1 % p)


def matmul(m: Sequence[int], n: Sequence[int], p: int) -> Mat:
    """Return ``m @ n`` modulo ``p``."""
    a, b, c, d = m
    e, f, g, h = n
    return ((a * e + b * g) % p, (a * f + b * h) % p, (c * e + d * g) % p, (c * f + d * h) % p)


def det(m: Sequence[int], p: int) -> int:
    a, b, c, d = m
    return (a * d - b * c) % p


def inverse(m: Sequence[int], p: int) -> Mat:
    """Inverse of a matrix of determinant 1: ``[[d, -b], [-c, a]]``."""
    if det(m, p) != 1 % p:
        raise ValueError(f"{m} does not have determinant 1 modulo {p}")
    a, b, c, d = m
    return (d % p, (-b) % p, (-c) % p, a % p)


def mat_pow(m: Sequence[int], k: int, p: int) -> Mat:
    """``m ** k`` modulo ``p`` for an integer ``k`` (negative ``k`` uses the inverse)."""
    if k < 0:
        m, k = inverse(m, p), -k
    result = identity(p)
    base = reduce_mod(m, p)
    while k:
        if k & 1:
            result = matmul(result, base, p)
        base = matmul(base, base, p)
        k >>= 1
    return result


def order(m: Sequence[int], p: int, limit: int | None = None) -> int:
    """Multiplicative order of ``m`` in SL_2(Z/pZ) (smallest ``k >= 1`` with ``m^k = I``).

    The order always divides ``|SL_2(Z/pZ)| = p(p^2 - 1)``, so the search is finite.
    """
    limit = limit or p * (p * p - 1)
    e = identity(p)
    x = reduce_mod(m, p)
    for k in range(1, limit + 1):
        if x == e:
            return k
        x = matmul(x, m, p)
    raise ValueError("order not found below the limit")


def generators(p: int, convention: str = "directed", base: Iterable[Sequence[int]] = (S, T)) -> list[Mat]:
    """The generating set for one of the two conventions, reduced modulo ``p``.

    ``convention="directed"``   -> the matrices in ``base`` (default ``S``, ``T``).
    ``convention="undirected"`` -> ``base`` together with the inverses, duplicates removed
    (for ``p = 2`` the inverses coincide with the generators and nothing new is added).
    """
    gens = [reduce_mod(g, p) for g in base]
    if convention == "directed":
        return gens
    if convention == "undirected":
        out: list[Mat] = []
        for g in gens + [inverse(g, p) for g in gens]:
            if g not in out:
                out.append(g)
        return out
    raise ValueError("convention must be 'directed' or 'undirected'")
