"""A bijection between SL_2(Z/pZ) and the integers ``0, 1, ..., p^3 - p - 1``.

Why we need this
----------------
A breadth-first search has to remember which group elements it has already seen.
Storing matrices in a Python ``set`` works for small ``p`` but is slow and memory
hungry.  If instead every element has a *unique index* below ``N = |SL_2(Z/pZ)|``,
the "seen" information is a single array of ``N`` bytes and the search can be
vectorised with NumPy.

The bijection
-------------
A matrix ``[[a, b], [c, d]]`` with ``ad - bc = 1`` is determined by three of its
entries once we know whether ``a`` is zero:

* If ``a != 0``: choose ``a`` in ``{1, ..., p-1}`` and ``b, c`` freely in ``{0, ..., p-1}``;
  then ``d = (1 + bc) a^{-1}`` is forced.  That is ``(p-1) p^2`` matrices.
* If ``a == 0``: the determinant is ``-bc = 1``, so ``b != 0`` and ``c = -b^{-1}`` is
  forced, while ``d`` is free.  That is ``(p-1) p`` matrices.

The two counts add up to ``p^3 - p^2 + p^2 - p = p^3 - p = p(p-1)(p+1)``, which is the
order of ``SL_2(Z/pZ)``.  We index the first family first:

    a != 0 :  idx = ((a - 1) p + b) p + c            in  [0, (p-1) p^2)
    a == 0 :  idx = (p-1) p^2 + (b - 1) p + d        in  [(p-1) p^2, p^3 - p)
"""

from __future__ import annotations

import numpy as np

from .arith import Mat


def group_order(p: int) -> int:
    """``|SL_2(Z/pZ)| = p (p^2 - 1)``."""
    return p * (p * p - 1)


def inverse_table(p: int) -> np.ndarray:
    """``inv[x] = x^{-1} mod p`` for ``1 <= x < p`` (and ``inv[0] = 0`` as a harmless filler).

    Uses Fermat's little theorem ``x^{p-2} = x^{-1}`` for prime ``p``; this is the only
    place the primality of ``p`` is used by the indexing code.
    """
    inv = np.zeros(p, dtype=np.int64)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    return inv


def index_dtype(p: int):
    """Smallest signed integer dtype in which all intermediate products fit.

    The largest intermediate is ``p^3`` (from ``((a-1)p + b)p + c`` and from
    ``(1 + bc) * inv[a]``), so int32 is safe exactly when ``p^3 < 2^31``.
    """
    return np.int32 if p**3 < 2**31 else np.int64


def encode_scalar(m: Mat, p: int) -> int:
    a, b, c, d = m
    if a != 0:
        return ((a - 1) * p + b) * p + c
    return (p - 1) * p * p + (b - 1) * p + d


def decode_scalar(idx: int, p: int) -> Mat:
    big = (p - 1) * p * p
    if idx < big:
        a = idx // (p * p) + 1
        r = idx % (p * p)
        b, c = r // p, r % p
        d = (1 + b * c) * pow(a, p - 2, p) % p
        return (a, b, c, d)
    j = idx - big
    b = j // p + 1
    d = j % p
    c = (-pow(b, p - 2, p)) % p
    return (0, b, c, d)


def encode(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, p: int) -> np.ndarray:
    """Vectorised :func:`encode_scalar`; the inputs are integer arrays with entries in ``[0, p)``."""
    big = (p - 1) * p * p
    return np.where(a != 0, ((a - 1) * p + b) * p + c, big + (b - 1) * p + d)


def decode(idx: np.ndarray, p: int, inv: np.ndarray):
    """Vectorised :func:`decode_scalar`.  Returns the four entry arrays ``a, b, c, d``."""
    big = (p - 1) * p * p
    is_big = idx < big
    r = idx % (p * p)
    j = idx - big
    a = np.where(is_big, idx // (p * p) + 1, 0)
    b = np.where(is_big, r // p, j // p + 1)
    # inv[b] is needed only where a == 0; there b lies in [1, p) so the lookup is valid.
    c = np.where(is_big, r % p, (-inv[b]) % p)
    d = np.where(is_big, (1 + b * c) * inv[a] % p, j % p)
    return a, b, c, d
