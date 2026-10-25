"""Tools for the diameter ("God's number") of SL_2(Z/pZ) with respect to S and T.

Conventions used throughout the package
---------------------------------------
* ``p`` is an odd prime unless stated otherwise (``p = 2`` is allowed and handled,
  but there ``S = S^{-1}`` and ``T = T^{-1}``, so the two conventions below coincide).
* A matrix is the tuple ``(a, b, c, d)`` meaning ``[[a, b], [c, d]]`` with entries
  reduced to ``0 <= entry < p``.
* ``S = [[0, -1], [1, 0]]`` and ``T = [[1, 1], [0, 1]]``.
* A *word* is a product ``g_1 g_2 ... g_n`` of generators, built by multiplying the
  next generator on the **right**.  The length function
  ``ell(gamma) = min{n : gamma is a product of n generators}``
  does not depend on left/right conventions because the set of products of
  length ``n`` is the same either way.
* ``convention="directed"``  : generators ``{S, T}``            (no inverses).
* ``convention="undirected"``: generators ``{S, S^-1, T, T^-1}`` (inverses allowed).
"""

from .arith import (
    S,
    T,
    U,
    det,
    generators,
    identity,
    inverse,
    mat_pow,
    matmul,
    order,
    reduce_mod,
)
from .index import decode, decode_scalar, encode, encode_scalar, group_order, inverse_table
from .bfs import BFSResult, bfs, bfs_reference, counting_lower_bound

__all__ = [
    "S",
    "T",
    "U",
    "det",
    "generators",
    "identity",
    "inverse",
    "mat_pow",
    "matmul",
    "order",
    "reduce_mod",
    "decode",
    "decode_scalar",
    "encode",
    "encode_scalar",
    "group_order",
    "inverse_table",
    "BFSResult",
    "bfs",
    "bfs_reference",
    "counting_lower_bound",
]
