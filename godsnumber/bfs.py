"""Breadth-first search on the Cayley graph of SL_2(Z/pZ).

The algorithm
-------------
Let ``B_n`` be the set of elements of word length at most ``n`` (the *ball* of radius
``n``) and ``L_n = B_n \\ B_{n-1}`` the *sphere* or *layer* of elements of length exactly
``n``.  Then ``L_0 = {I}`` and

    L_{n+1} = { x g : x in L_n, g a generator } minus B_n.

So one pass of "multiply every element of the current layer by every generator and
discard what was seen before" produces the next layer.  The diameter is the last ``n``
with ``L_n`` non-empty, and the sizes ``|L_0|, |L_1|, ...`` are the whole distribution of
the length function.

Two implementations are provided:

* :func:`bfs_reference` - plain Python dictionaries, for tiny ``p`` and for testing.
* :func:`bfs` - NumPy, using the index of :mod:`godsnumber.index` so that "seen
  before" is a byte array and a whole layer is processed with array operations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .arith import Mat, generators, identity, matmul
from .index import decode, decode_scalar, encode, group_order, index_dtype, inverse_table


@dataclass
class BFSResult:
    p: int
    convention: str
    gens: list[Mat]
    layer_sizes: list[int]
    farthest: Mat  # one element at maximal distance from the identity
    seconds: float
    dist: np.ndarray | None = field(default=None, repr=False)  # dist[idx] = ell(element idx)
    last_layer: list[Mat] = field(default_factory=list, repr=False)  # all elements at maximal distance (capped)

    @property
    def diameter(self) -> int:
        return len(self.layer_sizes) - 1

    @property
    def order(self) -> int:
        return sum(self.layer_sizes)

    @property
    def mean_length(self) -> float:
        n = np.arange(len(self.layer_sizes))
        return float((n * np.asarray(self.layer_sizes)).sum() / self.order)

    def ball_sizes(self) -> list[int]:
        return list(np.cumsum(self.layer_sizes))


def bfs_reference(p: int, convention: str = "directed", gens: Sequence[Mat] | None = None) -> BFSResult:
    """Dictionary-based BFS; fine for ``p`` up to about 30."""
    t0 = time.perf_counter()
    gens = list(gens) if gens is not None else generators(p, convention)
    e = identity(p)
    dist = {e: 0}
    layer = [e]
    layers = [1]
    while layer:
        nxt = []
        for x in layer:
            for g in gens:
                y = matmul(x, g, p)
                if y not in dist:
                    dist[y] = len(layers)
                    nxt.append(y)
        if nxt:
            layers.append(len(nxt))
        farthest = layer[0]
        layer = nxt
    if len(dist) != group_order(p):
        raise RuntimeError(f"reached {len(dist)} elements, expected {group_order(p)}: not a generating set?")
    last = [m for m, n in dist.items() if n == len(layers) - 1]
    return BFSResult(p, convention, gens, layers, farthest, time.perf_counter() - t0, last_layer=last)


def bfs(
    p: int,
    convention: str = "directed",
    gens: Sequence[Mat] | None = None,
    *,
    keep_dist: bool = False,
    chunk: int = 1 << 21,
    last_layer_cap: int = 10_000,
) -> BFSResult:
    """Vectorised BFS.  Returns the layer sizes (hence the diameter) and the farthest elements.

    ``keep_dist=True`` also returns the full array ``dist`` with ``dist[idx] = ell`` for every
    element, which costs ``p^3 - p`` bytes of memory.
    ``chunk`` bounds how many frontier elements are expanded at once, which bounds memory.
    ``last_layer`` lists every element at maximal distance (at most ``last_layer_cap`` of them).
    """
    t0 = time.perf_counter()
    gens = list(gens) if gens is not None else generators(p, convention)
    N = group_order(p)
    dt = index_dtype(p)
    inv = inverse_table(p).astype(dt)
    g_arr = np.array(gens, dtype=dt) % p  # shape (k, 4)

    dist = np.full(N, -1, dtype=np.int8)
    start = np.array([((1 - 1) * p + 0) * p + 0], dtype=dt)  # index of the identity: a=1,b=0,c=0
    dist[start] = 0
    layer = start
    layers = [1]
    farthest_idx = int(start[0])

    while True:
        level = len(layers)
        if level > 126:
            raise RuntimeError("diameter exceeds the int8 distance storage")
        parts = []
        for s in range(0, len(layer), chunk):
            a, b, c, d = decode(layer[s : s + chunk], p, inv)
            for g0, g1, g2, g3 in g_arr:
                # Right-multiply [[a,b],[c,d]] by [[g0,g1],[g2,g3]].
                na = (a * g0 + b * g2) % p
                nb = (a * g1 + b * g3) % p
                nc = (c * g0 + d * g2) % p
                nd = (c * g1 + d * g3) % p
                idx = encode(na, nb, nc, nd, p)
                idx = idx[dist[idx] < 0]
                if idx.size:
                    idx = np.unique(idx)
                    dist[idx] = level
                    parts.append(idx)
        if not parts:
            break
        farthest_idx = int(parts[0][0])
        layer = np.concatenate(parts) if len(parts) > 1 else parts[0]
        layers.append(int(layer.size))

    if sum(layers) != N:
        raise RuntimeError(f"reached {sum(layers)} elements, expected {N}: not a generating set?")
    last = [decode_scalar(int(i), p) for i in layer[:last_layer_cap]]  # ``layer`` is the last non-empty layer
    return BFSResult(
        p,
        convention,
        gens,
        layers,
        decode_scalar(farthest_idx, p),
        time.perf_counter() - t0,
        dist if keep_dist else None,
        last_layer=last,
    )


def counting_lower_bound(p: int, convention: str = "directed") -> int:
    """The smallest ``n`` for which the trivial word count can reach all ``p^3 - p`` elements.

    directed   (2 generators, no inverses): at most ``2^{n+1} - 1`` products of length <= n.
    undirected (4 generators, p odd):       at most ``2 * 3^n - 1`` reduced words of length <= n
               (a reduced word never follows a letter by its inverse: 4 * 3^{n-1} of length n).
    For p = 2 both conventions have the two involutions S, T and the bound is ``2n + 1``.
    Every element must be a product of length <= diameter, so ``diameter >= `` this value.
    """
    N = group_order(p)
    if p == 2:
        return math.ceil((N - 1) / 2)
    n = 0
    while True:
        ball = 2 ** (n + 1) - 1 if convention == "directed" else 2 * 3**n - 1
        if ball >= N:
            return n
        n += 1
