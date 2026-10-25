"""Growth of SL_2(Z) itself (no reduction mod p) with respect to S and T.

Why this matters for the mod-p problem
--------------------------------------
Reduction modulo ``p`` is a homomorphism ``pi : SL_2(Z) -> SL_2(Z/pZ)`` that sends ``S`` to
``S`` and ``T`` to ``T``.  A product of ``n`` generators modulo ``p`` is the reduction of the
same product in ``SL_2(Z)``, so the ball of radius ``n`` mod ``p`` is the image of the ball of
radius ``n`` in ``SL_2(Z)``:

    B_n(p) = pi( B_n(Z) ),   hence   |B_n(p)| <= |B_n(Z)|.

Consequently the diameter of ``SL_2(Z/pZ)`` is at least the smallest ``n`` with
``|B_n(Z)| >= p^3 - p``.  This is a much better lower bound than the naive word count
``2^{n+1} - 1`` or ``2 * 3^n - 1`` because ``SL_2(Z)`` has many relations
(``S^4 = I``, ``(ST)^3 = S^2``), so its balls grow much more slowly than a free group's.

Moreover, two words of length ``<= n`` with different values in ``SL_2(Z)`` can only
collide mod ``p`` if some entry differs by a nonzero multiple of ``p``.  All entries of a
product of ``n`` generators are bounded by the Fibonacci number ``F_{n+2}`` (see
:func:`entry_bound`), so while ``2 F_{n+2} < p`` the map ``pi`` is injective on ``B_n(Z)``
and the layer sizes mod ``p`` are *exactly* those of ``SL_2(Z)``.  This is why the first
layer sizes in the mod-``p`` tables do not depend on ``p``.

Empirical recurrences (verified by :func:`sl2z_layers`; a proof is still to be written)
---------------------------------------------------------------------------------------
    undirected:  |L_n| = |L_{n-1}| + |L_{n-2}|  for n >= 5,  in fact |L_n| = 4 F_{n+2} for n >= 3
    directed  :  |L_n| = |L_{n-1}| + |L_{n-3}|  for n >= 8
so the growth rates are the golden ratio ``phi = 1.6180...`` and the real root
``psi = 1.4655...`` of ``x^3 = x^2 + 1``.
"""

from __future__ import annotations

from functools import lru_cache

from .arith import generators, identity, matmul

# The layers of SL_2(Z) are enumerated exactly up to these radii (pure Python, a few seconds);
# beyond that the recurrences above are used.
EXACT_RADIUS = {"directed": 30, "undirected": 24}
RECURRENCE = {"directed": ((1, 3), 8), "undirected": ((1, 2), 5)}  # (lags, first n where it holds)


def entry_bound(n: int) -> int:
    """``F_{n+2}``: every entry of a product of ``n`` matrices from ``{S^±1, T^±1}`` has absolute
    value at most ``F_{n+2}`` (``F_1 = F_2 = 1``).

    Proof.  Write ``M = [v w]`` by columns and let ``x = max|v|``, ``y = max|w|`` (sup norms).
    Right-multiplying by ``T^±1`` replaces ``w`` by ``w ± v``, so ``(x, y) -> (x, <= x + y)``;
    right-multiplying by ``S^±1`` swaps the columns up to sign, so ``(x, y) -> (y, x)``.
    Claim: after ``n`` steps the sorted pair is at most ``(F_{n+1}, F_{n+2})`` termwise.
    For ``n = 0`` the pair is ``(1, 1) = (F_1, F_2)``.  If it holds after ``n`` steps, then a
    ``T`` step gives a pair at most ``(F_{n+2}, F_{n+1} + F_{n+2}) = (F_{n+2}, F_{n+3})`` and an
    ``S`` step does not change the sorted pair.  Hence every entry is at most ``F_{n+2}``.
    (The true maxima are far smaller - about ``1.37^n`` - because ``S`` steps cost a letter
    without growing entries; see the notebook.)"""
    a, b = 1, 1  # F_1, F_2
    for _ in range(n):
        a, b = b, a + b
    return b


def sl2z_layers(nmax: int, convention: str = "directed") -> list[int]:
    """Exact sizes ``|L_0|, ..., |L_nmax|`` of the spheres of ``SL_2(Z)`` by brute-force BFS.

    Matrices are stored as integer 4-tuples; ``matmul`` with a huge modulus is just integer
    multiplication (we pass ``p = 0``-free arithmetic by using a modulus larger than any entry).
    """
    # Use an enormous modulus so that "mod p" never triggers: entries stay below F_{nmax+1}.
    BIG = 10 ** (len(str(entry_bound(nmax))) + 2)
    gens = [tuple(x % BIG for x in g) for g in generators(BIG, convention)]
    # generators() reduced S's -1 to BIG-1; convert back to signed arithmetic via a tiny helper
    def mul(m, g):
        a, b, c, d = m
        e, f, gg, h = (x if x < BIG // 2 else x - BIG for x in g)
        return (a * e + b * gg, a * f + b * h, c * e + d * gg, c * f + d * h)

    e = identity(BIG)
    seen = {e}
    layer = [e]
    sizes = [1]
    for _ in range(nmax):
        nxt = []
        for m in layer:
            for g in gens:
                y = mul(m, g)
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        sizes.append(len(nxt))
        layer = nxt
    return sizes


@lru_cache(maxsize=None)
def _exact(convention: str) -> tuple[int, ...]:
    return tuple(sl2z_layers(EXACT_RADIUS[convention], convention))


def layers_extended(nmax: int, convention: str = "directed") -> list[int]:
    """``|L_n(Z)|`` for ``n <= nmax``: exact where enumerated, by the recurrence beyond."""
    sizes = list(_exact(convention))
    lags, _ = RECURRENCE[convention]
    while len(sizes) <= nmax:
        sizes.append(sum(sizes[-k] for k in lags))
    return sizes[: nmax + 1]


def growth_lower_bound(p: int, convention: str = "directed") -> int:
    """Smallest ``n`` with ``|B_n(Z)| >= p^3 - p``; the diameter of ``SL_2(Z/pZ)`` is at least this.

    Exact for ``n <= EXACT_RADIUS``; relies on the empirical recurrence beyond that radius.
    """
    N = p * (p * p - 1)
    total, n = 0, 0
    sizes = layers_extended(200, convention)
    for n, s in enumerate(sizes):
        total += s
        if total >= N:
            return n
    raise ValueError("p too large for the tabulated range")


def injective_radius(p: int) -> int:
    """Largest ``n`` with ``2 F_{n+2} < p``: reduction mod ``p`` is then injective on ``B_n(Z)``.

    Two matrices of ``B_n(Z)`` have entries in ``[-F_{n+2}, F_{n+2}]``; if they are congruent
    mod ``p`` every entry differs by a multiple of ``p`` of size ``<= 2 F_{n+2} < p``, so they
    are equal.  Hence ``|L_m(p)| = |L_m(Z)|`` for all ``m <= n`` (a proven, but crude, radius).
    """
    n = 0
    while 2 * entry_bound(n + 1) < p:
        n += 1
    return n
