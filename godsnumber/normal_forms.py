"""Amalgam normal forms in SL_2(Z) and an exact formula for the undirected word length.

Background
----------
Let ``S = [[0,-1],[1,0]]`` and ``R = S T = [[0,-1],[1,1]]``.  Then ``S`` has order 4, ``R`` has
order 6, and ``S^2 = R^3 = -I``.  It is classical that

    SL_2(Z) = <S> *_{Z} <R>,     Z = {I, -I},

the free product of the cyclic groups ``<S>`` and ``<R>`` amalgamated over the common centre.
By the normal form theorem for amalgamated products, every ``g`` in ``SL_2(Z)`` can be written
in exactly one way as

    g = z * c_1 c_2 ... c_m,     z in Z,  m >= 0,

where the *syllables* ``c_i`` alternate between the two sets ``{S}`` and ``{R, R^-1}`` (these
are transversals of ``Z`` in ``<S>`` and ``<R>``: ``<S>/Z = {Z, SZ}`` and ``<R>/Z = {Z, RZ, R^-1 Z}``,
using ``R^2 = -R^-1``).  We encode a syllable by a character: ``'S'``, ``'R'``, ``'r'`` (= ``R^-1``).

Right multiplication by a syllable changes the normal form only at its end (``rewrite``):
``S*S = -I``, ``R*R = -R^-1``, ``R*R^-1 = I``, ``R^-1*R^-1 = -R``.  This is the whole content of
the normal form theorem that we use: it is what makes the length formula *local*.

The length formula (undirected generating set {S, S^-1, T, T^-1})
-------------------------------------------------------------------
Lift each syllable to a word:  ``S -> S``,  ``R -> S T``,  ``R^-1 -> T^-1 S^-1``.  Concatenating
the lifts of the syllables gives a word for ``c_1 ... c_m``.  Runs of consecutive S-letters have
length at most 3; a run of length 2 or 3 contains a pair ``S S``, ``S S^-1``, ``S^-1 S`` or
``S^-1 S^-1`` which equals ``±I`` and can be deleted (2 letters saved, sign tracked).  Finally, if
the sign is wrong and the word contains an S-letter, replace one ``S`` by ``S^-1`` or vice versa
(``S^-1 = -S``): same length, sign fixed.  The only elements whose word contains no S-letter are
the powers ``T^k`` (normal form ``±(SR)^k`` or ``±(R^-1 S)^k``); for ``-T^k`` we append ``S^2``.

``L(g)`` is the length of the resulting word, so ``ell(g) <= L(g)`` trivially.  The theorem
``ell = L`` follows from the inequality ``L(g x) <= L(g) + 1`` for every ``g`` and generator ``x``
(then every word of length n reaches only elements with ``L <= n``).  Because ``L`` is a sum of
costs of windows of three consecutive syllables and ``x`` modifies at most the last two syllables,
that inequality is a finite check (``verify_lipschitz``), plus the explicitly handled ``±T^k``.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

from .arith import Mat

S_MAT: Mat = (0, -1, 1, 0)
R_MAT: Mat = (0, -1, 1, 1)  # S T
RINV_MAT: Mat = (1, 1, -1, 0)
SYL_MAT = {"S": S_MAT, "R": R_MAT, "r": RINV_MAT}


def imul(m: Mat, n: Mat) -> Mat:
    a, b, c, d = m
    e, f, g, h = n
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


# --------------------------------------------------------------------------- normal forms
# A normal form is a pair (sign, syllables): sign in {+1, -1}, syllables a string over 'S','R','r'
# in which 'S' alternates with 'R'/'r'.


def rewrite(sign: int, syl: str, c: str) -> tuple[int, str]:
    """Normal form of ``(sign * syl) * c`` for a syllable ``c`` in {'S', 'R', 'r'}."""
    last = syl[-1] if syl else None
    if c == "S":
        if last == "S":
            return -sign, syl[:-1]  # S S = -I
        return sign, syl + "S"
    if c == "R":
        if last == "R":
            return -sign, syl[:-1] + "r"  # R R = -R^-1
        if last == "r":
            return sign, syl[:-1]  # R^-1 R = I
        return sign, syl + "R"
    if c == "r":
        if last == "r":
            return -sign, syl[:-1] + "R"  # R^-1 R^-1 = -R
        if last == "R":
            return sign, syl[:-1]  # R R^-1 = I
        return sign, syl + "r"
    raise ValueError(c)


def times_generator(sign: int, syl: str, x: str) -> tuple[int, str]:
    """Right-multiply by a generator x in {'S', 's', 'T', 't'} (lower case = inverse).

    S^-1 = -S,  T = S^-1 R = -S R,  T^-1 = R^-1 S.
    """
    if x == "S":
        return rewrite(sign, syl, "S")
    if x == "s":
        sg, sy = rewrite(sign, syl, "S")
        return -sg, sy
    if x == "T":
        sg, sy = rewrite(sign, syl, "S")
        return rewrite(-sg, sy, "R")
    if x == "t":
        sg, sy = rewrite(sign, syl, "r")
        return rewrite(sg, sy, "S")
    raise ValueError(x)


def matrix_of(sign: int, syl: str) -> Mat:
    m: Mat = (1, 0, 0, 1)
    for c in syl:
        m = imul(m, SYL_MAT[c])
    return tuple(sign * v for v in m)


def all_syllable_strings(max_len: int):
    """Every alternating string over {'S','R','r'} of length <= max_len (both starting types)."""
    yield ""
    for m in range(1, max_len + 1):
        for start in ("S", "X"):
            seqs = []
            for i in range(m):
                is_s = (i % 2 == 0) == (start == "S")
                seqs.append(["S"] if is_s else ["R", "r"])
            for combo in itertools.product(*seqs):
                yield "".join(combo)


def normal_forms_in_ball(radius: int) -> dict[Mat, tuple[int, str]]:
    """Normal forms of all elements of the undirected ball of radius ``radius`` in SL_2(Z),
    computed by BFS using ``times_generator`` (and checked against matrix multiplication)."""
    e = ((1, 0, 0, 1), (1, ""))
    seen = {e[0]: e[1]}
    layer = [e]
    for _ in range(radius):
        nxt = []
        for m, (sg, sy) in layer:
            for x in "SsTt":
                nsg, nsy = times_generator(sg, sy, x)
                nm = matrix_of(nsg, nsy)
                if nm not in seen:
                    seen[nm] = (nsg, nsy)
                    nxt.append((nm, (nsg, nsy)))
        layer = nxt
    return seen


# --------------------------------------------------------------------------- the length formula
LIFT = {"S": "S", "R": "ST", "r": "ts"}  # lower case = inverse letter


def lifted_word(syl: str) -> str:
    return "".join(LIFT[c] for c in syl)


def reduce_s_runs(word: str) -> tuple[str, int]:
    """Delete pairs of adjacent S-letters (each pair is ±I).  Returns (word, sign of deleted part)."""
    out: list[str] = []
    sign = 1
    for ch in word:
        if ch in "Ss" and out and out[-1] in "Ss":
            prev = out.pop()
            # S S = -I, s s = -I, S s = s S = I
            if prev == ch:
                sign = -sign
        else:
            out.append(ch)
    return "".join(out), sign


def is_pure_t_power(syl: str) -> bool:
    """Normal forms of T^k (k != 0) are (SR)^k and (rS)^k; the empty form is T^0 = I."""
    if syl == "":
        return True
    if len(syl) % 2:
        return False
    return syl == "SR" * (len(syl) // 2) or syl == "rS" * (len(syl) // 2)


def length_formula(sign: int, syl: str) -> tuple[int, str]:
    """``(L(g), word)``: the explicit word for ``g = sign * c_1 ... c_m`` and its length."""
    word, s = reduce_s_runs(lifted_word(syl))
    # the word represents s * (c_1 ... c_m) = s * sign * g, so it represents g iff s*sign == 1
    if s * sign == 1:
        return len(word), word
    if any(ch in "Ss" for ch in word):  # flip one S-letter: S^-1 = -S
        i = next(i for i, ch in enumerate(word) if ch in "Ss")
        word = word[:i] + ("s" if word[i] == "S" else "S") + word[i + 1 :]
        return len(word), word
    return len(word) + 2, "SS" + word  # pure T power with the wrong sign: -T^k = S^2 T^k


def word_matrix(word: str) -> Mat:
    gens = {"S": S_MAT, "s": (0, 1, -1, 0), "T": (1, 1, 0, 1), "t": (1, -1, 0, 1)}
    m: Mat = (1, 0, 0, 1)
    for ch in word:
        m = imul(m, gens[ch])
    return m


# --------------------------------------------------------------------------- the finite check
def local_cost(syl: str) -> int:
    """L_0(syl): length of the lifted word after S-run reduction (no sign considerations).

    This equals  sum(len(LIFT[c]))  minus  2 * (number of S-runs of length >= 2).  Runs of
    S-letters arise only at syllable boundaries and have length <= 3, so each run is determined
    by at most three consecutive syllables: the quantity is a sum of local window costs.
    """
    word, _ = reduce_s_runs(lifted_word(syl))
    return len(word)


def verify_lipschitz(tail_len: int = 8) -> dict:
    """Machine check of  L(g x) <= L(g) + 1  for all g in SL_2(Z) and x in {S, S^-1, T, T^-1}.

    * Elements with at most ``tail_len`` syllables are checked one by one, with the full L
      (sign included).
    * For longer normal forms, L(gx) - L(g) depends only on the last ``tail_len`` syllables
      (multiplication by a generator rewrites at most two syllables and appends at most two;
      window costs further left are untouched) *unless* g or gx is a pure T-power, whose sign
      term is global.  Pure T-powers and their neighbours are handled in ``verify_pure_family``.
      So we check  L_0(tail x) - L_0(tail) <= 1  for every alternating tail of exactly
      ``tail_len`` syllables, where the rewritten tail is computed by the same rewriting rules
      (a deletion reaching past the start of the tail cannot happen for tail_len >= 2).
    """
    worst_short = -10
    n_short = 0
    for syl in all_syllable_strings(tail_len):
        for sign in (1, -1):
            L, _ = length_formula(sign, syl)
            for x in "SsTt":
                sg, sy = times_generator(sign, syl, x)
                L2, _ = length_formula(sg, sy)
                worst_short = max(worst_short, L2 - L)
                n_short += 1
    worst_tail = -10
    n_tail = 0
    for syl in all_syllable_strings(tail_len):
        if len(syl) != tail_len:
            continue
        L0 = local_cost(syl)
        for x in "SsTt":
            _, sy = times_generator(1, syl, x)
            assert len(sy) >= tail_len - 2
            worst_tail = max(worst_tail, local_cost(sy) - L0)
            n_tail += 1
    return {"max_increase_short_elements": worst_short, "checked_short": n_short,
            "max_increase_tails": worst_tail, "checked_tails": n_tail, "tail_len": tail_len}


def verify_pure_family(kmax: int = 12) -> dict:
    """L on the pure powers ±T^k and on their S-neighbours ±T^k S^±1, for 0 <= k <= kmax.

    Expected: L(T^k) = k, L(-T^k) = k + 2 (k >= 1; L(-I) = 2), L(±T^k S^±1) = k + 1 (k >= 1).
    Since L_0 of the periodic forms (SR)^k, (SR)^k S, (rS)^k, S(rS)^k is affine in k (adding a
    period adds the same window costs), the identities for k <= kmax imply them for all k.
    """
    bad = []
    for k in range(0, kmax + 1):
        for syl, sign0 in ((("SR" * k), (-1) ** k), (("rS" * k), 1)):  # T^k = (-1)^k (SR)^k, T^-k = (rS)^k
            for sign in (1, -1):
                # the element sign * T^{±k} has normal form (sign * sign0, syl)
                L, w = length_formula(sign0 * sign, syl)
                assert word_matrix(w) == matrix_of(sign0 * sign, syl)
                expect = k if sign == 1 else k + 2
                if L != expect:
                    bad.append(("pure", k, syl, sign, L, expect))
                for x in "Ss":
                    sg, sy = times_generator(sign0 * sign, syl, x)
                    Lx, wx = length_formula(sg, sy)
                    assert word_matrix(wx) == matrix_of(sg, sy)
                    if k >= 1 and Lx != k + 1:
                        bad.append(("neighbour", k, syl, sign, x, Lx, k + 1))
    return {"kmax": kmax, "violations": bad}


# --------------------------------------------------------------------------- counting by L
def count_by_L0(nmax: int) -> list[int]:
    """N_0(n) = number of syllable strings (normal forms up to sign) with L_0 = n, by a transfer
    matrix over states = (last two syllables).  Local costs are computed by re-deriving the lifted
    word of the last three syllables, which is legitimate because S-runs have length <= 3 and
    never span more than three syllables."""
    from collections import defaultdict

    # state: the string of the last two syllables (or fewer at the start); value: count by cost
    counts = [0] * (nmax + 1)
    counts[0] = 1  # the empty string
    frontier = {"": 0}  # syllable string (full, since costs are local we may truncate) -> cost ... use full strings up to the window
    # We grow strings syllable by syllable and keep only the last two syllables as the state,
    # together with the cost so far.  cost(syl + c) - cost(syl) depends on the last two syllables.
    # Appending one syllable changes L_0 by -1, 0, +1 or +2, and any two consecutive syllables
    # change it by at least +1 (checked in the tests), so strings with L_0 <= nmax have at most
    # 2 nmax + 1 syllables and a partial cost above nmax + 1 can never come back down to nmax.
    states: dict[tuple[str, int], int] = defaultdict(int)
    states[("", 0)] = 1
    for _ in range(2 * nmax + 2):
        new: dict[tuple[str, int], int] = defaultdict(int)
        for (tail, cost), mult in states.items():
            last = tail[-1] if tail else None
            if last is None:
                nxt = ["S", "R", "r"]
            elif last == "S":
                nxt = ["R", "r"]
            else:
                nxt = ["S"]
            for c in nxt:
                delta = local_cost(tail + c) - local_cost(tail)
                nc = cost + delta
                if nc > nmax + 1:
                    continue
                new[((tail + c)[-2:], nc)] += mult
                if nc <= nmax:
                    counts[nc] += mult
        states = new
        if not states:
            break
    return counts


# --------------------------------------------------------------------------- directed case
LIFT_DIR = {"S": "S", "R": "ST", "r": "SSSTST"}  # R^-1 = -R^2 = S^2 (ST)(ST) = S^3 T S T


def directed_runs(syl: str) -> tuple[list[int], int]:
    """Lift the syllables with ``LIFT_DIR`` and split the word into S-runs and T's.

    Returns ``(seq, flips)`` where ``seq`` lists, in order, the exponent of each S-run reduced
    modulo 2 and ``-1`` for each ``T``; ``flips`` is the number of runs whose exponent mod 4 was 2
    or 3, i.e. the number of factors ``S^2 = -I`` that were deleted (each flips the sign).
    """
    word = "".join(LIFT_DIR[c] for c in syl)
    seq: list[int] = []
    flips = 0
    i = 0
    while i < len(word):
        if word[i] == "S":
            j = i
            while j < len(word) and word[j] == "S":
                j += 1
            e = (j - i) % 4
            if e >= 2:
                flips += 1
            seq.append(e % 2)
            i = j
        else:
            seq.append(-1)
            i += 1
    return seq, flips


def directed_length_formula(sign: int, syl: str) -> tuple[int, str]:
    """Exact formula (proved by the finite check below) for the *directed* length, letters S, T only.

    Lift the syllables (``R -> S T``, ``R^-1 -> S^3 T S T``), delete every ``S^2 = -I`` from the
    S-runs (so each run has exponent 0 or 1), and track the sign: the reduced word represents
    ``(-1)^flips * c_1 ... c_m = (-1)^flips * sign * g``.  If that sign is wrong, prepend ``S^2``
    (two letters).  So  L_dir(g) = (#S-letters after reduction) + (#T-letters) + 2*[sign wrong].
    """
    seq, flips = directed_runs(syl)
    out = "".join("T" if e == -1 else "S" * e for e in seq)
    if (-1) ** flips * sign == -1:
        out = "SS" + out
    return len(out), out


def directed_local_cost(syl: str) -> tuple[int, int]:
    """``(L_0', flips mod 2)`` for a syllable string: the reduced length without the sign term,
    and the parity of the number of deleted ``S^2`` factors.  Both are sums over S-runs, hence
    local (a run spans at most three syllables)."""
    seq, flips = directed_runs(syl)
    return sum(e if e >= 0 else 1 for e in seq), flips % 2


def verify_lipschitz_directed(tail_len: int = 8) -> dict:
    """Machine check of  L_dir(g x) <= L_dir(g) + 1  for x in {S, T} (and the reverse inequality
    for x^-1, i.e. L_dir(g) <= L_dir(g x) + 1 -- both are needed since the metric is directed:
    every word of length n reaches only elements with L_dir <= n requires only the first).

    Short normal forms are checked completely.  For long ones the difference depends on the last
    ``tail_len`` syllables and on the parity of the number of S^2 deletions in the prefix, which is
    carried as an extra bit ``f``; the sign term is 2*[(-1)^(f + flips(tail)) * sign = -1].
    """
    worst_short = -10
    n_short = 0
    for syl in all_syllable_strings(tail_len):
        for sign in (1, -1):
            L, w = directed_length_formula(sign, syl)
            assert word_matrix(w) == matrix_of(sign, syl)
            for x in "ST":
                sg, sy = times_generator(sign, syl, x)
                L2, _ = directed_length_formula(sg, sy)
                worst_short = max(worst_short, L2 - L)
                n_short += 1
    worst_tail = -10
    n_tail = 0
    for syl in all_syllable_strings(tail_len):
        if len(syl) != tail_len:
            continue
        L0, par = directed_local_cost(syl)
        for f in (0, 1):
            for sign in (1, -1):
                L = L0 + 2 * ((-1) ** (f + par) * sign == -1)
                for x in "ST":
                    sg, sy = times_generator(sign, syl, x)
                    assert len(sy) >= tail_len - 2
                    L0b, parb = directed_local_cost(sy)
                    Lb = L0b + 2 * ((-1) ** (f + parb) * sg == -1)
                    worst_tail = max(worst_tail, Lb - L)
                    n_tail += 1
    return {"max_increase_short_elements": worst_short, "checked_short": n_short,
            "max_increase_tails": worst_tail, "checked_tails": n_tail, "tail_len": tail_len}


def count_by_directed_L0(nmax: int) -> list[list[int]]:
    """``N'[par][n]`` = number of syllable strings with reduced length ``n`` and deletion parity ``par``.
    The number of group elements with L_dir = n is then  N'[0][n] + N'[1][n] (right sign)
    + N'[0][n-2] + N'[1][n-2] (wrong sign, +2)  -- because for each string exactly one of the two
    signs is "right".  Transfer matrix over (last two syllables, parity)."""
    from collections import defaultdict

    counts = [[0] * (nmax + 1), [0] * (nmax + 1)]
    counts[0][0] = 1
    states: dict[tuple[str, int, int], int] = defaultdict(int)
    states[("", 0, 0)] = 1
    for _ in range(2 * nmax + 4):
        new: dict[tuple[str, int, int], int] = defaultdict(int)
        for (tail, cost, par), mult in states.items():
            last = tail[-1] if tail else None
            nxt = ["S", "R", "r"] if last is None else (["R", "r"] if last == "S" else ["S"])
            for c in nxt:
                L0a, pa = directed_local_cost(tail)
                L0b, pb = directed_local_cost(tail + c)
                nc = cost + (L0b - L0a)
                npar = (par + pb - pa) % 2
                if nc > nmax + 2:
                    continue
                new[((tail + c)[-2:], nc, npar)] += mult
                if nc <= nmax:
                    counts[npar][nc] += mult
        states = new
        if not states:
            break
    return counts


def directed_sphere_sizes_from_formula(nmax: int) -> list[int]:
    N = count_by_directed_L0(nmax)
    tot = [N[0][n] + N[1][n] for n in range(nmax + 1)]
    return [tot[n] + (tot[n - 2] if n >= 2 else 0) for n in range(nmax + 1)]


def sphere_sizes_from_formula(nmax: int) -> list[int]:
    """|L_n| predicted by the length formula: 2 N_0(n), corrected for the pure T-powers whose
    negative costs two more (|L_n| = 2N_0(n) - #pure(n) + #pure(n-2))."""
    N0 = count_by_L0(nmax)

    def pure(n):  # number of syllable strings that are pure T-power patterns with L_0 = n
        return 1 if n == 0 else 2

    return [2 * N0[n] - pure(n) + (pure(n - 2) if n >= 2 else 0) for n in range(nmax + 1)]
