"""An unambiguous finite automaton for the geodesic words of SL_2(Z) (undirected generators).

By ``normal_forms.length_formula`` every element g of SL_2(Z) has one distinguished geodesic word
w(g) over the alphabet {S, s, T, t} (s = S^-1, t = T^-1).  This module builds a finite automaton
whose accepting paths of length n are in bijection with the elements of word length n:

    #accepting paths of length n  =  |L_n|,   and the values of the paths run over L_n exactly once.

Consequently the *sphere operators*  A_n^pi = sum_{g in L_n} pi(g)  of any representation pi are
read off from powers of one matrix, the twisted transfer operator

    M^pi = sum over transitions (q --a--> q') of E_{q q'} (x) pi(a)      on  C^Q (x) V_pi,

as  A_n^pi = (e_start (x) 1)^T (M^pi)^n (e_accept (x) 1).  Its spectral gap on nontrivial pi is
what controls the diameter of the finite Cayley graphs (docs/transfer_operator.md).

The construction (undirected case).  A normal form is a syllable string alternating S and R/r
(r = R^-1); its reduced lift is obtained by concatenating S -> "S", R -> "ST", r -> "ts" and deleting
adjacent pairs of S-letters from left to right.  Only two letters can ever be deleted later: the "s"
ending an r-lift (deleted if an S-syllable follows) and the "S" of an S-syllable (deleted if R
follows).  Emitting these two letters *lazily* gives the six-state automaton for reduced lifts

    A  --T--> Q    (syllables S R:   "S"+"ST" -> "T")         A  --S--> I_T --T--> Q   (R: "ST")
    A  --S--> I_t  (S, possibly followed by r)                A  --t--> P   (r: "t", pending s)
    Q  --T--> Q    (S R after R: "...T"+"S"+"ST" -> "...TT")  Q  --S--> I_t (S after R; end or r next)
    I_t --t--> P   (r after S: "S"+"ts")                      I_T --T--> Q
    P  --s--> F    (end of word: the pending s is emitted)    P  --S--> I_T --T--> Q  (S cancels the pending s, then R: "ST")
    P  --t--> P    (S cancels the pending s, then r: "t", pending s)

with accepting states A (identity), Q, P (an element "...rS", whose word ends with t), I_t (an element
"...S"), F (an element "...r").  The sign of the element is fixed by flipping the first S-letter, so
every path is doubled at its first S-letter (layer 0 = no S-letter yet, layer 1 = flipped or not);
paths without any S-letter are the words T^k and t^k, which represent one sign only, and the other
sign is represented by S S T^k, S S t^k (a separate 4-state branch).  ``verify`` checks the
bijection; the docstring above is the explanation, the check is the proof.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .normal_forms import length_formula, normal_forms_in_ball

PHI = (1 + 5**0.5) / 2


class Automaton:
    def __init__(self, states, start, accept, transitions):
        self.states = list(states)
        self.start = start
        self.accept = set(accept)
        self.transitions = list(transitions)  # (from, letter, to)
        self.index = {s: i for i, s in enumerate(self.states)}

    def counting_matrix(self):
        n = len(self.states)
        M = np.zeros((n, n), dtype=object)
        for a, _, b in self.transitions:
            M[self.index[a], self.index[b]] += 1
        return M

    def path_counts(self, nmax: int) -> list[int]:
        """Number of accepting paths of length n, n = 0..nmax (exact integers)."""
        n = len(self.states)
        v = np.zeros(n, dtype=object)
        v[self.index[self.start]] = 1
        M = self.counting_matrix()
        acc = np.array([1 if s in self.accept else 0 for s in self.states], dtype=object)
        out = []
        for _ in range(nmax + 1):
            out.append(int(sum(v[i] * acc[i] for i in range(n))))
            v = v.dot(M)
        return out

    def words(self, nmax: int) -> dict[str, int]:
        """Accepted words of length <= nmax, with the number of accepting paths of each."""
        out: dict[str, int] = defaultdict(int)
        frontier = {(self.start, ""): 1}
        for length in range(nmax + 1):
            for (q, w), m in frontier.items():
                if q in self.accept:
                    out[w] += m
            if length == nmax:
                break
            nxt: dict = defaultdict(int)
            for (q, w), m in frontier.items():
                for a, let, b in self.transitions:
                    if a == q:
                        nxt[(b, w + let)] += m
            frontier = nxt
        return dict(out)


def build_geodesic_automaton() -> Automaton:
    # core automaton for the reduced lifts (no sign fix yet)
    core = [("A", "T", "Q"), ("A", "S", "I_T"), ("A", "S", "I_t"), ("A", "t", "P"),
            ("Q", "T", "Q"), ("Q", "S", "I_t"), ("I_t", "t", "P"), ("I_T", "T", "Q"),
            ("P", "s", "F"), ("P", "S", "I_T"), ("P", "t", "P")]
    core_accept = {"A", "Q", "P", "I_t", "F"}
    # layer the first S-letter: layer 0 = no S-letter emitted yet, layer 1 = one has been (flipped or not)
    trans = []
    for a, let, b in core:
        if let in "Tt":
            trans.append(((a, 0), let, (b, 0)))
            trans.append(((a, 1), let, (b, 1)))
        else:
            flipped = "s" if let == "S" else "S"
            trans.append(((a, 0), let, (b, 1)))       # unflipped: one of the two elements ±c_1...c_m
            trans.append(((a, 0), flipped, (b, 1)))   # flipped:   the other
            trans.append(((a, 1), let, (b, 1)))
    # the pure powers of T with the other sign: S S T^k and S S t^k (k >= 0)
    trans += [(("A", 0), "S", "Z1"), ("Z1", "S", "Z2"), ("Z2", "T", "ZT"), ("ZT", "T", "ZT"), ("Z2", "t", "Zt"), ("Zt", "t", "Zt")]
    states = sorted({x for tr in trans for x in (tr[0], tr[2])}, key=str)
    reachable = {("A", 0)}
    changed = True
    while changed:
        changed = False
        for a, _, b in trans:
            if a in reachable and b not in reachable:
                reachable.add(b); changed = True
    states = [s for s in states if s in reachable]
    trans = [t for t in trans if t[0] in reachable]
    accept = {s for s in states if (isinstance(s, tuple) and s[0] in core_accept) or s in ("Z2", "ZT", "Zt")}
    return Automaton(states, ("A", 0), accept, trans)


def verify(nmax_words: int = 14, nmax_counts: int = 40) -> dict:
    """Check (i) accepting-path counts against Theorem 1's sphere sizes, (ii) the accepted words of
    length <= nmax_words against the geodesic words of the ball, one path per word."""
    from .normal_forms import sphere_sizes_from_formula

    aut = build_geodesic_automaton()
    counts = aut.path_counts(nmax_counts)
    formula = sphere_sizes_from_formula(nmax_counts)
    words = aut.words(nmax_words)
    nf = normal_forms_in_ball(nmax_words)
    true_words = {length_formula(sg, sy)[1] for (sg, sy) in nf.values()}
    perron = max(abs(np.linalg.eigvals(aut.counting_matrix().astype(float))))
    return {
        "states": len(aut.states),
        "counts_match_theorem_1": counts == formula,
        "words_match_ball": set(words) == true_words,
        "unambiguous": all(m == 1 for m in words.values()),
        "perron_root": float(perron),
        "phi": PHI,
    }
