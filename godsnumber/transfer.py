"""Twisted transfer operators of the geodesic automaton, one per irreducible representation of SL_2(F_p).

For an irreducible pi of G_p = SL_2(F_p) with generator matrices pi(S), pi(s), pi(T), pi(t)
(s = S^-1, t = T^-1), the transfer operator of the automaton with state set Q is

    M^pi = sum_{(q, a, q') in transitions} E_{q q'} (x) pi(a)      on C^Q (x) V_pi,

and the sphere operators are read off as compressions of its powers:

    A_n^pi := sum_{g in L_n} pi(g) = (e_start (x) 1)^T (M^pi)^n (e_acc (x) 1).

On the trivial representation M is the counting matrix of the automaton (Perron root phi) and
A_n = |L_n|.  Two different growth rates matter and must not be confused:

* rho(M^pi), the spectral radius of the transfer matrix;
* the *visible* growth rate  rho_vis(pi) := limsup ||A_n^pi||^{1/n},  which is the inverse of the
  smallest pole of the rational matrix function  A^pi(z) = sum_n A_n^pi z^n
  = (e_start (x) 1)^T (1 - z M^pi)^{-1} (e_acc (x) 1).  Always rho_vis(pi) <= rho(M^pi), and the
  inequality can be strict (eigenvectors of M^pi invisible from the start/accept vectors): for p = 31
  the block with the largest rho(M^pi) = 1.36 has bounded ||A_n^pi||.

The diameter bound of docs/transfer_operator.md needs  ||A_n^pi|| <= C phi^{beta n}  for all nontrivial
pi, i.e. a uniform bound on the visible rates (Hypothesis E(beta)); ``visible_rates`` measures them.

Bookkeeping of the irreducibles (see reps.py): principal series Ind(chi_k) for 0 < 2k < p-1
(dimension p+1), Ind(chi_0) = trivial + Steinberg, Ind(quadratic) = two of dimension (p+1)/2, the
Weil blocks V_theta_k for 0 < 2k < p+1 (discrete series, dimension p-1) and V_theta_{(p+1)/2} = two of
dimension (p-1)/2.  Plancherel:  N sum_gamma c_n(gamma)^2 = sum over irreducibles of dim(pi) ||A_n^pi||_HS^2,
which ``plancherel_check`` tests against the exact lift counts of the spheres of SL_2(Z).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .automaton import build_geodesic_automaton
from .reps import calibrate_weil, principal_series_matrices, quadratic_space, weil_isotypic_matrices

PHI = (1 + 5**0.5) / 2


@dataclass
class Block:
    label: str
    dim: int
    weight: int  # multiplicity of the block's irreducible constituents in the regular representation
    mats: dict  # letter -> matrix


def representation_blocks(p: int, gamma_c=None) -> list[Block]:
    """Generator matrices (letters S, s, T, t) for every irreducible of SL_2(F_p), grouped as in reps.adjacency_blocks."""
    gens = [tuple(x % p for x in g) for g in ((0, -1, 1, 0), (0, 1, -1, 0), (1, 1, 0, 1), (1, -1, 0, 1))]  # S, s, T, t
    blocks = []
    for k in range(0, (p - 1) // 2 + 1):
        mats = principal_series_matrices(p, k, gens)
        d = {"S": mats[0], "s": mats[1], "T": mats[2], "t": mats[3]}
        if k == 0:
            blocks.append(Block("Ind(1) = trivial + Steinberg", p + 1, p, d))
        elif 2 * k == p - 1:
            blocks.append(Block("Ind(quadratic)", p + 1, (p + 1) // 2, d))
        else:
            blocks.append(Block(f"principal series chi_{k}", p + 1, p + 1, d))
    Q = quadratic_space(p)
    gamma, c = gamma_c if gamma_c is not None else calibrate_weil(p)
    wm = weil_isotypic_matrices(Q, gamma, c)
    for k in range(1, (p + 1) // 2 + 1):
        Sk, Tk = wm[k]
        d = {"S": Sk, "s": Sk.conj().T, "T": Tk, "t": Tk.conj().T}
        if 2 * k == p + 1:
            blocks.append(Block("Weil, theta of order 2", p - 1, (p - 1) // 2, d))
        else:
            blocks.append(Block(f"discrete series theta_{k}", p - 1, p - 1, d))
    return blocks


def transfer_matrix(block: Block, aut=None) -> np.ndarray:
    aut = aut or build_geodesic_automaton()
    q, d = len(aut.states), block.dim
    M = np.zeros((q * d, q * d), dtype=complex)
    for a, letter, b in aut.transitions:
        i, j = aut.index[a], aut.index[b]
        M[i * d:(i + 1) * d, j * d:(j + 1) * d] += block.mats[letter]
    return M


def sphere_operators(block: Block, nmax: int, aut=None) -> list[np.ndarray]:
    """A_n^pi for n = 0..nmax, as compressions of the powers of the transfer matrix."""
    aut = aut or build_geodesic_automaton()
    q, d = len(aut.states), block.dim
    M = transfer_matrix(block, aut)
    acc = np.zeros(q)
    acc[[aut.index[s] for s in aut.accept]] = 1
    acc_op = np.kron(acc[:, None], np.eye(d))  # (q d) x d
    X = np.zeros((d, q * d), dtype=complex)
    i0 = aut.index[aut.start]
    X[:, i0 * d:(i0 + 1) * d] = np.eye(d)
    out = []
    for _ in range(nmax + 1):
        out.append(X @ acc_op)
        X = X @ M
    return out


def steinberg_part(A: np.ndarray) -> np.ndarray:
    """Remove the trivial constituent (the constant function) from an Ind(1) sphere operator."""
    d = A.shape[0]
    ones = np.ones(d) / math.sqrt(d)
    return A - float(np.real(ones @ A @ ones)) * np.outer(ones, ones)


def visible_rates(p: int, nmax: int = 48, n0: int = 24, aut=None) -> list[dict]:
    """Growth rate of ||A_n^pi||_op over n0 <= n <= nmax for every nontrivial block (Steinberg separated)."""
    aut = aut or build_geodesic_automaton()
    out = []
    for b in representation_blocks(p):
        ops = sphere_operators(b, nmax, aut)
        if b.label.startswith("Ind(1)"):
            ops = [steinberg_part(A) for A in ops]
            label = "Steinberg"
        else:
            label = b.label
        norms = np.array([np.linalg.norm(A, 2) for A in ops])
        rate = math.exp(np.polyfit(range(n0, nmax + 1), np.log(norms[n0:nmax + 1] + 1e-300), 1)[0])
        out.append({"p": p, "block": label, "dim": b.dim, "rate": rate, "beta": math.log(rate) / math.log(PHI),
                    "norm_at_nmax": float(norms[nmax])})
    return out


def spectral_radii(p: int, aut=None) -> list[dict]:
    """rho(M^pi) for every block (an upper bound for the visible rate; can be strictly larger)."""
    aut = aut or build_geodesic_automaton()
    out = []
    for b in representation_blocks(p):
        ev = np.linalg.eigvals(transfer_matrix(b, aut))
        out.append({"p": p, "block": b.label, "dim": b.dim, "rho": float(np.max(np.abs(ev)))})
    return out


def plancherel_check(p: int, nmax: int = 12) -> float:
    """max relative deviation of  sum_pi dim(pi) ||A_n^pi||_HS^2  from  N sum_gamma c_n(gamma)^2."""
    from .index import encode_scalar, group_order
    from .normal_forms import imul

    aut = build_geodesic_automaton()
    N = group_order(p)
    gens = [(0, -1, 1, 0), (0, 1, -1, 0), (1, 1, 0, 1), (1, -1, 0, 1)]
    seen = {(1, 0, 0, 1)}
    layer = [(1, 0, 0, 1)]
    lhs = [float(N)]
    for n in range(1, nmax + 1):
        nxt = []
        for m in layer:
            for g in gens:
                y = imul(m, g)
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        layer = nxt
        c = np.zeros(N)
        for (a, b, cc, d) in layer:
            c[encode_scalar((a % p, b % p, cc % p, d % p), p)] += 1
        lhs.append(float(N * np.sum(c * c)))
    rhs = np.zeros(nmax + 1)
    for b in representation_blocks(p):
        ops = sphere_operators(b, nmax, aut)
        for n, A in enumerate(ops):
            hs = float(np.sum(np.abs(A) ** 2))
            if b.label.startswith("Ind(1)"):
                ones = np.ones(p + 1) / math.sqrt(p + 1)
                Ln = float(np.real(ones @ A @ ones))
                rhs[n] += Ln**2 + p * (hs - Ln**2)
            else:
                rhs[n] += b.weight * hs
    return float(np.max(np.abs(rhs - np.array(lhs)) / np.array(lhs)))


def free_monoid_rates(p: int) -> list[dict]:
    """The {T, U} instance: spectral radius of pi(T) + pi(U) on each nontrivial irreducible (the sphere
    operators of the free monoid are its powers, so here the visible rate is the spectral radius)."""
    T, U = (1, 1, 0, 1), (1, 0, 1, 1)
    out = []
    for k in range(0, (p - 1) // 2 + 1):
        mT, mU = principal_series_matrices(p, k, [T, U])
        A = mT + mU
        if k == 0:
            ones = np.ones(p + 1) / math.sqrt(p + 1)
            A = A - 2 * np.outer(ones, ones)
            label = "Steinberg"
        else:
            label = "Ind(quadratic)" if 2 * k == p - 1 else f"principal series chi_{k}"
        out.append({"p": p, "block": label, "rho": float(np.max(np.abs(np.linalg.eigvals(A))))})
    Q = quadratic_space(p)
    g, c = calibrate_weil(p)
    wm = weil_isotypic_matrices(Q, g, c)
    for k in range(1, (p + 1) // 2 + 1):
        Sk, Tk = wm[k]
        Uk = Sk @ Tk.conj().T @ Sk.conj().T  # U = S T^-1 S^-1
        label = "Weil, theta of order 2" if 2 * k == p + 1 else f"discrete series theta_{k}"
        out.append({"p": p, "block": label, "rho": float(np.max(np.abs(np.linalg.eigvals(Tk + Uk))))})
    return out
