"""The spectrum of Cay(SL_2(F_p), A) through the representation theory of SL_2(F_p).

Why this works
--------------
Let G = SL_2(F_p) and let A = sum_{g in A} R(g) be the adjacency operator on functions on G,
where R(g)f(x) = f(xg) is the right regular representation.  The regular representation splits
as a direct sum over the irreducible representations pi of G, each appearing dim(pi) times.
Hence the multiset of eigenvalues of the Cayley graph is

    union over pi of  dim(pi)  copies of  spec( sum_{g in A} pi(g) ),

and the (dim pi)-by-(dim pi) matrices sum_g pi(g) are tiny compared with the p^3 - p vertices.

The irreducible representations of SL_2(F_p), p odd (dimensions: 1, p, p+1, (p+1)/2, p-1, (p-1)/2):

* principal series:  Ind_B^G(chi) for a character chi of F_p^*, realised on functions f on
  F_p^2 \ {0} with f(lambda v) = chi(lambda) f(v), G acting by (g f)(v) = f(v g).  It is
  irreducible of dimension p+1 when chi^2 != 1 (and Ind(chi) ~ Ind(chi^-1)); for chi = 1 it is
  trivial + Steinberg; for chi = the quadratic character it splits into two pieces of dimension (p+1)/2.
* discrete (cuspidal) series: inside the Weil representation omega of G on functions on the
  quadratic space F_{p^2} (norm form).  omega(T) is multiplication by psi(N(x)) and omega(S) is a
  Fourier transform for the trace form, up to a constant (the Weil index).  The norm-one group
  U_1 = {z : N(z) = 1} (cyclic of order p+1) commutes with omega, and the theta-isotypic part
  V_theta = {f : f(z x) = theta(z) f(x)} is the cuspidal representation pi_theta (dimension p-1)
  for theta^2 != 1 (with pi_theta ~ pi_{theta^-1}), the sum of the two (p-1)/2-dimensional
  representations for theta of order 2, and V_1 (dimension p) is the Steinberg representation.

Every claim above is *verified numerically* by :func:`check_against_bruteforce`, which assembles
the multiset for small p and compares it with a direct eigen-decomposition of the p^3 - p vertex graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .arith import Mat, generators


# --------------------------------------------------------------------------- finite-field helpers
def primitive_root(p: int) -> int:
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in _prime_factors(p - 1)):
            return g
    raise ValueError


def _prime_factors(n: int) -> set[int]:
    out, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            out.add(d)
            n //= d
        d += 1
    if n > 1:
        out.add(n)
    return out


def nonresidue(p: int) -> int:
    return next(d for d in range(2, p) if pow(d, (p - 1) // 2, p) == p - 1)


# --------------------------------------------------------------------------- principal series
def principal_series_matrices(p: int, k: int, gens: list[Mat]):
    """Matrices of the generators in Ind(chi_k), chi_k(g0^j) = exp(2 pi i k j/(p-1)), on the basis
    of the p+1 lines of F_p^2 (representatives (1, t) and (0, 1))."""
    g0 = primitive_root(p)
    dlog = {}
    x = 1
    for j in range(p - 1):
        dlog[x] = j
        x = x * g0 % p
    chi = np.exp(2j * np.pi * k * np.arange(p - 1) / (p - 1))
    reps = [(1, t) for t in range(p)] + [(0, 1)]
    line_index = {r: i for i, r in enumerate(reps)}
    mats = []
    for a, b, c, d in gens:
        M = np.zeros((p + 1, p + 1), dtype=complex)
        for j, (v1, v2) in enumerate(reps):
            w1, w2 = (v1 * a + v2 * c) % p, (v1 * b + v2 * d) % p
            if w1 != 0:
                mu, r = w1, (1, w2 * pow(w1, p - 2, p) % p)
            else:
                mu, r = w2, (0, 1)
            M[j, line_index[r]] = chi[dlog[mu]]
        mats.append(M)
    return mats


# --------------------------------------------------------------------------- Weil representation
@dataclass
class QuadraticSpace:
    p: int
    d: int  # a non-residue; F_{p^2} = F_p[sqrt d]
    elems: np.ndarray  # (p^2, 2) all elements (u, v) = u + v sqrt d
    norm: np.ndarray  # N(u + v sqrt d) = u^2 - d v^2
    U1: list[tuple[int, int]]  # norm-one elements in cyclic order z0^0, z0^1, ...
    class_rep: dict  # t -> a representative of norm t


def quadratic_space(p: int) -> QuadraticSpace:
    d = nonresidue(p)
    elems = np.array([(u, v) for u in range(p) for v in range(p)])
    norm = (elems[:, 0] ** 2 - d * elems[:, 1] ** 2) % p

    def mul(x, y):
        return ((x[0] * y[0] + d * x[1] * y[1]) % p, (x[0] * y[1] + x[1] * y[0]) % p)

    ones = [tuple(e) for e, n in zip(elems.tolist(), norm) if n == 1]
    assert len(ones) == p + 1
    for z0 in ones:  # find a generator of the cyclic group U_1
        orbit, z = [], (1, 0)
        for _ in range(p + 1):
            orbit.append(z)
            z = mul(z, z0)
        if len(set(orbit)) == p + 1:
            break
    class_rep = {}
    for e, n in zip(elems.tolist(), norm):
        if n != 0 and n not in class_rep:
            class_rep[int(n)] = tuple(e)
    return QuadraticSpace(p, d, elems, norm, orbit, class_rep)


def weil_isotypic_matrices(Q: QuadraticSpace, gamma: complex = 1.0, c: int = 1):
    """For every character theta_k of U_1 (k = 0..p), the matrices of omega(S) and omega(T) on
    V_{theta_k}, in the basis of norm classes (plus the delta at 0 when k = 0).

    omega(T) f(x) = psi(c N(x)) f(x),   omega(S) f(x) = gamma/p * sum_y psi(-c B(x, y)) f(y),
    with B(x, y) = tr(x conj(y)) = 2(u u' - d v v').  (The relative sign between the two
    formulas is forced by the relations S^4 = I, (ST)^3 = S^2; gamma = -1 is the Weil index,
    consistent with the Gauss sum  sum_x psi(N(x)) = -p  of the anisotropic plane.)
    Returns a dict k -> (S_k, T_k).  All k are computed at once with an FFT over U_1.
    """
    p, d = Q.p, Q.d
    psi = lambda t: np.exp(2j * np.pi * (np.asarray(t) % p) / p)  # noqa: E731
    ts = sorted(Q.class_rep)  # the p-1 nonzero norms, in order
    X = np.array([Q.class_rep[t] for t in ts])  # (p-1, 2) representatives x_t
    Z = np.array(Q.U1)  # (p+1, 2) norm-one elements z_j = z0^j
    # y = z * x_{t'} for all j, t':  (p+1, p-1, 2)
    zu, zv = Z[:, 0][:, None], Z[:, 1][:, None]
    xu, xv = X[:, 0][None, :], X[:, 1][None, :]
    Yu = (zu * xu + d * zv * xv) % p
    Yv = (zu * xv + zv * xu) % p
    # B(x_t, y) = 2 (u_t Yu - d v_t Yv)  ->  shape (p-1, p+1, p-1) indexed [t, j, t']
    B = 2 * (X[:, 0][:, None, None] * Yu[None, :, :] - d * X[:, 1][:, None, None] * Yv[None, :, :])
    Psi = psi(-c * B)  # (p-1, p+1, p-1)
    # K_k[t, t'] = gamma/p * sum_j Psi[t, j, t'] * theta_k(z_j),  theta_k(z0^j) = exp(2 pi i k j/(p+1))
    # = gamma/p * (p+1) * inverse-DFT along j  -> use np.fft.ifft (which has the + sign and 1/(p+1))
    K = np.fft.ifft(Psi, axis=1) * (p + 1) * gamma / p  # (p-1, p+1, p-1), axis 1 = k
    Tdiag = psi(c * np.array(ts))
    out = {}
    for k in range(p + 1):
        Sk = K[:, k, :]
        Tk = np.diag(Tdiag)
        if k == 0:  # add e_0 = sqrt(p+1) * delta_0 as an extra basis vector (last coordinate);
            # the scaling makes the basis orthonormal up to a common factor, since ||f_t||^2 = p+1
            Sk = np.pad(Sk, ((0, 1), (0, 1)))
            Sk[:-1, -1] = gamma / p * math.sqrt(p + 1)  # coefficient of f_t in omega(S) e_0
            Sk[-1, :-1] = gamma / p * math.sqrt(p + 1)  # coefficient of e_0 in omega(S) f_t
            Sk[-1, -1] = gamma / p
            Tk = np.pad(Tk, ((0, 1), (0, 1)))
            Tk[-1, -1] = 1.0
        out[k] = (Sk, Tk)
    return out


def calibrate_weil(p: int):
    """Find (gamma, c) such that the Weil matrices satisfy S^4 = I and (S T)^3 = S^2 on every V_theta."""
    Q = quadratic_space(p)
    for c in (1,):
        for gamma in (-1, 1, 1j, -1j):
            mats = weil_isotypic_matrices(Q, gamma, c)
            ok = True
            for k, (Sk, Tk) in mats.items():
                n = Sk.shape[0]
                S2 = Sk @ Sk
                if not (np.allclose(np.linalg.matrix_power(Sk, 4), np.eye(n), atol=1e-8)
                        and np.allclose(np.linalg.matrix_power(Sk @ Tk, 3), S2, atol=1e-8)
                        and np.allclose(Sk.conj().T @ Sk, np.eye(n), atol=1e-8)):
                    ok = False
                    break
            if ok:
                return gamma, c
    raise RuntimeError("no calibration found")


# --------------------------------------------------------------------------- assembling the spectrum
def adjacency_blocks(p: int, gens: list[Mat] | None = None, gamma_c=None):
    """The blocks (matrix, multiplicity, label) whose eigenvalues, with multiplicities, form the
    spectrum of the Cayley graph of SL_2(F_p) with respect to ``gens`` (default S^±1, T^±1).
    Only the generators S^±1, T^±1 are supported for the Weil part (it needs S and T)."""
    gens = gens if gens is not None else generators(p, "undirected")
    blocks = []
    # principal series
    for k in range(0, (p - 1) // 2 + 1):
        mats = principal_series_matrices(p, k, gens)
        A = sum(mats)
        if k == 0:
            blocks.append((A, (1, p), "Ind(1) = trivial + Steinberg"))  # special: handled below
        elif 2 * k == p - 1:
            blocks.append((A, (p + 1) // 2, "Ind(quadratic) = two reps of dim (p+1)/2"))
        else:
            blocks.append((A, p + 1, f"principal series chi_{k}"))
    # discrete series via the Weil representation (needs S, T, S^-1, T^-1 exactly)
    std = set(generators(p, "undirected"))
    if set(gens) == std:
        Q = quadratic_space(p)
        gamma, c = gamma_c if gamma_c is not None else calibrate_weil(p)
        mats = weil_isotypic_matrices(Q, gamma, c)
        for k in range(1, (p + 1) // 2 + 1):
            Sk, Tk = mats[k]
            A = Sk + Sk.conj().T + Tk + Tk.conj().T
            if 2 * k == p + 1:
                blocks.append((A, (p - 1) // 2, "Weil, theta of order 2 = two reps of dim (p-1)/2"))
            else:
                blocks.append((A, p - 1, f"discrete series theta_{k}"))
    return blocks


def full_spectrum(p: int, gens: list[Mat] | None = None) -> np.ndarray:
    """All p^3 - p eigenvalues (with multiplicity), sorted, from the blocks."""
    vals = []
    for A, mult, label in adjacency_blocks(p, gens):
        ev = np.linalg.eigvalsh((A + A.conj().T) / 2)
        if isinstance(mult, tuple):  # Ind(1): the eigenvalue k (= |gens|) once, the rest p times
            k = len(gens) if gens is not None else 4
            i = int(np.argmin(np.abs(ev - k)))
            vals.append(np.array([ev[i]]))
            rest = np.delete(ev, i)
            vals.append(np.repeat(rest, p))
        else:
            vals.append(np.repeat(ev, mult))
    return np.sort(np.concatenate(vals))


def second_eigenvalue(p: int) -> dict:
    """lambda_2 and lambda_min of Cay(SL_2(F_p), {S^±1, T^±1}) and the block that attains lambda_2."""
    best, worst = -np.inf, np.inf
    best_label = ""
    for A, mult, label in adjacency_blocks(p):
        ev = np.linalg.eigvalsh((A + A.conj().T) / 2)
        if isinstance(mult, tuple):
            ev = np.delete(ev, int(np.argmin(np.abs(ev - 4))))
        if ev.max() > best:
            best, best_label = float(ev.max()), label
        worst = min(worst, float(ev.min()))
    return {"p": p, "lambda2": best, "lambda_min": worst, "block": best_label}


def check_against_bruteforce(p: int) -> float:
    """Max deviation between the assembled spectrum and a direct diagonalisation of the Cayley graph."""
    from .spectral import adjacency

    direct = np.linalg.eigvalsh(adjacency(p).toarray())
    assembled = full_spectrum(p)
    assert direct.size == assembled.size, (direct.size, assembled.size)
    return float(np.max(np.abs(np.sort(direct) - assembled)))
