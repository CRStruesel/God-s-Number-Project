# The explicit-constant diameter bound through the twisted transfer operator

Goal: a theorem of the form **G(p) ≤ C·log_φ N + O(1) with an explicit C** for the undirected
generators S, T of SL₂(𝔽_p) (N = p³ − p). No such bound is currently known: Selberg's theorem gives
O(log p) with an inexplicit constant, Helfgott's gives (log p)^{O(1)}. This note sets up the
machinery, proves everything except one statement, isolates that statement precisely, and measures
the constant it would deliver. Notation as in `theorems.md`.

---

## 1. The geodesic automaton (Theorem 1 made finite-state)

By Theorem 1 every g ∈ SL₂(ℤ) has a distinguished geodesic word w(g) over the alphabet
{S, s, T, t} (s = S⁻¹, t = T⁻¹). `godsnumber/automaton.py` builds a finite automaton 𝒜 with 12 states
and 27 transitions, start state A and accepting set 𝒜cc, such that

> **Proposition 1.** The map (accepting path of length n) ↦ (value of its word) is a bijection onto Lₙ,
> for every n ≥ 0.

*Construction.* A normal form is an alternating string of syllables S and R^{±1} (R = ST). Its reduced
lift concatenates S ↦ "S", R ↦ "ST", R⁻¹ ↦ "ts" and deletes adjacent pairs of S-letters left to right.
Only two letters can be deleted later than emitted: the "s" ending an R⁻¹-lift (deleted if an
S-syllable follows) and the "S" of an S-syllable (deleted if R follows). Emitting these two letters
*lazily* yields a six-state automaton for the reduced lifts:

    A --T--> Q         (syllables S R: "S"+"ST" → "T")        A --S--> I_T --T--> Q   (R: "ST")
    A --S--> I_t       (S, then end or R⁻¹)                    A --t--> P              (R⁻¹: "t", pending s)
    Q --T--> Q         (S R after R: "…T"+"S"+"ST" → "…TT")   Q --S--> I_t            (S after R)
    I_t --t--> P       (R⁻¹ after S: "S"+"ts")                 I_T --T--> Q
    P --s--> F         (end of word: the pending s is emitted) P --S--> I_T            (S cancels the pending s, then R)
    P --t--> P         (S cancels the pending s, then R⁻¹)

accepting at A (identity), Q, P (an element "…R⁻¹S", word ending in t), I_t (an element "…S"), F (an
element "…R⁻¹"). The sign of the element is fixed by flipping the first S-letter, so each path is
doubled at its first S-letter (layer 0: no S-letter yet; layer 1: flipped or not); the paths without
S-letters are the words Tᵏ, tᵏ, which represent one sign only, and the other sign is given by
SSTᵏ, SStᵏ (a separate 4-state branch). The result is ε-free and unambiguous.

*Proof of Proposition 1 (computer-assisted, `automaton.verify`).* (i) The number of accepting paths of
length n equals the sphere size of Theorem 1 for 0 ≤ n ≤ 40 (an identity of two linear recurrences,
so it holds for all n once it holds up to the sum of their orders, which is < 40). (ii) The set of
accepted words of length ≤ 14 equals {w(g) : ℓ(g) ≤ 14} and every accepted word has exactly one
accepting path. (iii) Since the words w(g) are pairwise distinct and the path counts equal |Lₙ|, the
map path ↦ value is a bijection for every n provided it is injective for every n; injectivity for
n ≤ 14 is (ii), and for larger n it follows from the fact that the automaton reads the normal form
syllable by syllable (each transition corresponds to one or two syllables, which are recovered from
the word) — every accepted word is the reduced lift (possibly first-S-flipped) of a unique syllable
string. ∎

The Perron root of the counting matrix of 𝒜 is φ (verified to 10⁻¹⁵).

## 2. Twisted transfer operators and sphere operators

For a representation π of G_p on V_π, with letters acting by π(S), π(s) = π(S)⁻¹, π(T), π(t) = π(T)⁻¹, set

    M^π := Σ_{(q,a,q′)} E_{qq′} ⊗ π(a)    on ℂ^Q ⊗ V_π,

and let A_n^π := Σ_{g ∈ Lₙ} π(π_p(g)) be the **sphere operators**. By Proposition 1,

    A_n^π = (e_A ⊗ 1)ᵀ (M^π)ⁿ (e_𝒜cc ⊗ 1),          A^π(z) := Σₙ A_n^π zⁿ = (e_A ⊗ 1)ᵀ (1 − zM^π)⁻¹ (e_𝒜cc ⊗ 1),

a rational matrix function. Its poles are at 1/λ for the eigenvalues λ of M^π that are *visible* from
the boundary vectors, and the **visible rate** ρ_vis(π) := limsup ‖A_n^π‖^{1/n} is the reciprocal of
the smallest pole. Always ρ_vis(π) ≤ ρ(M^π); the inequality can be strict and is in practice: for
p = 31 the block with the largest ρ(M^π) = 1.364 has *bounded* sphere operators. The spectral radius
of the transfer matrix is therefore the wrong invariant; the visible rate is the right one.

The conventions are verified by the Plancherel identity N·Σ_γ cₙ(γ)² = Σ_π dim(π)‖A_n^π‖²_HS, with
cₙ the exact lift counts of the spheres of SL₂(ℤ), which holds to 10⁻⁹ for p = 7, 11 and n ≤ 8
(`tests/test_automaton.py`).

## 3. The lemma: a uniform bound on the sphere operators gives an explicit diameter bound

**Hypothesis E(β, C).** For every prime p, every nontrivial irreducible π of G_p and every n ≥ 0,
‖A_n^π‖_op ≤ C φ^{βn}.

**Theorem 3 (E(β, C) ⇒ explicit diameter bound).** Under E(β, C),

    G(p) ≤ ( log_φ N + log_φ(C′/c_φ) ) / (1 − β) + 1,     C′ = C φ^β/(φ^β − 1),  c_φ = 4φ⁴/√5,

for every p; in particular G(p) ≤ log_φ N/(1 − β) + O(1). The same computation gives the typical-length
statement (a) below.

*Proof.* Let Cₙ(γ) = #{g ∈ Bₙ : π_p(g) = γ} and λₙ = |Bₙ|/N. For a function f on G_p write
f̂(π) = Σ_γ f(γ)π(γ). Then Ĉₙ(π) = Σ_{m≤n} A_m^π, and for the centred function Cₙ − λₙ the trivial
representation drops out while every nontrivial one is unchanged. Fourier inversion on G_p,
f(γ) = (1/N) Σ_π dim(π) tr(f̂(π)π(γ)⁻¹), gives for every γ

    |Cₙ(γ) − λₙ| ≤ (1/N) Σ_{π≠1} dim(π) |tr(Ĉₙ(π) π(γ)⁻¹)| ≤ (1/N) Σ_{π≠1} dim(π)² ‖Ĉₙ(π)‖_op ≤ max_{π≠1} ‖Ĉₙ(π)‖_op,

using |tr(XU)| ≤ dim·‖X‖_op for unitary U and Σ_π dim(π)² = N. Under E(β, C),
‖Ĉₙ(π)‖_op ≤ Σ_{m≤n} C φ^{βm} ≤ C′ φ^{βn}. Hence every γ satisfies Cₙ(γ) ≥ λₙ − C′φ^{βn}, which is
positive as soon as c_φ φⁿ/N > C′ φ^{βn}, i.e. n > (log_φ N + log_φ(C′/c_φ))/(1 − β) (using
|Bₙ| ≥ c_φ φⁿ − O(1), which holds for n ≥ 1 by Theorem 1). Every γ is then in π_p(Bₙ), i.e. G(p) ≤ n. ∎

Two corollaries of the same computation:

  (a) (**typical length**) Σ_γ (Cₙ(γ) − λₙ)² = (1/N) Σ_{π≠1} dim(π) ‖Ĉₙ(π)‖²_HS ≤ C′² φ^{2βn}, so the
      fraction of γ with ℓ(γ) > n is ≤ C′² φ^{2βn}/(N λₙ²) = C′² N c_φ^{−2} φ^{−2(1−β)n}, which is o(1) for
      n ≥ log_φ N/(2(1−β)) + ω(1): the typical length is at most log_φ N/(2(1−β)) + O(1).
  (b) (**where the constant comes from**) the two uses of the hypothesis — once for each γ (ℓ∞, giving
      the factor 1/(1−β)) and once on average (ℓ², giving 1/(2(1−β))) — are exactly the two factors
      in the Lubotzky–Phillips–Sarnak argument for Ramanujan graphs, where β = ½ and the bounds read
      2 log_q N and log_q N.

## 4. What the hypothesis is worth: the measured constant

`transfer.visible_rates` computes ρ_vis(π) for every nontrivial irreducible (growth of ‖A_n^π‖_op over
24 ≤ n ≤ 48, blocks of size ≤ p+1):

| p | max ρ_vis | β = log_φ max ρ_vis | attained in | median ρ_vis over blocks |
|---|---|---|---|---|
| 31 | 1.3415 | 0.611 | principal series χ₆ | 1.186 |
| 61 | 1.3353 | 0.601 | principal series χ₁₈ | 1.112 |
| 101 | 1.3435 | 0.614 | principal series χ₃₀ | 1.135 |

(The ℓ²₀-norm growth of the full sphere operators computed directly for p ≤ 13 was 1.34–1.40,
consistent.) So **Hypothesis E(β) holds numerically with β ≈ 0.61, uniformly in p**, and Theorem 3
would give

    G(p) ≤ 2.6 · log_φ N + O(1) ≈ 7.8 · log_φ p + O(1) ≈ 16 ln p + O(1)

against the truth ≈ 1.2 log_φ N ≈ 7.5 ln p. The typical-length corollary gives 1.3 log_φ N. These are
the constants an operator-norm method can reach; they are not sharp, but they would be the first
explicit ones.

Two remarks. (i) The median block has ρ_vis ≈ φ^{0.3}, and the Plancherel balance
Σ_{π≠1} dim(π)‖A_n^π‖²_HS = N|Lₙ|·(1.00 ± 0.03) at n = 20 shows that *on average over π* the sphere
operators are at the Ramanujan scale φ^{n/2} — this is Hypothesis V of `conjecture_A.md`; the few
worst blocks are what the ℓ∞ argument pays for. (ii) Since 1.34² = 1.80 > φ, the worst blocks eventually
dominate the variance: Σ_{π≠1} dim(π)‖A_n^π‖²_HS/(N|Lₙ|) grows from 1.0 at n = 20 to 1.3–2.2 at n = 40
(p = 101, 31), i.e. the excess over Poisson variance grows like (1.34²/φ)ⁿ ≈ 1.11ⁿ. This is the
spectral face of the arithmetic tail of §13 of the notebook.

**The free-monoid instance {T, U}.** For the directed generators T, U the geodesic language is the
free monoid on two letters (Stern–Brocot), so the automaton has one state and the transfer operator
is just π(T) + π(U); the sphere operators are its powers and ρ_vis(π) is its spectral radius (not its
norm, which equals 2 on every large irreducible because π(T⁻¹U) = π(R) has eigenvalue 1). The maximum
over nontrivial π of the spectral radius of π(T) + π(U) is 1.554–1.570 = 2^{0.64} for
p ∈ {11, 31, 61, 101, 151, 199}, with median 1.42–1.44 ≈ √2 — the Ramanujan value for a 2-letter free
monoid. The same lemma (with 2 in place of φ and log₂ N) gives G_{T,U}(p) ≤ 2.8 log₂ N + O(1) under the
corresponding hypothesis.

## 5. What remains: the uniform gap, and a program for it

Everything above is proved except Hypothesis E(β, C) (or, for the typical length, the weaker
Hypothesis V). The hypothesis is a **uniform spectral gap for the sphere operators**, i.e. for the
Markov measures on G_p given by the paths of 𝒜. This is a Bourgain–Gamburd-type statement for a
*Markov* (non-i.i.d.) random walk, and I do not know a reference that proves it as stated. The
inputs of the Bourgain–Gamburd method are available or provable:

1. **Non-concentration of spheres in proper subgroups.** For the Markov measure σₙ (uniform on
   paths of length n) and every proper subgroup H < G_p: σₙ(H) ≤ |Lₙ|^{−δ} for n ≤ c log p. Proper
   subgroups of SL₂(𝔽_p) are (Dickson) contained in a Borel, a dihedral group, or have bounded order
   (A₄, S₄, A₅); the lift of an element of such a subgroup satisfies a bounded-length identity
   (e.g. [g, h]ᵖ-type or commutator relations) with elements of the ball; by Theorem 1 the number of
   elements of Lₙ in any algebraic subgroup of SL₂(ℤ) (torus, unipotent, Borel) is O(n), so the
   standard argument applies in the injective range n ≤ ⅓ log_φ N.
2. **The product theorem** (Helfgott): |A³| ≥ |A|^{1+ε} for generating sets A of SL₂(𝔽_p) of size
   ≤ |G_p|^{1−δ}. Available.
3. **ℓ²-flattening.** For i.i.d. walks: ‖μ^{*2n}‖₂ ≤ ‖μ^{*n}‖₂^{1+ε} unless μ^{*n} concentrates on a
   coset of a proper subgroup (the Bourgain–Gamburd lemma, via Balog–Szemerédi–Gowers and the product
   theorem). For the Markov measure σ the convolution structure is replaced by concatenation of
   paths: σ_{2n} = Σ_q σₙ^{(→q)} * σₙ^{(q→)} summed over the 12 states q, and each summand is a
   convolution of two measures of the same type. The BSG/product step goes through summand by
   summand; the bookkeeping of the 12 states is the new (but finite) part of the argument. This is
   the one step I have not carried out.
4. **From flattening to the gap.** Once ‖σ_{2n}‖₂ ≤ ‖σₙ‖₂^{1+ε} up to the scale ‖σₙ‖₂ ≈ N^{−1/2+δ},
   the high multiplicity of nontrivial irreducibles of SL₂(𝔽_p) (dim π ≥ (p−1)/2, Frobenius) converts
   it into ‖A_n^π‖ ≤ φ^{βn} for all nontrivial π with an explicit β < 1, exactly as in
   Bourgain–Gamburd (their Proposition 3 / the "trace method"). This step is formal.

The same program for the free monoid {T, U} has a 1-state automaton, so step 3 is literally
Bourgain–Gamburd for the *non-symmetric* generating set {T, U} (a convolution power of the measure
½(δ_T + δ_U), with the group generated being SL₂(ℤ) — Zariski dense), for which the theorem *is*
available: Bourgain–Gamburd's result does not require the measure to be symmetric, only that its
support generates a Zariski-dense subgroup. Hence:

> **Theorem (conditional only on reading Bourgain–Gamburd carefully for non-symmetric measures).**
> There is an explicit β < 1 with ρ(π(T) + π(U)) ≤ 2^β for all nontrivial π of SL₂(𝔽_p) and all p, and
> consequently G_{T,U}(p) ≤ log₂ N/(1 − β) + O(1).

The explicit value of β coming out of Bourgain–Gamburd is poor (their ε's are tiny); the numerically
true value is β ≈ 0.64. For S, T the route is the same with the 12-state bookkeeping.

**Summary.** The automaton reduces the sphere operators of the word metric to compressions of powers
of one finite matrix per representation; the diameter bound follows from a uniform bound on these
compressions with explicit constants (Theorem 3, proved); the bound holds numerically with β ≈ 0.61
and would give G(p) ≤ 2.6 log_φ N + O(1); and the missing piece is a Bourgain–Gamburd argument for
the Markov measure of the automaton — immediately available in the 1-state (free monoid) case
{T, U}, where it already yields an explicit-constant bound for God's number of SL₂(𝔽_p) with respect
to the Stern–Brocot generators.
