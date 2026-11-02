# Conjecture A (the bulk limit law): what is proved, what it reduces to, and why the obvious tools stop short

Notation as in `theorems.md`: G = SL₂(ℤ), A = {S^{±1}, T^{±1}}, Lₙ and Bₙ the sphere and ball of radius n in G,
G_p = SL₂(𝔽_p), N = p³ − p, π_p : G → G_p reduction, ℓ = ℓ_p the word length in G_p. By Theorem 1,
|Lₙ| = 4F_{n+2} for n ≥ 3 and |Bₙ| ~ c_φ φⁿ with c_φ = 4φ⁴/√5 = 12.26…; log_φ N = 3 log_φ p + o(1).

**Conjecture A.** For γ uniform in G_p, as p → ∞,

    P(ℓ(γ) ≤ n) = 1 − exp(−|Bₙ|/N) + o(1),   uniformly in n.

Equivalently, with λₙ := |Bₙ|/N: the number of elements of G_p *not* in π_p(Bₙ) is N e^{−λₙ}(1 + o(1)) whenever
λₙ is bounded, and ℓ − log_φ(N/c_φ) converges in law to the integer-valued distribution F(t) = 1 − exp(−φ^{t}).

Everything below is organised around the **lift counts**

    cₙ(γ) := #{ g ∈ Lₙ : π_p(g) = γ },      Cₙ(γ) := #{ g ∈ Bₙ : π_p(g) = γ } = Σ_{m ≤ n} c_m(γ),

so that ℓ(γ) ≤ n ⟺ Cₙ(γ) ≥ 1, and Σ_γ cₙ(γ) = |Lₙ| exactly. Conjecture A says that Cₙ(γ) behaves, as γ varies,
like a Poisson variable with mean λₙ.

---

## 1. What is proved unconditionally

**Theorem A.1 (exact law in the injective range).** Let n₀(p) = max{n : 2F_{n+2} < p} (so n₀(p) = log_φ p − O(1)
= ⅓ log_φ N − O(1)). For n ≤ n₀(p), π_p is injective on Bₙ (Proposition 5 of `theorems.md`), hence
Cₙ(γ) ∈ {0, 1} for every γ and

    P(ℓ ≤ n) = |Bₙ|/N   exactly.

Since |Bₙ|/N ≤ c_φ φ^{n₀}/N = O(p⁻²), this agrees with 1 − exp(−|Bₙ|/N) up to O(p⁻⁴). ∎

**Theorem A.2 (two trivial bounds, valid for all n).** P(ℓ ≤ n) ≤ |Bₙ|/N, and
P(ℓ ≤ n) ≥ |Bₙ|²/(N·Σ_γ Cₙ(γ)²) (Cauchy–Schwarz applied to Σ_γ Cₙ(γ) = |Bₙ| over the support of Cₙ). ∎

The second bound is the engine of everything that follows: it converts a **variance bound** into a lower bound for
the coverage. Note the first bound is tight in the injective range and the Poisson prediction 1 − e^{−λ} ≤ λ is
consistent with it.

**Theorem A.3 (what the adjacency spectral gap gives, and why it is not enough).** Let λ₂(p) be the second
eigenvalue of X_p (Theorem 3 gives λ₂ ≥ 3.6157 − o(1); Selberg's 3/16 theorem gives sup_p λ₂ < 4 with an
inexplicit constant; numerically λ₂ ≈ 3.93). Write g := log(4/λ₂) > 0. Then for all p and n:

  (a) (random-walk mixing) the uniform measure μ on A satisfies ‖μ^{*n} − 1/N‖₂ ≤ (λ₂/4)ⁿ ‖δ_I − 1/N‖₂, so
      μ^{*n} is within ε of uniform in ℓ¹ as soon as n ≥ (log N + 2 log(1/ε))/(2g); since supp μ^{*n} ⊂ π_p(Bₙ),
      P(ℓ ≤ n) ≥ 1 − ε for such n;
  (b) (Chung) diam X_p ≤ ⌈log(N−1)/g⌉.

With the numerical value λ₂ ≈ 3.93 one has g ≈ 0.0177, so (a) gives "typical distance ≤ 28 log N" and (b)
"diameter ≤ 57 log N", against the truths 2.1 log N and ≈ 2.5 log N. The gap controls *mixing*, which
happens at time ≍ (1/g) log N ≫ log_φ N; the typical distance is governed by *growth* (the constant φ), not by
mixing. No improvement of the gap short of the Ramanujan value 2√3 — which Theorem 3 rules out — changes this
conclusion: even with a Ramanujan gap the same argument only reaches n ≈ 2 log_φ N (that is the factor 2 in the
Lubotzky–Phillips–Sarnak diameter bound). ∎

---

## 2. Rigorous reductions

Define the (centred) sphere counting measures and their second moments

    Vₙ(p) := Σ_γ (cₙ(γ) − |Lₙ|/N)²  =  #{(g, h) ∈ Lₙ² : g ≡ h mod p} − |Lₙ|²/N .

**Hypothesis V(c).** There is a constant C such that Vₙ(p) ≤ C nᶜ |Lₙ| for all p and all n ≤ 2 log_φ N.

(The Poisson model predicts Vₙ(p) ≈ |Lₙ|: variance = mean. Hypothesis V says the sphere counting measure
exhibits square-root cancellation *on the vector δ_I − 1/N*: ‖Aₙ(δ_I − 1/N)‖₂² ≤ C nᶜ φⁿ, where
Aₙ = Σ_{g ∈ Lₙ} R(π_p(g)).)

**Theorem A.4 (V ⇒ concentration at the growth scale).** Assume V(c). Then for all t ≥ 0 and all p,

    P(ℓ > log_φ(N/c_φ) + t) ≤ C′ (log N)^{c} φ^{−t},      P(ℓ < log_φ(N/c_φ) − t) ≤ C″ φ^{−t}.

In particular ℓ(γ) = log_φ(N/c_φ) + O_P((c+1) log log N) and the median of ℓ is log_φ(N/c_φ) + O(log log N).

*Proof.* The lower bound is Theorem A.2: P(ℓ ≤ n) ≤ |Bₙ|/N ≤ C″ φ^{n − log_φ(N/c_φ)}. For the upper bound, let
n = ⌈log_φ(N/c_φ) + t⌉, so λₙ = |Bₙ|/N ≥ φ^{t}/2. By Minkowski, (Σ_γ (Cₙ(γ) − λₙ)²)^{1/2} ≤ Σ_{m≤n} Vₘ(p)^{1/2}
≤ Σ_{m ≤ n} (C mᶜ |L_m|)^{1/2} ≤ C₁ nᶜ/² |Bₙ|^{1/2}, hence Σ_γ (Cₙ(γ) − λₙ)² ≤ C₁² nᶜ |Bₙ|. The elements with
ℓ(γ) > n are those with Cₙ(γ) = 0, each contributing λₙ² to the sum, so

    #{γ : ℓ(γ) > n} ≤ C₁² nᶜ |Bₙ| / λₙ² = C₁² nᶜ N / λₙ ≤ 2 C₁² nᶜ N φ^{−t}.  ∎

Theorem A.4 is the **second-moment form of Conjecture A**: it pins the typical length to within O(log log N)
of log_φ(N/c_φ) (with c = 0, to within O(1)), and gives a tail bound of the right exponential shape, with
exponent 1 instead of the conjectured 3 (Conjecture B). It does *not* give the Poisson law itself.

**Hypothesis M(k).** For the k-th factorial moment, Σ_γ Cₙ(γ)(Cₙ(γ)−1)⋯(Cₙ(γ)−k+1) = N λₙᵏ (1 + o(1)) uniformly
for λₙ in compact subsets of (0, ∞), as p → ∞.

(M(1) is the identity Σ_γ Cₙ(γ) = |Bₙ|. M(2) is V with the error term o(N λₙ²) instead of O(|Bₙ|); at λₙ ≍ 1
these are the same order, so M(2) is slightly stronger than V(0). M(k) says that the number of k-subsets of Bₙ
whose elements are pairwise congruent mod p is N λₙᵏ/k! (1 + o(1)): "k-fold coincidences are as frequent as for
a random map".)

**Theorem A.5 (method of moments).** If M(k) holds for every k ≥ 1, then Conjecture A holds: for each fixed t,
P(ℓ ≤ log_φ(N/c_φ) + t) → 1 − exp(−φ^{t}) and, more precisely, the empirical distribution of Cₙ(γ) over γ
converges to Poisson(λₙ) whenever λₙ → λ ∈ (0, ∞).

*Proof.* Let X_p be the random variable Cₙ(γ) for γ uniform in G_p. M(k) says E[X_p(X_p−1)⋯(X_p−k+1)] → λᵏ for
every k, which are the factorial moments of Poisson(λ). The Poisson law is determined by its moments (its
moment generating function is entire), so X_p → Poisson(λ) in distribution (Fréchet–Shohat), and in particular
P(X_p = 0) → e^{−λ}, i.e. P(ℓ > n) → e^{−λₙ}. The identification of the limit of ℓ − log_φ(N/c_φ) follows from
λₙ = |Bₙ|/N = c_φ φⁿ/N (1 + o(1)). ∎

(A quantitative version, with explicit o(1)'s, follows from the Bonferroni inequalities
Σ_{k=1}^{2K}(−1)^{k+1}E[C(X,k)] ≤ P(X ≥ 1) ≤ Σ_{k=1}^{2K−1}(−1)^{k+1}E[C(X,k)] applied with K ≍ log(1/ε)/log(1/λ).)

So **Conjecture A is equivalent to the statement that k-fold coincidences in the balls of SL₂(ℤ) modulo p are
Poisson-frequent for every k**, and its second-moment shadow (Theorem A.4) needs only the pair-coincidence count.
This is the precise form in which it should be attacked.

---

## 3. Numerical status of the hypotheses

`notebooks/gods_number.ipynb`, §13.1, computes cₙ(γ) exactly for n ≤ 22 (|L₂₂| = 185 472) and p ∈ {31, 61, 101, 151}.

* **Variance.** Vₙ(p)/|Lₙ| lies in [0.91, 1.07] for every n ≤ 22 and every p tested — through the collision
  regime (p = 31: λ₂₂ = 6.2). So Hypothesis V holds numerically with C = 1.1, c = 0: the sphere counting
  measures have *exactly* Poisson variance.
* **Law.** At p = 31, n = 20 (λ = 2.38) the empirical frequencies of cₙ(γ) = 0, 1, 2, 3, 4 are
  0.110, 0.195, 0.261, 0.206, 0.138 against the Poisson values 0.093, 0.220, 0.262, 0.208, 0.124; the second and
  third factorial moments are 5.70 and 13.43 against λ² = 5.67 and λ³ = 13.49. For small λ the third moment
  shows an excess (p = 31, n = 18: 0.89 vs 0.75), the first trace of the arithmetic tail.
* **Operator norms.** ‖Aₙ‖ on ℓ²₀(G_p) grows like (1.34–1.40)ⁿ ≈ φ^{0.63 n} for p = 5, 7, 11, 13 — strictly
  between √φ = 1.272 and φ = 1.618. So there is a **uniform spectral gap for the sphere operators** (numerically
  β ≈ 0.63 in ‖Aₙ‖₀ ≤ C φ^{βn}), but no square-root cancellation on worst-case vectors: the cancellation that
  V needs is specific to the vector δ_I − 1/N. (Curiously, 1.37 is also the growth rate of the largest matrix
  entry over Lₙ; whether the extremal vectors live on the unipotent subgroup ⟨T⟩ is worth checking.)

A sphere-operator gap with exponent β gives, by the argument of Theorem A.4 with ‖Aₙ‖₀ in place of V:
typical distance ≤ log_φ N/(2(1−β)) + O(1) and diam ≤ log_φ N/(1−β) + O(1) — with β = 0.63 that is
1.35 log_φ N and 2.7 log_φ N. These would be genuine theorems about G(p) if the gap were proved, and they show
the shape of what operator-norm methods can deliver: constants, never the limit law.

---

## 4. Routes to a proof

1. **The twisted transfer operator.** By §8.4 the geodesic words form a regular language 𝓛 with one word per
   element (the output of a finite-state transducer applied to the amalgam normal forms). Let 𝒜 be its
   automaton, with state set Q. For a representation π of G_p set
   M^π := Σ_{q →a q′} E_{qq′} ⊗ π(a) on ℂ^Q ⊗ V_π. Then Aₙ^π := Σ_{g ∈ Lₙ} π(g) = (row vector of the start state)
   · (M^π)ⁿ · (column vector of accepting states), and the whole question is the growth of ‖(M^π)ⁿ‖ for
   nontrivial π. For π trivial, M^π is the transition-count matrix with Perron root φ (Theorem 1).
   A *uniform spectral gap* — spectral radius of M^π ≤ φ^{1−ε} for all nontrivial π of G_p, all p — is a
   "Bourgain–Gamburd theorem for the automaton walk". The sphere measure is a Markov measure on 𝒜, not a
   convolution power, so Bourgain–Gamburd does not apply verbatim; but Bourgain–Kontorovich (Zaremba, Ann. of
   Math. 2014, §§ on the "congruence transfer operator") and Magee–Oh–Winter (Crelle 2019) prove exactly such
   uniform gaps for congruence twists of transfer operators of Schottky semigroups in SL₂(ℤ), with the
   product theorem of Helfgott as the input. Our automaton is a finite-type symbolic system for the Zariski-dense
   group SL₂(ℤ); adapting their argument from the norm grading to the symbolic-length grading looks feasible and
   would give E(β) for some explicit β < 1, hence (§3) **diam G(p) ≤ C log_φ N with an explicit C** — a theorem
   of the LPS type, and the first unconditional O(log p) with a known constant for S, T.
2. **The variance (Hypothesis V) directly.** V is a count of pairs (g, h) ∈ Lₙ² with g⁻¹h ∈ Γ(p):
   Vₙ(p) + |Lₙ|²/N = Σ_{γ ∈ Γ(p)} rₙ(γ), rₙ(γ) = #{g ∈ Lₙ : gγ ∈ Lₙ}. The γ = I term is |Lₙ|; V says the rest is
   |Lₙ|²/N + O(nᶜ|Lₙ|). A route: g ↦ (g, g⁻¹h) and equidistribution of the *pair* in G_p × G_p — i.e. a gap for
   M^{π⊗π′} — which is route 1 again, for tensor products. The numerics (Var/mean ≈ 1) say the truth is far
   better than any gap-type bound; proving that is a Ramanujan-type problem (compare: for LPS graphs the analogue
   is Deligne's bound), and I see no modular-forms input for S, T spheres. This is where I expect Conjecture A
   to be genuinely hard.
3. **Averaging over p.** Statements averaged over primes p ≤ P are often provable when individual ones are not
   (large sieve). The average of Vₙ(p) over p is a count of pairs (g, h) ∈ Lₙ² weighted by the number of primes
   dividing all entries of g − h; the divisor-type bound ω(g − h) = O(n/log P) is too weak by a factor n, but a
   more careful count using the explicit continued-fraction structure of Lₙ (Theorem 1) might give
   "Hypothesis V on average over p", hence Theorem A.4 for a density-one set of primes.

**Assessment.** The limit law itself (Conjecture A) is, in my judgement, out of reach of current tools: it is a
Poisson statement for all k-fold coincidences, equivalent to "the balls of SL₂(ℤ) project as if by a random map",
and already for LPS Ramanujan graphs the corresponding O(1)-scale law is not known. What *is* within reach, and
would already be new for S, T: (i) an explicit-constant diameter bound via route 1; (ii) Theorem A.4 — the
typical length is log_φ(N/c_φ) + O(1) — if Hypothesis V can be proved, perhaps first on average over p.
