# Theorems and proofs

This note collects the statements that the project can now *prove*, with complete proofs. Where a
proof uses a finite computation, the computation is specified exactly and is reproduced by
`python -m pytest tests/test_normal_forms.py tests/test_reps.py`. Everything else is classical and
cited. Notation: p is an odd prime, 𝔽_p = ℤ/pℤ,

    S = [[0, −1], [1, 0]],   T = [[1, 1], [0, 1]],   R = S T = [[0, −1], [1, 1]],

ℓ(γ) is the word length of γ with respect to the *undirected* generating set {S, S⁻¹, T, T⁻¹}
and ℓ_dir(γ) with respect to the *directed* set {S, T} (no inverses). Lₙ and Bₙ denote the sphere
and ball of radius n (in SL₂(ℤ) unless a prime is mentioned). Fₙ are the Fibonacci numbers
(F₁ = F₂ = 1), φ = (1+√5)/2, and ψ = 1.46557… is the real root of x³ = x² + 1.

---

## Theorem 1 (sphere sizes of SL₂(ℤ))

**(a) Undirected.** Σₙ |Lₙ| xⁿ = (1 + x)(1 + 2x + 4x² + x³) / (1 − x − x²). Equivalently
|L₀|, |L₁|, |L₂| = 1, 4, 11 and |Lₙ| = 4F_{n+2} for n ≥ 3; in particular |Lₙ| = |L_{n−1}| + |L_{n−2}| for n ≥ 5
and |Lₙ|^{1/n} → φ.

**(b) Directed.** Σₙ |Lₙ^{dir}| xⁿ = (1 + x)(1 + x²)(1 − x + x²)(1 + x + x²) / (1 − x − x³). Equivalently
|Lₙ^{dir}| = 1, 2, 4, 7, 11, 17, 25, 37 for n ≤ 7 and |Lₙ^{dir}| = |L_{n−1}^{dir}| + |L_{n−3}^{dir}| for n ≥ 8;
|Lₙ^{dir}|^{1/n} → ψ.

The proof has four steps: a normal form, an explicit word for every element, a Lipschitz inequality
proving the explicit word is a geodesic, and a transfer-matrix count.

### 1.1 Normal forms

S has order 4, R has order 6, and S² = R³ = −I. It is classical that SL₂(ℤ) is the free product of
⟨S⟩ ≅ ℤ/4 and ⟨R⟩ ≅ ℤ/6 amalgamated over the common centre Z = {±I}. The normal form theorem for
amalgamated products (with the transversals {I, S} of Z in ⟨S⟩ and {I, R, R⁻¹} of Z in ⟨R⟩, using
R² = −R⁻¹) says that every g ∈ SL₂(ℤ) has a **unique** expression

    g = z · c₁ c₂ ⋯ c_m,    z ∈ Z,  m ≥ 0,

in which the syllables cᵢ alternate between the sets {S} and {R, R⁻¹}. We write syllables as the
letters S, R, r (r = R⁻¹) and call (z, c₁⋯c_m) the normal form of g. Right multiplication by a
syllable acts on the normal form only at its end, by the rules

    S·S = −I,   R·R = −R⁻¹,   R·R⁻¹ = I,   R⁻¹·R⁻¹ = −R

(append the syllable; if it has the same type as the last one, combine them by one of these rules,
which deletes or replaces the last syllable and possibly flips z). This is implemented in
`normal_forms.rewrite`; `normal_forms_in_ball` checks, for every element of the undirected ball of
radius 12 (and radius 16 in the session log), that the normal form produced by the rules multiplies
out to the right matrix and that distinct elements get distinct normal forms.

The generators act on normal forms through S⁻¹ = −S, T = S⁻¹R = −SR and T⁻¹ = R⁻¹S, so right
multiplication by a generator changes **at most the last two syllables** and appends at most two.

### 1.2 The explicit word (undirected)

Lift the syllables to words: S ↦ S, R ↦ S T, R⁻¹ ↦ T⁻¹ S⁻¹. Concatenating the lifts of c₁⋯c_m
gives a word whose value is c₁⋯c_m. Runs of consecutive S-letters in this word have length at most 3
(S-letters occur only at the ends of lifts). Delete, left to right, every pair of adjacent S-letters;
each deleted pair has value S^{±1}S^{±1} = ±I, so after deleting the word has value s · c₁⋯c_m with
a known sign s. The element g = z c₁⋯c_m is represented by the word if s·z = +1. If s·z = −1 and the
word contains an S-letter, replace one S by S⁻¹ (or S⁻¹ by S): since S⁻¹ = −S this changes the
value by −1 and not the length. If s·z = −1 and the word contains no S-letter, prepend S² = −I.

Let L(g) be the length of the resulting word w(g). Then **ℓ(g) ≤ L(g)**, because w(g) is a word for g
(`length_formula` returns the word and the tests verify `word_matrix(w) == g`).

Two facts about this construction, both verified by exhaustive computation over all alternating
syllable strings of length ≤ 14 (`test_auxiliary_claims`), and clear from the lifts:

* The reduced word contains no S-letter exactly when the syllable string is a *pure pattern*
  (SR)ᵏ or (R⁻¹S)ᵏ (k ≥ 0), i.e. exactly for the elements ±Tᵏ (indeed Tᵏ = (−1)ᵏ(SR)ᵏ and
  T⁻ᵏ = (R⁻¹S)ᵏ). So L(Tᵏ) = |k| and L(−Tᵏ) = |k| + 2 for every k.
* L₀(σ) := length of the reduced word of the syllable string σ (before the sign step) is a sum of
  local terms: Σ (length of the lift of each syllable) − 2·(number of S-runs of length ≥ 2), and a
  run is determined by at most three consecutive syllables.

### 1.3 The Lipschitz inequality

**Lemma.** For every g ∈ SL₂(ℤ) and every generator x ∈ {S, S⁻¹, T, T⁻¹}: L(gx) ≤ L(g) + 1.

*Proof.* Let σ be the syllable string of g. Three cases.

*(i) g has at most 8 syllables.* Then (z, σ) ranges over a finite set; `verify_lipschitz(8)` computes
L(g) and L(gx) for all 2 × 424 such (z, σ) and all four x (848 checks) and finds max L(gx) − L(g) = 1.

*(ii) Neither g nor gx is of the form ±Tᵏ, and g has more than 8 syllables.* Then the sign step is free
for both, so L(g) = L₀(σ) and L(gx) = L₀(σ′), where σ′ is the syllable string of gx. By 1.1, σ and σ′
agree except in the last two syllables of σ (and σ′ may have up to two more). By 1.2, L₀ is a sum of
window costs over windows of three consecutive syllables, so L₀(σ′) − L₀(σ) only involves windows
meeting the last four syllables of σ; it therefore equals L₀(τ′) − L₀(τ) where τ is the string of
the last 8 syllables of σ and τ′ is obtained from τ by the same rewriting (which stays inside the
last two syllables of τ because τ has 8 ≥ 2 syllables; the sign is irrelevant for L₀).
`verify_lipschitz(8)` checks L₀(τ′) − L₀(τ) ≤ 1 for all 128 strings τ of exactly 8 syllables and all x.

*(iii) g or gx is ±Tᵏ.* If both are, then {g, gx} ⊂ {±Tᵏ, ±T^{k±1}} and L(±T^{k±1}) ≤ L(±Tᵏ) + 1 by
the formulas L(Tᵏ) = |k|, L(−Tᵏ) = |k| + 2. Otherwise exactly one of them is a pure power and x = S^{±1}
(multiplying ±Tᵏ by T^{±1} gives a pure power). The elements ±TᵏS^{±1} have normal forms
±(SR)ᵏS, ±(R⁻¹S)ᵏS, and L of each is |k| + 1 (its reduced word is T^{k}S^{±1}). Then
L(±TᵏS^{±1}) = |k| + 1 ≤ L(±Tᵏ) + 1, and L(Tᵏ) = |k| ≤ (|k|+1) + 1, L(−Tᵏ) = |k| + 2 ≤ (|k|+1) + 1.
`verify_pure_family(14)` recomputes these values from the definitions for all |k| ≤ 14 and both
signs; since L₀ of the periodic strings (SR)ᵏ, (SR)ᵏS, (R⁻¹S)ᵏ, S(R⁻¹S)ᵏ is affine in k with slope 1
(each added period adds the same windows, by locality), the identities hold for all k. ∎

**Corollary.** ℓ(g) = L(g) for every g. *Proof.* ℓ ≤ L by 1.2. Conversely, if g = x₁⋯xₙ with
generators xᵢ, apply the Lemma n times starting from L(I) = 0: L(g) ≤ n. Hence L(g) ≤ ℓ(g). ∎

(The corollary is also confirmed directly on the whole ball of radius 16, 27 044 elements, by
comparing L with breadth-first distances.)

### 1.4 Counting

By the Corollary, |Lₙ| = #{g : L(g) = n}. Elements are pairs (z, σ); for a non-pure σ both signs have
L = L₀(σ), and for the pure patterns the two signs have lengths k and k + 2. Hence, with
N₀(n) = #{σ : L₀(σ) = n}, and since there are exactly two pure patterns with L₀ = k for k ≥ 1
(one for k = 0):

    |Lₙ| = 2N₀(n) − pure(n) + pure(n−2),   pure(0) = 1, pure(k) = 2 for k ≥ 1.

N₀ is counted by a transfer matrix whose states are the last two syllables and whose transition
weights are x^{Δ}, Δ = the change of L₀ when a syllable is appended (a local quantity by 1.2).
Solving the linear system symbolically (sympy) gives

    Σₙ N₀(n) xⁿ = (1 + x)(1 + x + x²) / (1 − x − x²),

and substituting into the displayed formula gives the generating function of (a). The denominator
1 − x − x² and a numerator of degree 4 give |Lₙ| = |L_{n−1}| + |L_{n−2}| for n ≥ 5, and the initial
values force |Lₙ| = 4F_{n+2} for n ≥ 3 (`test_counting_reproduces_enumeration_and_recurrences`
checks the closed form to n = 40 and the agreement with brute-force enumeration to n = 18).

### 1.5 The directed case

Lift S ↦ S, R ↦ S T, R⁻¹ ↦ S³ T S T (= −R² = R⁻¹), concatenate, and reduce every S-run by deleting
factors S² = −I, so that each run has exponent 0 or 1; let f be the number of deleted factors. The
reduced word has value (−1)^f c₁⋯c_m = (−1)^f z g; if that sign is −1 prepend S². This gives a word in
S and T only, of length L_dir(g) = (#S-letters) + (#T-letters) + 2·[sign wrong]; so ℓ_dir ≤ L_dir.
The Lipschitz inequality L_dir(gx) ≤ L_dir(g) + 1 for x ∈ {S, T} is checked by
`verify_lipschitz_directed(8)`: all elements with ≤ 8 syllables directly (424 checks), and all
8-syllable tails with the parity f of the invisible prefix carried as an extra bit (256 checks); the
locality argument is the same as in 1.3, and no special family is needed because the sign term
2·[sign wrong] is already part of the local-plus-parity formula. Hence ℓ_dir = L_dir (confirmed on
the 14 582 elements of directed length ≤ 22). Counting with states (last two syllables, parity) and
the fact that for each σ exactly one sign is "right" gives
|Lₙ^{dir}| = N′(n) + N′(n−2) with N′ the count of strings by reduced length, and sympy returns the
generating function of (b). Its denominator 1 − x − x³ gives the recurrence for n ≥ 8. ∎

---

## Corollary 2 (lower bounds for God's number)

Let G(p) and G_dir(p) be God's numbers of SL₂(𝔽_p). Reduction mod p sends words to words, so the
ball of radius n in SL₂(𝔽_p) is the image of Bₙ ⊂ SL₂(ℤ) and |Bₙ(p)| ≤ |Bₙ|. Hence

    G(p) ≥ min{ n : |Bₙ| ≥ p³ − p },   G_dir(p) ≥ min{ n : |Bₙ^{dir}| ≥ p³ − p },

with |Bₙ| = Σ_{m≤n}|L_m| given exactly by Theorem 1. Since |Bₙ| ~ c φⁿ and |Bₙ^{dir}| ~ c′ ψⁿ,

    G(p) ≥ 3 log_φ p − O(1) = (3/ln φ) ln p − O(1) ≈ 6.23 ln p,
    G_dir(p) ≥ 3 log_ψ p − O(1) = (3/ln ψ) ln p − O(1) ≈ 7.85 ln p.

For comparison, the fitted slopes of the computed values (p ≥ 100) are ≈ 7.5 and ≈ 8.8, and these
bounds account for 75–80 % of G(1009). The naive word-counting bounds have slopes 3/ln 3 ≈ 2.73
and 3/ln 2 ≈ 4.33.

---

## Theorem 3 (the Cayley graphs are not Ramanujan)

Let X̃ = Cay(SL₂(ℤ), {S^{±1}, T^{±1}}) and X_p = Cay(SL₂(𝔽_p), {S^{±1}, T^{±1}}) (4-regular, p odd),
and let ρ(X̃) be the spectral radius of the adjacency operator of X̃ on ℓ². Then

    liminf_{p→∞} λ₂(X_p) ≥ ρ(X̃) ≥ 3.6157 > 2√3 = 3.4641.

In particular only finitely many X_p are Ramanujan. (Numerically none with p ≥ 5 is, and
ρ(X̃) ≈ 3.927–3.931.)

*Proof.* (1) For p odd the quotient map SL₂(ℤ) → SL₂(𝔽_p) induces a covering map X̃ → X_p: it is a
graph homomorphism, and it is injective on the neighbourhood of each vertex because two generators
a ≠ a′ with ga ≡ ga′ mod p would give a′a⁻¹ ∈ Γ(p); but a′a⁻¹ ∈ {S^{±2}, S^{±1}T^{±1}, T^{±1}S^{±1}, T^{±2}}
and none of these is ≡ I mod p (S² = −I and p is odd; the others are not scalar).
(2) *Greenberg's theorem* (Y. Greenberg, thesis, Hebrew University 1995; see Lubotzky–Nagnibeda,
JCTB 1998, and Grigorchuk–Żuk 1999): if finite graphs Xᵢ of bounded degree are covered by a fixed
infinite graph X̃ and |Xᵢ| → ∞, then for every ε > 0 a positive proportion of the eigenvalues of Xᵢ
lie in [ρ(X̃) − ε, deg]; in particular liminf λ₂(Xᵢ) ≥ ρ(X̃).
(3) Let c_{2n} be the number of closed walks of length 2n at I in X̃. Then
c_{2n} = ⟨A^{2n}δ_I, δ_I⟩ = ‖Aⁿδ_I‖² ≤ ‖A‖^{2n} = ρ^{2n}, so c_{2n}^{1/2n} ≤ ρ(X̃) for every n. The
numbers c_{2n} = Σ_x μₙ(x)², where μₙ(x) is the number of walks of length n from I to x, are computed
exactly in integers by `scripts/closed_walks.py`; c₅₆^{1/56} = 3.61577… .
(4) Alternatively, Kesten's theorem (Trans. AMS 92, 1959): for a group with a symmetric generating set
of 2k elements, ρ ≥ 2√(2k−1), with equality iff the group is free on the k generators; SL₂(ℤ) is not
free on S, T (S⁴ = I), so ρ(X̃) > 2√3 without any computation. ∎

*Remarks.* (i) The extrapolation c_{2n} ~ C ρ^{2n} n^{−3/2} (valid for random walks on virtually free
groups) gives ρ(X̃) ≈ 3.93; the same extrapolation recovers 2√3 to four decimals on the 4-regular
tree. (ii) The computed λ₂(X_p) for 37 ≤ p ≤ 101 lie in [3.916, 3.940]: within the precision available,
λ₂(X_p) ≈ ρ(X̃), i.e. no eigenvalue is seen above the spectral radius of the cover. Whether
λ₂(X_p) → ρ(X̃) ("X̃-Ramanujan") is the graph analogue of Selberg's eigenvalue conjecture for the
surfaces Γ(p)\ℍ and is, to my knowledge, open. (iii) Theorem 3 says nothing about the directed
graphs, whose adjacency operator is not normal.

---

## Proposition 4 (half-ball upper bound)

In either convention, if |B_r(p)| > |SL₂(𝔽_p)|/2 then G(p) ≤ 2r.

*Proof.* For g ∈ G the sets B_r and g·B_r⁻¹ = {g y⁻¹ : y ∈ B_r} both have more than |G|/2 elements,
so they meet: x = g y⁻¹ with ℓ(x), ℓ(y) ≤ r, hence g = xy has length ≤ 2r. ∎

## Proposition 5 (the first layers do not depend on p)

Every entry of a product of n matrices from {S^{±1}, T^{±1}} has absolute value ≤ F_{n+2}. Hence if
2F_{n+2} < p then reduction mod p is injective on Bₙ, and |L_m(p)| = |L_m| for all m ≤ n.

*Proof.* Write the matrix by columns [v w] and let (x, y) = (max|v|, max|w|). Right multiplication
by T^{±1} replaces w by w ± v, so (x, y) ↦ (x, ≤ x + y); by S^{±1} it swaps the columns up to sign.
By induction the sorted pair after n steps is ≤ (F_{n+1}, F_{n+2}) termwise (true for n = 0 with
(1, 1) = (F₁, F₂), and (F_{n+1}, F_{n+2}) ↦ (F_{n+2}, F_{n+3}) under a T-step). Two integer matrices
with entries in [−M, M] that agree mod p are equal when 2M < p. ∎

## Proposition 6 (symmetries)

(a) In the undirected convention ℓ(γ⁻¹) = ℓ(γ). (b) Conjugation by D = diag(1, −1) ∈ GL₂(𝔽_p) is an
automorphism of SL₂(𝔽_p) with DSD⁻¹ = S⁻¹ and DTD⁻¹ = T⁻¹; it preserves the undirected generating set,
hence ℓ(DγD⁻¹) = ℓ(γ); in particular ℓ(Tᵏ) = ℓ(T⁻ᵏ). (c) Left multiplication by any h is an
automorphism of the Cayley graph, so all vertices have the same eccentricity: diameter = radius = G(p).

*Proof.* (a) reverse the word and invert each letter. (b) direct computation. (c) x ↦ hx maps the
edge x → xg to hx → hxg. ∎

---

## Proposition 7 (the spectrum of X_p from representations)

The multiset of eigenvalues of X_p is the union over the irreducible representations π of SL₂(𝔽_p)
of dim(π) copies of the spectrum of π(S) + π(S)⁻¹ + π(T) + π(T)⁻¹. Concretely, with
χ_k (0 ≤ k ≤ (p−1)/2) the characters of 𝔽_p^× up to inversion and θ_k (1 ≤ k ≤ (p+1)/2) the characters
of the norm-one group U₁ ⊂ 𝔽_{p²}^× up to inversion:

| block | realisation | size | multiplicity |
|---|---|---|---|
| Ind(χ₀) | functions on 𝔽_p² ∖ 0, homogeneous of degree χ₀ = 1 | p+1 | eigenvalue 4 once; the other p eigenvalues p times (Steinberg) |
| Ind(χ_k), 0 < 2k < p−1 | homogeneous of degree χ_k, (gf)(v) = f(vg) | p+1 | p+1 |
| Ind(χ_{(p−1)/2}) | the quadratic character | p+1 | (p+1)/2 (two representations of dimension (p+1)/2) |
| V_{θ_k}, 0 < 2k < p+1 | θ_k-isotypic part of the Weil representation | p−1 | p−1 |
| V_{θ_{(p+1)/2}} | θ of order 2 | p−1 | (p−1)/2 (two representations of dimension (p−1)/2) |

where the Weil representation on functions on 𝔽_{p²} is ω(T)f(x) = ψ(N(x))f(x) and
ω(S)f(x) = −(1/p) Σ_y ψ(−tr(x ȳ)) f(y), ψ(t) = e^{2πit/p}. The V_{θ=1} part (dimension p) is the
Steinberg representation and is not counted again.

*Verification.* `godsnumber.reps` constructs the blocks (the Weil blocks for all θ at once by a
discrete Fourier transform over U₁) and `check_against_bruteforce(p)` compares the assembled
multiset with a direct diagonalisation of the p³ − p vertex graph: the maximal discrepancy is below
10⁻¹³ for p = 5, 7, 11, 13 (all 120, 336, 1320, 2184 eigenvalues). The relations S⁴ = I and
(ST)³ = S² and unitarity are checked on every block, which fixes the sign conventions and the Weil
index γ = −1 (= (1/p) Σ_x ψ(N(x)), the Gauss sum of the anisotropic plane). With this, λ₂(X_p) costs
O(p⁴) operations instead of a p³-vertex eigenproblem, and one learns which representation attains it.

## Proposition 7b (half of the spectrum is explicit)

For every j = 0, …, p−1 the number 2cos(2πj/p) is an eigenvalue of X_p with multiplicity at least
(p²−1)/2, and at least p²−1 when j ≠ 0 (j and p−j give the same cosine). These eigenvalues (with
these multiplicities) are exactly half of the spectrum, and the
other half is the spectrum of 2π(S) + π(T) + π(T)⁻¹ over the *even* irreducible representations
π (those with π(−I) = 1), each counted dim(π) times.

*Proof.* The subgroup H = {±I} × ⟨T⟩ ≅ ℤ/2 × ℤ/p acts freely on G by right multiplication, so ℓ²(G)
restricted to H is (p³−p)/(2p) = (p²−1)/2 copies of the regular representation of H; thus for each
character (ε, ζʲ) of H the joint eigenspace E_{ε,j} = {f : f(−x) = ε f(x), f(xT) = ζʲ f(x)} has
dimension (p²−1)/2, where ζ = e^{2πi/p}. For f ∈ E_{−1,j}: f(xS⁻¹) = f(x(−I)S) = −f(xS), so
(R(S) + R(S⁻¹))f = 0, and (R(T) + R(T⁻¹))f = (ζʲ + ζ⁻ʲ)f = 2cos(2πj/p) f. Hence A f = 2cos(2πj/p) f on
E_{−1,j}. The odd functions ⊕_j E_{−1,j} form half of ℓ²(G). On the even functions, R(S⁻¹) = R(S),
so A = 2R(S) + R(T) + R(T⁻¹), and R(S)² = R(−I) = 1 there. Decomposing the even part into irreducibles
gives the last statement. ∎

This is why the eigenvalue density of X_p shows a "comb" on [−2, 2] with spikes at ±2 (the arcsine
law of 2cos), and why λ₂, which is about 3.93, comes from even representations.

---

## Theorem 8 (Lubotzky–Phillips–Sarnak; recalled) and the numerical finding

For primes p, q ≡ 1 (mod 4) with q a square mod p, the q+1 quaternions of norm q (a₀ odd > 0,
a₁, a₂, a₃ even) give a symmetric generating set of PSL₂(𝔽_p) whose Cayley graph X^{p,q} is
Ramanujan: all nontrivial eigenvalues satisfy |λ| ≤ 2√q (LPS 1988; the input is the
Ramanujan–Petersson bound for weight-2 cusp forms via Jacquet–Langlands). Consequently
diam X^{p,q} ≤ 2 log_q N + 2 log_q 2 + 1 with N = |PSL₂(𝔽_p)| (LPS), and Sardari (Combinatorica 2019)
proved diam X^{p,q} ≥ (4/3) log_q N − O(1); Sarnak conjectures 4/3 is sharp.

Computed values (`scripts/compute_lps.py`, `data/lps_q*.jsonl`): for q = 5 and p ≤ 1009 the ratio
diam/log₅N stays within about [1.25, 1.5]; see the notebook, §12, for the table and for q = 13, 17.
