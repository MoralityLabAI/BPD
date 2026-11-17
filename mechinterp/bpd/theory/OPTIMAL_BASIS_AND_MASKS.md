# Expected-L0 BPD identifiability: optimal bases and invariant interventions

**Research addendum, 17 November 2026.** This result addresses an idealized exact full-rank square linear dictionary with unregularized expected L0 sparsity. The results do not establish recovery by a trained BPD, and leave the B1–B4 receipts and the pinned VPD/phase-disjoint gates untouched.

## 1. Objective, filtration, and optimal basis theorem

Let \(a\) be an activation random vector in \(\mathbb R^d\), let \(M\) have independent nonzero rows \(m_1,\ldots,m_d\), and define the projective hyperplane masses
\[
z(m)=\Pr[m^\top a=0],\qquad J(M)=\mathbb E\|Ma\|_0
=d-\sum_{i=1}^{d}z(m_i).
\]
Nonzero rescaling of a row preserves its mass. Minimize \(J\) by choosing a maximum-weight linear-matroid basis of read directions.

For \(t>0\), let \(E_t=\{m\ne0:z(m)\ge t\}\) and \(F_t=\operatorname{span}E_t\). Assume a maximizing basis exists, as it does for a finite candidate universe and a finitely supported activation law. A basis \(\mathcal B\) is optimal precisely when
\[
\operatorname{span}(\mathcal B\cap E_t)=F_t
\quad\text{at every positive weight threshold }t.
\]
The proof uses layer-cake and matroid greedy selection:
\[
\sum_{m\in\mathcal B}z(m)
=\int_0^1|\mathcal B\cap E_t|dt
\le\int_0^1\dim F_t\,dt.
\]
A greedy maximum-weight basis saturates the upper bound at each threshold. Whenever a candidate basis fails the filtration condition, its rank deficit persists on a positive-width interval below one of the finitely many positive weight levels, forcing a strictly smaller sum.

The entire filtration \(F_t\) is therefore common to every minimizer. A positive strict margin isolates a reference basis up to row rescaling and permutation in the declared competitor universe; a zero margin permits ties without specifying their shape; a negative margin excludes the reference from the optimum set. The maximum-weight-basis theorem is established here as a mathematical argument and by finite rational checks. The general vector-matroid theorem is not yet Lean-formalized in this repository.

## 2. Optimal-set geometries and exact finite witnesses

**Free remixing.** If every nonzero read direction has the same hyperplane mass, any invertible read basis is optimal. For example, a full-dimensional absolutely continuous distribution gives \(z(m)=0\) for all \(m\ne0\). The point mass at zero gives \(z(m)=1\) for all such directions.

**Nested support.** Give probability \(1/2\) to each of \((1,0)\) and \((0,0)\). The projective direction \((0,1)\) has mass \(1\); all other noncollinear reads have mass \(1/2\). Every minimizing basis contains \((0,1)\), while the other row is free along \((1,\lambda)\). The high-mass direction survives even though shearing can link the two intervention coordinates.

**Atomic cancellation.** Put equal probability \(1/3\) on \((1,0),(0,1),(1,1)\). The only positive-weight projective reads are
\[
(1,0),\quad (0,1),\quad (1,-1)
\]
with \(z=1/3\) each. Any two span, producing exactly three unordered optimal projective bases. Reads \((1,1)\) and \((1,2)\) each have zero mass. Equal optimum values do not permit arbitrary remixing.

**Unique optimum rotated away from the reference.** Put equal probability \(1/2\) on \((1,1)\) and \((1,-1)\). The only positive-weight read directions are \((1,1)\) and \((1,-1)\), each with mass \(1/2\), and their pair is the unique optimum. The reference coordinate pair \(e_1,e_2\) has total mass zero. An absolute graph linking components touched together by an optimal read would merge both *reference* coordinates, although the optimal dictionary has two individually identifiable rows.

These latter cases have exhaustive exact-integer Lean certificates on five declared candidate directions and independent exact-Fraction Python enumeration. The assertion about all real projective directions is a mathematical consequence of their explicitly enumerated activation hyperplanes, not of finite five-direction testing.

## 3. Relative-optimum identification and common mask algebra

Choose an optimal anchor \(M_*\) rather than a possibly suboptimal reference, and compare any other optimum \(M\) by
\[
R=M M_*^{-1},
\]
with consistent row label, permutation, and nonzero scaling alignment. Monomial transformations amount only to renaming/rescaling components.

Suppose a full-rank factorization \(W=U_*M_*\) has an injective write map \(U_*\), while \(M_*\) has full row rank. Reparameterization gives \(M=RM_*\) and \(U=U_*R^{-1}\). A diagonal \(0/1\) intervention mask \(D\) is representation invariant exactly when
\[
U_*R^{-1}DRM_*=U_*DM_*
\quad\Longleftrightarrow\quad
R^{-1}DR=D
\quad\Longleftrightarrow\quad DR=RD.
\]
If the available activation domain spans only a proper subspace or output writes identify distinct latent vectors, the equivalence must be restricted to the actually reached/observable image.

For \(D=\mathrm{diag}(d_1,\ldots,d_d)\), the matrix commutator is entrywise
\[
(DR-RD)_{ij}=(d_i-d_j)R_{ij}.
\]
Thus every nonzero off-diagonal edge of an aligned relative optimal transformation forces \(d_i=d_j\) for invariant masks. Connected components of the union of those edges describe the finest intervention-block partition under this alignment. Intrinsically, the invariant diagonal masks form the diagonal part of the common centralizer of all relative optimum transformations.

For a block projector \(D_B\) commuting with all relative transformations, the block weight contribution
\[
W_B=U_*D_BM_*=U_*R^{-1}D_BRM_*
\]
is invariant. Since \(R\) respects the invariant block decomposition and is invertible, the aggregate block-activity event \(D_BM_*a=0\) is equivalent to \(D_BRM_*a=0\). Single-component activity and singleton interventions may change when shears are allowed. A unique optimal basis rotated relative to the original reference has monomial relative symmetries and hence retains its singleton intervention masks.

**Proof scope.** Optimal readout filtration, invariant intervention algebra, and block-level represented weight contributions are distinct notions. With a specified anchor and alignment, the graph is a computational description of common mask commutation. A complete anchor-independent classification of arbitrary-dimensional projective optimum families remains open; it requires careful treatment of row-permutation gauge and possibly nonunique alignment. Residual matrices, approximate ties, overcomplete dictionaries, trained gates, and nonlinear behavior require separate analysis.

Lean source OptimalBasis.lean proves: finite optimal-set witnesses; a general equivalence between commutation and conjugated-mask invariance for mutually inverse reparameterizations; a surjective encoder/injective decoder bridge; a concrete two-coordinate shear preserving full/zero masks but changing a singleton mask.

## 4. Composition, finite-context support, and temporal gates

**Finite context:** a sample-count sufficient condition for basis identification should be described as sufficient; the exact condition is a hyperplane mass comparison. Atomic cancellation generalizes to all hyperplanes, including multivariate cancellations outside pairwise support tests.

**Reduced recurrent composition:** write \(s_{k+1}=F(s_k)\), let \(\pi\) be a projection, and let \(\hat s_{k+1}=\hat F(\hat s_k)\) with \(\hat s_0=\pi(s_0)\). Assume
\[
\|\pi(F(s_k))-\hat F(\pi(s_k))\|\le\varepsilon
\]
for each actual intervened network state \(s_k\), and that \(\hat F\) is \(L\)-Lipschitz on a domain including the projected states and the reduced path. Then
\[
e_{k+1}\le\varepsilon+Le_k,\qquad
e_n\le\varepsilon\sum_{j=0}^{n-1}L^j.
\]
When \(L<1\) the bound is uniformly finite; at \(L=1\) it grows linearly. The Lean file proves a natural-number algebraic recurrence analogue. The full real-norm Lipschitz bound is a written paper argument.

**Temporal switch optimization:** a finite switch coefficient cannot establish lexicographic priority over sparse execution. Use two constrained stages: first train read/write factors and gates for reconstruction and sparsity; freeze factors; then refit gates with a switch penalty subject to registered tolerances on both stage-one losses. Positive tolerances yield approximate lexicographic preference. Gates may stay active through gaps, so evaluate against the phase-disjoint control with the dictionary trained throughout.

Garfinkle–Hillar (2016), https://arxiv.org/abs/1606.06997, treats noisy sparse dictionary uniqueness with oversized dictionaries under its hypotheses. A bridge from their loss and assumptions to expected-L0 BPD gate minimizers is outstanding.

## 5. Evidence and continuation

- Lean 4.19, Init-only, finite atomic examples and abstract mask equivalence are in OptimalBasis.lean; the repository CI must check axioms and kernel status.
- The source-only Python script finite_optima.py enumerates four exact-rational models, checks filtration rank at each mass threshold, and computes relative optimum blocks. Unit tests guard all four models and the rotated nonoptimal reference.
- The general finite vector-matroid greedy theorem, anchor-independent invariant-block theorem, and real norm inequality are **paper derivations**, separate from the Lean witness certificates.
- The learned BPD claim remains a hypothesis with unchanged empirical gates: pinned VPD parity and phase-disjoint control while training the dictionary. The broken Haar read-rotation null and full-rank gate nonidentifiability remain independent obstacles.
