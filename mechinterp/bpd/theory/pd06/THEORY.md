# PD-06 mathematics: rotated support hyperplanes and strict read-basis identifiability

**Analytic source-law argument, registered source dated 2026-11-17.** This note derives the exact idealized expected-L0 optimum of the prospective 4D source and separates that mathematical optimum from finite-update BPD training. The associated [Lean file](RotatedSupportMass.lean) checks the finite support weights and the exclusion of held-out support combinations. The full real-distribution and matroid exchange arguments here are paper derivations, not yet Lean 4.19 theorems.

## 1. Orthogonal rotations preserve the hyperplane-mass problem

Let Q be a fixed orthogonal 4×4 matrix drawn independently for each seed, and define observed row vectors x=a Qᵀ, with latent coordinate support T⊂{0,1,2,3}. For any nonzero read direction m in observed coordinates, write c=Qᵀm so that

\[
m^\top x^\top = m^\top Q a^\top = c^\top a^\top.
\]

Conditional on T, the active latent coordinates are independently drawn from distributions with a continuous density supported on [-1.25,-0.75]∪[0.75,1.25]. If any c_i is nonzero for i∈T, the linear form cᵀa vanishes with probability zero. Therefore

\[
z(m)=\Pr[m^\top x^\top=0]=\Pr[T\cap\mathrm{supp}(c)=\varnothing].
\]

The observed source has almost surely nonzero values in **all four** observed coordinates, because the random rotation contains no structurally zero entries. Its sparse support is visible only after applying the *unknown* inverse rotation. This is a fixed per-seed rotation, so the experiment tests transfer to new supports under the same rotation, rather than transfer to a new untrained orthogonal basis.

## 2. Training weights and the expected-L0 optimum

The training distribution samples k∈{1,2,3} with probability 1/3 each. It then uniformly selects among four singletons, four cycle pairs, and three triples. With common denominator 36, each singleton or pair support has weight 3 and each triple weight 4, as [Lean verifies](RotatedSupportMass.lean).

For latent coordinate reads:

\[
z(Qe_0)=z(Qe_2)=z(Qe_3)=\frac{19}{36},\quad
z(Qe_1)=\frac{15}{36}=\frac5{12}.
\]

The total hyperplane mass of the four true coordinate read directions is 72/36=2. Hence their expected execution cost is

\[
J_{\mathrm{true}}=4-\sum_{i=0}^3z(Qe_i)=2
\]

executed factors per pass. The expectation agrees with E[k]=(1+2+3)/3=2.

Every latent read with support size at least two has strictly less hyperplane mass than each constituent coordinate read. Indeed for i in the read support S and j another member of S, the singleton source T={j} has probability 1/12. It contributes to z(e_i), since i∉T, while it cannot contribute to z(S), since S∩T is nonempty.

Now choose a maximizing linear read basis and suppose it contains a mixed read c supported on S with |S|≥2. Since c is in the span of coordinate basis elements {e_j:j∈S}, linear basis exchange yields at least one such coordinate e_j that can replace c while keeping all basis reads independent. That replacement strictly increases the total hyperplane mass, contradicting maximality. Repeating this argument shows that every globally maximizing basis is the coordinate basis up to permutations and nonzero scaling. The same holds in observed coordinates after rotation by Q.

This exact population-optimum theorem assumes independent continuously distributed active latent amplitudes, projective nonzero read directions, a square invertible read dictionary, and an expected-L0 objective. It does not guarantee convergence of an Adam/STE optimizer, empirical identification from a finite sample, or clean causal intervention recovery in a nonlinear network.

## 3. Why the held-out support families matter

Training has all four singleton supports, four adjacent pairs and three triples. The two unseen pairs {0,2} and {1,3}, plus the unseen triple {0,2,3}, have zero training probability even though their corresponding latent dimensions and cardinalities are present in training. A learner can fit the 11 seen supports without discovering four reusable read factors. Exact support recovery on the withheld combinations therefore probes *compositional factor use*.

Six held-out strata are evaluated: singleton, seen pair, unseen pair, seen triple, unseen triple, and a sequential combination [singleton, unseen pair, unseen triple, unseen pair]. The same global factor permutation must explain every held-out stratum. High reconstruction alone is insufficient: the identity teacher can be approximated by an always-on, rank-four factorization with low MSE while failing sparse intervention identification.

A count-supervised arm gets only k labels, which cannot identify **which** of four dimensions are active. The support-supervised arm receives privileged coordinate labels and explicitly breaks permutation gauge. Oracle-support and planted-frozen controls delimit how much of the task is due to dictionary learning versus the learned gate.

## 4. Next formalization boundary

The Lean source verifies the exact finite mass values, that all six candidate two-coordinate read supports carry less mass than their respective singleton read supports, and that unseen training support combinations are structurally absent. Proving the full continuous zero-probability argument and its real vector-matroid basis-exchange consequence in Lean would require measure-theoretic and linear-algebraic imports beyond the present Init-only certificate. The existing [FiniteFiltration.lean](../FiniteFiltration.lean) and [GreedyRank.lean](../GreedyRank.lean) provide conditional parts of that formal path.
