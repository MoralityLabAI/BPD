# BPD-PD-11 | Global pair-angle search to escape mixed-read local extrema

**Prospective registration: November 18, 2026.** [CONFIG.json](CONFIG.json) was committed first, before any BPD-PD-11 training, source sampling, or experimental outcomes. Preserve the source lock, original data, and scientific PASS/FAIL even when the registered hypothesis fails.

## Hypothesis

[PD-09](../pd09/RESULTS_20261117.md) and [PD-10](../pd10/RESULTS_20261117.md) found that the unlabelled Gaussian smoothed-\(L_0\) dictionary objective can recover two exact latent axes yet remain trapped in a near-45° mixture of the other two. Narrow \(\sigma=0.15\) improves the *global* coordinate solution's objective but leaves the 45° mixture a local maximum. PD-10's cold-start gradient descent recovered 2/24 mechanisms; the original warm-start schedule recovered 0/24. Even the privileged 42° pair-mixed initialization could not escape its basin under gradient descent.

PD-11 tests whether searching **nonlocally over the full angle of a pair of orthogonal read directions**, rather than making local angular updates, can escape this mixed-read attractor. The primary and matched comparator are unsupervised: their optimizer takes **only observed dense \(x\)**, the current 4×4 orthogonal read matrix, and the registered smoothing parameter \(\sigma=0.15\). Neither latent source support/count labels nor the true rotation \(Q\) enter their training objective or read update.

## Objective and deterministic Givens updates

The physical teacher remains a shared bias-free \(W=I_4\). Observed row vectors \(x=aQ^\top\) use the unchanged PD-06 sparse latent generator; Q is independently sampled once per model seed using Gaussian QR. At all times the learned read basis V is orthogonal, and its physical reconstruction is \(VU=I\), with \(U=V^\top\).

For a pair of learned read columns \(V_i,V_j\), denote their current activations \(u=xV_i,v=xV_j\). Every candidate angle \(\theta\) replaces those columns by

\[
V_i'=V_i\cos\theta+V_j\sin\theta,\quad
V_j'=-V_i\sin\theta+V_j\cos\theta.
\]

Because all other read columns are unchanged, the objective can be compared using only the new pair's Gaussian near-zero mass:

\[
S_\sigma(\theta)=\mathbb E_\mathrm{train}\left[
\exp\left(-\frac{(u\cos\theta+v\sin\theta)^2}{2\sigma^2}\right)
+
\exp\left(-\frac{(-u\sin\theta+v\cos\theta)^2}{2\sigma^2}\right)
\right].
\]

Choose the maximum-scoring candidate angle, with deterministic tie-breaking that favors the smallest absolute update, then lower signed angle. Update pairs in the fixed order \((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)\). Make **six complete sweeps**, exactly 36 pair optimizations per arm. All pairwise comparisons use the *same fixed 1024-board training sample* per model seed, containing four independent passes (4096 observed vectors). This is a derivative-free optimizer. It has no first-step gradient requirement; audit the number of evaluated candidates, the total nonzero angle updates, orthogonality and the initial/final near-zero score.

### Matched angular-search arms

The **global_givens** primary searches exactly 73 angles over \([-90°,90°]\) inclusive, with step \(2.5°\). The **local_givens** comparator searches exactly 73 angles over \([-5°,5°]\) inclusive. Both include zero angle; both evaluate the **same number of candidates**, use the same initial random QR read basis, see identical observed source samples and execute the same ordered 36 pair updates. Different angular reach is the intervention under test.

Two **privileged stress arms** begin from the true rotated source axes with the first two columns deliberately mixed by **44°**. The global and local angular search rules are otherwise identical. They measure whether a known mixed-pair trap can be escaped under a nonlocal rotation. Because their initialization uses true Q, their result is explicitly excluded from unlabelled recovery claims and from the preregistered primary comparison.

The **random_orthogonal** negative control retains the matched random QR basis without updates. A separate **count_supervised_positive** arm repeats the unmodified PD-06 privileged count-supervised 360-step Adam factor training on the fresh source. This is a pipeline solvability control, not part of the unlabelled comparison.

## Original source and held-out transfer

Use the original PD-06 \`rotation_for_seed\` and \`sample_batch\` functions unchanged. Training samples supports with k=1,2,3 among all four singletons, four cycle pairs and three triples. Latent nonzero amplitudes have independent signed values with magnitudes uniform in \([0.75,1.25]\). The **unseen diagonal pairs** (0,2), (1,3), the **withheld triple** (0,2,3), and the four-pass unseen-support transition are excluded from every training sample and remain in the original six-family test set.

The 24 new model/rotation seeds are 1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831. Excluded engineering seed **2029** has no overlap. A seed's training sample generator is 100000+seed. Only *after* all basis updates finish, generate each of six held-out families from 200000+seed+10000×family-index, with 192 four-pass trajectories per family. The seed plus independent rotation is the statistical unit.

## Physical component evaluation and scientific gate

Install each final V and U=Vᵀ into the actual shared \`BPDModel/TiedComponentLinear\` 4D identity teacher. Zero the residual delta mask; zero-ablation semantics apply, with sphere rescaling disabled. At each of four tied passes, execute exactly the binary component gate

\[
D_j(x)=\mathbf1[\,|xV_j/\|V_j\||\ge 0.5\,].
\]

There is **no post-optimization gate training or threshold selection**. Match learned read directions to the hidden source rotation Q with one globally optimal absolute-cosine permutation over all four read columns, shared across every held-out family. Record every matched cosine, full physical output MSE, exact complete-four-pass support, and false-off/false-on counts by family.

A trained arm succeeds for a model seed only if mean matched read cosine≥0.90, mean total MSE≤0.03, each *seen* family's MSE≤0.04 and full four-pass exact support≥0.80, and each *unseen* family's MSE≤0.06 and full four-pass exact support≥0.70. Physical masks must be binary, trained Givens arms must have at least one nonzero angular update, and the physical factorization must remain orthogonal. The random negative is reported separately without an update requirement.

The **preregistered primary claim** is that global_givens beats local_givens under these exact criteria. PASS requires **at least 18/24** complete global successes, **at least six net paired wins**, and exact two-sided **McNemar \(p\le0.05\)**. Preserve the first-run FAIL unchanged if any criterion fails. Report seed-level Wilson 95% intervals, the entire paired contingency table, source hashes and exact evaluation counts. The global and local arms have the same 36×73=2628 candidate-angle evaluations per seed.

## Scope and evidence protection

The full-angle grid and source-specific small dimension are explicit inductive biases. A result in this experiment concerns global-vs-local coordinate search on a **four-dimensional orthogonal synthetic identity teacher** under the particular Gaussian smoothing surrogate. It cannot establish polynomial-time unsupervised discovery in arbitrary dimension, a general global expected-\(L_0\) optimizer, or mechanistic identification of the 7M Sudoku TRM.

Only the named count-supervised and true-Q pairmix controls may access privileged source information. Excluded-seed smoke tests verify QR factorization, matched initial conditions, exact candidate-grid sizes, label isolation, monotonic pair-score updates, physical BPD masked execution and forbidden support exclusion. A single GitHub Actions workflow begins after the source files are committed and frozen by one execution lock. Archive the complete first-run JSON even if its scientific gate fails.
