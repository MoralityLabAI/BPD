# BPD-PD-12 | Scaling nonlocal Givens search to 8D and 16D

**Prospective registration, November 18, 2026.** [CONFIG.json](CONFIG.json) was committed at c1c49a6f924edfdfb54b5fa7c0cebae6f367a1c5, before any PD-12 model data. All 64 new seeds, eight factorial cells, three trained arms, the planted-Q oracle, source laws, costs and pass criteria were frozen there.

## Research question

[PD-11](../pd11/RESULTS_20261118.md) recovered 24/24 four-dimensional source dictionaries with full-angle Givens search, against 1/24 for equal-budget local-angle search. PD-12 tests whether this transfers to **8 and 16 dimensions**, to a **clustered coactivation law**, and to a **weaker minimum nonzero amplitude**. Candidate angle evaluations, observed sample volume and CPU wall time are measured alongside full physical intervention recovery.

Every source has an independently rotated latent vector x=aQᵀ where Q is an orthogonal d×d matrix generated with seed 900000+model_seed, d=8 or 16. The active latent cardinality is k∈{1,2,3,4}, sampled uniformly each pass. Each active coordinate has a random balanced sign, with amplitude uniform on [.75,1.25] for the strong-margin condition or [.25,1.25] for the weak-margin condition. The model receives only dense observed x. The exact latent support and true Q remain evaluation data, except that the explicitly privileged planted oracle uses Q for its basis.

## Frozen eight-cell factorial and support distribution

The Cartesian product is d∈{8,16}, sampling∈{uniform,clustered}, margin∈{strong,weak}. Each of the eight cells has eight distinct model seeds and rotations, yielding **64 independent statistical units**. Each seed occurs in one cell only; all 64 are listed in CONFIG.

For each k, the uniform training source chooses a support uniformly among all k-subsets of {0,…,d−1}, excluding preregistered held-out combinations. The clustered source uses the same training support bank. For k≥2, it selects a contiguous cyclic k-window anchored at indices divisible by four with probability 0.85, drawing from the full training support bank with probability 0.15. Singleton supports remain uniform. Both source laws retain positive singleton probability, making exact population expected-L0 coordinate-read identification meaningful.

Withheld k=2 supports are antipodal pairs {i,(i+d/2) mod d}, deduplicated. Withheld k=3 supports are stride-two triples {i,(i+2) mod d,(i+4) mod d}; withheld k=4 supports are stride-two quartets {i,(i+2) mod d,(i+4) mod d,(i+6) mod d}. These are excluded from every training bank and sampled only in evaluation. Exact support-set disjointness is a source-only regression gate.

Train on 256 four-pass trajectories, providing 1024 observed read vectors, using RNG 100000+seed. The eight evaluation families each use 96 independent four-pass trajectories, RNG 200000+seed+10000×family_index: seen k1, k2, k3, k4; withheld k2, k3, k4; and a transition with seen singleton then unseen pair, triple and quartet. The held-out generator is never opened during training or pair-angle selection.

## Frozen unlabelled pair-search algorithms

Each arm starts from an identical random d×d orthogonal read matrix produced by deterministic-sign QR from a Gaussian matrix with generator 610000+seed. The physical write matrix is its transpose U=Vᵀ. For pair (i,j), let u=xV_i and v=xV_j. At each candidate θ, score:

\[
S_\sigma(\theta)=\mathbb E\big[
\exp(-\tfrac12((u\cos\theta+v\sin\theta)/\sigma)^2)
+\exp(-\tfrac12((-u\sin\theta+v\cos\theta)/\sigma)^2)
\big].
\]

The source-only optimizer maximizes this smoothed near-zero mass with σ=.15 for strong or .05 for weak margin. The Givens update preserves orthogonality and only changes the two selected read columns. All unordered column pairs are visited in lexicographic order for three sweeps. Tied scores within 1e−12 prefer smallest absolute angle and then smaller signed angle. Every candidate grid contains the zero-angle no-op.

| Arm | Angle range | Angles / pair | Candidate angle scores in 8D | In 16D |
|---|---|---:|---:|---:|
| **global37 primary** | −90° to +90° | 37 | **3,108** | **13,320** |
| **local37 equal-budget control** | −5° to +5° | 37 | **3,108** | **13,320** |
| global13 cost-sensitive secondary | −90° to +90° | 13 | **1,092** | **4,680** |
| planted_Q_oracle privileged positive | true Q supplied | 0 | 0 | 0 |

Global37 versus local37 is a strictly matched **candidate-evaluation-budget** comparison. Global13 is separately reported at 13/37, or about 35.1%, of that cost. All three trained arms use the same 256 training boards and identical starting factor matrix for a given seed. Candidate angle scores and sample-weighted scoring operations are recorded separately from CPU wall time. The 16D high budget has 4.29 times the candidate angles of the 8D high budget, as determined by the d(d−1)/2 read pairs.

## Physical BPD intervention evaluation

Install V and U=Vᵀ in the actual shared BPDModel/TiedComponentLinear physical identity target with d rank-one factor components and four tied passes. Apply zero ablation, disable sphere rescaling and zero the residual delta mask. The frozen binary execution gate is

\[
D_j(x)=1[|xV_j/\|V_j\||\ge t],
\]

with t=.375 strong or .125 weak. No gate is trained or calibrated. An exact deterministic assignment algorithm aligns all learned read columns to Q by maximum absolute cosine under **one global permutation** across all held-out families. Report mean/minimum matched cosines, MSE, exact four-pass support, false-on/off counts, objective gains, candidate evaluations and execution time.

A seed counts as a complete recovery iff mean cosine≥.90, overall MSE≤.035, each family MSE≤.05, every seen-family four-pass support≥.80, every unseen-family support≥.70, every physical mask binary, and the trained search arm makes at least one nonzero pair update. The oracle positive control is exempt only from that final training-update condition.

## Preregistered primary scientific test

The primary claim is **global37 superiority over equal-budget local37** across the full eight-cell factorial. A scientific PASS needs all of:

1. Global37 succeeds in at least **40/64** independent seed+rotation units.
2. Within 8D, it succeeds in at least **24/32**; within 16D, at least **12/32**.
3. It gains at least **16 net paired seed successes** over local37.
4. Exact two-sided seed-paired McNemar p≤.05.

All 64 paired seeds remain counted. Report each eight-seed cell separately with no cell-level significance claim, plus dimension-pooled Wilson 95% intervals, and the full 64-unit paired contingency table. Global13 measures the cost/accuracy frontier as a secondary, explicitly unregistered-for-significance comparison.

## Limits and first-run discipline

The original exact expected-L0 source optimum has rotated coordinate reads because each latent singleton has positive mass. This does not guarantee successful finite-sigma search, adequate sample sizes, or strong-margin transfer at d=16. The source remains a specially constructed orthogonal linear sparse-latent identity teacher.

Source-only engineering uses excluded seed 3001, checking support disjointness, intended correlations, amplitude floors, exact BPD binary execution at both dimensions, identical within-seed initializations, candidate budgets and unsupervised label isolation. After tests pass, a one-time EXECUTE_LOCK.json pins CONFIG, PROTOCOL, implementation and test files. The original 256-arm result is archived unchanged even when the scientific gate fails. These outcomes do not establish mechanism identification in a learned recurrent transformer.
