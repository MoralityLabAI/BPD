# BPD-PD-16: permutation placebo for the source-observable norm-weighted objective

**Prospective protocol registration: November 18, 2026.** CONFIG.json was committed first at **225d6ab8dca1ad7b8676105446467655935e00cd**, before any PD-16 scientific-seed training. The exact source, 32 fresh source rotations, shuffle RNG, weights, 128 original arms, thresholds and decision rules are frozen in CONFIG. This first experiment will retain its scientific outcome regardless of PASS or FAIL.

## Scientific motivation

[PD-15](../pd15/RESULTS_20261118.md) passed its registered strongly clustered 16D recovery test: 32/32 complete physical mechanisms from the norm-weighted Gaussian near-zero-mass objective at lambda=1, sigma=.15, versus 0/32 for the equal-budget unweighted control. The explicitly predeclared objective-consistency gate passed 32/32.

That study showed a substantial joint change in the Gaussian objective and in **effective sample size**: fixed w(x)=exp(-||x||²) preferentially weights sparse low-energy observations in its original support law. PD-16 tests the more precise claim that the **alignment between each observation and its norm-derived weight** drives identification, rather than the mere presence of heterogeneous weights or the reduced effective sample size.

The primary comparison is norm_aligned against a **seeded randomly permuted placebo**. Both have the *exact same weight multiset*, ESS, nominal sample count, training source and optimizer cost. The placebo assigns those weights to different observed physical vectors through an independent, prespecified permutation. No source label, Q, cardinality, held-out result or learned model outcome enters the permutation.

## Unchanged source, 32 new independent rotations

Reuse PD-15's frozen source_for_seed and all PD-12 source/physical functions exactly: dimension 16, strong active amplitudes uniform [.75,1.25], source cardinality 1–4 with equal probability, clustered anchored contiguous supports chosen with probability .85 for k>=2, otherwise any allowed training support. The original withheld antipodal pairs, stride-two triples and quartets remain excluded from training. True hidden Q is a seeded orthogonal 16×16 Gaussian QR matrix. It is never an optimizer input.

CONFIG fixes **32 fresh seeds (3301–3539)**, each assigned once to the same original clustered strong-margin condition. Their seed roster is disjoint from all PD-12 through PD-15 scientific seeds. Excluded engineering seed **4001** is not in the cohort.

For a given seed, draw **one fixed 2048-board four-pass training source pool** with RNG 100000+seed, yielding N=8192 observed dense x vectors. All learned arms use this same x, the same initial orthogonal read basis from 610000+seed and the same source Q only when evaluating after training. Evaluation uses the original eight PD-12 support families, 96 heldout four-pass trajectories each, with the same independent family RNG.

## Exact controls and objective budget

Define w_n=exp(-||x_n||²) using observed physical x only. The three searches are:

| Arm | Observed training weights | Candidate angles |
|---|---|---:|
| **norm_aligned (primary)** | w_n assigned to its own x_n | **39,960** |
| **norm_shuffled (primary placebo)** | w_{pi(n)} for independent random permutation pi | **39,960** |
| uniform_control | all 1, original unweighted Gaussian search | **39,960** |
| planted_Q_oracle | privileged Q, no training | 0 |

The exact permutation is generated once per seed with **torch.randperm(N, generator=manual_seed(910000+seed))**, independently of any training-label information. The shuffled arm has the exact same multiset of positive weights, sum(w), sum(w²), ESS, and marginal weight distribution as norm_aligned, with the weight-to-example association destroyed. This random assignment is computed *once* per seed before the corresponding placebo search and never selected according to training objective or heldout metrics.

Each learned arm maximizes the same Gaussian weighted objective at fixed sigma=.15:

\[
J(V)=\frac{\sum_n w_n\,\sum_{j=1}^{16}
       \exp[-(x_n^\top V_j)^2/(2\sigma^2)]}
          {16\sum_n w_n}.
\]

Use exactly nine sweeps over all 120 unordered pairs in original lexicographic order. Evaluate 37 angular candidates across [-90°,90°] including zero and the original absolute-angle/negative tie break at tolerance 1e-12. The read factor V remains orthogonal after each 64-bit Givens update, and the decoder U=Vᵀ. Log the complete mass path, 39,960 candidate-angle evaluations, **327,352,320 candidate-by-observed-vector scores**, nonzero updates, weight ESS, shuffled permutation SHA256 and wall time. The uniform arm retains the exact unweighted PD15 implementation for parity. Weight computation and weighted reductions are separately accounted for.

## Physical heldout mechanism endpoint

Use the exact PD-12/PD-15 rank-16 physical BPDModel/TiedComponentLinear identity teacher and four physical calls. The source Q and support masks enter only post-fit read-axis Hungarian assignment and physical evaluation. The same assignment is used across every held-out family. The fixed strong read-gate threshold is **0.375**, the residual delta mask is zero, masks must be exact 0/1, and sphere rescaling remains disabled.

A learned seed counts as a **complete mechanism recovery** only if mean matched absolute cosine≥.90, mean heldout MSE≤.035, each family MSE≤.05, exact four-pass support≥.80 in every seen family and ≥.70 in every unseen family, every binary mask correct in type, at least one optimizer update and orthogonality error<1e-4. The privileged planted-Q oracle is exempt only from the optimizer update requirement.

## Primary paired scientific gate

The 32 independent seed+rotation units each have one norm_aligned and one norm_shuffled complete-mechanism boolean. A preregistered mechanism **PASS** requires all:

1. **At least 18/32** norm_aligned full mechanism recoveries.
2. **At least 10 net norm_aligned-only paired wins** versus norm_shuffled.
3. Exact **two-sided McNemar p≤0.05** using the 32 independent seed-level paired results.

The uniform_control and performance/cost details are secondary descriptive evidence, with no additional retrospective significance claim.

## Predeclared post-fit norm/cardinality diagnostic

Only after fitting all models for a seed, inspect the source generator's hidden training support masks to compute the actual per-vector active-cardinality k. For each seed, report Pearson correlation of k with the observed norm-derived **aligned weights**, and correlation of k with **shuffled weights**, as well as the absolute difference between the two ESS values. This is an independent audit of the placebo construction; no k, Q, signs or support masks are consulted by training or permutation generation.

The diagnostic gate requires **at least 30/32** seeds simultaneously satisfying corr(aligned weight,k)≤−0.70, abs(corr(shuffled weight,k))≤0.05, and exact aligned/shuffled ESS difference≤1e-10. This test checks that aligned weights carry the registered source-observable cardinality signal while the placebo has lost it. The full study scientific PASS requires **both this diagnostic and the primary mechanism gate**. Both will be reported separately.

## Compute and evidence

Excluded engineering seed 4001 verifies PD15/PD12 source parity, deterministic weight permutation, identical weight sorted multisets and ESS, label-free fit API, exact angular and scalar budget parity, monotone objective, orthogonal factors, physical 0/1 masks, privileged oracle success, source hash ancestry and synthetic seed-level PASS/FAIL logic.

The CONFIG and this PROTOCOL must be committed before implementation. Once excluded-seed CI and Lean accounting are green, commit one EXECUTE_LOCK.json with a protected source SHA. A dedicated GitHub Actions workflow then executes exactly **32×4=128 original arms**, retains every source and model receipt and archives the complete original scientific JSON whether it passes or fails. Protected scientific source files and older PD-01–PD-15 outcomes remain untouched.

This study isolates the weight-to-example association for a **specific artificial orthogonally rotated sparse-linear identity teacher**. A PASS would support the source-proxy explanation of PD-15 within that generator. Generality to other source distributions or learned nonlinear recurrent transformers remains a separately testable question.
