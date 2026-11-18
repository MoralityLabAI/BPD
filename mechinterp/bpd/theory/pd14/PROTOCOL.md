# BPD-PD-14: source-locked bandwidth correction and matched Givens restart test

**Prospective registration, November 18, 2026.** CONFIG.json was committed first at **09a2034d7c905fe93777ea11b656990c894f2311**, before running any of the 32 PD-14 scientific seeds. The frozen first-run result remains a scientific PASS or FAIL regardless of its outcome.

## Rationale

[PD-13's original registered result](../pd13/RESULTS_20261118.md) was a scientific FAIL (priority2048 2/32, lex2048 2/32, planted-Q oracle 32/32). Its separately labeled [post-hoc population analysis](../pd13/POSTHOC_OBJECTIVE_INVERSION_20261118.md) found an explicit 45-degree anchored-pair remix whose Gaussian near-zero mass exceeds that of the planted Q at the PD-13 *strong*-margin sigma=0.15. The population ranking reverses around sigma=0.0688879 for amplitudes [0.75,1.25]. This is a demonstrable **objective inconsistency** at the previously registered strong width, independently of the observed optimizer failure.

PD-14 prospectively tests one predeclared narrower width, sigma=0.05 for strong and 0.02 for weak, and separates the bandwidth change from the finite-search problem through equal-budget nine-sweep and three-restart regimes. The strong cell is the single primary mechanism comparison. The weak cell and restart contrasts are secondary, without recycled primary significance claims.

## Source and independence

Exactly 32 new seeds and rotated 16-dimensional source models appear in CONFIG.json. Sixteen are designated clustered strong-margin, sixteen clustered weak-margin. All differ from PD-12 and PD-13 scientific seeds. Excluded engineering seed 4001 is forbidden in the scientific cohort.

Reuse the **unchanged PD-12** rotation_for_seed, sample_batch, support_banks, initial_random_basis, best_pair_angle, physical_model, basis_alignment and evaluate_family. Source latent supports have cardinalities 1–4, independent random signs, and the original 0.85 probability of one anchored contiguous motif when cardinality >=2. All withheld antipodal pairs, stride-two triples and quartets remain outside the training bank.

Every seed is generated once with 2048 four-pass training trajectories, RNG offset +100000. All four unsupervised arms see exactly this same dense observed x pool and the same initial QR basis for their first start. Their optimizer receives **no Q, latent amplitudes, support masks, family labels, target outcomes or held-out inputs**. Each retains the same physical rank-16 identity teacher with U=V-transpose and four tied BPD passes.

The eight PD-12 held-out families use 96 trajectories each and RNG 200000+seed+10000*family_index, generated after training for evaluation. The separate **source-objective diagnostic** generates 1024 fresh four-pass trajectories per seed from the same registered training support distribution with RNG 310000+seed. It uses the hidden Q only as a privileged post-fit reference, never as a search input. Both source and diagnostic samples are independent of the training pool.

## Matched candidate-angle budget and fixed widths

The four unlabelled learned arms form a 2x2 factorial at **identical 39,960 evaluated candidate angles per seed**, 327,352,320 candidate-by-observed-vector scores, on the same 8192 training pass vectors:

| Arm | Width strong / weak | Unlabelled optimization | Candidate-angle evaluations |
| --- | --- | --- | ---: |
| **wide_single (primary control)** | 0.15 / 0.05 | 9 lexicographic full-angle Givens sweeps from original QR | 39,960 |
| **narrow_single (primary)** | 0.05 / 0.02 | 9 matching sweeps from the same original QR | 39,960 |
| wide_restart | 0.15 / 0.05 | 3 independent QR starts, 3 sweeps each; choose best *training objective* | 39,960 |
| narrow_restart | 0.05 / 0.02 | Same three starts and selection rule under narrower width | 39,960 |
| planted_Q_oracle | privileged true Q | No optimization, physical positive control only | 0 |

Each sweep processes all 16 choose 2 = 120 pairs in the exact same lexicographic order and uses the frozen PD-12 global37 angle grid, [-90°, +90°] inclusive, zero-angle candidate and tie-break tolerance 1e-12. For all four unlabelled arms, V is maintained orthogonal by 64-bit Givens updates; the training objective is the mean of exp(-0.5*(xV/sigma)^2). No annealing and no adaptive sigma selection occur.

Single_long visits every pair nine times starting from the original deterministic-sign QR seeded 610000+model_seed. Three_restarts visits each pair three times from each of three deterministic-sign QR starts: r=0 original QR, r>0 seeds 610000+model_seed+1000003*r. It chooses the candidate with the maximum final **in-sample unlabelled Gaussian objective**; score ties within 1e-12 go to the lowest r. This may expose model-selection overfitting, so every arm uses an identical independent held-out evaluation.

Log every initial basis hash, selected restart, per-restart final objective, exact scores, orthogonality error, nonzero updates and wall time. The nine-sweep and 3x3 regimes have equal total candidate and scalar-score budgets, although QR initialization and final-selection overhead are extra in the restart regime and separately reported.

## Frozen causal mechanism evaluation

Use the PD-12 actual BPDModel/TiedComponentLinear wrapper, four physical pass calls, exact zero residual delta, no sphere rescaling and no gate fitting. Set read thresholds to **0.375 strong / 0.125 weak**, exactly as PD-12 and PD-13. Use one Hungarian read-axis assignment per seed and learned model across all eight families.

A learned arm recovers a model **only if** mean matched absolute read cosine>=0.90, overall MSE<=0.035, every family MSE<=0.05, every seen family exact four-pass component-support fraction>=0.80, every unseen family exact support>=0.70, all masks binary, at least one nonzero angle update and orthogonality error<1e-4. The planted-Q oracle is exempt only from nonzero optimizer updates.

## Predeclared primary mechanism gate

The **only primary statistical test** is on the 16 independently rotated clustered *strong*-margin seeds: narrow_single versus wide_single, identically initialized, trained on the same 2048 trajectories, each with exactly 39,960 evaluated angles. A prospective scientific mechanism PASS requires all of:

1. **At least 8/16** complete mechanism recoveries for narrow_single.
2. **At least six net narrow-only paired wins** above wide_single.
3. Exact **two-sided** McNemar p<=0.05, using only seed-level discordant pairs.

The weak margin, restart comparisons, MSE, read cosines, objective values, per-family supports and training times are registered secondary descriptive outcomes, reported by margin, with all 32 seed receipts. The original PD-13 data are historical context only.

## Independent fixed-remix objective diagnostic

For every new seed's independent 1024-board diagnostic sample, compute the Gaussian mass of the planted Q read basis and of an explicit Q × R basis, where R applies 45-degree orthogonal rotations to anchored latent pairs (0,1), (4,5), (8,9), (12,13). This is a **privileged diagnostic**, not a label-free training arm. Use both fixed widths for that seed's margin and keep both scores in the archived JSON.

The predeclared objective-consistency gate is: in at least **15 of the 16 strong-margin seeds**, wide sigma=0.15 favors the remix above Q while narrow sigma=0.05 favors Q above the remix. These dual signs are one prospective descriptive gate, **not** an independently significant second recovery claim. The study-level scientific PASS requires both the objective-consistency gate and the primary mechanism gate. Report the two separately even when one fails.

This diagnostic directly tests the population inversion observation on fresh source realizations and avoids using a full cohort's after-the-fact recovery threshold to decide whether the objective improved.

## Evidence and execution

Excluded engineering seed 4001 verifies shared source hashes, uniqueness of scientific seeds, independent diagnostic sampling, deterministic QR restart generation, no true Q in the optimizer signature, equality of candidate evaluation counts, monotone within-sweep Gaussian mass, exact physical 0/1 masks, source lock, and the two-stage scientific decision function.

Before the first run on scientific seeds, the preflight and Lean CI must pass. A single EXECUTE_LOCK.json then pins the final source commit SHA. The GitHub Actions registered workflow verifies protected-file ancestry and unchanged source blobs, installs Python 3.12, Torch 2.5.1+cpu and SciPy 1.14.1, runs tests and the first 32×5=160 arms, and retains the original immutable full JSON for 90 days, regardless of PASS or FAIL. A failed engineering gate is reported as such; it cannot be silently recast as a scientific FAIL or rerun with modified frozen decisions.

The source is a synthetic orthogonally rotated sparse-linear identity teacher. Even complete recovery would support only this controlled identification setting, not mechanistic discovery in a pretrained nonlinear recurrent transformer. 
