# BPD-PD-17: radial camouflage, exact-zero identifiability and a source-observable angular control

**Prospective protocol: November 18, 2026.** The CONFIG file was committed FIRST at 2bdea40b83f60f17d9af2f396c69a6d62a0952c2, before implementation or any of the 32 registered source rotations. This protocol freezes the source, estimator, causal endpoint and preregistered interpretation. Preserve every original PASS or FAIL.

## Scientific question

PD-15 recovered all 32 strongly clustered 16D mechanisms using exp(-||x||²) norm weights at fixed sigma .15, versus 0/32 for the unweighted search (original run 44233123380). Its population support law draws one active latent coordinate 25% of the time. Exponential reweighting shifts the effective singleton share to 63.73%, with an effective sample fraction 48.43%. PD-16 independently tests weight reassignment against a shuffle with identical weight histogram and ESS; its outcome is a separate source-locked experiment.

PD-17 tests a different possible confound. Does norm weighting lose its advantage when radial energy is independent of support cardinality, despite the same source supports and identifiable latent Q? A source-blind angular-ray comparator measures whether the underlying components remain accessible to a different unsupervised method.

Related recurrent evidence motivates the scope boundary: NC-03 found gauge-covariant projectors in constrained models; NC-04 showed matched-sham effects while missing the absolute keep-only NLL gate; LOOPSCOPE-08 found a shared energy descending over four actual TRM loops without reliable matched-dose terminal steering; ROC-05B failed against physical source-rule edits. Accordingly, PD-17 tests sparse-source identifiability and four-pass physical masks, reserving true recurrent task causality for separately registered roots.

## Source laws and their mathematical difference

Both registered source laws share 32 fresh orthogonal rotations Q (seeds fixed in CONFIG), 2048 four-pass training trajectories per root, original 16D PD-12 support banks, source support sizes K uniformly in {1,2,3,4}, and anchored coactivation probability .85 for K>=2. All withheld antipodal pairs and stride-two triples/quartets remain absent from training. The source generator is the frozen PD-12 function imported through PD-15. For a given root, original and radial-camouflage laws use **the exact same training and heldout support masks and hidden Q**; the two laws are paired observations of one independent root.

**Original source:** independent Rademacher signs and nonzero magnitudes Uniform[.75,1.25]. For X=Q A and lambda 1, w(X)=exp(-||X||²) induces P_lambda(K=k) proportional to P(K=k)*phi(lambda)^k, phi(lambda)=E exp(-lambda A²). Hence input norm reveals K.

**Radial camouflage:** draw iid nonzero magnitudes B_i Uniform[.9,1.1], independent signs epsilon_i, and an independent per-observation scalar R Uniform[1.0,1.1]. Then for i in active support S,

A_i = R epsilon_i B_i / sqrt(sum over j in S of B_j²).

This gives **||X||²=R² independent of S and K** in the population. Each nonzero active magnitude exceeds .9/(2*1.1)=.40909, above the original fixed physical gate .375. Thus the privileged planted-Q oracle is eligible for exact masks under either law. Training amplitude RNG offsets are 710000 and 810000 plus seed; withheld amplitude offsets are 730000 and 830000 plus seed plus 10000 times family index. These source streams are pinned in CONFIG.

Both laws have the same ideal **exact-zero hyperplane-mass optimizer**. For a read m with latent-coordinate support T, and independent continuously distributed magnitudes conditional on S,

z(m) = P(m^T A=0) = P(S intersection T = empty).

For an invertible matrix of read columns, the nonzero determinant certifies a perfect matching of its columns to canonical coordinates. The exact-zero mass of each mixed column is bounded above by its matched singleton coordinate's mass. The strict gap is at least 1/64 per mixed column because each alternative singleton support occurs with probability 1/(4*16). Dividing by 16 reads gives a population exact-zero objective loss at least **1/1024** whenever any column is mixed. Signed coordinate permutation is the only optimal gauge. This statement concerns exact-zero probabilities, not arbitrary finite-sigma kernels.

## Observed-data estimators and four arms per law

| Arm | Identification input | Relative compute |
| --- | --- | --- |
| norm_givens | unchanged PD-15 weight exp(-||x||²), sigma .15 | 39,960 angles |
| uniform_givens | unchanged PD-15 uniform weight, sigma .15 | 39,960 angles |
| angular_atoms | unsupervised repeated ± ray frequency, first 1024 observed vectors against all 8192 | separately logged O(1024*8192*16) |
| planted_Q_oracle | privileged true Q, positive control | zero |

Both Givens arms use the **exact unchanged PD-15 fit_basis**: nine sweeps, all 120 lexicographic pairs, the same 37-angle global grid, same initialized Q-independent random QR dictionary, same 8192 observed pass vectors, exactly 39,960 candidate-angle evaluations and 327,352,320 candidate-by-vector scores per learned arm.

The label-free angular control normalizes each observed x. It counts directions with absolute cosine >=.99999999 for each of exactly the first 1024 source vectors against all 8192 vectors, then greedily selects 16 peaks by descending neighbor count (tie by increasing candidate index), separated from previous picks by absolute cosine <.999. Require at least 12 neighbors per selected peak; otherwise abstain and mark mechanism failure. The selected directions are polar-orthogonalized using SVD. The angular baseline has a distinct computational budget and is a **feasibility/identifiability comparator**, never the matched-budget primary arm. It uses no source labels, cardinalities or Q.

## Physical mechanism and heldout families

Use the actual PD-12 BPDModel/TiedComponentLinear rank-16 W=I identity teacher with four physical calls, decoder U=V-transpose, zero residual delta and no sphere rescale. For each source law, generate each of the eight original 96-board four-pass seen/withheld families. Use the same fixed .375 read threshold, exact binary component masks and one deterministic Hungarian alignment per model across families. The learned complete-mechanism endpoint is the ORIGINAL PD-15 rule: mean matched cosine >= .90, overall MSE <= .035, each family MSE <= .05, exact four-pass support >= .80 in every seen and >= .70 in every withheld family, all masks binary, nonzero learned updates (or 16 selected angular peaks), and orthogonality error < 1e-4. The oracle is exempt only from update requirements.

Both registered source laws are evaluated against their corresponding support/amplitude source; cross-law transfer of a frozen fitted basis is a separate descriptive outcome. A source oracle failure is treated as a failed source/evaluator admission control, not reinterpreted as a negative learned mechanism effect.

## Primary registered source-dependence hypothesis

Every root has complete-recovery booleans in both source laws for both norm and uniform Givens. Define root-level interaction

D_i = [success(norm,original)-success(uniform,original)]
    - [success(norm,camouflage)-success(uniform,camouflage)].

The single primary hypothesis is that the relative advantage of the norm-weighted optimizer decreases under radial camouflage. The frozen scientific PASS requires **all**:

1. At least **24/32** original-law norm-Givens complete mechanism recoveries, independently replicating PD-15.
2. At least **24/32** camouflaged-law angular-ray complete recoveries, demonstrating source-observable factor feasibility at an explicitly unmatched compute budget.
3. At least **30/32 roots** with original corr(K,||X||²) >= .70 and camouflaged abs(corr(K,||X||²)) <= .05, using true K **only after fitting**.
4. Mean 32-root interaction **Dbar >= .40**, and the lower 2.5th percentile of a fixed **10,000-repetition root-cluster bootstrap > 0**, bootstrap seed 170017.
5. Planted-Q oracle **32/32 in each law**, establishing physical evaluator admission.

The paired bootstrap resamples the full root with all its source laws and arms, preserving dependence. It is an approximate confidence interval, **not an exact randomization p-value**. The registered hypothesis may FAIL, including if norm weighting succeeds under radial camouflage. Report that transfer as meaningful negative evidence against the radial-cardinality explanation. Do not relabel a failed gate, change the roster, or use boards as independent statistical units.

Descriptive secondary outputs include both laws' successes, exact per-law McNemar contingencies, physical false-on/off masks, read cosines, MSE, effective sample size, radial-support correlations, angular selection costs and failed/abstained roots. The original 32 statistical units, rather than 64 law conditions or 256 arms, determine inference.

## Future recurrent experiment, separate registration

The follow-up to PD-17 needs independently trained physical recurrent models with real state-dependent tied updates and a fresh terminal task. Carry forward NC-04 absolute keep-only NLL fidelity <= .05 nats, gauge-covariant projectors, matched rank/norm/write-dose shams, clean/full-mask closure at every depth, physical source-donor reachability, and the EvalMath independent-model-root variation preflight. In light of LOOPSCOPE-08's native curl and weak matched-dose improvement, measure terminal loss after several remaining-loop horizons. This future recurrent gate remains **unrun** and must be registered and source locked separately.

## Execution/provenance

Excluded seed 4001 alone is available for engineering: verify exact matched supports, source-independent radius, lower amplitude bound, deterministic angular selection, original PD-15 Givens parity, oracle physical masks, root bootstrap and registered positive/negative decisions. Do not fit any registered seed during smoke. After engineering CI and math certificates pass, a single EXECUTE_LOCK.json pins the final source SHA and runs exactly **32 roots x 2 source laws x 4 arms = 256 original receipts**. Archive all failures, full JSON, source hashes, per-root interaction values and the original scientific verdict. PD-01 through PD-16 historical outcomes remain unchanged.
