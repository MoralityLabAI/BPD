# BPD-ECA-01/02: Sufficient-statistic geometry versus native causal mechanism

**Research note, November 18, 2026.** Companion to the merged [cross-program synthesis](BPD_ECA_CROSS_PROGRAM_SYNTHESIS_20261118.md), [BPD-ECA-01](https://github.com/MoralityLabAI/Control-Harness/pull/55) and [BPD-ECA-02](https://github.com/MoralityLabAI/Control-Harness/pull/56).

## Scientific distinction

The finite ECA-04 GRU benchmark supplies gold sentinel labels and three possible noise regimes. For each regime, the false-positive rate is the complement of sensitivity. Every report has probability `t_k` of matching its gold label under regime `k`, regardless of label polarity. For a fixed sequence of six events, the entire likelihood is

```
L_k = t_k^m (1-t_k)^(6-m),
m = number of reported values matching their gold label.
```

The exact three-regime posterior log odds are affine in the single scalar `m` at fixed depth. Over the seven possible counts, their centered 2D log-odds array has singular values approximately **27.7151** and **1.65×10⁻¹⁵**. This is an *exact sufficiency and target-rank* fact about the finite synthetic channel. A learned recurrent implementation can encode that scalar in distributed coordinates and nonlinear interactions.

The question for mechanistic interpretation is whether a physical linear subspace extracted from the frozen GRU **causally controls** its later native computation in a distinctive way, beyond edits along random subspaces at equal displacement. Decoding a sufficient statistic does not, on its own, identify its causal implementation.

## Registered BPD-ECA-01: overcomplete rank two

The first development experiment fitted a rank-two ridge decoder from the native GRU's sixth hidden state to both exact posterior log odds. The same donor and source six-step trajectories defined a model-consistent donor-prefix/source-suffix target. The candidate physical state edit was compared to eight rank-two Gaussian random projectors of exactly matched L2 dose.

| Quantity | Source-frozen first development run | Independent CPU refit |
|---|---:|---:|
| Candidate minus median random sham | 0.11357 | 0.04983 |
| 95th-quantile/95%-confidence sham variation floor | 0.53357 | 0.48509 |
| EvalMath confirmation gate | **REDESIGN_OR_PILOT_ONLY** | **REDESIGN_OR_PILOT_ONLY** |

The extracted second direction was not constrained by the intrinsically one-dimensional supervised target. In an independent CPU replay of the same source-registered roots the projector hash changed, causing a maximal per-root intervention score drift of **0.16454**. This source-locked result and the ungenerated confirmation cohort are preserved.

## Registered BPD-ECA-02: physical rank one

The second experiment froze a scalar match-count ridge decoder and saved the actual physical float32 16×16 rank-one projection matrix, then sampled new independent roots. The [original readout](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44239424896) used 1,024 training roots, 64 sham-calibration roots, 96 development-pilot roots and **zero** protected confirmation roots. The candidate donor-prefix model counterfactual and eight random rank-one controls had the same source/donor pair and precisely matched physical L2 intervention dose.

| Pre-registered measurement | Rank-one development outcome |
|---|---:|
| Gold match-count train MSE | **0.0194358** |
| Train-half squared subspace overlap | **0.988918** |
| Required train-half overlap | **0.990000** |
| Candidate mean donor-prefix logit restoration | 0.104235 |
| Median random-sham restoration | 0.030316 |
| Candidate minus median sham | **0.073919** |
| Separate 95/95 sham-variation floor | **0.360384** |
| Candidate above median sham | 91/96 |
| Candidate above every sham | **23/96** |
| Max L2 dose mismatch | 5.96×10⁻⁸ |

The rank-one probe removed the earlier *numerical* ambiguity. The [independent archived-projector replay](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44239887030) verified nine source hashes, seven original file hashes, and all 160 source/donor root-pair replays with 1,280 random shams. It reproduced the candidate effect within **1.9×10⁻⁸** on the mean and every per-root score within **6.1×10⁻⁶**. Refitting on the same 1,024 roots produced projector overlap 1.0000.

Its independent **512/512 training-half overlap**, however, remained **0.988918**, below the registered stability threshold. The 96-root causal advantage **0.073919** was also smaller than the calibrated sham floor **0.360384**, so strict EvalMath returned **REDESIGN_OR_PILOT_ONLY**. The planned protected 128-root confirmation was never sampled.

The result is a useful negative constraint: **the exact Bayesian sufficient-statistic dimension and a stable physical state projector do not guarantee a distinctive native causal effect at the tested step, intervention dose, and model-internal target.**

## Implications for the broader BPD, VPD and compactification program

**Causal instrumentation.** The fixed rank-one direction is reproducible, so the remaining failure is scientific: physically matched random directions can produce substantial restoration of a donor-prefix native-logit target. A follow-up must select a more discriminating *task-grounded* endpoint with the candidate chosen on disjoint training roots, evaluate correct authorization changes against host-independent labels, and retain random/donor/zero-patch controls. Re-running the same target with more roots cannot replace a failed mechanistic variation test.

**Representation theory.** The fixed-depth `m` match count supports a one-dimensional sufficient statistic. At variable depth, the pair `(n,m)` supplies the sufficient history counts if time is not provided separately. Under genuinely noncomplementary channels, two likelihood counts may be needed (for example the true-positive and false-positive report counts under separately varying `t` and `f`). This suggests a new *designed* rank-transition gym for Compactification-Theory, coupled to exact Bayes and a state-update closure test.

**Formal mathematics.** The rank-one likelihood factorization is a natural Lean target in MathCan. It can be stated independently of the GRU: under `f=1-t` and a known prefix length, likelihood depends only on the match count. The corresponding mechanistic learning claim remains a measured causal hypothesis, not a consequence of the Lean theorem.

**EvalMath.** Both attempts validate the need for separate independent sham-calibration roots, a 95/95 tolerance floor, and an explicit pre-data practical effect. A positive mean effect or frequent wins against the *median* sham leaves mechanism specificity unresolved when an occasional same-dose random direction produces comparable or greater change. The independent unit remains the issuer root, with intervention repetitions and the fifteen action contexts nested within it.

## Next source registration

A future **BPD-ECA-03** should prefer a *behaviorally grounded* endpoint over model-self-counterfactual restoration. Register an independent authorization/correctness target, two-direction and rank-one ablations, a full native step-by-step closure check, and a pre-data sham-discrimination floor calibrated on a fresh cohort. Consider an alternative noncomplementary-channel world where posterior rank truly grows, and compare match-count-only and two-statistic observers. All protected BPD test seals remain unchanged.

### Links and evidence boundary

[Original BPD-ECA-01 CI](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44236416558) · [BPD-ECA-02 first pilot](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44239424896) · [Rank-one independent replay](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44239887030) · [EvalMath mechanistic-resolution skill](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power)

The issuer roots, gold labels, HMAC keys and noise channels are synthetic. These results are narrow development-only causal investigations of a frozen small GRU. They establish neither real-world authority verification nor generalization to a larger looped transformer.
