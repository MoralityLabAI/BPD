# BPD-SF03 — signed-gradient-matched dark-fiber null and layerwise source sensitivity

Prospective **2026-11-17**, separately registered after [SF02](https://github.com/MoralityLabAI/BPD/pull/34). Research question: does a calibrated BPD component coalition cause secret-recipient transfer effects **beyond local future-readout first-order sensitivity**?

## Source, tuning exclusion and scope

Reuse frozen [SF02 teacher/factors/selection source](../sf02_secret_circuit/experiment.py) without changes; use the same nonlinear two-step shared-weight physical teacher, source-role-conditioned synthetic secret task, 12 full-rank jointly learned BPD components and development-greedy six-component selection. SF02 roots (1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051) and previously used engineering seeds **901, 902** are excluded from both SF03 calibration and confirmation. Gradient control construction was piloted only on excluded 901 and 902 to verify code geometry and establish that a tangent-matched control is powerful; no SF03 calibration or confirmation results have been examined yet.

Set **59 independent calibration roots: 4101..4159 inclusive**, then after writing a one-time threshold from their results freeze **32 independent confirmation roots: 5101..5132 inclusive**. Each root retrains an independent physical teacher for 850 updates and a separately initialized BPD dictionary for 400 updates. All task families remain drawn from the *same* continuous scalar source generator, so the confirmation is **root/seed-disjoint, not source-family-disjoint**. Models are synthetic numerical networks, not natural-language secret-keeping systems.

## Stronger matched-sham construction

At the first post-transition state `h_1` let `q(h)=h_0` denote the actual current readout. Let `K=ker Dq` and let `δ∈K` be the selected six-component donor-minus-recipient BPD write, projected to preserve the complete current readout. The measured one-step future output is `Y=q(F(h_1+δ))`. Define the dark future-readout gradient `g=P_K ∇_h(q∘F)(h_1)` and unit `u=g/||g||` when `||g||>0`.

Decompose `δ=αu+δ_{\perp}`, where `α=u·δ` and `δ_⊥∈K∩u^\perp`. Sample seeded random `z∈K∩u^\perp` with `||z||=||δ_⊥||`. The **signed-gradient matched** sham is

\[
\eta_{\rm signed}=\alpha u+z,
\quad \|\eta_{\rm signed}\|=\|\delta\|,
\quad g^\top\eta_{\rm signed}=g^\top\delta.
\]

The **unsigned-gradient matched** control uses `η_unsigned=σ αu+z` for independent `σ∈{−1,+1}` with equal probability. It matches norm and the absolute gradient magnitude, while randomizing first-order sign. Both exactly preserve `q(h_1)`. For zero-gradient or zero-radius cases use a deterministic documented fallback and record degeneracy; treat any nonfinite values as engineering failures. Numerical tolerances: max norm error 1e-4, absolute/signed derivative error 1e-4, current readout equality 1e-6.

The signed sham is a *privileged local-linear predictor control*: it uses the BPD selected intervention's exact signed gradient effect. If it performs as well as the original BPD mask, the observed output transfer may be fully explained by first-order sensitivity. A null failure does not prove model mechanistic irrelevance; it limits the incremental explanatory claim.

## Fixed per-root assay

For each trained root, generate the same independent SF02 heldout stream (384 source-matched pairs, calibration 240), and apply the original readout-preserving selection. Authorized eligibility and recovery tolerance are **unchanged from SF02**: correct clean answer within .08, recipient/donor first readout within .04, clean donor difference ≥.20, exact donor readout recovery tolerance ±.10. Use **64** independently seeded signed-gradient matched sham arms, **16** unsigned shams, all against the same eligible pairs. These 64 arms are within-root repeated null directions; independent root **not pair nor intervention count** is the inferential unit. Track complete denominators and within-root distribution.

Let `R_B` be original six-component BPD recovery, `R_j` signed-sham recovery for j=0..63, and define

\[
G_B=R_B-\frac{1}{64}\sum_jR_j,\qquad
G_{\rm null}=R_0-\frac{1}{63}\sum_{j=1}^{63}R_j.
\]

The heldout root-level `G_B` asks if BPD provides additional predictive ability beyond the signed tangent. The pseudo-selected null `G_null` is a fixed preregistered sham-versus-shams contrast. Its 59 calibration-root values are independent samples of a null tolerance statistic.

## 59-root one-sided calibration and prospective confirmation

Construct `T=max_{i=4101}^{4159}G_{{null},i}`. For independent identically distributed roots, the maximum of 59 observations is an upper bound on the population 95th percentile with confidence `1-0.95^{59}≈95.15%`, assuming the calibration and confirmation null laws match. This is a *calibrated upper-tail tolerance*, **not** a direct exchangeable randomization p-value for BPD versus sham or proof of family-wide security. Preserve the empirical population conditions.

Lock `T` and the calibration receipt on the branch **before launching** 32 confirmation roots. Registered strong incremental mechanism criterion: at least **24/32** independent confirmation roots must satisfy `G_B≥T+0.05`, with factor physical reconstruction MSE ≤1e-4 and the three geometric numerical tolerances. The +0.05 effect floor is absolute 5 percentage points in authorized donor recovery.

Also report a separate descriptive **equivalence check**, never infer it solely from failing the incremental gate: root-mean gap `\bar G_B` and its two-sided 95% bootstrap interval from 32 independent roots (10,000 resamples, deterministic RNG). Evidence of effect equivalence within ±0.05 requires that entire interval lie in [−0.05,+0.05], and at least 28/32 confirmation roots have |G_B|≤0.05 and pass factor reconstruction. Explain failures separately.

## Layerwise source-Jacobian profile

For independently generated heldout inputs, evaluate the true full-state derivative of the first and second readouts with respect to the recipient's authorized private input. Report:

- `|∂q(h_1)/∂S_{\rm target}|` under roles A,B (early readout should be insensitive)
- `|∂q(h_2)/∂S_{\rm target}|` under A,B (later readout should encode authorized secret)
- `|∂q(h_2)/∂S_{\rm other}|` under A,B (source binding)
- `|∂q(h_2)/∂S_A|` under unauthorized U (private influence on public output)

Gradients are *local functional sensitivity*, not an information-theoretic secrecy guarantee. Condition public and authorized records on the same source law.

## Exact evidence boundary

This is a numeric derivative control on a tiny synthetic teacher. SF02 source-locked positive donor transfer and 7/8 engineering outcome remain intact. No Fable B1–B4, CQ-03/04, ARG-03-N sealed 168-case data, or real model tests are modified. Calibration and confirmation constitute a new independent-root prospective study. The 95/95 claim applies narrowly to the stated pseudo-sham null-root distribution conditional on reproducible execution. 
