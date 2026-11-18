# BPD-EFD-01 × LOOPSCOPE-10B: a native causal null at small energy-tangent dose

**Post-result cross-program technical note, 18 November 2026.** Original EFD-01 source-only geometry instrument and its prospective source-only claims remain intact. This documents the distinct [original LGA-07B native evaluation](https://github.com/MoralityLabAI/LoopedTransformers/blob/main/mechinterp/loopscope10/RESULTS_LGA07B_FIRST_NATIVE_20261118.md), frozen before opening its new 31060 source and independently audited [in Actions #44277352248](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44277352248). **Registered first native development verdict: `DEVELOPMENT_NULL_OR_INELIGIBLE`.** Three old teacher seeds 103/211/307 remain development-only; no BPD factorization or protected native pretrained test is newly confirmed.

## What was tested

At one physical goal-token h[k,j] in each old 44,001-parameter tied four-pass model, the source-reachable donor difference u was projected into

`K_j = ker(W_head) ∩ ker(∇_{h_j} E_shared(h))`.

The head has rank ten and the energy gradient removes one further independent physical dimension, so `dim(K_j)=37` in **all 1,536 native k2 source-goal cases**. The selected donor direction was normalized to `rho=.25||F(h)[j]-h[j]||`. Its matched comparator was the within-source median of eight isotropic random directions from this *exact same K_j* at identical physical dose. Every original case passed native geometric admission. The alternative donor target was fixed by an actual legal input predecessor-digit change, independent of the model's selected class.

Across 64 independent source strings per teacher, eight nested goal slots and true continued native k2→4 loops, the paired donor-correctness advantages were +0.390625, 0 and +0.585938 percentage points for seeds 103/211/307. Registered engineering minimum +15 pp failed all three. All source-only and independent no-Torch audit jobs passed; scientific verdict remains negative. The exact first uncompressed test is SHA256 `1b2c53f5175dd95fef8aed06a38aecc36950ebbf946c4a0819706f5b51171c64` and the first independent audit is SHA256 `993a573efc86a68cc74330e102e91fca432a01b468874f0d18ac79fefc833148`.

A natural **single goal-token** donor replacement obtained donor-target correctness in 336/495, 320/489 and 320/501 clean-competent pairs respectively (67.88%,65.44%,63.87%), missing its preregistered 95% positive-control requirement. Median `||P_K u||/||u||` was 0.329/0.345/0.337; median natural donor-token dose was 7.51/7.39/7.22 times the matched experimental radius. The selected gradient-tangent intervention did lower donor-target CE by 0.248/0.252/0.281 nats over the median matched K sham, a secondary continuous observable; exact new correct digit recovery stayed near zero. These are nested observations in only three exposed model roots, and the CE comparison is descriptive/post-result with no familywise correction.

## Why this changes BPD causal instrumentation

The first original native causal failure is a **specific local transfer and physical-dose limit**, rather than evidence that a learned scalar energy captures all algorithmic computation. A directional derivative witness `rank(DG_future|K)>0` and a continuous target-logit shift may coexist with practically zero exact donor-target transfer. More importantly, even the natural larger-dose one-token positive control failed to transplant the full donor outcome.

For source-reachable multi-token donor states, one can define an exact physical global tangent fiber without building a dense 3888×3888 identity matrix. Let `N=81`, `d=48`, `W_head∈R^{10×48}`, and `H∈R^{N×d}`. The current full-head Jacobian is block-diagonal in token position, with rank `81 rank(W_head)=810`. Define the row-null projector `P_H` in the 48D physical channel space and apply it independently to every token:
 
`U_0 = (H_donor - H_recipient) P_H`, `G_0 = ∇E(H_recipient) P_H`.

The **whole-state head-and-energy-tangent** source direction is then

`U_K = U_0 - (⟨U_0,G_0⟩/||G_0||_F²) G_0` when `||G_0||>0`, with the zero-gradient case handled separately.

This is a projection onto `ker(D head_all_tokens) ∩ ker(dE)` in the physical Euclidean geometry. Its expected dimension with independent energy row is `81(48−10)−1 = 3077`. Actual rank, singular conditioning and projected norm must be measured on each new source state. A physical BPD factor-space projector can be compared only after appropriate gauge transport; arbitrary individual V/U component masks cannot substitute for this whole-state physical statement.

## Next prospective design requirements

Any new evaluation should use an entirely new source roster and a separate registration committed **before** constructing that source. Its key **positive control** is the entire naturally reached donor state substituted into the recipient recurrence at depth k=2; by deterministic parity this must recover the donor model's original clean four-pass output, subject to finite numerical closure. Require source-specific true donor correctness, ideally ≥95% on properly jointly competent cases. Treat full-state donor replacement as a natural-dose oracle, then compare (a) physical globally energy+head-null projected donor, (b) physical head-only-null donor, and (c) equal-L2 random globally tangent physical shams at a fixed natural donor dose, recording finite head/E changes. A smaller dose can serve as a separately registered secondary dose–response control. Source-local predecessor/goal two-token interventions are useful diagnostic site controls, with all chosen supports frozen before correctness scoring.

A positive full-state dark-fiber contrast would establish context-specific, native **task-relevant delayed observability** of a physical tangent subspace in the fixed 44K family. It would require larger independent teacher roots, EvalMath variation/sham calibration and protected confirmation to generalize. The FSI/SF05 nonlinear finite-radius caution remains: exact first-order head and energy tangent conditions do not guarantee finite preservation.

**Evidence boundary:** LGA-07B did *not* use native trained BPDModel/TiedComponentLinear factors or validate compression, neuron communities, global Hodge decomposition, or an internally computed objective. The original EFD-01 mathematical tests remain source-only.
