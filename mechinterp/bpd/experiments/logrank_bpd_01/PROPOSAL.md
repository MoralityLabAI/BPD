# LOGRANK-BPD-01: quadratic component communication

Development proposal only. Canonical exact fixtures and source receipt: https://github.com/MoralityLabAI/LoopedTransformers/tree/6b72a5cf78135c746cefcf444921098d087d691f/experiments/logrank_bpd_01.

The existing TiedComponentLinear owns V [d_in,C], U [C,d_out] with W_hat=(VU)^T, pass-specific masks and residual Delta_W. At each reached local state h (row convention), write z=hV. Add a separate proposed binary-message head, not a replacement for W_target:
m=1{z w + z Q z^T + b >= 0}, Q symmetric. Pair budget s can restrict Q's upper triangle; the diagonal is separately controlled. Existing ComponentIntervention may alter z before recomputing the message; pair-only ablation instead zeros a selected coefficient without changing z. Those are different causal interventions and must have separate estimands.

## Gauge audit

For any invertible component reparameterization R, V'=VR, U'=R^{-1}U, z'=zR; linear fidelity is unchanged. The quadratic head is invariant when w'=R^{-1}w and Q'=R^{-1}Q R^{-T}. Thus the ambient bilinear form V Q V^T is invariant; a sparse list of off-diagonal component pairs is generally basis-dependent. A shear can turn an interaction into a mix of interaction and squared terms. Explicit example in column convention: z'=(z1+z2,z2), f=2z1z2=2z'1z'2-2(z'2)^2. Omitting the diagonal changes the readout. A gauge-invariance check must include the entire transformed quadratic form, not just the selected pairs. The identity applies to the full unmasked factor map; independent component masks and existing pass-specific ablations fix a gauge and need not preserve outputs under general mixing.

For the native pilot pin the trained factor gauge, then audit sign/scale/permutation equivalence and full-head parity (including Delta_W, bias and original normalizations). Calibrate on independent development roots. Select a sparse pair mask using development only; compare zeroed-pair intervention, dose-matched random pair, same-size shuffled correspondence, linear matched head, and sham. Match actual message logit displacement and separately report reached-state radius and displacement; do not force equal hidden-state norm if the pair-only intervention acts on the head alone. Report bit flips, end-task recovery/corruption and exact all-pair success; distinguish causal effect on the message from causal effect on downstream reasoning. The two agents must communicate only charged binary messages, with no shared float features or verifier at inference.

The parity fixture is an analytical ceiling, one generator and no native BPD training. Use the EvalMath mechanistic-resolution preflight with actual independent-root paired effects and matched sham variation before confirmation. Suggested minimum effect for planning: +8 percentage points of root-level exact task correctness over matched sham at fixed bit budget, subject to actual pilot floor and headroom. No confirmation cohort defined or released. No existing BPD experiment seals touched.

Read: `mechinterp/bpd/components.py`, the EvalMath mechinterp-power skill, and the canonical receipt above.
