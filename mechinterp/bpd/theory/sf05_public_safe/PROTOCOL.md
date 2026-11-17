# BPD-SF05: Public-safe causal subspaces in a source-binding recurrent model

**Stage A prospective engineering registration, 2026-11-18.** This branch follows [BPD-SF04 #41](https://github.com/MoralityLabAI/BPD/pull/41) and [SF05 issue #42](https://github.com/MoralityLabAI/BPD/issues/42). Stage A is a **development-only** study on already excluded source roots **901 and 902**. It must not be scored as new science. A separate, frozen Stage B registration and source-family-disjoint test is required before any new-root inference. SF04's previous eight roots 1301,1303,1307,1319,1321,1327,1361,1367 and all SF03 roots are excluded from SF05 confirmation.

## What SF04 established

Trained shared recurrent teachers learned which of two users last wrote an independent scalar value (8/8 roots), but an arbitrary gauge-dependent 14/20-factor intervention recovered only 74.8% of 1,993 eligible source-tag-only donor targets and met the *joint* public-safe criterion in 3/8 roots. Exact full physical write and exact dark donor-state transplants recovered 100%. Under SF03's privileged signed future-Jacobian matching, a different 2-pass synthetic task showed 0/32 strong incremental BPD effects. SF05 explicitly distinguishes accessible physical information from additional explanatory contribution of BPD factor coordinates.

## Stage A mathematical instrument

For the SF04 physical teacher \`F(h,x)=h+tanh(hA+xB+b)C\`, define the actual prequery visible readout \`q(h)=h[0]\`. For each recipient's prequery state \`h2\`, define *different future maps on the same state*:

- \`G_U(h)=q(F(h,x_public))\`: the unauthorized/public query with role one-hot \`U\`, preserving the same public scalar and query marker.
- \`G_A(h)=q(F(h,x_authorized))\`: the recipient's A/B role query. For authorized target analysis, restrict to the SF04 frozen eligible paired donor set.

Let \`P_0\` set the first coordinate of a vector to zero. For \`d=P_0 δ\`, \`g=P_0 ∇G_U(h2)\`, define

\[
\Pi_{\rm safe}(d)=
\begin{cases}
d-\frac{\langle d,g\rangle}{\langle g,g\rangle}g,& \|g\|>10^{-8},\\
d,&\text{otherwise}.
\end{cases}
\]

This is the minimum-distance projection to \`K=ker Dq∩ker DG_U\`, conditional on nondegenerate gradient. The edited state has the **exact same current scalar readout** and **zero first-order public readout change**. Report actual finite-radius \`|G_U(h2+Π_safe(d))-G_U(h2)|\`; tangent equality alone never certifies finite public safety. Also report the dose removed by projection and near-degenerate cases.

Define \`a=P_K∇G_A(h2)\`, and record \`||a||\` and the authorized linear upper bound \`||Π_safe(d)|| ||a||\`. Random matched directions in K and **signed-authorized-gradient-and-norm matched** random directions in K serve as progressively privileged nulls. For the signed null, each matched vector \`η\` obeys \`||η||=||Π_safe(d)||\` and \`∇G_A·η=∇G_A·Π_safe(d)\`, in addition to current and public tangent constraints. Distinguish \`η\` from controls matched to the *unprojected* BPD write.

## Gauge-stable physical interventions

Freeze the teacher \`C\` and an independently trained full rank factorization \`W_hat=VU\` with 20 components. For a source-label-only donor pair, define \`w=(φ_d-φ_r)W_hat\`, with \`φ=tanh(h1 A+x2 B+b)\`. This **full write** is invariant under any invertible \`V→VR,U→R^{-1}U\`. For fixed output rank \`k∈{8,12,16,20}\`, use the top-k output-state right singular subspace of \`W_hat\`, \`P_k=V_k V_k^T\` with an orthogonal basis \`V_k\`, and evaluate \`w P_k\`. Distinct singular values and a nonzero singular gap stabilize the rank-k projector; explicitly record the boundary gap. If the rank gap is too small, flag indeterminate and use the complete physical operator as the primary gauge-stable control.

Compare, each on the **same recipient state and exact source-tag-only donor pair**:

1. Frozen SF04 14-factor coalition (gauge-dependent descriptive baseline), both raw-dark and public-tangent-projected.
2. Full BPD physical write and fixed-rank SVD physical subspace writes, both raw-dark and public-tangent-projected.
3. True teacher physical full write and actual dark-projected donor-state difference (privileged positive controls).
4. Equal-dose random intersection-kernel edits and signed-authorized-gradient-and-dose-matched shams. Include a chosen-rank **teacher-C** SVD control to show whether any effect arises from factorization rather than ordinary physical singular geometry.

All factors/teacher weights must be frozen during assays. Source labels are synthetic two-way identifiers with scalar payloads, and there is no language-model transcript channel.

## Stage A engineering-only assay

Reuse SF04 excluded seeds 901 and 902 with exactly the frozen teacher architecture, 1200 teacher steps, 400 separate factor steps, 240 development pair selection and 512 diagnostic pairs per root. Do not access SF04's fresh eight roots or any ARG-03-N 168-case sealed data. Freeze a code-level seed for random shams independent of the data generator. Run a fixed 16 shams per root for Stage A. Report eligibility/complete denominators and both public and authorized outputs, all gradient and norm errors, singular gaps, factor weight MSE and physical gauge drift. **Stage A has no scientific success threshold.** It serves to debug the geometry, demonstrate that the public and authorized gradients are distinct, and choose one physically meaningful, falsifiable Stage B arm.

## Stage B boundary

Before any new-root training, author a separate \`STAGE_B_PROTOCOL.md\` and immutable \`STAGE_B_LOCK.json\` containing independent root roster (minimum 59 sham-calibration roots, plus 32 confirmation roots), a genuinely **source-family-disjoint** test generator, ranks, gradient-matched and teacher-C controls, radius ladder, finite public-output bound, effect floor, exact scenario source list and source-hash digests.

The current Stage A system is a **mechanistic stress test**, not a field security certificate or a trained LM source-privacy result. Synthetic conditional public output deviation under an intervention can arise from moving the state off its reachable manifold. No merge or scientific claim should imply a novel secret module without stronger controls.
